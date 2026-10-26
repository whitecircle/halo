#!/usr/bin/env python
"""PP+EP gradient equivalence vs the unsplit model (pp2 × ep2 on 4 GPUs).

The pipeline schedule + EP-aware stage FSDP + the deferred post-backward sweep must together
reproduce the unsplit model's gradients: FSDP reduce-scatter covers the stage's non-expert
DTensors, the sweep DP-averages the FSDP-ignored routers over the stage block and applies the
expert DP divide (the DeepEP combine already summed each expert's grad over the EP group == the
stage's DP replicas), and the EP-aware clip aggregates the norm over dispatch group + pipeline
chain. Two unsplit references on the same bf16-rounded weights carry the checks:

  * an fp32 raw-HF reference anchors the absolute scale — per-replica loss, router grad
    norm-ratio + cosine, and the clip norm (flip-tolerant bands: fp32-vs-bf16 hidden states flip
    a few near-tied top-k picks, which no gate on data-dependent routing can avoid)
  * a bf16 ep1-wrapper twin (identical routing decisions and grouped-mm kernels) carries the
    tight per-expert gate: local shards gathered over the dispatch group vs the twin's per-expert
    grads — every expert of every kept layer covered (anti-vacuity counted)
  * router grads additionally bit-identical across each stage's ranks (the sweep's AVG collective)
  * the installed clip returns one identical norm on all 4 ranks
  * a second step proves kept-layer dispatch still works after the dropped-layer dispatcher
    destroy in _maybe_prepare_pipeline_model

Topology: 4 GPUs as 2 simulated NVLink domains of 2 (pp2 crosses the domain edge, ep2 fills a
stage). ``--gc`` re-runs the same equivalence with activation checkpointing on the stage.
Run: torchrun --nproc_per_node=4 tests/gpu/parallelism/pp/test_pp_ep_vs_unsplit.py [--gc]
"""

import sys

import torch

from src.distributed.expert_parallel.moe_layer.layer import EPMoELayerBase
from src.distributed.parallelism_config import ParallelismConfig
from tests.common import pp_equivalence as ppeq
from tests.common.harness import gpu_test_main

PP_SIZE = 2
EP_SIZE = 2
GLOBAL_BATCH = 4  # 2 DP replicas × per-device batch 2 (2 microbatches of 1)
PER_DEVICE_BATCH = 2
DP_SIZE = 2
N_MICROBATCHES = 2

# --gc: same equivalence with non-reentrant activation checkpointing on the stage. The recompute
# must replay each checkpoint frame's DeepEP dispatch/combine instead of issuing a second one (a
# second dispatch reuses the ElasticBuffer and invalidates the handle the original forward's
# backward node holds), and must not re-count expert load. Both are gated below.
USE_GC = False


@gpu_test_main(exact_world_size=4, prefix="pp_ep_vs_unsplit")
def run(ctx):
    torch.set_float32_matmul_precision("highest")  # keep the fp32 reference an actual fp32 reference
    checks, metrics = {}, {}

    rows, labels = ppeq.build_ragged_batch(GLOBAL_BATCH)
    reference = ppeq.fp32_reference(rows, labels, ctx.device, DP_SIZE)
    twin = ppeq.bf16_twin_reference(ctx, rows, labels)

    # ── PP+EP path through the real trainer ─────────────────────────────────────────────────────
    config = ParallelismConfig(
        world_size=ctx.world_size, gpus_per_node=2, nvlink_domain_size=2, pp_size=PP_SIZE, ep_size=EP_SIZE
    )
    model = ppeq.build_patched_moe_model(config)
    checks["unsplit_model_has_8_ep_layers"] = ppeq.count_ep_layers(model) == 8

    trainer = ppeq.build_pp_sft_trainer(
        ctx,
        config,
        model,
        rows,
        labels,
        per_device_batch=PER_DEVICE_BATCH,
        n_microbatches=N_MICROBATCHES,
        gradient_checkpointing=USE_GC,
    )
    stage = trainer._top_level_model()
    ep_modules = [(name, m) for name, m in stage.named_modules() if isinstance(m, EPMoELayerBase)]
    checks["stage_keeps_4_ep_layers"] = len(ep_modules) == 4
    checks["defer_grad_sync_active"] = bool(trainer._ep_config.defer_grad_sync)
    checks["no_in_backward_ep_hooks"] = all(not m._grad_sync_hooks for _, m in ep_modules)

    gc_calls = None
    if USE_GC:
        checks["gc_is_non_reentrant"] = trainer.args.gradient_checkpointing_kwargs["use_reentrant"] is False
        gc_calls = ppeq.count_checkpoint_calls(stage)
        # Expert-load accounting is only observable with the balancing counters live.
        for _name, module in ep_modules:
            module.enable_bias_balancing()

    dp_rank = config.get_data_parallel_rank()
    batch = ppeq.dp_slice_batch(rows, labels, dp_rank, PER_DEVICE_BATCH)

    loss = trainer.training_step(trainer.model, dict(batch))
    clip_norm = trainer.accelerator.clip_grad_norm_(trainer.model.parameters(), 1e6)  # runs sweep + norm, no scaling

    c, m = ppeq.check_loss_vs_reference(loss, reference, dp_rank)
    checks.update(c)
    metrics.update(m)

    c, m = ppeq.check_router_grads(stage, ep_modules, reference, trainer._pp_stage_group)
    checks.update(c)
    metrics.update(m)

    c, m = ppeq.check_expert_grads_vs_reference(ctx, stage, ep_modules, twin)
    checks.update(c)
    metrics.update(m)

    c, m = ppeq.check_clip_norm(ctx, clip_norm, reference)
    checks.update(c)
    metrics.update(m)

    if USE_GC:
        c, m = ppeq.check_gc_engaged(stage, gc_calls, N_MICROBATCHES)
        checks.update(c)
        metrics.update(m)
        # Every local token counted once per top-k slot: the recompute re-runs the router on the
        # same tokens, so an unrecognised recompute doubles this.
        c, m = ppeq.check_expert_load_counts(
            stage,
            ep_modules,
            gc_calls,
            tokens_per_forward=PER_DEVICE_BATCH // N_MICROBATCHES * ppeq.SEQ_LEN,
            n_microbatches=N_MICROBATCHES,
        )
        checks.update(c)
        metrics.update(m)

    checks["second_step_dispatch_works"] = ppeq.second_step_dispatch_works(trainer, batch, loss)

    ctx.on_teardown(trainer.cleanup_ep)
    return {"checks": checks, "metrics": metrics}


if __name__ == "__main__":
    if "--gc" in sys.argv:
        USE_GC = True
        sys.argv.remove("--gc")
    sys.exit(run())
