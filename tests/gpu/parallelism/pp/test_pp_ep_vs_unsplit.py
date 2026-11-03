#!/usr/bin/env python
"""PP+EP gradient equivalence against the unsplit model (pp2 x ep2), on two references over the same weights.

An fp32 raw-HF reference anchors the scale in flip-tolerant bands, since fp32-vs-bf16 hidden states flip near-tied
picks. A bf16 ep1 twin, with identical routing, carries the tight per-expert gate. A second step proves dispatch
survives the dropped-layer destroy.
"""

import sys

import torch

from halo.distributed.expert_parallel.moe_layer.layer import named_ep_layers
from halo.distributed.parallelism_config import ParallelismConfig
from tests.common import pp_equivalence as ppeq
from tests.common.harness import gpu_test_main

PP_SIZE = 2
EP_SIZE = 2
GLOBAL_BATCH = 4  # 2 DP replicas x per-device batch 2 (2 microbatches of 1).
PER_DEVICE_BATCH = 2
DP_SIZE = 2
N_MICROBATCHES = 2

# --gc: the recompute must replay each checkpoint frame's DeepEP dispatch and combine: a second dispatch reuses the
# ElasticBuffer and invalidates the handle the original backward node holds. It must also not re-count expert load.
USE_GC = False


@gpu_test_main(exact_world_size=4, prefix="pp_ep_vs_unsplit")
def run(ctx):
    torch.set_float32_matmul_precision("highest")  # Keeps the fp32 reference an actual fp32 reference.
    checks, metrics = {}, {}

    rows, labels = ppeq.build_ragged_batch(GLOBAL_BATCH)
    reference = ppeq.fp32_reference(rows, labels, ctx.device, DP_SIZE)
    twin = ppeq.bf16_twin_reference(ctx, rows, labels)

    config = ParallelismConfig(
        world_size=ctx.world_size, gpus_per_node=2, nvlink_domain_size=2, pp_size=PP_SIZE, ep_size=EP_SIZE
    )
    model = ppeq.build_patched_moe_model(config)
    checks["unsplit_model_has_8_ep_layers"] = len(named_ep_layers(model)) == 8

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
    ep_modules = named_ep_layers(stage)
    checks["stage_keeps_4_ep_layers"] = len(ep_modules) == 4
    checks["sweep_reduces_across_replicas"] = bool(trainer._ep_config.sweep_reduces_across_replicas)
    checks["no_in_backward_ep_hooks"] = all(not m._grad_sync_hooks for _, m in ep_modules)

    gc_calls = None
    if USE_GC:
        checks["gc_is_non_reentrant"] = trainer.args.gradient_checkpointing_kwargs["use_reentrant"] is False
        gc_calls = ppeq.count_checkpoint_calls(stage)
        # Only observable with the balancing counters live.
        for _name, module in ep_modules:
            module.enable_bias_balancing()

    dp_rank = config.get_data_parallel_rank()
    batch = ppeq.dp_slice_batch(rows, labels, dp_rank, PER_DEVICE_BATCH)

    loss = trainer.training_step(trainer.model, dict(batch))
    clip_norm = trainer.accelerator.clip_grad_norm_(trainer.model.parameters(), 1e6)  # Runs sweep + norm, no scaling.

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
        # The recompute re-runs the router on the same tokens, so an unrecognized recompute doubles this.
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
