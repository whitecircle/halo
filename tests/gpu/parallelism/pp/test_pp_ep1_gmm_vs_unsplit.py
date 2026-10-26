#!/usr/bin/env python
"""PP × ep1 grouped-GEMM MoE vs the unsplit model (pp2 × dp2, fsdp_shard_ep1_experts=True).

At ep_size==1 the grouped-GEMM EP wrappers stay on every MoE model and fsdp_shard_ep1_experts
folds the replicated experts into the stage FSDP mesh: no EP hooks, no deferred sweep — FSDP's
reduce-scatter is the sole grad sync and every parameter (experts included) is a stage-mesh
DTensor, so the plain ``pp_clip_grad_norm`` covers the whole stage. The reference is the same
model with the same ep1 wrappers, single-process (its EPConfig is world_size=1, so every hook is
inert), run on the full global batch — identical parametrization, so gradients compare per param
through ``global_parameter_name`` with a full_tensor() gather.

Checks: per-replica loss vs the reference slice loss; experts are DTensors on the stage mesh; no
grad-sync hooks registered; every stage param's grad matches the reference (worst rel err, bf16
gate) and the union of stage param names covers the reference exactly; the installed clip is
``pp_clip_grad_norm`` and returns one identical norm on all ranks.

Topology: 4 GPUs as 2 simulated NVLink domains of 2. Run:
    torchrun --nproc_per_node=4 tests/gpu/parallelism/pp/test_pp_ep1_gmm_vs_unsplit.py
"""

import sys

from torch.distributed.tensor import DTensor

from halo.distributed.expert_parallel.moe_layer.layer import named_ep_layers
from halo.distributed.parallelism_config import ParallelismConfig
from halo.trainers.parallelism.grad_clipping import pp_clip_grad_norm
from tests.common import pp_equivalence as ppeq
from tests.common.harness import gpu_test_main

PP_SIZE = 2
GLOBAL_BATCH = 4  # 2 DP replicas × per-device batch 2 (2 microbatches of 1)
PER_DEVICE_BATCH = 2
DP_SIZE = 2
GRAD_REL_TOL = 5e-2
CLIP_NORM_REL_TOL = 2e-2


@gpu_test_main(exact_world_size=4, prefix="pp_ep1_gmm")
def run(ctx):
    checks, metrics = {}, {}
    rows, labels = ppeq.build_ragged_batch(GLOBAL_BATCH)
    reference = ppeq.ep1_reference(rows, labels, ctx.device, DP_SIZE, ctx.rank)

    # ── PP × ep1-gmm × dp2 through the real trainer ─────────────────────────────────────────────
    config = ParallelismConfig(world_size=ctx.world_size, gpus_per_node=2, nvlink_domain_size=2, pp_size=PP_SIZE)
    model = ppeq.build_patched_moe_model(config)
    trainer = ppeq.build_pp_sft_trainer(
        ctx, config, model, rows, labels, per_device_batch=PER_DEVICE_BATCH, n_microbatches=2, max_steps=1
    )
    stage = trainer._top_level_model()
    ep_modules = named_ep_layers(stage)
    checks["stage_keeps_4_ep_layers"] = len(ep_modules) == 4

    # ep1-shard contract: experts are stage-mesh DTensors, no EP hooks, plain PP clip installed.
    expert_params = [p for _, m in ep_modules for _, p in m.expert_named_params()]
    checks["experts_are_stage_mesh_dtensors"] = all(
        isinstance(p, DTensor) and p.device_mesh.size() == config.stage_world_size for p in expert_params
    )
    checks["no_grad_sync_hooks"] = all(not m._grad_sync_hooks for _, m in ep_modules)
    checks["clip_is_pp_clip"] = trainer.accelerator.clip_grad_norm_ is pp_clip_grad_norm
    checks["no_deferred_sync"] = not trainer._ep_config.defer_grad_sync

    dp_rank = config.get_data_parallel_rank()
    batch = ppeq.dp_slice_batch(rows, labels, dp_rank, PER_DEVICE_BATCH)
    loss = trainer.training_step(trainer.model, dict(batch))
    clip_norm = trainer.accelerator.clip_grad_norm_(trainer.model.parameters(), 1e6)

    c, m = ppeq.check_loss_vs_reference(loss, reference, dp_rank)
    checks.update(c)
    metrics.update(m)

    c, m = ppeq.compare_stage_grads_vs_reference(ctx, stage, reference, GRAD_REL_TOL)
    checks.update(c)
    metrics.update(m)

    c, m = ppeq.check_clip_norm(ctx, clip_norm, reference, tol=CLIP_NORM_REL_TOL)
    checks.update(c)
    metrics.update(m)

    ctx.on_teardown(trainer.cleanup_ep)
    return {"checks": checks, "metrics": metrics}


if __name__ == "__main__":
    sys.exit(run())
