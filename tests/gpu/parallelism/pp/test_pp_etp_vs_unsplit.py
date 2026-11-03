#!/usr/bin/env python
"""PP + pure ETP gradient equivalence against the unsplit model, in fp32.

ETP partners share a batch, so the sweep divides by ``world_size // expert_tp_size``; counting them as replicas shrinks
expert gradients. fp32: ETP reorders each expert's sum, which in bf16 flips ~1% of near-tied picks and moves expert
gradient mass by 0.1-0.5 relative with no sync bug. In fp32 the worst errors drop to the 1e-6 range.
"""

import sys

import torch
import torch.distributed as dist

from halo.distributed.expert_parallel.moe_layer.layer import named_ep_layers
from halo.distributed.parallelism_config import ParallelismConfig
from tests.common import pp_equivalence as ppeq
from tests.common.harness import gpu_test_main
from tests.common.models import TINY_GPTOSS_CONFIG

PP_SIZE = 2
ETP_SIZE = 2
PER_DEVICE_BATCH = 4  # 4 rows per replica keeps every one of the 8 experts fed (no vacuous compare).
N_MICROBATCHES = 2
TOTAL_LAYERS = TINY_GPTOSS_CONFIG["num_hidden_layers"]

# One fp32 tolerance for loss, grads, expert shards and the clip norm: loose enough for the reference's per-expert loop
# against ETP's fused-GLU path, ~3 orders under a stray /expert_tp_size (a 0.5 relative miss).
REL_TOL = 2e-3
ROUTER_NORM_BAND = (0.995, 1.005)
ROUTER_COSINE_MIN = 0.999


@gpu_test_main(min_world_size=4, prefix="pp_etp_vs_unsplit")
def run(ctx):
    torch.set_float32_matmul_precision("highest")  # Keeps the fp32 reference an actual fp32 reference.
    checks, metrics = {}, {}

    config = ParallelismConfig(
        world_size=ctx.world_size,
        gpus_per_node=2,
        nvlink_domain_size=2,
        pp_size=PP_SIZE,
        expert_tp_size=ETP_SIZE,
    )
    dp_size = config.data_parallel_size
    checks["dp_size_matches_topology"] = dp_size == ctx.world_size // PP_SIZE // ETP_SIZE

    rows, labels = ppeq.build_ragged_batch(dp_size * PER_DEVICE_BATCH)
    reference = ppeq.ep1_reference(
        rows, labels, ctx.device, dp_size, ctx.rank, dtype=torch.float32, use_grouped_gemm=False
    )

    model = ppeq.build_patched_moe_model(config, dtype=torch.float32)
    checks["unsplit_model_has_all_ep_layers"] = len(named_ep_layers(model)) == TOTAL_LAYERS

    trainer = ppeq.build_pp_sft_trainer(
        ctx,
        config,
        model,
        rows,
        labels,
        per_device_batch=PER_DEVICE_BATCH,
        n_microbatches=N_MICROBATCHES,
        bf16=False,
    )
    stage = trainer._top_level_model()
    ep_modules = named_ep_layers(stage)
    ep_cfg = trainer._ep_config
    checks["stage_keeps_half_the_ep_layers"] = len(ep_modules) == TOTAL_LAYERS // PP_SIZE
    checks["sweep_reduces_across_replicas"] = bool(ep_cfg.sweep_reduces_across_replicas)
    checks["no_in_backward_ep_hooks"] = all(not m._grad_sync_hooks for _, m in ep_modules)
    # Without this an etp_size silently collapsed to 1 would pass every equivalence gate below.
    shard = TINY_GPTOSS_CONFIG["intermediate_size"] // ETP_SIZE
    checks["expert_ffn_sharded_over_etp_group"] = dist.get_world_size(ep_cfg.expert_tp_group) == ETP_SIZE and all(
        m.gate_proj.shape[2] == shard and m.down_proj.shape[1] == shard for _, m in ep_modules
    )
    if dp_size > 1:
        # The replica-SUM leg only exists above one ETP group per stage.
        checks["expert_replica_group_spans_dp"] = (
            ep_cfg.expert_replica_group is not None and dist.get_world_size(ep_cfg.expert_replica_group) == dp_size
        )

    dp_rank = config.get_data_parallel_rank()
    batch = ppeq.dp_slice_batch(rows, labels, dp_rank, PER_DEVICE_BATCH)

    loss = trainer.training_step(trainer.model, dict(batch))
    clip_norm = trainer.accelerator.clip_grad_norm_(trainer.model.parameters(), 1e6)  # Runs sweep + norm, no scaling.

    c, m = ppeq.check_loss_vs_reference(loss, reference, dp_rank, tol=REL_TOL)
    checks.update(c)
    metrics.update(m)

    # By global name; experts carry no sharding-independent name under ETP, so the layer-side gather below covers them.
    c, m = ppeq.compare_stage_grads_vs_reference(ctx, stage, reference, REL_TOL, exclude_experts=True)
    checks.update(c)
    metrics.update(m)

    c, m = ppeq.check_router_grads(
        stage, ep_modules, reference, trainer._pp_stage_group, band=ROUTER_NORM_BAND, cos_min=ROUTER_COSINE_MIN
    )
    checks.update(c)
    metrics.update(m)

    c, m = ppeq.check_expert_grads_vs_reference(ctx, stage, ep_modules, reference, tol=REL_TOL)
    checks.update(c)
    metrics.update(m)

    c, m = ppeq.check_clip_norm(ctx, clip_norm, reference, tol=REL_TOL)
    checks.update(c)
    metrics.update(m)

    checks["second_step_works"] = ppeq.second_step_dispatch_works(trainer, batch, loss)

    ctx.on_teardown(trainer.cleanup_ep)
    return {"checks": checks, "metrics": metrics}


if __name__ == "__main__":
    sys.exit(run())
