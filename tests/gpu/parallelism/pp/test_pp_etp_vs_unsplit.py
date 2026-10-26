#!/usr/bin/env python
"""PP + pure Expert-Tensor Parallelism gradient equivalence vs the unsplit model, in fp32.

pp2 × etp2, at world 4 (one ETP group per stage, dp1) or world 8 (two ETP groups per stage, dp2 —
the shape that also drives the deferred cross-replica expert sweep). ``ep_size`` is 1 throughout:
every rank holds every expert, sharded ``expert_tp_size``-way along the FFN intermediate dim, and
``ReduceFromExpertTP`` sums the partners' outputs in token space.

What the composition has to get right, and what each check pins:

  * the deferred post-backward sweep divides the expert leg by ``world_size // expert_tp_size`` —
    ETP partners hold shards of one expert and share a batch, so they are not DP replicas. Counting
    them as replicas would shrink every expert gradient by ``expert_tp_size``.
  * the router and the other FSDP-ignored EP submodules take the stage-wide DP average, while the
    non-expert FSDP shards are averaged by reduce-scatter over the stage mesh.
  * the EP-aware clip must count each ETP shard exactly once and the replicated router exactly once.

**Why fp32 against an unsplit reference, and not the bf16 ep1 twin** the PP+EP tests use: ETP
reorders the summation that produces each expert's output (partial sums from ``expert_tp_size``
ranks, all-reduced) instead of one contiguous bf16 GEMM. In bf16 that reordering perturbs the
hidden states enough to flip ~1% of near-tied top-k router picks, which moves whole-expert gradient
mass by 0.1-0.5 relative — with PP removed entirely, and with no sync bug present. A twin gate
therefore measures routing divergence, not the gradient sync under test. Running both sides in fp32
removes the flips (measured worst errors drop to the 1e-6 range) and lets one tolerance cover loss,
every non-expert parameter, every expert shard and the clip norm.

The reference is an unsplit single-process model carrying the same ep1 EP wrappers with grouped-GEMM
off, so its expert compute is the per-expert loop over plain torch ops — no kernel is asked to run
outside a dtype it supports, and its parameter names match the distributed run's for everything
except the experts, which are compared through the layer's own ``gather_expert_grads``.

Run: torchrun --nproc_per_node=8 tests/gpu/parallelism/pp/test_pp_etp_vs_unsplit.py
     torchrun --nproc_per_node=4 tests/gpu/parallelism/pp/test_pp_etp_vs_unsplit.py
"""

import sys

import torch
import torch.distributed as dist

from src.distributed.expert_parallel.moe_layer.layer import EPMoELayerBase
from src.distributed.parallelism_config import ParallelismConfig
from tests.common import pp_equivalence as ppeq
from tests.common.harness import gpu_test_main
from tests.common.models import TINY_GPTOSS_CONFIG

PP_SIZE = 2
ETP_SIZE = 2
PER_DEVICE_BATCH = 4  # 4 rows per replica keeps every one of the 8 experts fed (no vacuous compare)
N_MICROBATCHES = 2
TOTAL_LAYERS = TINY_GPTOSS_CONFIG["num_hidden_layers"]

# One fp32 tolerance for every quantity: loss, per-parameter grads, per-expert shards and the clip
# norm. Loose enough for the kernel difference between the reference's per-expert loop and the ETP
# fused-GLU path, ~3 orders tighter than the divisor error it has to catch (a stray /expert_tp_size
# is a 0.5 relative miss).
REL_TOL = 2e-3
ROUTER_NORM_BAND = (0.995, 1.005)
ROUTER_COSINE_MIN = 0.999


@gpu_test_main(min_world_size=4, prefix="pp_etp_vs_unsplit")
def run(ctx):
    torch.set_float32_matmul_precision("highest")  # keep the fp32 reference an actual fp32 reference
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
    checks["unsplit_model_has_all_ep_layers"] = ppeq.count_ep_layers(model) == TOTAL_LAYERS

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
    ep_modules = [(name, m) for name, m in stage.named_modules() if isinstance(m, EPMoELayerBase)]
    ep_cfg = trainer._ep_config
    checks["stage_keeps_half_the_ep_layers"] = len(ep_modules) == TOTAL_LAYERS // PP_SIZE
    checks["defer_grad_sync_active"] = bool(ep_cfg.defer_grad_sync)
    checks["no_in_backward_ep_hooks"] = all(not m._grad_sync_hooks for _, m in ep_modules)
    # The ETP axis must actually be live: partners each hold 1/etp of every expert's FFN. Without
    # this an etp_size that silently collapsed to 1 would pass every equivalence gate below.
    shard = TINY_GPTOSS_CONFIG["intermediate_size"] // ETP_SIZE
    checks["expert_ffn_sharded_over_etp_group"] = dist.get_world_size(ep_cfg.expert_tp_group) == ETP_SIZE and all(
        m.gate_proj.shape[2] == shard and m.down_proj.shape[1] == shard for _, m in ep_modules
    )
    if dp_size > 1:
        # The deferred sweep's replica-SUM leg only exists above one ETP group per stage.
        checks["expert_replica_group_spans_dp"] = (
            ep_cfg.expert_replica_group is not None and dist.get_world_size(ep_cfg.expert_replica_group) == dp_size
        )

    dp_rank = config.get_data_parallel_rank()
    batch = ppeq.dp_slice_batch(rows, labels, dp_rank, PER_DEVICE_BATCH)

    loss = trainer.training_step(trainer.model, dict(batch))
    clip_norm = trainer.accelerator.clip_grad_norm_(trainer.model.parameters(), 1e6)  # runs sweep + norm, no scaling

    c, m = ppeq.check_loss_vs_reference(loss, reference, dp_rank, tol=REL_TOL)
    checks.update(c)
    metrics.update(m)

    # Every non-expert stage parameter by global name; the experts carry no sharding-independent
    # name under ETP and are covered by the layer-side gather below.
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
