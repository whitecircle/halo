#!/usr/bin/env python
"""PP composed with every supported intra-stage mode at world=8, over small NVLink domains so the pipeline crosses
them.

The MoE modes drive the real trainer with the fp32-reference + bf16-twin methodology of test_pp_ep_vs_unsplit.
"""

import argparse
import sys
from dataclasses import dataclass

import torch
import torch.distributed as dist
from torch.distributed.tensor import DTensor

from halo.distributed.expert_parallel.moe_layer.layer import named_ep_layers
from halo.distributed.parallelism_config import ParallelismConfig
from halo.trainers.parallelism.grad_clipping import pp_clip_grad_norm
from tests.common import pp_equivalence as ppeq
from tests.common.distributed import world_min
from tests.common.harness import gpu_test_main
from tests.common.models import TINY_QWEN3_CONFIG
from tests.common.rank_log import log

MODE = ""  # Set from --mode before the harness runs.

GLOBAL_BATCH = 8  # dp x per-device batch is 8 in every mode.
DENSE_SEQ_LEN = 32
DENSE_GRAD_TOLERANCE = 2e-3  # fp32 runner-level exactness.
DENSE_LOSS_TOLERANCE = 1e-4
EP1_GRAD_REL_TOL = 5e-2  # bf16 same-parametrization reference (matches the 4-GPU ep1 gate).
EP1_CLIP_NORM_REL_TOL = 2e-2


@dataclass(frozen=True)
class MoEModeSpec:
    pp_size: int
    ep_size: int
    ep_scope: str
    per_device_batch: int
    n_microbatches: int
    kept_ep_layers: int  # 8 total layers / pp_size.
    gpus_per_node: int = 2
    nvlink_domain_size: int = 2


MOE_MODES = {
    # One 4-rank EP group is the whole stage. The cross-domain spelling (nvlink_domain_size=2 + ep_scope="global") is
    # asserted accepted at config time, but its DeepEP transport needs the inter-node NCCL GIN path and RDMA NICs a
    # single host lacks, so the run widens the domain to 4. The node widens too: a 4-GPU domain over 2-GPU nodes is
    # MNNVL, which validate_nvlink_domain_against_fabric rejects on a fabric-free host.
    "ep4": MoEModeSpec(
        pp_size=2,
        ep_size=4,
        ep_scope="node",
        per_device_batch=2,
        n_microbatches=2,
        kept_ep_layers=4,
        gpus_per_node=4,
        nvlink_domain_size=4,
    ),
    # 1F1B needs a full pipeline: 4 microbatches at pp4, so per-device batch 4.
    "pp4_ep2": MoEModeSpec(
        pp_size=4, ep_size=2, ep_scope="node", per_device_batch=4, n_microbatches=4, kept_ep_layers=2
    ),
    "ep2_dp2": MoEModeSpec(
        pp_size=2, ep_size=2, ep_scope="node", per_device_batch=2, n_microbatches=2, kept_ep_layers=4
    ),
}


def run_dense_pp2_dp4(ctx):
    """Every stage-local grad equals the unsplit fp32 model's. No ignore_index: every DP replica holds the same token
    count, so the runner's rank-local normalizer equals the reference's G/dp and losses compare exactly."""
    checks, metrics = {}, {}
    device = ctx.device
    config = ParallelismConfig(world_size=ctx.world_size, gpus_per_node=2, nvlink_domain_size=2, pp_size=2)
    assert config.stage_world_size == 4, config.stage_world_size

    rows, labels = ppeq.build_unmasked_batch(TINY_QWEN3_CONFIG["vocab_size"], GLOBAL_BATCH, DENSE_SEQ_LEN)
    reference = ppeq.run_unsplit_reference(
        ppeq.build_dense_model(attn_implementation="eager").to(device), rows, labels, device, dp_size=4
    )

    stage_module, runner, _ = ppeq.build_runner_stage(
        ppeq.build_dense_model(attn_implementation="eager").to(device),
        config,
        ppeq.build_stage_groups(config),
        device,
        2,
    )

    dp_rank = config.get_data_parallel_rank()
    batch = ppeq.dp_slice_batch(rows, labels, dp_rank, GLOBAL_BATCH // 4)
    loss = runner.step(batch["input_ids"].to(device), batch["labels"].to(device))

    # Only last-stage ranks hold a loss; reduce the error so every rank asserts it.
    loss_err_t = torch.zeros(1, device=device)
    if config.is_last_pp_stage:
        expected = reference.slice_losses[dp_rank]
        loss_err_t[0] = abs(float(loss) - expected) / abs(expected)
        log(f"[rank {ctx.rank}] pp loss={float(loss):.8f} ref={expected:.8f} rel_err={loss_err_t.item():.3e}")
    dist.all_reduce(loss_err_t, op=dist.ReduceOp.MAX)
    checks["loss_matches_reference"] = loss_err_t.item() < DENSE_LOSS_TOLERANCE
    metrics["loss_rel_err"] = loss_err_t.item()

    c, m = ppeq.compare_stage_grads_vs_reference(ctx, stage_module, reference, DENSE_GRAD_TOLERANCE)
    checks.update(c)
    metrics.update(m)

    # Or the split silently produced empty stages.
    checks["every_stage_has_params"] = world_min(m["params_compared"], device) > 0
    return {"checks": checks, "metrics": metrics}


def _stage_dtensor_mesh_sizes(stage) -> set[int]:
    """Mesh widths of every DTensor parameter on the stage (EP experts and routers are plain tensors)."""
    return {p.data.device_mesh.size() for _, p in stage.named_parameters() if isinstance(p.data, DTensor)}


def _assert_ep4_topology(trainer, config, checks):
    ep_cfg = trainer._ep_config
    checks["single_stage_wide_ep_group"] = ep_cfg.num_ep_groups == 1 and dist.get_world_size(ep_cfg.process_group) == 4
    checks["not_deferred_dp"] = not ep_cfg.is_deferred_dp
    checks["nonexpert_fsdp_over_stage"] = _stage_dtensor_mesh_sizes(trainer._top_level_model()) == {4}
    # Node scope must stay config-rejected (ep_group 4 > domain 2) and the cross-domain global scope config-accepted:
    # it is the real multi-node spelling, and only its inter-node transport (NCCL GIN / RDMA) can't run on a single
    # host.
    try:
        ParallelismConfig(
            world_size=config.world_size, gpus_per_node=2, nvlink_domain_size=2, pp_size=2, ep_size=4, ep_scope="node"
        )
        checks["node_scope_rejected_at_config_time"] = False
    except ValueError:
        checks["node_scope_rejected_at_config_time"] = True
    try:
        accepted = ParallelismConfig(
            world_size=config.world_size,
            gpus_per_node=2,
            nvlink_domain_size=2,
            pp_size=2,
            ep_size=4,
            ep_scope="global",
        )
        checks["global_scope_accepted_at_config_time"] = accepted.num_ep_groups == 1
    except ValueError:
        checks["global_scope_accepted_at_config_time"] = False


def _assert_pp4_ep2_topology(trainer, config, checks):
    ep_cfg = trainer._ep_config
    checks["single_domain_ep_group_per_stage"] = (
        ep_cfg.num_ep_groups == 1 and dist.get_world_size(ep_cfg.process_group) == 2
    )
    checks["not_deferred_dp"] = not ep_cfg.is_deferred_dp
    checks["nonexpert_fsdp_over_stage"] = _stage_dtensor_mesh_sizes(trainer._top_level_model()) == {2}


def _assert_ep2_dp2_topology(trainer, config, checks):
    """Two EP groups per 4-rank stage (R=2). A config collapsed to one group would pass the equivalence gates
    trivially, so the group structure is asserted: non-expert params FSDP-sharded over the EP group (2), not the stage
    (4)."""
    ep_cfg = trainer._ep_config
    checks["two_ep_groups_per_stage"] = ep_cfg.num_ep_groups == 2
    checks["deferred_dp_engaged"] = bool(ep_cfg.is_deferred_dp) and bool(ep_cfg.sweep_reduces_across_replicas)
    checks["dispatch_group_is_2_rank"] = dist.get_world_size(ep_cfg.process_group) == 2
    # Same ep_rank in the other domain of the same stage.
    expected_replicas = [
        config.stage_base_rank + ep_cfg.ep_rank,
        config.stage_base_rank + 2 + ep_cfg.ep_rank,
    ]
    checks["replica_group_spans_stage_domains"] = (
        ep_cfg.expert_replica_group is not None and sorted(ep_cfg.expert_replica_ranks) == expected_replicas
    )
    # The stage's four ranks split 2+2 over group indices {0, 1}.
    idx_t = torch.tensor([float(ep_cfg.ep_group_idx)], device=torch.cuda.current_device())
    gathered = [torch.zeros_like(idx_t) for _ in range(4)]
    dist.all_gather(gathered, idx_t, group=trainer._pp_stage_group)
    checks["stage_splits_into_two_dispatch_groups"] = sorted(int(t.item()) for t in gathered) == [0, 0, 1, 1]
    mesh_sizes = _stage_dtensor_mesh_sizes(trainer._top_level_model())
    checks["nonexpert_fsdp_over_ep_group"] = mesh_sizes == {2}


def _check_replica_expert_grads_identical(rank, ep_modules, replica_group) -> bool:
    """Post-sweep expert grads are bitwise identical across the replica group (R=2). final = (1/4) · Σ_{r=0..3} d(S_r ·
    4/G) = d(S)/G, the twin's full-batch gradient; skipping the replica SUM covers half the batch (rel_err ≈ 1)."""
    identical = True
    for _name, module in ep_modules:
        for attr, param in module.expert_named_params():
            grad = param.grad.float().flatten()
            gathered = [torch.zeros_like(grad) for _ in range(dist.get_world_size(replica_group))]
            dist.all_gather(gathered, grad, group=replica_group)
            if not all(torch.equal(grad, g) for g in gathered):
                log(f"[rank {rank}] expert grad differs across replicas: {_name}.{attr}")
                identical = False
    return identical


_TOPOLOGY_ASSERTS = {
    "ep4": _assert_ep4_topology,
    "pp4_ep2": _assert_pp4_ep2_topology,
    "ep2_dp2": _assert_ep2_dp2_topology,
}


def run_moe_mode(ctx, mode: str):
    spec = MOE_MODES[mode]
    torch.set_float32_matmul_precision("highest")  # Keeps the fp32 reference an actual fp32 reference.
    checks, metrics = {}, {}

    config = ParallelismConfig(
        world_size=ctx.world_size,
        gpus_per_node=spec.gpus_per_node,
        nvlink_domain_size=spec.nvlink_domain_size,
        pp_size=spec.pp_size,
        ep_size=spec.ep_size,
        ep_scope=spec.ep_scope,
    )
    dp_size = config.data_parallel_size
    assert dp_size * spec.per_device_batch == GLOBAL_BATCH, (dp_size, spec)

    rows, labels = ppeq.build_ragged_batch(GLOBAL_BATCH)
    reference = ppeq.fp32_reference(rows, labels, ctx.device, dp_size)
    twin = ppeq.bf16_twin_reference(ctx, rows, labels)

    model = ppeq.build_patched_moe_model(config)
    checks["unsplit_model_has_8_ep_layers"] = len(named_ep_layers(model)) == 8

    trainer = ppeq.build_pp_sft_trainer(
        ctx,
        config,
        model,
        rows,
        labels,
        per_device_batch=spec.per_device_batch,
        n_microbatches=spec.n_microbatches,
    )
    stage = trainer._top_level_model()
    ep_modules = named_ep_layers(stage)
    checks["stage_keeps_expected_ep_layers"] = len(ep_modules) == spec.kept_ep_layers
    checks["sweep_reduces_across_replicas"] = bool(trainer._ep_config.sweep_reduces_across_replicas)
    checks["no_in_backward_ep_hooks"] = all(not m._grad_sync_hooks for _, m in ep_modules)
    _TOPOLOGY_ASSERTS[mode](trainer, config, checks)

    dp_rank = config.get_data_parallel_rank()
    batch = ppeq.dp_slice_batch(rows, labels, dp_rank, spec.per_device_batch)

    loss = trainer.training_step(trainer.model, dict(batch))
    clip_norm = trainer.accelerator.clip_grad_norm_(trainer.model.parameters(), 1e6)  # Runs sweep + norm, no scaling.

    c, m = ppeq.check_loss_vs_reference(loss, reference, dp_rank)
    checks.update(c)
    metrics.update(m)

    c, m = ppeq.check_router_grads(
        stage, ep_modules, reference, trainer._pp_stage_group, twin_router_grads=twin.router_grads
    )
    checks.update(c)
    metrics.update(m)

    c, m = ppeq.check_expert_grads_vs_reference(ctx, stage, ep_modules, twin)
    checks.update(c)
    metrics.update(m)

    if mode == "ep2_dp2":
        checks["expert_grads_bitwise_identical_across_replicas"] = _check_replica_expert_grads_identical(
            ctx.rank, ep_modules, trainer._ep_config.expert_replica_group
        )

    c, m = ppeq.check_clip_norm(ctx, clip_norm, reference)
    checks.update(c)
    metrics.update(m)

    checks["second_step_dispatch_works"] = ppeq.second_step_dispatch_works(trainer, batch, loss)

    ctx.on_teardown(trainer.cleanup_ep)
    return {"checks": checks, "metrics": metrics}


def run_ep1_dp4(ctx):
    """pp2 x ep1 grouped-GEMM x dp4: the ep1-shard contract at stage width 4, every stage param compared by global
    name."""
    checks, metrics = {}, {}
    rows, labels = ppeq.build_ragged_batch(GLOBAL_BATCH)
    reference = ppeq.ep1_reference(rows, labels, ctx.device, dp_size=4, rank=ctx.rank)

    config = ParallelismConfig(world_size=ctx.world_size, gpus_per_node=2, nvlink_domain_size=2, pp_size=2)
    assert config.stage_world_size == 4, config.stage_world_size
    model = ppeq.build_patched_moe_model(config)
    trainer = ppeq.build_pp_sft_trainer(ctx, config, model, rows, labels, per_device_batch=2, n_microbatches=2)
    stage = trainer._top_level_model()
    ep_modules = named_ep_layers(stage)
    checks["stage_keeps_expected_ep_layers"] = len(ep_modules) == 4

    # Stage-mesh DTensor experts (width 4), no EP hooks, plain PP clip.
    expert_params = [p for _, m in ep_modules for _, p in m.expert_named_params()]
    checks["experts_are_stage_mesh_dtensors"] = all(
        isinstance(p, DTensor) and p.device_mesh.size() == config.stage_world_size for p in expert_params
    )
    checks["no_grad_sync_hooks"] = all(not m._grad_sync_hooks for _, m in ep_modules)
    checks["clip_is_pp_clip"] = trainer.accelerator.clip_grad_norm_ is pp_clip_grad_norm
    checks["no_deferred_sync"] = not trainer._ep_config.sweep_reduces_across_replicas

    dp_rank = config.get_data_parallel_rank()
    batch = ppeq.dp_slice_batch(rows, labels, dp_rank, 2)
    loss = trainer.training_step(trainer.model, dict(batch))
    clip_norm = trainer.accelerator.clip_grad_norm_(trainer.model.parameters(), 1e6)

    c, m = ppeq.check_loss_vs_reference(loss, reference, dp_rank)
    checks.update(c)
    metrics.update(m)

    c, m = ppeq.compare_stage_grads_vs_reference(ctx, stage, reference, EP1_GRAD_REL_TOL)
    checks.update(c)
    metrics.update(m)

    c, m = ppeq.check_clip_norm(ctx, clip_norm, reference, tol=EP1_CLIP_NORM_REL_TOL)
    checks.update(c)
    metrics.update(m)

    checks["second_step_dispatch_works"] = ppeq.second_step_dispatch_works(trainer, batch, loss)

    ctx.on_teardown(trainer.cleanup_ep)
    return {"checks": checks, "metrics": metrics}


@gpu_test_main(exact_world_size=8, prefix="pp_combos_8gpu")
def run(ctx):
    if MODE == "dense_pp2_dp4":
        return run_dense_pp2_dp4(ctx)
    if MODE == "ep1_dp4":
        return run_ep1_dp4(ctx)
    return run_moe_mode(ctx, MODE)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", required=True, choices=("dense_pp2_dp4", *sorted(MOE_MODES), "ep1_dp4"))
    MODE = parser.parse_args().mode
    sys.exit(run())
