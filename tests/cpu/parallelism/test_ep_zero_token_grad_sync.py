#!/usr/bin/env python
"""The EP expert-grad sync must never ride an in-backward hook.

``register_post_accumulate_grad_hook`` fires only where a grad was actually accumulated. A rank whose
DeepEP dispatch delivered ZERO tokens for a layer returns early from ``_compute_experts`` without
touching that layer's expert weights (``moe_layer/compute.py`` — both the grouped-GEMM and the per-expert
loop template short-circuit on ``tokens.shape[0] == 0``), so those weights end the step with
``grad is None`` and their hook never runs. If that hook carried the cross-replica ``all_reduce``,
the rank's replicas — which DID receive tokens — would block forever inside a collective it never
entered.

The guarantee is structural, not a guard: ``EPConfig.sweep_reduces_across_replicas`` covers EVERY multi-EP-group
topology (not just the cross-node ``is_deferred_dp`` one), routing the sum through the trainer's
post-backward sweep, which walks ``named_parameters()`` and agrees grad presence over its reduce group
(zero-filling a grad only a peer holds) so membership is rank-uniform. Grad-equivalence holds: a zero
grad contributes nothing to the SUM. With one EP group the sweep only divides the experts, by
``world_size // expert_tp_size``, with no collective, so a rank holding no expert grad skips nothing its
peers enter (``EPConfig.runs_post_backward_sweep``; a hook would divide once per nested backward of a
reentrant recompute).

The end-to-end proof is a GPU test: an 8-GPU ``ep_size=2`` run (4 EP groups, one NVLink domain — the
shape an ``is_deferred_dp``-only rule leaves on the hook) with a batch small enough for a rank to be
served no tokens, e.g. the ``--mode ep`` body of ``tests/gpu/parallelism/ep/test_ep_modes_vs_reference.py``
at ``--nproc_per_node=8``. Not run here.

Run: python tests/cpu/parallelism/test_ep_zero_token_grad_sync.py
"""

from __future__ import annotations

from types import MethodType, SimpleNamespace
from unittest.mock import patch

import pytest
import torch
import torch.distributed as dist
import torch.nn as nn

from halo.distributed.expert_parallel.config import EPConfig
from halo.trainers.distributed_trainer import DistributedTrainerMixin
from tests.common.parallelism import stubbed_process_groups


def _make_ep(
    rank: int, *, world_size: int, stage_world_size: int, fsdp_shard_ep1_experts: bool = True, **kwargs
) -> EPConfig:
    """Build the REAL ``EPConfig`` for ``rank`` of a ``world_size`` job split into stage-wide blocks, with the
    ``experts_fsdp_managed`` verdict ``ParallelismConfig`` passes for ``fsdp_shard_ep1_experts``."""
    stage_base_rank = (rank // stage_world_size) * stage_world_size
    managed = fsdp_shard_ep1_experts and kwargs["ep_size"] * kwargs.get("expert_tp_size", 1) == 1
    with stubbed_process_groups(world_size, rank):
        return EPConfig(
            stage_world_size=stage_world_size, stage_base_rank=stage_base_rank, experts_fsdp_managed=managed, **kwargs
        )


# EPConfig: every multi-group topology defers


def test_single_domain_multigroup_ep_defers():
    """``ep_size=2`` on 8 GPUs = 4 EP groups in ONE NVLink domain — the supported single-domain
    multi-group shape. ``is_deferred_dp`` is False there (it needs >1 domain), so an
    ``is_deferred_dp``-gated rule lands this on the in-backward hook path with a live cross-replica
    all-reduce."""
    cfg = _make_ep(0, world_size=8, stage_world_size=8, ep_size=2, nvlink_domain_size=8, node_local=True)
    assert cfg.num_ep_groups == 4
    assert cfg.needs_expert_grad_sync
    assert not cfg.is_deferred_dp, "single-domain multi-group EP is NOT is_deferred_dp — the whole point"
    assert cfg.sweep_reduces_across_replicas


def test_single_ep_group_keeps_the_router_hook_and_defers_the_expert_divide():
    """One EP group spanning the world needs no cross-replica reduce, so the router's in-backward hook
    stays and keeps its backward overlap for the single-group majority; only the expert divide, which a
    hook would repeat per nested backward, moves to the sweep."""
    cfg = _make_ep(0, world_size=8, stage_world_size=8, ep_size=8, nvlink_domain_size=8, node_local=True)
    assert cfg.num_ep_groups == 1
    assert not cfg.needs_expert_grad_sync
    assert not cfg.sweep_reduces_across_replicas
    assert cfg.runs_post_backward_sweep


def test_fsdp_managed_ep1_experts_never_defer():
    """``ep_group_size == 1`` + ``fsdp_shard_ep1_experts``: FSDP2's reduce-scatter is the sole sync and
    the EP layer registers no hooks at all. ``num_ep_groups`` is world-wide here, so a naive
    "defer whenever needs_expert_grad_sync" would double-sync every expert."""
    cfg = _make_ep(
        0,
        world_size=8,
        stage_world_size=8,
        ep_size=1,
        nvlink_domain_size=8,
        node_local=True,
        fsdp_shard_ep1_experts=True,
    )
    assert cfg.ep_group_size == 1
    assert cfg.needs_expert_grad_sync
    assert cfg.experts_fsdp_managed
    assert not cfg.sweep_reduces_across_replicas


def test_no_topology_leaves_a_reducing_hook_on_any_rank():
    """The invariant, swept over every rank of every EP shape this box admits: a cross-replica reduce
    is never left to the router hook path, and the expert grads are the sweep's wherever FSDP2 leaves
    them out."""
    world = 8
    shapes = [
        {"ep_size": 1, "fsdp_shard_ep1_experts": True},
        {"ep_size": 1, "fsdp_shard_ep1_experts": False},
        {"ep_size": 1, "expert_tp_size": 2},
        {"ep_size": 2},
        {"ep_size": 2, "expert_tp_size": 2},
        {"ep_size": 4, "expert_tp_size": 2},
        {"ep_size": 8},
    ]
    for shape in shapes:
        for rank in range(world):
            cfg = _make_ep(
                rank, world_size=world, stage_world_size=world, nvlink_domain_size=world, node_local=True, **shape
            )
            assert not (cfg.uses_backward_grad_hooks and cfg.needs_expert_grad_sync), (shape, rank)
            assert cfg.runs_post_backward_sweep == (not cfg.experts_fsdp_managed), (shape, rank)


# The single-group expert sweep is collective-free


def _single_group_sweep(expert_grad: torch.Tensor | None) -> nn.Parameter:
    """The trainer's real sweep over one expert of a single EP group (8 ranks, expert TP 2)."""
    expert = nn.Parameter(torch.zeros(2, 2))
    expert.grad = expert_grad
    model = nn.Module()
    model.expert = expert
    host = SimpleNamespace(
        _ep_config=SimpleNamespace(
            sweep_reduces_across_replicas=False,
            runs_post_backward_sweep=True,
            expert_replica_group=None,
            stage_world_size=8,
            expert_tp_size=2,
            fp32_grad_reduce=False,
        ),
        state=SimpleNamespace(global_step=0),
        _get_sharded_expert_param_ids=lambda: {id(expert)},
        _get_ep_param_ids=lambda: {id(expert)},
        _top_level_model=lambda: model,
    )
    host._classify_deferred_params = MethodType(DistributedTrainerMixin._classify_deferred_params, host)
    DistributedTrainerMixin._sync_deferred_expert_grads(host)
    return expert


@pytest.mark.parametrize("has_grad", [True, False], ids=["tokens", "zero_tokens"])
def test_single_group_expert_sweep_only_scales_and_issues_no_collective(has_grad):
    """The sweep divides in place by ``world_size // expert_tp_size`` and calls no collective, so a rank
    whose dispatch delivered no tokens (``grad is None``) skips nothing its peers enter."""

    def collective(*args, **kwargs):
        raise AssertionError("the single-group expert sweep issued a collective")

    with patch.object(dist, "all_reduce", collective):
        expert = _single_group_sweep(torch.full((2, 2), 8.0) if has_grad else None)
    if has_grad:
        assert torch.equal(expert.grad, torch.full((2, 2), 2.0)), expert.grad
    else:
        assert expert.grad is None


def test_every_multi_group_rank_holds_an_expert_replica_group():
    """The invariant the deferred sweep reduces over: on every rank of every multi-EP-group shape,
    ``EPConfig`` must have matched this rank to a replica set. A rank left without one has nothing to
    reduce over, and its replicas diverge silently — so the config raises instead."""
    world = 8
    for shape in ({"ep_size": 2}, {"ep_size": 2, "expert_tp_size": 2}):
        checked = 0
        for rank in range(world):
            cfg = _make_ep(
                rank, world_size=world, stage_world_size=world, nvlink_domain_size=world, node_local=True, **shape
            )
            assert cfg.needs_expert_grad_sync, shape
            assert cfg.expert_replica_group is not None, (shape, rank)
            assert rank in cfg.expert_replica_ranks, (shape, rank)
            checked += 1
        # Per shape, not a total: a shape contributing zero ranks would otherwise hide behind another.
        assert checked == world, (shape, checked)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
