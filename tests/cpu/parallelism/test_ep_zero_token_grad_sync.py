#!/usr/bin/env python
"""The EP expert-grad sync must never ride an in-backward hook, which fires only where a grad accumulated: a rank
served zero tokens would skip the ``all_reduce`` its replicas block in."""

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
    """With the ``experts_fsdp_managed`` verdict ``ParallelismConfig`` passes."""
    stage_base_rank = (rank // stage_world_size) * stage_world_size
    managed = fsdp_shard_ep1_experts and kwargs["ep_size"] * kwargs.get("expert_tp_size", 1) == 1
    with stubbed_process_groups(world_size, rank):
        return EPConfig(
            stage_world_size=stage_world_size, stage_base_rank=stage_base_rank, experts_fsdp_managed=managed, **kwargs
        )


def test_single_domain_multigroup_ep_defers():
    """``is_deferred_dp`` needs >1 domain, so a rule gated on it leaves this shape on the hook path with a live
    all-reduce."""
    cfg = _make_ep(0, world_size=8, stage_world_size=8, ep_size=2, nvlink_domain_size=8, node_local=True)
    assert cfg.num_ep_groups == 4
    assert cfg.needs_expert_grad_sync
    assert not cfg.is_deferred_dp, "single-domain multi-group EP is NOT is_deferred_dp — the whole point"
    assert cfg.sweep_reduces_across_replicas


def test_single_ep_group_keeps_the_router_hook_and_defers_the_expert_divide():
    """Only the expert divide, which a hook would repeat per nested backward, moves to the sweep."""
    cfg = _make_ep(0, world_size=8, stage_world_size=8, ep_size=8, nvlink_domain_size=8, node_local=True)
    assert cfg.num_ep_groups == 1
    assert not cfg.needs_expert_grad_sync
    assert not cfg.sweep_reduces_across_replicas
    assert cfg.runs_post_backward_sweep


def test_fsdp_managed_ep1_experts_never_defer():
    """``num_ep_groups`` is world-wide here, so deferring on needs_expert_grad_sync would double-sync every expert."""
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


def _single_group_sweep(expert_grad: torch.Tensor | None) -> nn.Parameter:
    """8 ranks, expert TP 2."""
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
    DistributedTrainerMixin._run_deferred_grad_sweep(host)
    return expert


@pytest.mark.parametrize("has_grad", [True, False], ids=["tokens", "zero_tokens"])
def test_single_group_expert_sweep_only_scales_and_issues_no_collective(has_grad):
    """A rank served no tokens (``grad is None``) skips nothing its peers enter."""

    def collective(*args, **kwargs):
        raise AssertionError("the single-group expert sweep issued a collective")

    with patch.object(dist, "all_reduce", collective):
        expert = _single_group_sweep(torch.full((2, 2), 8.0) if has_grad else None)
    if has_grad:
        assert torch.equal(expert.grad, torch.full((2, 2), 2.0)), expert.grad
    else:
        assert expert.grad is None


def test_every_multi_group_rank_holds_an_expert_replica_group():
    """A rank left without a replica set has nothing to reduce over and its replicas diverge silently, so the config
    raises."""
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
        # Per shape, since a shape contributing zero ranks would hide behind another.
        assert checked == world, (shape, checked)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
