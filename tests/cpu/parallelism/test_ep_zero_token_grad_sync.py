#!/usr/bin/env python
"""The EP cross-replica expert-grad reduce must never ride an in-backward hook.

``register_post_accumulate_grad_hook`` fires only where a grad was actually accumulated. A rank whose
DeepEP dispatch delivered ZERO tokens for a layer returns early from ``_compute_experts`` without
touching that layer's expert weights (``base_layer.py`` — both the grouped-GEMM and the per-expert
loop template short-circuit on ``tokens.shape[0] == 0``), so those weights end the step with
``grad is None`` and their hook never runs. If that hook carried the cross-replica ``all_reduce``,
the rank's replicas — which DID receive tokens — would block forever inside a collective it never
entered.

The guarantee is structural, not a guard: ``EPConfig.defer_grad_sync`` covers EVERY multi-EP-group
topology (not just the cross-node ``is_deferred_dp`` one), routing the sum through the trainer's
post-backward sweep, which walks ``named_parameters()`` and zero-fills missing grads so membership is
rank-uniform. Grad-equivalence holds: a zero grad contributes nothing to the SUM, and the sweep's
expert divisor is the same ``world_size // expert_tp_size`` a hook would use. ``create_expert_grad_hook``
refuses to build a reducing hook at all, so a regression fails loudly instead of hanging.

The end-to-end proof is a GPU test: an 8-GPU ``ep_size=2`` run (4 EP groups, one NVLink domain — the
shape an ``is_deferred_dp``-only rule leaves on the hook) with a batch small enough for a rank to be
served no tokens, e.g. ``tests/gpu/parallelism/ep/test_ep_correctness.py`` at
``--nproc_per_node=8`` with ``ep_size=2``. Not run here.

Run: python tests/cpu/parallelism/test_ep_zero_token_grad_sync.py
"""

from __future__ import annotations

import types
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
import torch.distributed as dist
import torch.nn as nn

from src.distributed.expert_parallel.config import EPConfig
from src.distributed.expert_parallel.grad_sync import create_expert_grad_hook

_EP_MOD = "src.distributed.expert_parallel.config"
_GS_MOD = "src.distributed.expert_parallel.grad_sync"


def _recording_dist(calls: list):
    """Stub ``torch.distributed``: ``new_group`` records the rank list and returns it as the handle."""

    def _new_group(ranks, timeout=None):
        calls.append(tuple(ranks))
        return tuple(ranks)

    return types.SimpleNamespace(is_initialized=lambda: True, new_group=_new_group)


def _make_ep(rank: int, *, world_size: int, stage_world_size: int, **kwargs) -> EPConfig:
    """Build the REAL ``EPConfig`` for ``rank`` of a ``world_size`` job split into stage-wide blocks."""
    rank_offset = (rank // stage_world_size) * stage_world_size
    with (
        patch(f"{_EP_MOD}.dist", _recording_dist([])),
        patch(f"{_EP_MOD}.get_global_rank", return_value=rank),
        patch(f"{_EP_MOD}.get_global_world_size", return_value=world_size),
        patch(f"{_EP_MOD}.get_local_world_size", return_value=8),
        patch(f"{_EP_MOD}.get_nccl_timeout", return_value=None),
        patch(f"{_EP_MOD}.is_global_main_process", return_value=False),
    ):
        return EPConfig(world_size=stage_world_size, rank_offset=rank_offset, **kwargs)


def _expert_hook_from(**ep_attrs):
    """Build the expert grad hook off a stub carrying every attribute a real ``EPConfig`` sets, with
    ``dist`` reporting peers so the hook is the distributed one rather than the single-process no-op."""
    defaults = {
        "world_size": 8,
        "expert_tp_size": 1,
        "needs_expert_grad_sync": False,
        "fp32_grad_reduce": False,
        "dp_scope_group": None,
    }
    ep = SimpleNamespace(**{**defaults, **ep_attrs})
    fake_dist = types.SimpleNamespace(is_available=lambda: True, is_initialized=lambda: True, ReduceOp=dist.ReduceOp)
    with patch(f"{_GS_MOD}.dist", fake_dist):
        return create_expert_grad_hook(ep)


# EPConfig: every multi-group topology defers


def test_single_domain_multigroup_ep_defers():
    """``ep_size=2`` on 8 GPUs = 4 EP groups in ONE NVLink domain — the supported single-domain
    multi-group shape. ``is_deferred_dp`` is False there (it needs >1 domain), so an
    ``is_deferred_dp``-gated rule lands this on the in-backward hook path with a live cross-replica
    all-reduce."""
    cfg = _make_ep(0, world_size=8, stage_world_size=8, ep_size=2, gpus_per_node=8, node_local=True)
    assert cfg.num_ep_groups == 4
    assert cfg.needs_expert_grad_sync
    assert not cfg.is_deferred_dp, "single-domain multi-group EP is NOT is_deferred_dp — the whole point"
    assert cfg.defer_grad_sync


def test_single_ep_group_still_hooks():
    """One EP group spanning the world needs no cross-replica reduce, so the cheap in-backward hook
    path stays. Guards against "just defer everything", which would drop backward overlap for the
    single-group majority."""
    cfg = _make_ep(0, world_size=8, stage_world_size=8, ep_size=8, gpus_per_node=8, node_local=True)
    assert cfg.num_ep_groups == 1
    assert not cfg.needs_expert_grad_sync
    assert not cfg.defer_grad_sync


def test_fsdp_managed_ep1_experts_never_defer():
    """``ep_group_size == 1`` + ``fsdp_shard_ep1_experts``: FSDP2's reduce-scatter is the sole sync and
    the EP layer registers no hooks at all. ``num_ep_groups`` is world-wide here, so a naive
    "defer whenever needs_expert_grad_sync" would double-sync every expert."""
    cfg = _make_ep(
        0, world_size=8, stage_world_size=8, ep_size=1, gpus_per_node=8, node_local=True, fsdp_shard_ep1_experts=True
    )
    assert cfg.ep_group_size == 1
    assert cfg.needs_expert_grad_sync
    assert cfg.experts_fsdp_managed
    assert not cfg.defer_grad_sync


def test_no_topology_leaves_a_reducing_hook_on_any_rank():
    """The invariant, swept over every rank of every EP shape this box admits: a cross-replica reduce
    is never left to the hook path. Equivalently — ``create_expert_grad_hook`` never raises for a config
    that reaches it."""
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
                rank, world_size=world, stage_world_size=world, gpus_per_node=world, node_local=True, **shape
            )
            hooks_registered = not cfg.experts_fsdp_managed and not cfg.defer_grad_sync
            assert not (hooks_registered and cfg.needs_expert_grad_sync), (shape, rank)


# The hook itself is collective-free, and says so loudly


def test_expert_hook_refuses_to_carry_a_cross_replica_reduce():
    """A hand-built config that selects hooks while needing the cross-replica sum must fail loudly at
    hook creation rather than hang mid-backward on the first zero-token rank."""
    with pytest.raises(RuntimeError, match="zero-token rank"):
        _expert_hook_from(needs_expert_grad_sync=True)


def test_expert_hook_only_scales_and_issues_no_collective():
    """The surviving hook path divides in place by ``world_size // expert_tp_size`` and calls no
    collective — so a rank that never fires it costs nothing but its own (zero) contribution."""
    hook = _expert_hook_from(world_size=8, expert_tp_size=2)
    param = nn.Parameter(torch.zeros(2, 2))
    param.grad = torch.full((2, 2), 8.0)
    reduced: list = []
    with (
        patch(f"{_GS_MOD}.reduce_grad", lambda *a, **kw: reduced.append(kw)),
        patch(f"{_GS_MOD}.should_sync_gradients", return_value=True),
    ):
        hook(param)
    assert reduced == [], "the expert hook must issue no collective"
    # world_size // expert_tp_size == 4 — the same divisor the deferred sweep applies.
    assert torch.equal(param.grad, torch.full((2, 2), 2.0)), param.grad


def test_expert_hook_is_a_noop_without_peers():
    """Single-process runs have no peers: the hook must not even divide (divisor 1 semantics)."""
    hook = _expert_hook_from(world_size=1)
    param = nn.Parameter(torch.zeros(2))
    param.grad = torch.ones(2)
    with patch(f"{_GS_MOD}.should_sync_gradients", return_value=True):
        hook(param)
    assert torch.equal(param.grad, torch.ones(2))


def test_every_multi_group_rank_holds_an_expert_replica_group():
    """The invariant the deferred sweep reduces over: on every rank of every multi-EP-group shape,
    ``EPConfig`` must have matched this rank to a replica set. A rank left without one has nothing to
    reduce over, and its replicas diverge silently — so the config raises instead."""
    world = 8
    for shape in ({"ep_size": 2}, {"ep_size": 2, "expert_tp_size": 2}):
        checked = 0
        for rank in range(world):
            cfg = _make_ep(
                rank, world_size=world, stage_world_size=world, gpus_per_node=world, node_local=True, **shape
            )
            assert cfg.needs_expert_grad_sync, shape
            assert cfg.expert_replica_group is not None, (shape, rank)
            assert rank in cfg.expert_replica_ranks, (shape, rank)
            checked += 1
        # Per shape, not a total: a shape contributing zero ranks would otherwise hide behind another.
        assert checked == world, (shape, checked)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
