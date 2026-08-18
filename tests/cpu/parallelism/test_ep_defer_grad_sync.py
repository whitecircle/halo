#!/usr/bin/env python
"""EPConfig deferred-grad-sync + DP-scope-group contracts (pipeline-parallel rank blocks).

``defer_grad_sync`` decides whether EP grad sync rides per-backward hooks or the trainer's
post-backward sweep. It must be a strict superset of ``is_deferred_dp``: any rank-blocked job
(``num_rank_blocks > 1``, i.e. pipeline parallelism) also defers, because PP forces
``gradient_accumulation_steps`` to 1 so post-accumulate hooks would fire after EVERY microbatch
backward and re-scale already-synced gradients. The one exception is ep1 experts sharded by FSDP
(``fsdp_shard_ep1_experts`` + ``ep_group_size == 1``), whose reduce-scatter is accumulation-safe.

``dp_scope_group`` is the collective domain for router/replicated-param DP averages under PP (one
group per rank block); ``None`` without PP keeps the non-PP collective sequence bit-identical.

These tests drive the REAL ``EPConfig`` and the grad-hook factories built off it, with
``torch.distributed`` stubbed (``new_group`` records and returns the rank tuple), following
``test_pp_group_construction.py``.

Run: python tests/cpu/parallelism/test_ep_defer_grad_sync.py
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
from src.distributed.expert_parallel.grad_sync import create_router_grad_hook
from src.trainers.mixins.base import DistributedTrainerMixin

_EP_MOD = "src.distributed.expert_parallel.config"
_GS_MOD = "src.distributed.expert_parallel.grad_sync"
# The trainer-side sweep, whose globals the deferred-sync tests patch.
_MIXIN_MOD = "src.trainers.mixins.grad_sync"


def _recording_dist(calls: list):
    """Stub ``torch.distributed``: ``new_group`` records the rank list and returns it as the handle."""

    def _new_group(ranks, timeout=None):
        calls.append(tuple(ranks))
        return tuple(ranks)

    return types.SimpleNamespace(is_initialized=lambda: True, new_group=_new_group)


def _make_ep(rank: int, *, world_size: int, stage_world_size: int, **kwargs):
    """Build the REAL ``EPConfig`` for ``rank`` of a ``world_size`` job split into stage-wide blocks.

    Returns ``(config, calls)`` where ``calls`` is the ordered ``new_group`` rank-list sequence."""
    calls: list = []
    rank_offset = (rank // stage_world_size) * stage_world_size
    with (
        patch(f"{_EP_MOD}.dist", _recording_dist(calls)),
        patch(f"{_EP_MOD}.get_global_rank", return_value=rank),
        patch(f"{_EP_MOD}.get_global_world_size", return_value=world_size),
        patch(f"{_EP_MOD}.get_local_world_size", return_value=8),
        patch(f"{_EP_MOD}.get_nccl_timeout", return_value=None),
        patch(f"{_EP_MOD}.is_global_main_process", return_value=False),
    ):
        return EPConfig(world_size=stage_world_size, rank_offset=rank_offset, **kwargs), calls


# defer_grad_sync truth table


def test_defer_false_without_pp_single_node():
    """pp1, single node, single EP group: hooks stay (defer False), no DP-scope group."""
    cfg, _ = _make_ep(0, world_size=8, stage_world_size=8, ep_size=8, gpus_per_node=8, node_local=True)
    assert cfg.num_rank_blocks == 1
    assert not cfg.is_deferred_dp
    assert not cfg.defer_grad_sync
    assert cfg.dp_scope_group is None


def test_defer_true_for_multinode_deferred_dp_without_pp():
    """pp1 multi-node multi-group EP (is_deferred_dp) implies defer_grad_sync — superset check."""
    cfg, _ = _make_ep(0, world_size=16, stage_world_size=16, ep_size=8, gpus_per_node=8, node_local=True)
    assert cfg.num_rank_blocks == 1
    assert cfg.is_deferred_dp
    assert cfg.defer_grad_sync
    assert cfg.dp_scope_group is None  # single rank block → default (world) group


def test_defer_true_under_pp_even_without_deferred_dp():
    """PP (2 rank blocks), single-node single-group EP: is_deferred_dp is False, but the
    per-microbatch hook firing forces the deferral anyway."""
    cfg, _ = _make_ep(0, world_size=16, stage_world_size=8, ep_size=8, gpus_per_node=8, node_local=True)
    assert cfg.num_rank_blocks == 2
    assert not cfg.is_deferred_dp
    assert cfg.defer_grad_sync


def test_defer_false_under_pp_for_fsdp_sharded_ep1_experts():
    """PP + ep1 + fsdp_shard_ep1_experts: no EP hooks exist at all (FSDP reduce-scatter is the sole,
    accumulation-safe sync) — deferring would double-sync. defer must stay False."""
    cfg, _ = _make_ep(
        0,
        world_size=16,
        stage_world_size=8,
        ep_size=1,
        gpus_per_node=8,
        node_local=True,
        fsdp_shard_ep1_experts=True,
    )
    assert cfg.num_rank_blocks == 2
    assert cfg.ep_group_size == 1
    assert not cfg.defer_grad_sync


def test_defer_true_under_pp_for_unsharded_ep1_experts():
    """PP + ep1 WITHOUT the FSDP sharding: the EP hooks are the only sync, so they must defer."""
    cfg, _ = _make_ep(
        0,
        world_size=16,
        stage_world_size=8,
        ep_size=1,
        gpus_per_node=8,
        node_local=True,
        fsdp_shard_ep1_experts=False,
    )
    assert cfg.defer_grad_sync


def test_defer_true_under_pp_for_pure_etp():
    """PP + pure ETP (ep_size=1, expert_tp_size=2): ep_group_size is 2, so the ep1-FSDP exception
    must NOT apply even though ep_size == 1 and fsdp_shard_ep1_experts is on."""
    cfg, _ = _make_ep(
        0,
        world_size=16,
        stage_world_size=8,
        ep_size=1,
        expert_tp_size=2,
        gpus_per_node=8,
        node_local=True,
        fsdp_shard_ep1_experts=True,
    )
    assert cfg.ep_group_size == 2
    assert cfg.defer_grad_sync


def test_single_process_mode_defaults():
    """Bare ``python`` launch (dist never initialized): one block, no deferral, world-scope group."""
    cfg = EPConfig(ep_size=1, world_size=1, gpus_per_node=1)
    assert cfg.num_rank_blocks == 1
    assert not cfg.defer_grad_sync
    assert cfg.dp_scope_group is None


# dp_scope_group construction discipline


def test_dp_scope_groups_block_confined_and_identical_order():
    """Every rank creates BOTH blocks' DP-scope groups in one order and keeps exactly its own."""
    built = {
        rank: _make_ep(rank, world_size=16, stage_world_size=8, ep_size=8, gpus_per_node=8, node_local=True)
        for rank in range(16)
    }
    for rank, (cfg, calls) in built.items():
        assert cfg.dp_scope_group == tuple(range((rank // 8) * 8, (rank // 8) * 8 + 8)), (rank, cfg.dp_scope_group)
        # The DP-scope groups are the LAST two new_group calls, block-major, on every rank.
        assert calls[-2:] == [tuple(range(8)), tuple(range(8, 16))], (rank, calls[-2:])
    reference = built[0][1]
    for rank, (_, calls) in built.items():
        assert calls == reference, (rank, calls, reference)


def test_pp1_new_group_sequence_bit_identical_to_pre_pp():
    """Without PP no DP-scope groups are created: the exact pre-PP pinned call list is unchanged."""
    cfg, calls = _make_ep(3, world_size=8, stage_world_size=8, ep_size=2, gpus_per_node=8, node_local=True)
    assert cfg.dp_scope_group is None
    assert calls == [(0, 1), (2, 3), (4, 5), (6, 7), (0, 2, 4, 6), (1, 3, 5, 7)], calls


# Router-hook plumbing: the DP-scope group and divisor come straight off EPConfig


def _router_hook_from(**ep_attrs):
    # The stub carries every attribute a real EPConfig sets: the hook factory reads them directly, so
    # a getattr default would mask a typo into "no expert grad sync".
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
        return create_router_grad_hook(ep)


def test_router_hook_reduces_over_dp_scope_group():
    """The router DP-average must run over dp_scope_group (None = world) with the block-width divisor.

    A world-group all-reduce under PP would blend gradients of DIFFERENT layers across stages."""
    for scope_group in (None, object()):
        hook = _router_hook_from(dp_scope_group=scope_group)
        recorded: dict = {}

        def _fake_reduce_grad(grad, *, op, divisor, group, fp32, _rec=recorded):
            _rec.update(op=op, divisor=divisor, group=group, fp32=fp32)

        param = nn.Parameter(torch.randn(2, 2))
        param.grad = torch.randn(2, 2)
        with (
            patch(f"{_GS_MOD}.reduce_grad", _fake_reduce_grad),
            patch(f"{_GS_MOD}.should_sync_gradients", return_value=True),
        ):
            hook(param)
        assert recorded["group"] is scope_group, recorded
        assert recorded["divisor"] == 8, recorded  # block width, not global world


class _FakeDTensor:
    """Stand-in patched over ``DTensor`` in the sweep: only the isinstance test and the mesh matter."""

    def __init__(self, ndim: int, size: int):
        self.device_mesh = SimpleNamespace(ndim=ndim, size=lambda: size)


class _FakeParam:
    def __init__(self, data):
        self.requires_grad = True
        self.data = data
        self.grad = SimpleNamespace(_local_tensor=torch.zeros(2))


def _deferred_sweep_host(**ep_attrs):
    """Minimal host running the REAL deferred sweep over one FSDP-sharded non-expert param."""
    ep = SimpleNamespace(
        **{
            "defer_grad_sync": True,
            "expert_replica_group": None,
            "is_deferred_dp": True,
            "ep_group_size": 4,
            "world_size": 8,
            "expert_tp_size": 1,
            "fp32_grad_reduce": False,
            "dp_scope_group": None,
            **ep_attrs,
        }
    )
    return SimpleNamespace(
        _ep_config=ep,
        state=SimpleNamespace(global_step=1),
        _deferred_sweep_last_step=None,
        _get_sharded_expert_param_ids=lambda: set(),
        _get_ep_param_ids=lambda: set(),
        _top_level_model=lambda: SimpleNamespace(
            named_parameters=lambda: [("layers.0.mlp.gate.weight", _FakeParam(_FakeDTensor(1, 4)))]
        ),
    )


def test_deferred_replica_average_refuses_the_world_group_fallback():
    """``group=None`` means the WORLD group, which is never the expert-replica scope.

    Unreachable today — these grads are collected only under ``is_deferred_dp``, which is exactly
    what builds ``expert_replica_group`` — but the fallback is silent where it is wrong: under PP
    the world group spans stages holding different layers, so the reduce averages unrelated
    parameters (or hangs) instead of naming the missing group.
    """
    host = _deferred_sweep_host()
    with (
        patch(f"{_MIXIN_MOD}.DTensor", _FakeDTensor),
        pytest.raises(RuntimeError, match="no expert replica group"),
    ):
        DistributedTrainerMixin._sync_deferred_expert_grads(host)


def test_deferred_replica_average_uses_the_replica_group_when_there_is_one():
    """Anti-vacuity: with the group present the same sweep reduces over it and raises nothing."""
    replica_group = object()
    host = _deferred_sweep_host(expert_replica_group=replica_group)
    reduced: list = []
    with (
        patch(f"{_MIXIN_MOD}.DTensor", _FakeDTensor),
        patch(
            f"{_MIXIN_MOD}.reduce_grads_bucketed",
            lambda grads, **kwargs: reduced.append((len(grads), kwargs.get("group"))),
        ),
    ):
        DistributedTrainerMixin._sync_deferred_expert_grads(host)
    assert (1, replica_group) in reduced, f"the sharded non-expert grad must average over the replicas: {reduced}"


def test_canonical_fsdp_shard_group_falls_back_to_the_stage_not_the_world():
    """No FSDP-sharded backbone param: the norm reduce must stay inside the pipeline stage.

    ``None`` is the default world group, which under PP spans stages holding different layers — the
    chain reduction would then count those norms a second time."""
    stage_group = object()
    host = SimpleNamespace(
        _pp_stage_group=stage_group,
        _fsdp_shard_group=lambda mesh: pytest.fail("no DTensor param exists to derive a group from"),
    )
    model = SimpleNamespace(named_parameters=lambda: [("embed.weight", nn.Parameter(torch.zeros(2)))])

    assert DistributedTrainerMixin._canonical_fsdp_shard_group(host, model, set()) is stage_group


def test_restore_time_zero_lr_step_does_not_run_the_sweep():
    """torch's ``set_optimizer_state_dict`` materializes empty optimizer state via a zero-LR
    ``optimizer.step()`` (``_init_optim_state``). That step fires the registered pre-step hook at
    the RESTORED ``global_step``; without the restore flag the hook runs the deferred sweep there,
    and the first real training step's sweep then trips its own non-idempotency guard."""
    calls: list[int] = []
    param = nn.Parameter(torch.zeros(2))
    fake = SimpleNamespace(
        optimizer=torch.optim.SGD([param], lr=0.0),
        _ep_config=SimpleNamespace(defer_grad_sync=True),
        state=SimpleNamespace(global_step=4),
        _deferred_sweep_last_step=None,
        _sync_deferred_expert_grads=lambda: calls.append(1),
    )
    DistributedTrainerMixin._register_deferred_ep_grad_sync_hook(fake)
    assert fake._deferred_ep_grad_sync_hook_registered

    fake._restoring_optimizer_state = True
    fake.optimizer.step()  # the restore-time _init_optim_state step
    assert calls == [], "the sweep must not run during optimizer-state restore"

    fake._restoring_optimizer_state = False
    fake.optimizer.step()  # the first real step (clip disabled -> backstop path)
    assert calls == [1], "the first real step's sweep must still run exactly once"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
