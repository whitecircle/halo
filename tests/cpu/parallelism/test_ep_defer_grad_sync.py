#!/usr/bin/env python
"""EPConfig deferred-grad-sync + DP-scope-group contracts (pipeline-parallel rank blocks).

``sweep_reduces_across_replicas`` decides whether the router and replicated EP params ride per-backward hooks or the
trainer's post-backward sweep. It must be a strict superset of ``is_deferred_dp``: any rank-blocked job
(``num_rank_blocks > 1``, i.e. pipeline parallelism) also defers, because PP forces
``gradient_accumulation_steps`` to 1 so post-accumulate hooks would fire after EVERY microbatch
backward and re-scale already-synced gradients. The one exception is ep1 experts sharded by FSDP
(``fsdp_shard_ep1_experts`` + ``ep_group_size == 1``), whose reduce-scatter is accumulation-safe.
``runs_post_backward_sweep`` widens it to every FSDP-ignored expert topology with peers: the expert divide
runs in the sweep even where the router keeps its hook.

``dp_scope_group`` is the collective domain for router/replicated-param DP averages under PP (one
group per rank block); ``None`` without PP keeps the non-PP collective sequence bit-identical.

These tests drive the REAL ``EPConfig`` and the grad-hook factories built off it, with
``torch.distributed`` stubbed (``new_group`` records and returns the rank tuple), following
``test_pp_group_construction.py``.

Run: python tests/cpu/parallelism/test_ep_defer_grad_sync.py
"""

from __future__ import annotations

from types import MethodType, SimpleNamespace
from unittest.mock import patch

import pytest
import torch
import torch.nn as nn

from halo.distributed.expert_parallel.config import EPConfig
from halo.distributed.expert_parallel.moe_layer.grad_sync import build_router_grad_hook
from halo.distributed.parallelism_config import ParallelismConfig
from halo.trainers.distributed_trainer import DistributedTrainerMixin
from tests.common.parallelism import built_ep_config, make_parallelism_config, stubbed_process_groups

_GS_MOD = "halo.distributed.expert_parallel.moe_layer.grad_sync"
# The trainer-side sweep, whose globals the deferred-sync tests patch.
_MIXIN_MOD = "halo.trainers.parallelism.grad_sync"


def _make_ep(rank: int, *, world_size: int, stage_world_size: int, fsdp_shard_ep1_experts: bool = True, **kwargs):
    """Build the REAL ``EPConfig`` for ``rank`` of a ``world_size`` job split into stage-wide blocks, with the
    ``experts_fsdp_managed`` verdict ``ParallelismConfig`` passes for ``fsdp_shard_ep1_experts``.

    Returns ``(config, calls)`` where ``calls`` is the ordered ``new_group`` rank-list sequence."""
    calls: list = []
    stage_base_rank = (rank // stage_world_size) * stage_world_size
    managed = fsdp_shard_ep1_experts and kwargs["ep_size"] * kwargs.get("expert_tp_size", 1) == 1
    with stubbed_process_groups(world_size, rank, calls):
        config = EPConfig(
            stage_world_size=stage_world_size, stage_base_rank=stage_base_rank, experts_fsdp_managed=managed, **kwargs
        )
        return config, calls


# sweep_reduces_across_replicas truth table


def test_defer_false_without_pp_single_node():
    """pp1, single node, single EP group: the router hooks stay (defer False) and the expert divide defers,
    with no DP-scope group."""
    cfg, _ = _make_ep(0, world_size=8, stage_world_size=8, ep_size=8, nvlink_domain_size=8, node_local=True)
    assert cfg.num_rank_blocks == 1
    assert not cfg.is_deferred_dp
    assert not cfg.sweep_reduces_across_replicas
    assert cfg.uses_backward_grad_hooks
    assert cfg.runs_post_backward_sweep
    assert cfg.dp_scope_group is None


def test_defer_true_for_multinode_deferred_dp_without_pp():
    """pp1 multi-node multi-group EP (is_deferred_dp) implies sweep_reduces_across_replicas — superset check."""
    cfg, _ = _make_ep(0, world_size=16, stage_world_size=16, ep_size=8, nvlink_domain_size=8, node_local=True)
    assert cfg.num_rank_blocks == 1
    assert cfg.is_deferred_dp
    assert cfg.sweep_reduces_across_replicas
    assert cfg.dp_scope_group is None  # single rank block → default (world) group


def test_defer_true_under_pp_even_without_deferred_dp():
    """PP (2 rank blocks), single-node single-group EP: is_deferred_dp is False, but the
    per-microbatch hook firing forces the deferral anyway."""
    cfg, _ = _make_ep(0, world_size=16, stage_world_size=8, ep_size=8, nvlink_domain_size=8, node_local=True)
    assert cfg.num_rank_blocks == 2
    assert not cfg.is_deferred_dp
    assert cfg.sweep_reduces_across_replicas


def test_defer_false_under_pp_for_fsdp_sharded_ep1_experts():
    """PP + ep1 + fsdp_shard_ep1_experts: no EP hooks exist at all (FSDP reduce-scatter is the sole,
    accumulation-safe sync) — deferring would double-sync. defer must stay False."""
    cfg, _ = _make_ep(
        0,
        world_size=16,
        stage_world_size=8,
        ep_size=1,
        nvlink_domain_size=8,
        node_local=True,
        fsdp_shard_ep1_experts=True,
    )
    assert cfg.num_rank_blocks == 2
    assert cfg.ep_group_size == 1
    assert not cfg.sweep_reduces_across_replicas


def test_defer_true_under_pp_for_unsharded_ep1_experts():
    """PP + ep1 WITHOUT the FSDP sharding: the EP hooks are the only sync, so they must defer."""
    cfg, _ = _make_ep(
        0,
        world_size=16,
        stage_world_size=8,
        ep_size=1,
        nvlink_domain_size=8,
        node_local=True,
        fsdp_shard_ep1_experts=False,
    )
    assert cfg.sweep_reduces_across_replicas


def test_defer_true_under_pp_for_pure_etp():
    """PP + pure ETP (ep_size=1, expert_tp_size=2): ep_group_size is 2, so the ep1-FSDP exception
    must NOT apply even though ep_size == 1 and fsdp_shard_ep1_experts is on."""
    cfg, _ = _make_ep(
        0,
        world_size=16,
        stage_world_size=8,
        ep_size=1,
        expert_tp_size=2,
        nvlink_domain_size=8,
        node_local=True,
        fsdp_shard_ep1_experts=True,
    )
    assert cfg.ep_group_size == 2
    assert cfg.sweep_reduces_across_replicas


def _parallelism_config(world, stage, ep_size, etp, ep_scope, fsdp_shard_ep1_experts) -> ParallelismConfig:
    return make_parallelism_config(
        world_size=world,
        pp_size=world // stage,
        ep_size=ep_size,
        expert_tp_size=etp,
        ep_scope=ep_scope,
        fsdp_shard_ep1_experts=fsdp_shard_ep1_experts,
    )


def _admitted(*topology) -> bool:
    """Whether ``ParallelismConfig``, the gate every run's ``EPConfig`` is built behind, admits the shape."""
    try:
        _parallelism_config(*topology)
    except ValueError:
        return False
    return True


_TOPOLOGIES = [
    # (world, stage, ep_size, expert_tp_size, ep_scope, fsdp_shard_ep1_experts) of every shape the config admits.
    (world, stage, ep, etp, ep_scope, managed)
    for world, stage in ((8, 8), (16, 16), (16, 8), (32, 16))
    for ep, etp in ((1, 1), (1, 2), (2, 1), (4, 2), (8, 1))
    for ep_scope in ("node", "global")
    for managed in (True, False)
    if _admitted(world, stage, ep, etp, ep_scope, managed)
]


@pytest.mark.parametrize(("world", "stage", "ep_size", "etp", "ep_scope", "managed"), _TOPOLOGIES)
def test_every_topology_with_expert_replicas_defers_their_sync(world, stage, ep_size, etp, ep_scope, managed):
    """The global norm sums expert shards within the dispatch group only, which is right only if the
    clip's sweep has already made every replica's expert grads identical: so every topology with
    expert replicas (more than one EP group) defers, unless FSDP2 owns the experts — and then they
    are DTensors outside the expert bucket, normed over their FSDP shard group instead."""
    cfg = built_ep_config(_parallelism_config(world, stage, ep_size, etp, ep_scope, managed))
    if cfg.num_ep_groups > 1:
        assert cfg.sweep_reduces_across_replicas or cfg.experts_fsdp_managed, (
            world,
            stage,
            ep_size,
            etp,
            ep_scope,
            managed,
        )
    if cfg.experts_fsdp_managed:
        assert cfg.ep_group_size == 1
    # The expert divide, which a reentrant recompute would repeat on a hook, defers wherever FSDP2 leaves
    # the experts out.
    assert cfg.runs_post_backward_sweep == (not cfg.experts_fsdp_managed), (world, stage, ep_size, etp, ep_scope)


@pytest.mark.parametrize("managed", [True, False])
def test_single_process_mode_defaults(managed):
    """Bare ``python`` launch (dist never initialized): one block, no deferral, world-scope group; with the
    experts left out of FSDP2 too, since the sweep's collectives have no group to run on."""
    cfg = EPConfig(ep_size=1, stage_world_size=1, nvlink_domain_size=1, experts_fsdp_managed=managed)
    assert cfg.num_rank_blocks == 1
    assert not cfg.sweep_reduces_across_replicas
    assert not cfg.runs_post_backward_sweep
    assert cfg.dp_scope_group is None


# dp_scope_group construction discipline


def test_dp_scope_groups_block_confined_and_identical_order():
    """Every rank creates BOTH blocks' DP-scope groups in one order and keeps exactly its own."""
    built = {
        rank: _make_ep(rank, world_size=16, stage_world_size=8, ep_size=8, nvlink_domain_size=8, node_local=True)
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
    cfg, calls = _make_ep(3, world_size=8, stage_world_size=8, ep_size=2, nvlink_domain_size=8, node_local=True)
    assert cfg.dp_scope_group is None
    assert calls == [(0, 1), (2, 3), (4, 5), (6, 7), (0, 2, 4, 6), (1, 3, 5, 7)], calls


# Router-hook plumbing: the DP-scope group and divisor come straight off EPConfig


def _router_hook_from(**ep_attrs):
    # The stub carries every attribute a real EPConfig sets: the hook factory reads them directly, so
    # a getattr default would mask a typo into "no expert grad sync".
    defaults = {
        "stage_world_size": 8,
        "expert_tp_size": 1,
        "needs_expert_grad_sync": False,
        "fp32_grad_reduce": False,
        "dp_scope_group": None,
    }
    ep = SimpleNamespace(**{**defaults, **ep_attrs})
    with patch(f"{_GS_MOD}.has_process_group", return_value=True):
        return build_router_grad_hook(ep)


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
            "sweep_reduces_across_replicas": True,
            "runs_post_backward_sweep": True,
            "expert_replica_group": None,
            "is_deferred_dp": True,
            "ep_group_size": 4,
            "stage_world_size": 8,
            "expert_tp_size": 1,
            "fp32_grad_reduce": False,
            "dp_scope_group": None,
            **ep_attrs,
        }
    )
    host = SimpleNamespace(
        _ep_config=ep,
        state=SimpleNamespace(global_step=1),
        _deferred_sweep_last_step=None,
        _get_sharded_expert_param_ids=lambda: set(),
        _get_ep_param_ids=lambda: set(),
        _top_level_model=lambda: SimpleNamespace(
            named_parameters=lambda: [("layers.0.mlp.gate.weight", _FakeParam(_FakeDTensor(1, 4)))]
        ),
    )
    host._classify_deferred_params = MethodType(DistributedTrainerMixin._classify_deferred_params, host)
    return host


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


def test_the_sweep_refuses_a_missing_ep_config():
    """Both callers run only with EP layers present, so a missing config is a setup that never
    captured it; returning would skip the cross-replica average and let expert replicas drift."""
    host = _deferred_sweep_host()
    host._ep_config = None
    with pytest.raises(RuntimeError, match="no EPConfig was captured"):
        DistributedTrainerMixin._sync_deferred_expert_grads(host)


def test_the_etp_generation_broadcast_refuses_a_missing_ep_config():
    """Under ETP the group comes off the EP config; ``None`` would skip the broadcast and let the
    expert-TP partners generate different batches."""
    host = SimpleNamespace(
        parallelism_config=SimpleNamespace(is_tp_mode=False, is_expert_tp_mode=True), _ep_config=None
    )
    with pytest.raises(RuntimeError, match="no EPConfig was captured"):
        DistributedTrainerMixin._get_tp_or_etp_process_group(host)


def test_deferred_replica_average_uses_the_replica_group_when_there_is_one():
    """Anti-vacuity: with the group present the same sweep reduces over it and raises nothing."""
    replica_group = object()
    host = _deferred_sweep_host(expert_replica_group=replica_group)
    reduced: list = []
    with (
        patch(f"{_MIXIN_MOD}.DTensor", _FakeDTensor),
        # The membership mask's MAX over a group this stub cannot reduce: the one grad is present.
        patch(f"{_MIXIN_MOD}.dist.all_reduce"),
        patch(
            f"{_MIXIN_MOD}.reduce_grads_bucketed",
            lambda grads, **kwargs: reduced.append((len(grads), kwargs.get("group"))),
        ),
    ):
        DistributedTrainerMixin._sync_deferred_expert_grads(host)
    assert (1, replica_group) in reduced, f"the sharded non-expert grad must average over the replicas: {reduced}"


def test_the_fsdp_norm_reduce_falls_back_to_the_stage_not_the_world():
    """No FSDP-sharded backbone param: the norm reduce must stay inside the pipeline stage.

    ``None`` is the default world group, which under PP spans stages holding different layers — the
    chain reduction would then count those norms a second time."""
    stage_group = object()
    model = nn.Linear(2, 2)
    host = SimpleNamespace(
        _pp_stage_group=stage_group,
        _pp_chain_group=None,
        _device_mesh=None,
        _fsdp_wrapped=True,
        _ep_config=None,
        _top_level_model=lambda: model,
        _get_sharded_expert_param_ids=lambda: set(),
        _tp_sharded_plain_param_ids=lambda: set(),
        _fsdp_shard_group=lambda mesh: pytest.fail("no DTensor param exists to derive a group from"),
    )
    host._bucket_grad_shards = MethodType(DistributedTrainerMixin._bucket_grad_shards, host)
    groups: list = []
    with patch(
        "halo.trainers.parallelism.grad_clipping.dist.all_reduce", lambda t, op=None, group=None: groups.append(group)
    ):
        DistributedTrainerMixin._compute_global_grad_norm(host, list(model.parameters()))
    assert groups == [stage_group]


def test_each_run_sweeps_its_own_first_step():
    """``global_step`` restarts at 0 with every ``train()``'s fresh state: a sweep taken before it (a hand-called
    optimizer step) must neither stop the run's own step 0 from sweeping nor make it raise, while a second sweep
    within one step still raises."""
    expert = nn.Parameter(torch.zeros(2))
    model = nn.Module()
    model.expert = expert
    host = SimpleNamespace(
        _ep_config=SimpleNamespace(
            sweep_reduces_across_replicas=False,
            runs_post_backward_sweep=True,
            expert_replica_group=None,
            stage_world_size=4,
            expert_tp_size=1,
            fp32_grad_reduce=False,
        ),
        state=SimpleNamespace(global_step=0),
        _get_sharded_expert_param_ids=lambda: {id(expert)},
        _get_ep_param_ids=lambda: {id(expert)},
        _top_level_model=lambda: model,
    )
    host._classify_deferred_params = MethodType(DistributedTrainerMixin._classify_deferred_params, host)
    expert.grad = torch.full((2,), 8.0)
    DistributedTrainerMixin._sync_deferred_expert_grads(host)
    with pytest.raises(RuntimeError, match="ran twice"):
        DistributedTrainerMixin._sync_deferred_expert_grads(host)
    host.state = SimpleNamespace(global_step=0)
    expert.grad = torch.full((2,), 8.0)
    DistributedTrainerMixin._sync_deferred_expert_grads(host)
    assert torch.equal(expert.grad, torch.full((2,), 2.0))


def test_restore_time_zero_lr_step_does_not_run_the_sweep():
    """torch's ``set_optimizer_state_dict`` materializes empty optimizer state via a zero-LR
    ``optimizer.step()`` (``_init_optim_state``). That step fires the registered pre-step hook at
    the RESTORED ``global_step``; without the restore flag the hook runs the deferred sweep there,
    and the first real training step's sweep then trips its own non-idempotency guard."""
    calls: list[int] = []
    param = nn.Parameter(torch.zeros(2))
    fake = SimpleNamespace(
        optimizer=torch.optim.SGD([param], lr=0.0),
        _ep_config=SimpleNamespace(sweep_reduces_across_replicas=True, runs_post_backward_sweep=True),
        state=SimpleNamespace(global_step=4),
        _deferred_sweep_last_step=None,
        _sync_deferred_expert_grads=lambda: calls.append(1),
        parallelism_config=SimpleNamespace(is_tp_mode=False),
        _sync_qlora_grads=lambda: None,
    )
    DistributedTrainerMixin._register_grad_sync_step_hooks(fake)
    assert fake._deferred_ep_grad_sync_hook_registered

    fake._materializing_optimizer_state = True
    fake.optimizer.step()  # the restore-time _init_optim_state step
    assert calls == [], "the sweep must not run during optimizer-state restore"

    fake._materializing_optimizer_state = False
    fake.optimizer.step()  # the first real step (clip disabled -> backstop path)
    assert calls == [1], "the first real step's sweep must still run exactly once"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
