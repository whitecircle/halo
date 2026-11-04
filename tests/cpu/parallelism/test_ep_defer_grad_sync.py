#!/usr/bin/env python
"""EPConfig deferred-grad-sync and DP-scope-group contracts. PP forces ``gradient_accumulation_steps`` to 1, so
post-accumulate hooks would fire per microbatch and re-scale synced gradients."""

from __future__ import annotations

from types import MethodType, SimpleNamespace
from unittest.mock import patch

import pytest
import torch
import torch.nn as nn

from halo.distributed.expert_parallel.config import EPConfig
from halo.distributed.expert_parallel.moe_layer.grad_sync import build_dp_average_grad_hook
from halo.distributed.parallelism_config import ParallelismConfig
from halo.trainers.distributed_trainer import DistributedTrainerMixin
from tests.common.parallelism import built_ep_config, make_parallelism_config, stubbed_process_groups

_GS_MOD = "halo.distributed.expert_parallel.moe_layer.grad_sync"
# The sweep's globals, which the deferred-sync tests patch.
_MIXIN_MOD = "halo.trainers.parallelism.grad_sync"


def _make_ep(rank: int, *, world_size: int, stage_world_size: int, fsdp_shard_ep1_experts: bool = True, **kwargs):
    """Returns ``(config, calls)`` where ``calls`` is the ordered ``new_group`` rank-list sequence."""
    calls: list = []
    stage_base_rank = (rank // stage_world_size) * stage_world_size
    managed = fsdp_shard_ep1_experts and kwargs["ep_size"] * kwargs.get("expert_tp_size", 1) == 1
    with stubbed_process_groups(world_size, rank, calls):
        config = EPConfig(
            stage_world_size=stage_world_size, stage_base_rank=stage_base_rank, experts_fsdp_managed=managed, **kwargs
        )
        return config, calls


def test_defer_false_without_pp_single_node():
    """Router hooks stay, the expert divide defers."""
    cfg, _ = _make_ep(0, world_size=8, stage_world_size=8, ep_size=8, nvlink_domain_size=8, node_local=True)
    assert cfg.num_rank_blocks == 1
    assert not cfg.is_deferred_dp
    assert not cfg.sweep_reduces_across_replicas
    assert cfg.uses_backward_grad_hooks
    assert cfg.runs_post_backward_sweep
    assert cfg.dp_scope_group is None


def test_defer_true_for_multinode_deferred_dp_without_pp():
    cfg, _ = _make_ep(0, world_size=16, stage_world_size=16, ep_size=8, nvlink_domain_size=8, node_local=True)
    assert cfg.num_rank_blocks == 1
    assert cfg.is_deferred_dp
    assert cfg.sweep_reduces_across_replicas
    assert cfg.dp_scope_group is None  # Single rank block → default (world) group.


def test_defer_true_under_pp_even_without_deferred_dp():
    """The per-microbatch hook firing forces the deferral."""
    cfg, _ = _make_ep(0, world_size=16, stage_world_size=8, ep_size=8, nvlink_domain_size=8, node_local=True)
    assert cfg.num_rank_blocks == 2
    assert not cfg.is_deferred_dp
    assert cfg.sweep_reduces_across_replicas


def test_defer_false_under_pp_for_fsdp_sharded_ep1_experts():
    """FSDP reduce-scatter is the sole, accumulation-safe sync, so deferring would double-sync."""
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
    """The EP hooks are the only sync, so they must defer."""
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
    """ep_group_size is 2, so the ep1-FSDP exception does not apply despite ep_size == 1."""
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
    """The global norm sums expert shards within the dispatch group, right only if the sweep already made replicas
    identical."""
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
    # A reentrant recompute would repeat the divide on a hook.
    assert cfg.runs_post_backward_sweep == (not cfg.experts_fsdp_managed), (world, stage, ep_size, etp, ep_scope)


@pytest.mark.parametrize("managed", [True, False])
def test_single_process_mode_defaults(managed):
    """The sweep's collectives have no group to run on."""
    cfg = EPConfig(ep_size=1, stage_world_size=1, nvlink_domain_size=1, experts_fsdp_managed=managed)
    assert cfg.num_rank_blocks == 1
    assert not cfg.sweep_reduces_across_replicas
    assert not cfg.runs_post_backward_sweep
    assert cfg.dp_scope_group is None


def test_dp_scope_groups_block_confined_and_identical_order():
    """Every rank creates both blocks' groups in one order and keeps its own."""
    built = {
        rank: _make_ep(rank, world_size=16, stage_world_size=8, ep_size=8, nvlink_domain_size=8, node_local=True)
        for rank in range(16)
    }
    for rank, (cfg, calls) in built.items():
        assert cfg.dp_scope_group == tuple(range((rank // 8) * 8, (rank // 8) * 8 + 8)), (rank, cfg.dp_scope_group)
        # The last two new_group calls, block-major, on every rank.
        assert calls[-2:] == [tuple(range(8)), tuple(range(8, 16))], (rank, calls[-2:])
    reference = built[0][1]
    for rank, (_, calls) in built.items():
        assert calls == reference, (rank, calls, reference)


def test_pp1_new_group_sequence_bit_identical_to_pre_pp():
    cfg, calls = _make_ep(3, world_size=8, stage_world_size=8, ep_size=2, nvlink_domain_size=8, node_local=True)
    assert cfg.dp_scope_group is None
    assert calls == [(0, 1), (2, 3), (4, 5), (6, 7), (0, 2, 4, 6), (1, 3, 5, 7)], calls


def _router_hook_from(**ep_attrs):
    # Every attribute a real EPConfig sets: a getattr default would mask a typo into no expert grad sync.
    defaults = {
        "stage_world_size": 8,
        "expert_tp_size": 1,
        "needs_expert_grad_sync": False,
        "fp32_grad_reduce": False,
        "dp_scope_group": None,
    }
    return build_dp_average_grad_hook(SimpleNamespace(**{**defaults, **ep_attrs}))


def test_router_hook_reduces_over_dp_scope_group():
    """A world-group all-reduce under PP would blend gradients of different layers across stages."""
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
        assert recorded["divisor"] == 8, recorded  # Block width, not global world.


class _FakeDTensor:
    """Only the isinstance test and the mesh matter."""

    def __init__(self, ndim: int, size: int):
        self.device_mesh = SimpleNamespace(ndim=ndim, size=lambda: size)


class _FakeParam:
    def __init__(self, data):
        self.requires_grad = True
        self.data = data
        self.grad = SimpleNamespace(_local_tensor=torch.zeros(2))


def _deferred_sweep_host(**ep_attrs):
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
    """Under PP the world group spans stages holding different layers, so the reduce averages unrelated parameters
    or hangs."""
    host = _deferred_sweep_host()
    with (
        patch(f"{_MIXIN_MOD}.DTensor", _FakeDTensor),
        pytest.raises(RuntimeError, match="no expert replica group"),
    ):
        DistributedTrainerMixin._run_deferred_grad_sweep(host)


def test_the_sweep_refuses_a_missing_ep_config():
    """Returning would skip the cross-replica average and let expert replicas drift."""
    host = _deferred_sweep_host()
    host._ep_config = None
    with pytest.raises(RuntimeError, match="no EPConfig was captured"):
        DistributedTrainerMixin._run_deferred_grad_sweep(host)


def test_the_etp_generation_broadcast_refuses_a_missing_ep_config():
    """``None`` would skip the broadcast and let expert-TP partners generate different batches."""
    host = SimpleNamespace(
        parallelism_config=SimpleNamespace(is_tp_mode=False, is_expert_tp_mode=True), _ep_config=None
    )
    with pytest.raises(RuntimeError, match="no EPConfig was captured"):
        DistributedTrainerMixin._get_tp_or_etp_process_group(host)


def test_deferred_replica_average_uses_the_replica_group_when_there_is_one():
    replica_group = object()
    host = _deferred_sweep_host(expert_replica_group=replica_group)
    reduced: list = []
    with (
        patch(f"{_MIXIN_MOD}.DTensor", _FakeDTensor),
        # The membership mask's MAX over a group this stub cannot reduce.
        patch(f"{_MIXIN_MOD}.dist.all_reduce"),
        patch(
            f"{_MIXIN_MOD}.reduce_grads_bucketed",
            lambda grads, **kwargs: reduced.append((len(grads), kwargs.get("group"))),
        ),
    ):
        DistributedTrainerMixin._run_deferred_grad_sweep(host)
    assert (1, replica_group) in reduced, f"the sharded non-expert grad must average over the replicas: {reduced}"


def test_the_fsdp_norm_reduce_falls_back_to_the_stage_not_the_world():
    """The world group would count other stages' norms a second time in the chain reduction."""
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
    """``global_step`` restarts at 0 with every ``train()``; an earlier sweep must not stop step 0, while a second
    sweep within one step still raises."""
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
    DistributedTrainerMixin._run_deferred_grad_sweep(host)
    with pytest.raises(RuntimeError, match="ran twice"):
        DistributedTrainerMixin._run_deferred_grad_sweep(host)
    host.state = SimpleNamespace(global_step=0)
    expert.grad = torch.full((2,), 8.0)
    DistributedTrainerMixin._run_deferred_grad_sweep(host)
    assert torch.equal(expert.grad, torch.full((2,), 2.0))


def test_restore_time_zero_lr_step_does_not_run_the_sweep():
    """``_init_optim_state``'s zero-LR step fires the pre-step hook at the restored ``global_step``, which would
    trip the sweep's non-idempotency guard."""
    calls: list[int] = []
    param = nn.Parameter(torch.zeros(2))
    fake = SimpleNamespace(
        optimizer=torch.optim.SGD([param], lr=0.0),
        _ep_config=SimpleNamespace(sweep_reduces_across_replicas=True, runs_post_backward_sweep=True),
        state=SimpleNamespace(global_step=4),
        _deferred_sweep_last_step=None,
        _run_deferred_grad_sweep=lambda: calls.append(1),
        parallelism_config=SimpleNamespace(is_tp_mode=False),
        _sync_qlora_grads=lambda: None,
    )
    DistributedTrainerMixin._register_grad_sync_step_hooks(fake)
    assert fake._deferred_ep_grad_sync_hook_registered

    fake._materializing_optimizer_state = True
    fake.optimizer.step()  # The restore-time _init_optim_state step.
    assert calls == [], "the sweep must not run during optimizer-state restore"

    fake._materializing_optimizer_state = False
    fake.optimizer.step()  # The first real step (clip disabled -> backstop path).
    assert calls == [1], "the first real step's sweep must still run exactly once"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
