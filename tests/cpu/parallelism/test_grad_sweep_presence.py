#!/usr/bin/env python
"""The three post-backward grad sweeps agree grad presence over their reduce group, on 2 gloo ranks.

Grad presence is rank-local, so each sweep (QLoRA's world DP average, the TP replicated sweep, the
deferred-EP cross-replica sweep) agrees which params enter its collective first
(``agree_grad_presence``). Two properties, per sweep:

* a param one rank produced a grad for enters on every rank, zero-filled where absent, so the
  reduced value is the true average and no rank skips a collective its peer enters;
* a param NO rank touched keeps ``grad is None`` everywhere: a materialized zero would hand AdamW a
  weight-decay and momentum step on a parameter the step should have left alone;
* presence is agreed over the sweep's own reduce group, never a wider one: on 4 ranks with
  2-rank groups, a param only rank 0 touched is zero-filled on its group peer and stays ``None``
  on the other group's ranks — for the expert, router and TP sweeps, and for an FSDP-sharded
  DTensor on the deferred sweep's replica leg;
* a rank holding no grad at all still enters the mask's all-reduce its group peer is in.

Every run carries a 60 s collective timeout on the default group and on every subgroup it builds,
so a rank skipping the mask's all-reduce fails fast instead of waiting out the spawn bound.

    python tests/cpu/parallelism/test_grad_sweep_presence.py
"""

import datetime
import json
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import DTensor, Shard

from src.trainers.mixins.base import DistributedTrainerMixin
from tests.common.gloo import new_groups, run_gloo_ranks

WORLD = 2
GROUPED_WORLD = 4
PG_TIMEOUT = datetime.timedelta(seconds=60)


class _Model(nn.Module):
    """``both`` gets a grad on every rank, ``rank0`` on rank 0 only, ``untouched`` on none."""

    def __init__(self):
        super().__init__()
        self.both = nn.Parameter(torch.zeros(2))
        self.rank0 = nn.Parameter(torch.zeros(2))
        self.untouched = nn.Parameter(torch.zeros(2))


def _backward(model: _Model, rank: int) -> None:
    model.both.grad = torch.full((2,), float(rank + 1))
    if rank == 0:
        model.rank0.grad = torch.full((2,), 4.0)


def _qlora(model: _Model) -> None:
    host = SimpleNamespace(
        _qlora_grad_sync=True,
        state=SimpleNamespace(global_step=0),
        parallelism_config=SimpleNamespace(fp32_grad_reduce=False),
        _top_level_model=lambda: model,
    )
    DistributedTrainerMixin._sync_qlora_grads(host)


def _tp(model: _Model, tp_group=None) -> None:
    host = SimpleNamespace(
        state=SimpleNamespace(global_step=0),
        parallelism_config=SimpleNamespace(fp32_grad_reduce=False),
        parallel_dims=SimpleNamespace(tp_group=lambda: dist.group.WORLD if tp_group is None else tp_group),
        _tp_sharded_plain_param_ids=lambda: set(),
        _tp_per_head_norm_param_ids=lambda: set(),
    )
    DistributedTrainerMixin._sync_tp_replicated_grads(host, list(model.parameters()))


def _deferred_ep(model: _Model, *, experts: bool, replica_group, dp_scope_group=None, world_size=WORLD) -> None:
    """The sweep over ``model``'s params as expert shards (SUM / replicas) or as routers (AVG)."""
    ids = {id(p) for p in model.parameters()}
    host = SimpleNamespace(
        _ep_config=SimpleNamespace(
            defer_grad_sync=True,
            expert_replica_group=replica_group,
            is_deferred_dp=False,
            world_size=world_size,
            expert_tp_size=1,
            fp32_grad_reduce=False,
            dp_scope_group=dp_scope_group,
        ),
        state=SimpleNamespace(global_step=0),
        _get_sharded_expert_param_ids=lambda: ids if experts else set(),
        _get_ep_param_ids=lambda: ids,
        _top_level_model=lambda: model,
    )
    DistributedTrainerMixin._sync_deferred_expert_grads(host)


SWEEPS = {
    "qlora": _qlora,
    "tp": _tp,
    "deferred_ep_experts": lambda model: _deferred_ep(model, experts=True, replica_group=dist.group.WORLD),
    "deferred_ep_routers": lambda model: _deferred_ep(model, experts=False, replica_group=dist.group.WORLD),
}


def _worker(rank: int, out: str) -> None:
    results = {}
    for name, sweep in SWEEPS.items():
        model = _Model()
        _backward(model, rank)
        sweep(model)
        results[name] = {
            param_name: None if param.grad is None else param.grad.tolist()
            for param_name, param in model.named_parameters()
        }
    # R == 1: one EP group spans the replicas, so the expert leg is a local divide with no collective.
    model = _Model()
    _backward(model, rank)
    _deferred_ep(model, experts=True, replica_group=None)
    results["deferred_ep_one_replica"] = {
        param_name: None if param.grad is None else param.grad.tolist()
        for param_name, param in model.named_parameters()
    }
    with open(f"{out}.{rank}", "w") as fh:
        json.dump(results, fh)


@pytest.fixture(scope="module")
def per_rank(tmp_path_factory):
    out = str(tmp_path_factory.mktemp("sweeps") / "grads")
    run_gloo_ranks(_worker, WORLD, out, pg_timeout=PG_TIMEOUT)
    results = []
    for rank in range(WORLD):
        with open(f"{out}.{rank}") as fh:
            results.append(json.load(fh))
    return results


@pytest.mark.parametrize("sweep", sorted(SWEEPS))
def test_a_param_one_rank_touched_is_averaged_on_every_rank(per_rank, sweep):
    for rank, results in enumerate(per_rank):
        grads = results[sweep]
        assert grads["both"] == [1.5, 1.5], f"rank {rank}: {grads}"
        assert grads["rank0"] == [2.0, 2.0], f"rank {rank}: the absent contribution must be a zero ({grads})"


@pytest.mark.parametrize("sweep", [*sorted(SWEEPS), "deferred_ep_one_replica"])
def test_a_param_no_rank_touched_keeps_no_grad(per_rank, sweep):
    for rank, results in enumerate(per_rank):
        assert per_rank[rank][sweep]["untouched"] is None, f"rank {rank}: {sweep} materialized a zero grad"


def test_one_replica_divides_locally_and_fills_nothing(per_rank):
    """With no replica group the combine already summed the replicas; nothing is zero-filled."""
    rank1 = per_rank[1]["deferred_ep_one_replica"]
    assert rank1["rank0"] is None, "a grad this rank never produced must not be materialized locally"
    assert per_rank[0]["deferred_ep_one_replica"]["rank0"] == [2.0, 2.0]


def _replica_leg(rank: int, ep_group, replica_group) -> list[float] | None:
    """A non-expert FSDP shard over the EP group under multi-node deferred DP: the sweep averages its
    local shard over the expert replicas. Only rank 0 produced a grad for it."""
    mesh = DeviceMesh.from_group(ep_group, "cpu")
    model = nn.Module()
    model.shard = nn.Parameter(DTensor.from_local(torch.zeros(2), mesh, [Shard(0)], run_check=False))
    if rank == 0:
        model.shard.grad = DTensor.from_local(torch.full((2,), 4.0), mesh, [Shard(0)], run_check=False)
    host = SimpleNamespace(
        _ep_config=SimpleNamespace(
            defer_grad_sync=True,
            expert_replica_group=replica_group,
            is_deferred_dp=True,
            ep_group_size=2,
            world_size=GROUPED_WORLD,
            expert_tp_size=1,
            fp32_grad_reduce=False,
            dp_scope_group=None,
        ),
        state=SimpleNamespace(global_step=0),
        _get_sharded_expert_param_ids=lambda: set(),
        _get_ep_param_ids=lambda: set(),
        _top_level_model=lambda: model,
    )
    DistributedTrainerMixin._sync_deferred_expert_grads(host)
    return None if model.shard.grad is None else model.shard.grad.to_local().tolist()


def _grad_less_rank(rank: int, tp_group) -> list[float] | None:
    """The TP sweep over a pair in which rank 1 and rank 3 produced no grad for any param."""
    model = _Model()
    if rank in (0, 2):
        model.both.grad = torch.full((2,), float(rank + 1))
    _tp(model, tp_group=tp_group)
    return None if model.both.grad is None else model.both.grad.tolist()


def _grouped_worker(rank: int, out: str) -> None:
    """Two EP groups [0, 1], [2, 3] whose expert replicas are [0, 2], [1, 3]; the TP groups and the
    pipeline-stage DP scopes are [0, 1], [2, 3]. Only rank 0 touches ``rank0``."""
    pairs = new_groups(rank, [[0, 1], [2, 3]], PG_TIMEOUT)
    replicas = new_groups(rank, [[0, 2], [1, 3]], PG_TIMEOUT)
    results = {}
    for name, sweep in {
        "expert_over_replicas": lambda m: _deferred_ep(m, experts=True, replica_group=replicas, world_size=4),
        "router_over_stage": lambda m: _deferred_ep(
            m, experts=False, replica_group=replicas, dp_scope_group=pairs, world_size=2
        ),
        "tp_over_pair": lambda m: _tp(m, tp_group=pairs),
    }.items():
        model = _Model()
        _backward(model, rank)
        sweep(model)
        results[name] = None if model.rank0.grad is None else model.rank0.grad.tolist()
    results["replica_leg_over_replicas"] = _replica_leg(rank, pairs, replicas)
    results["grad_less_rank_joins"] = _grad_less_rank(rank, pairs)
    with open(f"{out}.{rank}", "w") as fh:
        json.dump(results, fh)


def test_presence_is_agreed_over_the_reduce_group_not_the_world(tmp_path):
    """A mask agreed world-wide (or over another sweep's group) would zero-fill ``rank0`` on ranks
    whose group never touched it, stepping weight decay onto a param their replicas left alone."""
    out = str(tmp_path / "grouped")
    run_gloo_ranks(_grouped_worker, GROUPED_WORLD, out, pg_timeout=PG_TIMEOUT)
    results = []
    for rank in range(GROUPED_WORLD):
        with open(f"{out}.{rank}") as fh:
            results.append(json.load(fh))
    expected = {
        # rank 0's 4.0 summed with rank 2's zero over the replica pair, divided by the 4-rank world
        "expert_over_replicas": [[1.0, 1.0], None, [1.0, 1.0], None],
        # averaged over the stage pair [0, 1]; the other stage never saw it
        "router_over_stage": [[2.0, 2.0], [2.0, 2.0], None, None],
        "tp_over_pair": [[2.0, 2.0], [2.0, 2.0], None, None],
        # rank 0's local 4.0 averaged with rank 2's zero over the replica pair; ranks 1 and 3 untouched
        "replica_leg_over_replicas": [[2.0, 2.0], None, [2.0, 2.0], None],
        # each pair averages its one grad with its grad-less peer's zero
        "grad_less_rank_joins": [[0.5, 0.5], [0.5, 0.5], [1.5, 1.5], [1.5, 1.5]],
    }
    for name, per_rank_grad in expected.items():
        assert [results[rank][name] for rank in range(GROUPED_WORLD)] == per_rank_grad, name


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
