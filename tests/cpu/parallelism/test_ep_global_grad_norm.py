#!/usr/bin/env python
"""The EP clip's global norm on a 4-rank gloo world over two EP-group replicas. It sums each group's shards and never
the replicas, which the sweep made identical; EP+ETP inside one group needs both legs."""

import datetime
import json
import math
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from halo.trainers.parallelism.grad_clipping import GradientClippingMixin
from halo.trainers.parallelism.grad_sync import GradientSyncMixin
from tests.common.gloo import new_groups, run_gloo_ranks

WORLD = 4
MAX_NORM = 1.0
PG_TIMEOUT = datetime.timedelta(seconds=60)
# ETP partners share a batch, dispatch peers do not.
TOPOLOGIES = {"pure_etp": ("expert_tp_group", WORLD // 2), "plain_ep": ("dispatch_ep_group", WORLD)}


def _expert_grad(rank: int) -> torch.Tensor:
    return torch.full((3,), float(rank + 1))


def _router_grad(rank: int) -> torch.Tensor:
    return torch.full((2,), float(2 * rank))


def _swept_expert(position: int, divisor: int) -> torch.Tensor:
    """SUM over its replicas / divisor."""
    return (_expert_grad(position) + _expert_grad(position + 2)) / divisor


def _reference_norm(divisor: int) -> float:
    """Both in-group shards after the sweep, and the replicated router averaged over the world."""
    shards = [_swept_expert(position, divisor) for position in (0, 1)]
    router = sum(_router_grad(rank) for rank in range(WORLD)) / WORLD
    return math.sqrt(sum(float(shard.pow(2).sum()) for shard in shards) + float(router.pow(2).sum()))


class _Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.expert = nn.Parameter(torch.zeros(3))
        self.router = nn.Parameter(torch.zeros(2))


class _Host(GradientClippingMixin, GradientSyncMixin):
    def __init__(self, model, ep_config):
        self.model = model
        self.optimizer = None
        self.accelerator = SimpleNamespace(clip_grad_norm_=None)
        self.parallelism_config = SimpleNamespace(is_tp_mode=False)
        self.state = SimpleNamespace(global_step=0)
        self._ep_config = ep_config
        self._device_mesh = None
        self._fsdp_wrapped = False
        self._pp_chain_group = None
        self._ep_layers = lambda: [model.expert]
        self._install_ep_clip()

    def _top_level_model(self):
        return self.model

    def _get_sharded_expert_param_ids(self):
        return {id(self.model.expert)}

    def _get_ep_param_ids(self):
        return {id(p) for p in self.model.parameters()}

    def _tp_sharded_plain_param_ids(self):
        return set()


def _worker(rank: int, out: str) -> None:
    in_group = new_groups(rank, [[0, 1], [2, 3]], PG_TIMEOUT)
    replica_group = new_groups(rank, [[0, 2], [1, 3]], PG_TIMEOUT)
    results = {}
    for topology, (group_attr, divisor) in TOPOLOGIES.items():
        ep_config = SimpleNamespace(
            sweep_reduces_across_replicas=True,
            runs_post_backward_sweep=True,
            expert_replica_group=replica_group,
            is_deferred_dp=False,
            stage_world_size=WORLD,
            expert_tp_size=WORLD // divisor,
            expert_tp_group=None,
            dispatch_ep_group=None,
            num_ep_groups=2,
            fp32_grad_reduce=False,
            dp_scope_group=None,
        )
        setattr(ep_config, group_attr, in_group)
        model = _Model()
        model.expert.grad = _expert_grad(rank)
        model.router.grad = _router_grad(rank)
        host = _Host(model, ep_config)

        norm = host.accelerator.clip_grad_norm_(model.parameters(), MAX_NORM)
        results[topology] = {
            "norm": float(norm),
            "expert": model.expert.grad.tolist(),
            "router": model.router.grad.tolist(),
        }
    results["ep_etp"] = _ep_etp(rank, dispatch_group=in_group, expert_tp_group=replica_group)
    with open(f"{out}.{rank}", "w") as fh:
        json.dump(results, fh)


def _ep_etp(rank: int, dispatch_group, expert_tp_group) -> dict:
    """No replicas, so the sweep only divides the experts; the router arrives synced."""
    ep_config = SimpleNamespace(
        sweep_reduces_across_replicas=False,
        runs_post_backward_sweep=True,
        fp32_grad_reduce=False,
        expert_replica_group=None,
        stage_world_size=WORLD,
        expert_tp_size=2,
        expert_tp_group=expert_tp_group,
        dispatch_ep_group=dispatch_group,
        num_ep_groups=1,
    )
    model = _Model()
    model.expert.grad = _expert_grad(rank)
    model.router.grad = sum(_router_grad(r) for r in range(WORLD)) / WORLD
    host = _Host(model, ep_config)
    norm = host.accelerator.clip_grad_norm_(model.parameters(), MAX_NORM)
    return {"norm": float(norm), "expert": model.expert.grad.tolist()}


# world_size // expert_tp_size: the expert-TP partners share a batch.
EP_ETP_DIVISOR = WORLD // 2


def _ep_etp_reference_norm() -> float:
    router = sum(_router_grad(rank) for rank in range(WORLD)) / WORLD
    shards = sum(float((_expert_grad(rank) / EP_ETP_DIVISOR).pow(2).sum()) for rank in range(WORLD))
    return math.sqrt(shards + float(router.pow(2).sum()))


@pytest.fixture(scope="module")
def per_rank(tmp_path_factory):
    out = str(tmp_path_factory.mktemp("norm") / "norm")
    run_gloo_ranks(_worker, WORLD, out, pg_timeout=PG_TIMEOUT)
    results = []
    for rank in range(WORLD):
        with open(f"{out}.{rank}") as fh:
            results.append(json.load(fh))
    return results


@pytest.mark.parametrize("topology", sorted(TOPOLOGIES))
def test_every_rank_clips_by_the_norm_of_the_swept_grads(per_rank, topology):
    divisor = TOPOLOGIES[topology][1]
    reference = _reference_norm(divisor)
    coefficient = MAX_NORM / (reference + 1e-6)
    for rank, results in enumerate(per_rank):
        result = results[topology]
        assert result["norm"] == pytest.approx(reference, rel=1e-6), f"rank {rank}: {result['norm']} vs {reference}"
        swept = _swept_expert(rank % 2, divisor)
        assert result["expert"] == pytest.approx((swept * coefficient).tolist(), rel=1e-5), f"rank {rank}"
    assert per_rank[0][topology]["router"] == per_rank[3][topology]["router"], "the router must clip identically"


def test_ep_etp_sums_the_expert_norm_over_both_legs(per_rank):
    """Either leg alone counts half the shards."""
    reference = _ep_etp_reference_norm()
    coefficient = MAX_NORM / (reference + 1e-6)
    for rank, results in enumerate(per_rank):
        result = results["ep_etp"]
        assert result["norm"] == pytest.approx(reference, rel=1e-6), f"rank {rank}: {result['norm']} vs {reference}"
        expected = _expert_grad(rank) / EP_ETP_DIVISOR * coefficient
        assert result["expert"] == pytest.approx(expected.tolist(), rel=1e-5)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
