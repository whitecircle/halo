#!/usr/bin/env python
"""What a rank's rollout manager is built from. A loopback rollout URL means another engine per node, so the count it is
refused on takes in both ways a run spans nodes: the launch, and a Ray cluster its actors spill over. A head on this host
alone must still pass."""

import types

import pytest

from halo.environments.rollout_pool import RolloutPoolPlan
from halo.trainers.grpo.environmental import async_rollouts
from halo.trainers.grpo.environmental.async_rollouts import AsyncRolloutMixin


def _host(ray_address):
    return types.SimpleNamespace(async_config=types.SimpleNamespace(ray_address=ray_address))


def _count(monkeypatch, *, launch_nodes: int, cluster_nodes: list[bool] | None, ray_address) -> int:
    monkeypatch.setattr(async_rollouts, "get_num_nodes", lambda: launch_nodes)
    nodes = [{"Alive": alive} for alive in cluster_nodes or []]
    monkeypatch.setattr(async_rollouts, "ray", types.SimpleNamespace(nodes=lambda: nodes))
    return AsyncRolloutMixin._rollout_node_count(_host(ray_address))


@pytest.mark.parametrize(
    ("launch_nodes", "cluster_nodes", "ray_address", "expected"),
    [
        (1, None, None, 1),
        (2, None, None, 2),
        (1, [True], "ray://head:10001", 1),
        (1, [True, True, True], "ray://head:10001", 3),
        (1, [True, False, False], "ray://head:10001", 1),
        (2, [True], "ray://head:10001", 2),
    ],
    ids=[
        "one-node",
        "two-node-launch",
        "head-on-this-host",
        "shared-cluster",
        "dead-nodes-do-not-count",
        "launch-wins",
    ],
)
def test_the_node_count(monkeypatch, launch_nodes, cluster_nodes, ray_address, expected):
    assert (
        _count(monkeypatch, launch_nodes=launch_nodes, cluster_nodes=cluster_nodes, ray_address=ray_address)
        == expected
    )


def test_each_rank_starts_its_round_robin_at_its_data_parallel_rank(monkeypatch):
    class _Built(Exception):
        """Past the manager the component start needs actors and an event loop this test does not."""

    captured = {}

    def build(**kwargs):
        captured.update(kwargs)
        raise _Built

    monkeypatch.setattr(async_rollouts, "RolloutManager", build)
    monkeypatch.setattr(async_rollouts, "ray", types.SimpleNamespace(is_initialized=lambda: True))
    host = types.SimpleNamespace(
        async_config=types.SimpleNamespace(
            ray_address=None,
            num_rollout_workers=1,
            max_concurrent_rollouts=1,
            get_server_urls=lambda: ["http://10.0.0.1:8000"],
            get_rollout_config=lambda **kwargs: None,
        ),
        accelerator=types.SimpleNamespace(num_processes=8),
        parallelism_config=types.SimpleNamespace(get_data_parallel_rank=lambda: 5),
        _resolve_rollout_pool=lambda: RolloutPoolPlan(actors_per_rank=1, max_concurrent=1, warnings=()),
        _environment_spec="env",
        _env_config_dict={},
        _rollout_stop_token_ids=None,
        _reasoning_end_ids=None,
    )
    with pytest.raises(_Built):
        AsyncRolloutMixin._start_rollout_pool(host)
    assert captured["first_server"] == 5


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
