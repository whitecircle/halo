#!/usr/bin/env python
"""Rollout metrics are sliced by every categorical fact an episode carries: the effort level and whatever an env
stamps under ``info["slices"]``. Each value gets the same breakdown under ``<slice>/<value>/*``."""

import types
from collections import defaultdict

import pytest

from halo.environments.episode import RolloutResult
from halo.environments.episode_keys import EPISODE_SLICES_KEY
from halo.environments.messages import Trajectory
from halo.rewards.terms import REWARD_COMPONENT_PREFIX
from halo.trainers import world_metrics
from halo.trainers.grpo.environmental import rollout_metrics
from halo.trainers.grpo.environmental.trainer import DistributedAsyncEnvironmentalGRPOTrainer
from tests.common.world_metrics import pooled_states, rank_states


def _host():
    host = object.__new__(DistributedAsyncEnvironmentalGRPOTrainer)
    host._metrics = {"train": defaultdict(list)}
    host._total_rollouts = 0
    host._total_rollout_latency = 0.0
    host._total_generation_tokens = 0
    host.async_config = types.SimpleNamespace(truncation_alarm_rate=None)
    return host


def _result(effort, slices, solve, test_calls, reward=1.0, tokens=100):
    traj = Trajectory()
    traj.reasoning_effort = effort
    if slices is not None:
        traj.info[EPISODE_SLICES_KEY] = slices
    return RolloutResult(
        prompt="p",
        trajectory=traj,
        episode_length=2,
        total_reward=reward,
        terminated_naturally=True,
        latency=1.0,
        generation_tokens=tokens,
        metrics={"outcome/solve_rate": solve, "episode/test_calls": test_calls},
    )


def test_episode_slices_take_the_effort_level_and_the_envs_string_stamps():
    traj = types.SimpleNamespace(
        reasoning_effort="high", info={EPISODE_SLICES_KEY: {"language": "cpp", "n": 3, "e": ""}}
    )
    assert rollout_metrics.RolloutMetricsMixin._episode_slices(traj) == {"effort": "high", "language": "cpp"}
    assert (
        rollout_metrics.RolloutMetricsMixin._episode_slices(types.SimpleNamespace(reasoning_effort=None, info={}))
        == {}
    )
    assert rollout_metrics.RolloutMetricsMixin._episode_slices(None) == {}
    listed = types.SimpleNamespace(reasoning_effort="low", info={EPISODE_SLICES_KEY: ["cpp"]})
    assert rollout_metrics.RolloutMetricsMixin._episode_slices(listed) == {"effort": "low"}, (
        "a non-mapping stamp is ignored"
    )
    hijack = types.SimpleNamespace(reasoning_effort="low", info={EPISODE_SLICES_KEY: {"effort": "high"}})
    assert rollout_metrics.RolloutMetricsMixin._episode_slices(hijack) == {"effort": "low"}, "effort is the trainer's"


def test_metrics_are_sliced_per_value_of_every_slice():
    host = _host()
    results = [
        _result("high", {"language": "cpp"}, solve=1.0, test_calls=2.0, reward=1.2),
        _result("high", {"language": "cpp"}, solve=0.0, test_calls=4.0, reward=0.2),
        _result("low", {"language": "python"}, solve=1.0, test_calls=1.0, reward=1.0),
        _result("low", None, solve=0.0, test_calls=3.0, reward=0.0),
    ]
    host._log_rollout_metrics(results, "train", [[10]] * len(results))
    m = host._metrics["train"]
    assert m["language/cpp/count"] == [2.0] and m["language/python/count"] == [1.0]
    assert m["language/cpp/solve_rate"] == [pytest.approx(0.5)]
    assert m["language/cpp/reward"] == [pytest.approx(0.7)]
    assert m["language/cpp/test_calls"] == [pytest.approx(3.0)]
    assert m["language/python/test_calls"] == [1.0]
    assert m["effort/high/count"] == [2.0] and m["effort/low/count"] == [2.0]
    assert m["effort/low/solve_rate"] == [pytest.approx(0.5)]
    assert "language/None/count" not in m, "an episode without the stamp joins no language slice"
    assert m["outcome/solve_rate"] == [pytest.approx(0.5)], "the unsliced means stay"


def test_slices_and_the_p90_pool_every_ranks_episodes(monkeypatch):
    """A slice's count and means and the token p90 are the world's: cpp reward (1.2 + 0.2 + 0.4) / 3, where a mean of
    the two ranks' means reads 0.55."""
    peer = [
        _result("high", {"language": "cpp"}, 1.0, 1.0, reward=0.4, tokens=500),
        _result("low", {}, 0.0, 1.0, tokens=50),
    ]
    payloads = []

    def keep_the_peers_states(records, layout, group, key_mismatch_error):
        payloads.append(rank_states(records))
        return payloads[-1], layout

    monkeypatch.setattr(world_metrics, "_world_states", keep_the_peers_states)
    _host()._log_rollout_metrics(peer, "train", [[10]] * len(peer))
    monkeypatch.setattr(
        world_metrics,
        "_world_states",
        lambda records, layout, group, key_mismatch_error: (
            pooled_states([rank_states(records), *payloads], key_mismatch_error),
            layout,
        ),
    )
    host = _host()
    own = [
        _result("high", {"language": "cpp"}, 1.0, 1.0, reward=1.2, tokens=100),
        _result("high", {"language": "cpp"}, 0.0, 1.0, reward=0.2, tokens=300),
    ]
    host._log_rollout_metrics(own, "train", [[10]] * len(own))
    m = host._metrics["train"]
    assert m["language/cpp/count"] == [3.0] and m["language/cpp/reward"] == [pytest.approx(0.6)]
    assert m["episode/generation_tokens_p90"] == [500.0] and m["episode/generation_tokens"] == [950 / 4]
    assert host.cumulative_rollout_metrics()["async/total_generation_tokens"] == 950.0


def test_the_composition_residue_is_the_mean_shortfall_either_way():
    """Components 0.5 under a reward of 1.0 and 1.5 over it: each misses by 0.5, which signed would cancel."""
    host = _host()
    results = [_result("high", None, 1.0, 1.0), _result("high", None, 1.0, 1.0)]
    for result, component in zip(results, (0.5, 1.5), strict=True):
        result.metrics[f"{REWARD_COMPONENT_PREFIX}correctness"] = component
    host._log_rollout_metrics(results, "train", [[10]] * len(results))
    assert host._metrics["train"]["reward/composition_residue"] == [pytest.approx(0.5)]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
