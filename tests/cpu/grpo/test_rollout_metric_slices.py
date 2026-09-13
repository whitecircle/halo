#!/usr/bin/env python
"""Rollout metrics are sliced by every categorical fact an episode carries.

The effort level is one slice; an env stamps others under ``info["slices"]`` (the language a
code-contests episode settled on). Each value gets the same breakdown — count, reward, tokens,
turns, truncation, solve rate and every ``episode/*`` metric — so a run's balance across the values
and each value's outcome read off ``<slice>/<value>/*``.

Run: python tests/cpu/grpo/test_rollout_metric_slices.py  (or pytest)
"""

import types
from collections import defaultdict

import pytest

from src.environments.base import EPISODE_SLICES_KEY, Trajectory
from src.environments.episode import RolloutResult
from src.trainers.grpo.environmental import DistributedAsyncEnvironmentalGRPOTrainer
from src.trainers.grpo.rollout import rollout_metrics


def _host():
    host = object.__new__(DistributedAsyncEnvironmentalGRPOTrainer)
    host._metrics = {"train": defaultdict(list)}
    host._total_rollouts = 0
    host._total_rollout_latency = 0.0
    host._total_generation_tokens = 0
    host._assistant_turn_reasoning_tokens = lambda traj: [10]
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
        success=True,
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


def test_metrics_are_sliced_per_value_of_every_slice(monkeypatch):
    monkeypatch.setattr(rollout_metrics, "gather_object", lambda values: values)
    host = _host()
    results = [
        _result("high", {"language": "cpp"}, solve=1.0, test_calls=2.0, reward=1.2),
        _result("high", {"language": "cpp"}, solve=0.0, test_calls=4.0, reward=0.2),
        _result("low", {"language": "python"}, solve=1.0, test_calls=1.0, reward=1.0),
        _result("low", None, solve=0.0, test_calls=3.0, reward=0.0),
    ]
    host._log_rollout_metrics(results, "train")
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


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
