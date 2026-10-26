#!/usr/bin/env python
"""CPU tests: the env-GRPO ``async/*`` cumulative counters are JOB totals, not one rank's shard.

``RolloutManager`` is built on every rank and each one collects its own DP shard, so counters kept
manager-side under-report a 512-GPU job by ~512x while ``async/total_rollouts`` reads as
authoritative. The totals are therefore accumulated from the world sums :meth:`_log_rollout_metrics`
already folds for its means — no second collective.

Run::

    python tests/cpu/grpo/test_env_rollout_totals.py
"""

from __future__ import annotations

import types
from collections import defaultdict

import pytest

import halo.trainers.grpo.environmental.rollout_metrics as rm
from halo.trainers import world_metrics


class _MetricsHost(rm.RolloutMetricsMixin):
    """Minimal stand-in exposing exactly what ``_log_rollout_metrics`` reads."""

    def __init__(self):
        self._metrics = {"train": defaultdict(list), "eval": defaultdict(list)}
        self.async_config = types.SimpleNamespace(truncation_alarm_rate=None)


def _episode(latency: float, tokens: int, requests_expired_in_sync: int = 0):
    return types.SimpleNamespace(
        latency=latency,
        generation_tokens=tokens,
        requests_expired_in_sync=requests_expired_in_sync,
        episode_length=1,
        success=True,
        trajectory=None,
        error=None,
        total_reward=1.0,
        metrics={},
    )


def _fake_dp_world(monkeypatch, world: int):
    """Make the metrics flush gather ``world`` ranks holding this rank's identical shard."""
    monkeypatch.setattr(world_metrics, "_gathered", lambda payload, group: [payload] * world)


def _log_round(host, results: list, mode: str = "train") -> None:
    """``_log_rollout_metrics`` over episodes that carry no reasoning."""
    host._log_rollout_metrics(results, mode, [[] for _ in results])


def test_cumulative_totals_count_the_whole_world_not_this_rank(monkeypatch) -> None:
    """A rank-local count would report 2 rollouts for a 4-rank step that ran 8."""
    _fake_dp_world(monkeypatch, world=4)
    host = _MetricsHost()

    _log_round(host, [_episode(1.0, 100), _episode(3.0, 300)])

    metrics = host.cumulative_rollout_metrics()
    assert metrics["async/total_rollouts"] == 8.0
    assert metrics["async/total_generation_tokens"] == 1600.0
    assert metrics["async/cumulative_mean_rollout_latency"] == pytest.approx(2.0)


def test_totals_accumulate_across_steps_and_modes(monkeypatch) -> None:
    """Cumulative since train start: a second round adds to the first rather than replacing it, and
    the latency mean is over every episode, not over the per-step means."""
    _fake_dp_world(monkeypatch, world=2)
    host = _MetricsHost()

    _log_round(host, [_episode(1.0, 10)])
    _log_round(host, [_episode(5.0, 30)], "eval")

    metrics = host.cumulative_rollout_metrics()
    assert metrics["async/total_rollouts"] == 4.0
    assert metrics["async/total_generation_tokens"] == 80.0
    assert metrics["async/cumulative_mean_rollout_latency"] == pytest.approx(3.0)


def test_totals_are_per_trainer_not_shared_by_the_class(monkeypatch) -> None:
    """The counters start as class attributes so no ``__init__`` is needed; ``+=`` must rebind them
    on the instance, or a second trainer in the same process would inherit the first's totals."""
    _fake_dp_world(monkeypatch, world=1)
    first, second = _MetricsHost(), _MetricsHost()

    _log_round(first, [_episode(1.0, 10)])

    assert first.cumulative_rollout_metrics()["async/total_rollouts"] == 1.0
    assert second.cumulative_rollout_metrics()["async/total_rollouts"] == 0.0
    assert rm.RolloutMetricsMixin._total_rollouts == 0


def test_no_rollouts_yet_reports_zero_without_dividing_by_zero() -> None:
    assert _MetricsHost().cumulative_rollout_metrics() == {
        "async/total_rollouts": 0.0,
        "async/cumulative_mean_rollout_latency": 0.0,
        "async/total_generation_tokens": 0.0,
    }


def test_requests_expired_in_sync_is_the_worlds_count_not_a_mean(monkeypatch) -> None:
    """Each expiry is a turn the engine generated twice, so the round logs how many there were across
    the world: two ranks of (1 + 0 + 2) is 6, where a per-episode mean would read 1."""
    _fake_dp_world(monkeypatch, world=2)
    host = _MetricsHost()

    _log_round(host, [_episode(1.0, 10, 1), _episode(1.0, 10, 0), _episode(1.0, 10, 2)])

    assert host._metrics["train"]["async/requests_expired_in_sync"] == [6.0]


def test_an_environment_metric_named_like_a_total_stays_the_environments_mean(monkeypatch) -> None:
    """The totals fold under the trainer's own ``async/total_*`` names: an environment reporting a bare
    ``generation_tokens`` or ``rollouts`` logs its own mean and leaves the totals alone."""
    _fake_dp_world(monkeypatch, world=2)
    host = _MetricsHost()
    episode = _episode(1.0, 10)
    episode.metrics = {"generation_tokens": 7.0, "rollouts": 3.0}

    _log_round(host, [episode])

    logged = host._metrics["train"]
    assert logged["generation_tokens"] == [7.0] and logged["rollouts"] == [3.0]
    assert not any(key.startswith("async/total_") for key in logged), sorted(logged)
    totals = host.cumulative_rollout_metrics()
    assert totals["async/total_rollouts"] == 2.0 and totals["async/total_generation_tokens"] == 20.0


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
