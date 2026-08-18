#!/usr/bin/env python
"""CPU tests: the env-GRPO ``async/*`` cumulative counters are JOB totals, not one rank's shard.

``RolloutManager`` is built on every rank and each one collects its own DP shard, so counters kept
manager-side under-report a 512-GPU job by ~512x while ``async/total_rollouts`` reads as
authoritative. The totals are therefore accumulated from the gathered-global per-episode population
:meth:`_log_rollout_metrics` already builds for its means — no second collective.

Run::

    python tests/cpu/grpo/test_env_rollout_totals.py
"""

from __future__ import annotations

import types
from collections import defaultdict

import pytest

import src.trainers.grpo.rollout.rollout_metrics as rm


class _MetricsHost(rm.RolloutMetricsMixin):
    """Minimal stand-in exposing exactly what ``_log_rollout_metrics`` reads."""

    def __init__(self):
        self._metrics = {"train": defaultdict(list), "eval": defaultdict(list)}


def _episode(latency: float, tokens: int):
    return types.SimpleNamespace(
        latency=latency,
        generation_tokens=tokens,
        episode_length=1,
        success=True,
        trajectory=None,
        error=None,
        total_reward=1.0,
        metrics={},
    )


def _fake_dp_world(monkeypatch, world: int):
    """Make ``gather_object`` behave like a ``world``-rank all-gather of identical shards."""
    monkeypatch.setattr(rm, "gather_object", lambda values: [item for _ in range(world) for item in values])
    monkeypatch.setattr(rm, "is_multi_rank_run", lambda: world > 1)


def test_cumulative_totals_count_the_whole_world_not_this_rank(monkeypatch) -> None:
    """A rank-local count would report 2 rollouts for a 4-rank step that ran 8."""
    _fake_dp_world(monkeypatch, world=4)
    host = _MetricsHost()

    host._log_rollout_metrics([_episode(1.0, 100), _episode(3.0, 300)], "train")

    metrics = host.cumulative_rollout_metrics()
    assert metrics["async/total_rollouts"] == 8.0
    assert metrics["async/total_generation_tokens"] == 1600.0
    assert metrics["async/cumulative_mean_rollout_latency"] == pytest.approx(2.0)


def test_totals_accumulate_across_steps_and_modes(monkeypatch) -> None:
    """Cumulative since train start: a second round adds to the first rather than replacing it, and
    the latency mean is over every episode, not over the per-step means."""
    _fake_dp_world(monkeypatch, world=2)
    host = _MetricsHost()

    host._log_rollout_metrics([_episode(1.0, 10)], "train")
    host._log_rollout_metrics([_episode(5.0, 30)], "eval")

    metrics = host.cumulative_rollout_metrics()
    assert metrics["async/total_rollouts"] == 4.0
    assert metrics["async/total_generation_tokens"] == 80.0
    assert metrics["async/cumulative_mean_rollout_latency"] == pytest.approx(3.0)


def test_totals_are_per_trainer_not_shared_by_the_class(monkeypatch) -> None:
    """The counters start as class attributes so no ``__init__`` is needed; ``+=`` must rebind them
    on the instance, or a second trainer in the same process would inherit the first's totals."""
    _fake_dp_world(monkeypatch, world=1)
    first, second = _MetricsHost(), _MetricsHost()

    first._log_rollout_metrics([_episode(1.0, 10)], "train")

    assert first.cumulative_rollout_metrics()["async/total_rollouts"] == 1.0
    assert second.cumulative_rollout_metrics()["async/total_rollouts"] == 0.0
    assert rm.RolloutMetricsMixin._total_rollouts == 0


def test_no_rollouts_yet_reports_zero_without_dividing_by_zero() -> None:
    assert _MetricsHost().cumulative_rollout_metrics() == {
        "async/total_rollouts": 0.0,
        "async/cumulative_mean_rollout_latency": 0.0,
        "async/total_generation_tokens": 0.0,
    }


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
