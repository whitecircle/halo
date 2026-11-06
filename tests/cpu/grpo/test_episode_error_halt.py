#!/usr/bin/env python
"""An errored or timed-out episode trains as a masked row, so a run whose slow episodes all time out keeps training on
the fast ones. Rounds whose world-wide error rate stays over the limit halt the run instead."""

import logging
from collections import defaultdict

import pytest

from halo.configs.grpo_environmental import AsyncRolloutConfig
from halo.environments.episode import RolloutResult
from halo.trainers.grpo.environmental.rollout_metrics import RolloutMetricsMixin
from tests.common.env_grpo_batch import episode
from tests.common.world_metrics import attach_world_metrics


class _Host(RolloutMetricsMixin):
    def __init__(self, **config):
        self.async_config = AsyncRolloutConfig(**config)
        self._metrics = {"train": defaultdict(list), "eval": defaultdict(list)}
        attach_world_metrics(self)


def _round(host, mode: str, failed: int, total: int = 4) -> None:
    results = [episode(1.0, valid=False) for _ in range(failed)] + [episode(1.0) for _ in range(total - failed)]
    for result in results:
        result.trajectory.truncated = False
    host._log_rollout_metrics(results, mode, [[1]] * total)


def test_rounds_over_the_limit_in_a_row_raise_at_the_round_limit(caplog):
    host = _Host(episode_error_rate_limit=0.5, episode_error_round_limit=3)
    with caplog.at_level(logging.WARNING):
        _round(host, "train", 3)
        _round(host, "train", 4)
    assert len([r for r in caplog.records if "errored or timed out" in r.getMessage()]) == 1, "warned once, first"
    with pytest.raises(RuntimeError, match=r"75%|100%") as err:
        _round(host, "train", 3)
    assert "request_timeout" in str(err.value) and "boom" in str(err.value)


def test_a_healthy_round_resets_the_count():
    host = _Host(episode_error_rate_limit=0.5, episode_error_round_limit=2)
    _round(host, "train", 3)
    _round(host, "train", 1)
    _round(host, "train", 3)
    assert host._episode_error_rounds == 1


def test_a_rate_at_the_limit_is_not_over_it():
    host = _Host(episode_error_rate_limit=0.5, episode_error_round_limit=1)
    _round(host, "train", 2)
    assert host._episode_error_rounds == 0


def test_an_eval_round_neither_counts_nor_halts():
    host = _Host(episode_error_rate_limit=0.5, episode_error_round_limit=1)
    _round(host, "eval", 4)
    assert host._episode_error_rounds == 0


def test_a_null_limit_trains_through_anything():
    host = _Host(episode_error_rate_limit=None, episode_error_round_limit=1)
    for _ in range(3):
        _round(host, "train", 4)
    assert host._episode_error_rounds == 0


@pytest.mark.parametrize("field", ["episode_error_rate_limit"])
@pytest.mark.parametrize("value", [1.0, -0.1])
def test_a_limit_outside_the_unit_interval_is_refused(field, value):
    with pytest.raises(ValueError, match=field):
        AsyncRolloutConfig(**{field: value})


def test_a_non_positive_round_limit_is_refused():
    with pytest.raises(ValueError, match="episode_error_round_limit"):
        AsyncRolloutConfig(episode_error_round_limit=0)


def test_the_defaults_leave_a_round_with_a_minority_of_failures_alone():
    host = _Host()
    for _ in range(10):
        _round(host, "train", 2)
    assert host._episode_error_rounds == 0
    assert RolloutResult  # the episodes above are real results


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
