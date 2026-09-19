#!/usr/bin/env python
"""Parse-time range checks on ``AsyncTrainingConfig``.

* ``skip_update_masked_frac`` is a fraction in (0, 1]: 0 trips the breaker on every step, above 1 never.
  Checked with the other knobs at parse time, not at trainer construction after the servers are up.
* ``max_train_row_tokens`` must exceed ``rollout_max_tokens``: a training row is prompt + completion
  and the completion alone may run to the per-turn budget, so a cap at or below it leaves out every
  turn that used its budget — a length bias against long turns, not the memory bound the knob is.

    python tests/cpu/config/test_env_trainer_config_ranges.py
"""

import math

import pytest

from src.configs.async_training_config import AsyncTrainingConfig


@pytest.mark.parametrize("bad", [0.0, -0.1, 1.5, math.nan])
def test_skip_update_masked_frac_outside_the_unit_interval_is_refused_at_parse_time(bad):
    with pytest.raises(ValueError, match="skip_update_masked_frac must be in \\(0, 1\\]"):
        AsyncTrainingConfig(skip_update_masked_frac=bad)


@pytest.mark.parametrize("ok", [0.3, 1.0, None])
def test_skip_update_masked_frac_in_range_passes(ok):
    assert AsyncTrainingConfig(skip_update_masked_frac=ok).skip_update_masked_frac == ok


@pytest.mark.parametrize("cap", [1, 999, 1000])
def test_a_row_cap_at_or_below_the_per_turn_budget_is_refused(cap):
    with pytest.raises(ValueError, match="must be above rollout_max_tokens"):
        AsyncTrainingConfig(rollout_max_tokens=1000, max_train_row_tokens=cap)


def test_a_row_cap_above_the_per_turn_budget_passes():
    assert AsyncTrainingConfig(rollout_max_tokens=1000, max_train_row_tokens=1001).max_train_row_tokens == 1001
    assert AsyncTrainingConfig(rollout_max_tokens=1000).max_train_row_tokens is None


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
