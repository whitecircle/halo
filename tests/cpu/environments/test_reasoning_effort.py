"""CPU tests for reasoning-effort resolution (incl. "random") and the per-level CoT budget.

Covers the general facility on BaseEnvironment (validation, the resolve_reasoning_effort helper,
the thinking_budget_for_effort hook) and the CodeContestsEnvironment override that binds each level
to a token budget.

    python tests/cpu/environments/test_reasoning_effort.py
"""

import sys

import pytest

from src.environments.base import VALID_REASONING_EFFORTS, BaseEnvironment, resolve_reasoning_effort
from src.environments.envs.tasks.coding.code_contests import CodeContestsEnvironment
from src.environments.episode import reasoning_calibration_penalty


class _MinimalEnv(BaseEnvironment):
    """Concrete BaseEnvironment for exercising the base reasoning-effort plumbing."""

    def _reset_single(self, prompt, context):  # pragma: no cover
        raise NotImplementedError

    def _step_single(self, episode_id, action, context):  # pragma: no cover
        raise NotImplementedError

    def _compute_reward(self, trajectory):  # pragma: no cover
        return 0.0


def test_resolve_passthrough_and_none():
    assert resolve_reasoning_effort(None) is None
    for level in VALID_REASONING_EFFORTS:
        assert resolve_reasoning_effort(level) == level


def test_resolve_random_returns_valid_level_and_covers_all():
    seen = {resolve_reasoning_effort("random") for _ in range(300)}
    assert seen <= set(VALID_REASONING_EFFORTS)
    assert seen == set(VALID_REASONING_EFFORTS), f"random did not cover all levels: {seen}"


def test_base_validates_effort():
    assert _MinimalEnv(reasoning_effort=None).reasoning_effort is None
    for level in (*VALID_REASONING_EFFORTS, "random"):
        assert _MinimalEnv(reasoning_effort=level).reasoning_effort == level
    with pytest.raises(ValueError, match="reasoning_effort"):
        _MinimalEnv(reasoning_effort="extreme")


def test_base_budget_hook_is_none_by_default():
    # No per-level budget → the rollout falls back to the global one.
    env = _MinimalEnv(reasoning_effort="high")
    assert env.thinking_budget_for_effort("high") is None


def test_codeforces_binds_level_to_budget():
    env = CodeContestsEnvironment(language="python", reasoning_effort="random")
    assert env.reasoning_effort == "random"
    assert env.thinking_budget_for_effort("low") == 4096
    assert env.thinking_budget_for_effort("medium") == 8192
    assert env.thinking_budget_for_effort("high") == 16384
    assert env.thinking_budget_for_effort("nonexistent") is None


def test_calibration_in_band_is_zero():
    B = 8000
    # Anywhere in [0.3B, 0.9B] is compliant → no penalty.
    for r in (int(0.3 * B), int(0.6 * B), int(0.9 * B)):
        assert reasoning_calibration_penalty([r], B) == 0.0


def test_calibration_asymmetric_overuse_punished_more():
    B = 8000
    under = reasoning_calibration_penalty([int(0.1 * B)], B)  # below 0.3B
    over = reasoning_calibration_penalty([B], B)  # at the cap (truncated)
    assert under < 0 and over < 0
    assert over < under, f"over-use {over} should be a stronger (more negative) penalty than under-use {under}"
    # Extremes hit the max weights: over_use 1.0 at the cap, the milder under_use 0.3 at r == 0.
    assert over == pytest.approx(-1.0, abs=1e-6)
    assert reasoning_calibration_penalty([0], B) == pytest.approx(-0.3, abs=1e-6)


def test_calibration_averages_over_turns_and_guards():
    B = 8000
    # One in-band turn + one truncated turn → average of 0 and -1.
    assert reasoning_calibration_penalty([int(0.5 * B), B], B) == pytest.approx(-0.5, abs=1e-6)
    # No budget / no turns → no penalty (guarded).
    assert reasoning_calibration_penalty([], B) == 0.0
    assert reasoning_calibration_penalty([5000], 0) == 0.0


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
