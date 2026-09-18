"""CPU tests for reasoning-effort resolution (incl. "random") and the per-level CoT budget.

Covers the general facility on BaseEnvironment (validation, the resolve_reasoning_effort helper,
the thinking_budget_for_effort hook) and the CodeContestsEnvironment override that binds each level
to a token budget.

    python tests/cpu/environments/test_reasoning_effort.py
"""

import sys

import pytest

from src.environments.base import VALID_REASONING_EFFORTS, BaseEnvironment, EpisodeGrade, resolve_reasoning_effort
from src.environments.envs.tasks.coding.code_contests import CodeContestsEnvironment


class _MinimalEnv(BaseEnvironment):
    """Concrete BaseEnvironment for exercising the base reasoning-effort plumbing."""

    def _reset_single(self, prompt, context):  # pragma: no cover
        raise NotImplementedError

    def _step_single(self, episode_id, action, context):  # pragma: no cover
        raise NotImplementedError

    def _grade_episode(self, trajectory, context=None):  # pragma: no cover
        return EpisodeGrade(0.0)


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


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
