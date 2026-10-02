"""CPU tests for reasoning-effort resolution (incl. "random") and the per-level CoT budget.

Covers the general facility on BaseEnvironment (validation, the resolve_reasoning_effort helper,
the thinking_budget_for_effort hook), the eval's stable per-task draw, which no process's hash seed
moves, and the CodeContestsEnvironment override that binds each level to a token budget.

    python tests/cpu/environments/test_reasoning_effort.py
"""

import json
import os
import subprocess
import sys

import pytest

from src.environments.base import (
    VALID_REASONING_EFFORTS,
    BaseEnvironment,
    EpisodeGrade,
    resolve_reasoning_effort,
    stable_reasoning_effort,
)
from src.environments.envs.tasks.coding.code_contests import CodeContestsEnvironment
from tests.common.utils import REPO_ROOT

_DRAWN_PROBLEMS = [f"problem {i}" for i in range(24)]
# Prints the stable draw of each of ``_DRAWN_PROBLEMS`` as JSON, for a fresh interpreter under a chosen hash seed.
_STABLE_DRAW_PROBE = (
    "import json; from src.environments.base import stable_reasoning_effort; "
    f"print(json.dumps([stable_reasoning_effort(p) for p in {_DRAWN_PROBLEMS!r}]))"
)


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


def _stable_draw_under_hash_seed(seed: str) -> list[str]:
    env = {**os.environ, "PYTHONHASHSEED": seed, "PYTHONPATH": str(REPO_ROOT)}
    result = subprocess.run(
        [sys.executable, "-c", _STABLE_DRAW_PROBE], capture_output=True, text=True, cwd=REPO_ROOT, env=env, timeout=600
    )
    assert result.returncode == 0, result.stderr[-2000:]
    return json.loads(result.stdout.strip().splitlines()[-1])


def test_the_stable_draw_is_the_same_in_every_process():
    """An eval scores each problem at one level from checkpoint to checkpoint, and each eval is a new
    process: a draw keyed on Python's per-process salted ``hash`` would move problems between levels."""
    first, second = _stable_draw_under_hash_seed("1"), _stable_draw_under_hash_seed("2")
    assert len(set(first)) > 1, "every problem draws one level, so agreeing on it proves nothing"
    assert first == second == [stable_reasoning_effort(problem) for problem in _DRAWN_PROBLEMS]


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
    raise SystemExit(pytest.main([__file__, "-v"]))
