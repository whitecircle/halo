"""Reasoning-effort resolution (incl. random) and the per-level reasoning terms: the base facility, the eval's stable
per-task draw, and the CodeContestsEnvironment reading each level's budget and price off its profile."""

import json
import os
import subprocess
import sys

import pytest

from halo.environments.environment_base import EnvironmentBase
from halo.environments.episode_scoring import EpisodeGrade
from halo.environments.reasoning_effort import resolve_reasoning_effort, stable_reasoning_effort
from halo.environments.rollout_config import VALID_REASONING_EFFORTS
from halo.environments.tasks.coding.code_contests import CodeContestsEnvironment
from tests.common.repo import REPO_ROOT

_DRAWN_PROBLEMS = [f"problem {i}" for i in range(24)]
# Prints the stable draw of each of ``_DRAWN_PROBLEMS`` as JSON, for a fresh interpreter under a chosen hash seed.
_STABLE_DRAW_PROBE = (
    "import json; from halo.environments.reasoning_effort import stable_reasoning_effort; "
    f"print(json.dumps([stable_reasoning_effort(p) for p in {_DRAWN_PROBLEMS!r}]))"
)


class _MinimalEnv(EnvironmentBase):
    """Concrete EnvironmentBase for exercising the base reasoning-effort plumbing."""

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


def test_base_reasoning_terms_are_unset_by_default():
    # No profile → no level is budgeted or priced, and an episode with no level reads the same.
    env = _MinimalEnv(reasoning_effort="high")
    for level in (*VALID_REASONING_EFFORTS, None):
        assert env.reasoning_terms_for_effort(level) == (None, 0.0), level


def test_codeforces_reads_each_levels_reasoning_terms_off_its_profiles():
    profiles = {
        "low": {"reasoning_budget": 40000, "reasoning_price": 0.008},
        "high": {"reasoning_budget": 60000, "reasoning_price": 0.004},
    }
    env = CodeContestsEnvironment(language="python", reasoning_effort="random", reasoning_effort_profiles=profiles)
    assert env.reasoning_effort == "random"
    assert env.reasoning_terms_for_effort("low") == (40000, 0.008)
    assert env.reasoning_terms_for_effort("high") == (60000, 0.004)
    assert env.reasoning_terms_for_effort("medium") == (None, 0.0), "a level the profiles leave out is unbudgeted"
    assert env.reasoning_terms_for_effort("nonexistent") == (None, 0.0)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
