#!/usr/bin/env python
"""CPU tests: a SWE grader that itself crashes must drop its episode from the GRPO group baseline.

``SweEnvironment._compute_reward`` scores a raising ``test_function`` as a failure. That verdict is
about the grader, not the completion, so the episode also has to carry ``EPISODE_INVALID_KEY`` — the
flag the trainer reads to exclude a row from the group mean every sibling's advantage is measured
against. A real pass/fail verdict must NOT carry it.

    python tests/cpu/environments/test_swe_grader_crash_invalid.py
"""

import sys

import pytest

from src.environments.base import Trajectory
from src.environments.envs.tasks.coding.swe import SweEnvironment
from src.environments.sandbox.base import SandboxExecutor, SandboxResult


class _StubSandbox(SandboxExecutor):
    """Never used: these episodes answer in plain text, so no tool ever opens a session."""

    def open_session(self):  # pragma: no cover
        raise NotImplementedError

    def run(self, code, *, stdin="", timeout=15.0, language="python", files=None):  # pragma: no cover
        return SandboxResult(stdout="", returncode=0)


def _finished_episode(test_function) -> Trajectory:
    """Drive one episode to a plain-text answer, so ``_compute_reward`` runs the grader."""
    env = SweEnvironment(sandbox=_StubSandbox(), test_function=test_function)
    ids, _ = env.reset(["fix the bug"], [{}])
    step = env.step(ids, ["all done"], [{}])[0]
    assert step.done and step.trajectory.info["completed"], "episode must reach the graded terminal step"
    return step.trajectory


def test_crashing_test_function_marks_the_episode_invalid():
    def boom(trajectory):
        raise RuntimeError("grading container died")

    traj = _finished_episode(boom)
    assert traj.episode_invalid is True, "a crashed grader carries no signal — it must leave the group baseline"
    assert traj.total_reward == pytest.approx(0.0), "and it still scores the failure reward"


def test_real_verdicts_stay_in_the_baseline():
    # The counterpart that keeps the flag honest: an episode the grader actually judged is signal,
    # whichever way it went, and must stay in the baseline.
    passed = _finished_episode(lambda trajectory: True)
    failed = _finished_episode(lambda trajectory: False)
    assert passed.episode_invalid is False
    assert failed.episode_invalid is False
    assert passed.total_reward == pytest.approx(1.0)
    assert failed.total_reward == pytest.approx(0.0)


def test_ungraded_episode_stays_in_the_baseline():
    # No grader at all is a configuration, not a fault: those episodes are graded by completion.
    traj = _finished_episode(None)
    assert traj.episode_invalid is False


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
