#!/usr/bin/env python
"""A SWE grader that crashes drops its episode from the GRPO group baseline: the verdict is about the grader, so the
episode carries ``EPISODE_INVALID_KEY``. A real pass or fail does not."""

import pytest

from halo.environments.episodes.episode_keys import EPISODE_INVALID_REASON_KEY
from halo.environments.episodes.messages import Trajectory
from halo.environments.sandbox.base import SandboxExecutor
from halo.environments.sandbox.results import SandboxResult
from halo.environments.tasks.coding.swe import SweEnvironment


class _StubSandbox(SandboxExecutor):
    """Never used: these episodes answer in plain text, so no tool ever opens a session."""

    def open_session(self):  # pragma: no cover
        raise NotImplementedError

    def run(self, code, *, stdin="", timeout=15.0, language="python", files=None):  # pragma: no cover
        return SandboxResult(stdout="", returncode=0)


def _finished_episode(test_function) -> Trajectory:
    """Drive one episode to a plain-text answer, so ``_grade_episode`` runs the grader."""
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
    assert traj.total_reward == pytest.approx(0.0), "and it still grades 0"
    # Named where the all-invalid step halt and the eval runner read the cause.
    assert traj.info[EPISODE_INVALID_REASON_KEY] == "test_function raised RuntimeError: grading container died"


def test_real_verdicts_stay_in_the_baseline():
    # An episode the grader judged is signal, whichever way it went, and stays in the baseline.
    passed = _finished_episode(lambda trajectory: True)
    failed = _finished_episode(lambda trajectory: False)
    assert passed.episode_invalid is False
    assert failed.episode_invalid is False
    assert passed.total_reward == pytest.approx(1.0)
    assert failed.total_reward == pytest.approx(0.0)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
