#!/usr/bin/env python
"""Grading runs a submission's tests through one sandbox session, so a compiled program is built
once, and a compile failure grades the whole pool once.

Run: python tests/cpu/environments/test_grading_session.py  (or pytest)
"""

import errno
import logging

import pytest

from src.environments.envs.tasks.coding.grading import GradingSpec, grade_solution, run_solution_against_tests
from src.environments.sandbox.base import SandboxExecutor, SandboxResult, SandboxSession
from src.environments.sandbox.local import LocalSubprocessSandbox

_TESTS = [{"input": f"{i}\n", "output": "ok\n"} for i in range(3)]
_OK = SandboxResult(stdout="ok\n", returncode=0)


class _Session(SandboxSession):
    def __init__(self, owner, result):
        self.owner, self.result, self.runs, self.closed, self.resets = owner, result, [], False, 0

    def run(self, code, *, stdin="", timeout=15.0, language="python", files=None):
        self.runs.append((language, stdin))
        return self.result

    def reset_to_staged(self):
        self.resets += 1

    def write_file(self, path, content):  # pragma: no cover
        raise NotImplementedError

    def read_file(self, path):  # pragma: no cover
        raise NotImplementedError

    def list_files(self):  # pragma: no cover
        raise NotImplementedError

    def close(self):
        self.closed = True


class _SessionSandbox(SandboxExecutor):
    """Counts the sessions it opens and the one-shot runs it is asked for."""

    def __init__(self, result=_OK, sessions=True):
        self.result, self.sessions, self.opened, self.one_shot = result, sessions, [], 0

    def open_session(self):
        if not self.sessions:
            raise NotImplementedError
        session = _Session(self, self.result)
        self.opened.append(session)
        return session

    def run(self, code, *, stdin="", timeout=15.0, language="python", files=None):
        self.one_shot += 1
        return self.result


class _DeniedSession(_Session):
    """Fails at ``stage`` with a host-side exception whose text quotes a name the program chose."""

    def __init__(self, stage):
        super().__init__(None, _OK)
        self.stage = stage

    def _deny(self, stage):
        if stage == self.stage:
            raise PermissionError(errno.EACCES, "Permission denied", "/tmp/work/HIDDEN-4217")

    def run(self, code, **kwargs):
        self._deny("run")
        return super().run(code, **kwargs)

    def reset_to_staged(self):
        self._deny("reset")


class _DeniedSandbox(_SessionSandbox):
    def __init__(self, stage):
        super().__init__()
        self.stage = stage

    def open_session(self):
        return _DeniedSession(self.stage)


def test_a_grade_runs_every_test_through_one_session_and_closes_it():
    sandbox = _SessionSandbox()
    grade = run_solution_against_tests("code", _TESTS, sandbox=sandbox, language="cpp")
    assert grade.passed == 3 and sandbox.one_shot == 0
    assert len(sandbox.opened) == 1
    (session,) = sandbox.opened
    assert session.runs == [("cpp", "0\n"), ("cpp", "1\n"), ("cpp", "2\n")]
    assert session.resets == 3, "the session is reset to its staged state after every test"
    assert session.closed


@pytest.mark.parametrize("stage", ["run", "reset"])
def test_a_host_fault_shows_its_class_alone_under_outcome(stage, caplog):
    grade = run_solution_against_tests("code", _TESTS[:1], sandbox=_DeniedSandbox(stage))
    assert grade.infra_errors == 1
    assert grade.details.splitlines()[1:] == ["Test 1: ERROR -- grading infrastructure failure"], grade.details
    assert "HIDDEN-4217" in caplog.text, "the log keeps what the verdict leaves out"
    full = run_solution_against_tests("code", _TESTS[:1], sandbox=_DeniedSandbox(stage), verdict_detail="full")
    assert "HIDDEN-4217" in full.details


def test_garbage_output_is_a_wrong_answer_not_an_infra_error():
    program = "import sys\nsys.stdout.buffer.write(b'\\x80\\x80\\n')"
    grade = run_solution_against_tests(program, [{"input": "", "output": "ok\n"}], sandbox=LocalSubprocessSandbox())
    assert (grade.passed, grade.infra_errors, grade.ran_ok) == (0, 0, 1), grade.details
    assert "FAIL" in grade.details and "ERROR" not in grade.details


def test_tests_of_one_grade_cannot_see_each_others_files():
    """A program that leaves a file behind in one test must not find it in the next: the session reuses
    the build, never the run's leftovers, so a grade matches a per-test fresh directory."""
    program = "import os\nprint('seen' if os.path.exists('marker') else 'fresh')\nopen('marker', 'w').write('x')\n"
    tests = [{"input": "", "output": "fresh\n"}] * 3
    grade = run_solution_against_tests(program, tests, sandbox=LocalSubprocessSandbox(), language="python")
    assert grade.passed == 3, grade.details


def test_an_executor_without_sessions_grades_one_shot_without_a_warning(caplog):
    sandbox = _SessionSandbox(sessions=False)
    with caplog.at_level(logging.WARNING, logger="src.environments.envs.tasks.coding.grading"):
        grade = run_solution_against_tests("code", _TESTS, sandbox=sandbox)
    assert grade.passed == 3 and sandbox.one_shot == 3
    assert caplog.records == []


def test_a_compile_failure_grades_the_whole_pool_once():
    failure = SandboxResult(compile_failed=True, returncode=1, stderr="main.cpp:1:11: error: expected ';'")
    sandbox = _SessionSandbox(result=failure)
    grade = run_solution_against_tests("code", _TESTS, sandbox=sandbox, language="cpp", verdict_detail="full")
    assert (grade.passed, grade.total, grade.ran_ok, grade.graded, grade.infra_errors) == (0, 3, 0, 3, 0)
    assert not grade.budget_hit
    assert "COMPILATION ERROR (every test fails)" in grade.details
    assert "expected ';'" in grade.details
    assert "RUNTIME ERROR" not in grade.details
    assert len(sandbox.opened[0].runs) == 1, "no further test runs a source that never built"


def test_grade_solution_passes_the_call_language_and_falls_back_to_the_contract():
    sandbox = _SessionSandbox()
    spec = GradingSpec(sandbox=sandbox, language="python")
    grade_solution("code", _TESTS[:1], spec)
    grade_solution("code", _TESTS[:1], spec, language="cpp")
    assert [session.runs[0][0] for session in sandbox.opened] == ["python", "cpp"]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
