#!/usr/bin/env python
"""CPU tests for the special-judge CheckerVerdict (coding/grading.py).

The Codeforces ``generated_checker`` contract: the checker prints its verdict to
stdout and the LAST whitespace token decides accept (``"1"``) vs reject. A crashing
checker or a final ``"0"`` rejects; a sandbox *backend* failure raises
``CheckerInfraError`` so the grader buckets the test into ``infra_errors``.

These drive ``CheckerVerdict`` against a stub sandbox whose ``run`` returns a canned
:class:`SandboxResult`, so the verdict logic is tested deterministically without
spawning subprocesses (no network, no real interpreter).

Run: python tests/cpu/environments/test_grading_verdict.py  (or pytest)
"""

import json
from unittest import mock

import pytest

import src.environments.envs.tasks.coding.grading as grading_module
from src.environments.envs.tasks.coding.code_contests import CodeContestsEnvironment
from src.environments.envs.tasks.coding.grading import (
    _MAX_FAILURE_DETAILS,
    _STDERR_EXCERPT_CHARS,
    CheckerInfraError,
    CheckerVerdict,
    GradeResult,
    GradingSpec,
    grade_solution,
    run_solution_against_tests,
    select_verdict,
)
from src.environments.sandbox.base import SANDBOX_DEFAULT_TIMEOUT, SandboxExecutor, SandboxResult
from tests.common.code_contests import StubSandbox


def _verdict(result: SandboxResult) -> bool:
    checker = CheckerVerdict("# unused", StubSandbox(result))
    return checker("input", "expected", "actual")


def test_last_token_one_accepts():
    """Final whitespace token "1" accepts even with debug noise before it."""
    assert _verdict(SandboxResult(stdout="debug 0\n1", returncode=0)) is True


def test_last_token_not_one_rejects():
    """Final token is not "1" (a "1" earlier in the stream does not count)."""
    assert _verdict(SandboxResult(stdout="1\nfailed", returncode=0)) is False


def test_empty_output_rejects():
    """No tokens at all -> reject (cannot read an accept token)."""
    assert _verdict(SandboxResult(stdout="", returncode=0)) is False
    assert _verdict(SandboxResult(stdout="   \n  ", returncode=0)) is False


def test_lone_one_accepts():
    assert _verdict(SandboxResult(stdout="1\n", returncode=0)) is True


def test_lone_zero_rejects():
    assert _verdict(SandboxResult(stdout="0\n", returncode=0)) is False


def test_crashing_checker_scores_fail_not_pass():
    """A checker that printed "1" but exited non-zero (result.ok False) must REJECT.

    This is the security-relevant case: the verdict gate is ``result.ok AND last
    token == "1"``. A non-zero exit (SystemExit(1), uncaught exception, segfault)
    means the checker did not complete cleanly, so its stdout cannot be trusted —
    even though "1" is the last token.
    """
    crashed = SandboxResult(stdout="1", returncode=1)
    assert crashed.ok is False
    assert _verdict(crashed) is False


def test_timed_out_checker_rejects():
    assert _verdict(SandboxResult(stdout="1", returncode=0, timed_out=True)) is False


def test_backend_error_raises_infra_error():
    """A sandbox backend failure is NOT a wrong answer: a plain False verdict would score a grading
    outage as a zero grade for the whole group (fake signal) and hide it from the
    ``_grading_infra_outage`` guard. It must raise for the grader to bucket into ``infra_errors``."""
    with pytest.raises(CheckerInfraError, match="backend down"):
        _verdict(SandboxResult(stdout="1", returncode=None, error="backend down"))


class _RaisingSandbox(SandboxExecutor):
    """A backend that fails by RAISING, the way a local executor does.

    ``RemoteSandbox`` reports a lost run as ``SandboxResult(error=...)``, but ``LocalSubprocessSandbox``
    lets the fault out: ``Popen`` raises on a missing interpreter/``bwrap`` or an exhausted fork table,
    and the working-dir write raises on ENOSPC.
    """

    def __init__(self, exc: Exception):
        self._exc = exc
        self.calls = 0

    def open_session(self):  # pragma: no cover
        raise NotImplementedError

    def run(self, code, *, stdin="", timeout=15.0, language="python", files=None):
        self.calls += 1
        raise self._exc


def test_executor_exception_counts_as_infra_error_not_a_failed_test():
    """A raising executor must bucket as ``infra_errors``, exactly like the remote backend's error
    result. Letting the exception out leaves ``submit_solution`` an ordinary tool error: the episode
    then grades 0 with the invalid-episode guard never firing, and a host-level fault
    is averaged into the GRPO group baseline as a wrong program."""
    sandbox = _RaisingSandbox(FileNotFoundError(2, "No such file or directory", "/bin/bash"))
    grade = run_solution_against_tests(
        "print(42)",
        [{"input": "a", "output": "42"}, {"input": "b", "output": "42"}],
        sandbox=sandbox,
        verdict_detail="full",
    )
    assert sandbox.calls == 2
    assert (grade.infra_errors, grade.passed, grade.ran_ok, grade.total) == (2, 0, 0, 2)
    assert "ERROR" in grade.details and "FileNotFoundError" in grade.details


def test_checker_executor_exception_raises_checker_infra_error():
    """The checker leg honours the same contract: a raising backend surfaces as CheckerInfraError
    (bucketed into infra_errors), never as a raw OSError escaping the grade."""
    checker = CheckerVerdict("# unused", _RaisingSandbox(OSError(28, "No space left on device")))
    with pytest.raises(CheckerInfraError, match="No space left on device"):
        checker("input", "expected", "actual")


class _ScriptedSandbox(SandboxExecutor):
    """Returns queued results in order and records every ``run`` call's kwargs."""

    def __init__(self, results):
        self._results = list(results)
        self.calls: list[dict] = []

    def open_session(self):  # pragma: no cover
        raise NotImplementedError

    def run(self, code, *, stdin="", timeout=15.0, language="python", files=None):
        self.calls.append({"timeout": timeout, "files": files})
        return self._results.pop(0)


def test_checker_backend_failure_buckets_into_infra_errors():
    """run_solution_against_tests must count a checker-run backend failure as an infra-lost test
    (no pass, no ran_ok), so an all-infra special-judge outage trips the no-signal guard."""
    # Per test: a clean solution run, then a checker run that fails at the backend.
    sandbox = _ScriptedSandbox(
        [
            SandboxResult(stdout="42\n", returncode=0),
            SandboxResult(error="backend down"),
            SandboxResult(stdout="42\n", returncode=0),
            SandboxResult(error="backend down"),
        ]
    )
    verdict = CheckerVerdict("# checker", sandbox)
    grade = run_solution_against_tests(
        "print(42)",
        [{"input": "a", "output": "42"}, {"input": "b", "output": "42"}],
        sandbox=sandbox,
        verdict_fn=verdict,
    )
    assert grade.infra_errors == 2
    assert grade.passed == 0
    assert grade.ran_ok == 0
    assert "ERROR" in grade.details


class _TickingSandbox(_ScriptedSandbox):
    """Scripted sandbox that advances a fake clock by ``seconds_per_run`` on every ``run``."""

    def __init__(self, results, clock, seconds_per_run):
        super().__init__(results)
        self._clock = clock
        self._seconds_per_run = seconds_per_run

    def run(self, code, **kwargs):
        self._clock["t"] += self._seconds_per_run
        return super().run(code, **kwargs)


def test_every_grade_states_how_many_tests_it_judged():
    """``graded`` separates a partial grade from a wrong solution, so no construction may fall back
    to a default: 0 is the wrong answer for a fully judged run, and ``total`` is the wrong answer for
    a budget stop. The empty-pool grade states 0 of 0 explicitly."""
    assert "graded" not in GradeResult._field_defaults, "graded must be stated at every construction"

    tests = [{"input": str(i), "output": "42"} for i in range(3)]
    ok = SandboxResult(stdout="42\n", returncode=0)
    full = run_solution_against_tests("print(42)", tests, sandbox=_ScriptedSandbox([ok, ok, ok]))
    assert (full.graded, full.total, full.budget_hit) == (3, 3, False)

    empty = run_solution_against_tests("print(42)", [], sandbox=_ScriptedSandbox([]))
    assert (empty.graded, empty.total) == (0, 0)


def test_max_grading_seconds_bounds_the_sequential_grading_cost():
    """Each test is a sequential sandbox run; the wall-clock budget must stop grading between tests
    — an unbounded several-hundred-test row otherwise stalls a whole rollout round behind one
    episode's grade. The FULL pool stays the denominator: the budget only ever stops slow solutions,
    so crediting the ungraded remainder would make being slow the higher-reward strategy."""
    tests = [{"input": str(i), "output": "42"} for i in range(50)]
    clock = {"t": 0.0}

    def ok():
        return SandboxResult(stdout="42\n", returncode=0)

    # 10s/test against a 25s budget: tests 1-3 run, t=30 stops grading before test 4.
    sandbox = _TickingSandbox([ok() for _ in range(3)], clock, seconds_per_run=10.0)
    with mock.patch.object(grading_module.time, "monotonic", lambda: clock["t"]):
        grade = run_solution_against_tests("print(42)", tests, sandbox=sandbox, max_grading_seconds=25.0)
    assert len(sandbox.calls) == 3
    assert grade.passed == 3
    assert grade.total == 50, "an ungraded test must not be scored as passed — that pays for slowness"
    assert "first 3 of 50" in grade.details and "grading budget reached" in grade.details

    # At least one test always runs, even when a single run overshoots the whole budget.
    clock["t"] = 0.0
    sandbox_one = _TickingSandbox([ok()], clock, seconds_per_run=10.0)
    with mock.patch.object(grading_module.time, "monotonic", lambda: clock["t"]):
        grade_one = run_solution_against_tests("print(42)", tests, sandbox=sandbox_one, max_grading_seconds=5.0)
    assert len(sandbox_one.calls) == 1
    assert grade_one.passed == 1 and grade_one.total == 50

    sandbox_all = _ScriptedSandbox([ok() for _ in range(50)])
    grade_all = run_solution_against_tests("print(42)", tests, sandbox=sandbox_all)
    assert grade_all.total == 50 and len(sandbox_all.calls) == 50


def test_partial_grading_reaches_the_episode_metrics():
    """A grade cut short must be visible per episode, or a whole run's partial grading is unreadable.

    ``outcome/test_pass_frac`` keeps the FULL pool as its denominator, so a budget stop looks exactly
    like a wrong solution there. ``episode/tests_graded_frac`` and ``episode/grading_budget_hit`` are
    the only signals that grading stopped early — they must survive the whole path (``GradeResult`` ->
    ``trajectory.info`` -> ``rollout_metrics``), not just the grader.
    """
    tests = [{"input": str(i), "output": "42"} for i in range(50)]
    ok = SandboxResult(stdout="42\n", returncode=0)
    submit = {
        "id": "c1",
        "type": "function",
        "function": {"name": "submit_solution", "arguments": json.dumps({"code": "print(42)"})},
    }
    context = {"answer": {"tests": tests}}
    clock = {"t": 0.0}

    # 10s/test against a 25s budget: tests 1-3 run, t=30 stops grading before test 4.
    stopped = CodeContestsEnvironment(
        language="python",
        max_submissions=1,
        max_grading_seconds=25.0,
        sandbox=_TickingSandbox([ok, ok, ok], clock, seconds_per_run=10.0),
    )
    eids, _ = stopped.reset(["Print 42."], [context])
    with mock.patch.object(grading_module.time, "monotonic", lambda: clock["t"]):
        stopped.step(eids, [""], [{"tool_calls": [submit]}])
    metrics = stopped.rollout_metrics(stopped.get_trajectories(eids)[0])

    assert metrics["episode/tests_graded_frac"] == pytest.approx(3 / 50)
    assert metrics["episode/grading_budget_hit"] == pytest.approx(1.0)
    # The scoring denominator is untouched: the prefix is diagnosis, never credit.
    assert metrics["outcome/test_pass_frac"] == pytest.approx(3 / 50)

    complete = CodeContestsEnvironment(
        language="python", max_submissions=1, sandbox=_ScriptedSandbox([ok] * len(tests))
    )
    eids, _ = complete.reset(["Print 42."], [context])
    complete.step(eids, [""], [{"tool_calls": [submit]}])
    full_metrics = complete.rollout_metrics(complete.get_trajectories(eids)[0])

    assert full_metrics["episode/tests_graded_frac"] == pytest.approx(1.0)
    assert full_metrics["episode/grading_budget_hit"] == pytest.approx(0.0)


def test_grade_solution_gives_checker_infra_timeout_not_solution_limit():
    """The checker runs at the infra default timeout, never the solution's clamped per-test limit —
    a tight C++-tuned time_limit must TLE the solution, not the trusted judge grading it."""
    sandbox = _ScriptedSandbox(
        [
            SandboxResult(stdout="ok\n", returncode=0),
            SandboxResult(stdout="1\n", returncode=0),
        ]
    )
    grade = grade_solution(
        "code",
        [{"input": "x", "output": "ok"}],
        # compiled: the 2.0s limit is not floored to the interpreter default
        GradingSpec(sandbox=sandbox, language="cpp"),
        checker="# checker",
        time_limit=2.0,
    )
    assert grade.passed == 1
    solution_call, checker_call = sandbox.calls
    assert solution_call["timeout"] == pytest.approx(2.0)
    assert checker_call["files"] is not None and "checker.py" in checker_call["files"]
    assert checker_call["timeout"] == pytest.approx(SANDBOX_DEFAULT_TIMEOUT)


def test_outcome_verdict_hides_expected_and_produced_output():
    """Under ``verdict_detail="outcome"``, the default, a wrong answer is a verdict, not a diff. With
    the expected output shown, the graded channel doubles as a free test oracle and submit-first
    out-earns test-first within a GRPO group."""
    sandbox = StubSandbox(SandboxResult(stdout="X\n", returncode=0))
    tests = [{"input": "1", "output": "Y"}, {"input": "2", "output": "X"}]
    full = run_solution_against_tests("code", tests, sandbox=sandbox, verdict_detail="full")
    outcome = run_solution_against_tests("code", tests, sandbox=sandbox)
    assert "Expected: Y" in full.details and "Got:      X" in full.details
    assert "Test 1: FAIL" in outcome.details
    assert "Expected" not in outcome.details and "Got:" not in outcome.details
    assert "Y" not in outcome.details, outcome.details
    assert (outcome.passed, outcome.total) == (full.passed, full.total) == (1, 2)
    with pytest.raises(ValueError, match="verdict_detail"):
        run_solution_against_tests("code", tests, sandbox=sandbox, verdict_detail="diff")


@pytest.mark.parametrize("seconds", [0.0, -1.0, float("nan"), float("inf")])
@pytest.mark.parametrize("field", ["default_timeout", "max_time_limit"])
def test_the_grading_contract_refuses_a_time_limit_that_is_not_positive(field, seconds):
    """A run's wait on its program cannot be bounded by a negative or non-finite limit."""
    with pytest.raises(ValueError, match=field):
        GradingSpec(sandbox=StubSandbox(), **{field: seconds})


def test_a_negative_timeout_per_test_is_refused_at_construction():
    with pytest.raises(ValueError, match="timeout_per_test"):
        CodeContestsEnvironment(language="python", sandbox=StubSandbox(), timeout_per_test=-1.0)


def test_runtime_errors_fold_across_exit_codes_under_outcome():
    """Each exit code is its own verdict under ``full``; under ``outcome`` every crash is one."""
    crashes = [SandboxResult(stdout="", returncode=code) for code in (65, 66, 67)]
    tests = [{"input": str(k), "output": "1"} for k in range(3)]
    outcome = run_solution_against_tests("code", tests, sandbox=_ScriptedSandbox(crashes))
    full = run_solution_against_tests("code", tests, sandbox=_ScriptedSandbox(crashes), verdict_detail="full")
    assert outcome.details.splitlines()[1:] == ["Tests 1, 2, 3: RUNTIME ERROR"], outcome.details
    assert full.details.count("RUNTIME ERROR (exit") == 3, full.details


def test_verdict_detail_travels_with_the_grading_contract():
    """The env builds the spec once and the offline re-grader takes it back through ``to_meta``."""
    sandbox = StubSandbox(SandboxResult(stdout="X\n", returncode=0))
    spec = GradingSpec(sandbox=sandbox, verdict_detail="outcome")
    assert spec.to_meta()["verdict_detail"] == "outcome"
    assert spec.with_meta({"verdict_detail": "full"}).verdict_detail == "full"
    env = CodeContestsEnvironment(language="python", sandbox=sandbox, verdict_detail="outcome")
    assert env.grading_spec.verdict_detail == "outcome"
    with pytest.raises(ValueError, match="verdict_detail"):
        CodeContestsEnvironment(language="python", sandbox=sandbox, verdict_detail="diff")


def test_select_verdict_prefers_checker():
    """A non-empty checker always wins over the comparison mode."""
    v = select_verdict("# checker", "tokens", StubSandbox(SandboxResult(stdout="1", returncode=0)))
    assert isinstance(v, CheckerVerdict)


def test_select_verdict_unknown_comparison_raises():
    with pytest.raises(ValueError):
        select_verdict(None, "bogus", StubSandbox(SandboxResult()))


def test_a_runtime_error_excerpt_ends_with_the_exception_line():
    """A traceback names the exception on its last line; a head excerpt of a long one showed the frames
    and dropped the error, and the policy then guessed at the cause."""
    frames = "".join(f'  File "main.py", line {n}, in solve\n    step_{n}()\n' for n in range(12))
    stderr = (
        "Traceback (most recent call last):\n" + frames + "AttributeError: module 'math' has no attribute 'gamma2'"
    )
    assert len(stderr) > 2 * _STDERR_EXCERPT_CHARS
    crash = StubSandbox(SandboxResult(stdout="", stderr=stderr, returncode=1))
    grade = run_solution_against_tests("code", [{"input": "1", "output": "1"}], sandbox=crash, verdict_detail="full")
    assert "AttributeError: module 'math' has no attribute 'gamma2'" in grade.details
    assert "Traceback (most recent call last)" not in grade.details
    assert "Stderr: …" in grade.details


def test_tests_failing_identically_fold_into_one_verdict_outside_the_cap():
    """Eight tests crash the same way and one fails on output: the crash is one entry naming its tests,
    it takes one slot of the cap, and the distinct failure is still shown, not suppressed."""
    crash = SandboxResult(stdout="", stderr="ZeroDivisionError: division by zero", returncode=1)
    n = _MAX_FAILURE_DETAILS + 3
    sandbox = _ScriptedSandbox([crash] * n + [SandboxResult(stdout="X\n", returncode=0)])
    tests = [{"input": str(k), "output": "1"} for k in range(n)] + [{"input": "z", "output": "Y"}]
    grade = run_solution_against_tests("code", tests, sandbox=sandbox, verdict_detail="full")
    assert (grade.passed, grade.total) == (0, n + 1)
    assert grade.details.count("RUNTIME ERROR") == 1
    assert f"Tests 1, 2, 3, 4, 5, 6 and {n - 6} more: RUNTIME ERROR (exit 1)" in grade.details
    assert "ZeroDivisionError: division by zero" in grade.details
    assert f"Test {n + 1}: FAIL" in grade.details
    assert "details omitted" not in grade.details


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
