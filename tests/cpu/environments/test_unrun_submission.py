#!/usr/bin/env python
"""The behaviour counter on a failed submission whose program never ran as submitted: an episode reads
``episode/unrun_failed_submission`` 1 when such a program was submitted while a run was left, and a failure with the
runs spent does not count. It prices nothing."""

import json

import pytest

from halo.environments.sandbox.base import SandboxExecutor
from halo.environments.sandbox.results import SandboxResult
from halo.environments.tasks.coding.code_contests import CodeContestsEnvironment
from halo.environments.tasks.coding.program_source import is_output_only_change, normalized_program
from halo.environments.tools.definitions import NativeToolCall

# Two hidden tests whose input is its own index; the stub passes test ``i`` when the program's text holds ``i``.
TESTS = {"answer": {"tests": [{"input": str(i), "output": "ok"} for i in range(2)]}}
PROFILES = {"high": {"max_submissions": 3, "max_test_calls": 2}}
FAILS = "x = 'none'\nprint(x)\n"
FAILS_TOO = "y = 'none'\nprint(y)\n"
PASSES = "x = '01'\nprint(x)\n"


class _PassesSandbox(SandboxExecutor):
    def open_session(self):
        raise NotImplementedError

    def run(self, code, *, stdin="", timeout=15.0, language="python", files=None):
        return SandboxResult(stdout="ok\n" if stdin and stdin in code else "no\n", returncode=0)


def _episode():
    env = CodeContestsEnvironment(sandbox=_PassesSandbox(), reasoning_effort_profiles=PROFILES)
    ids, _ = env.reset(["solve it"], [{"reasoning_effort": "high", **TESTS}])
    return env, env.get_trajectories(ids)[0]


def _turn(env, traj, *calls):
    """One turn: its calls in order, under the turn's own stamp of runs left and programs run."""
    batch = [NativeToolCall(id=f"c{i}", name=name, arguments=arguments) for i, (name, arguments) in enumerate(calls)]
    return env._execute_tool_calls(batch, traj)[0]


def _run(code):
    """A scratchpad call (a Python-only run names its scratchpad ``python_repl``)."""
    return ("python_repl", {"code": code, "stdin": "0"})


def _submit(code):
    return ("submit_solution", {"code": code})


def _unrun_mark(env, traj):
    env._settle_grade(traj, None)
    return env.rollout_metrics(traj)["episode/unrun_failed_submission"]


def _wire_call(name, call_id, **arguments):
    return {"id": call_id, "function": {"name": name, "arguments": json.dumps(arguments)}}


def test_a_price_on_the_mark_is_an_unknown_option():
    with pytest.raises(TypeError, match="unrun_submission_penalty"):
        CodeContestsEnvironment(sandbox=_PassesSandbox(), unrun_submission_penalty=0.1)


def test_the_mark_survives_the_episodes_settlement_through_env_step():
    """Driven to its end through ``env.step``, the episode drops its grading payload before its metrics are read:
    the behaviour counter still sees the unrun failure."""
    env = CodeContestsEnvironment(sandbox=_PassesSandbox(), reasoning_effort_profiles=PROFILES)
    ids, _ = env.reset(["solve it"], [{"reasoning_effort": "high", **TESTS}])
    for i, code in enumerate((FAILS, FAILS_TOO, "z = 'none'\n")):
        step = env.step(ids, [""], [{"tool_calls": [_wire_call("submit_solution", f"s{i}", code=code)]}])[0]
    assert step.done and "_failed_submission_ran" not in step.trajectory.info
    assert env.rollout_metrics(step.trajectory)["episode/unrun_failed_submission"] == 1.0


def test_a_refused_run_is_not_a_run():
    """A program the scratchpad refused (here mislabelled) never ran, so submitting it afterwards is unrun."""
    env = CodeContestsEnvironment(
        sandbox=_PassesSandbox(), language=["python", "cpp"], reasoning_effort_profiles=PROFILES
    )
    ids, _ = env.reset(["solve it"], [{"reasoning_effort": "high", **TESTS}])
    traj = env.get_trajectories(ids)[0]
    cpp = "#include <bits/stdc++.h>\nint main() { std::cout << 2; }\n"
    (refused,) = _turn(env, traj, ("run_code", {"code": cpp, "language": "python", "stdin": "0"}))
    assert refused.refused
    _turn(env, traj, ("submit_solution", {"code": cpp, "language": "cpp"}))
    assert _unrun_mark(env, traj) == 1.0


def test_failed_submissions_never_run_count_once():
    env, traj = _episode()
    _turn(env, traj, _submit(FAILS))
    _turn(env, traj, _submit(FAILS_TOO))
    assert _unrun_mark(env, traj) == 1.0


def test_a_failure_run_in_an_earlier_turn_does_not_count_whatever_its_comments_and_spacing():
    env, traj = _episode()
    _turn(env, traj, _run(FAILS))
    _turn(env, traj, _submit("# my attempt\nx = 'none'   \nprint(x)  # the answer\n\n"))
    assert _unrun_mark(env, traj) == 0.0


def test_a_failure_that_only_dropped_its_debug_prints_is_the_program_that_ran():
    env, traj = _episode()
    _turn(env, traj, _run("x = 'none'\nprint('debug', x, file=sys.stderr)\nprint(x)\n"))
    _turn(env, traj, _submit(FAILS))
    assert _unrun_mark(env, traj) == 0.0


def test_a_changed_line_after_the_last_run_counts():
    env, traj = _episode()
    _turn(env, traj, _run(FAILS))
    _turn(env, traj, _submit("x = 'none' * 2\nprint(x)\n"))
    assert _unrun_mark(env, traj) == 1.0


def test_a_run_in_the_submissions_own_turn_tested_nothing_it_was_written_from():
    env, traj = _episode()
    _turn(env, traj, _run(FAILS), _submit(FAILS))
    assert _unrun_mark(env, traj) == 1.0


def test_a_solve_keeps_the_mark_of_an_earlier_unrun_failure():
    env, traj = _episode()
    _turn(env, traj, _submit(FAILS))
    _turn(env, traj, _submit(PASSES))
    assert traj.info["submission_result"].startswith("Passed 2/2")
    assert _unrun_mark(env, traj) == 1.0


def test_a_failure_submitted_with_the_runs_spent_does_not_count():
    env, traj = _episode()
    _turn(env, traj, _run("print(1)\n"), _run("print(2)\n"))
    _turn(env, traj, _submit(FAILS))
    assert _unrun_mark(env, traj) == 0.0


@pytest.mark.parametrize(
    ("ran", "submitted", "language", "same"),
    [
        ("n = int(input())\nprint('dbg', n)\nprint(n * 2)\n", "n = int(input())\nprint(n * 2)\n", "python", True),
        ("int n; cin >> n;\ncerr << n << endl;\ncout << 2 * n;\n", "int n; cin >> n;\ncout << 2 * n;\n", "cpp", True),
        ("n = int(input())\nn += 1\nprint(n)\n", "n = int(input())\nprint(n)\n", "python", False),
        ("n = int(input())\nprint(n)\n", "n = int(input())\nn += 1\nprint(n)\n", "python", False),
        (
            "for i in range(3):\n    a = i\n    print('dbg', a)\nfor i in range(3):\n    a = i\n    print('dbg', a)\nprint(a)\n",
            "for i in range(3):\n    a = i\nfor i in range(3):\n    a = i\nprint(a)\n",
            "python",
            True,
        ),
    ],
    ids=["python-debug-print", "cpp-cerr", "dropped-logic", "added-line", "repeated-blocks"],
)
def test_is_output_only_change_allows_dropped_output_statements_and_nothing_else(ran, submitted, language, same):
    assert is_output_only_change(normalized_program(ran, language), normalized_program(submitted, language)) is same


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
