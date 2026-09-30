#!/usr/bin/env python
"""CPU tests: what a code-contests submission scores, and what the scratchpad says about a missing stdin.

The grade is the judge's accept: 1 when the submitted program passes every hidden test, 0 otherwise, so a
near miss scores like a wrong answer. Every graded submission after the first pays a flat price, and the
task message states it so that not resubmitting is an option the policy can weigh.

The scratchpad half: a program that reads input it was not given ends in a parse error or in
silence, and the result names the cause so the next run is not spent the same way.

Run: python tests/cpu/environments/test_submission_scoring.py  (or pytest)
"""

import pytest

from src.environments.base import OBJECTIVE_REWARD_KEY, REWARD_COMPONENTS_KEY
from src.environments.envs.tasks.coding.code_contests import (
    NO_STDIN_NOTE,
    STARVED_RUN_NOTE,
    SUBMISSION_PASS_FRACS_KEY,
    CodeContestsEnvironment,
)
from src.environments.sandbox.base import REPL_NO_OUTPUT_MESSAGE, SandboxExecutor, SandboxResult
from src.environments.tools.definitions import NativeToolCall
from tests.common.code_contests import StubSandbox

PENALTY = 0.2
PROFILES = {"high": {"max_submissions": 3, "max_test_calls": 8}}
# Four hidden tests whose input is its own index; ``_PassesSandbox`` passes test ``i`` when the
# submitted "program" contains the digit ``i``, so a program's text is its pass set.
TESTS = {"answer": {"tests": [{"input": str(i), "output": "ok"} for i in range(4)]}}
PRICE_RULE = "every resubmission costs part of the score"


class _PassesSandbox(SandboxExecutor):
    def open_session(self):
        raise NotImplementedError

    def run(self, code, *, stdin="", timeout=15.0, language="python", files=None):
        return SandboxResult(stdout="ok\n" if stdin and stdin in code else "no\n", returncode=0)


def _env(**kwargs):
    kwargs.setdefault("sandbox", _PassesSandbox())
    kwargs.setdefault("reasoning_effort_profiles", PROFILES)
    kwargs.setdefault("resubmission_penalty", PENALTY)
    return CodeContestsEnvironment(**kwargs)


def _episode(env):
    ids, _ = env.reset(["solve it"], [{"reasoning_effort": "high", **TESTS}])
    return env.get_trajectories(ids)[0]


def _call(env, traj, name, **arguments):
    results, _ = env._execute_tool_calls([NativeToolCall(id="c", name=name, arguments=arguments)], traj)
    return results[0].content


def _graded(env, programs):
    traj = _episode(env)
    for program in programs:
        _call(env, traj, "submit_solution", code=program)
    env._settle_grade(traj, None)
    components = traj.info[REWARD_COMPONENTS_KEY]
    assert traj.total_reward == pytest.approx(sum(components.values())), "the decomposition no longer sums"
    return components, traj


@pytest.mark.parametrize(
    ("programs", "objective"),
    [
        (["0123"], 1.0),
        (["012"], 0.0),  # 3 of 4: a near miss scores like a wrong answer
        (["0"], 0.0),
        (["0", "0123"], 1.0),  # the last submission is the graded one
        (["0123", "012"], 0.0),
    ],
)
def test_the_objective_is_the_judges_accept_of_the_last_submission(programs, objective):
    components, _ = _graded(_env(), programs)
    assert components[OBJECTIVE_REWARD_KEY] == objective


def test_a_never_submitted_episode_grades_zero():
    components, _ = _graded(_env(), [])
    assert components[OBJECTIVE_REWARD_KEY] == 0.0
    assert components["reward/submission"] == 0.0


@pytest.mark.parametrize("programs", [["0"], ["0", "012"], ["012", "0"], ["0", "012", "0123"]])
def test_every_graded_submission_after_the_first_pays_the_flat_price(programs):
    components, traj = _graded(_env(), programs)
    assert components["reward/resubmission"] == pytest.approx(-PENALTY * (len(programs) - 1))
    assert len(traj.info[SUBMISSION_PASS_FRACS_KEY]) == len(programs)


def test_the_task_message_states_the_price_only_when_a_resubmission_can_be_charged():
    stated = _episode(_env()).messages[-1].content
    assert PRICE_RULE in stated and "3 graded submissions" in stated

    assert PRICE_RULE not in _episode(_env(resubmission_penalty=0.0)).messages[-1].content, "nothing is charged"
    single = _env(reasoning_effort_profiles={"high": {"max_submissions": 1}})
    assert PRICE_RULE not in _episode(single).messages[-1].content, "one submission has no resubmission to price"


def test_the_behavior_counter_is_the_share_of_resubmissions_that_improved():
    env = _env()
    _, traj = _graded(env, ["0", "012", "01"])
    assert env.rollout_metrics(traj)["episode/resubmission_improved"] == pytest.approx(0.5)
    _, once = _graded(env, ["0123"])
    assert "episode/resubmission_improved" not in env.rollout_metrics(once), "no resubmission, no share"


@pytest.mark.parametrize(
    ("result", "stdin", "note"),
    [
        # Silent without input: nothing was learned, so the run is returned and the reply says so.
        (SandboxResult(stdout="", returncode=0), "", STARVED_RUN_NOTE),
        (SandboxResult(stdout="", stderr="ValueError: invalid literal for int()", returncode=1), "", NO_STDIN_NOTE),
        (SandboxResult(stdout="3\n", stderr="IndexError: list index out of range", returncode=1), "", NO_STDIN_NOTE),
        (SandboxResult(stdout="", stderr="ValueError: invalid literal for int()", returncode=1), "5\n", None),
        (SandboxResult(stdout="", returncode=0), "5\n", None),
        (SandboxResult(stdout="42\n", returncode=0), "", None),
        # A build that failed or a run that timed out says nothing about a missing input.
        (SandboxResult(stderr="main.py: error: bad", returncode=1, compile_failed=True), "", None),
        (SandboxResult(timed_out=True), "", None),
    ],
)
def test_a_starved_scratchpad_run_names_the_missing_stdin(result, stdin, note):
    env = _env(sandbox=StubSandbox(result))
    arguments = {"code": "print(int(input()))", **({"stdin": stdin} if stdin else {})}
    observation = _call(env, _episode(env), "python_repl", **arguments)
    assert [n for n in (NO_STDIN_NOTE, STARVED_RUN_NOTE) if n in observation] == ([note] if note else []), observation
    if result.stdout == "" and result.returncode == 0:
        assert observation.startswith(REPL_NO_OUTPUT_MESSAGE)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
