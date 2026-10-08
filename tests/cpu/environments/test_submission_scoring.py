#!/usr/bin/env python
"""CPU tests: what a code-contests submission scores, and what the scratchpad says about a missing stdin.

The grade is the judge's accept: 1 when the submitted program passes every hidden test, 0 otherwise, so a
near miss scores like a wrong answer. Every graded submission after the first pays a flat price; neither
the price nor the submission budget is stated to the model — a verdict is the grade alone, and the cap
ends the episode at its last graded submission.

The scratchpad half: a program run on no input says so, since one that reads input it was not given
ends in a parse error or in silence, and the result names the cause so the next run is not spent the
same way; a silent one counts in ``episode/starved_test_runs`` and its reply is marked uninformative.

Run: python tests/cpu/environments/test_submission_scoring.py  (or pytest)
"""

import json

import pytest

from src.environments.base import EPISODE_TOOL_BUDGETS_KEY, REWARD_COMPONENTS_KEY
from src.environments.envs.tasks.coding.code_contests import (
    NO_STDIN_NOTE,
    SUBMISSION_PASS_FRACS_KEY,
    CodeContestsEnvironment,
)
from src.environments.sandbox.base import REPL_NO_OUTPUT_MESSAGE, SandboxExecutor, SandboxResult
from src.environments.tools.definitions import NativeToolCall
from src.rewards.terms import OBJECTIVE_REWARD_KEY
from tests.common.code_contests import StubSandbox, retired_budget_phrases

PENALTY = 0.2
PROFILES = {"high": {"max_submissions": 3, "max_test_calls": 8}}
# Four hidden tests whose input is its own index; ``_PassesSandbox`` passes test ``i`` when the
# submitted "program" contains the digit ``i``, so a program's text is its pass set (submitted as a
# comment, which compiles as Python).
TESTS = {"answer": {"tests": [{"input": str(i), "output": "ok"} for i in range(4)]}}
# The resubmission-price rule no task message states.
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
        # A distinct program per entry (comments aside: a repeat is refused unrun), carrying the text the stub reads.
        _call(env, traj, "submit_solution", code=f"program = {program!r}")
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
        (["0123", "012"], 1.0),  # an accept stands: a later submission is not graded
    ],
)
def test_the_objective_is_the_judges_accept_of_the_last_submission(programs, objective):
    components, _ = _graded(_env(), programs)
    assert components[OBJECTIVE_REWARD_KEY] == objective


def _step_submit(env, ids, program):
    """One turn submitting ``program`` (as a string statement, like :func:`_graded`) through the protocol's step."""
    arguments = json.dumps({"code": f"program = {program!r}"})
    call = {"id": "s", "type": "function", "function": {"name": "submit_solution", "arguments": arguments}}
    return env.step(ids, [""], [{"tool_calls": [call]}])[0]


def test_a_submission_passing_every_test_ends_the_episode():
    """Past an accept a resubmission can only lose the solve, so the accept ends the episode the way
    the submission cap does, completed, with graded submissions still unspent."""
    env = _env()
    ids, _ = env.reset(["solve it"], [{"reasoning_effort": "high", **TESTS}])
    first = _step_submit(env, ids, "01")
    assert not first.done
    accepted = _step_submit(env, ids, "0123")
    traj = accepted.trajectory
    assert accepted.done and not accepted.truncated and traj.info["completed"]
    assert env._submissions(traj) == 2, "one of the three graded submissions is left unspent"
    assert traj.info[REWARD_COMPONENTS_KEY][OBJECTIVE_REWARD_KEY] == 1.0


def test_an_accept_stands_against_a_later_submission_in_the_same_turn():
    """The episode ends after the turn that submitted the accept, so a submission later in that turn is
    refused unspent and unpaid, and grades nothing that could replace the solve."""
    env = _env(tool_success_reward=0.05)
    ids, _ = env.reset(["solve it"], [{"reasoning_effort": "high", **TESTS}])
    calls = [
        {
            "id": cid,
            "type": "function",
            "function": {"name": "submit_solution", "arguments": json.dumps({"code": code})},
        }
        for cid, code in (("a", "# 0123"), ("b", "# 0"))
    ]
    (step,) = env.step(ids, [""], [{"tool_calls": calls}])
    traj = step.trajectory
    assert traj.messages[-1].content == (
        "Not graded: an earlier submission already passed every test, so it stands and the task ends."
    )
    assert step.done and env._submissions(traj) == 1
    assert traj.info["submission_result"] == "Passed 4/4 test cases."
    components = traj.info[REWARD_COMPONENTS_KEY]
    assert components[OBJECTIVE_REWARD_KEY] == 1.0
    assert components["reward/resubmission"] == 0.0
    assert components["reward/turn_shaping"] == pytest.approx(0.05), "the refused call is not paid"


def test_a_graded_verdict_is_the_grade_alone_and_the_cap_still_ends_the_episode():
    """Every verdict the model reads is the grade and nothing about what is left: the submissions left are
    never stated, while the third graded submission of the ``high`` budget still ends the episode."""
    env = _env()
    ids, _ = env.reset(["solve it"], [{"reasoning_effort": "high", **TESTS}])
    steps, replies = [], []
    for program in ("0", "01", "012"):
        steps.append(_step_submit(env, ids, program))
        replies.append(steps[-1].trajectory.messages[-1].content)
    assert replies == [
        "Passed 1/4 test cases.\nTests 2, 3, 4: FAIL",
        "Passed 2/4 test cases.\nTests 3, 4: FAIL",
        "Passed 3/4 test cases.\nTest 4: FAIL",
    ]
    assert [step.done for step in steps] == [False, False, True]
    traj = steps[-1].trajectory
    assert traj.info["completed"] and not traj.truncated
    assert traj.info[EPISODE_TOOL_BUDGETS_KEY]["submit_solution"] == 3 == env._submissions(traj)
    assert traj.info["submission_result"] == replies[-1], "the record keeps the verdict alone"

    ids, _ = env.reset(["solve it"], [{"reasoning_effort": "high", **TESTS}])
    accepted = _step_submit(env, ids, "0123").trajectory
    assert accepted.messages[-1].content == "Passed 4/4 test cases."
    assert not any(retired_budget_phrases(m.content) for m in accepted.messages if m.content)


def test_a_never_submitted_episode_grades_zero():
    components, _ = _graded(_env(), [])
    assert components[OBJECTIVE_REWARD_KEY] == 0.0
    assert components["reward/submission"] == 0.0


@pytest.mark.parametrize("programs", [["0"], ["0", "012"], ["012", "0"], ["0", "012", "0123"]])
def test_every_graded_submission_after_the_first_pays_the_flat_price(programs):
    components, traj = _graded(_env(), programs)
    assert components["reward/resubmission"] == pytest.approx(-PENALTY * (len(programs) - 1))
    assert len(traj.info[SUBMISSION_PASS_FRACS_KEY]) == len(programs)


def test_the_task_message_states_neither_the_price_nor_the_budget():
    """Whatever the ladder binds, the task message carries no price rule and no submission count; the
    price is still charged (``test_every_graded_submission_after_the_first_pays_the_flat_price``)."""
    for env in (
        _env(),
        _env(resubmission_penalty=0.0),
        _env(reasoning_effort_profiles={"high": {"max_submissions": 1}}),
    ):
        stated = _episode(env).messages[-1].content
        assert PRICE_RULE not in stated and "graded submission" not in stated, stated
        assert not retired_budget_phrases(stated), stated


def test_the_behavior_counter_is_the_share_of_resubmissions_that_improved():
    env = _env()
    _, traj = _graded(env, ["0", "012", "01"])
    assert env.rollout_metrics(traj)["episode/resubmission_improved"] == pytest.approx(0.5)
    _, rebound = _graded(env, ["012", "0", "01"])
    assert env.rollout_metrics(rebound)["episode/resubmission_improved"] == 0.0, (
        "beating the last is not beating the best"
    )
    _, once = _graded(env, ["0123"])
    assert "episode/resubmission_improved" not in env.rollout_metrics(once), "no resubmission, no share"


@pytest.mark.parametrize(
    ("result", "stdin", "noted", "starved"),
    [
        # Silent without input, stderr and blank lines aside (a clean exit's reply leaves stderr out).
        (SandboxResult(stdout="", returncode=0), "", True, True),
        (SandboxResult(stdout="", stderr="debug: read nothing", returncode=0), "", True, True),
        (SandboxResult(stdout=" \n", returncode=0), "", True, True),
        (SandboxResult(stdout="", stderr="ValueError: invalid literal for int()", returncode=1), "", True, False),
        (SandboxResult(stdout="3\n", stderr="IndexError: list index out of range", returncode=1), "", True, False),
        (SandboxResult(stdout="", stderr="ValueError: invalid literal for int()", returncode=1), "5\n", False, False),
        (SandboxResult(stdout="", returncode=0), "5\n", False, False),
        # Output computed from no input, and a loop on end-of-file, ran on nothing too.
        (SandboxResult(stdout="42\n", returncode=0), "", True, False),
        (SandboxResult(timed_out=True), "", True, False),
        # A build that failed, or ran past the compile limit, ran nothing.
        (SandboxResult(stderr="main.py: error: bad", returncode=1, compile_failed=True), "", False, False),
        (SandboxResult(stderr="compilation timed out after 10 s", compile_failed=True), "", False, False),
    ],
)
def test_a_scratchpad_run_on_no_input_names_it_and_a_silent_one_counts_as_starved(result, stdin, noted, starved):
    """Every call here spends its slot; ``episode/starved_test_runs`` counts the input-less runs that exited
    cleanly with nothing on stdout, and exactly those come back marked uninformative, the mark that flags a
    turn of nothing else."""
    env = _env(sandbox=StubSandbox(result))
    traj = _episode(env)
    arguments = {"code": "print(int(input()))", **({"stdin": stdin} if stdin else {})}
    (call,), _ = env._execute_tool_calls([NativeToolCall(id="c", name="python_repl", arguments=arguments)], traj)
    observation = call.content
    assert (NO_STDIN_NOTE in observation) is noted, observation
    assert call.uninformative is starved and call.success
    assert env._test_calls(traj) == 1
    assert env.rollout_metrics(traj)["episode/starved_test_runs"] == (1.0 if starved else 0.0)
    if starved and not result.stdout:
        assert observation == f"{REPL_NO_OUTPUT_MESSAGE}\n{NO_STDIN_NOTE}", observation


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
