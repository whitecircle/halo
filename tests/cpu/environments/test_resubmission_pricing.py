#!/usr/bin/env python
"""CPU tests: what a code-contests resubmission costs, and what the scratchpad says about a missing stdin.

A flat price on every graded submission after the first lands on a fix and on a re-roll alike. With
``improved_resubmission_refund`` a resubmission that beats every earlier result earns most of the
price back, so within a group the fix out-scores the re-roll by the price itself, and the task message
states the rule so that not resubmitting is an option the policy can weigh.

The scratchpad half: a program that reads input it was not given ends in a parse error or in
silence, and the result names the cause so the next run is not spent the same way.

Run: python tests/cpu/environments/test_resubmission_pricing.py  (or pytest)
"""

import pytest

from src.environments.base import REWARD_COMPONENTS_KEY
from src.environments.envs.tasks.coding.code_contests import (
    NO_STDIN_NOTE,
    SUBMISSION_PASS_FRACS_KEY,
    CodeContestsEnvironment,
)
from src.environments.sandbox.base import REPL_NO_OUTPUT_MESSAGE, SandboxExecutor, SandboxResult
from src.environments.tools.definitions import NativeToolCall
from tests.common.code_contests import StubSandbox

PENALTY = 0.2
REFUND = 0.75
PROFILES = {"high": {"max_submissions": 3, "max_test_calls": 8}}
# Four hidden tests whose input is its own index; ``_PassesSandbox`` passes test ``i`` when the
# submitted "program" contains the digit ``i``, so a program's text is its pass set.
TESTS = {"answer": {"tests": [{"input": str(i), "output": "ok"} for i in range(4)]}}


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


def _resubmission_charge(env, programs):
    traj = _episode(env)
    for program in programs:
        _call(env, traj, "submit_solution", code=program)
    env._settle_grade(traj, None)
    components = traj.info[REWARD_COMPONENTS_KEY]
    assert traj.total_reward == pytest.approx(sum(components.values())), "the decomposition no longer sums"
    return components["reward/resubmission"], traj


@pytest.mark.parametrize(
    ("programs", "priced_units"),
    [
        (["0"], 0.0),
        (["0", "012"], 1 - REFUND),  # a fix: 1/4 -> 3/4
        (["012", "0123"], 1 - REFUND),  # a rescue
        (["012", "013"], 1.0),  # a re-roll that ties the first result
        (["012", "0"], 1.0),  # a re-roll that regresses
        (["0", "012", "01"], (1 - REFUND) + 1.0),  # a fix, then a regression
        (["0", "012", "0123"], 2 * (1 - REFUND)),  # two fixes
        (["012", "0", "01"], 2.0),  # the third beats the second but not the best so far
    ],
)
def test_a_resubmission_is_refunded_only_when_it_beats_every_earlier_result(programs, priced_units):
    charge, traj = _resubmission_charge(_env(improved_resubmission_refund=REFUND), programs)
    assert charge == pytest.approx(-PENALTY * priced_units)
    assert len(traj.info[SUBMISSION_PASS_FRACS_KEY]) == len(programs)


@pytest.mark.parametrize("programs", [["0", "012"], ["012", "0"], ["0", "012", "0123"]])
def test_without_a_refund_every_resubmission_pays_the_full_price(programs):
    charge, _ = _resubmission_charge(_env(), programs)
    assert charge == pytest.approx(-PENALTY * (len(programs) - 1))


def test_within_a_group_a_fix_out_scores_a_re_roll_by_more_than_the_objective_gap():
    """The decision the term exists to price: the same first result, then a fix or a re-roll that lands
    on the same final pass fraction. Without the refund the two score alike."""
    flat_fix, _ = _resubmission_charge(_env(), ["0", "012"])
    flat_reroll, _ = _resubmission_charge(_env(), ["012", "013"])
    assert flat_fix == pytest.approx(flat_reroll)

    fix, _ = _resubmission_charge(_env(improved_resubmission_refund=REFUND), ["0", "012"])
    reroll, _ = _resubmission_charge(_env(improved_resubmission_refund=REFUND), ["012", "013"])
    assert fix - reroll == pytest.approx(PENALTY * REFUND)


def test_the_task_message_states_the_rule_only_when_a_refund_makes_it_a_decision():
    rule = "does not beat your best result so far costs part of the score"
    stated = _episode(_env(improved_resubmission_refund=REFUND)).messages[-1].content
    assert rule in stated and "3 graded submissions" in stated

    assert rule not in _episode(_env()).messages[-1].content, "a flat price states no rule today"
    unpriced = _env(resubmission_penalty=0.0, improved_resubmission_refund=REFUND)
    assert rule not in _episode(unpriced).messages[-1].content, "nothing is charged, so nothing is stated"
    single = _env(improved_resubmission_refund=REFUND, reasoning_effort_profiles={"high": {"max_submissions": 1}})
    assert rule not in _episode(single).messages[-1].content, "one submission has no resubmission to price"


@pytest.mark.parametrize("bad", [-0.1, 1.1, float("nan"), float("inf")])
def test_the_refund_is_a_fraction(bad):
    with pytest.raises(ValueError, match="improved_resubmission_refund"):
        _env(improved_resubmission_refund=bad)


def test_the_behavior_counter_is_the_share_of_resubmissions_that_improved():
    env = _env(improved_resubmission_refund=REFUND)
    _, traj = _resubmission_charge(env, ["0", "012", "01"])
    assert env.rollout_metrics(traj)["episode/resubmission_improved"] == pytest.approx(0.5)
    _, once = _resubmission_charge(env, ["0123"])
    assert "episode/resubmission_improved" not in env.rollout_metrics(once), "no resubmission, no share"


@pytest.mark.parametrize(
    ("result", "stdin", "noted"),
    [
        (SandboxResult(stdout="", returncode=0), "", True),
        (SandboxResult(stdout="", stderr="ValueError: invalid literal for int()", returncode=1), "", True),
        (SandboxResult(stdout="3\n", stderr="IndexError: list index out of range", returncode=1), "", True),
        (SandboxResult(stdout="", stderr="ValueError: invalid literal for int()", returncode=1), "5\n", False),
        (SandboxResult(stdout="", returncode=0), "5\n", False),
        (SandboxResult(stdout="42\n", returncode=0), "", False),
    ],
)
def test_a_starved_scratchpad_run_names_the_missing_stdin(result, stdin, noted):
    env = _env(sandbox=StubSandbox(result))
    arguments = {"code": "print(int(input()))", **({"stdin": stdin} if stdin else {})}
    observation = _call(env, _episode(env), "python_repl", **arguments)
    assert observation.endswith(NO_STDIN_NOTE) is noted, observation
    if result.stdout == "" and result.returncode == 0:
        assert observation.startswith(REPL_NO_OUTPUT_MESSAGE)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
