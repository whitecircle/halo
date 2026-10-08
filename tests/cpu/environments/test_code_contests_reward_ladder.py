"""CPU tests for the CodeContestsEnvironment agentic-loop reward ladder: the grade and shaping
``_grade_episode`` returns, priced into ``reward_components`` by the base's ``_settle_grade``.

The base model answers in plain text and rarely calls submit_solution, so the verifier alone gives no
gradient to learn the loop. Small shaping rungs bootstrap it and self-neutralize per GRPO group:

  plain-text giveup (0 tool calls)  -> -no_tool_use_penalty          (reward/tool_shaping)
  tool call, no submission          ->  0
  submitted (any test result)       -> +submission_reward            (reward/submission)
  every hidden test passed          -> +weight                       (reward/objective, dominant;
                                       the environment term, default weight 1)

A grade that says nothing about the code — a backend outage, or one that stopped before any judged
test failed — pays no rung and leaves the group baseline.

    python tests/cpu/environments/test_code_contests_reward_ladder.py
"""

import pytest

from src.environments.base import (
    EPISODE_ERROR_KEY,
    EPISODE_INVALID_REASON_KEY,
    REWARD_COMPONENTS_KEY,
    Message,
    Trajectory,
)
from src.environments.envs.tasks.coding.code_contests import CodeContestsEnvironment
from src.environments.envs.tasks.coding.swe import SweEnvironment
from src.environments.envs.tasks.qa import ExamQAEnvironment
from src.environments.sandbox.base import SandboxInfraError
from src.environments.tools.definitions import NativeToolResult
from src.rewards.terms import OBJECTIVE_REWARD_KEY

SUB, PEN = 0.1, 0.1


def _env(**kw):
    kw.setdefault("submission_reward", SUB)
    kw.setdefault("no_tool_use_penalty", PEN)
    return CodeContestsEnvironment(language="python", sandbox_backend="local", **kw)


def _traj(*, turns=1, tool_calls=0, submitted=False, passed=0, total=10, **grade):
    t = Trajectory()
    for _ in range(turns):
        t.add_message(Message.assistant("..."))
    t.info["total_tool_calls"] = tool_calls
    t.info["tool_call_counts"] = {"submit_solution": 1} if submitted else {}
    if submitted:
        t.info["submission_result"] = "ok"
        t.info["tests_passed"] = passed
        t.info["tests_total"] = total
        t.info.update(grade)
    return t


def _reward(env, traj):
    """Price the finished episode the way ``_finalize_step`` does and return its total."""
    env._settle_grade(traj, None)
    return traj.total_reward


def test_ladder_is_monotonic():
    env = _env()
    giveup = _reward(env, _traj(turns=1, tool_calls=0, submitted=False))
    tooled_no_submit = _reward(env, _traj(turns=2, tool_calls=1, submitted=False))
    submit_zero = _reward(env, _traj(turns=1, tool_calls=1, submitted=True, passed=0))
    near_miss = _reward(env, _traj(turns=3, tool_calls=2, submitted=True, passed=9))
    solved = _reward(env, _traj(turns=3, tool_calls=2, submitted=True, passed=10))
    assert giveup == pytest.approx(-PEN)
    assert tooled_no_submit == pytest.approx(0.0)
    assert submit_zero == pytest.approx(SUB)
    assert near_miss == pytest.approx(SUB), "9 of 10 tests is a wrong answer: no partial credit"
    assert solved == pytest.approx(1.0 + SUB)
    assert giveup < tooled_no_submit < submit_zero == near_miss < solved


def test_all_shaping_off_by_default():
    env = CodeContestsEnvironment(language="python", sandbox_backend="local")
    assert (env.submission_reward, env.no_tool_use_penalty, env.resubmission_penalty) == (0.0, 0.0, 0.0)
    # Shaping off: a giveup grades 0, not penalized.
    assert _reward(env, _traj(tool_calls=0, submitted=False)) == pytest.approx(0.0)
    assert _reward(env, _traj(tool_calls=2, submitted=True, passed=5)) == pytest.approx(0.0)
    assert _reward(env, _traj(tool_calls=2, submitted=True, passed=10)) == pytest.approx(1.0)


@pytest.mark.parametrize("value", [-0.1, True, False, float("nan")], ids=["negative", "true", "false", "nan"])
@pytest.mark.parametrize(
    "knob",
    [
        "submission_reward",
        "resubmission_penalty",
        "no_tool_use_penalty",
        "tool_error_penalty",
        "tool_success_reward",
        "turn_overflow_penalty",
        "length_cutoff_penalty",
    ],
)
def test_a_magnitude_refuses_a_negative_a_bool_or_a_nan(knob, value):
    """A bool is an int subclass a sign check passes, pricing the knob at 0 or 1."""
    with pytest.raises(ValueError, match=knob):
        CodeContestsEnvironment(language="python", sandbox_backend="local", **{knob: value})


def test_turn_overflow_penalty_on_truncation():
    """An episode that burns ``max_turns`` without terminating (``trajectory.truncated``) pays
    ``turn_overflow_penalty`` on top of whatever the ladder gave it — pricing the turn-cap overflow the
    ladder otherwise leaves free. Applies to submitted and unsubmitted episodes alike; default 0 = off."""
    env = _env(turn_overflow_penalty=0.2)
    capped = _traj(turns=6, tool_calls=5, submitted=False)
    capped.truncated = True
    assert _reward(env, capped) == pytest.approx(-0.2)
    capped_sub = _traj(turns=6, tool_calls=5, submitted=True, passed=10)
    capped_sub.truncated = True
    assert _reward(env, capped_sub) == pytest.approx(1.0 + SUB - 0.2)
    assert _reward(env, _traj(turns=6, tool_calls=5, submitted=False)) == pytest.approx(0.0)
    # Default 0: truncation alone changes nothing.
    off = _traj(turns=6, tool_calls=5, submitted=False)
    off.truncated = True
    assert _reward(_env(), off) == pytest.approx(0.0)


def test_an_episode_its_driver_lost_pays_no_turn_overflow():
    """A driver that lost the episode (a generation that raised) stamps ``EPISODE_ERROR_KEY`` before
    finalizing it truncated: the fault is not the policy's, so the overflow price stays off."""
    env = _env(turn_overflow_penalty=0.2)
    lost = _traj(turns=2, tool_calls=2, submitted=False)
    lost.truncated = True
    lost.info[EPISODE_ERROR_KEY] = "RuntimeError: http boom"
    assert _reward(env, lost) == pytest.approx(0.0)


def test_tool_error_penalty_is_applied_negative():
    """The per-call error knob is a MAGNITUDE: a failed tool call must REDUCE the reward. A signed knob
    would silently turn a positive YAML value into +0.05 per malformed call — farmable above a solve's
    objective reward and invisible in the logged components."""
    env = _env(tool_error_penalty=0.5, tool_success_reward=0.05)
    traj = _traj()
    traj.info["successful_tool_calls"] = 0  # set by _reset_single in a real episode
    failed = NativeToolResult(tool_call_id="c1", name="submit_solution", content="Error", success=False)
    assert env._account_tool_result(failed, traj) == pytest.approx(-0.5)
    ok = NativeToolResult(tool_call_id="c2", name="submit_solution", content="ok", success=True)
    assert env._account_tool_result(ok, traj) == pytest.approx(0.05)


# (passed, tests_graded, tests_infra_errors, tests_ran_ok) of a 10-test pool, then the verdict.
_GRADES = {
    "solved": ((10, 10, 0, 10), "valid"),
    "a real failure": ((9, 10, 0, 10), "valid"),
    "budget stop after a failure": ((2, 3, 0, 3), "valid"),
    "budget stop before any failure": ((3, 3, 0, 3), "inconclusive"),
    "partial backend loss, nothing failed": ((7, 10, 3, 7), "inconclusive"),
    "partial backend loss beside a failure": ((6, 10, 3, 7), "valid"),
    "backend outage": ((0, 10, 10, 0), "outage"),
    "a crash beside a backend outage": ((0, 10, 9, 0), "valid"),
}


@pytest.mark.parametrize("case", list(_GRADES))
def test_only_a_grade_that_judged_the_code_pays_and_trains(case):
    """All-or-nothing, a grade short of every test scores 0 whatever the tests it never judged would
    have said; one that saw no failure says nothing about the code and leaves the group baseline, as a
    backend outage does. A failure among the judged tests is a verdict, however the grade then ended."""
    (passed, graded, infra, ran_ok), verdict = _GRADES[case]
    env = _env()
    traj = _traj(
        tool_calls=2,
        submitted=True,
        passed=passed,
        tests_graded=graded,
        tests_infra_errors=infra,
        tests_ran_ok=ran_ok,
        grading_budget_hit=graded < 10,
    )
    _reward(env, traj)
    components = traj.info[REWARD_COMPONENTS_KEY]
    metrics = env.rollout_metrics(traj)
    assert traj.episode_invalid == (verdict != "valid")
    assert components[OBJECTIVE_REWARD_KEY] == (1.0 if case == "solved" else 0.0)
    assert components["reward/submission"] == pytest.approx(SUB if verdict == "valid" else 0.0)
    assert metrics["episode/grade_inconclusive"] == (1.0 if verdict == "inconclusive" else 0.0)
    assert metrics["episode/grading_infra_outage"] == (1.0 if verdict == "outage" else 0.0)
    if verdict == "inconclusive":
        assert traj.info[EPISODE_INVALID_REASON_KEY] == (
            f"code grade inconclusive: {passed} of 10 tests passed and none failed "
            f"({10 - graded} ungraded, {infra} lost to the sandbox backend)"
        )
    elif verdict == "outage":
        assert traj.info[EPISODE_INVALID_REASON_KEY] == (
            "code grade lost to the sandbox backend: no test passed or failed"
        )
    else:
        assert EPISODE_INVALID_REASON_KEY not in traj.info


def test_an_earlier_fault_keeps_its_own_reason_through_an_outage_grade():
    """A sandbox fault a tool call booked names what voided the episode; the outage grade that follows must
    not overwrite it with its own reason."""
    env = _env()
    traj = _traj(tool_calls=2, submitted=True, passed=0, tests_graded=10, tests_infra_errors=10, tests_ran_ok=0)
    env._book_sandbox_fault(traj, env.test_tool_name, SandboxInfraError("backend down"))
    booked = f"sandbox infrastructure fault in tool {env.test_tool_name!r}: backend down"
    assert traj.info[EPISODE_INVALID_REASON_KEY] == booked
    _reward(env, traj)
    assert traj.episode_invalid and env.rollout_metrics(traj)["episode/grading_infra_outage"] == 1.0
    assert traj.info[EPISODE_INVALID_REASON_KEY] == booked


def test_a_zero_test_grade_pays_nothing():
    """A grade over no tests passed all of none: read as a solve it would pay the objective and the
    submission rung for code nothing judged."""
    env = _env()
    traj = _traj(tool_calls=2, submitted=True, passed=0, total=0)
    assert _reward(env, traj) == 0.0
    components = traj.info[REWARD_COMPONENTS_KEY]
    assert components[OBJECTIVE_REWARD_KEY] == 0.0 and components["reward/submission"] == 0.0


def test_rollout_metrics_decomposition_sums_to_reward():
    # Components must sum EXACTLY to the scalar reward: the trainer's composition-residue metric
    # flags any channel bypassing them.
    env = _env()
    traj = _traj(turns=3, tool_calls=2, submitted=True, passed=10, total=10)
    reward = _reward(env, traj)
    metrics = env.rollout_metrics(traj)
    components = traj.info[REWARD_COMPONENTS_KEY]
    assert set(components) == {
        OBJECTIVE_REWARD_KEY,
        "reward/submission",
        "reward/resubmission",
        "reward/tool_shaping",
        "reward/turn_shaping",
    }
    assert sum(components.values()) == pytest.approx(reward)
    assert components["reward/turn_shaping"] == pytest.approx(0.0)  # no per-turn deltas in this mock

    # With per-turn deltas accumulated, turn_shaping must carry their VALUE and the sum must still hold.
    shaped = _traj(turns=3, tool_calls=2, submitted=True, passed=10, total=10)
    shaped.total_reward = 0.15
    reward_shaped = _reward(env, shaped)
    shaped_components = shaped.info[REWARD_COMPONENTS_KEY]
    assert shaped_components["reward/turn_shaping"] == pytest.approx(0.15)
    assert sum(shaped_components.values()) == pytest.approx(reward_shaped)
    assert reward_shaped == pytest.approx(reward + 0.15)
    assert components[OBJECTIVE_REWARD_KEY] == pytest.approx(1.0)  # a solve at weight 1
    assert components["reward/submission"] == pytest.approx(SUB)
    assert components["reward/tool_shaping"] == pytest.approx(0.0)  # tools were called, nothing overflowed
    assert metrics[OBJECTIVE_REWARD_KEY] == pytest.approx(1.0)


def test_rollout_metrics_outcome_and_behavior():
    env = _env()
    solved = env.rollout_metrics(_traj(turns=3, tool_calls=2, submitted=True, passed=10, total=10))
    assert solved["outcome/solve_rate"] == pytest.approx(1.0)
    assert solved["outcome/test_pass_frac"] == pytest.approx(1.0)
    assert solved["episode/submission_rate"] == pytest.approx(1.0)
    assert solved["episode/tool_calls"] == pytest.approx(2.0)
    partial = env.rollout_metrics(_traj(submitted=True, passed=4, total=10))
    assert partial["outcome/solve_rate"] == pytest.approx(0.0)
    assert partial["outcome/test_pass_frac"] == pytest.approx(0.4)
    giveup = env.rollout_metrics(_traj(tool_calls=0, submitted=False))
    assert giveup["outcome/solve_rate"] == pytest.approx(0.0)
    assert giveup["outcome/test_pass_frac"] == pytest.approx(0.0)
    assert giveup["episode/submission_rate"] == pytest.approx(0.0)


def _exam_traj(*, completed=True, answer="B", response="B", tool_calls=0, truncated=False):
    t = Trajectory()
    t.add_message(Message.assistant(response))
    t.truncated = truncated
    t.info.update(
        {
            "completed": completed,
            "expected_answer": answer,
            "choices": ["A) x", "B) y"],
            "is_multiple_choice": True,
            "final_response": response,
            "total_tool_calls": tool_calls,
            "successful_tool_calls": tool_calls,
        }
    )
    return t


def test_exam_qa_applies_tool_use_shaping():
    """A YAML ``turn_overflow_penalty`` / ``no_tool_use_penalty`` on exam_qa must reach the priced
    reward: the protocol's tool shaping settles alongside the task's own grade."""
    env = ExamQAEnvironment(turn_overflow_penalty=0.2)
    assert _reward(env, _exam_traj()) == pytest.approx(1.0)
    assert _reward(env, _exam_traj(truncated=True)) == pytest.approx(1.0 - 0.2)
    assert _reward(env, _exam_traj(completed=False, truncated=True)) == pytest.approx(-0.2)


def _swe_traj(*, tool_calls: int, truncated: bool = False) -> Trajectory:
    traj = Trajectory()
    traj.add_message(Message.assistant("done"))
    traj.truncated = truncated
    traj.info.update(
        {
            "completed": True,
            "final_response": "done",
            "total_tool_calls": tool_calls,
            "successful_tool_calls": tool_calls,
            "context": {"answer": "done"},
        }
    )
    return traj


def test_swe_applies_tool_use_shaping():
    """The SWE grade settles with the protocol's tool shaping, so a turn overflow is priced there too."""
    env = SweEnvironment(turn_overflow_penalty=0.3, sandbox_backend="local")
    assert _reward(env, _swe_traj(tool_calls=1)) == pytest.approx(1.0)
    assert _reward(env, _swe_traj(tool_calls=1, truncated=True)) == pytest.approx(1.0 - 0.3)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
