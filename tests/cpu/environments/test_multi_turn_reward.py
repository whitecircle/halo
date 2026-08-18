"""CPU tests for the CodeContestsEnvironment agentic-loop reward ladder (_compute_reward).

The base model answers in plain text and rarely calls submit_solution, so the verifier alone gives no
gradient to learn the loop. Small shaping rungs bootstrap it and self-neutralize per GRPO group:

  plain-text giveup (0 tool calls)  -> -no_tool_use_penalty
  tool call, no submission          ->  0
  submitted (any test result)       -> +submission_reward
  submitted + tested/iterated       -> +submission_reward + multi_turn_reward
  fraction of tests passed          -> frac * success_reward  (objective, dominant)

    python tests/cpu/environments/test_multi_turn_reward.py
"""

import sys

import pytest

from src.environments.base import Message, Trajectory
from src.environments.envs.tasks.coding.code_contests import CodeContestsEnvironment

SUB, PEN, MTR = 0.1, 0.1, 0.05


def _env(**kw):
    kw.setdefault("submission_reward", SUB)
    kw.setdefault("no_tool_use_penalty", PEN)
    kw.setdefault("multi_turn_reward", MTR)
    return CodeContestsEnvironment(language="python", success_reward=1.0, sandbox_backend="local", **kw)


def _traj(*, turns=1, tool_calls=0, submitted=False, passed=0, total=10):
    t = Trajectory()
    for _ in range(turns):
        t.add_message(Message.assistant("..."))
    t.info["total_tool_calls"] = tool_calls
    t.info["submission_count"] = 1 if submitted else 0
    if submitted:
        t.info["submission_result"] = "ok"
        t.info["tests_passed"] = passed
        t.info["tests_total"] = total
    return t


def test_ladder_is_monotonic():
    env = _env()
    giveup = env._compute_reward(_traj(turns=1, tool_calls=0, submitted=False))
    tooled_no_submit = env._compute_reward(_traj(turns=2, tool_calls=1, submitted=False))
    submit_zero = env._compute_reward(_traj(turns=1, tool_calls=1, submitted=True, passed=0))
    submit_tested = env._compute_reward(_traj(turns=3, tool_calls=2, submitted=True, passed=0))
    submit_partial = env._compute_reward(_traj(turns=3, tool_calls=2, submitted=True, passed=3))
    solved = env._compute_reward(_traj(turns=3, tool_calls=2, submitted=True, passed=10))
    assert giveup == pytest.approx(-PEN)
    assert tooled_no_submit == pytest.approx(0.0)
    assert submit_zero == pytest.approx(SUB)
    assert submit_tested == pytest.approx(SUB + MTR)
    assert submit_partial == pytest.approx(0.3 + SUB + MTR)
    assert solved == pytest.approx(1.0 + SUB + MTR)
    assert giveup < tooled_no_submit < submit_zero < submit_tested < submit_partial < solved


def test_solving_dominates_shaping():
    env = _env()
    solved = env._compute_reward(_traj(tool_calls=2, submitted=True, passed=10))
    best_non_solve = env._compute_reward(_traj(turns=3, tool_calls=2, submitted=True, passed=9))
    assert solved - best_non_solve > 0.05  # the last 10% of tests is worth more than all shaping


def test_all_shaping_off_by_default():
    env = CodeContestsEnvironment(language="python", success_reward=1.0, sandbox_backend="local")
    assert (env.submission_reward, env.no_tool_use_penalty, env.multi_turn_reward) == (0.0, 0.0, 0.0)
    # Shaping off: a giveup is plain failure_reward, not penalized.
    assert env._compute_reward(_traj(tool_calls=0, submitted=False)) == pytest.approx(0.0)
    assert env._compute_reward(_traj(tool_calls=2, submitted=True, passed=5)) == pytest.approx(0.5)


def test_negative_magnitudes_rejected():
    for bad in (
        "submission_reward",
        "no_tool_use_penalty",
        "multi_turn_reward",
        "tool_error_penalty",
        "tool_success_reward",
        "turn_overflow_penalty",
    ):
        with pytest.raises(ValueError, match=bad):
            CodeContestsEnvironment(language="python", sandbox_backend="local", **{bad: -0.1})


def test_turn_overflow_penalty_on_truncation():
    """An episode that burns ``max_turns`` without terminating (``trajectory.truncated``) pays
    ``turn_overflow_penalty`` on top of whatever the ladder gave it — pricing the turn-cap overflow the
    ladder otherwise leaves free. Applies to submitted and unsubmitted episodes alike; default 0 = off."""
    env = _env(turn_overflow_penalty=0.2)
    capped = _traj(turns=6, tool_calls=5, submitted=False)
    capped.truncated = True
    assert env._compute_reward(capped) == pytest.approx(-0.2)
    capped_sub = _traj(turns=6, tool_calls=5, submitted=True, passed=10)
    capped_sub.truncated = True
    assert env._compute_reward(capped_sub) == pytest.approx(1.0 + SUB + MTR - 0.2)
    assert env._compute_reward(_traj(turns=6, tool_calls=5, submitted=False)) == pytest.approx(0.0)
    # Default 0: truncation alone changes nothing.
    off = _traj(turns=6, tool_calls=5, submitted=False)
    off.truncated = True
    assert _env()._compute_reward(off) == pytest.approx(0.0)


def test_tool_error_penalty_is_applied_negative():
    """The per-call error knob is a MAGNITUDE: a failed tool call must REDUCE the reward. A signed knob
    silently turned a positive YAML value into +0.05 per malformed call — farmable above a solve's
    objective reward and invisible in the logged components."""
    from src.environments.tools.definitions import NativeToolResult

    env = _env(tool_error_penalty=0.5, tool_success_reward=0.05)
    traj = _traj()
    traj.info["successful_tool_calls"] = 0  # set by _reset_single in a real episode
    failed = NativeToolResult(tool_call_id="c1", name="submit_solution", content="Error", success=False)
    assert env._account_tool_result(failed, traj) == pytest.approx(-0.5)
    ok = NativeToolResult(tool_call_id="c2", name="submit_solution", content="ok", success=True)
    assert env._account_tool_result(ok, traj) == pytest.approx(0.05)


def test_zero_test_rows_pay_no_rungs():
    # A zero-test row says nothing about the code; paying its submission rung makes it a payout attractor.
    env = _env()
    traj = _traj(tool_calls=2, submitted=True, passed=0, total=0)
    reward = env._compute_reward(traj)
    components = traj.info["reward_components"]
    assert components["reward/submission"] == pytest.approx(0.0)
    assert components["reward/execution"] == pytest.approx(0.0)
    assert components["reward/objective"] == pytest.approx(env.failure_reward)
    assert reward == pytest.approx(env.failure_reward + MTR)  # only the generic engagement shaping remains


def test_grading_backend_outage_pays_no_rungs():
    # A TOTAL outage carries no signal (no rungs, failure_reward); a PARTIAL one is still graded content.
    env = _env()
    outage = _traj(tool_calls=2, submitted=True, passed=0, total=10)
    outage.info["tests_infra_errors"] = 10
    env._compute_reward(outage)
    assert outage.info["reward_components"]["reward/submission"] == pytest.approx(0.0)
    assert outage.info["reward_components"]["reward/objective"] == pytest.approx(env.failure_reward)
    assert env.rollout_metrics(outage)["episode/grading_infra_outage"] == pytest.approx(1.0)

    partial = _traj(tool_calls=2, submitted=True, passed=3, total=10)
    partial.info["tests_infra_errors"] = 5
    env._compute_reward(partial)
    assert partial.info["reward_components"]["reward/submission"] == pytest.approx(SUB)
    assert partial.info["reward_components"]["reward/objective"] == pytest.approx(0.3)
    assert env.rollout_metrics(partial)["episode/grading_infra_outage"] == pytest.approx(0.0)


def test_rollout_metrics_decomposition_sums_to_reward():
    # Components must sum EXACTLY to the scalar reward: the trainer's composition-residue metric
    # flags any channel bypassing them.
    env = _env()
    traj = _traj(turns=3, tool_calls=2, submitted=True, passed=3, total=10)
    reward = env._compute_reward(traj)
    metrics = env.rollout_metrics(traj)
    components = traj.info["reward_components"]
    assert set(components) == {
        "reward/objective",
        "reward/submission",
        "reward/execution",
        "reward/tested_submission",
        "reward/tool_shaping",
        "reward/turn_shaping",
    }
    assert sum(components.values()) == pytest.approx(reward)
    assert components["reward/turn_shaping"] == pytest.approx(0.0)  # no per-turn deltas in this mock

    # With per-turn deltas accumulated, turn_shaping must carry their VALUE and the sum must still hold.
    shaped = _traj(turns=3, tool_calls=2, submitted=True, passed=3, total=10)
    shaped.total_reward = 0.15
    reward_shaped = env._compute_reward(shaped)
    shaped_components = shaped.info["reward_components"]
    assert shaped_components["reward/turn_shaping"] == pytest.approx(0.15)
    assert sum(shaped_components.values()) == pytest.approx(reward_shaped)
    assert reward_shaped == pytest.approx(reward + 0.15)
    assert components["reward/objective"] == pytest.approx(0.3)  # 3/10 tests * success_reward
    assert components["reward/submission"] == pytest.approx(SUB)
    assert components["reward/execution"] == pytest.approx(0.0)  # rung off by default, no tests_ran_ok set
    assert components["reward/tool_shaping"] == pytest.approx(MTR)  # >1 tool call + submitted
    assert metrics["reward/objective"] == pytest.approx(0.3)


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
    """ExamQAEnvironment's _compute_reward override must include the parent's shaping term — a YAML
    turn_overflow_penalty / no_tool_use_penalty on exam_qa silently no-ops otherwise."""
    from src.environments.envs.tasks.qa import ExamQAEnvironment

    env = ExamQAEnvironment(turn_overflow_penalty=0.2, success_reward=1.0, failure_reward=0.0)
    assert env._compute_reward(_exam_traj()) == pytest.approx(1.0)
    assert env._compute_reward(_exam_traj(truncated=True)) == pytest.approx(1.0 - 0.2)
    assert env._compute_reward(_exam_traj(completed=False, truncated=True)) == pytest.approx(-0.2)


def test_swe_applies_tool_use_shaping():
    """SweEnvironment's _compute_reward override must include the parent's shaping term."""
    from src.environments.envs.tasks.coding.swe import SweEnvironment

    env = SweEnvironment(turn_overflow_penalty=0.3, success_reward=1.0, failure_reward=0.0, sandbox_backend="local")
    traj = Trajectory()
    traj.add_message(Message.assistant("done"))
    traj.info.update({"completed": True, "total_tool_calls": 1, "successful_tool_calls": 1, "context": {}})
    assert env._compute_reward(traj) == pytest.approx(1.0)
    traj.truncated = True
    assert env._compute_reward(traj) == pytest.approx(1.0 - 0.3)


def test_swe_zero_tool_call_completion_scores_full_failure_reward():
    """In the ungraded default path, completing without a single tool call is the worst behavior
    (a pure no-interaction answer) and must score the FULL failure_reward — softening it below a
    genuine failed attempt would reward the exploit."""
    from src.environments.envs.tasks.coding.swe import SweEnvironment

    env = SweEnvironment(success_reward=1.0, failure_reward=-1.0, sandbox_backend="local")
    traj = Trajectory()
    traj.add_message(Message.assistant("here is my answer"))
    traj.info.update({"completed": True, "total_tool_calls": 0, "successful_tool_calls": 0, "context": {}})
    assert env._compute_reward(traj) == pytest.approx(-1.0)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
