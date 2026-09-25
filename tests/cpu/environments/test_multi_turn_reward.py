"""CPU tests for the CodeContestsEnvironment agentic-loop reward ladder: the grade and shaping
``_grade_episode`` returns, priced into ``reward_components`` by the base's ``_settle_grade``.

The base model answers in plain text and rarely calls submit_solution, so the verifier alone gives no
gradient to learn the loop. Small shaping rungs bootstrap it and self-neutralize per GRPO group:

  plain-text giveup (0 tool calls)  -> -no_tool_use_penalty          (reward/tool_shaping)
  tool call, no submission          ->  0
  submitted (any test result)       -> +submission_reward            (reward/submission)
  submitted + tested/iterated       -> +submission_reward + multi_turn_reward
  fraction of tests passed          -> weight * frac ** exponent     (reward/objective, dominant;
                                       the environment term, default weight 1 / exponent 1)

    python tests/cpu/environments/test_multi_turn_reward.py
"""

import sys

import pytest

from src.environments.base import (
    EPISODE_ERROR_KEY,
    OBJECTIVE_REWARD_KEY,
    REWARD_COMPONENTS_KEY,
    Message,
    Trajectory,
)
from src.environments.envs.tasks.coding.code_contests import CodeContestsEnvironment
from src.environments.envs.tasks.coding.swe import SweEnvironment
from src.environments.envs.tasks.qa import ExamQAEnvironment
from src.environments.tools.definitions import NativeToolResult

SUB, PEN, MTR = 0.1, 0.1, 0.05


def _env(**kw):
    kw.setdefault("submission_reward", SUB)
    kw.setdefault("no_tool_use_penalty", PEN)
    kw.setdefault("multi_turn_reward", MTR)
    return CodeContestsEnvironment(language="python", sandbox_backend="local", **kw)


def _traj(*, turns=1, tool_calls=0, submitted=False, passed=0, total=10):
    t = Trajectory()
    for _ in range(turns):
        t.add_message(Message.assistant("..."))
    t.info["total_tool_calls"] = tool_calls
    t.info["tool_call_counts"] = {"submit_solution": 1} if submitted else {}
    if submitted:
        t.info["submission_result"] = "ok"
        t.info["tests_passed"] = passed
        t.info["tests_total"] = total
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
    submit_tested = _reward(env, _traj(turns=3, tool_calls=2, submitted=True, passed=0))
    submit_partial = _reward(env, _traj(turns=3, tool_calls=2, submitted=True, passed=3))
    solved = _reward(env, _traj(turns=3, tool_calls=2, submitted=True, passed=10))
    assert giveup == pytest.approx(-PEN)
    assert tooled_no_submit == pytest.approx(0.0)
    assert submit_zero == pytest.approx(SUB)
    assert submit_tested == pytest.approx(SUB + MTR)
    assert submit_partial == pytest.approx(0.3 + SUB + MTR)
    assert solved == pytest.approx(1.0 + SUB + MTR)
    assert giveup < tooled_no_submit < submit_zero < submit_tested < submit_partial < solved


def test_solving_dominates_shaping():
    env = _env()
    solved = _reward(env, _traj(tool_calls=2, submitted=True, passed=10))
    best_non_solve = _reward(env, _traj(turns=3, tool_calls=2, submitted=True, passed=9))
    assert solved - best_non_solve > 0.05  # the last 10% of tests is worth more than all shaping


def test_all_shaping_off_by_default():
    env = CodeContestsEnvironment(language="python", sandbox_backend="local")
    assert (env.submission_reward, env.no_tool_use_penalty, env.multi_turn_reward) == (0.0, 0.0, 0.0)
    # Shaping off: a giveup grades 0, not penalized.
    assert _reward(env, _traj(tool_calls=0, submitted=False)) == pytest.approx(0.0)
    assert _reward(env, _traj(tool_calls=2, submitted=True, passed=5)) == pytest.approx(0.5)


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
    assert _reward(env, capped) == pytest.approx(-0.2)
    capped_sub = _traj(turns=6, tool_calls=5, submitted=True, passed=10)
    capped_sub.truncated = True
    assert _reward(env, capped_sub) == pytest.approx(1.0 + SUB + MTR - 0.2)
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
    silently turned a positive YAML value into +0.05 per malformed call — farmable above a solve's
    objective reward and invisible in the logged components."""
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
    reward = _reward(env, traj)
    components = traj.info[REWARD_COMPONENTS_KEY]
    assert components["reward/submission"] == pytest.approx(0.0)
    assert components["reward/execution"] == pytest.approx(0.0)
    assert components[OBJECTIVE_REWARD_KEY] == pytest.approx(0.0)
    assert reward == pytest.approx(MTR)  # only the generic engagement shaping remains


def test_grading_backend_outage_pays_no_rungs():
    # A TOTAL outage carries no signal (no rungs, a 0 grade); a PARTIAL one is still graded content.
    env = _env()
    outage = _traj(tool_calls=2, submitted=True, passed=0, total=10)
    outage.info["tests_infra_errors"] = 10
    _reward(env, outage)
    assert outage.info[REWARD_COMPONENTS_KEY]["reward/submission"] == pytest.approx(0.0)
    assert outage.info[REWARD_COMPONENTS_KEY][OBJECTIVE_REWARD_KEY] == pytest.approx(0.0)
    assert env.rollout_metrics(outage)["episode/grading_infra_outage"] == pytest.approx(1.0)

    partial = _traj(tool_calls=2, submitted=True, passed=3, total=10)
    partial.info["tests_infra_errors"] = 5
    _reward(env, partial)
    assert partial.info[REWARD_COMPONENTS_KEY]["reward/submission"] == pytest.approx(SUB)
    assert partial.info[REWARD_COMPONENTS_KEY][OBJECTIVE_REWARD_KEY] == pytest.approx(0.3)
    assert env.rollout_metrics(partial)["episode/grading_infra_outage"] == pytest.approx(0.0)


def test_rollout_metrics_decomposition_sums_to_reward():
    # Components must sum EXACTLY to the scalar reward: the trainer's composition-residue metric
    # flags any channel bypassing them.
    env = _env()
    traj = _traj(turns=3, tool_calls=2, submitted=True, passed=3, total=10)
    reward = _reward(env, traj)
    metrics = env.rollout_metrics(traj)
    components = traj.info[REWARD_COMPONENTS_KEY]
    assert set(components) == {
        OBJECTIVE_REWARD_KEY,
        "reward/submission",
        "reward/execution",
        "reward/tested_submission",
        "reward/resubmission",
        "reward/tool_shaping",
        "reward/turn_shaping",
    }
    assert sum(components.values()) == pytest.approx(reward)
    assert components["reward/turn_shaping"] == pytest.approx(0.0)  # no per-turn deltas in this mock

    # With per-turn deltas accumulated, turn_shaping must carry their VALUE and the sum must still hold.
    shaped = _traj(turns=3, tool_calls=2, submitted=True, passed=3, total=10)
    shaped.total_reward = 0.15
    reward_shaped = _reward(env, shaped)
    shaped_components = shaped.info[REWARD_COMPONENTS_KEY]
    assert shaped_components["reward/turn_shaping"] == pytest.approx(0.15)
    assert sum(shaped_components.values()) == pytest.approx(reward_shaped)
    assert reward_shaped == pytest.approx(reward + 0.15)
    assert components[OBJECTIVE_REWARD_KEY] == pytest.approx(0.3)  # 3/10 tests at weight 1, exponent 1
    assert components["reward/submission"] == pytest.approx(SUB)
    assert components["reward/execution"] == pytest.approx(0.0)  # rung off by default, no tests_ran_ok set
    assert components["reward/tool_shaping"] == pytest.approx(MTR)  # >1 tool call + submitted
    assert metrics[OBJECTIVE_REWARD_KEY] == pytest.approx(0.3)


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
    sys.exit(pytest.main([__file__, "-v"]))
