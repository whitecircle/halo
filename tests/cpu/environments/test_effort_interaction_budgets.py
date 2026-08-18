#!/usr/bin/env python
"""CPU tests for effort-conditional interaction budgets and the failures-only verdict.

``reasoning_effort_profiles`` binds an effort level to a thinking budget and optional per-episode
``max_submissions``/``max_test_calls``, stamped at reset from a deterministic level (context-supplied
— the trainer stamps one per GRPO group — or a concrete env setting) and stated in the task message.
The grading verdict lists only non-passing tests, capped at ``_MAX_FAILURE_DETAILS``.

These drive ``CodeContestsEnvironment`` against a stub sandbox whose ``run`` returns a canned
:class:`SandboxResult` (no subprocesses, no network).

Run: python tests/cpu/environments/test_effort_interaction_budgets.py  (or pytest)
"""

import logging

import pytest

from src.environments.envs.tasks.coding.code_contests import _ACTIVE_TRAJECTORY, CodeContestsEnvironment
from src.environments.envs.tasks.coding.grading import _MAX_FAILURE_DETAILS, run_solution_against_tests
from src.environments.episode import bind_episode_effort
from src.environments.sandbox.base import SandboxExecutor, SandboxResult
from src.environments.tools.definitions import NativeToolCall, ToolBudgetExhausted


class _StubSandbox(SandboxExecutor):
    """Always returns the same canned result."""

    def __init__(self, result: SandboxResult):
        self._result = result

    def open_session(self):  # pragma: no cover
        raise NotImplementedError

    def run(self, code, *, stdin="", timeout=15.0, language="python", files=None):
        return self._result


_PROFILES = {
    "low": {"max_submissions": 2, "max_test_calls": 2},
    "high": {"max_submissions": 1, "max_test_calls": 6},
}


def _make_env(**kwargs):
    kwargs.setdefault("sandbox", _StubSandbox(SandboxResult(stdout="X\n", returncode=0)))
    kwargs.setdefault("reasoning_effort_profiles", _PROFILES)
    return CodeContestsEnvironment(**kwargs)


def _reset(env, context):
    ids, _ = env.reset(["solve it"], [context])
    return env.get_trajectories(ids)[0]


_TESTS = {"answer": {"tests": [{"input": "", "output": "X"}]}}


def test_verdict_lists_failures_only_and_caps_them():
    sandbox = _StubSandbox(SandboxResult(stdout="X\n", returncode=0))
    tests = [{"input": "", "output": "X"}] * 2 + [{"input": "", "output": "Y"}] * (_MAX_FAILURE_DETAILS + 3)
    grade = run_solution_against_tests("code", tests, sandbox=sandbox)
    assert grade.passed == 2
    assert ": PASS" not in grade.details
    assert grade.details.count("Test ") == _MAX_FAILURE_DETAILS
    assert "...and 3 more non-passing tests (details omitted)." in grade.details


def test_all_pass_verdict_is_summary_only():
    sandbox = _StubSandbox(SandboxResult(stdout="X\n", returncode=0))
    grade = run_solution_against_tests("code", [{"input": "", "output": "X"}] * 4, sandbox=sandbox)
    assert grade.passed == 4
    assert grade.details.strip() == "Passed 4/4 test cases."


def test_budgets_stamp_from_context_level_and_enforce_submission_cap():
    env = _make_env()
    traj = _reset(env, {"reasoning_effort": "high", **_TESTS})
    assert traj.info["episode_max_submissions"] == 1
    assert traj.info["episode_max_test_calls"] == 6
    assert env.thinking_budget_for_effort("high") == 16384  # interaction-only override keeps default tokens
    user = next(m for m in reversed(traj.messages) if m.role == "user")
    assert "Budgets for this task: 1 graded submission" in user.content
    assert "6 scratchpad runs" in user.content

    token = _ACTIVE_TRAJECTORY.set(traj)
    try:
        first = env._submit("print('X')")
        with pytest.raises(ToolBudgetExhausted, match=r"Submission limit reached \(1\)"):
            env._submit("print('X')")
    finally:
        _ACTIVE_TRAJECTORY.reset(token)
    assert "Passed 1/1" in first


def test_scratchpad_cap_reads_episode_budget():
    env = _make_env()
    traj = _reset(env, {"reasoning_effort": "low", **_TESTS})
    assert traj.info["episode_max_test_calls"] == 2
    token = _ACTIVE_TRAJECTORY.set(traj)
    try:
        env._run_test("print(1)")
        env._run_test("print(1)")
        with pytest.raises(ToolBudgetExhausted, match=r"Test limit reached \(2\)"):
            env._run_test("print(1)")
    finally:
        _ACTIVE_TRAJECTORY.reset(token)


def test_over_cap_call_classifies_as_tool_error_not_paid_success():
    # A refused call must charge tool_error_penalty, never pay the model for an exhausted budget.
    env = _make_env(tool_success_reward=0.02, tool_error_penalty=0.05)
    traj = _reset(env, {"reasoning_effort": "low", **_TESTS})

    def call(i: int) -> NativeToolCall:
        return NativeToolCall(id=f"c{i}", name="python_repl", arguments={"code": "print(1)"})

    results, reward = env._execute_tool_calls([call(0), call(1)], traj)
    assert all(r.success for r in results)
    assert reward == pytest.approx(2 * 0.02)
    results, reward = env._execute_tool_calls([call(2)], traj)
    assert results[0].success is False
    assert "Test limit reached (2)" in results[0].content
    assert reward == pytest.approx(-0.05)


def test_a_refused_over_cap_call_logs_no_traceback(caplog):
    """An exhausted budget is expected control flow (2 submissions / 5 test calls in a 15-turn
    episode), so it must not emit the WARNING+traceback that marks a tool which actually broke."""
    env = _make_env()
    traj = _reset(env, {"reasoning_effort": "low", **_TESTS})

    def call(i: int) -> NativeToolCall:
        return NativeToolCall(id=f"c{i}", name="python_repl", arguments={"code": "print(1)"})

    with caplog.at_level(logging.DEBUG, logger="src.environments.envs.protocols.native"):
        env._execute_tool_calls([call(0), call(1), call(2)], traj)

    assert [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING] == []
    assert any("refused the call" in r.getMessage() for r in caplog.records)


def test_undetermined_level_stamps_class_caps_and_states_them():
    # Tool descriptions defer to the task message here, so an undetermined level still owes a contract.
    env = _make_env(reasoning_effort="random")
    traj = _reset(env, dict(_TESTS))
    assert traj.info["episode_max_submissions"] == 2
    assert traj.info["episode_max_test_calls"] == 5
    user = next(m for m in reversed(traj.messages) if m.role == "user")
    assert "Budgets for this task: 2 graded submissions" in user.content
    assert "5 scratchpad runs" in user.content


def test_level_without_interaction_keys_states_class_caps():
    env = _make_env(reasoning_effort="medium")
    traj = _reset(env, dict(_TESTS))
    assert traj.info["episode_max_submissions"] == 2
    assert traj.info["episode_max_test_calls"] == 5
    user = next(m for m in reversed(traj.messages) if m.role == "user")
    assert "Budgets for this task: 2 graded submissions" in user.content


def test_thinking_only_profiles_do_not_defer_descriptions():
    env = _make_env(reasoning_effort_profiles={"high": {"thinking_tokens": 24000}})
    assert env.thinking_budget_for_effort("high") == 24000
    assert env.thinking_budget_for_effort("low") == 4096
    assert "You get up to 2 graded submissions" in env.registry.get("submit_solution").description
    traj = _reset(env, {"reasoning_effort": "high", **_TESTS})
    assert "episode_max_submissions" not in traj.info
    user = next(m for m in reversed(traj.messages) if m.role == "user")
    assert "Budgets for this task" not in user.content


def test_reset_effort_level_contract():
    env = _make_env(reasoning_effort="medium")
    assert env.reset_effort_level({"reasoning_effort": "high"}) == "high"
    assert env.reset_effort_level(None) == "medium"
    assert env.reset_effort_level({"reasoning_effort": "random"}) is None
    assert _make_env(reasoning_effort="random").reset_effort_level(None) is None


def test_context_level_overrides_env_level():
    env = _make_env(reasoning_effort="low")
    traj = _reset(env, {"reasoning_effort": "high", **_TESTS})
    assert traj.info["episode_max_submissions"] == 1


def test_concrete_env_level_applies_without_context():
    env = _make_env(reasoning_effort="low")
    traj = _reset(env, dict(_TESTS))
    assert traj.info["episode_max_submissions"] == 2
    assert traj.info["episode_max_test_calls"] == 2


def test_tool_descriptions_defer_when_profiles_bind_interaction():
    env = _make_env()
    submit = env.registry.get("submit_solution")
    assert "stated in the task message" in submit.description
    plain = CodeContestsEnvironment(
        sandbox=_StubSandbox(SandboxResult(stdout="X\n", returncode=0)),
        reasoning_effort_profiles=None,
    )
    assert "You get up to 2 graded submissions" in plain.registry.get("submit_solution").description


def test_tested_submission_bonus_pays_only_on_test_then_submit():
    profiles = {"high": {"max_submissions": 1, "max_test_calls": 6, "tested_submission_reward": 0.1}}
    env = _make_env(reasoning_effort_profiles=profiles)

    def run_episode(test_first: bool) -> tuple[float, dict]:
        traj = _reset(env, {"reasoning_effort": "high", **_TESTS})
        token = _ACTIVE_TRAJECTORY.set(traj)
        try:
            if test_first:
                env._run_test("print('X')")
            env._submit("print('X')")
            if not test_first:
                env._run_test("print('X')")
        finally:
            _ACTIVE_TRAJECTORY.reset(token)
        return env._compute_reward(traj), traj.info["reward_components"]

    tested_reward, tested_parts = run_episode(test_first=True)
    oneshot_reward, oneshot_parts = run_episode(test_first=False)
    assert tested_parts["reward/tested_submission"] == 0.1
    assert oneshot_parts["reward/tested_submission"] == 0.0  # a test AFTER the submission pays nothing
    assert tested_reward - oneshot_reward == pytest.approx(0.1)
    assert tested_reward == pytest.approx(sum(tested_parts.values()))  # composition residue stays 0


def test_bonus_only_profile_still_binds_and_scales_by_effort():
    profiles = {
        "medium": {"tested_submission_reward": 0.05},
        "high": {"tested_submission_reward": 0.1},
    }
    env = _make_env(reasoning_effort_profiles=profiles)
    assert env._profiles_bind_interaction
    med = _reset(env, {"reasoning_effort": "medium", **_TESTS})
    low = _reset(env, {"reasoning_effort": "low", **_TESTS})
    assert med.info["episode_tested_submission_reward"] == 0.05
    assert med.info["episode_max_submissions"] == 2  # class caps stated when the profile sets no caps
    assert low.info["episode_tested_submission_reward"] == 0.0


def test_token_cost_stamps_from_profile():
    profiles = {"low": {"token_cost": 0.05}, "high": {"max_test_calls": 6}}
    env = _make_env(reasoning_effort_profiles=profiles)
    low = _reset(env, {"reasoning_effort": "low", **_TESTS})
    high = _reset(env, {"reasoning_effort": "high", **_TESTS})
    assert low.info["episode_token_cost"] == 0.05
    assert high.info["episode_token_cost"] == 0.0  # no price set -> free


def test_effort_binding_caps_both_channels():
    # The budget must bound the WHOLE turn: an unbounded visible channel displaces the tool call.
    # Both budgets are stated here so the arithmetic below does not ride on the class defaults.
    env = _make_env(reasoning_effort_profiles={"low": {"thinking_tokens": 4096}, "high": {"thinking_tokens": 30000}})
    bound = bind_episode_effort({"reasoning_effort": "low"}, env, max_tokens=20000, max_thinking_tokens=18000)
    assert bound.thinking_budget == 4096
    assert bound.max_tokens == 4096 + 2000
    # Over-budget profiles clamp on BOTH channels, else they silently raise the run's CoT cap.
    over = bind_episode_effort({"reasoning_effort": "high"}, env, max_tokens=20000, max_thinking_tokens=18000)
    assert (over.thinking_budget, over.max_tokens) == (18000, 20000)
    no_global = bind_episode_effort({"reasoning_effort": "low"}, env, max_tokens=1024)
    assert (no_global.thinking_budget, no_global.max_tokens) == (4096, 1024)


def test_invalid_profiles_raise():
    with pytest.raises(ValueError, match="level must be one of"):
        _make_env(reasoning_effort_profiles={"extreme": {"max_submissions": 1}})
    with pytest.raises(ValueError, match="unknown keys"):
        _make_env(reasoning_effort_profiles={"low": {"max_turns": 3}})
    with pytest.raises(ValueError, match="must be >= 1"):
        _make_env(reasoning_effort_profiles={"low": {"max_submissions": 0}})
    with pytest.raises(ValueError, match="thinking_tokens.*must be >= 1"):
        _make_env(reasoning_effort_profiles={"low": {"thinking_tokens": 0}})
    with pytest.raises(ValueError, match="must be >= 0"):
        _make_env(reasoning_effort_profiles={"low": {"max_test_calls": -1}})
    with pytest.raises(ValueError, match="tested_submission_reward.*must be >= 0"):
        _make_env(reasoning_effort_profiles={"low": {"tested_submission_reward": -0.1}})
    with pytest.raises(ValueError, match="token_cost.*must be >= 0"):
        _make_env(reasoning_effort_profiles={"low": {"token_cost": -0.01}})


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
