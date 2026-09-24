#!/usr/bin/env python
"""CPU tests for effort-conditional interaction budgets and the failures-only verdict.

``reasoning_effort_profiles`` binds an effort level to a thinking budget and optional per-episode
``max_submissions``/``max_test_calls``, stamped at reset from a deterministic level (context-supplied
— the trainer stamps one per GRPO group — or a concrete env setting) as the per-tool caps the protocol
enforces, and stated in the task message. The grading verdict lists only non-passing tests, one entry
per distinct verdict, capped at ``_MAX_FAILURE_DETAILS``.

These drive ``CodeContestsEnvironment`` against a stub sandbox whose ``run`` returns a canned
result (no subprocesses, no network), through the protocol's tool dispatch.

Run: python tests/cpu/environments/test_effort_interaction_budgets.py  (or pytest)
"""

import logging

import pytest

from src.environments.base import (
    EPISODE_TOOL_BUDGETS_KEY,
    REWARD_COMPONENTS_KEY,
    THINKING_BUDGET_EXHAUSTED_KEY,
    TOOL_CALL_COUNTS_KEY,
)
from src.environments.envs.tasks.coding.code_contests import CodeContestsEnvironment
from src.environments.envs.tasks.coding.grading import (
    _MAX_FAILURE_DETAILS,
    VERDICT_DETAIL_FULL,
    run_solution_against_tests,
)
from src.environments.episode import (
    TurnGeneration,
    bind_episode_effort,
    reasoning_tokens_of,
    resolve_reasoning_end_token_id,
)
from src.environments.tools.definitions import NativeToolCall
from tests.common.code_contests import SINGLE_TEST_ANSWER, StubSandbox, call_tool, reset_episode

_PROFILES = {
    "low": {"max_submissions": 2, "max_test_calls": 2},
    "high": {"max_submissions": 1, "max_test_calls": 6},
}


def _make_env(**kwargs):
    kwargs.setdefault("sandbox", StubSandbox())
    kwargs.setdefault("reasoning_effort_profiles", _PROFILES)
    return CodeContestsEnvironment(**kwargs)


def test_verdict_lists_failures_only_and_caps_them():
    # Distinct expected outputs make every failure its own verdict under ``full``; identical ones, or
    # the ``outcome`` default, fold into one entry.
    sandbox = StubSandbox()
    tests = [{"input": "", "output": "X"}] * 2 + [
        {"input": "", "output": f"Y{k}"} for k in range(_MAX_FAILURE_DETAILS + 3)
    ]
    grade = run_solution_against_tests("code", tests, sandbox=sandbox, verdict_detail=VERDICT_DETAIL_FULL)
    assert grade.passed == 2
    assert ": PASS" not in grade.details
    assert grade.details.count("Test ") == _MAX_FAILURE_DETAILS
    assert "...and 3 more non-passing tests (details omitted)." in grade.details


def test_all_pass_verdict_is_summary_only():
    sandbox = StubSandbox()
    grade = run_solution_against_tests("code", [{"input": "", "output": "X"}] * 4, sandbox=sandbox)
    assert grade.passed == 4
    assert grade.details.strip() == "Passed 4/4 test cases."


def test_budgets_stamp_from_context_level_and_enforce_submission_cap():
    env = _make_env()
    traj = reset_episode(env, {"reasoning_effort": "high", **SINGLE_TEST_ANSWER})
    assert traj.info[EPISODE_TOOL_BUDGETS_KEY] == {"python_repl": 6, "submit_solution": 1}
    assert env.thinking_budget_for_effort("high") == 16384  # interaction-only override keeps default tokens
    user = next(m for m in reversed(traj.messages) if m.role == "user")
    assert "Budgets for this task: 1 graded submission" in user.content
    assert "6 scratchpad runs" in user.content

    first = call_tool(env, traj, "submit_solution")
    second = call_tool(env, traj, "submit_solution")
    assert "Passed 1/1" in first
    assert "Submission limit reached (1); this submission is not graded." in second
    assert traj.info[TOOL_CALL_COUNTS_KEY]["submit_solution"] == 1, "a refused call spends nothing"


def test_scratchpad_cap_reads_episode_budget():
    env = _make_env()
    traj = reset_episode(env, {"reasoning_effort": "low", **SINGLE_TEST_ANSWER})
    assert traj.info[EPISODE_TOOL_BUDGETS_KEY]["python_repl"] == 2
    call_tool(env, traj, "python_repl")
    call_tool(env, traj, "python_repl")
    third = call_tool(env, traj, "python_repl")
    assert "Test limit reached (2); the scratchpad is exhausted. Submit your solution with submit_solution." in third
    assert traj.info[TOOL_CALL_COUNTS_KEY]["python_repl"] == 2


def test_resubmission_penalty_prices_each_graded_submission_after_the_first():
    """Three graded submissions of which only the last counts make the judge a free test oracle: the
    penalty prices every probe after the first, a single submission pays nothing, and the component
    stays inside the decomposition the trainer's residue check sums."""
    profiles = {"high": {"max_submissions": 3, "max_test_calls": 6}}
    env = _make_env(reasoning_effort_profiles=profiles, resubmission_penalty=0.1)
    traj = reset_episode(env, {"reasoning_effort": "high", **SINGLE_TEST_ANSWER})
    for _ in range(3):
        call_tool(env, traj, "submit_solution")
    env._settle_grade(traj, None)
    components = traj.info[REWARD_COMPONENTS_KEY]
    assert components["reward/resubmission"] == pytest.approx(-0.2)
    assert traj.total_reward == pytest.approx(sum(components.values()))

    once = _make_env(reasoning_effort_profiles=profiles, resubmission_penalty=0.1)
    traj_once = reset_episode(once, {"reasoning_effort": "high", **SINGLE_TEST_ANSWER})
    call_tool(once, traj_once, "submit_solution")
    once._settle_grade(traj_once, None)
    assert traj_once.info[REWARD_COMPONENTS_KEY]["reward/resubmission"] == 0.0

    # A sign check alone lets NaN and infinity through, and either poisons the whole reward.
    for bad in (-0.1, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="resubmission_penalty"):
            _make_env(resubmission_penalty=bad)


def test_over_cap_call_classifies_as_tool_error_not_paid_success():
    # A refused call must charge tool_error_penalty, never pay the model for an exhausted budget.
    env = _make_env(tool_success_reward=0.02, tool_error_penalty=0.05)
    traj = reset_episode(env, {"reasoning_effort": "low", **SINGLE_TEST_ANSWER})

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
    traj = reset_episode(env, {"reasoning_effort": "low", **SINGLE_TEST_ANSWER})

    def call(i: int) -> NativeToolCall:
        return NativeToolCall(id=f"c{i}", name="python_repl", arguments={"code": "print(1)"})

    with caplog.at_level(logging.DEBUG, logger="src.environments.envs.protocols.native"):
        env._execute_tool_calls([call(0), call(1), call(2)], traj)

    assert [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING] == []
    assert any("refused the call" in r.getMessage() for r in caplog.records)


def test_undetermined_level_stamps_class_caps_and_states_them():
    # Tool descriptions defer to the task message here, so an undetermined level still owes a contract.
    env = _make_env(reasoning_effort="random")
    traj = reset_episode(env, dict(SINGLE_TEST_ANSWER))
    assert traj.info[EPISODE_TOOL_BUDGETS_KEY] == {"python_repl": 5, "submit_solution": 2}
    user = next(m for m in reversed(traj.messages) if m.role == "user")
    assert "Budgets for this task: 2 graded submissions" in user.content
    assert "5 scratchpad runs" in user.content


def test_level_without_interaction_keys_states_class_caps():
    env = _make_env(reasoning_effort="medium")
    traj = reset_episode(env, dict(SINGLE_TEST_ANSWER))
    assert traj.info[EPISODE_TOOL_BUDGETS_KEY] == {"python_repl": 5, "submit_solution": 2}
    user = next(m for m in reversed(traj.messages) if m.role == "user")
    assert "Budgets for this task: 2 graded submissions" in user.content


def test_thinking_only_profiles_do_not_defer_descriptions():
    env = _make_env(reasoning_effort_profiles={"high": {"thinking_tokens": 24000}})
    assert env.thinking_budget_for_effort("high") == 24000
    assert env.thinking_budget_for_effort("low") == 4096
    assert "You get up to 2 graded submissions" in env.registry.get("submit_solution").description
    traj = reset_episode(env, {"reasoning_effort": "high", **SINGLE_TEST_ANSWER})
    assert traj.info[EPISODE_TOOL_BUDGETS_KEY] == {"python_repl": 5, "submit_solution": 2}, "class caps still bind"
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
    traj = reset_episode(env, {"reasoning_effort": "high", **SINGLE_TEST_ANSWER})
    assert traj.info[EPISODE_TOOL_BUDGETS_KEY]["submit_solution"] == 1


def test_concrete_env_level_applies_without_context():
    env = _make_env(reasoning_effort="low")
    traj = reset_episode(env, dict(SINGLE_TEST_ANSWER))
    assert traj.info[EPISODE_TOOL_BUDGETS_KEY] == {"python_repl": 2, "submit_solution": 2}


def test_tool_descriptions_defer_when_profiles_bind_interaction():
    env = _make_env()
    submit = env.registry.get("submit_solution")
    assert "stated in the task message" in submit.description
    plain = CodeContestsEnvironment(
        sandbox=StubSandbox(),
        reasoning_effort_profiles=None,
    )
    assert "You get up to 2 graded submissions" in plain.registry.get("submit_solution").description


def test_tested_submission_bonus_pays_only_on_test_then_submit():
    profiles = {"high": {"max_submissions": 1, "max_test_calls": 6, "tested_submission_reward": 0.1}}
    env = _make_env(reasoning_effort_profiles=profiles)

    def run_episode(test_first: bool) -> tuple[float, dict]:
        traj = reset_episode(env, {"reasoning_effort": "high", **SINGLE_TEST_ANSWER})
        if test_first:
            call_tool(env, traj, "python_repl")
        call_tool(env, traj, "submit_solution")
        if not test_first:
            call_tool(env, traj, "python_repl")
        env._settle_grade(traj, None)
        return traj.total_reward, traj.info[REWARD_COMPONENTS_KEY]

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
    med = reset_episode(env, {"reasoning_effort": "medium", **SINGLE_TEST_ANSWER})
    low = reset_episode(env, {"reasoning_effort": "low", **SINGLE_TEST_ANSWER})
    assert med.info["episode_tested_submission_reward"] == 0.05
    assert (
        med.info[EPISODE_TOOL_BUDGETS_KEY]["submit_solution"] == 2
    )  # class caps stated when the profile sets no caps
    assert "episode_tested_submission_reward" not in low.info  # no bonus at that level


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


_SCOPED_PROFILES = {"low": {"thinking_tokens": 4096}, "high": {"thinking_tokens": 30000}}
# The reasoning-end marker's id: one no reasoning id below carries.
_END = 151668


def _bind_scoped(level, scope, max_thinking_tokens=18000):
    env = _make_env(reasoning_effort_profiles=_SCOPED_PROFILES)
    return bind_episode_effort(
        {"reasoning_effort": level},
        env,
        max_tokens=30000,
        max_thinking_tokens=max_thinking_tokens,
        scope=scope,
        turn_reserve=512,
    )


def test_episode_scope_binds_the_level_budget_as_the_episode_total():
    """Under the episode scope the level's budget is the whole task's, so the run's per-turn ceiling must
    not clamp it (that clamp belongs to the per-turn scope) — it bounds only what one turn may take of it,
    so the first turn's cap, and with it the turn's total, still respect the ceiling."""
    high = _bind_scoped("high", "episode")
    assert (high.thinking_budget, high.max_tokens) == (30000, 30000)
    # The engine cap per turn: what the budget has left, never above the ceiling, never below the reserve.
    assert high.turn_thinking_cap(0) == 18000
    assert high.turn_thinking_cap(17000) == 13000
    assert high.turn_thinking_cap(29600) == 512
    assert high.turn_thinking_cap(40000) == 512
    assert high.budget_exhausted(29600) is True
    assert high.budget_exhausted(1000) is False
    low = _bind_scoped("low", "episode")
    assert (low.thinking_budget, low.max_tokens) == (4096, 4096 + 12000)
    assert [low.turn_thinking_cap(spent) for spent in (0, 3000, 4000)] == [4096, 1096, 512]


def test_turn_scope_caps_every_turn_alike_and_never_exhausts():
    turn = _bind_scoped("high", "turn")
    assert turn.thinking_budget == 18000, "the per-turn scope clamps the level's budget to the ceiling"
    assert {turn.turn_thinking_cap(spent) for spent in (0, 17000, 40000)} == {18000}
    assert turn.budget_exhausted(40000) is False


def test_episode_scope_without_a_ceiling_hands_a_turn_the_whole_remainder():
    high = _bind_scoped("high", "episode", max_thinking_tokens=None)
    assert high.thinking_budget == 30000
    assert high.turn_thinking_cap(1000) == 30000 - 1000


def _gen(token_ids):
    return TurnGeneration(text="", tool_calls=[], reasoning="", tokens=5, token_ids=token_ids)


def test_reasoning_tokens_of_counts_the_sampled_ids_through_the_marker():
    """The engine's budget counts the close it forces, so the marker is the last reasoning token; a turn
    cut before it closed spent every sampled id on reasoning. Without the ids, or without the marker's
    id, there is nothing exact to count and the scope must say so rather than count zero."""
    assert reasoning_tokens_of(_gen([11, 12, 13, _END, 14]), _END) == 4
    assert reasoning_tokens_of(_gen([11, 12, 13, 14, 15]), _END) == 5
    with pytest.raises(ValueError, match="sampled token ids"):
        reasoning_tokens_of(_gen(None), _END)
    with pytest.raises(ValueError, match="reasoning_end_token_id"):
        reasoning_tokens_of(_gen([11, _END]), None)


def test_spend_of_counts_only_under_the_episode_scope():
    """The drivers charge every turn through ``spend_of`` without branching on the scope, so under the
    per-turn scope it must charge nothing — with or without ids, since that scope never asked for them —
    and under the episode scope it is the marker count."""
    per_turn = _bind_scoped("high", "turn")
    assert per_turn.spend_of(_gen([11, 12, 13, _END, 14]), _END) == 0
    assert per_turn.spend_of(_gen(None), None) == 0
    shared = _bind_scoped("high", "episode")
    assert shared.spend_of(_gen([11, 12, 13, _END, 14]), _END) == 4
    assert shared.spend_of(_gen([11, 12, 13, 14, 15]), _END) == 5
    with pytest.raises(ValueError, match="sampled token ids"):
        shared.spend_of(_gen(None), _END)


class _Tokenizer:
    """The two attributes the resolver reads, over a fixed vocabulary."""

    def __init__(self, vocab, unk_token_id=0):
        self._vocab = vocab
        self.unk_token_id = unk_token_id

    def convert_tokens_to_ids(self, token):
        return self._vocab.get(token, self.unk_token_id)


def test_resolve_reasoning_end_token_id_requires_a_token_of_the_tokenizer():
    """A marker the tokenizer does not know would count every turn's whole generation as reasoning and
    starve the episode after its first turn, whether the tokenizer answers with unk or with None."""
    assert resolve_reasoning_end_token_id(_Tokenizer({"</think>": _END}), "</think>") == _END
    with pytest.raises(ValueError, match="not a token of this tokenizer"):
        resolve_reasoning_end_token_id(_Tokenizer({"</think>": _END}), "<|end_reasoning|>")
    with pytest.raises(ValueError, match="not a token of this tokenizer"):
        resolve_reasoning_end_token_id(_Tokenizer({}, unk_token_id=None), "</think>")


def test_stamp_records_budget_exhaustion_only_under_the_episode_scope():
    """The exhaustion flag is the driver's verdict on the spend it counted, so it exists only where a
    budget was shared across turns — and the metric follows the flag, absent rather than 0.0 otherwise."""
    env = _make_env(reasoning_effort_profiles=_SCOPED_PROFILES)

    def stamped(scope, reasoning_spent):
        traj = reset_episode(env, {"reasoning_effort": "high", **SINGLE_TEST_ANSWER})
        _bind_scoped("high", scope).stamp(traj, reasoning_spent=reasoning_spent)
        return traj

    exhausted = stamped("episode", 29600)
    assert exhausted.info[THINKING_BUDGET_EXHAUSTED_KEY] is True
    assert env.rollout_metrics(exhausted)["episode/thinking_budget_exhausted"] == 1.0
    within = stamped("episode", 1000)
    assert within.info[THINKING_BUDGET_EXHAUSTED_KEY] is False
    assert env.rollout_metrics(within)["episode/thinking_budget_exhausted"] == 0.0
    per_turn = stamped("turn", 40000)
    assert THINKING_BUDGET_EXHAUSTED_KEY not in per_turn.info
    assert "episode/thinking_budget_exhausted" not in env.rollout_metrics(per_turn)
    assert (per_turn.reasoning_effort, per_turn.reasoning_budget) == ("high", 18000)


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


def test_recovery_cap_tightens_per_level_and_never_exceeds_the_env_cap():
    env = _make_env(
        max_length_cutoff_recoveries=3,
        reasoning_effort_profiles={
            "low": {"max_submissions": 2, "max_test_calls": 2, "max_length_cutoff_recoveries": 1}
        },
    )
    low = reset_episode(env, {"reasoning_effort": "low", **SINGLE_TEST_ANSWER})
    assert low.info["episode_max_length_cutoff_recoveries"] == 1
    high = reset_episode(env, {"reasoning_effort": "high", **SINGLE_TEST_ANSWER})
    assert "episode_max_length_cutoff_recoveries" not in high.info, "a level without the key runs under the env cap"
    with pytest.raises(
        ValueError, match=r"max_length_cutoff_recoveries \(4\) exceeds the env's max_length_cutoff_recoveries \(3\)"
    ):
        _make_env(
            max_length_cutoff_recoveries=3, reasoning_effort_profiles={"low": {"max_length_cutoff_recoveries": 4}}
        )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
