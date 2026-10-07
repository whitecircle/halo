#!/usr/bin/env python
"""CPU tests for effort-conditional interaction budgets and the failures-only verdict.

``reasoning_effort_profiles`` binds an effort level to a thinking budget and optional per-episode
``max_submissions``/``max_test_calls``, stamped at reset from a deterministic level (context-supplied
— the trainer stamps one per GRPO group — or a concrete env setting) as the per-tool caps the protocol
enforces. Nothing states them to the model: not the task message, not the tool descriptions, not a
reply — a call past its cap is refused with the spent-budget reply, which names no count. The grading
verdict lists only non-passing tests, one entry per distinct verdict, capped at ``_MAX_FAILURE_DETAILS``.

These drive ``CodeContestsEnvironment`` against a stub sandbox whose ``run`` returns a canned
result (no subprocesses, no network), through the protocol's tool dispatch.

Run: python tests/cpu/environments/test_effort_interaction_budgets.py  (or pytest)
"""

import json
import logging
import re

import pytest

from src.environments.base import (
    EPISODE_TOOL_BUDGETS_KEY,
    OUTPUT_BUDGET_EXHAUSTED_KEY,
    REWARD_COMPONENTS_KEY,
    TOOL_CALL_COUNTS_KEY,
)
from src.environments.envs.tasks.coding.code_contests import (
    SCRATCHPAD_BUDGET_SPENT_REPLY,
    SUBMISSION_BUDGET_SPENT_REPLY,
    SUBMIT_TOOL,
    CodeContestsEnvironment,
)
from src.environments.envs.tasks.coding.grading import (
    _MAX_FAILURE_DETAILS,
    VERDICT_DETAIL_FULL,
    run_solution_against_tests,
)
from src.environments.episode import (
    TurnGeneration,
    bind_episode_effort,
    resolve_reasoning_end_ids,
    resolve_reasoning_end_token_id,
    sampled_reasoning_tokens,
    step_context_from_generation,
)
from src.environments.tools.definitions import NativeToolCall
from tests.common.code_contests import (
    SINGLE_TEST_ANSWER,
    StubSandbox,
    call_tool,
    counts_beside_budget_words,
    reset_episode,
    retired_budget_phrases,
)

_PROFILES = {
    "low": {"max_submissions": 2, "max_test_calls": 2},
    "high": {"max_submissions": 1, "max_test_calls": 6},
}
# A hidden test the stub's ``X`` fails, so no submission is an accept and only the cap ends the episode.
_FAILING_ANSWER = {"answer": {"tests": [{"input": "", "output": "Y"}]}}


def _make_env(**kwargs):
    kwargs.setdefault("sandbox", StubSandbox())
    kwargs.setdefault("reasoning_effort_profiles", _PROFILES)
    return CodeContestsEnvironment(**kwargs)


def _task_message(traj) -> str:
    return next(m for m in reversed(traj.messages) if m.role == "user").content


def _step_call(env, ids, name, code="print('X')"):
    """One turn calling ``name`` through the protocol's step, so the episode's end is the env's verdict."""
    call = {"id": "s", "type": "function", "function": {"name": name, "arguments": json.dumps({"code": code})}}
    return env.step(ids, [""], [{"tool_calls": [call]}])[0]


def _assert_no_count(reply: str) -> None:
    assert not re.search(r"\d", reply) and not retired_budget_phrases(reply), reply


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
    assert not retired_budget_phrases(_task_message(traj)), "the task message states no budget"

    first = call_tool(env, traj, "submit_solution")
    second = call_tool(env, traj, "submit_solution")
    assert "Passed 1/1" in first
    assert second == f"Error: {SUBMISSION_BUDGET_SPENT_REPLY}"
    _assert_no_count(second)
    assert traj.info[TOOL_CALL_COUNTS_KEY]["submit_solution"] == 1, "a refused call spends nothing"


def test_the_cap_still_ends_the_episode_at_the_last_graded_submission():
    """Unstated is not unenforced: under ``low`` the second graded submission ends the episode, completed,
    with no budget statement anywhere in the conversation it leaves."""
    env = _make_env()
    ids, _ = env.reset(["solve it"], [{"reasoning_effort": "low", **_FAILING_ANSWER}])
    first = _step_call(env, ids, SUBMIT_TOOL)
    assert not first.done
    last = _step_call(env, ids, SUBMIT_TOOL)
    traj = last.trajectory
    assert last.done and traj.info["completed"] and not traj.truncated
    assert env._submissions(traj) == 2
    assert traj.info[EPISODE_TOOL_BUDGETS_KEY] == {"python_repl": 2, "submit_solution": 2}
    assert not any(retired_budget_phrases(m.content) for m in traj.messages if m.content)


def test_scratchpad_cap_reads_episode_budget():
    env = _make_env()
    traj = reset_episode(env, {"reasoning_effort": "low", **SINGLE_TEST_ANSWER})
    assert traj.info[EPISODE_TOOL_BUDGETS_KEY]["python_repl"] == 2
    call_tool(env, traj, "python_repl")
    call_tool(env, traj, "python_repl")
    third = call_tool(env, traj, "python_repl")
    assert third == f"Error: {SCRATCHPAD_BUDGET_SPENT_REPLY}"
    assert SUBMIT_TOOL in third, "the refusal points at the graded channel"
    _assert_no_count(third)
    assert traj.info[TOOL_CALL_COUNTS_KEY]["python_repl"] == 2


def test_the_scratchpad_refusal_reads_the_same_whatever_is_left_to_submit():
    """The refusal states the spent budget and the way out, never how many submissions remain: a
    spent submission changes nothing in it."""
    env = _make_env()
    traj = reset_episode(env, {"reasoning_effort": "low", **SINGLE_TEST_ANSWER})
    call_tool(env, traj, "submit_solution")
    call_tool(env, traj, "python_repl")
    call_tool(env, traj, "python_repl")
    assert call_tool(env, traj, "python_repl") == f"Error: {SCRATCHPAD_BUDGET_SPENT_REPLY}"


def test_resubmission_penalty_prices_each_graded_submission_after_the_first():
    """Three graded submissions of which only the last counts make the judge a free test oracle: the
    penalty prices every probe after the first, a single submission pays nothing, and the component
    stays inside the decomposition the trainer's residue check sums."""
    profiles = {"high": {"max_submissions": 3, "max_test_calls": 6}}
    # The stub prints X, so a hidden test expecting Y fails every submission and none is an accept.
    failing = {"answer": {"tests": [{"input": "", "output": "Y"}]}}
    env = _make_env(reasoning_effort_profiles=profiles, resubmission_penalty=0.1)
    traj = reset_episode(env, {"reasoning_effort": "high", **failing})
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
    assert results[0].content == f"Error: {SCRATCHPAD_BUDGET_SPENT_REPLY}"
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


def test_undetermined_level_stamps_class_caps_and_states_none():
    env = _make_env(reasoning_effort="random")
    traj = reset_episode(env, dict(SINGLE_TEST_ANSWER))
    assert traj.info[EPISODE_TOOL_BUDGETS_KEY] == {"python_repl": 5, "submit_solution": 2}
    assert not retired_budget_phrases(_task_message(traj))


def test_level_without_interaction_keys_stamps_class_caps():
    env = _make_env(reasoning_effort="medium")
    traj = reset_episode(env, dict(SINGLE_TEST_ANSWER))
    assert traj.info[EPISODE_TOOL_BUDGETS_KEY] == {"python_repl": 5, "submit_solution": 2}
    assert not retired_budget_phrases(_task_message(traj))


def test_thinking_only_profiles_keep_the_class_caps():
    env = _make_env(reasoning_effort_profiles={"high": {"thinking_tokens": 24000}})
    assert env.thinking_budget_for_effort("high") == 24000
    assert env.thinking_budget_for_effort("low") == 4096
    traj = reset_episode(env, {"reasoning_effort": "high", **SINGLE_TEST_ANSWER})
    assert traj.info[EPISODE_TOOL_BUDGETS_KEY] == {"python_repl": 5, "submit_solution": 2}, "class caps still bind"
    assert not retired_budget_phrases(_task_message(traj))


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


@pytest.mark.parametrize(
    "knobs",
    [
        {},
        {"reasoning_effort_profiles": None},
        {"reasoning_effort_profiles": None, "max_submissions": 1, "max_test_calls": 0},
        {"language": "cpp"},
    ],
    ids=["interaction-profiles", "class-caps", "one-submission-no-scratchpad", "cpp"],
)
def test_tool_descriptions_state_no_budget_whatever_binds_it(knobs):
    """The descriptions name what each tool does and the way to be graded, never how many calls the
    episode gets: no deferral to the task message, no count, no "only" or "disabled" spelling of a cap."""
    env = _make_env(**knobs)
    submit = env.registry.get(SUBMIT_TOOL).description
    scratchpad = env.registry.get(env.test_tool_name).description
    for description in (submit, scratchpad):
        assert not retired_budget_phrases(description), description
        assert not counts_beside_budget_words(description), description
    assert SUBMIT_TOOL in scratchpad and "does NOT submit" in scratchpad
    assert "the last graded submission is the one that counts" in submit


def test_effort_binding_caps_the_reasoning_channel_and_leaves_the_turn_total():
    """The level's budget caps the reasoning channel of every turn, clamped by the run's cap; the turn's
    total stays the run's ``max_tokens``, so the answer room is what the cap leaves of it. A level
    whose budget fills the turn would leave no room, and is refused."""
    env = _make_env(reasoning_effort_profiles={"low": {"thinking_tokens": 4096}, "high": {"thinking_tokens": 30000}})
    bound = bind_episode_effort({"reasoning_effort": "low"}, env, max_tokens=20000, max_thinking_tokens=18000)
    assert (bound.thinking_budget, bound.max_tokens) == (4096, 20000)
    # Over-budget profiles clamp to the run's cap, else they silently raise the run's CoT cap.
    over = bind_episode_effort({"reasoning_effort": "high"}, env, max_tokens=40000, max_thinking_tokens=18000)
    assert (over.thinking_budget, over.max_tokens) == (18000, 40000)
    no_global = bind_episode_effort({"reasoning_effort": "low"}, env, max_tokens=20000)
    assert (no_global.thinking_budget, no_global.max_tokens) == (4096, 20000)
    with pytest.raises(
        ValueError, match=r"'high' level's thinking_tokens \(30000\) must sit below rollout_max_tokens"
    ):
        bind_episode_effort({"reasoning_effort": "high"}, env, max_tokens=30000)
    with pytest.raises(ValueError, match="must sit below rollout_max_tokens"):
        bind_episode_effort({"reasoning_effort": "low"}, env, max_tokens=1024)


_PROFILED = {"low": {"thinking_tokens": 4096}, "high": {"thinking_tokens": 30000}}
# The reasoning-end marker's id: one no reasoning id below carries.
_END = 151668


def _bind_level(level, max_thinking_tokens=18000, max_episode_tokens=None):
    env = _make_env(reasoning_effort_profiles=_PROFILED)
    return bind_episode_effort(
        {"reasoning_effort": level},
        env,
        max_tokens=30000,
        max_thinking_tokens=max_thinking_tokens,
        max_episode_tokens=max_episode_tokens,
    )


def test_every_turn_runs_under_the_levels_cap_whatever_was_generated():
    """Without an output budget a turn's caps never change: the level's cap clamped to the run's, and
    the run's turn total, however much the earlier turns sampled; nothing is ever exhausted."""
    high = _bind_level("high")
    assert high.thinking_budget == 18000, "the level's budget clamps to the run's per-turn cap"
    assert {tuple(high.turn_caps(generated).items()) for generated in (0, 17000, 40000)} == {
        (("max_tokens", 30000), ("max_thinking_tokens", 18000))
    }
    assert high.turn_caps(10**6) == {"max_tokens": 30000, "max_thinking_tokens": 18000}
    low = _bind_level("low")
    assert (low.thinking_budget, low.turn_caps(29000)["max_thinking_tokens"]) == (4096, 4096)


def test_an_output_budget_narrows_a_turns_caps_from_the_reasoning_side_first():
    """70000 for the episode over 30000-token turns with an 18000-token reasoning cap (12000 of answer
    room): a turn with 20000 left asks for 20000 and reasons 8000, one with exactly the room left
    keeps one reasoning token (never a cap of 0), and one token less leaves no turn to start."""
    high = _bind_level("high", max_episode_tokens=70000)
    assert high.episode_tokens == 70000
    assert high.turn_caps(40000) == {"max_tokens": 30000, "max_thinking_tokens": 18000}
    assert high.turn_caps(50000) == {"max_tokens": 20000, "max_thinking_tokens": 8000}
    assert high.turn_caps(58000) == {"max_tokens": 12000, "max_thinking_tokens": 1}
    assert high.turn_caps(58001) is None
    assert _bind_level("high").episode_tokens is None


def test_without_a_run_cap_the_levels_budget_is_the_turns_cap():
    low = _bind_level("low", max_thinking_tokens=None)
    assert (low.thinking_budget, low.max_tokens) == (4096, 30000)
    with pytest.raises(ValueError, match="must sit below rollout_max_tokens"):
        _bind_level("high", max_thinking_tokens=None)


def _gen(token_ids):
    return TurnGeneration(text="", tool_calls=[], reasoning="", tokens=5, token_ids=token_ids)


def test_sampled_reasoning_tokens_counts_through_the_marker_and_is_absent_without_ids_or_marker():
    """The marker is the last reasoning token counted (on a forced close, the count is the budget vLLM
    enforced: its counter takes the prompt's token after ``<think>`` but not the close); a turn cut
    before it closed spent every sampled id on reasoning. Without the ids, or without the marker's id,
    there is nothing exact to count and the count is absent rather than zero."""
    assert sampled_reasoning_tokens([11, 12, 13, _END, 14], _END) == 4
    assert sampled_reasoning_tokens([11, 12, 13, 14, 15], _END) == 5
    assert sampled_reasoning_tokens([], _END) == 0
    assert sampled_reasoning_tokens(None, _END) is None
    assert sampled_reasoning_tokens([11, _END], None) is None


def test_a_turns_step_context_carries_its_cap_and_the_reasoning_it_sampled():
    """The overlong charge reads the pair off each turn: the cap the turn's level set, and the reasoning
    the turn sampled, counted as the budget counts it (the close included; a cut turn's every id). Without
    the ids or the marker's id the count is absent rather than zero, which would read as a free turn."""
    closed = step_context_from_generation(
        None, _gen([11, 12, _END, 14]), thinking_cap=18000, reasoning_end_token_id=_END
    )
    assert (closed["thinking_cap"], closed["reasoning_tokens"]) == (18000, 3)
    cut = step_context_from_generation(None, _gen([11, 12, 13]), thinking_cap=3, reasoning_end_token_id=_END)
    assert cut["reasoning_tokens"] == 3
    for gen, end in ((_gen(None), _END), (_gen([11, _END]), None)):
        assert "reasoning_tokens" not in step_context_from_generation(None, gen, reasoning_end_token_id=end)
    assert "thinking_cap" not in step_context_from_generation(None, _gen([11]), reasoning_end_token_id=_END)


class _Tokenizer:
    """The two attributes the resolver reads: an encoding per text and the ids of the added tokens."""

    def __init__(self, encodings, added):
        self._encodings = encodings
        self.added_tokens_decoder = dict.fromkeys(added)

    def encode(self, text, add_special_tokens=True):
        assert not add_special_tokens, "the engine encodes its reasoning end string without special tokens"
        return list(self._encodings[text])


# ``</think>`` as a added token, a string that splits into plain text, and a close of several tokens
# around added tokens (gpt-oss's final-channel opener).
_VOCAB = _Tokenizer(
    {
        "</think>": [_END],
        "<|end_reasoning|>": [11, 12, 13],
        "": [],
        "<|start|>assistant<|channel|>final<|message|>": [70, 71, 72, 73, 74],
    },
    added={_END, 70, 72, 74},
)


def test_resolve_reasoning_end_ids_encodes_the_marker_as_the_engine_forces_it():
    """The ids vLLM appends at the budget are its parser's end string encoded without special tokens, one
    id or several; a string of plain text is no close the model writes."""
    assert resolve_reasoning_end_ids(_VOCAB, "</think>") == (_END,)
    assert resolve_reasoning_end_ids(_VOCAB, "<|start|>assistant<|channel|>final<|message|>") == (70, 71, 72, 73, 74)
    with pytest.raises(ValueError, match="not a reasoning marker of this tokenizer"):
        resolve_reasoning_end_ids(_VOCAB, "<|end_reasoning|>")
    with pytest.raises(ValueError, match="not a reasoning marker of this tokenizer"):
        resolve_reasoning_end_ids(_VOCAB, "")


def test_resolve_reasoning_end_token_id_requires_one_control_token():
    """The overlong charge counts a turn's reasoning up to one marker token: one the tokenizer does not
    write would count every turn's whole generation as reasoning. A marker of several tokens is refused,
    and the charge told to turn off."""
    assert resolve_reasoning_end_token_id(_VOCAB, "</think>") == _END
    with pytest.raises(ValueError, match="not a reasoning marker of this tokenizer"):
        resolve_reasoning_end_token_id(_VOCAB, "<|end_reasoning|>")
    opener = "<|start|>assistant<|channel|>final<|message|>"
    with pytest.raises(ValueError, match="encodes to 5 tokens.*turn it off for this model"):
        resolve_reasoning_end_token_id(_VOCAB, opener)


def test_stamp_records_output_budget_exhaustion_only_under_an_output_budget():
    """The exhaustion flag is the driver's verdict on what the episode generated against its output
    budget, so it exists only where one was set — and the metric follows the flag, absent rather than
    0.0 otherwise. The level and its per-turn cap are stamped either way."""
    env = _make_env(reasoning_effort_profiles=_PROFILED)

    def stamped(max_episode_tokens, generated):
        traj = reset_episode(env, {"reasoning_effort": "high", **SINGLE_TEST_ANSWER})
        _bind_level("high", max_episode_tokens=max_episode_tokens).stamp(traj, generated)
        return traj

    exhausted = stamped(70000, 58001)
    assert exhausted.info[OUTPUT_BUDGET_EXHAUSTED_KEY] is True
    assert env.rollout_metrics(exhausted)["episode/output_budget_exhausted"] == 1.0
    within = stamped(70000, 1000)
    assert within.info[OUTPUT_BUDGET_EXHAUSTED_KEY] is False
    assert env.rollout_metrics(within)["episode/output_budget_exhausted"] == 0.0
    unbounded = stamped(None, 10**6)
    assert OUTPUT_BUDGET_EXHAUSTED_KEY not in unbounded.info
    assert "episode/output_budget_exhausted" not in env.rollout_metrics(unbounded)
    assert (unbounded.reasoning_effort, unbounded.reasoning_budget) == ("high", 18000)


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


@pytest.mark.parametrize(
    ("knob", "value"), [("max_submissions", 1.5), ("max_submissions", True), ("max_test_calls", 2.0)]
)
def test_a_count_knob_takes_only_an_int(knob, value):
    """A budget is stamped as a per-tool call cap, so a float or a bool standing in for a count is refused."""
    with pytest.raises(ValueError, match=f"{knob} must be an int"):
        _make_env(**{knob: value})


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
