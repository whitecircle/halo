#!/usr/bin/env python
"""Per-tool episode budgets and effort profiles are protocol- and base-level features.

Every native or ReAct env can cap calls per tool per episode (``tool_budgets``): the protocol admits
a call — binds its arguments, checks the episode's cap, counts it — before the handler runs, so a call
the handler could never run or one past the cap is refused as a tool error without spending the
budget, and a one-call cap cannot be double-spent by two calls in one turn. Under the native protocol a
turn whose every call was refused ran nothing: it is flagged untrainable like a turn of unknown tools, so the
next turn retries on the recovery reserve; a call to a tool a cap of 0 disables is an error but no refusal.
Every env binds an
effort profile at reset: the thinking budget is the base key; a task adds its own through
``EFFORT_PROFILE_KEY_MINIMA`` and ``_apply_effort_profile``. A refusal states the fact, never the cap:
what an episode may do is the chat template's and the engine's to control, not a number the model reads.

Run: python tests/cpu/environments/test_tool_budgets.py  (or pytest)
"""

import asyncio
import json
import re

import pytest

from src.environments.base import EPISODE_TOOL_BUDGETS_KEY, TOOL_CALL_COUNTS_KEY
from src.environments.envs.protocols.native import AsyncNativeToolUseEnvironment, NativeToolUseEnvironment
from src.environments.envs.protocols.react import ReActEnvironment
from src.environments.episode import recovering_turn
from src.environments.tools.definitions import NativeTool, NativeToolRegistry, ToolParameter

# The generic refusal, as the protocol prefixes it: the tool's name and the fact, no cap and no count.
_ECHO_SPENT = "Error: Not executed: this task's budget for echo is spent."


def _registry(budget_message=None, slow=False):
    async def echo_async(code: str) -> str:
        await asyncio.sleep(0.01)
        return f"echo:{code}"

    registry = NativeToolRegistry()
    registry.register(
        NativeTool(
            name="echo",
            description="Echoes code.",
            parameters=[ToolParameter("code", "string", "text")],
            handler=lambda code: f"echo:{code}",
            async_handler=echo_async if slow else None,
            budget_message=budget_message,
        )
    )
    registry.register(NativeTool(name="ping", description="Pings.", handler=lambda: "pong"))
    return registry


def _call(cid, name, **arguments):
    return {"id": cid, "type": "function", "function": {"name": name, "arguments": json.dumps(arguments)}}


def _observations(traj):
    return [m.content for m in traj.messages if m.role == "tool"]


def _react_observations(traj):
    return [m.content.removeprefix("Observation: ") for m in traj.messages if m.content.startswith("Observation: ")]


def _flags(traj):
    return [m.calls_rejected for m in traj.messages if m.role == "assistant"]


def _step(env, ids, *calls):
    """One turn of ``calls`` through the protocol's step, sync or async by the environment's class."""
    contexts = [{"tool_calls": list(calls)}]
    if isinstance(env, AsyncNativeToolUseEnvironment):
        return asyncio.run(env.step_async(ids, [""], contexts))[0]
    return env.step(ids, [""], contexts)[0]


def test_native_budget_refuses_past_the_cap_and_counts_only_admitted_calls():
    env = NativeToolUseEnvironment(
        tool_registry=_registry(), tool_budgets={"echo": 1}, tool_error_penalty=0.1, tool_success_reward=0.0
    )
    ids, _ = env.reset(["t"])
    traj = env.get_trajectories(ids)[0]
    assert traj.info[EPISODE_TOOL_BUDGETS_KEY] == {"echo": 1}
    calls = [_call("a", "echo"), _call("b", "echo", code="x"), _call("c", "echo", code="y"), _call("d", "ping")]
    env.step(ids, [""], [{"tool_calls": calls}])
    observations = _observations(traj)
    assert "echo: missing a required argument: 'code'" in observations[0]
    assert observations[1] == "echo:x"
    assert observations[2] == _ECHO_SPENT
    assert observations[3] == "pong"
    assert traj.info[TOOL_CALL_COUNTS_KEY] == {"echo": 1, "ping": 1}
    assert traj.total_reward == pytest.approx(-0.2), "two refusals charged as tool errors; admitted calls pay 0"


def test_tool_budgets_must_name_registered_tools_with_non_negative_int_caps():
    for bad in ({"nope": 1}, {"echo": -1}, {"echo": True}, {"echo": "2"}):
        with pytest.raises(ValueError, match="tool_budgets"):
            NativeToolUseEnvironment(tool_registry=_registry(), tool_budgets=bad)


def test_a_zero_cap_disables_the_tool_for_the_episode():
    env = NativeToolUseEnvironment(tool_registry=_registry(), tool_budgets={"echo": 0})
    ids, _ = env.reset(["t"])
    env.step(ids, [""], [{"tool_calls": [_call("a", "echo", code="x")]}])
    assert _observations(env.get_trajectories(ids)[0]) == [_ECHO_SPENT]


def test_the_refusal_wording_is_the_tools_own():
    env = NativeToolUseEnvironment(
        tool_registry=_registry(budget_message="No more echo this task; ping instead."), tool_budgets={"echo": 1}
    )
    ids, _ = env.reset(["t"])
    env.step(ids, [""], [{"tool_calls": [_call("a", "echo", code="x"), _call("b", "echo", code="y")]}])
    assert _observations(env.get_trajectories(ids)[0])[1] == "Error: No more echo this task; ping instead."


def test_a_refusal_names_neither_the_cap_nor_what_another_tool_has_left():
    """The refusal is the same text at every cap and beside any other budget, with no number in it:
    the tool's message, or the generic one, is used verbatim."""
    env = NativeToolUseEnvironment(tool_registry=_registry(), tool_budgets={"echo": 1, "ping": 3})
    ids, _ = env.reset(["t"])
    calls = [_call("p", "ping"), _call("a", "echo", code="x"), _call("b", "echo", code="y")]
    env.step(ids, [""], [{"tool_calls": calls}])
    refusal = _observations(env.get_trajectories(ids)[0])[2]
    assert refusal == _ECHO_SPENT and not re.search(r"\d", refusal)
    tool = NativeTool(name="echo", description="Echoes.")
    assert tool.budget_exhausted_message() == _ECHO_SPENT.removeprefix("Error: ")
    assert (
        NativeTool(name="echo", description="Echoes.", budget_message="Spent.").budget_exhausted_message() == "Spent."
    )


def test_an_enum_argument_outside_the_schema_is_refused_unspent():
    registry = NativeToolRegistry()
    registry.register(
        NativeTool(
            name="pick",
            description="Picks.",
            parameters=[ToolParameter("choice", "string", "one of", enum=["a", "b"])],
            handler=lambda choice: f"picked:{choice}",
        )
    )
    env = NativeToolUseEnvironment(tool_registry=registry, tool_budgets={"pick": 1})
    ids, _ = env.reset(["t"])
    env.step(ids, [""], [{"tool_calls": [_call("1", "pick", choice="z"), _call("2", "pick", choice="a")]}])
    traj = env.get_trajectories(ids)[0]
    assert _observations(traj) == ["Error: pick: choice must be one of a, b, got 'z'", "picked:a"]
    assert traj.info[TOOL_CALL_COUNTS_KEY] == {"pick": 1}


@pytest.mark.parametrize(
    "call",
    [_call("a", "echo"), _call("a", "echo", code="x", mode="fast"), _call("a", "echo", code=["x"])],
    ids=["missing-argument", "undeclared-argument", "list-for-a-string"],
)
def test_a_turn_of_only_malformed_calls_is_flagged_and_the_next_turn_recovers(call):
    """A call whose arguments do not bind ran nothing and spent nothing: a turn of nothing else is flagged
    like one that named no tool, so it is never rewarded and the next turn retries on the recovery reserve.
    Its reply and its tool-error price are the ones it always had."""
    env = NativeToolUseEnvironment(tool_registry=_registry(), tool_error_penalty=0.1, tool_success_reward=0.0)
    ids, _ = env.reset(["t"])
    step = _step(env, ids, call)
    traj = step.trajectory
    assert _flags(traj) == [True] and recovering_turn(traj)
    assert _observations(traj)[0].startswith("Error: echo: ")
    assert step.reward == pytest.approx(-0.1) and traj.info[TOOL_CALL_COUNTS_KEY] == {}
    step = _step(env, ids, _call("b", "echo", code="x"))
    assert _flags(step.trajectory) == [True, False] and not recovering_turn(step.trajectory)


@pytest.mark.parametrize("cls", [NativeToolUseEnvironment, AsyncNativeToolUseEnvironment])
def test_a_turn_of_only_calls_past_their_budget_is_flagged(cls):
    """A call refused over its tool's episode cap ran nothing: a turn of nothing else is flagged and the next
    turn recovers, the refusal text and price unchanged."""
    env = cls(tool_registry=_registry(), tool_budgets={"echo": 1}, tool_error_penalty=0.1, tool_success_reward=0.0)
    ids, _ = env.reset(["t"])
    assert _flags(_step(env, ids, _call("a", "echo", code="x")).trajectory) == [False]
    step = _step(env, ids, _call("b", "echo", code="y"), _call("c", "echo", code="z"))
    traj = step.trajectory
    assert _flags(traj) == [False, True] and recovering_turn(traj)
    assert _observations(traj)[1:] == [_ECHO_SPENT, _ECHO_SPENT]
    assert step.reward == pytest.approx(-0.2) and traj.info[TOOL_CALL_COUNTS_KEY] == {"echo": 1}


@pytest.mark.parametrize("cls", [NativeToolUseEnvironment, AsyncNativeToolUseEnvironment])
def test_a_turn_of_only_calls_to_a_disabled_tool_is_an_error_and_not_flagged(cls):
    """A cap of 0 disables the tool: a call to it gets the spent-budget reply and price, but the run offered no
    budget to lose, so it is no refusal: the turn stays unflagged and the next one gets no recovery reserve."""
    env = cls(tool_registry=_registry(), tool_budgets={"echo": 0}, tool_error_penalty=0.1, tool_success_reward=0.0)
    ids, _ = env.reset(["t"])
    step = _step(env, ids, _call("a", "echo", code="x"), _call("b", "echo", code="y"))
    traj = step.trajectory
    assert _observations(traj) == [_ECHO_SPENT, _ECHO_SPENT]
    assert step.reward == pytest.approx(-0.2) and traj.info[TOOL_CALL_COUNTS_KEY] == {}
    assert _flags(traj) == [False] and not recovering_turn(traj)


def test_a_refused_call_beside_one_that_ran_leaves_the_turn_trainable():
    env = NativeToolUseEnvironment(tool_registry=_registry(), tool_budgets={"echo": 0})
    ids, _ = env.reset(["t"])
    step = _step(env, ids, _call("a", "echo", code="x"), _call("b", "echo"), _call("c", "ping"))
    assert _observations(step.trajectory)[2] == "pong"
    assert _flags(step.trajectory) == [False] and not recovering_turn(step.trajectory)


async def test_two_calls_in_one_async_turn_cannot_double_spend_a_one_call_cap():
    env = AsyncNativeToolUseEnvironment(tool_registry=_registry(slow=True), tool_budgets={"echo": 1})
    ids, _ = await env.reset_async(["t"])
    calls = [_call("a", "echo", code="x"), _call("b", "echo", code="y")]
    await env.step_async(ids, [""], [{"tool_calls": calls}])
    traj = env.get_trajectories(ids)[0]
    observations = _observations(traj)
    assert sorted(observations) == sorted(["echo:x", _ECHO_SPENT])
    assert traj.info[TOOL_CALL_COUNTS_KEY] == {"echo": 1}


def test_react_budget_refuses_past_the_cap():
    env = ReActEnvironment(tool_registry=_registry(), tool_budgets={"echo": 1}, require_thought=False)
    ids, _ = env.reset(["t"])
    for _ in range(2):
        env.step(ids, ['Thought: go\nAction: echo(code="x")'])
    traj = env.get_trajectories(ids)[0]
    observations = _react_observations(traj)
    assert observations[0] == "echo:x"
    assert observations[1] == _ECHO_SPENT
    assert traj.info[TOOL_CALL_COUNTS_KEY] == {"echo": 1}
    with pytest.raises(ValueError, match="tool_budgets"):
        ReActEnvironment(tool_registry=_registry(), tool_budgets={"nope": 1})


# --- effort profiles on the base ---


def test_base_profiles_bind_the_thinking_budget_per_level():
    env = NativeToolUseEnvironment(
        tool_registry=_registry(),
        reasoning_effort="high",
        reasoning_effort_profiles={"high": {"thinking_tokens": 2048}},
    )
    assert env.thinking_budget_for_effort("high") == 2048
    assert env.thinking_budget_for_effort("low") is None, "the base binds no budget to an unprofiled level"


def test_base_rejects_non_finite_and_non_numeric_profile_values():
    """NaN passes a plain minimum check and would reach the engine as a thinking budget."""
    for bad in (float("nan"), float("inf"), True, "0.05", None):
        with pytest.raises(ValueError, match="thinking_tokens for effort 'low' must be a finite number"):
            NativeToolUseEnvironment(
                tool_registry=_registry(), reasoning_effort_profiles={"low": {"thinking_tokens": bad}}
            )


def test_base_rejects_a_fractional_count_budget():
    """A cap of 1.5 calls admits two; the registry's int minimum marks the key a count."""
    with pytest.raises(ValueError, match="thinking_tokens for effort 'low' is a count"):
        NativeToolUseEnvironment(
            tool_registry=_registry(), reasoning_effort_profiles={"low": {"thinking_tokens": 1.5}}
        )


async def test_async_admission_binds_against_the_async_handler():
    """A tool whose sync and async handlers differ must be admitted against the one that will run,
    or a call the async handler cannot bind spends the budget before it is refused."""

    async def narrow(text: str) -> str:
        return f"async:{text}"

    registry = NativeToolRegistry()
    registry.register(
        NativeTool(
            name="dual",
            description="Two handlers.",
            parameters=[
                ToolParameter("code", "string", "sync only", required=False),
                ToolParameter("text", "string", "async only", required=False),
            ],
            handler=lambda code, text="": f"sync:{code}",
            async_handler=narrow,
        )
    )
    env = AsyncNativeToolUseEnvironment(tool_registry=registry, tool_budgets={"dual": 1})
    ids, _ = await env.reset_async(["t"])
    await env.step_async(ids, [""], [{"tool_calls": [_call("a", "dual", code="x")]}])
    await env.step_async(ids, [""], [{"tool_calls": [_call("b", "dual", text="y")]}])
    traj = env.get_trajectories(ids)[0]
    observations = _observations(traj)
    assert "dual: missing a required argument: 'text'" in observations[0]
    assert observations[1] == "async:y", "the refused call left the one-call budget intact"
    assert traj.info[TOOL_CALL_COUNTS_KEY] == {"dual": 1}


def test_base_rejects_profile_keys_it_does_not_declare():
    with pytest.raises(ValueError, match="unknown keys \\['max_submissions'\\]"):
        NativeToolUseEnvironment(tool_registry=_registry(), reasoning_effort_profiles={"low": {"max_submissions": 1}})


def test_a_subclass_extends_the_admitted_keys_over_the_mro():
    class _Task(NativeToolUseEnvironment):
        EFFORT_PROFILE_KEY_MINIMA = {"max_probes": 0}

        def _apply_effort_profile(self, trajectory, level, profile):
            trajectory.info["seen"] = (level, dict(profile))

    assert _Task.effort_profile_key_minima() == {"thinking_tokens": 1, "max_probes": 0}
    env = _Task(
        tool_registry=_registry(), reasoning_effort="medium", reasoning_effort_profiles={"medium": {"max_probes": 2}}
    )
    ids, _ = env.reset(["t"])
    assert env.get_trajectories(ids)[0].info["seen"] == ("medium", {"max_probes": 2})
    ids, _ = env.reset(["t"], [{"reasoning_effort": "random"}])
    assert env.get_trajectories(ids)[0].info["seen"] == (None, {}), "an undetermined level binds an empty profile"
    with pytest.raises(ValueError, match="max_probes for effort 'low' must be >= 0"):
        _Task(tool_registry=_registry(), reasoning_effort_profiles={"low": {"max_probes": -1}})


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
