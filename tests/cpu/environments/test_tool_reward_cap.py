#!/usr/bin/env python
"""CPU tests for the per-call tool reward and its episode cap, owned by ``BaseEnvironment``.

``tool_success_reward`` / ``tool_error_penalty`` price every executed call through one accounting
(``_credit_tool_call``) under both protocols, and ``tool_reward_cap`` bounds what an episode earns
from successful calls in total — default one paid call per turn of the budget. Without the cap five
paid calls a turn over ten turns pay 2.5 against a 1.0 solve, so call spam out-earns the objective.

Run: python tests/cpu/environments/test_tool_reward_cap.py
"""

import pytest

from src.environments.base import TOOL_REWARD_PAID_KEY
from src.environments.envs.protocols.mcp import NativeMCPClientEnvironment
from src.environments.envs.protocols.native import NativeToolUseEnvironment
from src.environments.envs.protocols.react import ReActEnvironment
from src.environments.envs.tasks.coding.code_contests import CodeContestsEnvironment
from src.environments.registry import resolve_environment
from src.environments.tools.definitions import NativeTool, NativeToolRegistry, ToolParameter


def _echo_registry():
    registry = NativeToolRegistry()
    registry.register(
        NativeTool(
            name="echo",
            description="echo",
            parameters=[ToolParameter("text", "string", "text")],
            handler=lambda text: text,
        )
    )
    return registry


def _native(**kwargs):
    return NativeToolUseEnvironment(tool_registry=_echo_registry(), **kwargs)


def _react(**kwargs):
    return ReActEnvironment(tool_registry=_echo_registry(), thought_reward=0.0, require_thought=False, **kwargs)


def _native_call(name: str, cid: str):
    return {"id": cid, "function": {"name": name, "arguments": '{"text": "hi"}'}}


def _native_turn(env, eid, *names):
    calls = [_native_call(name, f"c{i}") for i, name in enumerate(names)]
    return env.step([eid], ["calling"], [{"finish_reason": "tool_calls", "tool_calls": calls}])[0]


def _react_turn(env, eid, name):
    return env.step([eid], [f'Action: {name}(text="hi")'], [{"finish_reason": "stop"}])[0]


def test_default_cap_is_one_paid_call_per_turn():
    env = _native(max_turns=4, tool_success_reward=0.05)
    assert env.tool_reward_cap == pytest.approx(0.2)


def test_call_spam_earns_exactly_the_cap():
    env = _native(max_turns=2, tool_success_reward=0.05, max_tool_calls_per_turn=5)
    eid = env.reset(["task"], [{}])[0][0]

    first = _native_turn(env, eid, *["echo"] * 5)
    assert first.reward == pytest.approx(0.1), "five paid calls in one turn must stop at the cap"
    second = _native_turn(env, eid, *["echo"] * 5)
    assert second.reward == pytest.approx(0.0), "a call past the cap pays nothing"

    traj = env.get_trajectories([eid])[0]
    assert traj.info["successful_tool_calls"] == 10, "the cap bounds pay, not the count"
    assert traj.info[TOOL_REWARD_PAID_KEY] == pytest.approx(env.tool_reward_cap)


def test_explicit_cap_is_honoured_to_the_partial_call():
    env = _native(max_turns=4, tool_success_reward=0.05, tool_reward_cap=0.07)
    eid = env.reset(["task"], [{}])[0][0]
    step = _native_turn(env, eid, "echo", "echo", "echo")
    assert step.reward == pytest.approx(0.07)


def test_errors_are_still_charged_past_the_cap():
    env = _native(max_turns=4, tool_success_reward=0.05, tool_error_penalty=0.1, tool_reward_cap=0.05)
    eid = env.reset(["task"], [{}])[0][0]
    _native_turn(env, eid, "echo")
    step = _native_turn(env, eid, "echo", "nope")
    assert step.reward == pytest.approx(-0.1)


def test_react_and_native_pay_identically_for_the_same_call_sequence():
    """One accounting for both protocols: the same knobs, the same sequence, the same per-turn pay."""
    knobs = {"max_turns": 8, "tool_success_reward": 0.05, "tool_error_penalty": 0.1, "tool_reward_cap": 0.12}
    sequence = ["echo", "echo", "nope", "echo", "echo"]

    native = _native(**knobs)
    native_eid = native.reset(["task"], [{}])[0][0]
    native_pay = [_native_turn(native, native_eid, name).reward for name in sequence]

    react = _react(**knobs)
    react_eid = react.reset(["task"], [{}])[0][0]
    react_pay = [_react_turn(react, react_eid, name).reward for name in sequence]

    assert native_pay == pytest.approx([0.05, 0.05, -0.1, 0.02, 0.0])
    assert react_pay == pytest.approx(native_pay)
    for env, eid in ((native, native_eid), (react, react_eid)):
        info = env.get_trajectories([eid])[0].info
        assert (info["total_tool_calls"], info["successful_tool_calls"]) == (5, 4)


def test_mcp_default_pay_is_twice_native():
    env = NativeMCPClientEnvironment()
    assert env.tool_success_reward == pytest.approx(0.1)
    assert env.tool_error_penalty == pytest.approx(NativeToolUseEnvironment.DEFAULT_TOOL_ERROR_PENALTY)
    assert env.tool_reward_cap == pytest.approx(0.1 * env.max_turns)


def test_code_contests_turns_per_call_pay_off():
    env = CodeContestsEnvironment(language="python", sandbox_backend="local")
    assert (env.tool_success_reward, env.tool_error_penalty, env.tool_reward_cap) == (0.0, 0.0, 0.0)


@pytest.mark.parametrize("env_type", ["react_math", "native_math"])
def test_yaml_spellings_reach_the_base_through_the_registry(env_type):
    env = resolve_environment(
        env_type, {"tool_success_reward": 0.2, "tool_error_penalty": 0.3, "tool_reward_cap": 0.5, "max_turns": 3}
    )
    assert (env.tool_success_reward, env.tool_error_penalty, env.tool_reward_cap) == (0.2, 0.3, 0.5)


@pytest.mark.parametrize("knob", ["tool_success_reward", "tool_error_penalty", "tool_reward_cap"])
def test_a_negative_magnitude_is_refused(knob):
    with pytest.raises(ValueError, match=knob):
        _react(**{knob: -0.1})


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
