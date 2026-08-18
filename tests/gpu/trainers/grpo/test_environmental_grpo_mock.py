#!/usr/bin/env python
"""
Mock test for DistributedAsyncEnvironmentalGRPOTrainer and environment components.

Tests the environmental GRPO infrastructure with mocked components, validating:
1. Public surface - the exported classes keep the hierarchy and factory shapes the rollout layer
   branches on (the import line alone is the export check; `is not None` on top of it is not)
2. Environment registry - registration round-trips, every built-in resolves by name
3. Message/Trajectory classes - creation, mutation, serialization
4. Mock rollout - trajectory with tool calls simulates a multi-turn episode
5. Tool registry - NativeTool creation, execution, OpenAI schema generation
6. ReAct parsing - thought/action/final answer parsing from text

Nothing here needs Ray or a rollout server: it covers only the environment-side
components DistributedAsyncEnvironmentalGRPOTrainer depends on.

Usage:
    torchrun --nproc_per_node=2 \
        tests/gpu/trainers/grpo/test_environmental_grpo_mock.py
"""

import dataclasses

import torch

from tests.common.harness import gpu_test_main, record_check
from tests.common.utils import log


def test_import_base_environment():
    """The base environment surface the trainer reads BY NAME, not merely that the names import.

    An ``import`` line already fails on a missing name, so ``is not None`` on a class certifies
    nothing; a renamed ``EnvStep`` field or a severed Async/sync inheritance would pass it while the
    rollout layer breaks.
    """
    from src.environments.base import (
        AsyncBaseEnvironment,
        BaseEnvironment,
        EnvStep,
        Message,
        Trajectory,
    )

    # The async env must stay a BaseEnvironment: the trainer routes both through one code path.
    assert issubclass(AsyncBaseEnvironment, BaseEnvironment)
    # Attribute reads in the rollout/eval drivers, so a rename is silent until runtime.
    assert {"trajectory", "observation", "reward", "done", "truncated", "info"} <= set(EnvStep.__slots__)
    assert Message.assistant("answer", thinking="cot").thinking == "cot"
    assert Trajectory().total_reward == 0.0 and Trajectory().num_turns == 0


def test_react_and_native_tool_use_surface():
    """The ReAct / native-tool-use surface the rollout layer reaches for, by name AND by shape.

    The ``from ... import`` line is already the export check — ``is not None`` on top of it certifies
    nothing. What a rename cannot fake is the hierarchy and the factories' return type: both tool-use
    environments must stay ``NativeToolUseEnvironment`` subclasses (the rollout driver branches on
    that), and both ReAct factories must hand back a ``ReActEnvironment``, not a bare registry.
    """
    from src.environments.base import AsyncBaseEnvironment, BaseEnvironment
    from src.environments.envs.protocols.native import AsyncNativeToolUseEnvironment, NativeToolUseEnvironment
    from src.environments.envs.protocols.react import (
        ReActEnvironment,
        ReActStep,
        create_react_math_environment,
        create_react_search_environment,
        parse_react_output,
    )
    from src.environments.envs.tasks.coding.swe import SweEnvironment
    from src.environments.tools.definitions import NativeToolResult

    assert issubclass(AsyncNativeToolUseEnvironment, (NativeToolUseEnvironment, AsyncBaseEnvironment))
    assert issubclass(SweEnvironment, NativeToolUseEnvironment)
    assert issubclass(ReActEnvironment, BaseEnvironment)
    assert isinstance(create_react_math_environment(max_turns=2), ReActEnvironment)
    assert isinstance(create_react_search_environment(max_turns=2), ReActEnvironment)
    assert isinstance(parse_react_output("Final Answer: 7"), ReActStep)
    # Fields the tool loop reads off a result before deciding whether to retry or end the turn.
    assert {"tool_call_id", "name", "content", "success", "unknown_tool"} <= {
        f.name for f in dataclasses.fields(NativeToolResult)
    }


def test_registry_registration_round_trip():
    """``register_environment`` → ``get_registered_environments`` → ``resolve_environment``.

    The three functions are one contract: a name registered by an out-of-tree environment must show
    up in the listing and resolve through the same factory. Importing them proves nothing about that.
    """
    from src.environments.registry import get_registered_environments, register_environment, resolve_environment

    sentinel = object()
    name = "halo_test_registry_round_trip"
    register_environment(name, lambda config: sentinel, override=True)
    assert name in get_registered_environments()
    assert resolve_environment(name, {}) is sentinel


def test_registry_list_builtins():
    """Every built-in environment type stays registered.

    The list is the whole shipped roster, not a sample: a `resolve_environment` name that drops out
    of `src/environments/envs/` is only visible here, and a partial list lets the rest vanish.
    """
    from src.environments.registry import get_registered_environments

    available = get_registered_environments()
    expected = [
        "code_contests",
        "codeforces",
        "exam_qa",
        "mcp",
        "native_coding",
        "native_combined",
        "native_math",
        "qa_search",
        "react_math",
        "react_search",
        "swe",
    ]
    for name in expected:
        assert name in available, f"Missing builtin environment: '{name}'. Available: {available}"


def test_registry_resolve_react_math():
    """Verify react_math environment can be resolved by name."""
    from src.environments.registry import resolve_environment

    env = resolve_environment("react_math", {"max_turns": 5})
    assert env is not None

    # ReAct reads the action out of the assistant text: its tools live in the registry (named in
    # the system prompt), and no OpenAI tool schema is advertised to the server.
    assert env.get_tools_schema() is None, "ReAct must not advertise a tool schema"
    assert "calculate" in env.registry.names(), f"Expected 'calculate' in tools, got {env.registry.names()}"


def test_registry_resolve_native_math():
    """Verify native_math environment can be resolved by name."""
    from src.environments.registry import resolve_environment

    env = resolve_environment("native_math", {"max_turns": 5})
    assert env is not None

    tools_schema = env.get_tools_schema()
    tool_names = [t["function"]["name"] for t in tools_schema]
    assert "calculate" in tool_names


def test_registry_resolve_unknown_raises():
    """Verify resolving an unknown environment type raises ValueError."""
    from src.environments.registry import resolve_environment

    try:
        resolve_environment("nonexistent_env_type_xyz", {})
        raise AssertionError("Should have raised ValueError")
    except ValueError as e:
        assert "Unknown environment type" in str(e)
        assert "nonexistent_env_type_xyz" in str(e)


def test_message_creation():
    """Test Message creation and factory methods."""
    from src.environments.base import Message

    msg = Message(role="user", content="Hello")
    assert msg.role == "user"
    assert msg.content == "Hello"

    user_msg = Message.user("Question?")
    assert user_msg.role == "user"
    assert user_msg.content == "Question?"

    assistant_msg = Message.assistant("Answer!")
    assert assistant_msg.role == "assistant"

    system_msg = Message.system("You are helpful.")
    assert system_msg.role == "system"

    tool_msg = Message.tool("42", "call_123", "calculate")
    assert tool_msg.role == "tool"
    assert tool_msg.tool_call_id == "call_123"
    assert tool_msg.name == "calculate"


def test_message_to_dict():
    """Test Message serialization to dict."""
    from src.environments.base import Message

    msg = Message.user("Hello")
    d = msg.to_dict()
    assert d["role"] == "user"
    assert d["content"] == "Hello"


def test_message_from_dict():
    """Test Message deserialization from dict."""
    from src.environments.base import Message

    msg = Message.from_dict({"role": "assistant", "content": "Hi there"})
    assert msg.role == "assistant"
    assert msg.content == "Hi there"


def test_trajectory_creation():
    """Test Trajectory creation and basic operations."""
    from src.environments.base import Trajectory

    traj = Trajectory()
    assert traj.num_turns == 0
    assert traj.total_reward == 0.0
    assert not traj.done
    assert len(traj.messages) == 0


def test_trajectory_add_messages():
    """Test adding messages to a Trajectory."""
    from src.environments.base import Message, Trajectory

    traj = Trajectory()
    traj.add_message(Message.user("What is 2+2?"))
    assert len(traj.messages) == 1
    assert traj.num_turns == 0  # Only counts assistant messages

    traj.add_message(Message.assistant("4"))
    assert len(traj.messages) == 2
    assert traj.num_turns == 1


def test_trajectory_rewards():
    """Test Trajectory reward tracking."""
    from src.environments.base import Message, Trajectory

    traj = Trajectory()
    traj.add_message(Message.user("Q1"))
    traj.add_message(Message.assistant("A1"))

    traj.add_reward(0.5)
    assert traj.total_reward == 0.5

    traj.add_reward(0.3)
    assert abs(traj.total_reward - 0.8) < 1e-6


def test_trajectory_get_conversation():
    """Test Trajectory serialization to conversation format."""
    from src.environments.base import Message, Trajectory

    traj = Trajectory()
    traj.add_message(Message.user("Q"))
    traj.add_message(Message.assistant("A"))

    conv = traj.get_conversation()
    assert len(conv) == 2
    assert conv[0]["role"] == "user"
    assert conv[0]["content"] == "Q"
    assert conv[1]["role"] == "assistant"
    assert conv[1]["content"] == "A"


def test_mock_multi_turn_trajectory():
    """Simulate a multi-turn trajectory with tool calls (no real model)."""
    from src.environments.base import Message, Trajectory

    traj = Trajectory()

    traj.add_message(Message.user("What is the square root of 144?"))

    traj.add_message(
        Message.assistant('Thought: I need to calculate the square root.\nAction: calculate(expression="sqrt(144)")')
    )
    traj.add_reward(0.1)

    traj.add_message(Message.tool("12.0", "call_001", "calculate"))

    traj.add_message(Message.assistant("Thought: The result is 12.\nFinal Answer: 12"))
    traj.add_reward(1.0)
    traj.done = True

    assert traj.num_turns == 2
    assert abs(traj.total_reward - 1.1) < 1e-6
    assert traj.done

    conv = traj.get_conversation()
    assert len(conv) == 4
    assert conv[0]["role"] == "user"
    assert conv[1]["role"] == "assistant"
    assert conv[2]["role"] == "tool"
    assert conv[3]["role"] == "assistant"


def test_mock_truncated_trajectory():
    """Simulate a trajectory that gets truncated at max_turns."""
    from src.environments.base import Message, Trajectory

    max_turns = 3
    traj = Trajectory()
    traj.add_message(Message.user("What is 1+1?"))

    for i in range(max_turns):
        traj.add_message(Message.assistant(f"Thinking step {i + 1}..."))
        traj.add_reward(0.05)

    traj.done = True

    assert traj.num_turns == max_turns
    assert traj.done
    assert abs(traj.total_reward - 0.15) < 1e-6


def test_native_tool_creation():
    """Test creating a NativeTool with handler."""
    from src.environments.tools.definitions import NativeTool, ToolParameter

    tool = NativeTool(
        name="add",
        description="Add two numbers",
        parameters=[
            ToolParameter("a", "number", "First number"),
            ToolParameter("b", "number", "Second number"),
        ],
        handler=lambda a, b: str(a + b),
    )

    result = tool.execute(a=5, b=3)
    assert result == "8", f"Expected '8', got '{result}'"


def test_native_tool_openai_schema():
    """Test NativeTool OpenAI schema generation."""
    from src.environments.tools.definitions import NativeTool, ToolParameter

    tool = NativeTool(
        name="search",
        description="Search the web",
        parameters=[
            ToolParameter("query", "string", "Search query", required=True),
            ToolParameter("limit", "number", "Max results", required=False),
        ],
        handler=lambda query, limit=10: f"Results for: {query}",
    )

    schema = tool.to_openai_schema()
    assert schema["type"] == "function"
    assert schema["function"]["name"] == "search"
    assert "query" in schema["function"]["parameters"]["properties"]
    assert "limit" in schema["function"]["parameters"]["properties"]
    assert "query" in schema["function"]["parameters"]["required"]
    assert "limit" not in schema["function"]["parameters"]["required"]


def test_native_tool_registry():
    """Test NativeToolRegistry operations."""
    from src.environments.tools.definitions import NativeTool, NativeToolRegistry

    registry = NativeToolRegistry()

    tool1 = NativeTool(
        name="tool_a",
        description="First",
        parameters=[],
        handler=lambda: "a",
    )
    tool2 = NativeTool(
        name="tool_b",
        description="Second",
        parameters=[],
        handler=lambda: "b",
    )

    registry.register(tool1).register(tool2)
    assert len(registry) == 2
    assert "tool_a" in registry.names()
    assert "tool_b" in registry.names()
    assert registry.get("tool_a") is tool1
    assert registry.get("nonexistent") is None


def test_math_tools_factory():
    """Test create_native_math_tools factory."""
    from src.environments.tools.factories import create_native_math_tools

    registry = create_native_math_tools()
    assert "calculate" in registry.names()

    calc = registry.get("calculate")
    result = calc.execute(expression="2 + 3 * 4")
    assert result == "14", f"Expected '14', got '{result}'"


def test_native_tool_call_parsing():
    """Test NativeToolCall parsing from OpenAI format."""
    from src.environments.tools.definitions import NativeToolCall

    tc = NativeToolCall.from_openai_format(
        {"id": "call_abc", "function": {"name": "calculate", "arguments": '{"expression": "1 + 1"}'}}
    )
    assert tc.id == "call_abc"
    assert tc.name == "calculate"
    assert tc.arguments == {"expression": "1 + 1"}


def test_react_parse_thought_action():
    """Test parsing ReAct output with thought and action."""
    from src.environments.envs.protocols.react import parse_react_output

    text = """Thought: I need to calculate the sum.
Action: calculate(expression="2 + 2")"""

    step = parse_react_output(text)
    assert step.thought == "I need to calculate the sum."
    assert step.action == "calculate"
    assert step.action_args == {"expression": "2 + 2"}
    assert not step.has_final_answer


def test_react_parse_final_answer():
    """Test parsing ReAct output with final answer."""
    from src.environments.envs.protocols.react import parse_react_output

    text = """Thought: I now have the answer.
Final Answer: 42"""

    step = parse_react_output(text)
    assert step.thought == "I now have the answer."
    assert step.has_final_answer
    assert step.final_answer == "42"


def test_react_parse_json_action():
    """Test parsing ReAct output with JSON-format action."""
    from src.environments.envs.protocols.react import parse_react_output

    text = """Thought: Let me search for information.
Action: {"name": "web_search", "arguments": {"query": "python tutorials"}}"""

    step = parse_react_output(text)
    assert step.action == "web_search"
    assert step.action_args == {"query": "python tutorials"}


def test_react_environment_basic_flow():
    """Test ReActEnvironment basic reset/step/reward flow."""
    from src.environments.envs.protocols.react import ReActEnvironment
    from src.environments.tools.factories import create_native_math_tools

    registry = create_native_math_tools()
    env = ReActEnvironment(
        tool_registry=registry,
        max_turns=5,
        success_reward=1.0,
    )

    episode_ids, steps = env.reset(
        ["What is 25 * 4?"],
        [{"answer": "100"}],
    )
    assert len(episode_ids) == 1
    assert not steps[0].done

    action1 = """Thought: I need to multiply 25 by 4.
Action: calculate(expression="25 * 4")"""
    steps = env.step(episode_ids, [action1])
    assert not steps[0].done

    traj = env.get_trajectories(episode_ids)[0]
    assert traj.info["total_tool_calls"] == 1

    action2 = """Thought: The result is 100.
Final Answer: 100"""
    steps = env.step(episode_ids, [action2])
    assert steps[0].done

    traj = env.get_trajectories(episode_ids)[0]
    assert traj.info["completed"]
    assert traj.info["final_answer"] == "100"


def run(ctx) -> dict:
    """Run the mocked environment/trainer component checks."""
    log(f"\n{'=' * 70}")
    log("  Environmental GRPO Mock Test (DistributedAsyncEnvironmentalGRPOTrainer components)")
    log(f"  World size: {ctx.world_size}")
    log(f"  GPU: {torch.cuda.get_device_name(ctx.local_rank)}")
    log(f"{'=' * 70}")

    checks: dict[str, bool] = {}
    log("\n--- Public Surface Tests ---")
    record_check(checks, "BaseEnvironment surface", test_import_base_environment)
    record_check(checks, "ReAct + native-tool-use surface", test_react_and_native_tool_use_surface)

    log("\n--- Environment Registry Tests ---")
    record_check(checks, "Registry registration round-trip", test_registry_registration_round_trip)
    record_check(checks, "Registry list builtins", test_registry_list_builtins)
    record_check(checks, "Registry resolve react_math", test_registry_resolve_react_math)
    record_check(checks, "Registry resolve native_math", test_registry_resolve_native_math)
    record_check(checks, "Registry resolve unknown raises", test_registry_resolve_unknown_raises)

    log("\n--- Message/Trajectory Tests ---")
    record_check(checks, "Message creation", test_message_creation)
    record_check(checks, "Message to_dict", test_message_to_dict)
    record_check(checks, "Message from_dict", test_message_from_dict)
    record_check(checks, "Trajectory creation", test_trajectory_creation)
    record_check(checks, "Trajectory add messages", test_trajectory_add_messages)
    record_check(checks, "Trajectory rewards", test_trajectory_rewards)
    record_check(checks, "Trajectory get_conversation", test_trajectory_get_conversation)

    log("\n--- Mock Rollout Tests ---")
    record_check(checks, "Mock multi-turn trajectory", test_mock_multi_turn_trajectory)
    record_check(checks, "Mock truncated trajectory", test_mock_truncated_trajectory)

    log("\n--- Tool Registry Tests ---")
    record_check(checks, "NativeTool creation", test_native_tool_creation)
    record_check(checks, "NativeTool OpenAI schema", test_native_tool_openai_schema)
    record_check(checks, "NativeToolRegistry operations", test_native_tool_registry)
    record_check(checks, "Math tools factory", test_math_tools_factory)
    record_check(checks, "NativeToolCall parsing", test_native_tool_call_parsing)

    log("\n--- ReAct Parsing Tests ---")
    record_check(checks, "ReAct parse thought+action", test_react_parse_thought_action)
    record_check(checks, "ReAct parse final answer", test_react_parse_final_answer)
    record_check(checks, "ReAct parse JSON action", test_react_parse_json_action)
    record_check(checks, "ReAct environment basic flow", test_react_environment_basic_flow)

    return {"checks": checks}


main = gpu_test_main(min_world_size=1, prefix="environmental_grpo_mock", partial_state=False)(run)

if __name__ == "__main__":
    main()
