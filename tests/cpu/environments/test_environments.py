#!/usr/bin/env python
"""
Tests for environment classes without requiring vLLM or GPUs.

Run with:
    python tests/cpu/environments/test_environments.py

Tests cover:
- Base environment classes (Message, Trajectory, BaseEnvironment)
- Native tool use (NativeTool, NativeToolRegistry, NativeToolUseEnvironment)
- ReAct environment (parsing and execution)
"""

import asyncio
import sys

import pytest


@pytest.fixture
def allow_mock_search(monkeypatch):
    """Opt into the fabricated-results search backend for a test that BUILDS a mock-search tool.

    ``create_native_search_tools`` refuses an unselectable backend at construction, so every env
    named with ``search_backend="mock"`` needs the flag; the tests that exercise the per-call flag
    read itself set it inline instead.
    """
    monkeypatch.setenv("HALO_ALLOW_MOCK_SEARCH", "1")


def test_message_creation():
    """Test Message creation and conversion."""
    from src.environments.base import Message

    msg = Message(role="user", content="Hello")
    assert msg.role == "user"
    assert msg.content == "Hello"

    d = msg.to_dict()
    assert d["role"] == "user"
    assert d["content"] == "Hello"

    user_msg = Message.user("User content")
    assert user_msg.role == "user"

    assistant_msg = Message.assistant("Assistant content")
    assert assistant_msg.role == "assistant"

    system_msg = Message.system("System content")
    assert system_msg.role == "system"

    tool_msg = Message.tool("Result", "tool_123", "my_tool")
    assert tool_msg.role == "tool"
    assert tool_msg.tool_call_id == "tool_123"
    assert tool_msg.name == "my_tool"

    msg2 = Message.from_dict({"role": "user", "content": "Test"})
    assert msg2.role == "user"
    assert msg2.content == "Test"


def test_trajectory():
    """Test Trajectory class."""
    from src.environments.base import Message, Trajectory

    traj = Trajectory()

    assert traj.num_turns == 0
    assert traj.total_reward == 0.0
    assert not traj.done

    traj.add_message(Message.user("Hello"))
    assert len(traj.messages) == 1
    assert traj.num_turns == 0  # counts assistant messages only

    traj.add_message(Message.assistant("Hi there!"))
    assert len(traj.messages) == 2
    assert traj.num_turns == 1

    traj.add_reward(0.5)
    assert traj.total_reward == 0.5

    traj.add_reward(0.3)
    assert traj.total_reward == 0.8

    conv = traj.get_conversation()
    assert len(conv) == 2
    assert conv[0]["role"] == "user"
    assert conv[1]["role"] == "assistant"


class SimpleTestEnvironment:
    """Simple test environment implementation."""

    def __init__(self, max_turns: int = 5):
        from src.environments.base import BaseEnvironment, Message, Trajectory

        class TestEnv(BaseEnvironment):
            def _reset_single(self, prompt, context=None):
                traj = Trajectory()
                traj.add_message(Message.user(str(prompt)))
                traj.info["expected"] = (context or {}).get("answer", "42")
                return traj

            def _step_single(self, trajectory, action, context=None):
                done = "Final Answer:" in action
                reward = 0.1
                return trajectory, reward, done, False, {}

            def _compute_reward(self, trajectory, context=None):
                expected = trajectory.info.get("expected", "42")
                if trajectory.info.get("completed") and str(expected) in str(trajectory.messages[-1].content):
                    return trajectory.total_reward + 1.0
                return trajectory.total_reward

        self.env = TestEnv(max_turns=max_turns)

    def get_env(self):
        return self.env


def test_base_environment_reset():
    """Test BaseEnvironment reset."""
    test_env = SimpleTestEnvironment()
    env = test_env.get_env()

    prompts = ["What is 6 * 7?", "What is 2 + 2?"]
    contexts = [{"answer": "42"}, {"answer": "4"}]

    episode_ids, steps = env.reset(prompts, contexts)

    assert len(episode_ids) == 2
    assert len(steps) == 2

    assert steps[0].observation[0]["content"] == "What is 6 * 7?"
    assert not steps[0].done

    trajs = env.get_trajectories(episode_ids)
    assert len(trajs) == 2
    assert trajs[0].info["expected"] == "42"
    assert trajs[1].info["expected"] == "4"


def test_base_environment_step():
    """Test BaseEnvironment step."""
    test_env = SimpleTestEnvironment()
    env = test_env.get_env()

    prompts = ["What is 6 * 7?"]
    contexts = [{"answer": "42"}]

    episode_ids, steps = env.reset(prompts, contexts)
    episode_id = episode_ids[0]

    steps = env.step([episode_id], ["Let me think..."])
    assert len(steps) == 1
    assert not steps[0].done

    steps = env.step([episode_id], ["Final Answer: 42"])
    assert steps[0].done


def test_base_environment_max_turns():
    """Test max turns truncation."""
    test_env = SimpleTestEnvironment(max_turns=2)
    env = test_env.get_env()

    episode_ids, steps = env.reset(["Test"])
    episode_id = episode_ids[0]

    # max_turns=2, so the second step truncates.
    env.step([episode_id], ["Step 1"])
    steps = env.step([episode_id], ["Step 2"])

    assert steps[0].done
    assert steps[0].truncated


def test_environment_cleanup():
    """Test episode cleanup."""
    test_env = SimpleTestEnvironment()
    env = test_env.get_env()

    episode_ids, _ = env.reset(["Test"])

    trajs = env.get_trajectories(episode_ids)
    assert trajs[0] is not None

    env.step(episode_ids, ["Final Answer: done"])
    env.cleanup(episode_ids)

    trajs = env.get_trajectories(episode_ids)
    assert trajs[0] is None


def test_native_tool():
    """Test NativeTool class."""
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
    assert result == "8"

    schema = tool.to_openai_schema()
    assert schema["type"] == "function"
    assert schema["function"]["name"] == "add"
    assert "a" in schema["function"]["parameters"]["properties"]
    assert "b" in schema["function"]["parameters"]["properties"]


def test_native_tool_registry():
    """Test NativeToolRegistry class."""
    from src.environments.tools.definitions import NativeTool, NativeToolRegistry

    registry = NativeToolRegistry()

    tool1 = NativeTool(
        name="tool1",
        description="First tool",
        parameters=[],
        handler=lambda: "result1",
    )
    tool2 = NativeTool(
        name="tool2",
        description="Second tool",
        parameters=[],
        handler=lambda: "result2",
    )

    registry.register(tool1).register(tool2)

    assert len(registry) == 2
    assert registry.names() == ["tool1", "tool2"]
    assert registry.get("tool1") is tool1
    assert registry.get("nonexistent") is None

    tools = registry.to_openai_tools()
    assert len(tools) == 2


def test_create_native_math_tools():
    """Test math tools factory."""
    from src.environments.tools.factories import create_native_math_tools

    registry = create_native_math_tools()

    assert "calculate" in registry.names()

    calc_tool = registry.get("calculate")

    result = calc_tool.execute(expression="2 + 3 * 4")
    assert result == "14"

    result = calc_tool.execute(expression="sqrt(16)")
    assert result == "4.0"

    result = calc_tool.execute(expression="1/0")
    assert "Error" in result


def test_create_native_python_tools():
    """Test Python REPL tool."""
    from src.environments.tools.factories import create_native_python_tools

    registry = create_native_python_tools()
    python_tool = registry.get("python")

    result = python_tool.execute(code="2 ** 10")
    assert result == "1024"

    result = python_tool.execute(code="for i in range(3): print(i)")
    assert "0" in result and "1" in result and "2" in result


def test_create_native_search_tools(allow_mock_search):
    """Test search tools factory."""
    from src.environments.tools.factories import create_native_search_tools

    registry = create_native_search_tools(backend="mock")

    assert "web_search" in registry.names()
    assert len(registry) == 1

    search_tool = registry.get("web_search")
    result = search_tool.execute(query="Python programming")
    assert "Python programming" in result


def test_create_native_file_tools():
    """Test file system tools factory."""
    from src.environments.tools.factories import create_native_file_tools

    registry = create_native_file_tools()

    assert "list_files" in registry.names()
    assert "read_file" in registry.names()
    assert "write_file" in registry.names()
    assert len(registry) == 3

    list_tool = registry.get("list_files")
    result = list_tool.execute(directory="/home/user")
    assert "notes.txt" in result or "No files" not in result

    read_tool = registry.get("read_file")
    result = read_tool.execute(path="/home/user/notes.txt")
    assert "notes" in result.lower() or "Error" not in result or "My notes" in result

    write_tool = registry.get("write_file")
    result = write_tool.execute(path="/home/user/test.txt", content="Hello World")
    assert "Successfully" in result or "wrote" in result.lower()


def test_create_native_code_tools_named_repl():
    """Test the generic code-tool factory with a custom tool name (what a 'coding' preset covers)."""
    from src.environments.tools.factories import create_native_code_tools

    registry = create_native_code_tools(language="python", tool_name="python_repl")

    # tool_name overrides the canonical language name.
    assert "python_repl" in registry.names()
    assert len(registry) == 1

    python_repl = registry.get("python_repl")

    result = python_repl.execute(code="sum(range(10))")
    assert result == "45"

    # Multi-statement code forces the exec path.
    result = python_repl.execute(code="x = 5\nprint(x * 2)")
    assert "10" in result

    result = python_repl.execute(code="sqrt(144)")
    assert "12" in result


def test_create_all_native_tools():
    """Test all native tools factory (math + python + search + file)."""
    from src.environments.tools.factories import create_all_native_tools

    registry = create_all_native_tools()

    assert "calculate" in registry.names()
    assert "python" in registry.names()
    assert "web_search" in registry.names()
    assert "list_files" in registry.names()
    assert "read_file" in registry.names()
    assert "write_file" in registry.names()

    assert len(registry) >= 6


def test_native_tool_use_environment():
    """Test NativeToolUseEnvironment."""
    from src.environments.envs.protocols.native import NativeToolUseEnvironment
    from src.environments.tools.factories import create_native_math_tools

    registry = create_native_math_tools()
    env = NativeToolUseEnvironment(
        tool_registry=registry,
        max_turns=5,
        success_reward=1.0,
    )

    episode_ids, steps = env.reset(["Calculate 2 + 2"], [{"answer": "4"}])

    assert len(episode_ids) == 1
    assert len(steps[0].observation) > 0

    tool_calls = [{"id": "call_1", "function": {"name": "calculate", "arguments": '{"expression": "2 + 2"}'}}]

    steps = env.step(episode_ids, ["Using calculate tool"], [{"tool_calls": tool_calls}])

    traj = env.get_trajectories(episode_ids)[0]
    assert traj.info["total_tool_calls"] == 1
    assert traj.info["successful_tool_calls"] == 1


def test_parse_react_output_thought_action():
    """Test parsing ReAct output with thought and action."""
    from src.environments.envs.protocols.react import parse_react_output

    text = """Thought: I need to calculate 2 + 2
Action: calculate(expression="2 + 2")"""

    step = parse_react_output(text)

    assert step.thought == "I need to calculate 2 + 2"
    assert step.action == "calculate"
    assert step.action_args == {"expression": "2 + 2"}
    assert not step.has_final_answer


def test_parse_react_output_final_answer():
    """Test parsing ReAct output with final answer."""
    from src.environments.envs.protocols.react import parse_react_output

    text = """Thought: I have the answer now
Final Answer: 42"""

    step = parse_react_output(text)

    assert step.thought == "I have the answer now"
    assert step.has_final_answer
    assert step.final_answer == "42"


def test_parse_react_output_json_action():
    """Test parsing JSON format action."""
    from src.environments.envs.protocols.react import parse_react_output

    text = """Thought: Let me search
Action: {"name": "web_search", "arguments": {"query": "python tutorials"}}"""

    step = parse_react_output(text)

    assert step.action == "web_search"
    assert step.action_args == {"query": "python tutorials"}


def test_parse_react_output_malformed_json_degrades_instead_of_raising():
    """Model-authored JSON must not escape the parser.

    ``_parse_action`` reaching into ``data["function"]`` as an object raises AttributeError out of
    ``parse_react_output`` → ``env.step`` on a string there, and the Ray actor turns the whole
    episode into an errored zero. A malformed action must degrade to "no usable action" (or an
    unknown-tool call), which the protocol answers with a retry hint. The types matter too: a non-str
    name is unhashable in the registry lookup, and non-dict arguments break the ``**`` splat.
    """
    from src.environments.envs.protocols.react import parse_react_output

    malformed = [
        '{"function": "calculate"}',  # `function` a string, not an object
        '{"function": null, "name": null}',
        '{"name": {"nested": 1}, "arguments": {}}',
        '{"name": "calculate", "arguments": [1, 2]}',
    ]
    for action in malformed:
        step = parse_react_output(f"Thought: t\nAction: {action}")
        assert step.action is None or isinstance(step.action, str), action
        assert step.action_args is None or isinstance(step.action_args, dict), action


def test_parse_react_output_numeric_args():
    """Test parsing action with numeric arguments."""
    from src.environments.envs.protocols.react import parse_react_output

    text = """Thought: Calculate
Action: multiply(a=5, b=3.14)"""

    step = parse_react_output(text)

    assert step.action == "multiply"
    assert step.action_args["a"] == 5
    assert step.action_args["b"] == 3.14


def test_react_environment_basic():
    """Test ReActEnvironment basic flow."""
    from src.environments.envs.protocols.react import ReActEnvironment
    from src.environments.tools.factories import create_native_math_tools

    registry = create_native_math_tools()
    env = ReActEnvironment(
        tool_registry=registry,
        max_turns=5,
        success_reward=1.0,
    )

    episode_ids, steps = env.reset(["What is 25 * 4?"], [{"answer": "100"}])

    assert len(episode_ids) == 1

    action1 = """Thought: I need to multiply 25 by 4
Action: calculate(expression="25 * 4")"""

    steps = env.step(episode_ids, [action1])

    assert not steps[0].done
    traj = env.get_trajectories(episode_ids)[0]
    assert traj.info["total_tool_calls"] == 1

    action2 = """Thought: The result is 100
Final Answer: 100"""

    steps = env.step(episode_ids, [action2])

    assert steps[0].done
    traj = env.get_trajectories(episode_ids)[0]
    assert traj.info["completed"]
    assert traj.info["final_answer"] == "100"


def test_react_environment_reward():
    """Test ReActEnvironment reward computation."""
    from src.environments.envs.protocols.react import ReActEnvironment
    from src.environments.tools.factories import create_native_math_tools

    registry = create_native_math_tools()
    env = ReActEnvironment(
        tool_registry=registry,
        max_turns=5,
        success_reward=1.0,
        failure_reward=0.0,
    )

    episode_ids, _ = env.reset(["Test"], [{"answer": "42"}])

    action = """Thought: The answer is 42
Final Answer: 42"""

    env.step(episode_ids, [action])

    assert env.get_trajectories(episode_ids)[0].total_reward > 0


def test_react_environment_unknown_tool():
    """An invented tool name is answered with the registry's one unknown-tool wording.

    One wording for both protocols: a message built in the environment interpolates ``names()``
    itself, so the model reads a Python list repr (``Available: ['calculate']``) in registration
    order instead of the sorted prose the native protocol sends.
    """
    from src.environments.envs.protocols.react import ReActEnvironment
    from src.environments.tools.factories import create_native_math_tools

    registry = create_native_math_tools()
    env = ReActEnvironment(
        tool_registry=registry,
        max_turns=5,
        tool_error_penalty=0.1,
    )

    episode_ids, _ = env.reset(["Test"])

    action = """Thought: Use a fake tool
Action: fake_tool(arg="test")"""

    env.step(episode_ids, [action])

    traj = env.get_trajectories(episode_ids)[0]
    observations = traj.info.get("observations", [])
    assert len(observations) > 0
    assert "Unknown tool 'fake_tool'" in observations[0]
    assert f"Available tools: {', '.join(sorted(registry.names()))}" in observations[0]
    assert "[" not in observations[0], f"tool list rendered as a repr, not prose: {observations[0]}"


def test_react_tool_that_raises_is_logged_and_charged(caplog):
    """A ReAct tool that raises must reach the logs, not only the model's observation.

    Without the log the episode just scores ``failure_reward`` with nothing anywhere saying why: the
    observation is trained on, not read by an operator, and a submit/grading handler dying on a
    malformed payload looks exactly like a model that used the tool wrong.
    """
    import logging

    from src.environments.envs.protocols.react import ReActEnvironment
    from src.environments.tools.definitions import NativeTool, NativeToolRegistry, ToolParameter

    def _explode(**_kwargs):
        raise RuntimeError("tool backend is down")

    registry = NativeToolRegistry()
    registry.register(
        NativeTool(
            name="calculate",
            description="calculate",
            parameters=[ToolParameter("expression", "string", "expr")],
            handler=_explode,
        )
    )
    env = ReActEnvironment(tool_registry=registry, max_turns=5, tool_error_penalty=0.1, thought_reward=0.0)
    episode_ids, _ = env.reset(["Test"])

    with caplog.at_level(logging.WARNING, logger="src.environments.envs.protocols.react"):
        steps = env.step(episode_ids, ['Thought: compute\nAction: calculate(expression="1+1")'])

    assert any("raised during execution" in record.getMessage() for record in caplog.records), (
        "a raising ReAct tool must be logged; the observation alone never reaches an operator"
    )
    assert any(record.exc_info for record in caplog.records), "the traceback is what localizes the fault"
    assert steps[0].reward == pytest.approx(-0.1)


def test_react_math_factory():
    """ReAct advertises its tools in the system prompt, never as a native ``tools=`` schema: the
    action is parsed out of plain text, so a server-side parser would strip it into a burnt turn."""
    from src.environments.envs.protocols.react import create_react_math_environment

    env = create_react_math_environment(max_turns=10, success_reward=2.0)

    assert env.get_tools_schema() is None
    assert {"calculate", "python"}.issubset(env.registry.names())
    assert "calculate" in env.system_prompt and "python" in env.system_prompt


def test_rollout_manager_round_robin():
    """Test RolloutManager round-robin actor selection logic."""
    from src.environments.ray_actors import RolloutConfig, RolloutManager

    manager = RolloutManager(
        num_workers=2,
        env_type="react_math",
        env_config={},
        server_urls=["http://server1:8000", "http://server2:8000", "http://server3:8000"],
        rollout_config=RolloutConfig(),
    )

    # Stand-in actors: the real ones would need a live Ray cluster.
    manager._actors = ["actor0", "actor1"]
    manager._started = True

    actors = [manager._next_actor() for _ in range(4)]
    assert actors == ["actor0", "actor1", "actor0", "actor1"]


def test_create_environment_class_tuple():
    """Test create_environment with (class, kwargs) tuple."""
    from src.environments.envs.protocols.react import ReActEnvironment
    from src.environments.registry import create_environment
    from src.environments.tools.factories import create_native_math_tools

    registry = create_native_math_tools()
    env = create_environment(
        (ReActEnvironment, {"tool_registry": registry, "max_turns": 5}),
        {},
    )
    assert env is not None
    assert isinstance(env, ReActEnvironment)


def test_registry_resolve_unknown():
    """Test resolve_environment raises for unknown type."""
    from src.environments.registry import resolve_environment

    try:
        resolve_environment("nonexistent_env", {})
        raise AssertionError("Should raise ValueError")
    except ValueError as e:
        assert "Unknown environment type" in str(e)
        assert "nonexistent_env" in str(e)


def test_registry_forwards_environment_kwargs():
    """environment_kwargs must reach the env — a dropped sandbox_backend is a silent isolation
    downgrade, a dropped max_submissions silently keeps the default contest budget."""
    from src.environments.registry import resolve_environment

    env = resolve_environment(
        "code_contests",
        {"sandbox_backend": "local", "max_submissions": 3, "max_test_calls": 7, "max_turns": 9},
    )
    assert env.max_submissions == 3
    assert env.max_test_calls == 7
    assert env.max_turns == 9

    # The codeforces preset keeps its token-compare default while forwarding the extras.
    cf = resolve_environment("codeforces", {"max_submissions": 5})
    assert cf.grading_spec.comparison == "tokens"
    assert cf.max_submissions == 5


def test_registry_register_custom(isolated_registry):
    """Test registering a custom environment type."""
    from src.environments.base import BaseEnvironment, Message, Trajectory
    from src.environments.registry import register_environment, resolve_environment

    class TestCustomEnv(BaseEnvironment):
        def _reset_single(self, prompt, context=None):
            traj = Trajectory()
            traj.add_message(Message.user(str(prompt)))
            return traj

        def _step_single(self, trajectory, action, context=None):
            return trajectory, 0.0, True, False, {}

        def _compute_reward(self, trajectory, context=None):
            return 1.0

    def factory(config):
        return TestCustomEnv(max_turns=config.get("max_turns", 5))

    register_environment("test_custom_env", factory)

    env = resolve_environment("test_custom_env", {"max_turns": 3})
    assert env is not None
    assert isinstance(env, TestCustomEnv)


def test_registry_override(isolated_registry):
    """Test overriding a registered environment type."""
    from src.environments.registry import register_environment, resolve_environment

    def factory1(config):
        return "factory1"

    def factory2(config):
        return "factory2"

    register_environment("test_override_env", factory1)
    assert resolve_environment("test_override_env", {}) == "factory1"

    # Must raise without override=True, and leave the original factory in place.
    try:
        register_environment("test_override_env", factory2)
        raise AssertionError("Should raise ValueError")
    except ValueError:
        pass
    assert resolve_environment("test_override_env", {}) == "factory1", "a rejected register still replaced"

    # override=True must actually REPLACE the factory, not silently keep the first one.
    register_environment("test_override_env", factory2, override=True)
    assert resolve_environment("test_override_env", {}) == "factory2"


def test_registry_resolve_swe():
    """Test resolve_environment for 'swe' type (SweEnvironment)."""
    from src.environments.envs.tasks.coding.swe import SweEnvironment
    from src.environments.registry import resolve_environment

    env = resolve_environment("swe", {"max_turns": 5})
    assert env is not None
    assert isinstance(env, SweEnvironment)


def test_code_environment_init():
    """Test SweEnvironment initialization."""
    from src.environments.envs.tasks.coding.swe import SweEnvironment

    env = SweEnvironment(max_turns=10, success_reward=2.0)

    tools_schema = env.get_tools_schema()
    tool_names = [t["function"]["name"] for t in tools_schema]

    assert "list_files" in tool_names
    assert "read_file" in tool_names
    assert "write_file" in tool_names
    assert "run_code" in tool_names


def test_code_environment_reset_step():
    """Test SweEnvironment reset and step."""
    from src.environments.envs.tasks.coding.swe import SweEnvironment

    env = SweEnvironment(max_turns=5, success_reward=1.0)

    episode_ids, steps = env.reset(["Fix the bug in the code"], [{"answer": "fixed"}])

    assert len(episode_ids) == 1
    assert len(steps[0].observation) > 0

    # run_code executes in the episode's persistent workspace.
    tool_calls = [{"id": "call_1", "function": {"name": "run_code", "arguments": '{"code": "print(42)"}'}}]

    steps = env.step(episode_ids, ["Running code..."], [{"tool_calls": tool_calls}])
    assert not steps[0].done

    traj = env.get_trajectories(episode_ids)[0]
    assert traj.info["total_tool_calls"] == 1


async def test_async_base_environment():
    """Test AsyncBaseEnvironment."""
    from src.environments.base import AsyncBaseEnvironment, Message, Trajectory

    class TestAsyncEnv(AsyncBaseEnvironment):
        def _reset_single(self, prompt, context=None):
            traj = Trajectory()
            traj.add_message(Message.user(str(prompt)))
            return traj

        def _step_single(self, trajectory, action, context=None):
            done = "done" in action.lower()
            return trajectory, 0.1, done, False, {}

        def _compute_reward(self, trajectory, context=None):
            return trajectory.total_reward + 1.0

        async def _reset_single_async(self, prompt, context=None):
            await asyncio.sleep(0.01)
            return self._reset_single(prompt, context)

        async def _step_single_async(self, trajectory, action, context=None):
            await asyncio.sleep(0.01)
            return self._step_single(trajectory, action, context)

    env = TestAsyncEnv(max_turns=5)

    episode_ids, steps = await env.reset_async(["Test1", "Test2"])

    assert len(episode_ids) == 2
    assert len(steps) == 2

    steps = await env.step_async(episode_ids, ["action1", "done"])

    assert not steps[0].done
    assert steps[1].done


async def test_async_native_tool_use():
    """Test AsyncNativeToolUseEnvironment."""
    from src.environments.envs.protocols.native import AsyncNativeToolUseEnvironment
    from src.environments.tools.definitions import NativeTool, NativeToolRegistry, ToolParameter

    async def async_handler(query: str) -> str:
        await asyncio.sleep(0.01)
        return f"Result for: {query}"

    registry = NativeToolRegistry()
    registry.register(
        NativeTool(
            name="async_search",
            description="Async search tool",
            parameters=[ToolParameter("query", "string", "Search query")],
            async_handler=async_handler,
        )
    )

    env = AsyncNativeToolUseEnvironment(
        tool_registry=registry,
        max_turns=5,
    )

    episode_ids, steps = await env.reset_async(["Search for something"])

    tool_calls = [{"id": "call_1", "function": {"name": "async_search", "arguments": '{"query": "test"}'}}]

    await env.step_async(episode_ids, ["Using search"], [{"tool_calls": tool_calls}])

    traj = env.get_trajectories(episode_ids)[0]
    assert traj.info["total_tool_calls"] == 1


def test_mcp_server_presets():
    """Test MCP server preset configurations."""
    from src.environments.envs.protocols.mcp import MCP_SERVERS

    expected_servers = ["brave_search", "filesystem", "fetch", "memory", "github", "slack"]
    for server_name in expected_servers:
        assert server_name in MCP_SERVERS, f"Missing MCP server: {server_name}"

    for name, config in MCP_SERVERS.items():
        assert "command" in config, f"Missing 'command' in {name}"
        assert "args" in config, f"Missing 'args' in {name}"
        assert "env" in config, f"Missing 'env' in {name}"
        assert "description" in config, f"Missing 'description' in {name}"

        assert isinstance(config["args"], list), f"args should be list in {name}"
        assert isinstance(config["env"], list), f"env should be list in {name}"


def test_get_mcp_server_config():
    """Test get_mcp_server_config function."""
    from src.environments.envs.protocols.mcp import get_mcp_server_config

    config = get_mcp_server_config("filesystem")
    assert config["command"] == "npx"
    assert "@modelcontextprotocol/server-filesystem" in " ".join(config["args"])

    config = get_mcp_server_config("brave_search")
    assert "BRAVE_API_KEY" in config["env"]

    config = get_mcp_server_config("github")
    assert "GITHUB_TOKEN" in config["env"]

    try:
        get_mcp_server_config("nonexistent_server")
        raise AssertionError("Should raise ValueError for unknown server")
    except ValueError as e:
        assert "Unknown MCP server" in str(e)
        assert "nonexistent_server" in str(e)


def test_create_native_mcp_environment():
    """Test create_native_mcp_environment factory."""
    from src.environments.envs.protocols.mcp import create_native_mcp_environment

    env = create_native_mcp_environment(
        "filesystem",
        max_turns=10,
        success_reward=1.5,
    )

    assert env.server_command == "npx"
    assert env.max_turns == 10
    assert env.success_reward == 1.5

    env2 = create_native_mcp_environment(
        "brave_search",
        env_vars={"BRAVE_API_KEY": "test-key"},
        max_turns=5,
    )

    assert env2.server_env == {"BRAVE_API_KEY": "test-key"}


def test_native_mcp_client_environment_init():
    """Test NativeMCPClientEnvironment initialization."""
    from src.environments.envs.protocols.mcp import NativeMCPClientEnvironment

    env = NativeMCPClientEnvironment(
        server_command="npx",
        server_args=["-y", "@modelcontextprotocol/server-memory"],
        max_turns=10,
        success_reward=2.0,
        tool_success_reward=0.2,
    )

    assert env.transport == "stdio"
    assert env.server_command == "npx"
    assert env.max_turns == 10
    assert env.success_reward == 2.0
    assert env.tool_success_reward == 0.2

    env2 = NativeMCPClientEnvironment(
        server_url="http://localhost:8080/sse",
        transport="sse",
        max_turns=5,
    )

    assert env2.transport == "sse"
    assert env2.server_url == "http://localhost:8080/sse"


def test_multi_tool_environment_with_all_tools():
    """Test NativeToolUseEnvironment with all tools."""
    from src.environments.envs.protocols.native import NativeToolUseEnvironment
    from src.environments.tools.factories import create_all_native_tools

    registry = create_all_native_tools()
    env = NativeToolUseEnvironment(
        tool_registry=registry,
        max_turns=10,
        success_reward=1.0,
    )

    tools_schema = env.get_tools_schema()
    assert len(tools_schema) >= 6

    tool_names = [t["function"]["name"] for t in tools_schema]
    assert "calculate" in tool_names
    assert "python" in tool_names
    assert "web_search" in tool_names
    assert "list_files" in tool_names

    episode_ids, steps = env.reset(["Use various tools to help me"], [{"answer": "done"}])

    assert len(episode_ids) == 1

    tool_calls1 = [{"id": "call_1", "function": {"name": "calculate", "arguments": '{"expression": "2 * 3 + 4"}'}}]
    steps = env.step(episode_ids, ["Calculating..."], [{"tool_calls": tool_calls1}])
    assert not steps[0].done

    tool_calls2 = [{"id": "call_2", "function": {"name": "python", "arguments": '{"code": "sum([1,2,3,4,5])"}'}}]
    steps = env.step(episode_ids, ["Running Python..."], [{"tool_calls": tool_calls2}])
    assert not steps[0].done

    tool_calls3 = [{"id": "call_3", "function": {"name": "web_search", "arguments": '{"query": "Python tutorials"}'}}]
    steps = env.step(episode_ids, ["Searching..."], [{"tool_calls": tool_calls3}])
    assert not steps[0].done

    traj = env.get_trajectories(episode_ids)[0]
    assert traj.info["total_tool_calls"] == 3
    assert traj.info["successful_tool_calls"] == 3


def test_multi_tool_parallel_calls():
    """Test environment with multiple parallel tool calls in one step."""
    from src.environments.envs.protocols.native import NativeToolUseEnvironment
    from src.environments.tools.factories import create_all_native_tools

    registry = create_all_native_tools()
    env = NativeToolUseEnvironment(
        tool_registry=registry,
        max_turns=10,
        max_tool_calls_per_turn=5,
    )

    episode_ids, _ = env.reset(["Multi-tool task"])

    tool_calls = [
        {"id": "call_a", "function": {"name": "calculate", "arguments": '{"expression": "10 + 20"}'}},
        {"id": "call_b", "function": {"name": "python", "arguments": '{"code": "2 ** 8"}'}},
        {"id": "call_c", "function": {"name": "web_search", "arguments": '{"query": "math"}'}},
    ]

    env.step(episode_ids, ["Using multiple tools..."], [{"tool_calls": tool_calls}])

    traj = env.get_trajectories(episode_ids)[0]
    assert traj.info["total_tool_calls"] == 3
    assert traj.info["successful_tool_calls"] == 3

    tool_results = traj.info["tool_results"]
    assert len(tool_results) == 3
    assert all(r["success"] for r in tool_results)


def test_tool_error_handling():
    """Test error handling when tool execution fails."""
    from src.environments.envs.protocols.native import NativeToolUseEnvironment
    from src.environments.tools.definitions import NativeTool, NativeToolRegistry, ToolParameter

    def failing_tool(param: str) -> str:
        raise ValueError("Tool execution failed!")

    registry = NativeToolRegistry()
    registry.register(
        NativeTool(
            name="failing_tool",
            description="A tool that always fails",
            parameters=[ToolParameter("param", "string", "Input parameter")],
            handler=failing_tool,
        )
    )

    env = NativeToolUseEnvironment(
        tool_registry=registry,
        max_turns=5,
        tool_error_penalty=0.5,
    )

    episode_ids, _ = env.reset(["Test failure"])

    tool_calls = [{"id": "call_1", "function": {"name": "failing_tool", "arguments": '{"param": "test"}'}}]

    env.step(episode_ids, ["Using tool..."], [{"tool_calls": tool_calls}])

    traj = env.get_trajectories(episode_ids)[0]
    assert traj.info["total_tool_calls"] == 1
    assert traj.info["successful_tool_calls"] == 0

    tool_results = traj.info["tool_results"]
    assert len(tool_results) == 1
    assert tool_results[0]["success"] is False
    assert "Error" in tool_results[0]["content"]


def test_unknown_tool_handling():
    """Test handling of calls to unknown tools."""
    from src.environments.envs.protocols.native import NativeToolUseEnvironment
    from src.environments.tools.factories import create_native_math_tools

    registry = create_native_math_tools()
    env = NativeToolUseEnvironment(
        tool_registry=registry,
        max_turns=5,
        tool_error_penalty=0.2,
    )

    episode_ids, _ = env.reset(["Test unknown tool"])

    tool_calls = [{"id": "call_1", "function": {"name": "nonexistent_tool", "arguments": '{"x": 1}'}}]

    env.step(episode_ids, ["Using unknown tool..."], [{"tool_calls": tool_calls}])

    traj = env.get_trajectories(episode_ids)[0]
    tool_results = traj.info["tool_results"]

    assert len(tool_results) == 1
    assert tool_results[0]["success"] is False
    assert "Unknown tool" in tool_results[0]["content"] or "not found" in tool_results[0]["content"].lower()


def test_react_search_environment():
    """Test create_react_search_environment factory."""
    from src.environments.envs.protocols.react import create_react_search_environment

    env = create_react_search_environment(max_turns=8, success_reward=1.5)

    assert env.get_tools_schema() is None
    assert "web_search" in env.registry.names()
    assert "web_search" in env.system_prompt

    episode_ids, _ = env.reset(["What is the capital of France?"], [{"answer": "Paris"}])

    action1 = """Thought: I need to search for this information
Action: web_search(query="capital of France")"""

    steps = env.step(episode_ids, [action1])
    assert not steps[0].done

    traj = env.get_trajectories(episode_ids)[0]
    assert traj.info["total_tool_calls"] == 1

    action2 = """Thought: I found the answer
Final Answer: Paris"""

    steps = env.step(episode_ids, [action2])
    assert steps[0].done
    assert traj.info["final_answer"] == "Paris"


def test_react_environment_with_python_tool():
    """Test ReAct environment using Python tool for complex calculations."""
    from src.environments.envs.protocols.react import create_react_math_environment

    env = create_react_math_environment(max_turns=10)

    episode_ids, _ = env.reset(["Calculate the factorial of 10"], [{"answer": "3628800"}])

    action1 = """Thought: I'll use Python to calculate factorial
Action: python(code="factorial(10)")"""

    steps = env.step(episode_ids, [action1])
    assert not steps[0].done

    traj = env.get_trajectories(episode_ids)[0]
    assert traj.info["total_tool_calls"] == 1

    observations = traj.info.get("observations", [])
    assert len(observations) > 0
    assert "3628800" in observations[-1]


def test_create_environment_native_coding():
    """Test create_environment for native_coding type."""
    from src.environments.registry import create_environment

    env = create_environment("native_coding", {"max_turns": 5})
    assert env is not None

    tools_schema = env.get_tools_schema()
    tool_names = [t["function"]["name"] for t in tools_schema]
    assert "python_repl" in tool_names


def test_create_environment_native_combined():
    """Test create_environment for native_combined type."""
    from src.environments.registry import create_environment

    env = create_environment("native_combined", {"max_turns": 10})
    assert env is not None

    tools_schema = env.get_tools_schema()
    tool_names = [t["function"]["name"] for t in tools_schema]

    assert "calculate" in tool_names
    assert "python" in tool_names
    assert "web_search" in tool_names


def test_create_environment_react_search():
    """Test create_environment for react_search type."""
    from src.environments.registry import create_environment

    env = create_environment("react_search", {"max_turns": 5})
    assert env is not None

    # ReAct names its tools in the system prompt; it advertises no native tools= schema.
    assert env.get_tools_schema() is None
    assert "web_search" in env.registry.names()


def test_tool_openai_schema_complex():
    """Test complex tool schema generation."""
    from src.environments.tools.definitions import NativeTool, ToolParameter

    tool = NativeTool(
        name="complex_tool",
        description="A tool with multiple parameter types",
        parameters=[
            ToolParameter("required_string", "string", "A required string", required=True),
            ToolParameter("optional_number", "number", "An optional number", required=False),
            ToolParameter("enum_param", "string", "Choose one", enum=["option1", "option2", "option3"]),
        ],
        handler=lambda **kwargs: str(kwargs),
    )

    schema = tool.to_openai_schema()

    assert schema["type"] == "function"
    assert schema["function"]["name"] == "complex_tool"

    props = schema["function"]["parameters"]["properties"]
    assert "required_string" in props
    assert "optional_number" in props
    assert "enum_param" in props

    assert props["enum_param"]["enum"] == ["option1", "option2", "option3"]

    required = schema["function"]["parameters"]["required"]
    assert "required_string" in required
    assert "enum_param" in required  # enum with required=True by default
    assert "optional_number" not in required


# Test: Utility Functions


# Test: Async Multi-Tool Scenarios


async def test_async_concurrent_tool_execution():
    """Test concurrent tool execution in async environment."""
    from src.environments.envs.protocols.native import AsyncNativeToolUseEnvironment
    from src.environments.tools.definitions import NativeTool, NativeToolRegistry, ToolParameter

    # Create tools with varying delays
    async def slow_tool(x: str) -> str:
        await asyncio.sleep(0.1)
        return f"Slow: {x}"

    async def fast_tool(x: str) -> str:
        await asyncio.sleep(0.01)
        return f"Fast: {x}"

    registry = NativeToolRegistry()
    registry.register(
        NativeTool(
            name="slow_tool",
            description="A slow tool",
            parameters=[ToolParameter("x", "string", "Input")],
            async_handler=slow_tool,
        )
    )
    registry.register(
        NativeTool(
            name="fast_tool",
            description="A fast tool",
            parameters=[ToolParameter("x", "string", "Input")],
            async_handler=fast_tool,
        )
    )

    env = AsyncNativeToolUseEnvironment(
        tool_registry=registry,
        max_turns=5,
        max_tool_calls_per_turn=5,
    )

    episode_ids, _ = await env.reset_async(["Test concurrent"])

    # Call both tools concurrently
    import time

    start = time.time()

    tool_calls = [
        {"id": "call_1", "function": {"name": "slow_tool", "arguments": '{"x": "a"}'}},
        {"id": "call_2", "function": {"name": "fast_tool", "arguments": '{"x": "b"}'}},
        {"id": "call_3", "function": {"name": "slow_tool", "arguments": '{"x": "c"}'}},
    ]

    await env.step_async(episode_ids, ["Concurrent tools"], [{"tool_calls": tool_calls}])

    elapsed = time.time() - start

    # Should complete faster than sequential (3 * 0.1 = 0.3s)
    # With concurrent execution, should be closer to 0.1s
    assert elapsed < 0.25, f"Expected concurrent execution, but took {elapsed}s"

    traj = env.get_trajectories(episode_ids)[0]
    assert traj.info["total_tool_calls"] == 3
    assert traj.info["successful_tool_calls"] == 3


# Web Search Module


def test_web_search_mock_backend(monkeypatch):
    """Test web_search with mock backend (no network)."""
    from src.environments.tools.web_search import web_search, web_search_raw

    monkeypatch.setenv("HALO_ALLOW_MOCK_SEARCH", "1")

    # Formatted output
    result = web_search("quantum computing", backend="mock")
    assert "quantum computing" in result.lower()
    assert isinstance(result, str)
    assert len(result) > 0

    # Raw output
    raw = web_search_raw("quantum computing", max_results=2, backend="mock")
    assert isinstance(raw, list)
    assert len(raw) == 2
    for r in raw:
        assert "title" in r
        assert "url" in r
        assert "snippet" in r


async def test_web_search_async_mock(monkeypatch):
    """Test async_web_search with mock backend."""
    from src.environments.tools.web_search import async_web_search

    monkeypatch.setenv("HALO_ALLOW_MOCK_SEARCH", "1")

    result = await async_web_search("machine learning", backend="mock")
    assert "machine learning" in result.lower()
    assert isinstance(result, str)


def test_web_search_raises_on_backend_failure_no_mock_fabrication():
    """A failing real backend must raise — never silently return fabricated mock results.

    Returning _search_mock() output on error fed the model invented evidence and scored a
    tool success in the RL envs. The error must propagate so the tool layer records a real
    failure.
    """
    import dataclasses

    from src.environments.tools import web_search as ws

    def _boom(**kwargs):
        raise RuntimeError("network down")

    original = ws._BACKENDS["serper"]
    ws._BACKENDS["serper"] = dataclasses.replace(original, sync=_boom)
    try:
        with pytest.raises(RuntimeError):
            ws.web_search("anything", backend="serper")
        with pytest.raises(RuntimeError):
            ws.web_search_raw("anything", backend="serper")
    finally:
        ws._BACKENDS["serper"] = original


def test_web_search_format_results():
    """Test result formatting."""
    from src.environments.tools.web_search import _format_results

    # Empty results
    assert _format_results([]) == "No results found."

    # Normal results
    results = [
        {"title": "Title 1", "url": "https://example.com/1", "snippet": "Snippet 1"},
        {"title": "Title 2", "url": "https://example.com/2", "snippet": "Snippet 2"},
    ]
    formatted = _format_results(results, max_results=5)
    assert "Title 1" in formatted
    assert "Title 2" in formatted
    assert "https://example.com/1" in formatted

    # Respects max_results
    formatted_1 = _format_results(results, max_results=1)
    assert "Title 1" in formatted_1
    assert "Title 2" not in formatted_1


def test_web_search_invalid_backend():
    """Test error on invalid backend."""
    from src.environments.tools.web_search import web_search

    try:
        web_search("test", backend="nonexistent")
        raise AssertionError("Should have raised ValueError")
    except ValueError as e:
        assert "nonexistent" in str(e)


def test_mock_search_backend_is_not_selectable_from_a_config(monkeypatch):
    """Fabricated snippets pay ``tool_success_reward`` like a real search, so a training YAML naming
    ``mock`` must fail loudly instead of teaching the policy that invented evidence works."""
    from src.environments.tools.web_search import web_search

    monkeypatch.delenv("HALO_ALLOW_MOCK_SEARCH", raising=False)
    with pytest.raises(ValueError, match="HALO_ALLOW_MOCK_SEARCH"):
        web_search("test", backend="mock")


def test_mock_search_backend_is_available_to_an_opted_in_demo(monkeypatch):
    from src.environments.tools.web_search import web_search

    monkeypatch.setenv("HALO_ALLOW_MOCK_SEARCH", "1")
    assert "Wikipedia: pytest" in web_search("pytest", backend="mock")


def test_an_unselectable_search_backend_is_refused_when_the_ENV_IS_BUILT(monkeypatch):
    """Construction time, not first-search time.

    Every protocol wraps tool execution in ``except Exception`` → observation, so a gate that only
    fires inside the handler lets the whole run proceed with every ``web_search`` returning an error
    and charging ``tool_error_penalty`` — a broken environment measured to completion instead of a
    refused config.
    """
    from src.environments.envs.tasks.qa import create_qa_search_environment
    from src.environments.tools.factories import create_native_search_tools

    monkeypatch.delenv("HALO_ALLOW_MOCK_SEARCH", raising=False)
    with pytest.raises(ValueError, match="HALO_ALLOW_MOCK_SEARCH"):
        create_native_search_tools(backend="mock")
    with pytest.raises(ValueError, match="Unknown search backend: nonexistent"):
        create_qa_search_environment(max_turns=3, search_backend="nonexistent")


# Shared Reward Utilities


def test_normalize_text():
    """Test text normalization."""
    from src.environments.rewards import normalize_text

    assert normalize_text("  The Answer is 42  ") == "42"
    assert normalize_text("\\boxed{42}") == "42"
    assert normalize_text("**Paris**") == "paris"
    assert normalize_text("Therefore, 100.") == "100"
    assert normalize_text("HELLO") == "hello"


def test_exact_match():
    """Test exact match."""
    from src.environments.rewards import exact_match

    assert exact_match("Paris", "paris")
    assert exact_match("42", "42")
    assert exact_match("  Hello World  ", "hello world")
    assert not exact_match("Paris", "London")


def test_numeric_match():
    """Test numeric match with tolerance."""
    from src.environments.rewards import numeric_match

    assert numeric_match("42", "42")
    assert numeric_match("42.001", "42", rtol=0.01)
    assert numeric_match("3.14159", "3.14159")
    assert not numeric_match("42", "43")
    assert numeric_match("50%", "0.5")
    assert numeric_match("The answer is 110", "110")
    assert not numeric_match("no numbers here", "42")


def test_multiple_choice_match():
    """Test multiple-choice answer extraction."""
    from src.environments.envs.tasks.qa import multiple_choice_match

    assert multiple_choice_match("The answer is B", "B")
    assert multiple_choice_match("(A)", "A")
    assert multiple_choice_match("C. Saturn", "C")
    assert multiple_choice_match("A", "A")
    assert not multiple_choice_match("A", "B")
    assert multiple_choice_match("I think the answer is (D)", "D")
    assert multiple_choice_match("B) Jupiter", "B")

    # MMLU-Pro uses A-J (10 choices)
    assert multiple_choice_match("The answer is F", "F")
    assert multiple_choice_match("(G)", "G")
    assert multiple_choice_match("H. Some option", "H")
    assert multiple_choice_match("J", "J")
    assert not multiple_choice_match("F", "G")

    # Reward inflation guard: prose that merely *starts* with the expected letter must NOT
    # score as that choice (the removed startswith fallback granted full reward here).
    assert not multiple_choice_match("Although I'm not sure, the reasoning is complex", "A")
    assert not multiple_choice_match("Based on the above, several options apply", "B")


def test_validate_answer():
    """Test composite validation."""
    from src.environments.rewards import validate_answer

    assert validate_answer("42", "42")
    assert not validate_answer("wrong", "42")

    # Numeric match
    assert validate_answer("The result is 110", "110")

    # Substring containment is not in the default chain: it must not inflate the score
    # (an expected "7" inside "17" would otherwise score a full match).
    assert not validate_answer("17", "7")

    # A custom chain replaces the default one rather than extending it.
    assert validate_answer("17", "7", methods=[lambda p, e: e in p])


def test_compute_answer_reward():
    """Test reward computation."""
    from src.environments.rewards import compute_answer_reward

    # Correct answer
    assert compute_answer_reward("42", "42", 1.0, 0.0) == 1.0
    # Wrong answer
    assert compute_answer_reward("wrong", "42", 1.0, 0.0) == 0.0
    # Grading is all-or-nothing: a near-miss earns the failure reward, never a similarity-scaled one.
    assert compute_answer_reward("Leonardo da Vinchi", "Leonardo da Vinci", 1.0, 0.0) == 0.0


# NativeToolUse Reward Fix


def test_native_tool_use_reward_with_answer():
    """Test NativeToolUseEnvironment validates answers from context."""
    from src.environments.envs.protocols.native import NativeToolUseEnvironment
    from src.environments.tools.factories import create_native_math_tools

    registry = create_native_math_tools()
    env = NativeToolUseEnvironment(
        tool_registry=registry,
        max_turns=5,
        success_reward=1.0,
        failure_reward=0.0,
    )

    # Reset with answer context
    episode_ids, steps = env.reset(
        ["What is 2 + 2?"],
        [{"answer": "4"}],
    )

    # Simulate model response (no tool calls = done)
    env.step(episode_ids, ["The answer is 4"])
    traj = env.get_trajectories(episode_ids)[0]
    assert traj.done
    # Should get success_reward because answer matches
    assert traj.total_reward >= 1.0

    env.cleanup(episode_ids)


def test_native_tool_use_reward_wrong_answer():
    """Test NativeToolUseEnvironment gives failure for wrong answer."""
    from src.environments.envs.protocols.native import NativeToolUseEnvironment
    from src.environments.tools.factories import create_native_math_tools

    registry = create_native_math_tools()
    env = NativeToolUseEnvironment(
        tool_registry=registry,
        max_turns=5,
        success_reward=1.0,
        failure_reward=0.0,
    )

    episode_ids, steps = env.reset(
        ["What is 2 + 2?"],
        [{"answer": "4"}],
    )

    env.step(episode_ids, ["The answer is 7"])
    traj = env.get_trajectories(episode_ids)[0]
    assert traj.done
    # Should get failure_reward because answer is wrong
    assert traj.total_reward < 1.0

    env.cleanup(episode_ids)


def test_native_tool_use_reward_no_answer():
    """Test backward compatibility when no answer in context."""
    from src.environments.envs.protocols.native import NativeToolUseEnvironment
    from src.environments.tools.factories import create_native_math_tools

    registry = create_native_math_tools()
    env = NativeToolUseEnvironment(
        tool_registry=registry,
        max_turns=5,
        success_reward=1.0,
    )

    # No answer in context — should still give success_reward
    episode_ids, steps = env.reset(["Do something"], [{}])
    env.step(episode_ids, ["Done!"])
    traj = env.get_trajectories(episode_ids)[0]
    assert traj.total_reward >= 1.0

    env.cleanup(episode_ids)


def test_native_tool_use_null_answer_is_invalid_not_a_free_success():
    """A row that IS answer-graded but whose ``answer`` cell is null must not take the completion
    fallback above: that pays full success_reward to any episode that merely finished, and every
    sibling in its GRPO group finishes just as easily, so the whole group learns nothing but "stop"."""
    from src.environments.envs.protocols.native import NativeToolUseEnvironment
    from src.environments.tools.factories import create_native_math_tools

    env = NativeToolUseEnvironment(
        tool_registry=create_native_math_tools(),
        max_turns=5,
        success_reward=1.0,
        failure_reward=0.0,
    )

    episode_ids, _ = env.reset(["What is 2 + 2?"], [{"answer": None}])
    env.step(episode_ids, ["Done!"])
    traj = env.get_trajectories(episode_ids)[0]
    assert traj.info["completed"] is True
    assert traj.total_reward == 0.0  # failure_reward, never the completion payout
    assert traj.episode_invalid is True  # and dropped from the group baseline

    env.cleanup(episode_ids)


# SearchQAEnvironment


def test_qa_search_environment_init(allow_mock_search):
    """Test the create_qa_search_environment factory (preset over NativeToolUseEnvironment)."""
    from src.environments.envs.tasks.qa import create_qa_search_environment

    env = create_qa_search_environment(max_turns=10, search_backend="mock")
    tools = env.get_tools_schema()
    tool_names = [t["function"]["name"] for t in tools]
    assert "web_search" in tool_names


def test_qa_search_environment_correct_answer(allow_mock_search):
    """Test SearchQA rewards correct answer."""
    from src.environments.envs.tasks.qa import create_qa_search_environment

    env = create_qa_search_environment(
        max_turns=5,
        search_backend="mock",
        success_reward=1.0,
        failure_reward=0.0,
    )

    episode_ids, _ = env.reset(
        ["What year was the Eiffel Tower completed?"],
        [{"answer": "1889"}],
    )

    # Final answer without tools. ``require_tool_use`` only FLAGS the zero-tool-call finish; the price
    # is the episode-level no_tool_use_penalty (default 0), so a correct answer scores exactly 1.0 —
    # charging the per-call tool_error_penalty here too would double-bill the same condition.
    env.step(episode_ids, ["1889"])
    traj = env.get_trajectories(episode_ids)[0]
    assert traj.done
    assert traj.info["no_tool_use"] is True
    assert abs(traj.total_reward - 1.0) < 1e-9, traj.total_reward

    env.cleanup(episode_ids)


def test_no_tool_use_is_charged_once_by_the_dedicated_knob(allow_mock_search):
    """The zero-tool-call giveup costs exactly ``no_tool_use_penalty``, once.

    ``require_tool_use`` and ``_tool_use_shaping`` both fire on that condition; the terminal step must
    not add a second (per-call) charge on top of the episode-level one.
    """
    from src.environments.envs.tasks.qa import create_qa_search_environment

    env = create_qa_search_environment(
        max_turns=5, search_backend="mock", success_reward=1.0, failure_reward=0.0, no_tool_use_penalty=0.3
    )
    episode_ids, _ = env.reset(["Q?"], [{"answer": "A"}])
    env.step(episode_ids, ["A"])
    traj = env.get_trajectories(episode_ids)[0]
    assert abs(traj.total_reward - 0.7) < 1e-9, traj.total_reward  # 1.0 success - 0.3 once, NOT twice
    env.cleanup(episode_ids)


def test_qa_search_environment_wrong_answer(allow_mock_search):
    """Test SearchQA penalizes wrong answer."""
    from src.environments.envs.tasks.qa import create_qa_search_environment

    env = create_qa_search_environment(
        max_turns=5,
        search_backend="mock",
        success_reward=1.0,
        failure_reward=0.0,
    )

    episode_ids, _ = env.reset(
        ["What year was the Eiffel Tower completed?"],
        [{"answer": "1889"}],
    )

    env.step(episode_ids, ["2005"])
    traj = env.get_trajectories(episode_ids)[0]
    assert traj.done
    assert traj.total_reward < 0.5  # failure_reward (0.0) minus possible penalty

    env.cleanup(episode_ids)


def test_qa_search_registry(allow_mock_search):
    """Test qa_search resolves from registry."""
    from src.environments.registry import resolve_environment

    env = resolve_environment("qa_search", {"search_backend": "mock"})
    assert env is not None
    tools = env.get_tools_schema()
    assert any(t["function"]["name"] == "web_search" for t in tools)


# CodeContestsEnvironment


def test_code_contests_environment_init():
    """Test CodeContestsEnvironment initialization."""
    from src.environments.envs.tasks.coding.code_contests import CodeContestsEnvironment

    env = CodeContestsEnvironment(max_turns=10)
    tools = env.get_tools_schema()
    tool_names = [t["function"]["name"] for t in tools]
    assert "python_repl" in tool_names
    assert "submit_solution" in tool_names


def test_code_contests_run_tests():
    """Test run_solution_against_tests directly."""
    from src.environments.envs.tasks.coding.grading import run_solution_against_tests

    code = "n = int(input())\nprint(n * 2)"
    test_cases = [
        {"input": "5", "output": "10"},
        {"input": "0", "output": "0"},
        {"input": "3", "output": "6"},
    ]

    passed, total, details, *_ = run_solution_against_tests(code, test_cases)
    assert total == 3
    assert passed == 3
    assert "3/3" in details


def test_code_contests_registry():
    """Test code_contests resolves from registry."""
    from src.environments.registry import resolve_environment

    env = resolve_environment("code_contests", {})
    assert env is not None
    tools = env.get_tools_schema()
    assert any(t["function"]["name"] == "submit_solution" for t in tools)


# ExamQAEnvironment


def test_exam_qa_closed_book_refuses_a_search_backend():
    """A closed-book exam registers no search tool, so a configured backend reaches nothing.

    Refused at construction, not ignored: read only inside the open-book branch, a misspelled or
    unselectable backend name — refused at construction everywhere else — would launch a whole run
    that trains closed-book with no error and no warning.
    """
    from src.environments.envs.tasks.qa import ExamQAEnvironment

    with pytest.raises(ValueError, match="search_backend"):
        ExamQAEnvironment(max_turns=3, search_backend="duckduckgo")

    # Open-book still binds it, and an unselectable name still fails at construction.
    assert ExamQAEnvironment(max_turns=3, open_book=True, search_backend="duckduckgo").registry.names()
    with pytest.raises(ValueError, match="Unknown search backend"):
        ExamQAEnvironment(max_turns=3, open_book=True, search_backend="nonexistent")


def test_exam_qa_multiple_choice_correct():
    """Test correct multiple-choice answer."""
    from src.environments.envs.tasks.qa import ExamQAEnvironment

    env = ExamQAEnvironment(
        max_turns=3,
        success_reward=1.0,
        failure_reward=0.0,
    )

    episode_ids, _ = env.reset(
        ["Which is the largest planet?"],
        [{"answer": "B", "choices": ["A: Mars", "B: Jupiter", "C: Saturn", "D: Neptune"]}],
    )

    env.step(episode_ids, ["The answer is B"])
    traj = env.get_trajectories(episode_ids)[0]
    assert traj.done
    assert traj.total_reward >= 1.0

    env.cleanup(episode_ids)


def test_exam_qa_multiple_choice_wrong():
    """Test wrong multiple-choice answer."""
    from src.environments.envs.tasks.qa import ExamQAEnvironment

    env = ExamQAEnvironment(
        max_turns=3,
        success_reward=1.0,
        failure_reward=0.0,
    )

    episode_ids, _ = env.reset(
        ["Which is the largest planet?"],
        [{"answer": "B", "choices": ["A: Mars", "B: Jupiter", "C: Saturn", "D: Neptune"]}],
    )

    env.step(episode_ids, ["The answer is A"])
    traj = env.get_trajectories(episode_ids)[0]
    assert traj.done
    assert traj.total_reward < 1.0

    env.cleanup(episode_ids)


def test_exam_qa_index_answer_is_graded_as_its_choice_letter():
    """MMLU/ARC ship ``answer`` as a 0-based index into ``choices``.

    ``multiple_choice_match`` scores anything that is not a single letter as wrong, so an unconverted
    index gives EVERY completion ``failure_reward``: a GRPO group with zero variance, no gradient, and
    nothing in the logs saying the rows were never gradable."""
    from src.environments.envs.tasks.qa import ExamQAEnvironment

    env = ExamQAEnvironment(max_turns=3, success_reward=1.0, failure_reward=0.0)
    choices = ["Mars", "Jupiter", "Saturn", "Neptune"]

    episode_ids, _ = env.reset(["Which is the largest planet?"], [{"answer": 1, "choices": choices}])
    assert env.get_trajectories(episode_ids)[0].info["expected_answer"] == "B"
    env.step(episode_ids, ["The answer is B"])
    assert env.get_trajectories(episode_ids)[0].total_reward == 1.0

    # A digit string is the same shape; and the conversion must still grade a wrong letter as wrong.
    other_ids, _ = env.reset(["Which is the largest planet?"], [{"answer": "2", "choices": choices}])
    assert env.get_trajectories(other_ids)[0].info["expected_answer"] == "C"
    env.step(other_ids, ["The answer is B"])
    assert env.get_trajectories(other_ids)[0].total_reward == 0.0

    env.cleanup(episode_ids + other_ids)


def test_exam_qa_letter_answers_pass_through_and_bad_shapes_fail_loud():
    """A real letter keeps grading unchanged; any shape the grader cannot score raises at episode
    start, because a constant-reward run costs far more than a refused one."""
    import pytest

    from src.environments.envs.tasks.qa import ExamQAEnvironment

    env = ExamQAEnvironment(max_turns=3, success_reward=1.0, failure_reward=0.0)
    choices = ["Mars", "Jupiter", "Saturn", "Neptune"]

    episode_ids, _ = env.reset(["Which is the largest planet?"], [{"answer": "b", "choices": choices}])
    assert env.get_trajectories(episode_ids)[0].info["expected_answer"] == "B"

    with pytest.raises(ValueError, match="neither a choice letter"):
        env.reset(["Which is the largest planet?"], [{"answer": "Jupiter", "choices": choices}])
    with pytest.raises(ValueError, match="does not address"):
        env.reset(["Which is the largest planet?"], [{"answer": 9, "choices": choices}])
    with pytest.raises(ValueError, match="neither a choice letter"):
        # bool is an int subclass: True must not silently index choice "B".
        env.reset(["Which is the largest planet?"], [{"answer": True, "choices": choices}])

    # Open-ended rows (no choices) are untouched by the letter contract.
    open_ids, _ = env.reset(["What is the capital of France?"], [{"answer": "Paris"}])
    assert env.get_trajectories(open_ids)[0].info["expected_answer"] == "Paris"

    env.cleanup(episode_ids + open_ids)


def test_exam_qa_open_ended():
    """Test open-ended exam question."""
    from src.environments.envs.tasks.qa import ExamQAEnvironment

    env = ExamQAEnvironment(max_turns=3, success_reward=1.0, failure_reward=0.0)

    episode_ids, _ = env.reset(
        ["What is the capital of France?"],
        [{"answer": "Paris"}],
    )

    env.step(episode_ids, ["Paris"])
    traj = env.get_trajectories(episode_ids)[0]
    assert traj.done
    assert traj.total_reward >= 1.0

    env.cleanup(episode_ids)


def test_exam_qa_open_book(allow_mock_search):
    """Test open-book exam has search tools."""
    from src.environments.envs.tasks.qa import ExamQAEnvironment

    env = ExamQAEnvironment(max_turns=5, open_book=True, search_backend="mock")
    tools = env.get_tools_schema()
    tool_names = [t["function"]["name"] for t in tools]
    assert "web_search" in tool_names


# Test: EnvironmentConfig


def test_environment_config_defaults():
    """Test EnvironmentConfig default values."""
    from src.configs.environment_config import EnvironmentConfig

    config = EnvironmentConfig()
    assert config.environment_type == "react_math"
    assert config.success_reward == 1.0
    assert config.failure_reward == 0.0
    # None sentinel: the env class default wins unless the YAML sets max_turns explicitly.
    assert config.max_turns is None
    assert config.environment_kwargs == {}


def test_environment_config_to_env_config():
    """Test to_env_config() outputs correct dict with defaults."""
    from src.configs.environment_config import EnvironmentConfig

    config = EnvironmentConfig()
    env_dict = config.to_env_config()

    # Core reward shaping is always emitted; max_turns is omitted when unset so the env class
    # default (CodeContests 15, SWE 20, ExamQA 8, ...) wins.
    assert env_dict == {
        "success_reward": 1.0,
        "failure_reward": 0.0,
    }
    assert "max_turns" not in env_dict
    assert EnvironmentConfig(max_turns=12).to_env_config()["max_turns"] == 12


def test_environment_config_env_specific_kwargs():
    """Test to_env_config() merges environment_kwargs."""
    from src.configs.environment_config import EnvironmentConfig

    config = EnvironmentConfig(
        environment_type="qa_search",
        success_reward=2.0,
        max_turns=15,
        environment_kwargs={"search_backend": "mock", "include_python_tools": True},
    )
    env_dict = config.to_env_config()

    assert env_dict["success_reward"] == 2.0
    assert env_dict["max_turns"] == 15
    assert env_dict["search_backend"] == "mock"
    assert env_dict["include_python_tools"] is True


def test_all_registered_envs_resolvable():
    """Every registered env type resolves to a real BaseEnvironment honouring the config's max_turns.

    A factory that ignored ``env_config`` — or returned a stray non-environment object — would keep
    a plain ``is not None`` check green while every rollout ran the class default number of turns.
    """
    from src.configs.environment_config import EnvironmentConfig
    from src.environments.base import BaseEnvironment
    from src.environments.registry import get_registered_environments, resolve_environment

    registered = [name for name in get_registered_environments() if name != "mcp"]  # MCP needs a server
    assert len(registered) >= 5, f"the registry went (nearly) empty — the loop below tests nothing: {registered}"
    for env_name in registered:
        config = EnvironmentConfig(environment_type=env_name, max_turns=3)
        env = resolve_environment(config.environment_type, config.to_env_config())
        assert isinstance(env, BaseEnvironment), f"{env_name} resolved to {type(env).__name__}"
        assert env.max_turns == 3, f"{env_name} ignored max_turns from env_config (got {env.max_turns})"


# Registry Integration


def test_react_step_does_not_double_add_assistant_message():
    """BaseEnvironment.step already appends the assistant action message, so ReAct's _step_single
    must NOT append it again: a double-add corrupts the history and halves max_turns."""
    from src.environments.envs.protocols.react import ReActEnvironment
    from src.environments.tools.definitions import NativeToolRegistry

    env = ReActEnvironment(tool_registry=NativeToolRegistry(), require_thought=False)
    episode_ids, _ = env.reset(["Solve: 1+1"])
    eid = episode_ids[0]

    steps = env.step([eid], ["Thought: done\nFinal Answer: 2"])
    assert steps[0].done

    traj = env.get_trajectories([eid])[0]
    assert traj is not None
    roles = [m.role for m in traj.messages]
    assert roles.count("assistant") == 1, f"expected exactly 1 assistant message, got roles={roles}"
    assert not any(roles[i] == "assistant" and roles[i + 1] == "assistant" for i in range(len(roles) - 1)), roles


def test_grading_nonzero_exit_is_runtime_error_not_pass():
    """A solution that prints the correct answer but exits non-zero is a Runtime Error — it must fail
    even though stdout matches (reward-leakage regression)."""
    from src.environments.envs.tasks.coding.grading import run_solution_against_tests
    from src.environments.sandbox.base import SandboxResult

    class _FakeSandbox:
        def __init__(self, result):
            self._result = result

        def run(self, *args, **kwargs):
            return self._result

    tests = [{"input": "", "output": "42"}]

    crashed = _FakeSandbox(SandboxResult(stdout="42", returncode=1))
    assert run_solution_against_tests("code", tests, sandbox=crashed)[:2] == (0, 1), "non-zero exit must not pass"

    clean = _FakeSandbox(SandboxResult(stdout="42", returncode=0))
    assert run_solution_against_tests("code", tests, sandbox=clean)[:2] == (1, 1), "clean exit must pass"


def test_react_parse_empty_string_argument_preserved():
    """A quoted empty argument (expression=\"\") must parse to "" — truthiness-based group selection
    turned it into None, so the tool then executed with a None argument."""
    from src.environments.envs.protocols.react import parse_react_output

    step = parse_react_output('Thought: try it\nAction: calculate(expression="")')
    assert step.action == "calculate"
    assert step.action_args == {"expression": ""}

    step = parse_react_output("Thought: mix\nAction: run(code='', count=3)")
    assert step.action_args == {"code": "", "count": 3}


def test_native_step_tool_calls_counts_executed_not_requested():
    """step_tool_calls must count EXECUTED calls (post per-turn cap), matching total_tool_calls —
    counting the requested list made the two metrics disagree on a capped turn."""
    from src.environments.envs.protocols.native import NativeToolUseEnvironment
    from src.environments.tools.definitions import NativeTool, NativeToolRegistry

    registry = NativeToolRegistry()
    registry.register(NativeTool(name="echo", description="echo", parameters=[], handler=lambda **a: "ok"))
    env = NativeToolUseEnvironment(tool_registry=registry, max_tool_calls_per_turn=2)
    episode_ids, _ = env.reset(["task"])
    eid = episode_ids[0]

    calls = [{"id": f"c{i}", "function": {"name": "echo", "arguments": "{}"}} for i in range(5)]
    steps = env.step([eid], [""], [{"tool_calls": calls}])
    traj = env.get_trajectories([eid])[0]
    assert steps[0].info["step_tool_calls"] == 2  # executed (capped), not the 5 requested
    assert traj.info["total_tool_calls"] == 2  # and consistent with the per-episode counter


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
