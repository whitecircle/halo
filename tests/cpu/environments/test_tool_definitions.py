#!/usr/bin/env python
"""Tests for the native tool data structures, registry and the environment registry
(``src/environments/tools/`` + ``src/environments/registry.py``).

The vLLM parsing/formatting helpers are covered by
``tests/cpu/environments/test_openai_tool_format.py`` and the pre-built tool factories by
``test_environments.py``; this file owns the tool/registry/schema surface.

Run: ``pytest tests/cpu/environments/test_tool_definitions.py``.
"""

import asyncio
import functools
import inspect
import time

import pytest

from src.environments.envs.tasks.coding.code_contests import CodeContestsEnvironment
from src.environments.registry import (
    get_registered_environments,
    register_environment,
    resolve_environment,
)
from src.environments.sandbox.base import SANDBOX_DEFAULT_TIMEOUT
from src.environments.tools.definitions import (
    NativeTool,
    NativeToolCall,
    NativeToolRegistry,
    NativeToolResult,
    ToolParameter,
)
from src.environments.tools.factories import (
    create_all_native_tools,
    create_native_code_tools,
    create_native_file_tools,
    create_native_math_tools,
    create_native_python_tools,
    create_native_search_tools,
    create_session_code_tools,
    create_session_file_tools,
)

# NativeTool.to_openai_schema


def test_to_openai_schema_required_and_optional_params():
    tool = NativeTool(
        name="calculate",
        description="Evaluate a math expression",
        parameters=[
            ToolParameter("expression", "string", "The math expression", required=True),
            ToolParameter("precision", "integer", "Decimal places", required=False),
        ],
    )
    schema = tool.to_openai_schema()
    assert schema["type"] == "function"
    fn = schema["function"]
    assert fn["name"] == "calculate"
    assert fn["description"] == "Evaluate a math expression"
    props = fn["parameters"]["properties"]
    assert fn["parameters"]["type"] == "object"
    assert props["expression"]["type"] == "string"
    assert props["precision"]["type"] == "integer"
    assert fn["parameters"]["required"] == ["expression"]


def test_to_openai_schema_default_required_is_true():
    tool = NativeTool(name="t", description="d", parameters=[ToolParameter("x", "string", "desc")])
    assert tool.to_openai_schema()["function"]["parameters"]["required"] == ["x"]


def test_to_openai_schema_emits_enum():
    tool = NativeTool(
        name="format",
        description="Format output",
        parameters=[ToolParameter("style", "string", "Output style", enum=["json", "csv", "text"])],
    )
    props = tool.to_openai_schema()["function"]["parameters"]["properties"]
    assert props["style"]["enum"] == ["json", "csv", "text"]
    bare = NativeTool(name="b", description="d", parameters=[ToolParameter("y", "string", "d")])
    assert "enum" not in bare.to_openai_schema()["function"]["parameters"]["properties"]["y"]


def test_to_openai_schema_no_params():
    schema = NativeTool(name="noop", description="d").to_openai_schema()
    assert schema["function"]["parameters"]["properties"] == {}
    assert schema["function"]["parameters"]["required"] == []


# NativeTool.execute / execute_async


def _add_tool() -> NativeTool:
    return NativeTool(
        name="add",
        description="d",
        parameters=[ToolParameter("a", "number", "a"), ToolParameter("b", "number", "b")],
        handler=lambda a, b: a + b,
    )


def test_execute_invokes_sync_handler_and_stringifies():
    assert _add_tool().execute(a=3, b=4) == "7"


def test_execute_coerces_non_string_result():
    tool = NativeTool(name="get_list", description="d", handler=lambda: [1, 2, 3])
    result = tool.execute()
    assert result == "[1, 2, 3]" and isinstance(result, str)


def test_execute_without_handler_raises():
    with pytest.raises(NotImplementedError, match="noop"):
        NativeTool(name="noop", description="d").execute()


def test_execute_async_prefers_async_handler():
    async def ahandler(x):
        return x * 2

    tool = NativeTool(
        name="double", description="d", parameters=[ToolParameter("x", "number", "x")], async_handler=ahandler
    )
    assert asyncio.run(tool.execute_async(x=21)) == "42"


def test_execute_async_falls_back_to_sync_handler():
    # Without an async_handler the sync one is threaded off and str()-coerced: same -> str contract.
    assert asyncio.run(_add_tool().execute_async(a=1, b=2)) == "3"

    str_tool = NativeTool(name="echo", description="d", handler=lambda: "ok")
    assert asyncio.run(str_tool.execute_async()) == "ok"


def test_execute_async_without_any_handler_raises():
    with pytest.raises(NotImplementedError):
        asyncio.run(NativeTool(name="x", description="d").execute_async())


def test_execute_drops_arguments_the_tool_does_not_declare():
    """A model-supplied extra must not reach the handler.

    The handlers are ``functools.partial`` objects carrying pre-bound safety keywords, and a call-time
    keyword of the same name overrides them — so an undeclared argument was a way for the model to
    rewrite the tool's own configuration. ``execute`` now binds against the declared schema.
    """
    seen: dict = {}

    def handler(code, timeout=1.0, allow_imports=False):
        seen.update(code=code, timeout=timeout, allow_imports=allow_imports)
        return "ok"

    tool = NativeTool(
        name="python",
        description="d",
        parameters=[ToolParameter("code", "string", "code")],
        handler=functools.partial(handler, timeout=1.0, allow_imports=False),
    )
    tool.execute(code="print(1)", timeout=3600, allow_imports=True)
    assert seen == {"code": "print(1)", "timeout": 1.0, "allow_imports": False}


def test_execute_passes_arguments_through_when_the_tool_declares_no_schema():
    """No declared parameters = nothing to filter against (an MCP server may advertise a tool with no
    ``properties``); dropping its arguments would silently break the call."""
    seen: dict = {}
    tool = NativeTool(name="passthrough", description="d", handler=lambda **kw: seen.update(kw) or "ok")
    tool.execute(anything=1, else_=2)
    assert seen == {"anything": 1, "else_": 2}


def test_python_tool_cannot_have_its_timeout_raised_by_the_model():
    """End-to-end on the shipped in-process Python tool: the configured wall-clock cap must hold."""
    tool = create_native_python_tools(timeout=0.3).get("python")
    started = time.monotonic()
    result = tool.execute(code="while True:\n    pass", timeout=30)
    assert "0.3s timeout" in result
    assert time.monotonic() - started < 5, "the model-supplied timeout overrode the configured one"


# NativeToolCall.from_openai_format


def test_tool_call_parses_json_string_arguments():
    tc = NativeToolCall.from_openai_format(
        {"id": "call_123", "function": {"name": "calculate", "arguments": '{"expression": "2+3"}'}}
    )
    assert (tc.id, tc.name, tc.arguments) == ("call_123", "calculate", {"expression": "2+3"})


def test_tool_call_accepts_dict_arguments():
    tc = NativeToolCall.from_openai_format(
        {"id": "call_456", "function": {"name": "python", "arguments": {"code": "print(42)"}}}
    )
    assert tc.arguments == {"code": "print(42)"}


def test_tool_call_invalid_json_becomes_empty_args():
    tc = NativeToolCall.from_openai_format(
        {"id": "call_bad", "function": {"name": "calc", "arguments": "not valid json{{{"}}
    )
    assert tc.name == "calc" and tc.arguments == {}


def test_tool_call_missing_fields_defaults():
    tc = NativeToolCall.from_openai_format({})
    assert (tc.id, tc.name, tc.arguments) == ("", "", {})


def test_tool_call_null_function_degrades_instead_of_raising():
    """An explicit ``"function": null`` (or a non-dict) in a model's tool-call payload must degrade to
    an empty-name call the registry rejects as unknown-tool — an AttributeError here escapes env.step
    and kills the whole episode."""
    for malformed in ({"id": "c1", "function": None}, {"id": "c1", "function": "not-a-dict"}, {"function": []}):
        tc = NativeToolCall.from_openai_format(malformed)
        assert (tc.name, tc.arguments) == ("", {})
    tc = NativeToolCall.from_openai_format({"id": None, "function": {"name": None, "arguments": None}})
    assert (tc.id, tc.name, tc.arguments) == ("", "", {})


def test_tool_result_to_message():
    msg = NativeToolResult(tool_call_id="call_abc", name="calculate", content="42", success=True).to_message()
    assert msg.role == "tool"
    assert msg.content == "42"
    assert msg.tool_call_id == "call_abc"
    assert msg.name == "calculate"


# NativeToolRegistry


def test_registry_register_get_and_miss():
    reg = NativeToolRegistry()
    tool = NativeTool(name="t", description="d")
    assert reg.register(tool) is reg  # chainable
    assert reg.get("t") is tool
    assert reg.get("nope") is None


def test_registry_len_names_and_list():
    reg = NativeToolRegistry()
    reg.register(NativeTool(name="a", description="A"))
    reg.register(NativeTool(name="b", description="B"))
    assert len(reg) == 2
    assert set(reg.names()) == {"a", "b"}
    assert {t.name for t in reg.list_tools()} == {"a", "b"}


def test_registry_register_overwrites_same_name():
    reg = NativeToolRegistry()
    reg.register(NativeTool(name="dup", description="first"))
    reg.register(NativeTool(name="dup", description="second"))
    assert len(reg) == 1
    assert reg.get("dup").description == "second"


def test_registry_merge_is_in_place():
    reg1 = NativeToolRegistry().register(NativeTool(name="a", description="A"))
    reg2 = NativeToolRegistry().register(NativeTool(name="b", description="B"))
    assert reg1.merge(reg2) is reg1
    assert set(reg1.names()) == {"a", "b"}


def test_registry_combine_does_not_mutate_sources():
    r1 = NativeToolRegistry().register(NativeTool(name="x", description="X"))
    r2 = NativeToolRegistry().register(NativeTool(name="y", description="Y"))
    combined = NativeToolRegistry.combine(r1, r2)
    assert set(combined.names()) == {"x", "y"}
    assert len(r1) == 1 and len(r2) == 1  # sources untouched


def test_registry_to_openai_tools():
    reg = NativeToolRegistry().register(
        NativeTool(name="calc", description="Calculate", parameters=[ToolParameter("expr", "string", "Expression")])
    )
    tools = reg.to_openai_tools()
    assert len(tools) == 1
    assert tools[0]["type"] == "function"
    assert tools[0]["function"]["name"] == "calc"


# Pre-built tool factories


def test_factory_file_tools_execute_round_trip():
    reg = create_native_file_tools()
    reg.get("write_file").execute(path="/home/user/x.txt", content="hello")
    assert reg.get("read_file").execute(path="/home/user/x.txt") == "hello"
    assert "not found" in reg.get("read_file").execute(path="/missing").lower()


def test_file_tool_wording_is_per_store_while_the_schema_is_shared():
    """The two stores share one builder but not one set of instructions: a simulated absolute-path
    filesystem and a persistent relative-path workspace need different wording for the same three
    tools, so a fold that shared one description set would mis-instruct one of them."""
    simulated = create_native_file_tools()
    session = create_session_file_tools(lambda: None)
    for name in ("read_file", "write_file", "list_files"):
        assert simulated.get(name).description != session.get(name).description
        assert [p.name for p in simulated.get(name).parameters] == [p.name for p in session.get(name).parameters]
        assert [p.description for p in simulated.get(name).parameters] != [
            p.description for p in session.get(name).parameters
        ]
    assert "workspace" in session.get("read_file").description


# Environment registry


def test_env_registry_lists_builtins_sorted():
    envs = get_registered_environments()
    assert envs == sorted(envs)
    assert {"react_math", "native_math", "swe"} <= set(envs)


def test_env_registry_duplicate_without_override_raises():
    with pytest.raises(ValueError, match="already registered"):
        register_environment("react_math", lambda c: None, override=False)


def test_env_registry_resolve_unknown_raises():
    with pytest.raises(ValueError, match="Unknown environment type"):
        resolve_environment("totally_nonexistent_env_12345", {})


def test_env_registry_register_resolve_and_override_round_trip(isolated_registry):
    name = "unit_test_custom_env"
    sentinel_a, sentinel_b = object(), object()
    register_environment(name, lambda c: sentinel_a, override=True)
    assert resolve_environment(name, {}) is sentinel_a
    assert resolve_environment(name.upper(), {}) is sentinel_a  # names are lower-cased on both sides
    register_environment(name, lambda c: sentinel_b, override=True)
    assert resolve_environment(name, {}) is sentinel_b


# Declared parameter names vs the handlers they bind to


def _registries_under_test() -> dict[str, NativeToolRegistry]:
    """Every statically-built tool registry the toolkit ships, keyed by how it is built.

    Read off the factories and the environments that own their own tools, so a new tool arrives
    covered. MCP is out of scope by construction: its parameters come from the live server's schema
    and its handler is ``**kwargs``.
    """

    def no_session():
        """The session getter is never called here — only the declared schema is read."""
        return None

    return {
        "math": create_native_math_tools(),
        "code": create_native_code_tools(),
        "python": create_native_python_tools(),
        "search": create_native_search_tools(backend="duckduckgo"),
        "file": create_native_file_tools(),
        "session_code": create_session_code_tools(no_session),
        "session_file": create_session_file_tools(no_session),
        "all": create_all_native_tools(),
        "code_contests": CodeContestsEnvironment(language="python", sandbox_backend="local").registry,
    }


def _bindable(handler, name: str) -> bool:
    """True when ``handler`` accepts ``name`` as a keyword (directly or through ``**kwargs``)."""
    signature = inspect.signature(handler)
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in signature.parameters.values()):
        return True
    parameter = signature.parameters.get(name)
    return parameter is not None and parameter.kind in (
        inspect.Parameter.POSITIONAL_OR_KEYWORD,
        inspect.Parameter.KEYWORD_ONLY,
    )


@pytest.mark.parametrize("source", sorted(_registries_under_test()))
def test_every_declared_parameter_binds_to_its_handler(source):
    """A declared name the handler does not accept breaks that tool for the whole run.

    ``bind_arguments`` filters the model's arguments down to the DECLARED schema, so a declared name
    the handler has no keyword for is passed straight to it: the handler raises, and the protocol
    books a tool error on every call the model makes — while the keyword the handler does own can
    never be reached, silently keeping its default. Neither failure is visible in a config review.
    ``functools.partial`` is unwrapped because that is how the pre-bound safety keywords arrive.
    """
    mismatches = []
    for tool in _registries_under_test()[source].list_tools():
        for handler in (tool.handler, tool.async_handler):
            if handler is None:
                continue
            mismatches += [
                f"{tool.name}.{parameter.name} -> {getattr(handler, '__qualname__', handler)}"
                for parameter in tool.parameters
                if not _bindable(handler, parameter.name)
            ]
    assert not mismatches, f"declared tool parameters that reach no handler keyword: {mismatches}"


def test_the_binding_check_would_catch_a_misspelled_parameter():
    """Anti-vacuity: the check above must fail on the defect it exists to find, and nothing else
    catches it before a run does — the model's call raises inside the handler, once per call."""
    tool = NativeTool(
        name="misdeclared",
        description="",
        parameters=[
            ToolParameter("query", "string", "the query"),
            ToolParameter("max_result", "integer", "typo for max_results", required=False),
        ],
        handler=lambda query, max_results=5: f"{query}:{max_results}",
    )
    assert not _bindable(tool.handler, "max_result")
    with pytest.raises(TypeError, match="max_result"):
        tool.execute(query="q", max_result=99)
    # The keyword the handler does own is unreachable: undeclared names never survive the filter.
    assert tool.execute(query="q", max_results=99) == "q:5"


def test_code_tool_advertises_no_input_channel_it_cannot_bind():
    """The description is the model's only account of the tool. It once promised stdin while no
    ``stdin`` parameter was declared and none was bound, so the sandbox always ran with empty input
    and the model was told to write programs that read it."""
    for source in ("code", "python", "session_code"):
        for tool in _registries_under_test()[source].list_tools():
            declared = {parameter.name for parameter in tool.parameters}
            texts = [tool.description] + [parameter.description for parameter in tool.parameters]
            promises_stdin = any("stdin" in text and "no stdin" not in text for text in texts)
            assert not promises_stdin or "stdin" in declared, (
                f"{source}/{tool.name} tells the model about stdin but declares no stdin parameter: {texts}"
            )


def test_calculator_timeout_is_plumbed_from_the_factory():
    """``create_all_native_tools(timeout=...)`` must reach the calculator, not just the code tool.

    The bound value has to be a pre-bound handler keyword: a call-time one would be filtered out by
    ``bind_arguments`` (deliberately — a model must not raise its own execution cap).
    """
    assert create_all_native_tools(timeout=0.25).get("calculate").handler.keywords["timeout"] == 0.25
    assert create_native_math_tools().get("calculate").handler.keywords["timeout"] == SANDBOX_DEFAULT_TIMEOUT
    # And the model cannot raise it back: an undeclared call-time keyword is filtered out.
    tight = create_native_math_tools(timeout=0.25).get("calculate")
    assert tight.bind_arguments({"expression": "1+1", "timeout": 600}) == {"expression": "1+1"}


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
