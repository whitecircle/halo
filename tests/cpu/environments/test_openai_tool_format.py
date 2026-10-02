#!/usr/bin/env python
"""Tests for OpenAI tool-call format handling on the data model
(``src/environments/tools/definitions.py``): parsing one tool call out of a chat-completion message
and serializing a tool result back into an OpenAI ``tool`` message.

Tool-call *parsing* belongs on the data model (any OpenAI-compatible server, including vLLM,
returns this shape); picking a server-side ``--tool-call-parser`` is the inference server's job,
not the trainer's, so there is no parser-selection helper here.

Covers: dict-vs-JSON-string arguments, malformed/empty payloads, and the serialized tool message.

Run: ``pytest tests/cpu/environments/test_openai_tool_format.py``.
"""

import warnings

import pytest

from src.environments.envs.protocols.native import NativeToolUseEnvironment
from src.environments.tools.definitions import (
    NativeTool,
    NativeToolCall,
    NativeToolRegistry,
    NativeToolResult,
    ToolParameter,
)

# NativeToolCall.from_openai_format — single call


def test_from_openai_format_parses_json_string_arguments():
    tc = NativeToolCall.from_openai_format(
        {"id": "call_1", "function": {"name": "calculate", "arguments": '{"expression": "2 + 2"}'}}
    )
    assert tc.id == "call_1"
    assert tc.name == "calculate"
    assert tc.arguments == {"expression": "2 + 2"}


def test_from_openai_format_accepts_dict_arguments():
    tc = NativeToolCall.from_openai_format({"id": "c", "function": {"name": "f", "arguments": {"already": "parsed"}}})
    assert tc.arguments == {"already": "parsed"}


def test_from_openai_format_malformed_arguments_default_to_empty():
    tc = NativeToolCall.from_openai_format({"id": "c", "function": {"name": "f", "arguments": "{not json"}})
    assert tc.arguments == {}


@pytest.mark.parametrize(
    "raw",
    [
        "{[1]: 2}",
        "{'code': {[1]}}",
        "[" * 100_000 + "]" * 100_000,
        '{"a": ' * 100_000 + "1" + "}" * 100_000,
    ],
    ids=["unhashable-key", "unhashable-set-member", "deep-list", "deep-object"],
)
def test_from_openai_format_degrades_arguments_no_literal_can_build(raw):
    """An unhashable dict key or set member raises TypeError out of the literal reader, deep nesting a
    RecursionError out of the JSON one: neither is a syntax error, and uncaught either escapes
    ``env.step``, so the policy could void its own episode by writing one."""
    assert NativeToolCall.from_openai_format({"id": "c", "function": {"name": "f", "arguments": raw}}).arguments == {}


def test_from_openai_format_repairs_a_python_literal_without_a_syntax_warning():
    """An invalid escape in a model-written literal (a regex's ``\\d``) is read as Python reads it,
    silently: a ``SyntaxWarning`` per parse would print once per call of every episode."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        call = NativeToolCall.from_openai_format(
            {"id": "c", "function": {"name": "f", "arguments": "{'pattern': '\\d+'}"}}
        )
    assert call.arguments == {"pattern": "\\d+"}
    assert not [w for w in caught if issubclass(w.category, SyntaxWarning)]


def test_from_openai_format_repairs_a_python_literal_after_leading_blanks():
    """``ast.literal_eval`` strips leading blanks before it parses, and the warning-free parse in its place
    must too: a model that spaces its arguments string would otherwise bind no arguments."""
    call = NativeToolCall.from_openai_format({"id": "c", "function": {"name": "f", "arguments": " \t{'a': 1}"}})
    assert call.arguments == {"a": 1}


def test_a_call_no_literal_can_build_is_a_refused_call_not_a_lost_episode():
    registry = NativeToolRegistry().register(
        NativeTool(
            name="echo",
            description="echo",
            parameters=[ToolParameter(name="text", type="string", description="text")],
            handler=lambda text: text,
        )
    )
    env = NativeToolUseEnvironment(tool_registry=registry, max_turns=3)
    ids, _ = env.reset(["task"])
    call = {"id": "c1", "function": {"name": "echo", "arguments": "{[1]: 2}"}}
    (step,) = env.step(ids, ["calling"], [{"finish_reason": "tool_calls", "tool_calls": [call]}])
    assert not step.done
    assert (step.trajectory.info["total_tool_calls"], step.trajectory.info["successful_tool_calls"]) == (1, 0)
    assert "missing a required argument: 'text'" in step.trajectory.messages[-1].content


def test_from_openai_format_missing_fields_default():
    tc = NativeToolCall.from_openai_format({})
    assert tc.id == "" and tc.name == "" and tc.arguments == {}


# NativeToolResult serialization


def test_tool_result_serializes_to_an_openai_tool_message():
    """The wire shape the next generation turn is sent: role/content/name/tool_call_id, no extras."""
    result = NativeToolResult(tool_call_id="call_123", name="calculate", content="42")
    assert result.to_message().to_dict() == {
        "role": "tool",
        "content": "42",
        "name": "calculate",
        "tool_call_id": "call_123",
    }


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
