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

import pytest

from src.environments.tools.definitions import NativeToolCall, NativeToolResult

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
    raise SystemExit(pytest.main([__file__, "-q"]))
