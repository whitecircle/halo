#!/usr/bin/env python
"""A model-authored tool call that omits a schema-required argument is the model's error: the tool
reports which argument is missing instead of letting the handler die on a ``TypeError``, and the
protocol turns that into an ordinary tool-error observation.

Run: python tests/cpu/environments/test_tool_required_arguments.py
"""

import asyncio

import pytest

from src.environments.tools.definitions import MissingToolArguments, NativeTool, ToolParameter


def _tool() -> NativeTool:
    return NativeTool(
        name="run_test",
        description="run code",
        parameters=[
            ToolParameter(name="code", type="string", description="source"),
            ToolParameter(name="stdin", type="string", description="input", required=False),
        ],
        handler=lambda code, stdin="": f"{code}|{stdin}",
    )


def test_missing_required_argument_names_it():
    with pytest.raises(MissingToolArguments, match="run_test.*missing required argument\\(s\\): code") as excinfo:
        _tool().execute(stdin="x")
    assert excinfo.value.missing == ["code"]


def test_missing_required_argument_is_not_a_type_error():
    """The failure the guard replaces: the handler's own signature error, logged as a tool fault."""
    try:
        _tool().execute()
    except TypeError:
        pytest.fail("a missing required argument reached the handler as a TypeError")
    except MissingToolArguments:
        pass


def test_optional_argument_may_be_omitted_and_undeclared_ones_are_dropped():
    assert _tool().execute(code="print(1)") == "print(1)|"
    assert _tool().execute(code="c", stdin="s", timeout=999) == "c|s"


def test_async_path_applies_the_same_check():
    with pytest.raises(MissingToolArguments):
        asyncio.run(_tool().execute_async(stdin="x"))


def test_a_tool_without_a_schema_passes_arguments_through():
    tool = NativeTool(name="free", description="d", handler=lambda **kw: str(sorted(kw)))
    assert tool.execute(a=1, b=2) == "['a', 'b']"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
