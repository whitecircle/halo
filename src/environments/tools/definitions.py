"""OpenAI-compatible tool definitions and registry (the tool-calling format both rollout engines take)."""

import ast
import asyncio
import contextlib
import inspect
import json
import warnings
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from src.environments.base import Message
from src.environments.sandbox.results import SandboxFault

# What reading model-authored arguments as JSON or a Python literal raises on malformed text, past a
# syntax error: an unhashable dict key or set member (``{[1]: 2}``) is a TypeError, deep nesting a
# RecursionError or MemoryError, and an integer past Python's 4300-digit conversion limit a ValueError
# that ``json.loads`` raises outside ``JSONDecodeError``.
MALFORMED_LITERAL_ERRORS = (SyntaxError, ValueError, TypeError, MemoryError, RecursionError)


class ToolBudgetExhausted(Exception):
    """A tool refusing a call because the episode spent its per-episode budget for that tool: booked as a tool ERROR
    (an over-cap call never earns ``tool_success_reward``) but logged without a traceback, being expected control flow.
    """


class ToolDisabled(ToolBudgetExhausted):
    """A call to a tool whose episode budget is zero: the run disabled it (the ``leaderboard`` evaluation
    protocol's scratchpad, a ``tool_budgets`` cap of 0). Answered and booked like a spent budget, but no refusal
    (:func:`is_refused_unrun`): where the run offers no such tool the call is an ordinary failed one, so its turn
    stays trainable."""


class ToolArgumentError(TypeError):
    """A call whose model-authored arguments the tool refuses (a required parameter missing, a name its schema does not
    declare, a value outside its enum, a list or object for a ``string``), raised before the handler runs so the model
    reads ``Error: <tool>: missing a required argument: 'code'`` instead of a Python signature, or by a handler
    refusing an admitted call (:class:`ToolCallRefused`). Booked and logged like :class:`ToolBudgetExhausted`.
    """


class ToolCallRefused(ToolArgumentError):
    """A handler refusing an admitted call it will not run, the call returned to the episode's budget
    (code contests: a program identical to one already graded, or one evidently in another language than its
    call names)."""


# What refuses a call unrun, expected control flow logged without a traceback: its arguments failing to bind or
# its tool's episode budget spent, both at admission, or its handler (:class:`ToolCallRefused`).
TOOL_CALL_REFUSALS = (ToolBudgetExhausted, ToolArgumentError)


def is_refused_unrun(outcome: object) -> bool:
    """Whether a call's ``outcome`` refused it unrun (:data:`TOOL_CALL_REFUSALS`), a disabled tool's
    (:class:`ToolDisabled`) aside. Under the native protocol a turn whose every call was refused or named no
    existing tool is flagged ``calls_rejected``, so an episode that recovers never reinforces it."""
    return isinstance(outcome, TOOL_CALL_REFUSALS) and not isinstance(outcome, ToolDisabled)


def parse_python_expression(source: str) -> ast.expr:
    """``source`` parsed as one Python expression, without the ``SyntaxWarning`` an invalid escape in a
    model-written string (``"\\d"``) prints on every parse."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", SyntaxWarning)
        return ast.parse(source.lstrip(" \t"), mode="eval").body


@dataclass
class ToolParameter:
    """A single parameter for a tool. ``type`` is its JSON-schema type as declared, ``None`` where a
    schema declares none (an MCP property typed through ``anyOf`` or ``$ref``), which takes any value."""

    name: str
    type: str | None
    description: str
    enum: list[str] | None = None
    required: bool = True


@dataclass
class NativeTool:
    """Tool definition in OpenAI function-calling format (understood natively by the rollout engines)."""

    name: str
    description: str
    parameters: list[ToolParameter] = field(default_factory=list)
    handler: Callable[..., Any] | None = None
    async_handler: Callable[..., Any] | None = None
    # The observation for a call refused because the episode spent its budget for this tool; ``None``
    # takes the generic wording. States the refusal, never the budget's size.
    budget_message: str | None = None

    def to_openai_schema(self) -> dict[str, Any]:
        """Convert to OpenAI function calling schema."""
        properties = {}
        required = []

        for param in self.parameters:
            prop: dict[str, Any] = {"description": param.description}
            if param.type is not None:
                prop["type"] = param.type
            if param.enum:
                prop["enum"] = param.enum
            properties[param.name] = prop

            if param.required:
                required.append(param.name)

        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                },
            },
        }

    def bind(self, arguments: dict[str, Any], *, for_async: bool = False) -> dict[str, Any]:
        """Admit a call's model-authored arguments: the schema-checked set the handler can bind, or
        :class:`ToolArgumentError`. The protocols call it before spending the episode's budget, so a call the handler
        could never run is refused uncounted; ``for_async`` binds the handler :meth:`execute_async` will run.
        """
        handler = (self.async_handler or self.handler) if for_async else self.handler
        if handler is None:
            raise NotImplementedError(f"Tool '{self.name}' has no handler")
        return self._bind_for_call(handler, arguments)

    def _bind_for_call(self, handler: Callable[..., Any], arguments: dict[str, Any]) -> dict[str, Any]:
        """The call's arguments checked against the declared schema and ``handler``'s signature: a name outside the
        schema is refused, since dropped unread a garbled name loses its value and a keyword shadowing a pre-bound
        safety one (the sandbox ``timeout``) would lift its cap. A tool declaring no parameters takes its arguments as
        given, and a handler without an introspectable signature is called unchecked.
        """
        declared = [parameter.name for parameter in self.parameters]
        unknown = [name for name in arguments if name not in declared] if declared else []
        if unknown:
            raise ToolArgumentError(
                f"{self.name}: unknown argument {', '.join(map(repr, unknown))}; its arguments are {', '.join(declared)}"
            )
        bound = dict(arguments)
        for parameter in self.parameters:
            value = bound.get(parameter.name)
            if parameter.type == "string" and parameter.name in bound:
                # A JSON scalar reads as its str() (an engine's parser may send ``"stdin": 5``; ``true`` reads ``True``)
                # and a null as an omission; a list or an object has no such reading, and a list ``code``
                # passed on fails inside the sandbox, read as a backend fault.
                if value is None:
                    del bound[parameter.name]
                elif not isinstance(value, str | int | float):
                    raise ToolArgumentError(
                        f"{self.name}: {parameter.name} must be a string, got {type(value).__name__}"
                    )
                else:
                    bound[parameter.name] = value = str(value)
            # The schema is the contract the model was shown: a required parameter left out, or an enum
            # value outside it, is a malformed call refused before the handler (and before the episode's
            # budget), even when the handler would supply a default of its own.
            if parameter.required and parameter.name not in bound:
                raise ToolArgumentError(f"{self.name}: missing a required argument: {parameter.name!r}")
            if parameter.enum and parameter.name in bound and value not in parameter.enum:
                raise ToolArgumentError(
                    f"{self.name}: {parameter.name} must be one of {', '.join(parameter.enum)}, got {value!r}"
                )
        try:
            signature = inspect.signature(handler)
        except (TypeError, ValueError):
            return bound
        try:
            signature.bind(**bound)
        except TypeError as e:
            raise ToolArgumentError(f"{self.name}: {e}") from None
        return bound

    def budget_exhausted_message(self) -> str:
        """The observation for a call refused because the episode spent its budget for this tool: the
        tool's own ``budget_message``, else a generic one. It states the refusal, never the budget's
        size — what an episode may do is the chat template's and the engine's to control, not a number
        the model reads."""
        return self.budget_message or f"Not executed: this task's budget for {self.name} is spent."

    def execute(self, **kwargs) -> str:
        """Execute the tool synchronously."""
        if self.handler:
            return self._as_text(self.handler(**self._bind_for_call(self.handler, kwargs)))
        raise NotImplementedError(f"Tool '{self.name}' has no sync handler")

    async def execute_async(self, **kwargs) -> str:
        """Execute the tool asynchronously."""
        if self.async_handler:
            return self._as_text(await self.async_handler(**self._bind_for_call(self.async_handler, kwargs)))
        if self.handler:
            return self._as_text(await asyncio.to_thread(self.handler, **self._bind_for_call(self.handler, kwargs)))
        raise NotImplementedError(f"Tool '{self.name}' has no handler")

    @staticmethod
    def _as_text(result: Any) -> str:
        """Render a handler's return value as the string a tool observation must be."""
        return result if isinstance(result, str) else str(result)


class NativeToolRegistry:
    """Registry of OpenAI-format tools, with composition helpers (merge/combine)."""

    def __init__(self):
        self._tools: dict[str, NativeTool] = {}

    def register(self, tool: NativeTool) -> "NativeToolRegistry":
        """Register a tool. Returns self for chaining."""
        self._tools[tool.name] = tool
        return self

    def get(self, name: str) -> NativeTool | None:
        """Get a tool by name."""
        return self._tools.get(name)

    def list_tools(self) -> list[NativeTool]:
        """List all registered tools."""
        return list(self._tools.values())

    def names(self) -> list[str]:
        """Get all tool names."""
        return list(self._tools.keys())

    def unknown_tool_message(self, name: str) -> str:
        """The observation for a call naming an unregistered tool, one wording for every protocol, naming the real
        tools as sorted prose: a policy drifting off the tool syntax invents plausible names and burns turns probing
        for a listing.
        """
        return f"Error: Unknown tool '{name}'. Available tools: {', '.join(sorted(self.names()))}"

    def to_openai_tools(self) -> list[dict[str, Any]]:
        """OpenAI function-calling schemas, passed to the rollout engine via the ``tools`` parameter."""
        return [tool.to_openai_schema() for tool in self._tools.values()]

    def merge(self, other: "NativeToolRegistry") -> "NativeToolRegistry":
        """Merge another registry's tools into this one in-place. Returns self for chaining."""
        for tool in other.list_tools():
            self._tools[tool.name] = tool
        return self

    @classmethod
    def combine(cls, *registries: "NativeToolRegistry") -> "NativeToolRegistry":
        """Create a new registry combining tools from multiple registries."""
        result = cls()
        for reg in registries:
            result.merge(reg)
        return result

    def __len__(self) -> int:
        return len(self._tools)


@dataclass
class NativeToolCall:
    """A tool call extracted from model output in OpenAI format."""

    id: str
    name: str
    arguments: dict[str, Any]

    @classmethod
    def from_openai_format(cls, tool_call: dict[str, Any]) -> "NativeToolCall":
        """The call one OpenAI tool-call object (``{"id", "function": {"name", "arguments"}}``) describes; a malformed
        payload degrades to an empty-name call the registry rejects as unknown, never an exception ending the episode.
        """
        function = tool_call.get("function")
        if not isinstance(function, dict):
            function = {}
        # `or "{}"` handles null / "" arguments.
        args = function.get("arguments") or "{}"

        if isinstance(args, str):
            # ``ast.literal_eval`` repairs Python-dict literals (single quotes, trailing commas, ``True``/``None``);
            # text neither reads as falls back to no arguments.
            raw, args = args, {}
            with contextlib.suppress(*MALFORMED_LITERAL_ERRORS):
                try:
                    args = json.loads(raw)
                except json.JSONDecodeError:
                    args = ast.literal_eval(parse_python_expression(raw))

        return cls(
            id=tool_call.get("id") or "",
            name=function.get("name") or "",
            arguments=args if isinstance(args, dict) else {},
        )


@dataclass
class NativeToolResult:
    """Result of executing a native tool call."""

    tool_call_id: str
    name: str
    content: str
    success: bool = True
    # The environment refused the call because the registry holds no such tool — the model invented it,
    # nothing ran. Structural, never inferred from the observation text: a TOOL's own failure message
    # can reproduce any wording of it (an MCP server answering "Tool not found: x" is a real failure of
    # a real tool), and the protocol drops a turn from training on this flag alone.
    unknown_tool: bool = False
    # The call was refused and nothing ran (:func:`is_refused_unrun`): at admission, or by its handler
    # (:class:`ToolCallRefused`). Structural like ``unknown_tool``, and read the same way.
    refused: bool = False
    # The sandbox fault the call ended on, booked by type (infra or agent-caused) rather than as an
    # ordinary tool error, and ending the episode.
    sandbox_fault: SandboxFault | None = None

    def to_message(self):
        """Convert to a tool :class:`Message` for the conversation."""
        return Message.tool(self.content, self.tool_call_id, self.name)
