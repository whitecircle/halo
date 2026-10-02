"""OpenAI-compatible tool definitions and registry (the tool-calling format both rollout engines take)."""

import ast
import asyncio
import contextlib
import inspect
import json
import warnings
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

from src.environments.base import Message
from src.environments.sandbox.base import SandboxAgentFault, SandboxInfraError

# What reading model-authored arguments as JSON or a Python literal raises on malformed text, past a
# syntax error: an unhashable dict key or set member (``{[1]: 2}``) is a TypeError, deep nesting a
# RecursionError or MemoryError, and an integer past Python's 4300-digit conversion limit a ValueError
# that ``json.loads`` raises outside ``JSONDecodeError``.
MALFORMED_LITERAL_ERRORS = (SyntaxError, ValueError, TypeError, MemoryError, RecursionError)


class ToolBudgetExhausted(Exception):
    """A tool refusing a call because the episode spent its per-episode budget for that tool.

    Expected control flow, not a fault: the protocol still books it as a tool ERROR (so an over-cap
    call never earns ``tool_success_reward``) but logs it without a traceback, which is reserved for
    a tool that actually broke.
    """


class ToolArgumentError(TypeError):
    """A call whose model-authored arguments the tool refuses: a required parameter missing
    (``submit_solution`` with no ``code``), a name its schema does not declare, a value outside its enum,
    or a name its handler has no keyword for.

    Raised before the handler runs, so the model reads ``Error: <tool>: missing a required argument:
    'code'`` instead of a Python signature. Expected control flow like :class:`ToolBudgetExhausted`:
    booked as a tool ERROR, logged without a traceback.
    """


def parse_python_expression(source: str) -> ast.expr:
    """``source`` parsed as one Python expression, without the ``SyntaxWarning`` an invalid escape in a
    model-written string (``"\\d"``) prints on every parse."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", SyntaxWarning)
        return ast.parse(source.lstrip(" \t"), mode="eval").body


def _parse_tool_arguments(raw: str) -> dict[str, Any]:
    """Parse a tool-call ``arguments`` string; ``ast.literal_eval`` repairs Python-dict literals (single
    quotes, trailing commas, ``True``/``None``). Unrecoverable / non-dict input falls back to ``{}``."""
    with contextlib.suppress(*MALFORMED_LITERAL_ERRORS):
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            parsed = ast.literal_eval(parse_python_expression(raw))
        return parsed if isinstance(parsed, dict) else {}
    return {}


@dataclass
class ToolParameter:
    """A single parameter for a tool."""

    name: str
    type: str
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
    # The observation for a call refused over the episode's cap on this tool (``{cap}`` and ``{name}``
    # format fields); ``None`` takes the generic wording.
    budget_message: str | None = None

    def to_openai_schema(self) -> dict[str, Any]:
        """Convert to OpenAI function calling schema."""
        properties = {}
        required = []

        for param in self.parameters:
            prop = {
                "type": param.type,
                "description": param.description,
            }
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

    def _bind_for_call(self, handler: Callable[..., Any], arguments: dict[str, Any]) -> dict[str, Any]:
        """The call's arguments checked against the declared schema and ``handler``'s signature.

        The schema is what :meth:`to_openai_schema` showed the model, so a name outside it is refused,
        naming the declared ones: dropped unread, a garbled name loses its value, and a keyword shadowing
        a handler's pre-bound safety one (the sandbox ``timeout``) would lift that cap. A tool declaring no
        parameters (an MCP tool without ``properties``) takes its arguments as given. Binding up front
        turns the call's own ``TypeError`` into :class:`ToolArgumentError`; a handler without an
        introspectable signature is called unchecked.
        """
        declared = [parameter.name for parameter in self.parameters]
        unknown = [name for name in arguments if name not in declared] if declared else []
        if unknown:
            raise ToolArgumentError(
                f"{self.name}: unknown argument {', '.join(map(repr, unknown))}; its arguments are {', '.join(declared)}"
            )
        for parameter in self.parameters:
            # The schema is the contract the model was shown: a required parameter left out, or an enum
            # value outside it, is a malformed call refused before the handler (and before the episode's
            # budget), even when the handler would supply a default of its own.
            if parameter.required and parameter.name not in arguments:
                raise ToolArgumentError(f"{self.name}: missing a required argument: {parameter.name!r}")
            if parameter.enum and parameter.name in arguments and arguments[parameter.name] not in parameter.enum:
                raise ToolArgumentError(
                    f"{self.name}: {parameter.name} must be one of {', '.join(parameter.enum)}, "
                    f"got {arguments[parameter.name]!r}"
                )
        try:
            signature = inspect.signature(handler)
        except (TypeError, ValueError):
            return arguments
        try:
            signature.bind(**arguments)
        except TypeError as e:
            raise ToolArgumentError(f"{self.name}: {e}") from None
        return arguments

    def bind(self, arguments: dict[str, Any], *, for_async: bool = False) -> dict[str, Any]:
        """Admit a call's model-authored arguments: the schema-checked set the handler can bind, or
        :class:`ToolArgumentError`. The protocols call it before spending the episode's budget on the
        call, so a call the handler could never run is refused without being counted. ``for_async``
        binds against the handler :meth:`execute_async` will run, which may differ from the sync one; a sync
        call binds only the sync handler, so an async-only tool is refused before it is counted."""
        handler = (self.async_handler or self.handler) if for_async else self.handler
        if handler is None:
            raise NotImplementedError(f"Tool '{self.name}' has no handler")
        return self._bind_for_call(handler, arguments)

    def budget_exhausted_message(self, cap: int, left: Mapping[str, int]) -> str:
        """The observation for a call refused over the episode's cap of ``cap`` calls on this tool.

        ``left`` maps each capped tool to the calls it has left, which a ``budget_message`` names as
        ``{left_<tool>}``, so a refusal can point at the budget that still holds.
        """
        template = self.budget_message or "{name} limit reached ({cap}); this call was not executed."
        return template.format(name=self.name, cap=cap, **{f"left_{tool}": calls for tool, calls in left.items()})

    @staticmethod
    def _as_text(result: Any) -> str:
        """Render a handler's return value as the string a tool observation must be."""
        return result if isinstance(result, str) else str(result)

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
        """The observation for a call naming a tool that is not registered — one wording for every
        protocol.

        Names the real tools: a policy that has drifted off the tool syntax late in training invents
        plausible names (``test``, ``test_tool``, ``repl``) and then burns turns probing for a
        listing, so the correction has to be in the observation it already gets. Rendered as sorted
        prose — a bare ``list`` repr is not text a model reads.
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
        """Create from one OpenAI tool-call object (``{"id", "function": {"name", "arguments"}}``).

        Tolerant of a malformed payload (explicit ``"function": null`` or a non-dict): it degrades to
        an empty-name call the registry rejects as unknown-tool, instead of an AttributeError killing
        the whole episode.
        """
        function = tool_call.get("function")
        if not isinstance(function, dict):
            function = {}
        # `or "{}"` handles null / "" arguments.
        args = function.get("arguments") or "{}"

        if isinstance(args, str):
            args = _parse_tool_arguments(args)

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
    # The sandbox fault the call ended on, booked by type (infra or agent-caused) rather than as an
    # ordinary tool error, and ending the episode.
    sandbox_fault: SandboxInfraError | SandboxAgentFault | None = None

    def to_message(self):
        """Convert to a tool :class:`Message` for the conversation."""
        return Message.tool(self.content, self.tool_call_id, self.name)
