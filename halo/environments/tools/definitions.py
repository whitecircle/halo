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

from halo.environments.episodes.messages import Message
from halo.environments.sandbox.results import SandboxFault

# What reading model-authored arguments as JSON or a Python literal raises past a syntax error: an unhashable key
# (``{[1]: 2}``) is a TypeError, deep nesting a RecursionError or MemoryError, an integer past Python's 4300-digit
# limit a ValueError outside ``JSONDecodeError``.
MALFORMED_LITERAL_ERRORS = (SyntaxError, ValueError, TypeError, MemoryError, RecursionError)


class ToolBudgetExhausted(Exception):
    """The episode spent this tool's budget: booked as a tool error (it never earns ``tool_success_reward``), logged
    without a traceback."""


class ToolDisabled(ToolBudgetExhausted):
    """A zero episode budget: the run disabled the tool. Answered like a spent budget, no refusal
    (:func:`is_refused_unrun`): where the run offers no such tool the call simply failed, so its turn stays
    trainable."""


class ToolArgumentError(TypeError):
    """Model-authored arguments the tool refuses, raised before the handler runs (or by a handler,
    :class:`ToolCallRefused`), so the model reads ``Error: <tool>: missing a required argument: 'code'`` instead of a
    Python signature; a value ending in the closing tag of a parameter that refuses one (``stdin`` ending
    ``</stdin>``) is refused too. Booked like :class:`ToolBudgetExhausted`."""


class ToolCallRefused(ToolArgumentError):
    """A handler refusing an admitted call, returned to the budget (code contests: a program evidently in another
    language than its call names)."""


# What refuses a call unrun: expected control flow, logged without a traceback.
TOOL_CALL_REFUSALS = (ToolBudgetExhausted, ToolArgumentError)


def is_refused_unrun(outcome: object) -> bool:
    """Whether ``outcome`` refused the call unrun, a disabled tool aside. A turn whose every call was refused or unknown
    is flagged ``calls_rejected``, so an episode that recovers never reinforces it."""
    return isinstance(outcome, TOOL_CALL_REFUSALS) and not isinstance(outcome, ToolDisabled)


def parse_python_expression(source: str) -> ast.expr:
    """Parse without the ``SyntaxWarning`` an invalid escape in a model-written string (``"\\d"``) prints."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", SyntaxWarning)
        return ast.parse(source.lstrip(" \t"), mode="eval").body


@dataclass
class ToolParameter:
    """``type`` is the JSON-schema type, ``None`` where a schema declares none (an MCP property typed through ``anyOf``
    or ``$ref``), which takes any value. ``refuse_closing_tag`` refuses a ``string`` value ending in ``</name>``, the
    call syntax leaked into it: opt-in, since a parameter of an external tool may legitimately end in markup (an
    ``svg`` in ``</svg>``)."""

    name: str
    type: str | None
    description: str
    enum: list[str] | None = None
    required: bool = True
    refuse_closing_tag: bool = False


@dataclass
class NativeTool:
    name: str
    description: str
    parameters: list[ToolParameter] = field(default_factory=list)
    handler: Callable[..., Any] | None = None
    async_handler: Callable[..., Any] | None = None
    # Observation for a call over the episode budget; ``None`` takes the generic wording.
    budget_message: str | None = None
    # A JSON-schema object shown to the engine as ``parameters`` in place of one built from ``parameters`` (an
    # MCP server's ``inputSchema``), which keeps what a ToolParameter has no field for: array ``items``, nested
    # objects, ``anyOf``, ``$defs``. Admission still checks ``parameters``.
    input_schema: dict[str, Any] | None = None

    def to_openai_schema(self) -> dict[str, Any]:
        """The function declaration the engine reads. ``input_schema`` wins when set, so a server-authored schema
        reaches the model with its array items, nested objects and references intact."""
        if self.input_schema is not None:
            return {
                "type": "function",
                "function": {"name": self.name, "description": self.description, "parameters": self.input_schema},
            }
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
        """Admit a call's arguments or raise :class:`ToolArgumentError`. The protocols call it before spending the
        budget, so a call the handler could never run is refused uncounted; ``for_async`` binds what
        :meth:`execute_async` runs."""
        handler = (self.async_handler or self.handler) if for_async else self.handler
        if handler is None:
            raise NotImplementedError(f"Tool '{self.name}' has no handler")
        return self._bind_for_call(handler, arguments)

    def _bind_for_call(self, handler: Callable[..., Any], arguments: dict[str, Any]) -> dict[str, Any]:
        """Check against the schema and ``handler``'s signature. A name outside the schema is refused: dropped unread,
        a garbled name loses its value and a keyword shadowing a pre-bound safety one (the sandbox ``timeout``) would
        lift its cap. A tool declaring no parameters takes its arguments as given."""
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
                # A JSON scalar reads as its str() (an engine's parser may send ``"stdin": 5``) and a null as an
                # omission; a list ``code`` passed on would fail inside the sandbox, read as a backend fault.
                if value is None:
                    del bound[parameter.name]
                elif not isinstance(value, str | int | float):
                    raise ToolArgumentError(
                        f"{self.name}: {parameter.name} must be a string, got {type(value).__name__}"
                    )
                else:
                    bound[parameter.name] = value = str(value)
                    # Markup the call's own syntax leaked into the value: run as given, it is input the program was
                    # never meant to read, and stripped silently, a habit nothing prices.
                    if parameter.refuse_closing_tag and value.rstrip().endswith(f"</{parameter.name}>"):
                        raise ToolArgumentError(
                            f"{self.name}: not run — the {parameter.name} argument ends with a stray "
                            f"`</{parameter.name}>` tag, which is no part of its value; send it again without the tag"
                        )
            # The schema is the contract the model was shown: a missing required parameter or an out-of-enum value is
            # refused before the handler and the budget, even where the handler has its own default.
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
        """States the refusal, never the budget's size: what an episode may do is the chat template's and the engine's
        to control."""
        return self.budget_message or f"Not executed: this task's budget for {self.name} is spent."

    def execute(self, **kwargs) -> str:
        if self.handler:
            return self._as_text(self.handler(**self._bind_for_call(self.handler, kwargs)))
        raise NotImplementedError(f"Tool '{self.name}' has no sync handler")

    async def execute_async(self, **kwargs) -> str:
        if self.async_handler:
            return self._as_text(await self.async_handler(**self._bind_for_call(self.async_handler, kwargs)))
        if self.handler:
            return self._as_text(await asyncio.to_thread(self.handler, **self._bind_for_call(self.handler, kwargs)))
        raise NotImplementedError(f"Tool '{self.name}' has no handler")

    @staticmethod
    def _as_text(result: Any) -> str:
        return result if isinstance(result, str) else str(result)


class NativeToolRegistry:
    def __init__(self):
        self._tools: dict[str, NativeTool] = {}

    def register(self, tool: NativeTool) -> "NativeToolRegistry":
        self._tools[tool.name] = tool
        return self

    def get(self, name: str) -> NativeTool | None:
        return self._tools.get(name)

    def list_tools(self) -> list[NativeTool]:
        return list(self._tools.values())

    def names(self) -> list[str]:
        return list(self._tools.keys())

    def unknown_tool_message(self, name: str) -> str:
        """One wording for every protocol, naming the real tools as sorted prose: a policy drifting off the tool syntax
        invents plausible names and burns turns probing for a listing."""
        return f"Error: Unknown tool '{name}'. Available tools: {', '.join(sorted(self.names()))}"

    def to_openai_tools(self) -> list[dict[str, Any]]:
        return [tool.to_openai_schema() for tool in self._tools.values()]

    def merge(self, other: "NativeToolRegistry") -> "NativeToolRegistry":
        for tool in other.list_tools():
            self._tools[tool.name] = tool
        return self

    @classmethod
    def combine(cls, *registries: "NativeToolRegistry") -> "NativeToolRegistry":
        result = cls()
        for reg in registries:
            result.merge(reg)
        return result

    def __len__(self) -> int:
        return len(self._tools)


@dataclass
class NativeToolCall:
    id: str
    name: str
    arguments: dict[str, Any]

    @classmethod
    def from_openai_format(cls, tool_call: dict[str, Any]) -> "NativeToolCall":
        """A malformed payload degrades to an empty-name call the registry rejects as unknown, never an exception
        ending the episode."""
        function = tool_call.get("function")
        if not isinstance(function, dict):
            function = {}
        args = function.get("arguments") or "{}"

        if isinstance(args, str):
            # ``ast.literal_eval`` repairs Python-dict literals (single quotes, trailing commas, ``True``/``None``).
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
    tool_call_id: str
    name: str
    content: str
    success: bool = True
    # Nothing ran: the model invented it. Structural, since a tool's own failure can reproduce any wording, and the
    # protocol drops a turn from training on this flag alone.
    unknown_tool: bool = False
    # Refused and nothing ran (:func:`is_refused_unrun`); structural like ``unknown_tool``.
    refused: bool = False
    # Booked by type (infra or agent-caused) rather than as a tool error; it ends the episode.
    sandbox_fault: SandboxFault | None = None

    def to_message(self):
        return Message.tool(self.content, self.tool_call_id, self.name)
