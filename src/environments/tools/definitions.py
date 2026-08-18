"""OpenAI-compatible tool definitions and registry (vLLM tool-calling format)."""

import ast
import asyncio
import json
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from src.environments.base import Message


class ToolBudgetExhausted(Exception):
    """Raised when an episode has spent its per-episode call budget for a tool.

    Expected control flow rather than a fault: the protocol records a tool error (so an over-cap call
    does not earn ``tool_success_reward``) but logs it without a traceback.
    """


def _parse_tool_arguments(raw: str) -> dict[str, Any]:
    """Parse a tool-call ``arguments`` string; ``ast.literal_eval`` repairs Python-dict literals (single
    quotes, trailing commas, ``True``/``None``). Unrecoverable / non-dict input falls back to ``{}``."""
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        try:
            parsed = ast.literal_eval(raw)
        except (ValueError, SyntaxError):
            return {}
    return parsed if isinstance(parsed, dict) else {}


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
    """Tool definition in OpenAI function-calling format (understood natively by vLLM)."""

    name: str
    description: str
    parameters: list[ToolParameter] = field(default_factory=list)
    handler: Callable[..., Any] | None = None
    async_handler: Callable[..., Any] | None = None

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

    def bind_arguments(self, arguments: dict[str, Any]) -> dict[str, Any]:
        """Keep only the arguments this tool declares in :attr:`parameters`.

        The argument dict is model-authored and the handlers are ``functools.partial`` objects
        carrying pre-bound safety keywords (sandbox ``timeout``, ``allow_imports``); a call-time
        keyword of the same name would override them, raising the execution timeout or lifting the
        import guard. Filtering against the declared schema (the set advertised in
        :meth:`to_openai_schema`) drops any argument the tool does not declare.

        A tool that declares no parameters has no schema to filter against (an MCP server may
        advertise a tool without ``properties``), so its arguments pass through unchanged.
        """
        if not self.parameters:
            return arguments
        declared = {parameter.name for parameter in self.parameters}
        return {name: value for name, value in arguments.items() if name in declared}

    @staticmethod
    def _as_text(result: Any) -> str:
        """Render a handler's return value as the string a tool observation must be."""
        return result if isinstance(result, str) else str(result)

    def execute(self, **kwargs) -> str:
        """Execute the tool synchronously."""
        if self.handler:
            return self._as_text(self.handler(**self.bind_arguments(kwargs)))
        raise NotImplementedError(f"Tool '{self.name}' has no sync handler")

    async def execute_async(self, **kwargs) -> str:
        """Execute the tool asynchronously."""
        bound = self.bind_arguments(kwargs)
        if self.async_handler:
            return self._as_text(await self.async_handler(**bound))
        if self.handler:
            return self._as_text(await asyncio.to_thread(self.handler, **bound))
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
        """The observation returned for a call naming a tool that is not registered.

        The registered names are listed, sorted and comma-separated, so a model that invented a tool
        name can correct the call from the observation it already receives.
        """
        return f"Error: Unknown tool '{name}'. Available tools: {', '.join(sorted(self.names()))}"

    def to_openai_tools(self) -> list[dict[str, Any]]:
        """OpenAI function-calling schemas, passed to vLLM via the ``tools`` parameter."""
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

        A malformed payload (explicit ``"function": null`` or a non-dict) degrades to an empty-name
        call, which the registry rejects as an unknown tool, rather than raising an AttributeError
        that would end the episode.
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
    # Set when the registry held no such tool, so nothing ran. Determined structurally rather than by
    # matching the observation text: a registered tool can produce the same wording (an MCP server
    # answering "Tool not found: x" is a genuine failure of a real tool), and the protocol drops a
    # turn from training on this flag alone.
    unknown_tool: bool = False

    def to_message(self):
        """Convert to a tool :class:`Message` for the conversation."""
        return Message.tool(self.content, self.tool_call_id, self.name)
