"""MCP (Model Context Protocol) client environment: connects to MCP servers and auto-discovers tools in OpenAI-compatible format."""

import asyncio
import logging
from datetime import timedelta
from typing import Any

from mcp import ClientSession, StdioServerParameters
from mcp.client.sse import sse_client
from mcp.client.stdio import get_default_environment, stdio_client

from src.env import env_str
from src.environments.envs.protocols.native import AsyncNativeToolUseEnvironment
from src.environments.tools.definitions import NativeTool, NativeToolRegistry, ToolParameter

logger = logging.getLogger(__name__)

# Cap on the cross-thread wait for the connection owner to unwind (stdio subprocess teardown). Bounded
# because ``close()`` runs on the rollout shutdown path, where a wedged server must not hold the rank.
MCP_DISCONNECT_TIMEOUT_S = 30.0

# Deadline on ONE MCP request (initialize, tool discovery, a tool call). The SDK defaults to
# ``anyio.fail_after(None)``, so a server that ACKs and never answers parks the episode for the whole
# ``episode_timeout`` (20 min) and takes its rollout worker with it — with stdio there is not even a
# socket timeout underneath. Sized well above a slow fetch/search round trip and far below the episode
# deadline, so a hung call comes back as one tool error and the episode continues.
MCP_REQUEST_TIMEOUT_S = 120.0


class NativeMCPClientEnvironment(AsyncNativeToolUseEnvironment):
    """Environment connecting to MCP servers (via the MCP Python SDK), exposing discovered tools in native OpenAI format."""

    # Twice the native per-call rate: the one reward knob this env departs on, and the value
    # agent-docs/training-methods/grpo/environments/mcp.md states, so a change here has to follow.
    DEFAULT_TOOL_SUCCESS_REWARD = 0.1

    def __init__(
        self,
        server_command: str | None = None,
        server_args: list[str] | None = None,
        server_env: dict[str, str] | None = None,
        server_url: str | None = None,
        transport: str = "stdio",
        **kwargs,
    ):
        super().__init__(tool_registry=NativeToolRegistry(), **kwargs)

        self.transport = transport
        self.server_command = server_command
        self.server_args = server_args or []
        self.server_env = server_env
        self.server_url = server_url

        self._session = None
        self._conn_task: asyncio.Task | None = None
        self._close_task: asyncio.Task | None = None
        self._stop: asyncio.Event | None = None
        # Serialize the lazy first-episode connect so concurrent episodes don't each spawn a server.
        self._connect_lock = asyncio.Lock()

    @property
    def _connected(self) -> bool:
        return self._conn_task is not None and not self._conn_task.done()

    async def _reset_single_async(self, prompt, context=None):
        # No caller enters the env as a context manager, so connect lazily on the first episode.
        if not self._connected:
            async with self._connect_lock:
                if not self._connected:
                    await self.connect()
        return await super()._reset_single_async(prompt, context)

    def _transport_context(self):
        """Build the transport's async context manager (stdio subprocess or SSE stream)."""
        if self.transport == "stdio":
            if not self.server_command:
                raise ValueError("server_command required for stdio transport")
            # The SDK passes ``env`` verbatim when set, dropping PATH/HOME and breaking the spawn.
            env = {**get_default_environment(), **self.server_env} if self.server_env else None
            params = StdioServerParameters(command=self.server_command, args=self.server_args, env=env)
            return stdio_client(params)
        if self.transport == "sse":
            if not self.server_url:
                raise ValueError("server_url required for sse transport")
            return sse_client(self.server_url)
        raise ValueError(f"Unknown transport: {self.transport}")

    def _create_session(self, read, write):
        """Build the MCP ClientSession async context manager for a connected transport.

        ``read_timeout_seconds`` bounds every request the session makes (:data:`MCP_REQUEST_TIMEOUT_S`).
        """
        return ClientSession(read, write, read_timeout_seconds=timedelta(seconds=MCP_REQUEST_TIMEOUT_S))

    async def _connection_main(self, ready: asyncio.Future, stop: asyncio.Event) -> None:
        """Own the transport + session contexts in ONE task, from enter to exit.

        The MCP SDK's contexts hold anyio cancel scopes, which must be entered and exited by the same
        task — driving ``__aenter__``/``__aexit__`` from different tasks corrupts the scope stack.
        Any failure before ``ready`` resolves (spawn, initialize, discovery) unwinds the
        already-entered contexts via the ``async with`` stack, so a partial connect never leaks a
        half-initialized session or the server subprocess.
        """
        try:
            async with self._transport_context() as (read, write), self._create_session(read, write) as session:
                await session.initialize()
                self._discover_tools(await session.list_tools())
                self._session = session
                ready.set_result(None)
                await stop.wait()
        except Exception as e:
            if not ready.done():
                ready.set_exception(e)
            else:
                logger.exception("MCP connection task failed")
        finally:
            self._session = None

    def _discover_tools(self, list_tools_response) -> None:
        """Register the server's tools in native OpenAI format."""
        for mcp_tool in list_tools_response.tools:
            self.registry.register(self._create_mcp_tool(mcp_tool.name, mcp_tool))
        logger.info(f"MCP connected: {len(self.registry)} tools discovered")

    async def connect(self) -> None:
        """Start the connection-owner task; returns once tools are discovered (raises on failure)."""
        if self._connected:
            return
        self._stop = asyncio.Event()
        ready: asyncio.Future = asyncio.get_running_loop().create_future()
        task = asyncio.create_task(self._connection_main(ready, self._stop))
        try:
            await ready
        except Exception:
            await asyncio.gather(task, return_exceptions=True)  # ensure the partial connect unwound
            raise
        self._conn_task = task

    def _create_mcp_tool(self, name: str, mcp_tool: Any) -> NativeTool:
        """Create a NativeTool from an MCP tool definition."""
        parameters = []
        schema = getattr(mcp_tool, "inputSchema", {})
        properties = schema.get("properties", {})
        required = schema.get("required", [])

        for param_name, param_info in properties.items():
            parameters.append(
                ToolParameter(
                    name=param_name,
                    type=param_info.get("type", "string"),
                    description=param_info.get("description", ""),
                    enum=param_info.get("enum"),
                    required=param_name in required,
                )
            )

        async def mcp_handler(**kwargs) -> str:
            return await self._call_mcp_tool(name, kwargs)

        return NativeTool(
            name=name,
            description=mcp_tool.description or f"MCP tool: {name}",
            parameters=parameters,
            async_handler=mcp_handler,
        )

    async def _call_mcp_tool(self, name: str, arguments: dict[str, Any]) -> str:
        """Call an MCP tool and return its text content.

        Raises on failure so the wrapper marks ``success=False``; returning an error string would count
        as a successful call and wrongly earn the reward.
        """
        if not self._session:
            raise RuntimeError("MCP session not connected")

        try:
            result = await self._session.call_tool(name, arguments=arguments)

            content_parts = []
            for c in result.content:
                if hasattr(c, "text"):
                    content_parts.append(c.text)
                elif hasattr(c, "data"):
                    content_parts.append(f"[Binary data: {len(c.data)} bytes]")
                else:
                    content_parts.append(str(c))

            content = "\n".join(content_parts)
            if result.isError:
                raise RuntimeError(content or f"MCP tool {name} returned an error")
            return content

        except Exception:
            logger.exception(f"MCP tool error: {name}")
            raise

    async def disconnect(self) -> None:
        """Signal the connection-owner task to unwind its contexts and wait for it (idempotent)."""
        task, self._conn_task = self._conn_task, None
        if task is None:
            return
        if self._stop is not None:
            self._stop.set()
        try:
            await task
        except Exception:
            logger.exception("MCP disconnect failed")
        self.registry = NativeToolRegistry()

    def close(self) -> None:
        """Drive :meth:`disconnect` from the sync lifecycle hook (``EnvironmentActor.shutdown``).

        Without this override the base no-op ``close`` leaks the ClientSession and the stdio server
        subprocess on every rollout shutdown. Teardown must run on the loop that owns the connection
        task; when called FROM that loop (the actor's async shutdown) it cannot block, so the
        disconnect is scheduled as a task and a reference is kept until it completes.
        """
        task = self._conn_task
        if task is not None:
            loop = task.get_loop()
            try:
                running = asyncio.get_running_loop()
            except RuntimeError:
                running = None
            if running is loop:
                self._close_task = loop.create_task(self.disconnect())
            elif loop.is_running():
                asyncio.run_coroutine_threadsafe(self.disconnect(), loop).result(timeout=MCP_DISCONNECT_TIMEOUT_S)
            else:
                logger.warning("MCP close(): owning event loop already stopped; connection not unwound")
                self._conn_task = None
        super().close()


MCP_SERVERS = {
    "brave_search": {
        "command": "npx",
        "args": ["-y", "@modelcontextprotocol/server-brave-search"],
        "env": ["BRAVE_API_KEY"],
        "description": "Brave web search",
    },
    "filesystem": {
        "command": "npx",
        # The trailing path is the server's ENTIRE access allowlist — it refuses anything outside it —
        # so this is the policy's jail, not a scratch-dir preference. Deliberately NOT HALO_DATA_ROOT:
        # that volume holds the dataset cache and checkpoints, which a rollout must not be able to rewrite.
        "args": ["-y", "@modelcontextprotocol/server-filesystem", "/tmp"],
        "env": [],
        "description": "File system operations",
    },
    "fetch": {
        # Python package (uvx), not npm.
        "command": "uvx",
        "args": ["mcp-server-fetch"],
        "env": [],
        "description": "HTTP fetch for web content",
    },
    "memory": {
        "command": "npx",
        "args": ["-y", "@modelcontextprotocol/server-memory"],
        "env": [],
        "description": "Knowledge graph memory",
    },
    "github": {
        "command": "npx",
        "args": ["-y", "@modelcontextprotocol/server-github"],
        "env": ["GITHUB_TOKEN"],
        "description": "GitHub API",
    },
    "slack": {
        "command": "npx",
        "args": ["-y", "@modelcontextprotocol/server-slack"],
        "env": ["SLACK_TOKEN"],
        "description": "Slack messaging",
    },
}


# Preset a config gets when it names none: keyless, no network, and jailed to /tmp.
DEFAULT_MCP_SERVER = "filesystem"


def get_mcp_server_config(name: str) -> dict[str, Any]:
    """Get MCP server configuration by name."""
    if name not in MCP_SERVERS:
        available = list(MCP_SERVERS.keys())
        raise ValueError(f"Unknown MCP server: {name}. Available: {available}")
    return MCP_SERVERS[name]


def _resolve_required_env(server_name: str, var_names: list[str]) -> dict[str, str]:
    """Resolve a server's declared credential env vars from the process environment, failing loud
    at construction — a missing key otherwise surfaces only as an opaque tool failure mid-episode."""
    missing = sorted(name for name in var_names if not env_str(name))
    if missing:
        raise ValueError(
            f"MCP server '{server_name}' requires environment variable(s) {', '.join(missing)}; "
            "set them (or pass env_vars) before constructing the environment."
        )
    return {name: env_str(name, "") for name in var_names}


def create_native_mcp_environment(
    server_name: str | None = None, env_vars: dict[str, str] | None = None, **kwargs
) -> NativeMCPClientEnvironment:
    """Create a NativeMCPClientEnvironment for a known MCP server.

    ``server_name`` selects a stdio preset from :data:`MCP_SERVERS` (default
    :data:`DEFAULT_MCP_SERVER`). A remote server is reached with ``transport="sse"`` and
    ``server_url=...`` instead: there is no subprocess to spawn, so no preset describes it.

    The preset's declared credential requirements (``MCP_SERVERS[name]["env"]``) are resolved from
    the process environment and forwarded to the server subprocess; explicit ``env_vars`` entries
    both satisfy and override a declared name. A missing credential raises here, not at the first
    tool call."""
    if kwargs.get("transport") == "sse":
        if server_name is not None or env_vars:
            raise ValueError(
                f"transport='sse' connects to a running server at server_url, so it spawns nothing: "
                f"the stdio preset (mcp_server={server_name!r}) and its credential env_vars would be "
                f"silently ignored. Drop them, or drop transport='sse' to spawn the preset."
            )
        return NativeMCPClientEnvironment(**kwargs)
    server_name = server_name or DEFAULT_MCP_SERVER
    config = get_mcp_server_config(server_name)
    explicit = dict(env_vars or {})
    declared = [name for name in config["env"] if name not in explicit]
    server_env = {**_resolve_required_env(server_name, declared), **explicit}
    return NativeMCPClientEnvironment(
        server_command=config["command"],
        server_args=config["args"],
        server_env=server_env or None,
        **kwargs,
    )
