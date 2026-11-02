"""MCP (Model Context Protocol) client environment: connects to MCP servers and discovers their tools
in OpenAI-compatible format."""

import asyncio
import logging
import os
import shutil
import tempfile
from collections.abc import Awaitable, Callable
from datetime import timedelta
from typing import Any

from mcp import ClientSession, StdioServerParameters
from mcp.client.sse import sse_client
from mcp.client.stdio import get_default_environment, stdio_client

from halo.env import env_str
from halo.environments.protocols.native_tool_use import AsyncNativeToolUseEnvironment
from halo.environments.tools.definitions import NativeTool, NativeToolRegistry, ToolParameter

logger = logging.getLogger(__name__)

# Cap on the cross-thread wait for the connection owner to unwind (stdio subprocess teardown). Bounded
# because ``close()`` runs on the rollout shutdown path, where a wedged server must not hold the rank.
_MCP_DISCONNECT_TIMEOUT_S = 30.0

# Deadline on a single MCP request: the SDK waits forever by default, so a hung server would hold the episode
# to its deadline; sized above a slow fetch/search round trip, a hung call returns as one tool error.
MCP_REQUEST_TIMEOUT_S = 120.0

MCP_SERVERS = {
    "brave_search": {
        "command": "npx",
        "args": ["-y", "@modelcontextprotocol/server-brave-search"],
        "env": ["BRAVE_API_KEY"],
    },
    "filesystem": {
        "command": "npx",
        # A fresh scratch directory, appended as the trailing path, is the server's entire access allowlist: a
        # rollout reaches no file of the host, another process or (under episode_scratch) another episode.
        "args": ["-y", "@modelcontextprotocol/server-filesystem"],
        "env": [],
        "scratch_allowlist": True,
    },
    "fetch": {
        # Python package (uvx), not npm.
        "command": "uvx",
        "args": ["mcp-server-fetch"],
        "env": [],
    },
    "memory": {
        "command": "npx",
        "args": ["-y", "@modelcontextprotocol/server-memory"],
        "env": [],
        # Unset, the server keeps one graph file inside its npm package, which every episode and every run on
        # the host reads and writes; a file in the server's own scratch directory scopes the graph with it.
        "scratch_env": {"MEMORY_FILE_PATH": "memory.jsonl"},
    },
    "github": {
        "command": "npx",
        "args": ["-y", "@modelcontextprotocol/server-github"],
        "env": ["GITHUB_PERSONAL_ACCESS_TOKEN"],
    },
    "slack": {
        "command": "npx",
        "args": ["-y", "@modelcontextprotocol/server-slack"],
        "env": ["SLACK_BOT_TOKEN", "SLACK_TEAM_ID"],
    },
}


class _ServerConnection:
    """One spawned MCP server, held open by an owner task that enters and exits its transport and session contexts,
    as their anyio cancel scopes require. With ``scratch`` the transport is handed a fresh directory under the temp
    dir, kept across a respawn of a server that died and removed by :meth:`stop`; ``on_ready`` runs on the session
    before :meth:`start` returns."""

    def __init__(self, *, scratch: bool, on_ready: Callable[[ClientSession], Awaitable[None]] | None = None):
        self.session: ClientSession | None = None
        self._scratch = scratch
        self._on_ready = on_ready
        self._scratch_dir: str | None = None
        self._task: asyncio.Task | None = None
        self._stop: asyncio.Event | None = None
        self._lock = asyncio.Lock()

    @property
    def is_open(self) -> bool:
        return self._task is not None and not self._task.done()

    @property
    def loop(self) -> asyncio.AbstractEventLoop | None:
        """The loop running the owner task, ``None`` while no server runs."""
        return self._task.get_loop() if self.is_open else None

    async def start(self, transport_context: Callable[[str | None], Any]) -> ClientSession:
        """The server's session, starting it first unless it runs; concurrent callers share one start, and a failed
        start raises leaving no half-started server behind."""
        async with self._lock:
            if not self.is_open:
                await self._spawn(transport_context)
        return self.session

    async def _run(self, transport_context: Callable[[str | None], Any], ready: asyncio.Future, stop: asyncio.Event):
        try:
            async with (
                transport_context(self._scratch_dir) as (read, write),
                ClientSession(read, write, read_timeout_seconds=timedelta(seconds=MCP_REQUEST_TIMEOUT_S)) as session,
            ):
                await session.initialize()
                if self._on_ready is not None:
                    await self._on_ready(session)
                self.session = session
                ready.set_result(None)
                await stop.wait()
        except Exception as e:
            if not ready.done():
                ready.set_exception(e)
            else:
                logger.exception("MCP connection task failed")
        finally:
            self.session = None

    async def _spawn(self, transport_context: Callable[[str | None], Any]) -> None:
        if self._scratch and self._scratch_dir is None:
            self._scratch_dir = tempfile.mkdtemp(prefix="halo-mcp-")
        stop = asyncio.Event()
        ready: asyncio.Future = asyncio.get_running_loop().create_future()
        task = asyncio.create_task(self._run(transport_context, ready, stop))
        try:
            await ready
        except BaseException:
            # A start cancelled mid-connect (the episode deadline) must not leave the owner task and its server running.
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
            raise
        self._task, self._stop = task, stop

    async def stop(self) -> None:
        """Stop the server and wait for its owner task to unwind (idempotent), then remove its scratch directory."""
        task, self._task = self._task, None
        if task is not None:
            self._stop.set()
            try:
                await task
            except Exception:
                logger.exception("MCP disconnect failed")
        self._remove_scratch()

    def forget(self) -> None:
        """Drop a server whose loop can run no teardown (already unwound, or stopped), removing its scratch."""
        self._task = None
        self._remove_scratch()

    def _remove_scratch(self) -> None:
        if self._scratch_dir is not None:
            shutil.rmtree(self._scratch_dir, ignore_errors=True)
            self._scratch_dir = None


class NativeMCPClientEnvironment(AsyncNativeToolUseEnvironment):
    """Environment connecting to MCP servers via the MCP Python SDK, exposing the discovered tools in
    native OpenAI format."""

    # Twice the native per-call rate; also stated in
    # agent-docs/training-methods/async-grpo/environments/protocols/mcp.md, which must be updated alongside it.
    DEFAULT_TOOL_SUCCESS_REWARD = 0.1
    TOOLS_ARRIVE_ON_RESET = True

    def __init__(
        self,
        server_command: str | None = None,
        server_args: list[str] | None = None,
        server_env: dict[str, str] | None = None,
        server_url: str | None = None,
        transport: str = "stdio",
        scratch_allowlist: bool = False,
        scratch_env: dict[str, str] | None = None,
        episode_scratch: bool = False,
        **kwargs,
    ):
        """A stdio server can take a fresh scratch directory, removed when the server stops: ``scratch_allowlist``
        appends it to the server's arguments (the filesystem server's access allowlist), and ``scratch_env`` maps
        environment variables to file names inside it (the memory server's graph file). ``episode_scratch``
        scopes it to the episode: each episode starts a server of its own over its own directory, stopped with
        the episode; otherwise one server and its directory serve every episode."""
        super().__init__(tool_registry=NativeToolRegistry(), **kwargs)
        takes_scratch = scratch_allowlist or bool(scratch_env)
        if transport != "stdio" and (takes_scratch or episode_scratch):
            raise ValueError(
                f"scratch_allowlist, scratch_env and episode_scratch give a spawned server its scratch directory, "
                f"but transport={transport!r} connects to one running server and spawns none. Use the stdio "
                f"transport, or drop them."
            )
        if overlap := sorted(set(scratch_env or {}) & set(server_env or {})):
            raise ValueError(
                f"server_env sets {overlap}, which scratch_env would point into the scratch directory instead. "
                f"Drop the name from one of the two."
            )
        if episode_scratch and not takes_scratch:
            presets = sorted(
                name
                for name, config in MCP_SERVERS.items()
                if config.get("scratch_allowlist") or config.get("scratch_env")
            )
            raise ValueError(
                f"episode_scratch scopes the server's scratch directory to the episode, but this server takes "
                f"none (of the presets, {presets} take one). Drop episode_scratch."
            )

        self.transport = transport
        self.server_command = server_command
        self.server_args = server_args or []
        self.server_env = server_env
        self.server_url = server_url
        self.scratch_allowlist = scratch_allowlist
        self.scratch_env = scratch_env or {}
        self.episode_scratch = episode_scratch

        # Discovers the tools on the first episode; under episode_scratch it serves no call.
        self._server = _ServerConnection(scratch=takes_scratch, on_ready=self._register_tools)
        self._episode_servers: dict[int, _ServerConnection] = {}

    @property
    def _connected(self) -> bool:
        return self._server.is_open

    async def _reset_single_async(self, prompt, context=None):
        # No caller enters the env as a context manager, so connect lazily on the first episode.
        await self.connect()
        return await super()._reset_single_async(prompt, context)

    def _transport_context(self, scratch_dir: str | None):
        """Build the transport's async context manager (stdio subprocess or SSE stream); a stdio spawn is handed
        ``scratch_dir`` as its ``scratch_allowlist`` argument and its ``scratch_env`` files."""
        if self.transport == "stdio":
            if not self.server_command:
                raise ValueError("server_command required for stdio transport")
            args = list(self.server_args)
            server_env = dict(self.server_env or {})
            if scratch_dir is not None:
                if self.scratch_allowlist:
                    args.append(scratch_dir)
                server_env.update({name: os.path.join(scratch_dir, file) for name, file in self.scratch_env.items()})
            # The SDK passes ``env`` verbatim when set, dropping PATH/HOME and breaking the spawn.
            env = {**get_default_environment(), **server_env} if server_env else None
            return stdio_client(StdioServerParameters(command=self.server_command, args=args, env=env))
        if self.transport == "sse":
            if not self.server_url:
                raise ValueError("server_url required for sse transport")
            return sse_client(self.server_url)
        raise ValueError(f"Unknown transport: {self.transport}")

    async def _register_tools(self, session: ClientSession) -> None:
        for mcp_tool in (await session.list_tools()).tools:
            self.registry.register(self._build_mcp_tool(mcp_tool))
        logger.info(f"MCP connected: {len(self.registry)} tools discovered")

    async def connect(self) -> None:
        """Start the environment's server; returns once its tools are discovered (raises on failure)."""
        await self._server.start(self._transport_context)

    async def _call_session(self) -> ClientSession:
        """The session a tool call goes to: under ``episode_scratch`` the executing episode's own server, started on
        its first call, else the environment's."""
        server = self._server
        if self.episode_scratch:
            trajectory = self.active_trajectory()
            if trajectory is None:
                raise RuntimeError("An episode-scoped MCP tool was called outside an episode")
            server = self._episode_servers.setdefault(trajectory.info["episode_id"], _ServerConnection(scratch=True))
            await server.start(self._transport_context)
        if server.session is None:
            raise RuntimeError("MCP session not connected")
        return server.session

    def _build_mcp_tool(self, mcp_tool: Any) -> NativeTool:
        """A NativeTool from an MCP tool definition."""
        name = mcp_tool.name
        parameters = []
        schema = mcp_tool.inputSchema
        properties = schema.get("properties", {})
        required = schema.get("required", [])

        for param_name, param_info in properties.items():
            parameters.append(
                ToolParameter(
                    name=param_name,
                    type=param_info.get("type"),
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
        """Call an MCP tool and return its text content; a failure raises, since an error string would be booked
        as a successful call."""
        session = await self._call_session()
        result = await session.call_tool(name, arguments=arguments)

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

    def _release_episode(self, episode_id: int) -> None:
        """Stop the episode's own server, removing its scratch directory."""
        server = self._episode_servers.pop(episode_id, None)
        if server is not None:
            self._run_or_schedule(server.stop())

    def _servers(self) -> list[_ServerConnection]:
        return [self._server, *self._episode_servers.values()]

    async def disconnect(self) -> None:
        """Stop every server and wait for each to unwind (idempotent)."""
        servers = self._servers()
        self._episode_servers.clear()
        await asyncio.gather(*(server.stop() for server in servers))
        self.registry = NativeToolRegistry()

    def close(self) -> None:
        """Drive :meth:`disconnect` from the sync lifecycle hook on the loop owning the servers: scheduled as a task
        when called from that loop, which it cannot block, else awaited across threads."""
        loop = next((server.loop for server in self._servers() if server.is_open), None)
        if loop is None:
            # Already unwound: ``asyncio.run`` cancels the tasks its loop leaves behind (a CLI eval).
            self._forget_servers()
        else:
            try:
                running = asyncio.get_running_loop()
            except RuntimeError:
                running = None
            if running is loop:
                self._run_or_schedule(self.disconnect())
            elif loop.is_running():
                asyncio.run_coroutine_threadsafe(self.disconnect(), loop).result(timeout=_MCP_DISCONNECT_TIMEOUT_S)
            else:
                logger.warning("MCP close(): owning event loop already stopped; connection not unwound")
                self._forget_servers()
        super().close()

    def _forget_servers(self) -> None:
        for server in self._servers():
            server.forget()
        self._episode_servers.clear()


def get_mcp_server_config(name: str) -> dict[str, Any]:
    """Get MCP server configuration by name."""
    if name not in MCP_SERVERS:
        available = list(MCP_SERVERS.keys())
        raise ValueError(f"Unknown MCP server: {name}. Available: {available}")
    return MCP_SERVERS[name]


def _resolve_required_env(server_name: str, var_names: list[str]) -> dict[str, str]:
    """A server's declared credential env vars from the process environment; a missing one raises here rather
    than as an opaque tool failure mid-episode."""
    missing = sorted(name for name in var_names if not env_str(name))
    if missing:
        raise ValueError(
            f"MCP server '{server_name}' requires environment variable(s) {', '.join(missing)}; "
            "set them (or pass env_vars) before constructing the environment."
        )
    return {name: env_str(name, "") for name in var_names}


def build_mcp_environment(
    mcp_server: str | None = None, env_vars: dict[str, str] | None = None, **kwargs
) -> NativeMCPClientEnvironment:
    """A NativeMCPClientEnvironment for the stdio preset ``mcp_server`` names in :data:`MCP_SERVERS`, its declared
    credentials read from the process environment unless ``env_vars`` sets them, or, under ``transport="sse"``,
    for the running server at ``server_url``."""
    if kwargs.get("transport") == "sse":
        if mcp_server is not None or env_vars:
            raise ValueError(
                f"transport='sse' connects to a running server at server_url, so it spawns nothing: "
                f"the stdio preset (mcp_server={mcp_server!r}) and its credential env_vars would be "
                f"silently ignored. Drop them, or drop transport='sse' to spawn the preset."
            )
        return NativeMCPClientEnvironment(**kwargs)
    # A config naming no server gets the preset with no credentials and no network, held to its own scratch dir.
    mcp_server = mcp_server or "filesystem"
    config = get_mcp_server_config(mcp_server)
    explicit = dict(env_vars or {})
    declared = [name for name in config["env"] if name not in explicit]
    server_env = {**_resolve_required_env(mcp_server, declared), **explicit}
    scratch_allowlist = config.get("scratch_allowlist", False)
    # A file env_vars names outranks the preset's scratch file: a seeded MEMORY_FILE_PATH is one graph, served
    # to every episode by one server.
    preset_scratch_env = config.get("scratch_env", {})
    if (seeded := sorted(set(preset_scratch_env) & set(explicit))) and kwargs.get("episode_scratch"):
        raise ValueError(
            f"env_vars sets {seeded}, one file every episode's own server would read and write at once under "
            f"episode_scratch=True. Drop episode_scratch to serve that file from one server, or drop the env_vars "
            f"entry to give each episode a file of its own."
        )
    scratch_env = {name: file for name, file in preset_scratch_env.items() if name not in explicit}
    return NativeMCPClientEnvironment(
        server_command=config["command"],
        server_args=config["args"],
        server_env=server_env or None,
        scratch_allowlist=scratch_allowlist,
        scratch_env=scratch_env,
        # A preset's scratch directory is the episode's unless the config scopes it to the environment.
        episode_scratch=kwargs.pop("episode_scratch", scratch_allowlist or bool(scratch_env)),
        **kwargs,
    )
