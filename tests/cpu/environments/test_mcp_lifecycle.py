#!/usr/bin/env python
"""CPU tests for the NativeMCPClientEnvironment connection lifecycle (no mcp package needed).

The MCP SDK's transport/session contexts hold anyio cancel scopes, which must be entered and exited
by the SAME task. These tests stub the transport and session to pin the lifecycle contract:

- one connection-owner task enters AND exits both contexts (task affinity);
- a failure mid-connect (``initialize`` raising) unwinds the already-entered transport, so a partial
  connect never leaks a half-initialized session or the server subprocess;
- the sync ``close()`` hook (what ``EnvironmentActor.shutdown`` calls) actually drives disconnect —
  a no-op there leaks the ClientSession and the stdio subprocess on every rollout shutdown;
- a start cancelled mid-connect (the episode deadline) unwinds the server it was starting;
- ``scratch_allowlist`` (the filesystem preset) gives a server a fresh directory as its allowlist, kept across a
  respawn and removed when the server stops, and ``episode_scratch`` scopes it: each episode gets a server of
  its own over its own directory, so concurrent episodes (a GRPO group's siblings) share no files, while
  without it one server and directory serve the environment. A server that takes no directory refuses the
  scope instead of being handed an argument it rejects.

Run: python tests/cpu/environments/test_mcp_lifecycle.py  (or pytest)
"""

import asyncio
import importlib
import logging
import os
import types
from datetime import timedelta

import pytest

import halo.environments.protocols.mcp as mcp_module
from halo.environments.protocols.mcp import NativeMCPClientEnvironment, build_mcp_environment
from halo.environments.registry import build_environment
from halo.environments.rollout_config import RolloutConfig


class _StubTransport:
    """Async CM standing in for stdio_client/sse_client; records the spawn's extra args, enter/exit and their
    owning tasks, and hands the session stub on as its read stream."""

    def __init__(self, session: "_StubSession", args: list[str]):
        self.session = session
        self.args = list(args)
        self.entered = False
        self.exited = False
        self.enter_task = None
        self.exit_task = None

    async def __aenter__(self):
        self.entered = True
        self.enter_task = asyncio.current_task()
        return (self, "write")

    async def __aexit__(self, *exc):
        self.exited = True
        self.exit_task = asyncio.current_task()
        return False


class _StubSession:
    """Async CM standing in for mcp.ClientSession; a call answers with the scratch directory of its server, and
    ``initialize_gate`` holds ``initialize`` (which sets ``initializing``) until it is set."""

    def __init__(self, fail_initialize: bool = False, initialize_gate: asyncio.Event | None = None):
        self.fail_initialize = fail_initialize
        self.initialize_gate = initialize_gate
        self.initializing = asyncio.Event()
        self.entered = False
        self.exited = False
        self.args: list[str] = []

    async def __aenter__(self):
        self.entered = True
        return self

    async def __aexit__(self, *exc):
        self.exited = True
        return False

    async def initialize(self):
        self.initializing.set()
        if self.initialize_gate is not None:
            await self.initialize_gate.wait()
        if self.fail_initialize:
            raise RuntimeError("initialize failed")

    async def list_tools(self):
        tool = types.SimpleNamespace(name="stub_tool", description="a stub", inputSchema={})
        return types.SimpleNamespace(tools=[tool])

    async def call_tool(self, name, arguments):
        text = f"served from {self.args[-1] if self.args else None}"
        return types.SimpleNamespace(content=[types.SimpleNamespace(type="text", text=text)], isError=False)


class _StubMCPEnv(NativeMCPClientEnvironment):
    """MCP env with the SDK transport stubbed out; the first server takes ``stub_session``, every later one a
    fresh session, and :func:`_stub_client_session` hands each transport's session to the SDK seam."""

    def __init__(self, fail_initialize: bool = False, initialize_gate: asyncio.Event | None = None, **kwargs):
        super().__init__(server_command="unused", **kwargs)
        self.stub_session = _StubSession(fail_initialize=fail_initialize, initialize_gate=initialize_gate)
        self.transports: list[_StubTransport] = []

    @property
    def stub_transport(self) -> _StubTransport:
        return self.transports[0]

    def _transport_context(self, scratch_dir):
        session = _StubSession() if self.transports else self.stub_session
        session.args = [scratch_dir] if scratch_dir else []
        self.transports.append(_StubTransport(session, session.args))
        return self.transports[-1]


@pytest.fixture(autouse=True)
def _stub_client_session(monkeypatch):
    """``mcp.ClientSession`` as the stub session of the transport that just opened."""

    def _session(read, write, **kwargs):
        assert write == "write"
        return read.session

    monkeypatch.setattr(mcp_module, "ClientSession", _session)


async def test_the_session_bounds_every_request_it_makes(monkeypatch):
    """A ClientSession built without ``read_timeout_seconds`` runs ``anyio.fail_after(None)``, so a
    server that ACKs and never answers parks the episode for the whole ``episode_timeout`` and takes
    its rollout worker with it — over stdio there is no socket timeout underneath either. The
    deadline must also sit BELOW the episode deadline, or it can never fire first."""
    seen: dict = {}
    env = _StubMCPEnv()

    def _record(read, write, **kwargs):
        seen.update(read=read, write=write, **kwargs)
        return env.stub_session

    monkeypatch.setattr(mcp_module, "ClientSession", _record)
    await env.connect()
    await env.disconnect()
    assert seen["read_timeout_seconds"] == timedelta(seconds=mcp_module.MCP_REQUEST_TIMEOUT_S)
    assert 0 < mcp_module.MCP_REQUEST_TIMEOUT_S < RolloutConfig.episode_timeout


async def test_connect_discovers_tools_and_disconnect_unwinds_in_owner_task():
    env = _StubMCPEnv()
    await env.connect()
    assert env._connected
    assert env.registry.get("stub_tool") is not None
    assert env.stub_transport.entered and env.stub_session.entered

    await env.disconnect()
    assert env.stub_session.exited and env.stub_transport.exited
    assert env._server.session is None and not env._connected
    assert env.registry.get("stub_tool") is None
    # anyio cancel-scope contract: enter and exit happened in the SAME task.
    assert env.stub_transport.enter_task is env.stub_transport.exit_task
    assert env.stub_transport.enter_task is not asyncio.current_task()  # owned by the connection task


async def test_failed_connect_unwinds_partial_state():
    """initialize() raising mid-connect must exit the already-entered transport and leave the env
    fully disconnected: a split enter leaves _session half-initialized and the transport open."""
    env = _StubMCPEnv(fail_initialize=True)
    with pytest.raises(RuntimeError, match="initialize failed"):
        await env.connect()
    assert env.stub_transport.entered and env.stub_transport.exited
    # A session entered but never exited holds the anyio cancel scope open, so the unwind must exit it.
    assert env.stub_session.entered and env.stub_session.exited
    assert env._server.session is None and not env._connected


async def test_a_start_cancelled_mid_connect_leaves_no_server_running():
    """The episode deadline cancels a tool call whose first use is still starting the server: the owner task, and
    the server subprocess its transport holds, must unwind with it rather than run for the actor's lifetime
    outside every stop."""
    env = _StubMCPEnv(initialize_gate=asyncio.Event(), scratch_allowlist=True)
    start = asyncio.create_task(env.connect())
    await asyncio.wait_for(env.stub_session.initializing.wait(), timeout=5)

    start.cancel()
    with pytest.raises(asyncio.CancelledError):
        await start
    owner = env.stub_transport.enter_task
    assert owner.done(), "the owner task outlived the cancelled start"
    assert env.stub_session.exited and env.stub_transport.exited
    assert not env._connected

    await env.disconnect()
    assert not os.path.exists(env.stub_transport.args[-1])


async def test_sync_close_from_running_loop_drives_disconnect():
    """close() called on the owning loop (EnvironmentActor.shutdown's env.close()) cannot block that
    loop, so it schedules disconnect; the scheduled task must actually unwind the connection."""
    env = _StubMCPEnv()
    await env.connect()

    env.close()
    assert env._background_tasks, "close() on the owning loop schedules the disconnect"
    await asyncio.gather(*env._background_tasks)
    assert env.stub_session.exited and env.stub_transport.exited
    assert env._server.session is None and not env._connected


async def test_disconnect_is_idempotent():
    env = _StubMCPEnv()
    await env.connect()
    await env.disconnect()
    await env.disconnect()
    assert not env._connected


def test_close_after_asyncio_run_unwound_the_connection_warns_nothing(caplog):
    """A CLI eval drives its episodes under ``asyncio.run``, which cancels the connection task, and so
    unwinds the session and the transport, before the script calls ``close()``: a warning that the
    connection was left open is false on every such run."""
    env = _StubMCPEnv()
    asyncio.run(env.connect())
    assert env.stub_session.exited and env.stub_transport.exited
    caplog.clear()
    with caplog.at_level(logging.WARNING, logger=mcp_module.__name__):
        env.close()
    assert [r for r in caplog.records if r.levelno >= logging.WARNING] == []
    assert not env._connected


def test_close_still_warns_when_a_stopped_loop_left_the_connection_open(caplog):
    env = _StubMCPEnv()
    loop = asyncio.new_event_loop()
    loop.run_until_complete(env.connect())
    task, stop = env._server._task, env._server._stop
    assert not task.done()
    caplog.clear()
    with caplog.at_level(logging.WARNING, logger=mcp_module.__name__):
        env.close()
    assert [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING] == [
        "MCP close(): owning event loop already stopped; connection not unwound"
    ]
    stop.set()
    loop.run_until_complete(task)
    loop.close()


async def test_a_failing_mcp_tool_is_logged_once_by_the_protocol(caplog):
    """The tool's failure reaches the log through the protocol that ran it, traceback included; the MCP
    client logging it again before re-raising would log every failed call twice."""
    env = _StubMCPEnv()

    async def _call_tool(name, arguments):
        raise RuntimeError("server down")

    env.stub_session.call_tool = _call_tool
    ids, _ = await env.reset_async(["task"])
    call = {"id": "c1", "function": {"name": "stub_tool", "arguments": "{}"}}
    with caplog.at_level(logging.DEBUG, logger="halo.environments"):
        await env.step_async(ids, ["calling"], [{"finish_reason": "tool_calls", "tool_calls": [call]}])
    await env.disconnect()
    logged = [r for r in caplog.records if r.levelno >= logging.WARNING]
    assert len(logged) == 1 and logged[0].exc_info is not None
    assert "server down" in env.get_trajectories(ids)[0].messages[-1].content


async def test_an_mcp_property_without_a_declared_type_takes_any_value():
    """A property typed through ``anyOf`` (an int-or-null) declares no top-level ``type``: the schema
    shown to the engine carries none, and ``3`` and ``null`` both reach the server as sent. Only a
    declared ``string`` is held to one, so a list there is refused before the server is called."""
    env = _StubMCPEnv()
    input_schema = {
        "properties": {
            "url": {"type": "string", "description": "page"},
            "max_length": {"anyOf": [{"type": "integer"}, {"type": "null"}], "default": None},
        },
        "required": ["url"],
    }

    async def list_tools():
        tool = types.SimpleNamespace(name="fetch", description="fetch a page", inputSchema=input_schema)
        return types.SimpleNamespace(tools=[tool])

    received = []

    async def call_tool(name, arguments):
        received.append(arguments)
        return types.SimpleNamespace(content=[types.SimpleNamespace(type="text", text="ok")], isError=False)

    env.stub_session.list_tools = list_tools
    env.stub_session.call_tool = call_tool
    ids, _ = await env.reset_async(["task"])
    (schema,) = env.get_tools_schema()
    assert schema["function"]["parameters"]["properties"]["max_length"] == {"description": ""}
    calls = [
        {"id": "a", "function": {"name": "fetch", "arguments": '{"url": "x", "max_length": 3}'}},
        {"id": "b", "function": {"name": "fetch", "arguments": '{"url": "x", "max_length": null}'}},
        {"id": "c", "function": {"name": "fetch", "arguments": '{"url": ["x"]}'}},
    ]
    await env.step_async(ids, ["calling"], [{"finish_reason": "tool_calls", "tool_calls": calls}])
    await env.disconnect()
    assert received == [{"url": "x", "max_length": 3}, {"url": "x", "max_length": None}]
    replies = [m.content for m in env.get_trajectories(ids)[0].messages if m.role == "tool"]
    assert replies[2] == "Error: fetch: url must be a string, got list"


async def test_episode_scratch_gives_each_episode_its_own_server_and_directory():
    """Concurrent episodes on one environment (a GRPO group's siblings) must not see each other's files: each
    episode's calls reach a server of its own over a fresh directory, stopped and removed with the episode."""
    env = _StubMCPEnv(scratch_allowlist=True, episode_scratch=True)
    ids, _ = await env.reset_async(["task a", "task b"])
    call = {"id": "c1", "function": {"name": "stub_tool", "arguments": "{}"}}
    await env.step_async(ids, ["calling", "calling"], [{"finish_reason": "tool_calls", "tool_calls": [call]}] * 2)

    discovery, *episode_servers = env.transports
    scratch_dirs = [transport.args[-1] for transport in episode_servers]
    assert len(set(scratch_dirs)) == 2 and all(os.path.isdir(d) for d in scratch_dirs)
    assert discovery.args[-1] not in scratch_dirs
    replies = {
        trajectory.info["episode_id"]: next(m.content for m in trajectory.messages if m.role == "tool")
        for trajectory in env.get_trajectories(ids)
    }
    assert sorted(replies.values()) == sorted(f"served from {d}" for d in scratch_dirs), replies

    env.cleanup(ids)
    await asyncio.gather(*env._background_tasks)
    assert all(transport.exited for transport in episode_servers)
    assert not any(os.path.exists(d) for d in scratch_dirs), "an episode's scratch directory outlived it"

    await env.disconnect()
    assert discovery.exited and not os.path.exists(discovery.args[-1])


async def _call_from_two_episodes(env: _StubMCPEnv) -> list:
    ids, _ = await env.reset_async(["task a", "task b"])
    call = {"id": "c1", "function": {"name": "stub_tool", "arguments": "{}"}}
    await env.step_async(ids, ["calling", "calling"], [{"finish_reason": "tool_calls", "tool_calls": [call]}] * 2)
    return ids


async def test_one_server_serves_every_episode_without_episode_scratch():
    env = _StubMCPEnv()
    await _call_from_two_episodes(env)
    await env.disconnect()
    assert len(env.transports) == 1 and env.transports[0].args == []


async def test_an_environment_scoped_scratch_allowlist_is_one_server_over_its_own_directory():
    """``episode_scratch: false`` on the filesystem preset shares one server, which still needs its directory
    allowlist: spawned without one, the server exits on its usage error (or denies every path)."""
    env = _StubMCPEnv(scratch_allowlist=True)
    ids = await _call_from_two_episodes(env)
    ((scratch,),) = [transport.args for transport in env.transports]
    assert os.path.isdir(scratch)
    replies = [next(m.content for m in t.messages if m.role == "tool") for t in env.get_trajectories(ids)]
    assert replies == [f"served from {scratch}"] * 2
    await env.disconnect()
    assert not os.path.exists(scratch)


async def test_a_respawned_episode_server_keeps_the_episodes_directory():
    """A server that died mid-episode is respawned by the episode's next call over the same directory, so the
    files the episode wrote before the crash are still there and the first directory is not orphaned."""
    env = _StubMCPEnv(scratch_allowlist=True, episode_scratch=True)
    ids, _ = await env.reset_async(["task"])
    call = {"id": "c1", "function": {"name": "stub_tool", "arguments": "{}"}}
    await env.step_async(ids, ["calling"], [{"finish_reason": "tool_calls", "tool_calls": [call]}])
    (server,) = env._episode_servers.values()
    scratch = env.transports[-1].args[-1]
    with open(os.path.join(scratch, "notes.txt"), "w") as f:
        f.write("written before the crash")

    server._stop.set()  # the server ends without a stop(), as one that crashed does
    await server._task
    await env.step_async(ids, ["calling again"], [{"finish_reason": "tool_calls", "tool_calls": [call]}])
    assert len(env.transports) == 3 and env.transports[-1].args == [scratch]
    assert os.path.exists(os.path.join(scratch, "notes.txt"))

    env.cleanup(ids)
    await asyncio.gather(*env._background_tasks)
    await env.disconnect()
    assert not os.path.exists(scratch)


def test_the_filesystem_preset_scopes_its_allowlist_to_the_episode_unless_told_otherwise():
    env = build_mcp_environment("filesystem")
    assert env.scratch_allowlist and env.episode_scratch and "/tmp" not in env.server_args
    shared = build_mcp_environment("filesystem", episode_scratch=False)
    assert shared.scratch_allowlist and not shared.episode_scratch


def test_a_server_that_takes_no_directory_gets_none_and_refuses_episode_scratch():
    """``uvx mcp-server-fetch <dir>`` is an argparse error, so a directory goes only to an allowlist server."""
    env = build_mcp_environment("fetch")
    assert not env.scratch_allowlist and not env.episode_scratch
    with pytest.raises(ValueError, match=r"takes none .*\['filesystem', 'memory'\]"):
        build_mcp_environment("fetch", episode_scratch=True)


@pytest.mark.parametrize(
    "scratch", [{"episode_scratch": True}, {"scratch_allowlist": True}, {"scratch_env": {"MEMORY_FILE_PATH": "m"}}]
)
def test_a_scratch_directory_refuses_a_transport_that_spawns_nothing(scratch):
    with pytest.raises(ValueError, match="spawns none"):
        NativeMCPClientEnvironment(transport="sse", server_url="http://x/sse", **scratch)


class _MemoryServerSession(_StubSession):
    """``@modelcontextprotocol/server-memory`` as the stdio spawn would run it: entities created land in the file
    ``MEMORY_FILE_PATH`` names, else in the one file its package keeps (``default_file``), and ``read_graph``
    returns what that file holds."""

    def __init__(self, params, default_file: str):
        super().__init__()
        self.params = params
        self.graph_file = (params.env or {}).get("MEMORY_FILE_PATH", default_file)

    async def call_tool(self, name, arguments):
        if name == "create_entities":
            with open(self.graph_file, "a") as f:
                f.write(arguments["name"] + "\n")
            text = "created"
        elif os.path.exists(self.graph_file):
            with open(self.graph_file) as f:
                text = f.read()
        else:
            text = ""
        return types.SimpleNamespace(content=[types.SimpleNamespace(type="text", text=text)], isError=False)

    async def list_tools(self):
        tools = [
            types.SimpleNamespace(name="create_entities", description="add", inputSchema={"properties": {"name": {}}}),
            types.SimpleNamespace(name="read_graph", description="read", inputSchema={}),
        ]
        return types.SimpleNamespace(tools=tools)


async def test_the_memory_preset_gives_each_episode_its_own_graph(monkeypatch, tmp_path):
    """Two concurrent episodes (a GRPO group's siblings) must not read each other's memory: each episode's
    server keeps its graph in a file of its own scratch directory, removed with the episode, never in the one
    file the package keeps for every episode and run on the host."""
    default_file = tmp_path / "package-memory.jsonl"
    default_file.touch()
    spawned = []

    def _stdio_client(params):
        session = _MemoryServerSession(params, str(default_file))
        spawned.append(session)
        return _StubTransport(session, params.args)

    monkeypatch.setattr(mcp_module, "stdio_client", _stdio_client)
    env = build_mcp_environment("memory")
    assert env.episode_scratch
    ids, _ = await env.reset_async(["task a", "task b"])
    create = {"id": "c1", "function": {"name": "create_entities", "arguments": '{"name": "alpha"}'}}
    read = {"id": "c2", "function": {"name": "read_graph", "arguments": "{}"}}
    await env.step_async(
        ids, ["writing", "reading"], [{"finish_reason": "tool_calls", "tool_calls": [c]} for c in (create, read)]
    )

    _discovery, *episode_servers = spawned
    graph_files = [server.graph_file for server in episode_servers]
    assert len(set(graph_files)) == 2 and str(default_file) not in graph_files
    assert all(server.params.args == mcp_module.MCP_SERVERS["memory"]["args"] for server in spawned)
    reader = env.get_trajectories(ids)[1]
    assert next(m.content for m in reader.messages if m.role == "tool") == "", "episode b read episode a's graph"

    env.cleanup(ids)
    await asyncio.gather(*env._background_tasks)
    await env.disconnect()
    assert not any(os.path.exists(os.path.dirname(f)) for f in graph_files)
    assert not build_mcp_environment("memory", episode_scratch=False).episode_scratch


@pytest.mark.parametrize("scope", [{}, {"episode_scratch": False}], ids=["preset-scope", "environment-scope"])
async def test_a_memory_file_env_vars_names_is_the_graph_the_server_reads(monkeypatch, tmp_path, scope):
    """A seeded graph handed through ``env_vars`` is the file the server reads, served to every episode by one
    server; the preset's scratch file must not silently replace it."""
    seeded = tmp_path / "seeded-graph.jsonl"
    seeded.write_text("seed\n")
    spawned = []

    def _stdio_client(params):
        session = _MemoryServerSession(params, str(tmp_path / "package-memory.jsonl"))
        spawned.append(session)
        return _StubTransport(session, params.args)

    monkeypatch.setattr(mcp_module, "stdio_client", _stdio_client)
    env = build_mcp_environment("memory", env_vars={"MEMORY_FILE_PATH": str(seeded)}, **scope)
    assert not env.episode_scratch
    ids, _ = await env.reset_async(["task"])
    read = {"id": "c1", "function": {"name": "read_graph", "arguments": "{}"}}
    await env.step_async(ids, ["reading"], [{"finish_reason": "tool_calls", "tool_calls": [read]}])

    assert [server.graph_file for server in spawned] == [str(seeded)]
    reader = env.get_trajectories(ids)[0]
    assert next(m.content for m in reader.messages if m.role == "tool") == "seed\n"
    env.cleanup(ids)
    await asyncio.gather(*env._background_tasks)
    await env.disconnect()


def test_a_seeded_memory_file_refuses_a_server_per_episode():
    """Per-episode servers over one seeded file would interleave their writes to one graph."""
    with pytest.raises(ValueError, match=r"env_vars sets \['MEMORY_FILE_PATH'\].*episode_scratch=True"):
        build_mcp_environment("memory", env_vars={"MEMORY_FILE_PATH": "/data/graph.jsonl"}, episode_scratch=True)


def test_a_scratch_file_refuses_a_server_env_that_sets_it():
    """Both arguments name the variable, so either value would be a silent guess."""
    with pytest.raises(ValueError, match=r"server_env sets \['MEMORY_FILE_PATH'\]"):
        NativeMCPClientEnvironment(
            server_command="npx",
            server_env={"MEMORY_FILE_PATH": "/data/graph.jsonl"},
            scratch_env={"MEMORY_FILE_PATH": "memory.jsonl"},
        )


# Factory credential forwarding: MCP_SERVERS[...]["env"] must reach the server


def test_factory_forwards_declared_server_env(monkeypatch):
    """A key-requiring server (brave_search declares BRAVE_API_KEY) must get its credential from the
    process environment: dropping the declared ``env`` list launches the server without
    credentials."""
    monkeypatch.setenv("BRAVE_API_KEY", "sekrit")
    env = build_mcp_environment("brave_search")
    assert env.server_env == {"BRAVE_API_KEY": "sekrit"}


def test_factory_missing_required_env_fails_at_construction(monkeypatch):
    """A missing declared credential must fail loud at environment construction, not surface as an
    opaque tool failure on the first call."""
    monkeypatch.delenv("BRAVE_API_KEY", raising=False)
    with pytest.raises(ValueError, match="BRAVE_API_KEY"):
        build_mcp_environment("brave_search")


def test_factory_explicit_env_vars_satisfy_and_override(monkeypatch):
    """Explicit env_vars satisfy a declared requirement (no process-env lookup) and win over it."""
    monkeypatch.delenv("GITHUB_PERSONAL_ACCESS_TOKEN", raising=False)
    env = build_mcp_environment("github", env_vars={"GITHUB_PERSONAL_ACCESS_TOKEN": "tok"})
    assert env.server_env == {"GITHUB_PERSONAL_ACCESS_TOKEN": "tok"}


def test_factory_no_declared_env_passes_none():
    """Servers with no declared credentials keep server_env=None (SDK default environment)."""
    env = build_mcp_environment("filesystem")
    assert env.server_env is None


def test_sse_transport_is_registry_selectable_and_needs_a_url(monkeypatch):
    """A remote MCP server is reached by URL, so it names no stdio preset.

    ``mcp_server`` selects a command to spawn; ``transport="sse"`` spawns nothing. Resolving both
    would build an env whose preset (and its credential resolution) is silently dead, so the two are
    mutually exclusive and the registry picks the transport from the config.
    """
    mcp_module = importlib.import_module("halo.environments.protocols.mcp")
    monkeypatch.setattr(mcp_module, "sse_client", lambda url: ("sse-ctx", url))

    env = build_environment("mcp", {"transport": "sse", "server_url": "http://mcp.example:9000/sse"})
    assert env.transport == "sse"
    assert env._transport_context([]) == ("sse-ctx", "http://mcp.example:9000/sse")

    with pytest.raises(ValueError, match="server_url required"):
        NativeMCPClientEnvironment(transport="sse")._transport_context([])


def test_sse_config_refuses_a_stdio_preset(monkeypatch):
    """Anti-vacuity for the branch above: an sse config that also names a preset must fail loudly
    rather than build an env that ignores half its own configuration."""
    with pytest.raises(ValueError, match="spawns nothing"):
        build_environment("mcp", {"mcp_server": "filesystem", "transport": "sse", "server_url": "http://x/sse"})


def test_stdio_is_still_the_default_transport_and_preset():
    """The default path stays the keyless filesystem preset over stdio."""
    env = build_environment("mcp", {})
    assert (env.transport, env.server_command) == ("stdio", "npx")
    assert env.server_url is None


def test_stdio_transport_merges_credentials_over_sdk_default_env(monkeypatch):
    """The MCP SDK passes ``env`` verbatim when set — a credentials-only dict would drop PATH and
    break the server spawn. The stdio transport must merge our vars OVER the SDK defaults, and pass
    None (SDK default environment) when no vars are set."""
    captured = []

    class _Params:
        def __init__(self, command, args, env):
            self.command, self.args, self.env = command, args, env

    # The SDK names the module binds at import; patch them where they are USED.
    mcp_module = importlib.import_module("halo.environments.protocols.mcp")
    monkeypatch.setattr(mcp_module, "StdioServerParameters", _Params)
    monkeypatch.setattr(mcp_module, "get_default_environment", lambda: {"PATH": "/usr/bin", "HOME": "/root"})
    monkeypatch.setattr(mcp_module, "stdio_client", lambda params: captured.append(params) or "transport-ctx")
    monkeypatch.setattr(mcp_module, "sse_client", lambda url: "transport-ctx")

    env = NativeMCPClientEnvironment(server_command="npx", server_env={"KEY": "v", "PATH": "/custom"})
    assert env._transport_context(None) == "transport-ctx"
    assert captured[0].env == {"PATH": "/custom", "HOME": "/root", "KEY": "v"}

    env_plain = NativeMCPClientEnvironment(server_command="npx", server_args=["-y", "pkg"], scratch_allowlist=True)
    env_plain._transport_context("/scratch")
    assert captured[1].env is None
    assert captured[1].args == ["-y", "pkg", "/scratch"], "the allowlist directory must follow server_args"

    env_file = NativeMCPClientEnvironment(server_command="npx", scratch_env={"MEMORY_FILE_PATH": "memory.jsonl"})
    env_file._transport_context("/scratch")
    assert captured[2].args == [], "a scratch file goes through the environment, not the arguments"
    assert captured[2].env == {"PATH": "/usr/bin", "HOME": "/root", "MEMORY_FILE_PATH": "/scratch/memory.jsonl"}


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
