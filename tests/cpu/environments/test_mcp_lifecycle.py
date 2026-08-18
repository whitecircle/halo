#!/usr/bin/env python
"""CPU tests for the NativeMCPClientEnvironment connection lifecycle (no mcp package needed).

The MCP SDK's transport/session contexts hold anyio cancel scopes, which must be entered and exited
by the SAME task. These tests stub the transport and session to pin the lifecycle contract:

- one connection-owner task enters AND exits both contexts (task affinity);
- a failure mid-connect (``initialize`` raising) unwinds the already-entered transport, so a partial
  connect never leaks a half-initialized session or the server subprocess;
- the sync ``close()`` hook (what ``EnvironmentActor.shutdown`` calls) actually drives disconnect —
  a no-op there leaks the ClientSession and the stdio subprocess on every rollout shutdown.

Run: python tests/cpu/environments/test_mcp_lifecycle.py  (or pytest)
"""

import asyncio
import importlib
import sys
import types
from datetime import timedelta

import pytest

import src.environments.envs.protocols.mcp as mcp_module
from src.configs.rollout_config import DEFAULT_EPISODE_TIMEOUT_SECONDS
from src.environments.envs.protocols.mcp import NativeMCPClientEnvironment, create_native_mcp_environment
from src.environments.registry import resolve_environment


class _StubTransport:
    """Async CM standing in for stdio_client/sse_client; records enter/exit and their owning tasks."""

    def __init__(self):
        self.entered = False
        self.exited = False
        self.enter_task = None
        self.exit_task = None

    async def __aenter__(self):
        self.entered = True
        self.enter_task = asyncio.current_task()
        return ("read", "write")

    async def __aexit__(self, *exc):
        self.exited = True
        self.exit_task = asyncio.current_task()
        return False


class _StubSession:
    """Async CM standing in for mcp.ClientSession."""

    def __init__(self, fail_initialize: bool = False):
        self.fail_initialize = fail_initialize
        self.entered = False
        self.exited = False

    async def __aenter__(self):
        self.entered = True
        return self

    async def __aexit__(self, *exc):
        self.exited = True
        return False

    async def initialize(self):
        if self.fail_initialize:
            raise RuntimeError("initialize failed")

    async def list_tools(self):
        tool = types.SimpleNamespace(name="stub_tool", description="a stub", inputSchema={})
        return types.SimpleNamespace(tools=[tool])


class _StubMCPEnv(NativeMCPClientEnvironment):
    """MCP env with the SDK-touching seams stubbed out."""

    def __init__(self, fail_initialize: bool = False, **kwargs):
        super().__init__(server_command="unused", **kwargs)
        self.stub_transport = _StubTransport()
        self.stub_session = _StubSession(fail_initialize=fail_initialize)

    def _transport_context(self):
        return self.stub_transport

    def _create_session(self, read, write):
        assert (read, write) == ("read", "write")
        return self.stub_session


def test_the_session_bounds_every_request_it_makes(monkeypatch):
    """A ClientSession built without ``read_timeout_seconds`` runs ``anyio.fail_after(None)``, so a
    server that ACKs and never answers parks the episode for the whole ``episode_timeout`` and takes
    its rollout worker with it — over stdio there is no socket timeout underneath either. The
    deadline must also sit BELOW the episode deadline, or it can never fire first."""
    seen: dict = {}

    def _record(read, write, **kwargs):
        seen.update(read=read, write=write, **kwargs)
        return "session"

    monkeypatch.setattr(mcp_module, "ClientSession", _record)
    env = NativeMCPClientEnvironment(server_command="unused")

    assert env._create_session("r", "w") == "session"
    assert seen["read_timeout_seconds"] == timedelta(seconds=mcp_module.MCP_REQUEST_TIMEOUT_S)
    assert 0 < mcp_module.MCP_REQUEST_TIMEOUT_S < DEFAULT_EPISODE_TIMEOUT_SECONDS


async def test_connect_discovers_tools_and_disconnect_unwinds_in_owner_task():
    env = _StubMCPEnv()
    await env.connect()
    assert env._connected
    assert env.registry.get("stub_tool") is not None
    assert env.stub_transport.entered and env.stub_session.entered

    await env.disconnect()
    assert env.stub_session.exited and env.stub_transport.exited
    assert env._session is None and not env._connected
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
    assert env._session is None and not env._connected


async def test_sync_close_from_running_loop_drives_disconnect():
    """close() called on the owning loop (EnvironmentActor.shutdown's env.close()) cannot block that
    loop, so it schedules disconnect; the scheduled task must actually unwind the connection."""
    env = _StubMCPEnv()
    await env.connect()

    env.close()
    assert env._close_task is not None
    await env._close_task
    assert env.stub_session.exited and env.stub_transport.exited
    assert env._session is None and not env._connected


async def test_disconnect_is_idempotent():
    env = _StubMCPEnv()
    await env.connect()
    await env.disconnect()
    await env.disconnect()
    assert not env._connected


# Factory credential forwarding: MCP_SERVERS[...]["env"] must reach the server


def test_factory_forwards_declared_server_env(monkeypatch):
    """A key-requiring server (brave_search declares BRAVE_API_KEY) must get its credential from the
    process environment: dropping the declared ``env`` list launches the server without
    credentials."""
    monkeypatch.setenv("BRAVE_API_KEY", "sekrit")
    env = create_native_mcp_environment("brave_search")
    assert env.server_env == {"BRAVE_API_KEY": "sekrit"}


def test_factory_missing_required_env_fails_at_construction(monkeypatch):
    """A missing declared credential must fail loud at environment construction, not surface as an
    opaque tool failure on the first call."""
    monkeypatch.delenv("BRAVE_API_KEY", raising=False)
    with pytest.raises(ValueError, match="BRAVE_API_KEY"):
        create_native_mcp_environment("brave_search")


def test_factory_explicit_env_vars_satisfy_and_override(monkeypatch):
    """Explicit env_vars satisfy a declared requirement (no process-env lookup) and win over it."""
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    env = create_native_mcp_environment("github", env_vars={"GITHUB_TOKEN": "tok"})
    assert env.server_env == {"GITHUB_TOKEN": "tok"}


def test_factory_no_declared_env_passes_none():
    """Servers with no declared credentials keep server_env=None (SDK default environment)."""
    env = create_native_mcp_environment("filesystem")
    assert env.server_env is None


def test_sse_transport_is_registry_selectable_and_needs_a_url(monkeypatch):
    """A remote MCP server is reached by URL, so it names no stdio preset.

    ``mcp_server`` selects a command to spawn; ``transport="sse"`` spawns nothing. Resolving both
    would build an env whose preset (and its credential resolution) is silently dead, so the two are
    mutually exclusive and the registry picks the transport from the config.
    """
    mcp_module = importlib.import_module("src.environments.envs.protocols.mcp")
    monkeypatch.setattr(mcp_module, "sse_client", lambda url: ("sse-ctx", url))

    env = resolve_environment("mcp", {"transport": "sse", "server_url": "http://mcp.internal:9000/sse"})
    assert env.transport == "sse"
    assert env._transport_context() == ("sse-ctx", "http://mcp.internal:9000/sse")

    with pytest.raises(ValueError, match="server_url required"):
        NativeMCPClientEnvironment(transport="sse")._transport_context()


def test_sse_config_refuses_a_stdio_preset(monkeypatch):
    """Anti-vacuity for the branch above: an sse config that also names a preset must fail loudly
    rather than build an env that ignores half its own configuration."""
    with pytest.raises(ValueError, match="spawns nothing"):
        resolve_environment("mcp", {"mcp_server": "filesystem", "transport": "sse", "server_url": "http://x/sse"})


def test_stdio_is_still_the_default_transport_and_preset():
    """The default path stays the keyless filesystem preset over stdio."""
    env = resolve_environment("mcp", {})
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
    mcp_module = importlib.import_module("src.environments.envs.protocols.mcp")
    monkeypatch.setattr(mcp_module, "StdioServerParameters", _Params)
    monkeypatch.setattr(mcp_module, "get_default_environment", lambda: {"PATH": "/usr/bin", "HOME": "/root"})
    monkeypatch.setattr(mcp_module, "stdio_client", lambda params: captured.append(params) or "transport-ctx")
    monkeypatch.setattr(mcp_module, "sse_client", lambda url: "transport-ctx")

    env = NativeMCPClientEnvironment(server_command="npx", server_env={"KEY": "v", "PATH": "/custom"})
    assert env._transport_context() == "transport-ctx"
    assert captured[0].env == {"PATH": "/custom", "HOME": "/root", "KEY": "v"}

    env_plain = NativeMCPClientEnvironment(server_command="npx")
    env_plain._transport_context()
    assert captured[1].env is None


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
