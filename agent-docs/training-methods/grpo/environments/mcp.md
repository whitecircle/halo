# MCP Environment

`NativeMCPClientEnvironment` (`src/environments/envs/protocols/mcp.py`) connects to [Model Context Protocol](https://modelcontextprotocol.io/) servers and exposes their discovered tools to the model as native function calls. It extends `AsyncNativeToolUseEnvironment`, so turn limits and reward shaping match the [native tool-use environments](native-tool-use.md) with one exception: `tool_success_reward` defaults to `0.1` here, twice the native `0.05`. Every MCP request — `initialize`, tool discovery, a tool call — is bounded by `MCP_REQUEST_TIMEOUT_S` (120 s); without it a server that acknowledges and never answers parks the episode for the whole `episode_timeout` and holds its rollout worker.

Registry name `mcp`; in YAML, `mcp_server` selects a preset (default `filesystem`) to spawn over stdio. The MCP Python SDK (`mcp>=1.22.0,<2.0.0`) is a core dependency. `create_native_mcp_environment` resolves a preset's declared credential variables before it constructs the environment, so a missing one raises there rather than mid-episode; constructing `NativeMCPClientEnvironment` directly performs no credential check.

An already-running server is reached over SSE instead: `environment_kwargs: {transport: sse, server_url: "http://host:port/sse"}`. Nothing is spawned, so an SSE config names no `mcp_server` and no `env_vars` — passing either raises, rather than building an environment that ignores half its own configuration.

| Server | Required env vars | Tools |
|--------|-------------------|-------|
| `brave_search` | `BRAVE_API_KEY` | Brave web search |
| `filesystem` | — | File operations (default path `/tmp`) |
| `fetch` | — | HTTP fetch for web content |
| `memory` | — | Knowledge graph memory |
| `github` | `GITHUB_TOKEN` | GitHub API |
| `slack` | `SLACK_TOKEN` | Slack messaging |

```python
from src.environments.envs.protocols.mcp import create_native_mcp_environment

env = create_native_mcp_environment(
    server_name="brave_search",
    env_vars={"BRAVE_API_KEY": "your-api-key"},
    max_turns=10,
)
```

## Custom server

Any MCP-compatible server, via a command and args:

```python
from src.environments.envs.protocols.mcp import NativeMCPClientEnvironment

env = NativeMCPClientEnvironment(
    server_command="npx",
    server_args=["-y", "@your-org/mcp-server-custom"],
    server_env={"CUSTOM_API_KEY": "your-key"},
    max_turns=10,
)
```

The default `transport="stdio"` requires `server_command`; for an HTTP/SSE server set `transport="sse"` and pass `server_url` instead. Either way, both the transport and the session are entered and exited on one task — the MCP SDK's anyio cancel scopes require it.
