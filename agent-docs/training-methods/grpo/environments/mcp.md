# MCP Environment

`NativeMCPClientEnvironment` (`src/environments/envs/protocols/mcp.py`) connects to a [Model Context Protocol](https://modelcontextprotocol.io/) server and exposes its tools to the model as native function calls. Registry name `mcp`.

It extends `AsyncNativeToolUseEnvironment`, so the [native protocol's](native-tool-use.md) turn budget and reward apply unchanged but for one default: `tool_success_reward` is `0.1`, twice the native rate. The MCP SDK ships in the image; the server does not, so its launcher must be on the actor's `PATH`.

## Configuration

```yaml
environment_type: mcp

environment_kwargs:
  mcp_server: filesystem
```

| Knob | Default | Effect |
|---|---|---|
| `mcp_server` | `filesystem` | stdio preset to spawn; read by the factory, not the class |
| `transport` | `stdio` | `sse` connects to a running server instead |
| `server_url` | `None` | required under `sse` |
| `env_vars` | `{}` | satisfies or overrides a preset's credentials |

A preset's credentials are resolved at construction, so a missing key raises at launch, not as an opaque tool failure mid-episode. An `sse` config spawns nothing: it names no `mcp_server` and no `env_vars`, and passing either raises rather than silently ignoring half the configuration.

## Tools

Whatever the server advertises. Tools are discovered on the first episode from the server's JSON schema; a call that hangs returns as a tool error after 120 s, costing one turn rather than the whole episode deadline.

| `mcp_server` | Credential | Tools |
|---|---|---|
| `filesystem` | — | file operations, jailed to `/tmp` |
| `fetch` | — | HTTP fetch for web content (`uvx`) |
| `memory` | — | knowledge-graph memory |
| `brave_search` | `BRAVE_API_KEY` | Brave web search |
| `github` | `GITHUB_TOKEN` | GitHub API |
| `slack` | `SLACK_TOKEN` | Slack messaging |

## Evaluation

```bash
python scripts/environments/inference/run_env.py \
    --env_type mcp --env_kwargs '{"mcp_server": "fetch"}' \
    --dataset <hf-id-or-dir> --split test --prompt_field prompt --answer_field answer \
    --base_url http://localhost:8000/v1 --model <served-model> --num_examples 10
```

## From Python

```python
from src.environments.envs.protocols.mcp import NativeMCPClientEnvironment

env = NativeMCPClientEnvironment(
    server_command="npx",
    server_args=["-y", "@your-org/mcp-server-custom"],
    server_env={"CUSTOM_API_KEY": "..."},
)
```

## Related pages

- [Native Tool-Use](native-tool-use.md) — the protocol, its knobs and reward.
- [Environments](README.md) — registry and shared knobs.
