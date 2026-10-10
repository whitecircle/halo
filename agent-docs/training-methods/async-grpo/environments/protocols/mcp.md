# MCP Environment

`NativeMCPClientEnvironment` (`src/environments/envs/protocols/mcp.py`, registry name `mcp`) connects to a [Model Context Protocol](https://modelcontextprotocol.io/) server and exposes its tools as native function calls.

It extends `AsyncNativeToolUseEnvironment`, so the [native protocol's](native-tool-use.md) turn budget and reward apply, with `tool_success_reward` raised to `0.1`. The MCP SDK ships in the image; the server does not, so its launcher must be on the actor's `PATH`.

## Configuration

```yaml
environment_type: mcp

environment_kwargs:
  mcp_server: filesystem
```

| Knob | Default | Effect |
|---|---|---|
| `mcp_server` | `filesystem` | Stdio preset to spawn; read by the factory, not the class |
| `transport` | `stdio` | `sse` connects to a running server instead |
| `server_url` | `None` | Required under `sse` |
| `env_vars` | `{}` | Satisfies or overrides a preset's credentials |

- A preset's credentials are resolved at construction, so a missing key raises at launch.
- An `sse` config spawns nothing; passing `mcp_server` or `env_vars` with it raises.

## Tools

Whatever the server advertises, discovered on the first episode. A call that hangs returns a tool error after 120 s.

| `mcp_server` | Credential | Tools |
|---|---|---|
| `filesystem` | — | File operations, jailed to `/tmp` |
| `fetch` | — | HTTP fetch for web content (`uvx`) |
| `memory` | — | Knowledge-graph memory |
| `brave_search` | `BRAVE_API_KEY` | Brave web search |
| `github` | `GITHUB_PERSONAL_ACCESS_TOKEN` | GitHub API |
| `slack` | `SLACK_BOT_TOKEN`, `SLACK_TEAM_ID` | Slack messaging |

## Evaluation

```bash
python scripts/environments/inference/run_env.py \
    --env_type mcp --env_kwargs '{"mcp_server": "fetch"}' \
    --dataset <hf-id-or-dir> --split test --prompt_field prompt --answer_field answer \
    --base_url http://localhost:8000/v1 --model <served-model> --num_examples 10
```

Flags and the `--training_config` contract: [Evaluation](../../evaluation.md).

## From Python

A stdio server outside the presets is launched from the constructor:

```python
from src.environments.envs.protocols.mcp import NativeMCPClientEnvironment

env = NativeMCPClientEnvironment(
    server_command="npx",
    server_args=["-y", "<npm-package>"],
    server_env={"API_KEY": "..."},
)
```

## Related pages

- [Native Tool-Use](native-tool-use.md) — the protocol, its knobs and reward.
- [Environments](../README.md) — registry and common knobs.
