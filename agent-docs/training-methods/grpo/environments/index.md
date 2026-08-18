# Environments

Environments provide the multi-turn interaction loop for [Environmental GRPO](../environmental-grpo.md): the model generates an action, the environment executes it, returns an observation, and repeats until done. Shared interface: `reset(prompts, contexts)`, `step(episode_ids, actions, contexts)`, `get_trajectories(episode_ids)`.

**Location:** `src/environments/`. `envs/protocols/` holds tool-invocation protocols (native function calling, ReAct, and the MCP transport over native calling); `envs/tasks/` holds the concrete tasks (`coding/` with its grading and dataset adapters, `qa`). Tools in `tools/`, code-execution sandboxes in `sandbox/`.

## Class hierarchy

| Class | Base | Description |
|-------|------|-------------|
| `BaseEnvironment` | - | Abstract base for all environments |
| `AsyncBaseEnvironment` | `BaseEnvironment` | Async support for I/O |
| `ReActEnvironment` | `BaseEnvironment` | Thought/Action/Observation format |
| `NativeToolUseEnvironment` | `BaseEnvironment` | OpenAI/vLLM function calling |
| `AsyncNativeToolUseEnvironment` | `AsyncBaseEnvironment`, `NativeToolUseEnvironment` | Async native tool execution |
| `NativeMCPClientEnvironment` | `AsyncNativeToolUseEnvironment` | MCP server integration |
| `SweEnvironment` | `NativeToolUseEnvironment` | Stateful SWE-agent over a persistent session |
| `CodeContestsEnvironment` | `NativeToolUseEnvironment` | Competitive programming, hidden-test grading |
| `ExamQAEnvironment` | `NativeToolUseEnvironment` | Multiple-choice and open-ended exams |

Factual QA (`qa_search`) is a factory preset over `NativeToolUseEnvironment` (`create_qa_search_environment`), not a dedicated class.

## Registry names

Resolve built-in types by name via `src/environments/registry.py`:

| Registry name | Resolves to | Tools |
|---------------|-------------|-------|
| `react_math` | `ReActEnvironment` | `calculate`, `python` |
| `react_search` | `ReActEnvironment` | `web_search` |
| `native_math` | `NativeToolUseEnvironment` | `calculate`, `python` |
| `native_coding` | `NativeToolUseEnvironment` | `python_repl` |
| `native_combined` | `NativeToolUseEnvironment` | math + python + search + file |
| `swe` | `SweEnvironment` | `run_code` + persistent file ops (accepts `language`) |
| `mcp` | `NativeMCPClientEnvironment` | MCP server tools |
| `qa_search` | `NativeToolUseEnvironment` (factory) | `web_search` (+ optional `python`) |
| `code_contests` | `CodeContestsEnvironment` | test REPL + `submit_solution`, exact-match grading (accepts `language`) |
| `codeforces` | `CodeContestsEnvironment` | same env, token comparison — the only difference between the two presets |
| `exam_qa` | `ExamQAEnvironment` | none, or `web_search` in open-book mode |

**Runtime per env** — must exist on every Ray actor node
([Ray Cluster](../../../infrastructure/ray.md#multi-node)): a sandbox plus a large `TMPDIR` for
`code_contests` / `codeforces` / `swe`; outbound network plus the backend's key for `react_search` /
`qa_search` / open-book `exam_qa`; `npx` or `uvx` for `mcp`.

`native_coding` needs neither — its `python_repl` is the in-process restricted REPL (no imports).
Swapping in a `SandboxExecutor` there is a Python-only path through the tool factory: `native_coding`'s
chain takes no `sandbox` parameter, so setting it in its `environment_kwargs` raises `TypeError`. The
sandboxed `swe` / `code_contests` / `codeforces` envs do declare `sandbox=`, but only Python callers
can pass an executor object — YAML cannot.

## Passing environments to the trainer

`EnvironmentConfig` (`src/configs/environment_config.py`) has four top-level fields — `environment_type` (default `react_math`), `success_reward` (`1.0`), `failure_reward` (`0.0`), `max_turns` (`None`) — plus the nested `environment_kwargs` block that carries every per-env option. There is no `partial_reward`.

`max_turns: None` keeps the environment class's own default (`code_contests` / `codeforces` 15, `swe` 20, `exam_qa` 8, everything else 10); setting it overrides all of them, and it must be `>= 1`.

```yaml
environment_type: react_math
success_reward: 1.0
max_turns: 10

environment_kwargs:
  search_backend: duckduckgo   # qa_search / open-book exam_qa (react_search auto-selects its backend)
  open_book: true              # exam_qa
  timeout_per_test: 10         # code_contests / codeforces
  mcp_server: filesystem       # mcp
```

Registry factories forward the whole config, so any constructor kwarg of the target class can be set from `environment_kwargs`. Two keys the factories consume themselves are the exception: `react_math` / `react_search` drop `system_prompt` (each preset hardcodes its own), so setting it there is silently inert, and the `mcp` factory consumes `mcp_server` to **select the server preset** rather than discarding it.

A key no constructor in the chain binds — a typo, or an option belonging to a different `environment_type` — raises `TypeError` at environment construction rather than being absorbed and ignored.

For a custom environment, register a factory (`env_config` dict → `BaseEnvironment`) with `register_environment` and set `environment_type`, or pass `environment_cls` + `environment_kwargs` to the trainer directly. See [Custom Environments](custom-environments.md).

## Related pages

- [Environmental GRPO Trainer](../environmental-grpo.md) — training architecture and setup
- [ReAct](react.md) · [Native Tool-Use](native-tool-use.md) · [SWE](swe-environment.md) · [Code Contests](code-contests.md) · [MCP](mcp.md) · [Benchmarks](benchmarks.md)
- [Code Execution Sandboxes](sandbox.md) — backends, languages, limits, sessions
- [Custom Environments](custom-environments.md) — custom envs, tools, dataset format
