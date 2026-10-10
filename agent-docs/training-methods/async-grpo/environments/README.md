# Environments

An environment is the multi-turn task an [Async GRPO with Environments](../README.md) run trains against. The protocol (`src/environments/base.py`):

- `reset(prompts, contexts)` opens one episode per prompt.
- `step(episode_ids, actions, contexts)` executes a model turn and returns the next observation.
- `get_trajectories(episode_ids)` hands back the messages and the final reward.

An episode ends when the environment reports it done, or at `max_turns`.

## Registry

`environment_type` resolves through `src/environments/registry.py`. `get_registered_environments()` lists the live set; `register_environment` extends it ([Custom Environments](custom.md)).

| `environment_type` | Class | Tools |
|---|---|---|
| `react_math` | `ReActEnvironment` | `calculate`, `python` |
| `react_search` | `ReActEnvironment` | `web_search` |
| `native_math` | `NativeToolUseEnvironment` | `calculate`, `python` |
| `native_coding` | `NativeToolUseEnvironment` | `python_repl` |
| `native_combined` | `NativeToolUseEnvironment` | `calculate`, `python`, `web_search`, file ops |
| `qa_search` | `NativeToolUseEnvironment` (factory) | `web_search` (+ `python` under `include_python_tools`) |
| `exam_qa` | `ExamQAEnvironment` | none; `web_search` under `open_book` |
| `swe` | `SweEnvironment` | `run_code`, `run_bash_command`, workspace file ops |
| `code_contests` | `CodeContestsEnvironment` | scratchpad test tool, `submit_solution` |
| `codeforces` | `CodeContestsEnvironment` | same, graded by token comparison |
| `mcp` | `NativeMCPClientEnvironment` | whatever the server advertises |

The ReAct presets parse the action out of the text. Every other environment uses native tool calls, so the server needs the family's `--tool-call-parser` ([Tool calls](../episodes.md#tool-calls)).

## Actor runtime

Actors run environments on CPU and inherit the `ray start` daemon's environment, so export what they need before `ray start` on each node ([Ray Cluster](../../../infrastructure/ray.md#multi-node)).

| Environment | Actor nodes need |
|---|---|
| `code_contests`, `codeforces`, `swe` | A sandbox backend and a large `TMPDIR` ([Code Execution Sandboxes](sandbox.md)) |
| `react_math`, `native_math`, `native_coding`, closed-book `exam_qa` | Nothing beyond the image (in-process tools) |
| `react_search`, `qa_search`, `native_combined`, open-book `exam_qa` | Outbound network. `SERPER_API_KEY` / `BRAVE_API_KEY` / `TAVILY_API_KEY` picks a keyed search backend, else keyless DuckDuckGo. `search_backend` names one instead ([QA Benchmarks](tasks/qa-benchmarks.md#configuration)) |
| `mcp` | The preset's launcher on `PATH` (`npx`; `uvx` for `fetch`) and its credential. An SSE server needs only network to `server_url` |

## Configuration

`EnvironmentConfig` (`src/configs/environment_config.py`) has four top-level fields. Everything else goes in `environment_kwargs`.

```yaml
environment_type: react_math
rewards:
  - source: environment    # the environment's grade in [0, 1], priced weight × grade ^ exponent
max_turns: null            # null keeps the class default

environment_kwargs:
  carry_reasoning: false
  tool_error_penalty: 0.1
```

- Any other constructor parameter of the resolved class is settable from `environment_kwargs`. A key no constructor binds (a typo, or another `environment_type`'s option) raises `TypeError` at construction.
- The ReAct presets refuse a set `system_prompt`, `mcp` reads `mcp_server` in its factory, and `native_combined` reads `search_backend` in its own.
- `rewards` reaches the constructor as `reward_terms` ([Reward Terms](../../grpo/rewards/README.md#environment-arm)).

## Common knobs

Every environment accepts these. `max_turns` and `rewards` are top-level fields; the rest go in `environment_kwargs`.

| Knob | Default | Effect |
|---|---|---|
| `max_turns` | class default: 10; `exam_qa` 8, code contests 15, `swe` 20 | Turns before the episode truncates; ≥ 1 |
| `rewards` | `[{source: environment}]` | The reward terms ([Reward Terms](../../grpo/rewards/README.md)) |
| `tool_success_reward` / `tool_error_penalty` | `0.05` / `0.1` (`mcp` pays `0.1`; code contests `0` / `0`) | Paid per successful tool call, charged per failed one, logged as `reward/turn_shaping`. A call a sandbox outage ended is unpriced ([Sandbox faults](sandbox.md#sandbox-faults)) |
| `tool_reward_cap` | `tool_success_reward × max_turns` | Episode cap on successful-call pay, so call spam cannot out-earn the objective |
| `max_observation_chars` | `16384` | Longer tool observations are truncated |
| `reasoning_effort` | `None` (code contests `medium`) | `low` / `medium` / `high` / `random` CoT steer ([Effort levels](../reasoning.md#effort-levels)) |
| `carry_reasoning` | `false` | Sends the last assistant turn's reasoning back to the engine; vLLM only ([Carried reasoning](../episodes.md#carried-reasoning)) |
| `max_length_cutoff_recoveries` | `null` | Unproductive turns (engine-cut, or no visible content and no tool call) one episode may retry; `null` = all within `max_turns` ([Unproductive turns](../episodes.md#unproductive-turns)) |
| `requires_answer` | `false`; `true` for code contests, `exam_qa`, `qa_search`, the ReAct presets and [`swe`](tasks/swe.md#reward) (unless judge-only) | The reward grades against the dataset's `answer` column, so a dataset without one is refused at trainer construction |

Reward and penalty knobs are magnitudes (≥ 0); a negative value raises.

## Pages

**Tool protocols**

- [ReAct](protocols/react.md) — Thought / Action / Observation in plain text.
- [Native Tool-Use](protocols/native-tool-use.md) — function calling and tool factories; most environments extend it.
- [MCP](protocols/mcp.md) — tools from an MCP server.

**Tasks**

- [Code Contests](tasks/code-contests/README.md) — competitive programming, hidden-test grading.
- [SWE](tasks/swe.md) — persistent-workspace agent.
- [QA Benchmarks](tasks/qa-benchmarks.md) — `qa_search` and `exam_qa`.

**Infrastructure and authoring**

- [Code Execution Sandboxes](sandbox.md) — backends, faults, limits.
- [Custom Environments](custom.md) — write, register and test your own.
- [Evaluation](../evaluation.md) — run a model through an environment.
