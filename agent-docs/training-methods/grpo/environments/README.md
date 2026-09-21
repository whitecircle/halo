# Environments

An environment is the multi-turn task an [Async GRPO with Environments](../async-grpo/README.md) run trains against. `reset(prompts, contexts)` opens one episode per prompt; `step(episode_ids, actions, contexts)` executes a model turn and returns the next observation; the episode ends when the environment reports it done or at `max_turns`. `get_trajectories(episode_ids)` hands back the messages and final reward. Protocol: `src/environments/base.py`.

## Registry

`environment_type` resolves through `src/environments/registry.py`; `get_registered_environments()` lists the live set, which `register_environment` extends.

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

The ReAct presets parse the action out of the assistant text; the rest use native tool calls, so the server needs the family's `--tool-call-parser` ([Rollout Configuration](../async-grpo/rollouts.md#tool-calls)).

## Actor runtime

Actors run environments on CPU and inherit the `ray start` daemon's environment — export what they need before `ray start` on each node ([Ray Cluster](../../../infrastructure/ray.md#multi-node)).

| Environment | Actor nodes need |
|---|---|
| `code_contests`, `codeforces`, `swe` | A sandbox backend and a large `TMPDIR`: the local backend runs each program in a temp dir |
| `react_math`, `native_math`, `native_coding`, closed-book `exam_qa` | Nothing beyond the image: `calculate` and the `python` REPL run in-process with imports blocked |
| `react_search`, `qa_search`, `native_combined`, open-book `exam_qa` | Outbound network. `SERPER_API_KEY` / `BRAVE_API_KEY` / `TAVILY_API_KEY` picks a keyed search backend; with none, keyless DuckDuckGo |
| `mcp` | The preset's launcher on `PATH` (`npx`; `uvx` for `fetch`) and its credential. An SSE server needs only network to `server_url` |

The code-executing environments take `sandbox_backend` / `sandbox_url` in `environment_kwargs`, else `HALO_SANDBOX_BACKEND` / `HALO_SANDBOX_URL` — [Code Execution Sandboxes](sandbox.md).

## Configuration

`EnvironmentConfig` (`src/configs/environment_config.py`) has four top-level fields; everything else goes in `environment_kwargs`.

```yaml
environment_type: react_math
rewards:
  - source: environment    # the environment's grade in [0, 1], priced weight × grade ^ exponent
max_turns: null            # null keeps the class default

environment_kwargs:
  carry_reasoning: false
  tool_error_penalty: 0.1
```

The factory forwards the merged dict whole, so any constructor parameter of the resolved class is settable from `environment_kwargs`. A key no constructor binds — a typo, or an option of another `environment_type` — raises `TypeError` at construction. Two keys the factories consume themselves: the ReAct presets drop `system_prompt`, and `mcp` reads `mcp_server`. `rewards` reaches the constructor as `reward_terms`.

The episode reward is the environment's grade priced by the `rewards:` terms, plus the environment's own shaping ([Reward Terms](../rewards.md#environment-arm)).

Knobs every environment accepts (the first two are top-level fields):

| Knob | Default | Effect |
|---|---|---|
| `max_turns` | class default: 10; `exam_qa` 8, code contests 15, `swe` 20 | turns before the episode truncates; ≥ 1 |
| `rewards` | `[{source: environment}]` | the reward terms: the environment's grade at `weight` / `exponent`, plus external `judge` / `reward_model` terms ([Reward Terms](../rewards.md)) |
| `tool_success_reward` / `tool_error_penalty` | `0.05` / `0.1` (`mcp` pays `0.1`; code contests `0` / `0`) | paid per successful tool call, charged per failed one |
| `tool_reward_cap` | `tool_success_reward × max_turns` | episode total payable for successful calls |
| `max_observation_chars` | `16384` | longer tool observations are truncated at the source |
| `reasoning_effort` | `None` (code contests `medium`) | `low` / `medium` / `high` / `random` CoT steer |
| `carry_reasoning` | `false` | sends the last assistant turn's reasoning back to the engine; vLLM only |
| `max_length_cutoff_recoveries` | `null` | unproductive turns — engine-cut, or ended with no visible content and no tool call — one episode may retry; `null` = every one within `max_turns` |
| `requires_answer` | class default: `false`; `true` for code contests, `exam_qa`, `qa_search` and the ReAct presets | the reward grades against the dataset's `answer` column, so a dataset without one is refused at trainer construction |

## Pages

- [ReAct](react.md) — Thought / Action / Observation.
- [Native Tool-Use](native-tool-use.md) — function calling, tool factories.
- [SWE](swe-environment.md) — persistent workspace agent.
- [Code Contests](code-contests.md) — hidden-test grading.
- [MCP](mcp.md) — tools from an MCP server.
- [Benchmarks](benchmarks.md) — QA and exams.
- [Code Execution Sandboxes](sandbox.md) — backends and limits.
- [Evaluating on an Environment](evaluation.md) — run a model through one.
- [Custom Environments](custom-environments.md) — write, register, test your own.
