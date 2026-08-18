# Native Tool-Use Environments

`NativeToolUseEnvironment` uses the OpenAI/vLLM function-calling format. Works with any model whose chat template emits OpenAI-format tool calls (parsed server-side by the rollout engine's `--tool-call-parser`). Three registry presets wrap a tool registry in it:

| Registry | Factory | Tools |
|----------|---------|-------|
| `native_math` | `NativeToolRegistry.combine(create_native_math_tools(), create_native_python_tools())` | `calculate`, `python` |
| `native_coding` | `create_native_code_tools(language="python", tool_name="python_repl")` | `python_repl` |
| `native_combined` | `create_all_native_tools()` | math + python + search + file |

```python
from src.environments.tools.factories import create_all_native_tools
from src.environments.envs.protocols.native import NativeToolUseEnvironment

env = NativeToolUseEnvironment(tool_registry=create_all_native_tools(), max_turns=10)
```

## Reward knobs

Defaults: `max_turns` 10, `success_reward` 1.0, `failure_reward` 0.0, and at most `max_tool_calls_per_turn` 5 calls executed per turn (extras are dropped).

The terminal `success_reward` payout for a completed episode is reserved for contexts with **no** `answer` key at all — envs where completing is the objective. A row that carries an `answer` key holding null is a data fault: it scores `failure_reward` and is marked `episode_invalid`, so the trainer drops it from the group baseline instead of paying every completion full reward.

Per-call shaping: `tool_success_reward` (`0.05`) added per successful tool call, `tool_error_penalty` (`0.1`) subtracted per failed one. Per-episode shaping, all defaulting to `0`: `no_tool_use_penalty` (0-tool-call giveup), `multi_turn_reward` (more than one tool call, and the subclass's `_tool_use_engaged` gate passes), `turn_overflow_penalty` (turn count is otherwise free in the reward).

All five are **magnitudes** (≥ 0); the minus is applied at the use site and a negative config value raises at construction, so a sign flip in YAML can never turn a penalty into a farmable bonus.

`turn_overflow_penalty` is charged on any **truncated** episode, not only one that burned `max_turns` — `finalize_truncated` marks an episode killed mid-flight (a generation failure, an external abort) the same way, so it pays too.

**A turn cut off at its token cap** (`finish_reason == "length"`) **with no tool call is a failed turn, not an answer.** The episode appends a nudge naming what happened and retries on its remaining `max_turns` budget; the count surfaces as `episode/length_cutoff_turns`. A turn that emitted its tool call before the cap takes the normal tool path.

The cutoff is **unpriced** — a penalty would be avoidable only by reasoning well short of the budget. Bound the frequency structurally instead, through the per-effort caps and the answer headroom (`rollout_max_tokens - rollout_max_thinking_tokens`).

The cut-off turn is flagged on its `Message` (`truncated`) and the trainer skips it when building per-turn training rows: the model still conditions on the fragment in the next turn's prompt, but an unfinished turn is never reinforced by an episode that goes on to succeed.

**A turn whose every tool call named a tool that does not exist** is flagged the same way (`Message.calls_rejected`) and likewise skipped at tokenization. The rejection travels on the tool result's structured `unknown_tool` field, never on error-text matching: a real tool whose backend answers with its own "Tool not found …" message is a tool failure, not a model-invented call, and keeps its turn in training.

The unknown-tool observation names the real tools (`Error: Unknown tool 'X'. Available tools: ...`) — a policy that drifts off the tool syntax late in training invents plausible names and then burns turns probing for a listing that no tool provides.

`require_tool_use` only flags a zero-tool-call finish in the step info; it gates no reward. `no_tool_use_penalty` is charged once by `_tool_use_shaping` for any episode ending with zero tool calls, flag set or not.

## Tool factories

Factories in `src/environments/tools/factories.py` build tool registries. Stateless factories (`create_native_*`) make each call independent; session-backed factories (`create_session_*`) bind to a live [`SandboxSession`](sandbox.md#sessions-persistent-multi-turn-state) whose working directory persists across turns.

| Factory | Tools | Description |
|---------|-------|-------------|
| `create_native_math_tools()` | `calculate` | Restricted-builtins math evaluation |
| `create_native_python_tools()` | `python` | Sandboxed Python REPL |
| `create_native_code_tools(language=..., tool_name=..., sandbox=...)` | one code tool | Single-language code execution (python / cpp / c) |
| `create_native_search_tools()` | `web_search` | Pluggable web search (Serper / Brave / Tavily / DuckDuckGo) |
| `create_native_file_tools()` | `list_files`, `read_file`, `write_file` | Simulated in-memory filesystem (tests / closed-world demos) |
| `create_all_native_tools()` | math + python + search + file | All stateless tools |
| `create_session_code_tools(session_getter, language=..., tool_name=...)` | `run_<language>` | Code execution in the episode's persistent workspace |
| `create_session_file_tools(session_getter)` | `write_file`, `read_file`, `list_files` | Real persistent file ops over the session working dir |

Combine registries with `NativeToolRegistry.combine(a, b)`, or `registry.merge(other)` in place.

By default the `python` tool runs in the in-process restricted sandbox (`run_python_sandboxed`: restricted builtins, no imports). Pass `sandbox=` (a `SandboxExecutor` from `resolve_sandbox()`) to `create_native_python_tools` / `create_native_code_tools` to run the REPL in a real OS-isolated interpreter with imports available; `allow_imports=True` instead lifts the import ban inside the in-process REPL and is only safe when the whole process is already isolated. A non-Python `language` always runs through a `SandboxExecutor`. See [Code Execution Sandboxes](sandbox.md).

## Tool observations

Each tool observation is capped at `max_observation_chars` (default 16384, set in `environment_kwargs`) where the result is built (`BaseEnvironment._truncate_observation`, inherited by every tool-use env). An unbounded output — a `python_repl` that prints megabytes — otherwise bloats the trajectory and makes the per-turn chat-template re-render take minutes; capping at the source keeps the rollout and the trainer's recompute identical.

## Tool-call parsing and async

OpenAI tool-call parsing and serialization live on the data model in `src/environments/tools/definitions.py`; `registry.to_openai_tools()` supplies the schemas vLLM needs at generation time. `NativeTool.execute` / `execute_async` bind the model's argument dict against the tool's declared `parameters` and drop anything else: the handlers carry pre-bound configuration (sandbox `timeout`, `allow_imports`), and a call-time keyword of the same name would otherwise override it — so a tool's own limits stay out of the model's reach. See [Custom Environments](custom-environments.md#registering-custom-tools) for the method-level contract.

For async I/O use `AsyncNativeToolUseEnvironment` (`reset_async` / `step_async`, tool calls within a turn executed concurrently); `NativeMCPClientEnvironment` extends it, see [MCP](mcp.md).

The Ray rollout path drives sync and async environments alike. `RolloutManager` dispatches episodes to its `EnvironmentActor` pool, and each actor picks `step` / `step_async` from an `isinstance(env, AsyncBaseEnvironment)` check on its own instance, offloading a sync env to a thread so blocking tool work cannot stall the actor's loop.
