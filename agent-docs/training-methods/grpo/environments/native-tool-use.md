# Native Tool-Use Environments

`NativeToolUseEnvironment` runs the OpenAI/vLLM function-calling protocol: the environment advertises its tools as schemas, the rollout server's `--tool-call-parser` extracts the model's calls, and each result comes back as a `tool` message the next turn conditions on. Serve the model with the parser its family needs — without one the calls arrive as plain text, which the environment reads as a final answer and every episode ends on its first turn.

Three registry presets wrap a tool registry directly: `native_math` (`calculate`, `python`), `native_coding` (`python_repl`), `native_combined` (math, python, search, simulated files). `qa_search`, `exam_qa`, `swe`, `code_contests`, `codeforces` and `mcp` are subclasses or factory presets over the same protocol, so everything on this page applies to them.

## Configuration

```yaml
environment_type: native_math

environment_kwargs:
  max_tool_calls_per_turn: 5
  turn_overflow_penalty: 0.1
```

| Knob | Default | Effect |
|---|---|---|
| `max_tool_calls_per_turn` | `5` | calls executed per turn; extras are dropped and trimmed off the stored message, so every advertised call has a result |
| `require_tool_use` | `false` | flags a zero-tool-call finish in the step info; pays nothing by itself |
| `no_tool_use_penalty` | `0` | charged once on an episode that made no tool call |
| `multi_turn_reward` | `0` | paid once on more than one tool call |
| `turn_overflow_penalty` | `0` | charged once on a truncated episode |
| `length_cutoff_penalty` | `0` | charged per engine-cut turn the episode recovers from; the cut that exhausts the recovery cap pays the overflow price instead |
| `tool_budgets` | `{}` | per-tool episode caps, `{tool: cap}`; `0` disables a tool |
| `system_prompt` | `None` | prepended as the episode's system turn |
| `max_length_cutoff_recoveries` | `null` | engine-cut turns one episode may retry; `null` = every one within `max_turns` |

The knobs every environment shares — turn cap, per-call tool pay, observation cap, reasoning steer — are in the [overview](README.md#configuration).

## Tools

- `calculate` — restricted arithmetic and math functions, one `expression`.
- `python` / `python_repl` — Python REPL; in-process and import-free by default.
- `web_search` — `query` and optional `max_results` (default 5).
- `read_file`, `write_file`, `list_files` — a simulated per-episode file store, for tests and closed-world demos.

Registries are built by the factories in `src/environments/tools/factories.py` and composed with `NativeToolRegistry.combine(a, b)`. The `create_native_*` set is stateless; `create_session_*` binds the episode's persistent [sandbox session](sandbox.md) so files survive across turns. Pass `sandbox=` a `SandboxExecutor` to run code in a real isolated interpreter with imports, or `allow_imports=True` to lift the ban inside the in-process REPL — safe only when the whole process is already isolated.

## Reward

Per call: `+tool_success_reward` for a successful call, `-tool_error_penalty` for a failed one, paid up to `tool_reward_cap` per episode (default one paid call per turn of the budget, so call spam cannot out-earn the objective). A call refused over its `tool_budgets` cap, or one whose arguments cannot bind to the handler, books as a tool error without spending the budget.

The grade, in order: an episode that never completed grades 0; a `validator` callable in the row's context decides, 1 or 0; else the row's `answer` is graded all-or-nothing (exact match, then numeric); else completing the episode grades 1. The reward's `environment` term prices the grade as `weight × grade ^ exponent`, logged as `reward/objective` ([Reward Terms](../rewards.md#environment-arm)).

A row whose `answer` key holds null grades 0 and is marked `episode_invalid`, so the trainer drops it from the group baseline instead of grading every completion 1.

Per episode: `no_tool_use_penalty`, `multi_turn_reward` and `turn_overflow_penalty` are charged once each, by their conditions above, `length_cutoff_penalty` once per recovered cut, and they log together as `reward/tool_shaping`; the per-call deltas log as `reward/turn_shaping`. Overflow is charged on any truncated episode, including one killed mid-flight — except one its driver lost, where the fault is not the policy's. Every magnitude must be ≥ 0; the minus is applied at the use site, so a negative value raises instead of paying a penalty as a bonus.

Two kinds of turn earn nothing and carry no loss on either tokenization path ([Rollout Configuration](../async-grpo/rollouts.md#training-on-sampled-tokens)): a turn the engine cut off at its token cap or aborted, and a turn whose every call named a tool that does not exist.

A cut turn is nudged and retried within `max_turns` and `max_length_cutoff_recoveries`; a tool call salvaged from it is never executed. The unknown-tool observation lists the real tools, which stops a drifted policy from burning turns probing for a listing.

## Dataset

`{"prompt": str | list[dict], "answer": Any}`. For the native presets the `answer` column is optional: a row that carries one is graded against it, and without one completing the episode is the objective and every completion grades 1. The subclasses that grade only against it — [code contests](code-contests.md), [`exam_qa` and `qa_search`](benchmarks.md) — require the column (`requires_answer`). Extra columns reach the environment as context only when the training script's `context_fields` names them.

## Evaluation

```bash
python scripts/environments/inference/run_env.py \
    --env_type native_math --dataset <hf-id-or-dir> --split test \
    --prompt_field prompt --answer_field answer \
    --base_url http://localhost:8000/v1 --model <served-model> --num_examples 20
```

The endpoint needs tool calling enabled. Flags and the `--training_config` contract:
[Evaluating on an Environment](evaluation.md).

## From Python

```python
from src.environments.envs.protocols.native import NativeToolUseEnvironment
from src.environments.tools.factories import create_all_native_tools

env = NativeToolUseEnvironment(tool_registry=create_all_native_tools(), max_turns=10)
```

`AsyncNativeToolUseEnvironment` is the same protocol with `reset_async` / `step_async` and concurrent tool execution within a turn; use it for I/O-bound tools.

## Related pages

- [Environments](README.md) — registry and shared knobs.
- [Custom Environments](custom-environments.md) — your own tools and environments.
- [Code Execution Sandboxes](sandbox.md) — backends, languages, sessions.
