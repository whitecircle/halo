# Native Tool-Use Environments

`NativeToolUseEnvironment` (`src/environments/envs/protocols/native.py`) runs the OpenAI/vLLM function-calling protocol: the environment advertises tool schemas, the server's `--tool-call-parser` extracts the calls, and each result returns as a `tool` message.

Serve the model with its family's parser ([Tool calls](../../episodes.md#tool-calls)). Without one, vLLM rejects the request (400) and SGLang returns the calls as plain text, so every episode ends on its first turn.

Presets: `native_math` (`calculate`, `python`), `native_coding` (`python_repl`), `native_combined` (math, python, search, simulated files; `search_backend` names the search backend as on [`qa_search`](../tasks/qa-benchmarks.md#configuration)). `qa_search`, `exam_qa`, `swe`, `code_contests`, `codeforces` and `mcp` build on the same protocol, so this page applies to them.

## Configuration

```yaml
environment_type: native_math

environment_kwargs:
  max_tool_calls_per_turn: 5
  turn_overflow_penalty: 0.1
```

| Knob | Default | Effect |
|---|---|---|
| `max_tool_calls_per_turn` | `5` | Calls executed per turn; extras are dropped from the stored message |
| `no_tool_use_penalty` | `0` | Charged once on an episode that made no tool call |
| `turn_overflow_penalty` | `0` | Charged once on a truncated episode |
| `length_cutoff_penalty` | `0` | Charged per recovered unproductive turn (engine-cut, or neither a tool call nor visible content). The turn that exhausts the recovery cap, or lands on the last turn, pays `turn_overflow_penalty` instead |
| `tool_budgets` | `{}` | Per-tool episode caps, `{tool: cap}`; `0` disables a tool ([Tool budgets](#tool-budgets)) |
| `system_prompt` | `None` | Prepended as the episode's system turn |

The [common knobs](../README.md#common-knobs) apply too.

## Tools

- `calculate` — restricted arithmetic and math functions, one `expression`.
- `python` / `python_repl` — Python REPL; in-process and import-free by default.
- `web_search` — `query` and optional `max_results` (default 5).
- `read_file`, `write_file`, `list_files` — a simulated per-episode file store.

The factories in `src/environments/tools/factories.py` build registries: `create_native_*` stateless, `create_session_*` bound to the episode's persistent [sandbox session](../sandbox.md). `NativeToolRegistry.combine(a, b)` composes two. Pass `sandbox=` a `SandboxExecutor` to run code in a real interpreter with imports ([Choosing a backend](../sandbox.md#choosing-a-backend)).

## Reward

The grade, first rule that applies:

1. An episode that never completed grades 0.
2. A `validator` callable in the row's context decides, 1 or 0.
3. Else the row's `answer`, all-or-nothing (exact match, then numeric). A null `answer` cell grades 0 and marks the episode `episode_invalid`, so it leaves the group baseline.
4. Else completing the episode grades 1 (not under [`swe`](../tasks/swe.md#reward), which has no completion fallback).

The reward's `environment` term prices the grade as `reward/objective` ([Reward Terms](../../../grpo/rewards/README.md#environment-arm)). Shaping adds on top:

- Per call (`reward/turn_shaping`): `+tool_success_reward` / `-tool_error_penalty`, up to `tool_reward_cap`.
- Per episode (`reward/tool_shaping`): the three penalties in [Configuration](#configuration). Overflow is charged on any truncated episode, including one killed mid-flight, but not on one its driver lost.
- A call that ends on a sandbox fault is booked by its class and ends the episode ([Sandbox faults](../sandbox.md#sandbox-faults)).

### Tool budgets

- A call over its `tool_budgets` cap is refused as a tool error and spends no budget. The reply, `Error: Not executed: this task's budget for <tool> is spent.` (a tool's own `budget_message` replaces the text after `Error:`), names neither the cap nor the calls left.
- A call to a tool capped at `0` gets the same reply; it is a tool error but not a refusal (`ToolDisabled`).

### Argument binding

Every call is checked against the tool's schema before it runs or spends budget (`src/environments/tools/definitions.py`). A call that fails is refused as a tool error and never ends the episode:

- `Error: <tool>: unknown argument '<name>'; its arguments are …`
- `Error: <tool>: missing a required argument: '<name>'`
- `Error: <tool>: code must be a string, got list` — a list or object for a `string` parameter. A number or boolean there reads as its Python string (`5` as `5`, `true` as `True`), and a null as the argument left out.

Unparseable `arguments` (neither JSON nor a Python literal) bind as none. An MCP property with no top-level `type` takes any value. A call to a tool that does not exist gets an observation listing the real tools.

### Untrainable turns

Three kinds of turn earn nothing and train only on a negative advantage ([Untrainable turns](../../objective.md#untrainable-turns)):

- a turn the engine cut at its token cap;
- a turn with neither a tool call nor visible content (a stop inside its reasoning);
- a turn whose every call was unknown or refused unrun: arguments that do not bind, a spent `tool_budgets` cap, or its handler's `ToolCallRefused`.

A cut or empty turn is nudged and retried within one shared cap ([Unproductive turns](../../episodes.md#unproductive-turns)). A tool call salvaged from a cut turn is never executed; it stays on the turn for a judge's view ([What a judge reads](../../../grpo/rewards/judge.md#what-a-judge-reads)). When the engine's parser returns the unfinished call (vLLM's `qwen3_xml` does), the nudge says the call ran past the length limit.

## Dataset

`{"prompt": str | list[dict], "answer": Any}`.

- For the native presets `answer` is optional: without one, every completed episode grades 1. Subclasses that grade only against it require it ([`requires_answer`](../README.md#common-knobs)).
- Extra columns reach the environment only when the training script's `context_fields` names them.

## Evaluation

```bash
python scripts/environments/inference/run_env.py \
    --env_type native_math --dataset <hf-id-or-dir> --split test \
    --prompt_field prompt --answer_field answer \
    --base_url http://localhost:8000/v1 --model <served-model> --num_examples 20
```

The endpoint needs tool calling enabled. Flags and the `--training_config` contract: [Evaluation](../../evaluation.md).

## From Python

A custom tool set goes through the constructor ([A custom tool](../custom.md#a-custom-tool)):

```python
from src.environments.envs.protocols.native import NativeToolUseEnvironment
from src.environments.tools.factories import create_all_native_tools

env = NativeToolUseEnvironment(tool_registry=create_all_native_tools(), max_turns=10)
```

`AsyncNativeToolUseEnvironment` runs a turn's tool calls concurrently; use it for I/O-bound tools.

## Related pages

- [Environments](../README.md) — registry and common knobs.
- [Custom Environments](../custom.md) — your own tools and environments.
- [Code Execution Sandboxes](../sandbox.md) — backends, languages, sessions.
