# ReAct Environments

The model writes a `Thought:`, then an `Action:` or a `Final Answer:`, and the environment parses the action out of the plain text ([ReAct](https://arxiv.org/abs/2210.03629), `src/environments/envs/protocols/react.py`).

Serve the model **without** a tool-call parser: one would lift the call out of the text, leaving no `Action:` line to parse.

Registry names, each with a fixed system prompt: `react_math` (`calculate`, `python`) and `react_search` (`web_search`). Shipped config: `examples/grpo/environmental/qwen3_5/vllm/qwen3.6-35b-a3b-react-math-full-ep4.yaml`.

```text
Thought: I need 25 * 4 first.
Action: calculate(expression="25 * 4")
Observation: 100
Final Answer: 110
```

An action's arguments are read as a Python call of literals (a triple-quoted `code` argument arrives whole), else by a `name=value` pattern. Arguments neither reads bind as none, which a tool with a required argument refuses as a tool error. An `Action:` that spells no call gets the format hint. Neither is an episode error.

## Configuration

```yaml
environment_type: react_math

environment_kwargs:
  thought_reward: 0.02
```

| Knob | Default | Effect |
|---|---|---|
| `thought_reward` | `0.02` | Added on a turn that carries a Thought |
| `no_thought_penalty` | `0.05` | Subtracted on a turn that acts or answers with no Thought |
| `require_thought` | `true` | Gate for that penalty; a turn doing neither is free either way |
| `tool_budgets` | `{}` | Per-tool episode caps, `{tool: cap}`; `0` disables a tool, an over-cap call is refused as a tool error |
| `search_backend` | auto | `react_search` only: the `web_search` backend, named and validated as on [`qa_search`](../tasks/qa-benchmarks.md#configuration) |

The [common knobs](../README.md#common-knobs) apply too. The native protocol's `no_tool_use_penalty`, `turn_overflow_penalty` and `length_cutoff_penalty` raise `TypeError` here.

## Tools

- `calculate` — restricted arithmetic and math functions, one `expression`.
- `python` — the in-process restricted REPL, imports blocked.
- `web_search` — `query` and optional `max_results` (default 5).

## Reward

The grade is 1 when the final answer matches, 0 otherwise, priced as `reward/objective` ([Reward Terms](../../../grpo/rewards/README.md#environment-arm)).

- Matching is all-or-nothing: exact match after normalization (`\boxed{}`, bold, answer prefixes), then a numeric compare at 1% relative tolerance.
- An `answer_validator` `(final_answer, expected) -> bool` grades instead wherever one is passed, and clears `requires_answer` ([From Python](#from-python)). Without one the presets require the `answer` column.
- A validator that raises marks the episode `episode_invalid` and grades it 0.

Per-turn shaping, logged as `reward/turn_shaping`: `+thought_reward` or `-no_thought_penalty` for the Thought, and `+tool_success_reward` / `-tool_error_penalty` for the call, paid up to `tool_reward_cap`.

### Refused and unproductive turns

- An `Action:` naming an unregistered tool is a tool error whose observation lists the real tools; the turn trains only on a negative advantage ([Untrainable turns](../../objective.md#untrainable-turns)).
- An `Action:` whose arguments do not bind is refused unrun ([Argument binding](native-tool-use.md#argument-binding)): `tool_error_penalty`, no `tool_budgets` spent.
- A call that ends on a sandbox fault is booked by its class and ends the episode ([Sandbox faults](../sandbox.md#sandbox-faults)).
- A cut or empty turn is neither executed nor graded. It is nudged and retried ([Unproductive turns](../../episodes.md#unproductive-turns)) and trains only on a negative advantage.
- Text with neither an Action nor a Final Answer gets the format hint and stays trainable.

## Dataset

`{"prompt": str | list[dict], "answer": str | number}`; the `answer` column is required (`requires_answer`). Strip dataset markup from `answer` during preparation ([answer grading](../tasks/qa-benchmarks.md#reward)).

## Evaluation

```bash
python scripts/environments/inference/run_env.py \
    --env_type react_math --dataset <hf-id-or-dir> --split test \
    --prompt_field prompt --answer_field answer \
    --base_url http://localhost:8000/v1 --model <served-model> --num_examples 20
```

Flags and the `--training_config` contract: [Evaluation](../../evaluation.md).

## From Python

`answer_validator` is a callable, so only the Python path sets it:

```python
from src.environments.envs.protocols.react import create_react_math_environment

env = create_react_math_environment(max_turns=10, answer_validator=my_validator)
```

## Related pages

- [Environments](../README.md) — registry and common knobs.
- [Native Tool-Use](native-tool-use.md) — the function-calling protocol.
- [Async GRPO with Environments](../../README.md) — the trainer.
