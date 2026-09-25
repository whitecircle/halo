# ReAct Environments

The model writes a `Thought:`, then an `Action:` or a `Final Answer:`, and the environment parses the action out of the plain text ([ReAct](https://arxiv.org/abs/2210.03629), `src/environments/envs/protocols/react.py`). No tool schema reaches the server, so serve the model **without** a tool-call parser: a parser lifts the call out of the text, the environment finds no `Action:` line, and the turn burns on a format hint.

Two registry names: `react_math` (`calculate`, `python`) and `react_search` (`web_search`); each hardcodes its own system prompt. Shipped config: `examples/grpo/environmental/qwen3_5/vllm/qwen3.6-35b-a3b-react-math-full-ep4.yaml`.

```text
Thought: I need 25 * 4 first.
Action: calculate(expression="25 * 4")
Observation: 100
Final Answer: 110
```

## Configuration

```yaml
environment_type: react_math

environment_kwargs:
  thought_reward: 0.02
```

| Knob | Default | Effect |
|---|---|---|
| `thought_reward` | `0.02` | added on a turn that carries a Thought |
| `no_thought_penalty` | `0.05` | subtracted on a turn that acts or answers with no Thought |
| `require_thought` | `true` | gate for that penalty; a turn doing neither is free either way |
| `tool_budgets` | `{}` | per-tool episode caps, `{tool: cap}`; `0` disables a tool, an over-cap call is refused as a tool error |

The knobs every environment shares are in the [overview](README.md#configuration). The native protocol's episode-level knobs (`no_tool_use_penalty`, `multi_turn_reward`, `turn_overflow_penalty`, `require_tool_use`) are not parameters here and raise `TypeError`.

## Tools

- `calculate` — restricted arithmetic and math functions, one `expression`.
- `python` — in-process sandboxed REPL, imports blocked.
- `web_search` — `query` and optional `max_results` (default 5).

## Reward

The grade is 1 when the final answer matches, 0 otherwise, priced by the reward's `environment` term as `reward/objective` ([Reward Terms](../rewards.md#environment-arm)). Matching is all-or-nothing — exact match after normalization (`\boxed{}`, bold, answer prefixes), then a numeric compare at 1% relative tolerance. `answer_validator`, a `(final_answer, expected) -> bool`, grades instead wherever one is passed and clears `requires_answer`; without it an answer-less context would grade any Final Answer 1, so the presets require the column. A validator that raises marks the episode `episode_invalid` (its exception as the `episode_invalid_reason`) and grades it 0: no check stands in for it, since the default one would pay a Final Answer on a row with no expected answer.

Per turn: `+thought_reward` or `-no_thought_penalty` for the Thought, `+tool_success_reward` / `-tool_error_penalty` for the call, paid up to `tool_reward_cap`; these deltas log as `reward/turn_shaping`. Every magnitude must be ≥ 0; the minus is applied at the use site, so a negative value raises instead of paying a penalty as a bonus.

A call that ends on a sandbox fault is booked by its class and ends the episode ([Sandbox faults](sandbox.md#sandbox-faults)). An `Action:` naming an unregistered tool is a tool error whose observation lists the real tools, and that turn is dropped from training. A turn the engine cut short, or one that comes back empty, is neither executed nor graded: the environment appends a nudge asking for the Action or Final Answer and retries within `max_turns` and `max_length_cutoff_recoveries`; both kinds of turn are dropped from training. A turn with text but neither an Action nor a Final Answer gets the format hint instead and stays trainable.

## Dataset

`{"prompt": str | list[dict], "answer": str | number}`; the `answer` column is required (`requires_answer`). Strip dataset markup from `answer` during preparation ([answer grading](benchmarks.md#reward)).

## Evaluation

Run a checkpoint on a few rows first:

```bash
python scripts/environments/inference/run_env.py \
    --env_type react_math --dataset <hf-id-or-dir> --split test \
    --prompt_field prompt --answer_field answer \
    --base_url http://localhost:8000/v1 --model <served-model> --num_examples 20
```

Flags and the `--training_config` contract: [Evaluating on an Environment](evaluation.md).

## From Python

```python
from src.environments.envs.protocols.react import create_react_math_environment

env = create_react_math_environment(max_turns=10)
```

## Related pages

- [Environments](README.md) — registry and shared knobs.
- [Native Tool-Use](native-tool-use.md) — the function-calling protocol.
- [Async GRPO with Environments](../async-grpo/README.md) — the trainer.
