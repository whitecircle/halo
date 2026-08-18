# ReAct Environments

Implements the [ReAct paradigm](https://arxiv.org/abs/2210.03629): the model interleaves **Thought**, **Action**, and **Observation** steps, ending in a **Final Answer**. The action is parsed out of plain text (`src/environments/envs/protocols/react.py`), so no server-side tool-call parser is needed.

The rollout carries **no** `tools=` block: `get_tools_schema()` returns `None`, and the available tools are named in the system prompt instead. Serve the model without `--tool-call-parser` for this protocol — a parser lifts the call out of `content`, the env then finds no `Action:` line, and the turn burns on the format hint with no tool-error penalty and no counter to show for it.

```text
Thought: I need to calculate 25 * 4 first.
Action: calculate(expression="25 * 4")
Observation: 100
Thought: Now I need to add 10 to get the final answer.
Action: calculate(expression="100 + 10")
Observation: 110
Final Answer: 110
```

Two registry presets: `react_math` (`calculate` — restricted math eval; `python` — sandboxed REPL) and `react_search` (`web_search`, with `query` and optional `max_results`, default 5). Select either from YAML with `environment_type`; both read `max_turns`, `success_reward`, and `failure_reward` from it. Example config: `examples/grpo/environmental/qwen3_5/vllm/qwen3.6-35b-a3b-react-math-full-ep4.yaml`.

```python
from src.environments.envs.protocols.react import create_react_math_environment

env = create_react_math_environment(max_turns=10, success_reward=1.0)
```

## Reward structure

Terminal: `success_reward` (default `1.0`) on a matching final answer, `failure_reward` (default `0.0`) otherwise.

Per-step deltas accrue during the episode, all `ReActEnvironment` constructor kwargs: `+0.02` when a step includes reasoning (`thought_reward`), `−0.05` when a step that acts or answers omits one under `require_thought=True` (`no_thought_penalty`; a step doing neither is free), `+0.05` per successful tool call (`tool_success_reward`), `−0.1` on a tool error (`tool_error_penalty`). A reasoning step is any line headed `Thought:`, `Think:`, or `Reasoning:` (case-insensitive).

Every knob is a **magnitude** (≥ 0) — the minus is applied at the use site, and a negative config value raises at construction, so a sign flip cannot turn a penalty into a farmable bonus.

ReAct has no episode-level shaping: `no_tool_use_penalty`, `multi_turn_reward`, `turn_overflow_penalty`, and `require_tool_use` belong to the [native](native-tool-use.md) protocol and raise `TypeError` here.

**A turn cut off at its token cap** (`finish_reason == "length"`) **before it produced an Action or a Final Answer is a failed turn, not a format failure.** The episode appends a ReAct-shaped nudge (`ReActEnvironment.LENGTH_CUTOFF_NUDGE`, asking for the Action or Final Answer — never for shorter reasoning) and retries on its remaining `max_turns`; the count surfaces as `episode/length_cutoff_turns`.

Such a turn is **unpriced in full**: neither `thought_reward` nor `no_thought_penalty` applies to a turn the model never finished. A turn whose text already carries its `Action:` or `Final Answer:` takes the normal path.

### Answer validation

`_compute_reward` calls `compute_answer_reward` with the default chain `DEFAULT_METHODS = [exact_match, numeric_match]` — all-or-nothing: a match pays `success_reward`, anything else `failure_reward`. Validator definitions and tolerances: [Shared reward validation](benchmarks.md#shared-reward-validation).

Substring containment is deliberately not one of the shipped matchers — it inflates rewards (expected `"7"` matches `"17"`). Pass such a matcher via `methods=` only when substring semantics are intended.

Override the check with `answer_validator`, a `(final_answer, expected) -> bool` callable; one that raises is logged and falls back to the default chain.

```python
env = ReActEnvironment(tool_registry=registry, answer_validator=lambda a, e: a.strip().lower() == str(e).lower())
```
