# Reasoning Budgets

An episode's effort level reaches the model through the chat template and sets a per-turn reasoning budget, which
the engine enforces or treats as a soft target. Two trainer-side terms price the reasoning tokens.

## Effort levels

| Knob (`environment_kwargs`) | Default | Effect |
|---|---|---|
| `reasoning_effort` | `null` on `BaseEnvironment`, `medium` on `code_contests` | `low` / `medium` / `high` / `random` / `null` |
| `reasoning_effort_profiles` | the class's per-level table | per-level overrides, merged over that table |

- `random` draws one level per generation group, so a group shares its conditioning. An eval round draws it from the
  problem's text, so every checkpoint scores a problem at the same level.
- A profile takes `thinking_tokens` (int ≥ 1, the per-turn budget) and `enforce_thinking_cap` (default `true`).
  Environment classes add keys, such as code contests' tool budgets
  ([Reasoning effort](environments/tasks/code-contests/README.md#reasoning-effort)).
- An unknown profile key, a value below its minimum, or `enforce_thinking_cap: false` without `thinking_tokens`
  raises at environment construction.

## Effort templates

- gpt-oss's template renders the level natively and states no budget.
- The stock Qwen3.x and Gemma 4 templates have no effort variable. The code-contests recipes pin
  `jinja-templates/qwen3/qwen3.6-reasoning-effort.jinja` and `jinja-templates/gemma4/gemma4-reasoning-effort.jinja`: the hub
  template cut to text only with thinking always on, plus a system-block line built from `reasoning_effort`,
  `reasoning_budget` and `reasoning_budget_soft` that states the budget as a cut-off, or as a target when soft. An
  episode with no level gets no line.
- The Gemma 4 file opens every system turn with `<|think|>`, so it needs no `enable_thinking` variable.

Pin either with `chat_template:` + `force_chat_template: true` and serve the same file (`VLLM_CHAT_TEMPLATE` /
`SGLANG_CHAT_TEMPLATE` in the compose files; [Chat template](episodes.md#chat-template)).

`reasoning_effort` travels as the request's top-level field (copied into `chat_template_kwargs` on SGLang), the two
budget variables in `chat_template_kwargs`. Both engines derive their thinking toggle from that field
(`enable_thinking`, plus `thinking` on SGLang): on for any level but `none`, unless `rollout_chat_template_kwargs`
sets it. The trainer's own renders carry the same variables, but not the derived toggle, and
`rollout_chat_template_kwargs` refuses all three.

## Reasoning budget

The budget is per turn; `rollout_max_tokens` bounds the whole turn, reasoning and answer together.

| Knob | Default | Effect |
|---|---|---|
| `rollout_max_thinking_tokens` | `null` | run-wide per-turn reasoning cap (vLLM `thinking_token_budget`) |
| `rollout_max_answer_tokens` | `null` | the most a turn may generate past its reasoning cap ([Answer room](#answer-room)) |

Any reasoning cap on vLLM needs a reasoning parser and `VLLM_USE_V2_MODEL_RUNNER=0` on the server
([Rollout Servers](../../infrastructure/rollout-servers.md#vllm)).

### Enforced and soft budgets

- **Enforced** (default): the engine closes the turn's reasoning at the budget, mid-thought if need be.
- **Soft** (`enforce_thinking_cap: false`): the template states the budget as a target and the
  [reasoning length terms](#reasoning-length-reward) read it. The engine closes the reasoning only at the run's
  ceiling: `rollout_max_thinking_tokens`, else `rollout_max_tokens − rollout_max_answer_tokens`; none if both are unset.

With `T` the level's `thinking_tokens`, `M` `rollout_max_thinking_tokens`, `A` `rollout_max_answer_tokens` and `R`
`rollout_max_tokens`, each cell gives a turn's budget (what the template is handed and the reasoning terms read),
which is also its engine cap unless a cap is named:

| Level | `M` set | `A` set, `M` unset | neither |
|---|---|---|---|
| Enforced `T` | `min(T, M)` | `T` | `T` |
| Soft `T` | `T`, cap `M` | `T`, cap `R − A` | `T`, no cap |
| No `thinking_tokens` (or no level) | `M` | refused | no budget, no cap |

A turn's total is `R`, or `min(R, cap + A)` where a cap and `A` are both set. An episode output budget narrows a late
turn's cap and total ([Trajectory length](episodes.md#trajectory-length)), never the stated budget.

Refused at trainer construction and at eval start, for every level the environment can draw (all three under
`random`):

- a budget at or above `R`, which leaves no answer room;
- a soft `T` at or above its ceiling (`M`, or `R − A`), where the engine would enforce it;
- `A` set while a drawable level has no engine cap.

`M` or `A` outside `[1, R)` is refused when the config is parsed.

### Answer room

`rollout_max_answer_tokens` bounds what a turn generates past its reasoning cap. Without it, a turn whose reasoning
the engine closed can keep reasoning in its visible output (comment lines in a tool call, say) until `R` cuts it.

The answer room is `R − cap`, or `A` where smaller; the whole turn without a cap. It is never stated to the model.

### SGLang

SGLang takes no `thinking_token_budget`, the vLLM field every reasoning cap travels in.

- `rollout_max_thinking_tokens` is refused when the config is parsed.
- An enforced budget, a soft ceiling and a retry reserve are dropped from the request (warned once per process), so
  nothing caps reasoning below the turn's total.
- The template still states the budget and the reasoning terms still read it. Prefer a soft budget here: an enforced
  one is stated as a cut this engine never makes.
- `rollout_max_answer_tokens` still caps the turn at the cap plus `A`, since `max_tokens` binds on both engines.

### Retry reserve

The turn after an unproductive one ([Unproductive turns](episodes.md#unproductive-turns)) runs under a reserve: a
quarter of the episode's per-turn budget, which is `rollout_max_thinking_tokens` where the level sets none.

- It is a hard cap even under a soft budget, so a cut never buys a second budget.
- Under `rollout_max_answer_tokens` the retry's total is `min(R, reserve + A)`. Under an output budget the reserve
  clamps what is left and never shrinks with it.
- The template still states the level's budget. An episode with no budget retries uncapped.

## Thinking-cap metric

`episode/thinking_cap_turns` counts turns whose reasoning (sampled ids through `rollout_reasoning_end_token`) reached
the cap they ran under: an enforced budget, a soft ceiling, or a retry reserve. It ignores the narrower cap an output
budget leaves a late turn.

It is logged only with a vLLM cap that can bind, `train_on_sampled_tokens`, and a `rollout_reasoning_end_token` that
encodes to one added token. gpt-oss's five-token final-channel opener logs none, nor do the standalone eval scripts.

## Reasoning length reward

Two trainer-side terms add to the task reward before advantages.

| Knob | Default | Effect |
|---|---|---|
| `reasoning_floor` | `0` (off) | weight of the [shortfall penalty](#floor) |
| `reasoning_price` | `null` (off) | per-level map: reward units per 1,000 reasoning tokens |
| `reasoning_price_cap` | `0.1` | the most the price charges one episode; refused at another value while the price is off |
| `reasoning_price_past_budget` | `false` | price only reasoning past each turn's soft budget; needs `reasoning_price` |

- While `drop_degenerate_groups` is on (the default), both reach the gradient only through groups the environment's
  reward separates: it judges a group without them ([Advantages](objective.md#advantages)).
- Watch `reward/reasoning_price` and `reward/reasoning_floor`.

### Floor

The reference is three quarters of the episode's per-turn budget, enforced or soft
([table](#enforced-and-soft-budgets)).

- An episode whose reasoning tokens, summed over its turns, fall short pays `-floor × shortfall / reference`: the
  whole weight for no reasoning at all.
- It reads the total, not a per-turn mean, so an extra short turn never lowers the score.
- An episode with no budget or no assistant turn pays nothing.
- A run where no drawable level has a budget and `rollout_max_thinking_tokens` is unset is refused at trainer
  construction.

### Price

An episode pays `-min(reasoning_price_cap, price[level] × tokens / 1000)` on its reasoning tokens summed over turns.

- The map must name exactly `low`, `medium` and `high`, and the environment must set `reasoning_effort`; both are
  refused at trainer construction.
- Price the lowest level highest, so the same trace costs most where little reasoning was asked.
- Code and tool calls are free. An episode with no level pays nothing.

Under group-relative advantages, a price on every token rewards the sibling that reasons less, whatever the outcome.
Hence the cap. A price without the floor is accepted, and then nothing pays a level to reason more. Pair it with the
floor, which must out-slope the price below its reference, so a short episode is still paid to reason more
(`tests/cpu/config/test_env_grpo_reward_economy.py` holds every shipped recipe that sets a price to that).

### Price past the budget

`reasoning_price_past_budget: true` prices only what each turn reasons past its soft budget:
`-min(cap, price[level] × Σ max(0, turn − budget) / 1000)`.

- It is then a soft budget's only ceiling below the engine's. Reasoning within the budget is free.
- Every drawable level must have a soft budget, or trainer construction refuses the run.

### Sizing

Keep `reasoning_price_cap + reasoning_floor` below what the environment charges for the decisions it prices. Under
`reasoning_price_past_budget` each alone must sit below it: an episode pays one at most, since a turn past its budget
clears the floor's reference. The code-contests recipes run the past-budget mode and size both under their
resubmission price ([Shipped recipes](environments/tasks/code-contests/README.md#shipped-recipes)).
