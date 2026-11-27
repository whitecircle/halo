# Reasoning Budgets

The model is told one thing about its reasoning: the episode's effort level, through the chat template. A level's
profile can set a reasoning budget and price for the whole episode, which the trainer charges against; the engine's
per-turn reasoning cap is run-wide and the same at every level. No budget, price or cap reaches the prompt.

## Effort levels

| Knob (`environment_kwargs`) | Default | Effect |
|---|---|---|
| `reasoning_effort` | `null` on `EnvironmentBase`, `medium` on `code_contests` | `low` / `medium` / `high` / `random` / `null` |
| `reasoning_effort_profiles` | empty at every level | per-level keys |

- `random` draws one level per generation group, so a group shares its conditioning. An eval round draws it from the
  problem's text, so every checkpoint scores a problem at the same level.
- A row's own level reaches its episode as a `reasoning_effort` column named in `context_fields` and takes
  precedence over the environment's `reasoning_effort`; a null cell keeps the environment's. Under an environment
  level of `random`, the trainer stamps the group's draw over every row's level, in training and in its eval rounds;
  otherwise, and in the standalone eval, a row's cell is read, and one other than `low`, `medium`, `high` or null
  raises (a row's own `random` would draw differently across its group). The standalone eval draws from the problem's
  text only for a missing or null cell. The
  [reasoning length terms](#reasoning-length-reward) switch on from the levels the environment draws alone
  ([Construction checks](#construction-checks)), so a row's other level is priced only while one of those is.
- A profile takes `reasoning_budget` (int ≥ 0) and `reasoning_price` (≥ 0), the
  [reasoning length terms](#reasoning-length-reward). Environment classes add keys, such as code contests' tool
  budgets ([Reasoning effort](environments/tasks/code-contests/README.md#reasoning-effort)).
- An unknown level or key, a value below its minimum, a non-int count, or `reasoning_price` without
  `reasoning_budget` raises at environment construction.

## Effort templates

- gpt-oss's harmony template renders the level natively, `medium` when none is passed.
- The stock Qwen3.x and Gemma 4 templates have no effort variable. The code-contests recipes pin
  `jinja-templates/qwen3/qwen3.6-reasoning-effort.jinja` and `jinja-templates/gemma4/gemma4-reasoning-effort.jinja`: the hub
  template cut to text only, plus one system-block line read off `reasoning_effort`.
- The Qwen3.6 file thinks unless `enable_thinking` is false, which closes the block in the generation prompt as the
  hub template does ([Thinking off](#thinking-off)).
- The Gemma 4 file opens every system turn with `<|think|>`: thinking is always on.

| Level | Line |
|---|---|
| `low` | `Reasoning effort: low. Keep your reasoning short: take the most direct approach and act on it.` |
| `medium` | `Reasoning effort: medium. Reason as far as the problem needs, without exhaustive exploration.` |
| `high` | `Reasoning effort: high. Reason thoroughly: weigh alternatives and check your work before you act.` |

Any other level renders `Reasoning effort: <level>.` alone, and a render with no level gets no line.

Pin either with `chat_template:` + `force_chat_template: true` and serve the same file (`VLLM_CHAT_TEMPLATE` /
`SGLANG_CHAT_TEMPLATE` in the compose files; [Chat template](episodes.md#chat-template)).

`reasoning_effort` travels as the request's top-level field (copied into `chat_template_kwargs` on SGLang). The
trainer's own renders carry it too, and `rollout_chat_template_kwargs` refuses it.

## Thinking off

`rollout_thinking: false` samples the policy with no reasoning block, so it works in its visible messages and tool
calls; thinking is the default ([AsyncRolloutConfig](../../reference/configuration-reference.md#asyncrolloutconfig)).
Effort levels still apply: the template states the level, and a profile's non-reasoning keys (code contests' tool
budgets) still bind.

- The config adds `enable_thinking: false` to `rollout_chat_template_kwargs`, which then reaches every request and the
  trainer's renders alike. vLLM derives `enable_thinking` from the top-level level unless the request's own kwargs
  carry it, and its qwen3 reasoning parser reads a turn under `enable_thinking: false` as content.
- The chat template must read the switch. The training script renders a generation prompt with `enable_thinking`
  false and true before any server check and refuses a template that renders both alike.
  `qwen3/qwen3.6-reasoning-effort.jinja` reads it; the Gemma 4 file does not, so it is refused.
- Serve the same file, the server's other flags unchanged (`--reasoning-parser` included). The
  [startup probe](episodes.md#chat-template) refuses a server whose generation prompt differs from the trainer's at
  all, or that does not answer it: the request carries the switch, so no derived toggle explains a difference.

Refused when the config is parsed, each reading reasoning no turn has:

- `reasoning_floor` above 0: an episode with no reasoning falls short of its whole budget, so every failed episode
  would pay the full weight, a gradient back toward reasoning;
- `reasoning_price_cap` off its default, and `reasoning_terms_in_failed_groups`;
- `rollout_max_thinking_tokens` and `rollout_max_answer_tokens`: a cap counts a turn's sampled tokens as reasoning up
  to a close the turn never writes.

The environment's `reject_settings_without_thinking` refuses more, in the training script, again in the trainer's
constructor, and when an eval loads the training config:

- `carry_reasoning`: no turn reasons, so there is nothing to send back;
- a `reasoning_budget` or `reasoning_price` in any level's profile;
- code contests without the [notes harness](environments/tasks/code-contests/tools.md#notes), whose reply rule
  otherwise leaves a policy with no reasoning nowhere to work.

## Reasoning cap

The engine's reasoning cap is per turn and run-wide: every level runs under the same one. `rollout_max_tokens` bounds
the whole turn, reasoning and answer together.

`rollout_max_thinking_tokens` is the run-wide per-turn reasoning cap (vLLM `thinking_token_budget`), and
`rollout_max_answer_tokens` the most a turn may generate past it ([Answer room](#answer-room)); types and defaults are
in [AsyncRolloutConfig](../../reference/configuration-reference.md#asyncrolloutconfig).

- The cap is `rollout_max_thinking_tokens`, else `rollout_max_tokens − rollout_max_answer_tokens`, else none. The
  engine closes the reasoning there, mid-thought if need be.
- A turn's total is `rollout_max_tokens`, or the cap plus `rollout_max_answer_tokens` where smaller. An episode
  output budget narrows a late turn's cap and total ([Trajectory length](episodes.md#trajectory-length)).
- Either knob outside `[1, rollout_max_tokens)` is refused when the config is parsed.
- Any cap on vLLM needs a reasoning parser on the server, which enforces it on the image's Model Runner V2
  ([Rollout Servers](../../infrastructure/rollout-servers.md#vllm)).
- A vLLM cap on Mistral 4 is refused at trainer construction and when an eval loads a training config: vLLM serves the
  family with its `mistral_common` tokenizer, which encodes the `[THINK]`/`[/THINK]` markers as plain text, so the
  budget would never start counting ([Mistral4](../../models/mistral4.md#serving)). The fact is the vLLM client's
  `UNENFORCEABLE_THINKING_BUDGET_MODEL_TYPES` (`halo/weight_sync/clients/vllm.py`).

### Answer room

`rollout_max_answer_tokens` bounds what a turn generates past its reasoning cap. Without it, a turn whose reasoning
the engine closed can keep reasoning in its visible output (comment lines in a tool call, say) until
`rollout_max_tokens` cuts it.

The answer room is `rollout_max_tokens` less the cap, or `rollout_max_answer_tokens` where smaller; the whole turn
without a cap. It is never stated to the model.

### SGLang

SGLang takes no `thinking_token_budget`, the vLLM field the cap travels in.

- `rollout_max_thinking_tokens` is refused when the config is parsed.
- The cap `rollout_max_answer_tokens` leaves is dropped from the request (warned once per process), so nothing caps
  reasoning below `rollout_max_tokens`. The knob then only sets the answer room an output budget keeps a late turn.
- The level still reaches the template, and the [reasoning length terms](#reasoning-length-reward) alone hold
  reasoning length.

## Thinking-cap metric

`episode/thinking_cap_turns` counts turns whose reasoning (sampled ids through `rollout_reasoning_end_token`) reached
the run's reasoning cap. It ignores the narrower cap an output budget leaves a late turn.

It is logged only on a capped vLLM run, with `train_on_sampled_tokens` and a `rollout_reasoning_end_token` that
encodes to one added token. gpt-oss's six-token move into the final channel logs none, nor do the standalone eval scripts.

## Reasoning length reward

Two trainer-side terms add to the task reward before advantages. Both read an episode's reasoning tokens summed over
its turns against its level's `reasoning_budget`: the price charges what it reasoned past the budget, the floor a
failed episode's shortfall under it.

Each level's profile sets `reasoning_budget` (the reasoning tokens an episode spends free; `0` prices every token) and
`reasoning_price` (reward units per 1,000 reasoning tokens past it; off at `0`). `reasoning_price_cap` is the most the
price charges one episode, `reasoning_floor` weighs the [floor](#floor), and `reasoning_terms_in_failed_groups` charges
both terms in a [failed group](#failed-groups) too. Both terms are off by default; types and defaults are in
[AsyncRolloutConfig](../../reference/configuration-reference.md#asyncrolloutconfig). The per-level keys sit in the
effort profiles, the rest at the top level:

```yaml
environment_kwargs:
  reasoning_effort: random
  reasoning_effort_profiles:
    low: {reasoning_budget: 40000, reasoning_price: 0.008}
    medium: {reasoning_budget: 50000, reasoning_price: 0.006}
    high: {reasoning_budget: 60000, reasoning_price: 0.004}
reasoning_floor: 0.25
```

- Reasoning tokens only: visible output, code and tool calls are free. An episode whose level sets no budget pays
  neither term.
- The price is 0 within the budget and the floor 0 past it, so an episode pays one term at most and a short solve
  pays neither: the policy can stay short where the problem is easy.
- While `drop_degenerate_groups` is on (the default), both reach the gradient only through groups the environment's
  reward separates: it judges a group without them ([Advantages](objective.md#advantages)).
- Watch `reward/reasoning_price` and `reward/reasoning_floor`, each logged while its term is on.

### Price

An episode past its budget pays `-min(reasoning_price_cap, price × (tokens − budget) / 1000)`, a solve included.

- Price the lowest level highest, so the same trace costs most where little reasoning was asked.
- At `reasoning_budget: 0` every token is priced, and group-relative advantages then reward the sibling that reasons
  less, whatever the outcome. Pair a price with the floor, which pays a failed episode to reason up to its budget
  (`tests/cpu/config/test_env_grpo_reward_economy.py` holds every shipped recipe to that).

### Floor

A baseline episode that earned no objective and reasoned under its level's budget pays
`-reasoning_floor × (budget − tokens) / budget`: the whole weight for no reasoning at all.

- A solve never pays it, nor does an episode outside the group baseline or one with no assistant turn.
- It reads the total, not a per-turn mean, so an extra short turn never lowers the score.
- It is the one term that pays for more reasoning, and it can reward padding in failing episodes: watch
  first-submission pass against reasoning.

### Failed groups

A training group in which no baseline episode earned the objective (a positive `reward/objective`; a veto zeroes it)
is charged neither term.

- Shaping and judge terms can still separate such a group, so it can train, but the length terms would be most of what tells
  its attempts apart: the advantage would train reasoning length, not the task, wherever the task is out of reach.
- A group with a success is charged in full, its failures included. Eval episodes are always charged.
- `outcome/reasoning_terms_waived_frac` logs the waived share of training episodes.
- `reasoning_terms_in_failed_groups: true` charges them there too.

### Construction checks

Trainer construction reads the levels the environment can draw (all three under `random`) and refuses:

- a profile `reasoning_budget` or `reasoning_price` while the environment's `reasoning_effort` is unset;
- `reasoning_price_cap` at another value, or `reasoning_terms_in_failed_groups` with the floor off, while no drawn
  level prices reasoning;
- `reasoning_floor` while the environment draws no level, or a drawn level sets no `reasoning_budget` above 0;
- a drawn level's `reasoning_budget` that neither its price nor the floor reads;
- `reasoning_budget` or `reasoning_price` among `rollout_chat_template_kwargs`, where the template could state it;
- under a reward that prices no success above zero (no environment term, or one of weight ≤ 0), the floor, and the
  price while failed groups are waived: every episode would read as failed.

### Sizing

Keep `reasoning_price_cap` and `reasoning_floor` each below what the environment charges for the decisions it prices.
An episode pays one of them at most, so a failed one pays at most the larger. The code-contests recipes hold both
under one more failed submission, itself under running out
([Shipped recipes](environments/tasks/code-contests/README.md#shipped-recipes)).
