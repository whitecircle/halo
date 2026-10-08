# Rollout Configuration

Rollout knobs sit at the YAML top level on `AsyncTrainingConfig`
([fields](../../../reference/configuration-reference.md#asynctrainingconfig)); `reasoning_effort` and
`carry_reasoning` are `environment_kwargs`, and `max_turns` is a top-level `EnvironmentConfig` field.
Sampling is `rollout_temperature` (default `0.7`) and `rollout_top_p` (`0.95`), with
`rollout_top_k`, `rollout_min_p` and `rollout_repetition_penalty` off by default. Rollout and eval
requests send all five, so the model's `generation_config.json` defaults never apply
([Rollout Servers](../../../infrastructure/rollout-servers.md)). TRL's own sampling fields reach no
sampler ([Rollout backend](setup.md#rollout-backend)).

## Trajectory length

Three knobs bound what an episode generates. `rollout_max_tokens` (default `32768`) caps **one
turn**, reasoning and answer together. `max_turns` caps the turns; an episode that burns them ends
truncated. Its `null` default keeps the environment class's own — 15 on `code_contests` /
`codeforces`, 20 on `swe`, 8 on `exam_qa`, 10 everywhere else.
`rollout_max_episode_tokens` (default `null`) caps the **episode**: the most tokens its assistant
turns may sample together, reasoning and visible output alike, recovery turns included. Unset, an
episode may generate `max_turns × rollout_max_tokens`.

The episode budget is enforced by the engine per turn and never stated to the model. A turn's
`max_tokens` is the smaller of its own cap and what the episode has left, and its reasoning cap
([Reasoning budget](#reasoning-budget)) shrinks by the same amount, so the turn keeps its answer
room: `rollout_max_tokens` less the turn's reasoning cap, the whole turn without one. An episode
left with less than that room starts no further turn and ends truncated, priced like a `max_turns`
overflow; a cut or empty turn on the last turn the budget affords is closed the same way, never nudged into a
retry that could not run. `episode/output_budget_exhausted` is the fraction of episodes whose budget ran below it.
The budget must be at least `rollout_max_tokens`, so one whole turn fits; it is `max_tokens` on the
wire, so it holds on both engines. Every shipped code-contests recipe sets `81920`.

The trajectory accumulates across turns within the model context window and is **never
truncated**: a row over that window is recorded per rank, then raised on every rank together. The
startup check warns when `max_prompt_length` + the environment's preamble + what an episode may
generate (the smaller of `max_turns × rollout_max_tokens` and `rollout_max_episode_tokens`) exceeds the
served window ([Servers and Launch](setup.md#rollout-backend)); set the episode budget so that worst
case fits.

`max_prompt_length` (default `null`) is a dataset filter — rows whose rendered prompt exceeds it are
dropped. `max_completion_length` is no knob here: the script pins it to `rollout_max_tokens` and
raises on any other explicit value, and TRL reads it only as the `dr_grpo` normalizer.

`max_train_row_tokens` (default `null`: no cap beyond the context check above) is a memory bound on
a training row, and must exceed `rollout_max_tokens` (a row is prompt plus completion). A per-turn
row over it leaves the batch while the episode's other turns train; a whole-trajectory row trains at
zero weight. Setting it also turns on `sampling/rows_over_cap_frac`, the share of rows left out.
`rollout_max_episode_tokens` bounds a row's generated share (a per-turn row's prompt carries the
earlier turns' output) but not the tool output between turns, so it lowers how many rows reach this
cap without replacing it.

## Tool calls

Native-tool environments send the env's `tools` schema, so the server needs the matching
`--tool-call-parser`. A mismatched parser fails silently: calls come back as text with no
`tool_calls`, so every turn scores as a give-up. A missing one is a 400 on vLLM; SGLang returns the
call as text. Per-family values:
[Rollout Servers](../../../infrastructure/rollout-servers.md#vllm).
[ReAct](../environments/react.md) envs send no `tools` and need no parser.

`rollout_stop_tokens` (default empty) lists special-token strings the trainer resolves to ids and
sends as `stop_token_ids`. Set it to a tool-call terminator the model's `eos_token_id` omits: gpt-oss
with harmony disabled ends a call with `<|call|>` and otherwise hallucinates the result, playing the
episode out in one turn (the recipes set `["<|call|>"]`).

Every name must resolve: an unknown one raises at trainer construction, and the eval scripts resolve
`--training_config`'s list through the same function, so a recipe that trains also evaluates.

## Reasoning budget

A thinking budget is per turn. `rollout_max_thinking_tokens` (default `null`) caps a turn's
chain-of-thought (vLLM's `thinking_token_budget`), leaving the rest of `rollout_max_tokens` for the
answer. A value at or above `rollout_max_tokens` is refused, as is any value under
`rollout_backend: sglang`. The vLLM server needs a reasoning parser and Model Runner V2 off
([Rollout Servers](../../../infrastructure/rollout-servers.md#vllm)).

Steer depth with the env's `reasoning_effort`: `low`/`medium`/`high`/`random`/`null` (`null` on
`BaseEnvironment`, `medium` on `code_contests`). A `random` level is drawn **once per generation
group**, keeping the conditioning of a group identical. An eval round draws it from the problem's
text instead, so every checkpoint scores a problem at the same level.

The level reaches the model only through the chat template. gpt-oss's template renders it natively;
the stock Qwen3.x and Gemma 4 templates have no effort variable, so those recipes pin
`jinja-templates/qwen3/qwen3.6-reasoning-effort.jinja` and `jinja-templates/gemma4/gemma4-reasoning-effort.jinja`,
which state the level and its per-turn budget in the system block ("Think for at most N tokens per
turn") from the `reasoning_effort` and `reasoning_budget` variables below. A level the model cannot
see is not a policy it can learn.

`reasoning_effort_profiles` overrides the class's per-level table. A level's `thinking_tokens` caps
every turn's reasoning, clamped by `rollout_max_thinking_tokens` where that is set; an episode whose
level sets none runs under `rollout_max_thinking_tokens` alone, or uncapped. `rollout_max_tokens`
bounds the whole turn either way, so a level's budget must sit below it (refused at trainer
construction and at the start of an eval) and the difference is the turn's answer room. An episode output budget narrows both
caps together ([Trajectory length](#trajectory-length)); `reasoning_budget` stays the level's budget.

`thinking_tokens` is a **vLLM** request field (`thinking_token_budget`). On `rollout_backend:
sglang` a level's budget reaches no request field (warned once per process): nothing caps CoT
below `rollout_max_tokens`; the level still steers through the chat template (which states the
budget as a cut it cannot be on this engine), and the budget stays the reference of the
[reasoning floor](#reasoning-length-reward).

A turn the engine cuts at its token cap, or one the model ends with neither a tool call nor visible
content, is nudged and retried within `max_turns` and `max_length_cutoff_recoveries`
(`environment_kwargs`; `null` = every such turn within `max_turns`). A recovered turn lands in
`episode/length_cutoff_turns` or `episode/empty_turns` (a cut while the turn held an unfinished tool call
also in `episode/length_cutoff_in_call_turns`) and pays the protocol's `length_cutoff_penalty`
(default `0`); the turn that exhausts the cap, or lands on the last turn, ends the episode truncated,
priced like a `max_turns` overflow. Under carried reasoning a cut costs the policy only a turn and
the retry thinks on from where it stopped, so a per-turn budget binds only once the cut is priced.
The turn after an unproductive one — cut, empty, or every call unknown or refused unrun — gets a
quarter of its level's reasoning cap, not the whole budget again (`RECOVERY_THINKING_SHARE` in
`src/environments/episode.py`, clamping what the output budget leaves): room to read the nudge or the
refusal, fix and act, so a cut never buys a second budget; the template still states the level's
budget, and the nudge asks for the action. The episode
budget bounds what the recoveries may add in total.

`episode/thinking_cap_turns` counts the turns whose reasoning reached the cap they recorded: the
level's, or a retry's reserve, never the narrower request cap an output budget leaves a late turn. A
turn's reasoning is its sampled ids up to and including `rollout_reasoning_end_token`, all of them for
a turn cut before its close; on a turn vLLM force-closes at its recorded cap that is the cap, or one or
two past it where the template's or the model's own `<think>` sits, so such a turn always counts. The
metric needs a vLLM cap that can bind, `train_on_sampled_tokens`, and a marker that is one of the
tokenizer's added tokens (gpt-oss's five-token final-channel opener logs none); the standalone eval scripts log none.

An engine abort never reaches the environment: the actor re-issues the turn up to `max_retries` times
(default `3`) rather than charging a length cut; past that the episode errors into a masked row.

## Carried reasoning

`carry_reasoning: true` (`environment_kwargs`, default off) sends the most recent assistant turn's
reasoning back with the conversation, so the next turn continues that thought. Earlier turns stay
visible text, so a request grows by at most one reasoning budget.

The trainer refuses the knob under `rollout_backend: sglang`, where SGLang's handling of an assistant
`reasoning_content` is unverified. A step whose turns carry no reasoning while a consumer is on is
warned once — the server most likely runs without a reasoning parser.

Where a carried thought renders is the template's decision. `rollout_chat_template_kwargs`
(`{preserve_thinking: true}` on the stock Qwen3.x template; the shipped Qwen3.6 effort template always
renders carried reasoning) sends template variables with every request **and** applies
them to the trainer's own renders, so both sides see one template state. `reasoning_effort` and
`reasoning_budget` are refused there, being per episode.

## Reasoning length reward

Two trainer-side terms price an episode's reasoning tokens, summed over its assistant turns, on top of
the task reward and before advantages.

**The floor** (`reasoning_floor`, default `0` = off) pays for more reasoning. Its reference is three
quarters of the per-turn thinking budget the episode ran under (its level's `thinking_tokens`, clamped
by `rollout_max_thinking_tokens`). An episode whose reasoning tokens fall short of it pays
`-floor × shortfall / reference`: the whole weight for an episode that reasoned nothing. The
reference sits under one budget, so an episode of a single assistant turn can clear it without running
into the cap the engine enforces per turn. It reads the episode's total, not a per-turn mean, so a
terse repair turn after a verdict is not under-use and an extra tool turn never lowers the score.
An episode with no thinking budget or no assistant turn pays nothing. A run where no drawable level
sets a budget and `rollout_max_thinking_tokens` is unset is refused at trainer construction.

**The price** (`reasoning_price`, default `null` = off) maps each effort level to reward units per
1,000 reasoning tokens: an episode pays `-min(reasoning_price_cap, price[level] × tokens / 1000)`.
The map must name exactly `low`, `medium` and `high` whatever the environment draws, and the
environment must set `reasoning_effort` (both refused at trainer construction). Price the lowest level
highest, so the same trace costs most where little reasoning was asked. It prices reasoning tokens
only — code and tool calls are free. An episode with no level pays nothing.

A price is paid within the group, so the sibling that reasons less wins it whatever the outcome. That
is why `reasoning_price_cap` (default `0.1`; refused at another value while the price is off) caps it
per episode, and why it never runs alone: below its reference the floor must out-slope the price, so a
level short of it is still paid to reason more. The code-contests recipes run no price; their caps,
budgets and output budget are the dial.

Both terms reach the gradient only through groups the environment's reward separates:
`drop_degenerate_groups` judges a group without them ([Advantages](objective.md#advantages)).

Keep `reasoning_price_cap + reasoning_floor` below what the environment charges for the decisions it
prices: the code-contests recipes run a floor of `0.10` (the Qwen3.6 vLLM recipes) or `0.05`, under
the `0.2` resubmission price. Watch `reward/reasoning_price` and `reward/reasoning_floor`.

## Chat template

Two safe setups. **Built-in** (default): no `chat_template:` on the trainer and no `--chat-template`
on the server, so both load the checkpoint's. **Custom**:
`chat_template: <file>.jinja` plus `force_chat_template: true` on the trainer and
`--chat-template <same file>` on the server, the file visible inside its container.

`chat_template:` without `force_chat_template: true` is a no-op wherever the checkpoint ships a
template; with it, and no `--chat-template` on the server, the two sides diverge silently. The rule
governs the re-tokenization paths below: a differing template scores log-probs against a prompt the
policy never generated under. Per-turn rows take the engine's ids and cannot drift.

`reasoning_effort` goes out as the request's **top-level** field — the spelling both engines derive
their thinking toggle from and hand the template. SGLang lets a nested copy override that field (it
pops it into the top-level one before rendering), so on SGLang the same value is added nested too —
an exact copy, so the render reads one value whichever spelling the engine consults. The level's
per-turn thinking budget rides in the nested form as `reasoning_budget`, added per request; the
trainer's own renders carry the same variables, and `rollout_chat_template_kwargs` refuses both keys.

`jinja-templates/qwen3/qwen3.6-reasoning-effort.jinja` is the hub Qwen3.6 template cut to what the rollouts
use — text only, thinking always on, an assistant turn rendering whatever reasoning it carries, the
hub's tool-call and tool-response spellings — plus a system-block line stating the effort and budget.
`jinja-templates/gemma4/gemma4-reasoning-effort.jinja` is the same cut of the hub Gemma 4 template: the
`<|think|>` marker opens every system turn, so the server and the trainer's renders agree without the
`enable_thinking` variable the hub template keys on. Pin either with `chat_template:` +
`force_chat_template: true` and serve with the same file (`VLLM_CHAT_TEMPLATE` / `SGLANG_CHAT_TEMPLATE`
in the compose files).

## Training on sampled tokens

`train_on_sampled_tokens` (default on) trains on the ids the model actually sampled. Each assistant
turn becomes its own row: `prompt` is the engine's rendered prompt ids for that request, `completion`
that turn's sampled ids; rows share the trajectory's advantage.

vLLM returns the ids under `--return-tokens-as-token-ids`; SGLang requests them per call and needs no
flag. The importance-sampling correction additionally needs vLLM's
`--logprobs-mode processed_logprobs` ([Objective](objective.md#importance-sampling-correction)).

Turns the rollout marked untrainable — engine-cut (`truncated`), ended with neither a tool call nor
visible content (`empty`), or every tool call naming a nonexistent tool (`calls_rejected`) — become
rows too, tagged: a tagged row stays in the loss only when its trajectory's advantage is negative
([Objective](objective.md#untrainable-turns)). Such a turn stays in the next turn's prompt. An episode
whose turns are all untrainable trains only when its advantage is negative; one that yields no row at all yields
one fully masked row.

The negative-only rows need the turn's sampled ids. An untrainable turn without them, and every
untrainable turn on the single-row path below, trains on nothing: the re-render closes a cut turn with
template tokens the engine never sampled. `sampling/untrainable_turns_rowless_frac` is the share of
untrainable turns that became no row (no ids, zero tokens, a rejected prompt re-render, over
`max_train_row_tokens`, or a trajectory trained as the single re-tokenized row).

A trajectory where any other turn lost its completion ids drops whole to the single re-tokenized
row, warned once with the engine's remedy — all-or-nothing, never a partial capture.

A turn that kept its completion ids but lost its prompt ids re-renders only that prompt through the
serving template, silently; a prefix the template rejects invalidates the episode, not the batch, and on an
untrainable turn drops only that turn's row.

The single-row path renders the trajectory once and locates each assistant span inside that render. A
boundary it cannot pin **invalidates the episode** — a fully masked row, outside its group's baseline
— rather than training a guessed span.

`train_on_sampled_tokens: false` forces that path for every trajectory and disables the
importance-sampling correction, which needs the sampling log-probs: batches then train uncorrected,
with a warning, and a run whose vLLM thinking budget can bind and whose `rollout_reasoning_end_token`
resolves is refused at construction, since its forced reasoning closes are neutralized only through the ratio
([Objective](objective.md#importance-sampling-correction)).

## Saving trajectories

`save_completions` (default on) writes each log step's rollouts to
`<output_dir>/completions/completions_<step>.parquet` — columns `step`, `prompt`, `completion`,
`environment_reward`, `advantage`, `reward_components` and `reward_details` — plus a `completions`
table when wandb is in `report_to`. A file holds every row since the previous log (one round at
`logging_steps: 1`); eval logs keep the step number, take an `_eval` suffix and hold the whole eval set.

`reward_components` is the episode's settled `reward/*` components as JSON; `environment_reward` less
their sum is what the trainer's reasoning terms charged. `reward_details` holds each scored term's rationale by term
name — a veto judge's fired checks with the quotes behind them — so a vetoed episode is read off the
record without scoring it again. Both are `{}` for an episode that kept neither.

The `completion` column renders detokenized message text, unaffected by `train_on_sampled_tokens`
(raw ids feed the loss only). TRL's `log_completions` controls the console table alone, capped by
`num_completions_to_print`.

The writer rank comes from `fs_aware_save_rank`: global rank 0 on a shared output filesystem, one
writer per node on a per-node one, so a non-shared output does not lose every node but the first.
The console table prints on global rank 0 alone. A writer that fails (a full disk, a wandb outage)
fails the step on every rank.
