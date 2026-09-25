# Rollout Configuration

Rollout knobs sit at the YAML top level on `AsyncTrainingConfig`
([fields](../../../reference/configuration-reference.md#asynctrainingconfig)); `reasoning_effort` and
`carry_reasoning` are `environment_kwargs`, and `max_turns` is a top-level `EnvironmentConfig` field.
Sampling is `rollout_temperature` (default `0.7`) and `rollout_top_p` (`0.95`); TRL's own sampling
fields reach no sampler ([Rollout backend](setup.md#rollout-backend)).

## Trajectory length

`rollout_max_tokens` (default `32768`) caps **one turn** — it is the only generation budget.
`max_turns` caps the turns; an episode that burns them ends truncated. Its `null` default keeps the
environment class's own — 15 on `code_contests` / `codeforces`, 20 on `swe`, 8 on `exam_qa`, 10
everywhere else.

The trajectory accumulates across turns, bounded only by the model context window, and is **never
truncated**: a row over that window is recorded per rank, then raised on every rank together.

`max_prompt_length` (default `null`) is a dataset filter — rows whose rendered prompt exceeds it are
dropped. `max_completion_length` is no knob here: the script overwrites it with `rollout_max_tokens`,
and TRL reads it only as the `dr_grpo` normalizer.

`max_train_row_tokens` (default `null` = the model's context window) is a second, tighter bound on a training row, and
must exceed `rollout_max_tokens` (a row is prompt plus completion). A per-turn row over it leaves
the batch while the episode's other turns train; a whole-trajectory row trains at zero weight.
Setting it also turns on `sampling/rows_over_cap_frac`, the share of rows left out.

## Tool calls

Native-tool environments send the env's `tools` schema, so the server needs the matching
`--tool-call-parser`. A mismatched parser fails silently: calls come back as text with no
`tool_calls`, so every turn scores as a give-up. A missing one is a 400. Per-family values:
[Rollout Servers](../../../infrastructure/rollout-servers.md#vllm).
[ReAct](../environments/react.md) envs send no `tools` and need no parser.

`rollout_stop_tokens` (default empty) lists special-token strings the trainer resolves to ids and
sends as `stop_token_ids`. Set it to a tool-call terminator the model's `eos_token_id` omits: gpt-oss
with harmony disabled ends a call with `<|call|>` and otherwise hallucinates the result, playing the
episode out in one turn (the recipes set `["<|call|>"]`).

An unknown name is warned and skipped; a list where **none** resolves raises rather than running
unstopped.

## Reasoning budget

`rollout_max_thinking_tokens` (default `null` = unbounded) caps a turn's chain-of-thought (vLLM's
`thinking_token_budget`), leaving the rest of `rollout_max_tokens` for the answer. A value at or
above `rollout_max_tokens` is refused, as is any value under `rollout_backend: sglang`. The vLLM
server needs a reasoning parser and Model Runner V2 off
([Rollout Servers](../../../infrastructure/rollout-servers.md#vllm)).

Steer depth with the env's `reasoning_effort`: `low`/`medium`/`high`/`random`/`null` (`null` on
`BaseEnvironment`, `medium` on `code_contests`). A `random` level is drawn **once per generation
group**, keeping the conditioning of a group identical.

The level reaches the model only through the chat template. gpt-oss's template renders it natively;
the stock Qwen3.x and Gemma 4 templates have no effort variable, so those recipes pin
`jinja-templates/qwen3/qwen3.6-reasoning-effort.jinja` and `jinja-templates/gemma4/gemma4-reasoning-effort.jinja`,
which state the level and its budget in the system block from the `reasoning_effort` and
`reasoning_budget` variables below. A level the model cannot see is not a policy it can learn.

`reasoning_effort_profiles` overrides the class's per-level table. Under a set
`rollout_max_thinking_tokens`, a level's `thinking_tokens` is capped by it per turn, and the turn's
total drops to that per-turn cap plus the global answer headroom; left `null`, the level caps
reasoning alone and `rollout_max_tokens` still bounds the turn. A profile may also carry
`max_length_cutoff_recoveries`.

`thinking_tokens` is a **vLLM** request field (`thinking_token_budget`). On `rollout_backend:
sglang` a level's budget reaches no request field (warned once per process): nothing caps CoT
below `rollout_max_tokens`; the level still steers through the chat template, and the budget
stays the reference of the [effort length floor](#effort-length-reward).

A turn the engine cuts at its token cap, or one the model ends with neither a tool call nor visible
content, is nudged and retried within `max_turns` and the episode's `max_length_cutoff_recoveries`
(`environment_kwargs`; `null` = every such turn within `max_turns`). A recovered turn lands in
`episode/length_cutoff_turns` or `episode/empty_turns` and pays the protocol's `length_cutoff_penalty`
(default `0`); the turn that exhausts the cap, or lands on the last turn, ends the episode truncated,
priced like a `max_turns` overflow. Under carried reasoning a cut costs the policy only a turn and
the retry thinks on from where it stopped, so a per-turn budget binds only once the cut is priced.

**Budget scope.** `rollout_thinking_budget_scope` (default `turn`) says what a budget — a level's
`thinking_tokens`, else `rollout_max_thinking_tokens` — covers. Under `turn` every turn gets it whole,
so a cut or empty turn plus its recovery nudge buys another full budget: a cheap "continue thinking"
that lets an episode's reasoning grow to any per-turn cap. Under `episode` the budget is the episode's
total: a turn's engine cap is the budget minus the reasoning the earlier turns spent, never below
`rollout_thinking_turn_reserve` (default `512`, enough to close the reasoning and act) and never above
`rollout_max_thinking_tokens`, which under this scope is the ceiling one turn may take rather than a
clamp on the level's budget. The per-turn total (`rollout_max_tokens`, narrowed per level to the first
turn's cap plus the answer headroom) stays constant across the episode, and a recovery turn gets only
what is left. `episode/thinking_budget_exhausted` is the fraction of episodes whose budget ran down to
the reserve. Trainer construction and the eval scripts refuse a scope some episode could not run
under: with `rollout_max_thinking_tokens` unset, the environment must set `reasoning_effort` and
every level's `thinking_tokens`, and no level's budget may sit below the reserve.

A turn's spend is read off the engine's sampled ids as the ids up to and including
`rollout_reasoning_end_token` (default `</think>`, resolved through the tokenizer; the engine's budget
counts the close it forces, and a turn cut before closing its reasoning counts all of its ids), so the
scope is vLLM-only, requires `train_on_sampled_tokens`, and every request under it — the eval scripts'
too — asks for `return_token_ids`. The effort templates read the run-wide `reasoning_budget_scope`
variable and state the budget as a total across the task's turns; `reasoning_budget` stays the level's
budget, not the turn's narrowed cap. The floor term's reference scales with it
([Effort length reward](#effort-length-reward)).

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
them to the trainer's own renders, so both sides see one template state. `reasoning_effort` is
refused there.

## Effort length reward

Two trainer-side terms price an episode's reasoning tokens, summed over its assistant turns, by its
effort level. Both enter the task reward before advantages.

**The price** (`effort_length_penalty_k0`, default `None` = off) is
`-min(c_max, k(effort) × tokens / l_norm)` with `k(effort) = k0 × exp(-(effort - effort_min) / tau)`.
`effort_length_penalty_levels` gives each level its scalar (`low` 25, `medium` 50, `high` 100), so at
the default `tau` of 25 the same trace costs `low` about 20× what it costs `high`. It prices reasoning
tokens only — code and tool calls are free — and `c_max` caps it, so a long trace cannot outweigh the
task reward.

A price is paid within the group, so the sibling that reasons less wins it whatever the outcome. That
is why it is capped and near zero at the highest level, and why the recipes never run it alone:

**The floor** (`effort_length_floor_weight`, default `0` = off) is the one term that pays for more
reasoning. Its reference is `effort_length_floor_budgets` (default `0.75`) times the thinking budget
the episode ran under — a level's per-turn budget, or the episode's total under
`rollout_thinking_budget_scope: episode`. The floor moves with that budget, so the fraction is set
against the budgets a recipe runs: the episode-scope Qwen3.6 code-contests recipes pair `0.375` with
24,576–36,000-token budgets, a floor of 9,216–13,500 reasoning tokens. The default sits below 1 so
that an episode of a single assistant turn can clear its floor without running into the cap the engine
enforces per turn. An episode short of the floor pays `-weight × shortfall / floor`. It reads the episode's
total, not a per-turn mean, so a terse repair turn after a verdict is not under-use and an extra tool
turn never lowers the score. An episode with no thinking budget or no assistant turn pays nothing;
turns that carry no reasoning at all pay the whole weight.

Keep `c_max + floor weight` below what the environment charges for the decisions it prices (the
code-contests recipes: `0.1 + 0.05` under the `0.2` resubmission price). Smaller shaping terms, a
`0.05` tool error among them, can still be outweighed by a long trace at the lowest level. Watch
`reward/effort_length_penalty` and `reward/effort_length_floor`.

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
thinking budget rides in the nested form as `reasoning_budget`, added per request, and under the
episode scope the run-wide `reasoning_budget_scope: episode` rides beside it; the trainer's own renders
carry the same variables, and `rollout_chat_template_kwargs` refuses all three keys.

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

Turns the rollout marked unusable are left out: engine-cut (`truncated`), turns the model ended with
neither a tool call nor visible content (`empty`), and turns whose every tool call named a nonexistent
tool (`calls_rejected`). They stay in the next turn's prompt. An episode with every turn excluded
yields one fully masked row.

A trajectory where any trainable turn lost its completion ids drops whole to the single re-tokenized
row, warned once with the engine's remedy — all-or-nothing, never a partial capture.

A turn that kept its completion ids but lost its prompt ids re-renders only that prompt through the
serving template, silently; a prefix the template rejects invalidates the episode, not the batch.

The single-row path renders the trajectory once and locates each assistant span inside that render. A
boundary it cannot pin **invalidates the episode** — a fully masked row, outside its group's baseline
— rather than training a guessed span.

`train_on_sampled_tokens: false` forces that path for every trajectory and disables the
importance-sampling correction, which needs the sampling log-probs: every batch then trains
uncorrected on rollouts at least one weight sync stale, with a warning.

## Saving trajectories

`save_completions` (default on) writes each log step's rollouts to
`<output_dir>/completions/completions_<step>.parquet` — columns `step`, `prompt`, `completion`,
`environment_reward`, `advantage` — plus a `completions` table when wandb is in `report_to`. A file
holds every row since the previous log (one round at `logging_steps: 1`); eval logs keep the step
number, take an `_eval` suffix and hold the whole eval set.

The `completion` column renders detokenized message text, unaffected by `train_on_sampled_tokens`
(raw ids feed the loss only). TRL's `log_completions` controls the console table alone, capped by
`num_completions_to_print`.

The writer rank comes from `fs_aware_save_rank`: global rank 0 on a shared output filesystem, one
writer per node on a per-node one, so a non-shared output does not lose every node but the first.
