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

`reasoning_effort_profiles` overrides the class's per-level table. Under a set
`rollout_max_thinking_tokens`, a level's `thinking_tokens` applies as `min(level budget, that cap)`
and lowers the turn's total to that budget plus the global answer headroom; left `null`, the level
caps reasoning alone and `rollout_max_tokens` still bounds the turn. A profile may also carry
`max_length_cutoff_recoveries` and `token_cost`, reward units per 1k generated tokens.

`thinking_tokens` is a **vLLM** request field (`thinking_token_budget`). On `rollout_backend:
sglang` a level's budget reaches no request field (warned once per process): nothing caps CoT
below `rollout_max_tokens`, and the budget survives only as the target
`reasoning_compliance_weight` prices against.

A turn the engine cuts at its token cap is nudged and retried within `max_turns` and the episode's
`max_length_cutoff_recoveries` (`environment_kwargs`; `null` = every cut within `max_turns`). A
recovered cut lands in `episode/length_cutoff_turns` and carries no reward penalty; the cut that
exhausts the cap ends the episode truncated, priced like a `max_turns` overflow.

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
(`{preserve_thinking: true}` on Qwen3.x) sends template variables with every request **and** applies
them to the trainer's own renders, so both sides see one template state. `reasoning_effort` is
refused there.

## Reasoning compliance reward

`reasoning_compliance_weight` (default `0` = off) adds an asymmetric term matching CoT length to the
episode's applied budget `B`. Per assistant turn with `r` reasoning tokens: **0** inside
`[0.3·B, 0.9·B]`, a mild penalty below, a strong penalty above that reaches `-1` at `r ≥ B`. The term
is the mean over turns, scaled by the weight; an episode with no budget scores 0.

`reasoning_compliance_under_use_weight` (default `0.3`) scales the below-band side against the
over-use side's `1.0`; `0` prices over-use only, so a short repair turn pays nothing.

It enters the task reward before advantages, so keep it small (`0.15` in the code-contests
recipes); watch `reward/calibration`.

## Chat template

Two safe setups. **Built-in** (default): no `chat_template:` on the trainer and no `--chat-template`
on the server, so both load the checkpoint's. **Custom**:
`chat_template: <file>.jinja` plus `force_chat_template: true` on the trainer and
`--chat-template <same file>` on the server, the file visible inside its container.

`chat_template:` without `force_chat_template: true` is a no-op wherever the checkpoint ships a
template; with it, and no `--chat-template` on the server, the two sides diverge silently. The rule
governs the re-tokenization paths below: a differing template scores log-probs against a prompt the
policy never generated under. Per-turn rows take the engine's ids and cannot drift.

`reasoning_effort` goes out as the request's **top-level** field only — the spelling both engines
derive their thinking toggle from. The nested `chat_template_kwargs` form does not, and sending both
is ambiguous.

## Training on sampled tokens

`train_on_sampled_tokens` (default on) trains on the ids the model actually sampled. Each assistant
turn becomes its own row: `prompt` is the engine's rendered prompt ids for that request, `completion`
that turn's sampled ids; rows share the trajectory's advantage.

vLLM returns the ids under `--return-tokens-as-token-ids`; SGLang requests them per call and needs no
flag. The importance-sampling correction additionally needs vLLM's
`--logprobs-mode processed_logprobs` ([Objective](objective.md#importance-sampling-correction)).

Turns the rollout marked unusable are left out: engine-cut (`truncated`), and turns whose every tool
call named a nonexistent tool (`calls_rejected`). They stay in the next turn's prompt. An episode
with every turn excluded yields one fully masked row.

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
