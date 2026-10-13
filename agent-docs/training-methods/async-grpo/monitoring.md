# Monitoring

Async GRPO metrics are namespaced by the question they answer. The rollout, `sampling/*`, `reward/*` and `routing/*` metrics and the prefetch counters fold every rank's episodes or counts; timings are the main process's.

## Logged metrics

| Namespace | Answers | Keys to watch |
|---|---|---|
| `async/*` | rollout throughput | `mean_rollout_latency`; job totals `total_rollouts`, `cumulative_mean_rollout_latency`, `total_generation_tokens`; `prefetch_hit_rate`, `prefetch_hits` / `prefetch_misses` (multiple servers only); `requests_expired_in_sync` |
| `episode/*` | agent behavior | `turns`, `generation_tokens` (+`_max`, `_p90`), `natural_termination_rate`, `truncation_rate`, `truncation_alarm`, `error_rate`, `length_cutoff_turns`, `length_cutoff_in_call_turns`, `empty_turns`, `thinking_cap_turns`, `output_budget_exhausted`, `reasoning_cjk_rate`, `tool_calls`, `sandbox_infra_fault`, `sandbox_agent_fault`, `reward_scored` |
| `outcome/*` | task success | `solve_rate`, `all_pass_group_frac` / `all_fail_group_frac`, eval `success@k`, plus per-env keys such as `test_pass_frac` |
| `reward/*` | reward and decomposition | bare `reward` / `reward_std`; `within_group_std`; the components and `composition_residue` ([Reward decomposition](#reward-decomposition)) |
| `judge/*`, `reward_model/*` | external reward terms | `<source>/<name>/scored` plus per-source keys: [judge](../grpo/rewards/judge.md#metrics), [reward model](../grpo/rewards/reward-model.md) |
| `logps/*` | policy confidence | `sampling_mean`, `advantage_cov`; the related `kl`, `entropy`, `kl_clamp_frac` are unnamespaced |
| `advantage/*` | the round's push on the sampled tokens | `net_token_mass`, `token_mass_scale`, `negative_only_mass` ([Token-mass balance](objective.md#token-mass-balance)) |
| `sampling/*` | engine/trainer agreement | `is_correction_active`, `logratio_mean`, `is_ratio_mean` / `is_ratio_max`, `is_ratio_extreme_frac`, `is_ess_frac`, `is_correction_coverage`, `is_masked_frac`, `sampler_certain_frac`, `update_skipped`, `degenerate_group_frac`, `invalid_episode_frac`, `rows_over_cap_frac`, `forced_close_frac`, `untrainable_rows_frac`, `untrainable_rows_trained_frac`, `untrainable_turns_rowless_frac` |
| `routing/*` | MoE [routing replay](objective.md#routing-replay) | `replay_flip_rate`; under R3 ([rollout replay](objective.md#routing-replay)), the capture shape classes `rollout_full_frac`, `rollout_engine_omits_last_frac`, `rollout_completion_only_frac`, `rollout_unresolved_frac`, plus `rollout_prompt_len_mismatch_frac` |
| unnamespaced | TRL loss internals | `clip_ratio/*`, `cispo_clip_ratio` (under `loss_type: cispo`), `step_time` |

Environments add task-specific `outcome/*` and `reward/*` keys through their `rollout_metrics` hook.

### `async/*`

| Metric | Meaning |
|---|---|
| `prefetch_input_skips` | Logged once non-zero: prefetch submissions (each a round's prompts) skipped because a wedged worker's input queue stayed full; those prompts never train |
| `requests_expired_in_sync` | Per-round world count of request deadlines that expired across a weight sync (pause credited), each discarding a started turn. Nonzero: raise `request_timeout` |

### `episode/*`

| Metric | Meaning |
|---|---|
| `thinking_cap_turns` | Turns whose counted reasoning reached a vLLM cap ([Thinking-cap metric](reasoning.md#thinking-cap-metric)) |
| `output_budget_exhausted` | Under `rollout_max_episode_tokens` only: episodes whose output budget no longer held a turn's answer room ([Trajectory length](episodes.md#trajectory-length)) |
| `truncation_alarm` | 1 in a round whose `truncation_rate` exceeds [`truncation_alarm_rate`](../../reference/configuration-reference.md#asynctrainingconfig), which also warns each time the rate crosses it |

### Group and entropy diagnostics

| Metric | Meaning |
|---|---|
| `outcome/all_pass_group_frac` / `all_fail_group_frac` | Groups every / no valid member solved, by the environment's solve verdict (absent without one). High values: the task is too easy / too hard |
| `sampling/degenerate_group_frac` | Groups whose valid members' settled environment rewards are all equal, or with fewer than two valid members; logged in train rounds under `drop_degenerate_groups` |
| `logps/advantage_cov` | Covariance of each loss token's pre-update log-prob with its advantage. It pools contexts, so its sign need not match the `entropy` trend, which [`advantage/net_token_mass`](objective.md#token-mass-balance) predicts |

### `sampling/*`

| Metric | Meaning |
|---|---|
| `logratio_mean` | Unclamped mean log-ratio in nats, near 0 when healthy ([IS correction](objective.md#importance-sampling-correction)) |
| `is_correction_coverage` | Share of loss tokens that carry a correction: covered by the engine's sampling log-probs and not emitted with probability 1 |
| `is_masked_frac` | Share of those a mask stage zeroed |
| `is_ratio_extreme_frac` | Share of corrected tokens whose raw ratio lies outside `[1/c, c]` before any mask, `c` being `vllm_importance_sampling_clip_max` |
| `is_ess_frac` | Normalized effective sample size `(Σw)² / (n Σw²)` of the applied IS weights; it falls as a few tokens carry the correction |
| `is_ratio_mean` / `is_ratio_max` | The ratios that survive the mask stages |
| `sampler_certain_frac` | Tokens the engine emitted with probability 1, which carry no correction (a forced reasoning close instead gets ratio 0, `forced_close_frac`) |

### `routing/*`

| Metric | Meaning |
|---|---|
| `rollout_unresolved_frac` | The R3 shape class that says replay is degrading; a rank whose routed rows are all unresolved fails the step |
| `rollout_prompt_len_mismatch_frac` | Outside the shape classes' denominator |

### Slices

Each episode's resolved effort level slices the metrics as `effort/<level>/*`, and any string an environment stamps under `trajectory.info["slices"]` as `<slice>/<value>/*` (code contests, where the run offers a choice of languages: `language/cpp/*`). Each slice carries `count`, `reward`, `generation_tokens`, `reasoning_tokens`, `turns`, `truncation_rate` and `solve_rate`, plus the environment's own `episode/*` keys without that prefix (`effort/high/test_calls`).

### Step timings

`profiling/Time taken: <Trainer>.<name>` (main process) includes `rollout_acquire` (the generation wait, near zero on a prefetch hit), `build_training_tensors` and `weight_sync`.

## Reading the metrics

- **`reward_std` is not the learning signal.** Between-problem difficulty dominates it. The advantage uses `reward/within_group_std` (the mean per-group std): near zero means degenerate groups and no gradient ([Advantages](objective.md#advantages)).
- **`episode/natural_termination_rate` is not a solve rate.** It is the share of episodes that reached `done` untruncated; `outcome/solve_rate` comes from environments with a solve verdict (code contests).
- **Turn health is logged apart:** `episode/length_cutoff_turns` (turns cut at their token cap) and `episode/empty_turns` (neither a tool call nor visible content), both [unproductive turns](episodes.md#unproductive-turns), and `episode/reasoning_cjk_rate` (reasoning drifted into CJK script).
- **A rising `reward` is task progress only when `reward/objective` moves**; shaping and external terms move it too.
- **The headline `reward`, `reward_std` and `reward/within_group_std` cover valid episodes only**; `sampling/invalid_episode_frac` reports the rest in train rounds. The `reward/*` components, the reasoning charges and slice rewards also average invalid episodes ([Batch construction](objective.md#batch-construction)).

### Reward decomposition

The `reward/*` components separate objective from shaping ([Environment arm](../grpo/rewards/README.md#environment-arm)):

| Component | What it is |
|---|---|
| `reward/objective` | The environment's grade, priced by its term |
| `reward/turn_shaping`, `reward/tool_shaping` | Per-turn and episode-level shaping |
| `reward/<name>` | The environment's own terms (`reward/submission`, …) and each `judge` or `reward_model` term |
| `reward/reasoning_floor`, `reward/reasoning_price` | The trainer's charges, each while its term is on ([Reasoning length reward](reasoning.md#reasoning-length-reward)) |
| `reward/composition_residue` | Mean absolute gap between the components and the settled reward |

The environment's components (all but the trainer's charges) sum exactly to its settled reward, so a nonzero `composition_residue` means a channel bypasses them. The headline `reward` adds the trainer's reasoning charges.

### External terms

- `episode/reward_scored` below 1 means a judge or reward-model call reached no verdict; `<source>/<name>/scored` names the term. The term scores 0, and its `on_error` decides whether the episode stays in the baseline ([Outages](../grpo/rewards/README.md#outages)).
- `judge/<name>/veto` is the share of judged episodes a veto check fired on ([Veto judge](../grpo/rewards/judge.md#veto-judge)).

## Saved trajectories

`save_completions` (default on) writes the rollouts since the previous log to `<output_dir>/completions/completions_<step>.parquet`, plus a `completions` table when wandb is in `report_to`. Eval files take an `_eval` suffix and hold the whole eval set.

| Column | Holds |
|---|---|
| `completion` | Detokenized message text, whatever `train_on_sampled_tokens` |
| `environment_reward` | The episode's reward; less the `reward_components` sum, it is what the trainer's reasoning terms charged |
| `advantage` | The scaled value the trajectory's trainable rows carry |
| `reward_components` | The settled `reward/*` components, as JSON |
| `reward_details` | Each scored term's rationale by term name, such as a veto judge's fired checks and quotes |

Both JSON columns are `{}` when empty. TRL's `log_completions` controls only the console table, capped by `num_completions_to_print`.

### Cut tool calls

A turn the engine cut mid-tool-call shows the partial call after its text, tagged `cut by the engine before the turn closed; never run`. On vLLM, with the turn's sampled ids captured and a reasoning budget resolving the reasoning close to a single token, it carries the whole decoded text past the close; elsewhere (SGLang, which enforces no reasoning cap, included), what the tool parser salvaged. It never runs, and later turns never see it.

### Writer rank

`fs_aware_save_rank` picks the writer: global rank 0 on a shared output filesystem, one per node otherwise. A failed write (a full disk, a wandb outage) fails the step on every rank.

## Early stop

Off unless a condition is set. Each condition counts its consecutive breaching readings; when one reaches `early_stop_patience` (default 3), training ends on a verdict taken across ranks.

- **A reading** is one training log line (the window mean under `logging_steps` > 1). The gap and the [breaker](objective.md#breaker) (table below) come once per generation round, so a step reusing its round leaves their count unchanged.
- **The stop** skips that step's save and eval (an epoch save included), logs its reason, and exits non-zero on every rank. Resume from a checkpoint taken before the drift began.

| Knob | Stops when | Reads |
|---|---|---|
| `early_stop_entropy_band` | `entropy` stays outside `[low, high]` | TRL's `entropy` |
| `early_stop_logratio_gap` | the trainer-vs-sampler gap stays above it | the magnitude of `sampling/logratio_mean`; needs the IS correction |
| `early_stop_on_skipped_updates` | the breaker skipped every update | `sampling/update_skipped`; needs `skip_update_masked_frac` |

TRL's `entropy` counts a micro-batch whose loss tokens were all dropped (degenerate groups, invalid episodes, padding rows) as 0, so under drops it reads low and moves with the dropped share.

Read the band and the gap off a healthy run; both are model-specific. The gap shows a drift before entropy does.

- The Qwen3.6 vLLM code-contests recipes stop on a band of `[0.12, 0.40]` and a gap of `0.005`.
- Every code-contests recipe stops on skipped updates.
