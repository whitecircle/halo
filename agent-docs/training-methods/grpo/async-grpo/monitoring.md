# Metrics and Troubleshooting

Async GRPO metrics are namespaced by the question they answer. Rollout means are over the gathered episode population, and the `sampling/*`, `reward/*` and `routing/rollout_*` fractions and means fold every rank's counts in one collective per step, so neither is one rank's shard. The `async/prefetch_*` counters and `routing/replay_flip_rate` are the main process's own.

## Logged metrics

| Namespace | Answers | Keys to watch |
|---|---|---|
| `async/*` | rollout throughput | `mean_rollout_latency`; job totals `total_rollouts`, `cumulative_mean_rollout_latency`, `total_generation_tokens`; `prefetch_hit_rate`, `prefetch_hits` / `prefetch_misses`; `requests_expired_in_sync` |
| `episode/*` | agent behavior | `turns`, `generation_tokens` (+`_max`, `_p90`), `natural_termination_rate`, `truncation_rate`, `truncation_alarm`, `error_rate`, `length_cutoff_turns`, `empty_turns`, `thinking_budget_exhausted`, `reasoning_cjk_rate`, `tool_calls`, `sandbox_infra_fault`, `sandbox_agent_fault`, `reward_scored` |
| `outcome/*` | task success | `solve_rate`, `all_pass_group_frac` / `all_fail_group_frac`, eval `success@k`, plus per-env keys such as `test_pass_frac` |
| `reward/*` | reward and decomposition | bare `reward` / `reward_std`; `within_group_std`; the components `turn_shaping`, `tool_shaping`, `objective`, the environment's own terms (`submission`, …) and each external term's `<name>`; `composition_residue`; the trainer's `effort_length_penalty`, `effort_length_floor` |
| `judge/*`, `reward_model/*` | external reward terms | `judge/<name>/<requirement>`, `judge/<name>/completion_tokens`, `judge/<name>/cost_usd`; `reward_model/<name>/logit` |
| `logps/*` | policy confidence | `sampling_mean`, `advantage_cov`; the related `kl`, `entropy`, `kl_clamp_frac` are unnamespaced |
| `sampling/*` | engine/trainer agreement | `is_correction_active`, `logratio_mean`, `is_ratio_mean` / `is_ratio_max`, `is_ratio_extreme_frac`, `is_ess_frac`, `is_correction_coverage`, `is_masked_frac`, `sampler_certain_frac`, `update_skipped`, `degenerate_group_frac`, `invalid_episode_frac`, `rows_over_cap_frac` |
| `routing/*` | MoE routing replay | `replay_flip_rate`; under R3, the shape classes `rollout_full_frac`, `rollout_engine_omits_last_frac`, `rollout_completion_only_frac`, `rollout_unresolved_frac` (the one that says replay is degrading; a rank whose routed rows are all unresolved fails the step), plus `rollout_prompt_len_mismatch_frac`, which sits outside their denominator |
| unnamespaced | TRL loss internals | `clip_ratio/*`, `cispo_clip_ratio`, `step_time`, `num_tokens` |

`async/prefetch_*` appears only with multiple servers; `prefetch_input_skips` only once non-zero — a wedged prefetch worker dropped those prompts and they never trained. `async/requests_expired_in_sync` is a count per rollout round across the world, not a mean: the request deadlines that ran out on a request in flight across a weight sync even with the pause credited. Each threw away a turn the engine had already started (retried, or ending its episode on the last try), so nonzero means `request_timeout` is too tight for the backlog a sync leaves. `episode/thinking_budget_exhausted` appears only under `rollout_thinking_budget_scope: episode`: the fraction of episodes whose budget ran down to the per-turn reserve ([Reasoning budget](rollouts.md#reasoning-budget)). Task-specific `outcome/*` and `reward/*` components come from the environment's `rollout_metrics` hook.

Under `sampling/*`, `is_correction_coverage` is the share of loss tokens the engine's sampling log-probs cover, `is_masked_frac` the share of those a mask stage zeroed, `is_ratio_mean` / `is_ratio_max` the surviving ratios, and `sampler_certain_frac` the policy tokens the engine emitted with probability 1, which carry no correction. `is_ratio_extreme_frac` is the share of corrected tokens whose raw trainer-vs-sampler ratio lies outside `[1 / vllm_importance_sampling_clip_max, vllm_importance_sampling_clip_max]`, counted like `logratio_mean` over every corrected token before any mask or drop. `is_ess_frac` is the normalized effective sample size `(Σw)² / (n Σw²)` of the IS weights the loss applies over the corrected loss tokens, masked ones counting as zero: 1 when every weight is equal, falling as a few tokens carry the correction.

Three more diagnostics read the step without changing it. `outcome/all_pass_group_frac` and `outcome/all_fail_group_frac` split the groups by the environment's solve verdict (the one behind `solve_rate`; absent where the environment reports none): a group every valid member solved carries no objective signal, one none solved only partial credit where the objective pays it (the share of tests passed), and the two say whether the task is too easy or too hard; groups whose rewards are all equal are `sampling/degenerate_group_frac`. `logps/advantage_cov` is the covariance, over the loss tokens, of each token's log-prob under the pre-update policy and its advantage; a positive value drives entropy down and a negative one up (Cui et al., 2025), so a persistently positive value with a falling `entropy` is collapse under way. `episode/truncation_alarm` is 1 in a round whose `episode/truncation_rate` is over [`truncation_alarm_rate`](../../../reference/configuration-reference.md#asynctrainingconfig), which also warns when the rate first crosses it.

Every episode's categorical facts slice the metrics: its resolved effort level under `effort/<level>/*`, and any string an environment stamps under `trajectory.info["slices"]` as `<slice>/<value>/*`. Code-contests stamps the submission language where the run offers a choice of them, giving `language/cpp/*`.

Each slice carries `count`, `reward`, `generation_tokens`, `reasoning_tokens`, `turns`, `truncation_rate` and `solve_rate`, plus the environment's own `episode/*` keys with that prefix dropped (`effort/high/test_calls`).

Step timings land under `profiling/Time taken: <Trainer>.<name>` (main process only). The trainer adds `rollout_acquire` (the generation wait, near zero on a prefetch hit), `build_training_tensors` and `weight_sync`.

## Reading the metrics

Four common misreadings:

- **`reward_std` is not the learning signal.** It is the sample std over the gathered **valid** episodes, dominated by between-problem difficulty spread. `reward/within_group_std`, the mean of each group's reward std, is what the advantage uses: near zero means degenerate groups and no gradient ([Advantages](objective.md#advantages)).
- **`episode/natural_termination_rate` is not a solve rate.** That is `outcome/solve_rate`. It is the fraction that reached `done` without truncation, and says nothing about turns the engine cut at their token cap — those are `episode/length_cutoff_turns` ([Reasoning budget](rollouts.md#reasoning-budget)).
- **Two episode-health signals a termination rate hides.** `episode/empty_turns` counts the turns the model ended with neither a tool call nor visible content — a stop inside its reasoning, below the cap — which recover like cuts ([Reasoning budget](rollouts.md#reasoning-budget)). `episode/reasoning_cjk_rate` is the fraction whose reasoning drifted into CJK script.
- **The decomposition separates objective from shaping.** `reward/objective` is the environment's grade priced by its term; `reward/turn_shaping` and `reward/tool_shaping` are the per-turn and episode-level shaping, and the environment's own terms and each `judge` / `reward_model` term log by name ([Reward Terms](../rewards.md#environment-arm)). In every environment the components sum exactly to the episode reward; `reward/composition_residue` is the mean absolute gap, and nonzero means a channel bypasses them. A rising `reward` means task progress only when the objective component moves. `episode/reward_scored` below 1 means a judge or reward-model call failed: that term scores 0 and the episode leaves the baseline as invalid.

`reward` and `reward_std` cover the **valid** episodes only — the rows that train; `sampling/invalid_episode_frac` reports the rest ([Batch construction](objective.md#batch-construction)).

## Evaluation

Eval metrics carry the `eval_` prefix. The eval group size is `num_generations_eval` (TRL's `GRPOConfig`, `None` → `num_generations`); set it to `1` for a fast pass@1 monitoring eval. Above 1, and where the environment reports a solve verdict, `eval_outcome/success@<k>` is the share of prompts any of their `k` valid samples solved; `eval_outcome/solve_rate` stays the per-sample mean.

Eval rounds run without prefetch and a round waits for its slowest episode, so `eval_rollout_batch_size` widens the round — rows per rank, not prompts. It must be a multiple of `num_generations_eval`, at most `max_concurrent_rollouts`, and needs `dataloader_drop_last: false` — all three raise at startup. Make it a divisor of the per-rank share too, or the round pads duplicate rows ([`eval_rollout_batch_size`](../../../reference/configuration-reference.md#asynctrainingconfig)).

Left unset, one round is one eval batch, and the same check then requires `per_device_eval_batch_size` to hold whole groups. The loss forward chunks by it either way.

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `server unreachable at …`, or a stall in `init_communicator()` | The server is not serving, or never joins the NCCL group: an unroutable advertised master, or trainer and server on one GPU | Verify `/health`; set `VLLM_GROUP_HOST` / `SGLANG_GROUP_HOST` to a routable NIC; give them different `CUDA_VISIBLE_DEVICES` ([Rollout Servers](../../../infrastructure/rollout-servers.md#troubleshooting)) |
| `Could not bind the weight-transfer group port` | A live process still holds it; `SO_REUSEADDR` rules out `TIME_WAIT` | Give each server and trainer its own `group_port` / `vllm_group_port`; remove the stale container |
| `GENERATION is wedged (no response … to a 1-token probe)` | `probe_generation` found a scheduler stuck after a trainer died mid-sync | Restart the server container after every hard trainer crash |
| `vLLM server … is PAUSED for a weight update` | A trainer died without lifting its `/pause` | `POST /resume`; restart the server instead if that sync had already streamed part of the model |
| `Prefetch auto-disabled: 1 rollout server configured.` | Expected: a sync pauses the only server | Configure more servers ([Multiple servers and prefetch](setup.md#multiple-servers-and-prefetch)) |
| `DeadlineExpired: request exceeded its …s deadline` | Server overloaded or wedged; a weight-sync pause is already excluded. Actors retry with backoff (`request_timeout` 120 s, `max_retries` 3, `retry_base_wait` 1 s) | Check server health first; 4xx are terminal except 408 and 429 |
| Episodes booked into `episode/error_rate` near `episode_timeout` | The whole episode — generation, tools, grading — passed its deadline (default 1200 s) | Raise `DIST_NCCL_TIMEOUT_MINUTES` first, then `episode_timeout` ([bounds](performance.md#sizing-a-run)) |
| `404 … "The model <path> does not exist"` on every rollout | `model_name` is unset by default and filled with `model_name_or_path`, so a checkpoint path is sent as the id | Set `model_name` to the id at `/v1/models`, or serve with `--served-model-name` |
| `500 "Already borrowed"` | A vLLM fast-tokenizer race that scales with concurrency | Harmless unless `gave up after N tries` appears; then raise `max_retries` or `retry_base_wait` |
| `ray.exceptions.RayActorError` | The episode returns as an errored, masked-out row and the actor restarts (`max_restarts=-1`) | Check `ray logs` and per-node memory ([Ray Cluster](../../../infrastructure/ray.md#monitoring)) |
| `SweEnvironment has nothing to grade against` at environment construction | `requires_answer: false` with no `test_function` while the reward prices the environment's grade | Grade against an `answer` column, pass a `test_function`, or price the episode with external terms only ([SWE → Reward](../environments/swe-environment.md#reward)) |
| `the local sandbox waits on its programs through pidfd_open, which this environment refuses` at environment construction | The `local` / `bubblewrap` sandbox waits on every run through a pidfd, which the kernel (before 5.3) or the container's seccomp profile refuses | Run on Linux 5.3+ with `pidfd_open` allowed, or set `sandbox_backend: remote` ([Sandboxes](../environments/sandbox.md#choosing-a-backend)) |
