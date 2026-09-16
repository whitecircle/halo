# Objective and Stability

The loss is TRL's GRPO objective ([Online GRPO](../online-grpo.md#grpo-objective-for-verifiable-rewards)) at TRL's default `loss_type: dapo`, clipped by `epsilon` (`0.2`) and `epsilon_high` (`0.28` in the [shipped recipes](setup.md#shipped-recipes), inert at `num_iterations: 1`). Rollouts are off-policy by at least one weight sync, so an importance ratio corrects them, and the trust region is masks on that ratio, not a KL term.

## Importance sampling correction

The trainer recomputes the sampled tokens' log-probs and multiplies the per-token loss by `clamp(exp(logπ_recompute − logπ_sampling), max=vllm_importance_sampling_clip_max)` (TRL default `3.0`). On by default, it needs the log-probs [training on sampled tokens](rollouts.md#training-on-sampled-tokens) captures; without them it is disabled with a warning.

Consequences:

- `vllm_importance_sampling_mode` and `vllm_importance_sampling_clip_min` are ignored, with a warning when set: truncation is token-level, from above only.
- Set `rollout_top_p: 1.0` under the geometric band (default `0.95`): a nucleus cut shifts every uncertain position, which the band reads as drift. A server that renormalizes log-probs over the nucleus raises at startup.

Watch `sampling/logratio_mean` first: the unclamped mean log-ratio in nats, near 0 when healthy. A growing negative drift is the broken-weight-sync signature ([Weight synchronization](setup.md#weight-synchronization)).

## Trust region masks

The `isr_*` stages mask rather than reweight: a masked token loses its policy-gradient term but keeps its KL anchor. All default off, all raise without the IS correction, and they compose. The three trajectory stages pool a trajectory's turn rows; the token band is per token.

| Knobs | Stage | Metric |
|---|---|---|
| `isr_band_min` / `isr_band_max` | Mask a token whose raw ratio leaves the band. Start `[0.5, 2]`. | `sampling/is_token_band_masked_frac` |
| `isr_geo_band_min` / `isr_geo_band_max` | Mask a trajectory whose `exp(mean log-ratio)` leaves it. | `sampling/is_geo_band_masked_frac` |
| `isr_veto_min` | Mask a trajectory if any corrected token's raw ratio drops below it. | `sampling/is_veto_masked_frac` |
| `isr_opsm_delta` | Mask negative-advantage trajectories drifting past N nats. | `sampling/is_opsm_masked_frac` |

Paired bands need both bounds, `0 < min < 1 < max`. The code-contests recipes run the geometric band at `[0.95, 1.05]` with `isr_veto_min: 1.0e-4` and `isr_opsm_delta: 0.05`, and leave the token band off.

**Size the geometric band above the numerical floor.** Trainer and engine are different bf16 stacks: their mean log-ratio is negative at identical weights, steeper the flatter the distribution — ~0.002 nats/token at sampling entropy 0.4, ~0.04 at 1.0. A band inside it masks every step; sustained masking at low entropy is a disagreement to fix, not a band to widen.

`skip_update_masked_frac` (`(0, 1]`, off by default) is the breaker and needs at least one stage. It trips when `sampling/is_masked_traj_frac` or `sampling/is_masked_token_frac` passes it — the trajectory count alone reads a gutted step as healthy. A tripped round zeroes its advantages and drops its gradients to `None` (`sampling/update_skipped`).

`isr_engine_reference: true` moves the trainer↔engine floor out of the stages: after every sync the engine re-scores each train row carrying sampling log-probs, and the stages read that log-ratio. It needs `rollout_temperature` and `rollout_top_p` of `1.0` and one prefill per row, so serve with headroom ([Throughput](../../../infrastructure/rollout-servers.md#throughput)). Off in the recipes.

TRL's `off_policy_mask_threshold` is refused: it reads a batch key this trainer never emits, so it would threshold a KL of 0.

## Advantages

`advantage_mode` sets the baseline and the negative-side treatment:

- `mean` (default) — plain group-mean baseline.
- `qae` — per-group `advantage_quantile` baseline (default `0.4`), so only rare successes train.
- `asymmetric` — mean baseline, then `advantage_pos_scale` / `advantage_neg_scale` (`1.0` / `0.4`).
- `neg_mask_hard` — zero negatives in groups where no member's objective reward reached `advantage_hard_group_threshold` (`0.5`).

The code-contests recipes run `asymmetric` with `advantage_neg_scale: 0.7`. `scale_rewards` picks the divisor:

| Value | Effect |
|---|---|
| `batch` | Global batch std. The recipes' choice: degenerate groups stay near zero, scale holds across steps. |
| `none` | Dr.GRPO, unscaled. Unbiased, but gradient magnitude tracks the batch's raw reward spread. |
| `group` | Per-group std. TRL's default; on a sparse reward a near-degenerate std → 0 amplifies noise. |

`scale_rewards_std_floor` (default `0`, `0.2` in the code-contests recipes) divides by `max(std, floor)`, so a degenerate batch cannot inflate its noise into full-scale advantages. Set it below the healthy batch std.

`drop_degenerate_groups` defaults **on** here (off for online GRPO): all-alike groups carry no gradient but still inflate the DAPO normalizer (`sampling/degenerate_group_frac`). `mask_truncated_completions` is enforced here, not in TRL's generation path; the recipes leave it off.

## KL and template protection

`beta > 0` adds TRL's k3 KL term, whose estimator is unbounded where the policy suppresses a token the reference likes. An unconditional clamp (`clamp_ref_logps`) caps the log-ratio at 5 nats, bounding per-token KL at `exp(5) ≈ 148`. Watch `kl_clamp_frac`: persistently non-zero means the policy sits far from the reference on real tokens.

`top_entropy_quantile < 1.0` trains only the highest-entropy tokens. Structural tokens — role markers, tool delimiters, BOS/EOS — are the lowest-entropy ones, so that alone starves the template until tool calls stop parsing. `ProtectedTokenEntropyMixin` unions the tokenizer's special and added tokens back in. A mask whose shape does not match the completion ids stashed by the last log-prob forward **raises** rather than silently dropping the protection. The recipes leave `top_entropy_quantile: 1.0` and lean on a nonzero `tool_error_penalty` (`0.05`) on the reward side.

## Routing replay

`routing_replay` pins the update pass's top-k expert selection to a recorded mask, so the gradient forward trains through the distribution the IS ratios were computed on.

- `recompute` (R2) captures the mask in the trainer's no-grad log-prob pass, so it needs a config that runs that pass: the IS correction, a nonzero `beta`, `num_iterations > 1`, or misaligned gradient accumulation.
- `rollout` (R3) replays the engine's own selection, and needs `train_on_sampled_tokens` plus a capture-capable server.

Both need MoE EP wrappers; Gemma 4 and Zaya are rejected at construction. The mask costs 2 bytes per token per MoE layer per top-k slot; `routing/replay_flip_rate` reports the share of selections the live top-k would have flipped. Under `rollout`, a batch with no mask fails on every rank; the gpt-oss ep1 recipes ship it.

## Batch construction

TRL's `RepeatSampler` delivers each prompt `num_generations` consecutive times, rank-local; the trainer rolls out one trajectory per row. Advantages normalize within each group, so `per_device_train_batch_size × steps_per_generation` must divide by `num_generations` — stricter than TRL's global rule.

![Batch construction at the stage-1 code-contests shape: the sampler gives each of a rank's 3 prompts 8 consecutive rows, one rank's round is 24 rows (per_device_train_batch_size 1 × steps_per_generation 24), and six data-parallel ranks make one optimizer step of 144 rows = 18 prompts × 8; a row is one episode, a group is one prompt's rows and never straddles ranks](../../../assets/diagrams/batch_prompt_expansion.png)

A rollout with no learning signal — a raised episode, an `episode_timeout` cancellation, an invalid one — enters as a zero-masked row, out of its group's baseline (`sampling/invalid_episode_frac`). A step where no episode survived warns once, then **halts the run on the second**.

Rows carry two masks. `completion_mask` is attention-valid — every real completion token, tool results and generation-prompt headers included, since they conditioned the sampling — while `tool_mask` is the loss mask, `1` only on assistant spans. The loss and the DAPO normalizer use their intersection. The batch is never packed.

`per_device_train_batch_size` counts rows, that is completions; unique prompts per step are the rollout total ÷ `num_generations` ([online-GRPO count](../online-grpo.md#data-flow-and-batch-construction)). The sizing knobs:

| Knob | Sets | Sizing rule |
|---|---|---|
| `num_generations` | GRPO group size | 8 in the code-contests recipes, 4 in the templates. Trades unique prompts for samples each. |
| `max_concurrent_rollouts` | Per-rank cap on rollouts in flight | Defaults to 4× this rank's actor share; above the round's count it never binds. |
| `num_rollout_workers` | Ray actor pool per rank (default `64`) | The environment's blocking per-episode cost ([Pool sizing](../../../infrastructure/ray.md#pool-sizing)). |

## Learning rate

Async GRPO refines an already tuned policy, so the rate sits near the SFT floor: `2e-7` on the code-contests full fine-tunes (`3e-7` on the Qwen3.6 curriculum stages), `3e-6` on their LoRA siblings, `1e-6` on the lighter single-answer tasks, `5e-6` on the two templates. All use a cosine schedule over the run's useful length, not the dataset's ([SFT](../../sft.md#learning-rate-and-global-batch-size)).
