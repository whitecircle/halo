# Objective and Stability

The loss is TRL's GRPO objective ([Online GRPO](../online-grpo.md#grpo-objective-for-verifiable-rewards)) at TRL's default `loss_type: dapo`, clipped by `epsilon` (`0.2`; no recipe sets `epsilon_high`, which falls back to `epsilon`); the clip is inert at `num_iterations: 1`. Rollouts differ from the trainer's policy by the engine↔trainer numerics gap, and by up to one sync interval under prefetch or `sync_weights_every_n_steps > 1`, so an importance ratio corrects them, and the trust region is masks on that ratio, not a KL term.

## Importance sampling correction

The trainer recomputes the sampled tokens' log-probs and multiplies the per-token loss by `clamp(exp(logπ_recompute − logπ_sampling), max=vllm_importance_sampling_clip_max)` (TRL default `3.0`). On by default, it needs the log-probs [training on sampled tokens](rollouts.md#training-on-sampled-tokens) captures; without them it is disabled with a warning.

Consequences:

- Truncation is token-level, from above only: TRL's `token_truncate` at an unset lower clip. `vllm_importance_sampling_mode` passes at TRL's default or `token_truncate` and is refused at any other value at trainer construction; `vllm_importance_sampling_clip_min` is never read, and refused when set.
- `loss_type: vespo` is refused at trainer construction, in every config. VESPO weighs each turn row by the sum of its per-token log ratios in place of multiplying each token by its own, so whenever a mask stage drops a token or the engine forces a reasoning close (below), that token's ratio of 0 takes the whole row's weight to ~1e-15 with no error. TRL's own VESPO mode check never runs on this trainer. Use a per-token loss: `dapo`, `dr_grpo`, `cispo` or `grpo`.
- Set `rollout_top_p: 1.0` (default `0.95`) and leave `rollout_top_k` and `rollout_min_p` off under the geometric band or OPSM (`isr_opsm_delta`): a cut shifts every uncertain position, which the band reads as drift. A server that renormalizes log-probs over the cut raises at startup.

Watch `sampling/logratio_mean` first: the unclamped mean log-ratio in nats, near 0 when healthy. A growing negative drift is the broken-weight-sync signature ([Weight synchronization](setup.md#weight-synchronization)); a KL-free run drifting widens it too, ahead of a climbing `entropy` and with `advantage/net_token_mass` staying negative, which `balance_token_mass` and the [early stop](monitoring.md#early-stop) address.

A reasoning close the engine forced at the thinking budget gets ratio 0 like any masked token, keeping the DAPO normalizer: it was not the policy's choice, and trained with the episode's advantage it moves the model's own probability of ending its reasoning. The close is `rollout_reasoning_end_token` encoded as vLLM encodes its parser's end string (one token for Qwen3.x `</think>` and Gemma 4 `<channel|>`, the five-token final-channel opener `<|start|>assistant<|channel|>final<|message|>` for gpt-oss), and a forced one is a run of those ids every one at sampling log-prob exactly 0. `sampling/forced_close_frac` is their share of the completion tokens that carry sampling log-probs. The ratio reaches the loss only through this correction, so a run whose vLLM thinking budget can bind (a level's `thinking_tokens` or `rollout_max_thinking_tokens`) and whose tokenizer resolves the marker refuses to start without it. A marker the tokenizer does not write (an encoding with none of its added tokens, as `</think>` on Gemma 4 or gpt-oss) keeps the forced closes in the loss, with a warning.

## Trust region masks

The `isr_*` stages mask rather than reweight: a masked token loses its policy-gradient term but keeps its KL anchor. All default off, all raise without the IS correction, and they compose. Each stage pools a trajectory's turn rows.

| Knobs | Stage | Metric |
|---|---|---|
| `isr_geo_band_min` / `isr_geo_band_max` | Mask a trajectory whose `exp(mean log-ratio)` leaves it. | `sampling/is_geo_band_masked_frac` |
| `isr_veto_min` | Mask a trajectory if any corrected token's raw ratio drops below it. | `sampling/is_veto_masked_frac` |
| `isr_opsm_delta` | Mask negative-advantage trajectories drifting past N nats. | `sampling/is_opsm_masked_frac` |

The geometric band needs both bounds, `0 < min < 1 < max`. The code-contests recipes run it at `[0.95, 1.05]` with `isr_veto_min: 1.0e-4` and `isr_opsm_delta: 0.05`.

**Size the geometric band above the numerical floor.** Trainer and engine are different bf16 stacks: their mean log-ratio is negative at identical weights, steeper the flatter the distribution — ~0.002 nats/token at sampling entropy 0.4, ~0.04 at 1.0. A band inside it masks every step; sustained masking at low entropy is a disagreement to fix, not a band to widen.

`skip_update_masked_frac` (`(0, 1]`, off by default) is the breaker and needs at least one stage. It trips when `sampling/is_masked_traj_frac` or `sampling/is_masked_token_frac` passes it — the trajectory count alone reads a gutted step as healthy. A tripped round zeroes its advantages and drops its gradients to `None` (`sampling/update_skipped`).

`isr_engine_reference: true` moves the trainer↔engine floor out of the stages: after every sync the engine re-scores each train row carrying sampling log-probs, and the stages read that log-ratio. It needs `rollout_temperature` and `rollout_top_p` of `1.0`, `rollout_top_k`, `rollout_min_p` and `rollout_repetition_penalty` off, and one prefill per row, so serve with headroom ([Throughput](../../../infrastructure/rollout-servers.md#throughput)). At startup the trainer scores a 1,100-token counting probe whole and its prefixes alone at CUDA-graph sizes (8–448 tokens), and refuses a server whose shared positions disagree by more than 2 nats or whose whole probe averages over 4 nats per token: unpatched vLLM 0.26.0 under MTP returns garbage prompt log-probs for every graph-run prefill ([vLLM server patches](../../../infrastructure/rollout-servers.md#vllm-server-patches)). Off in the recipes.

TRL's `off_policy_mask_threshold` is refused: it reads a batch key this trainer never emits, so it would threshold a KL of 0.

## Advantages

A trajectory's advantage is its total reward minus the mean over its group's valid members. `scale_rewards` picks the divisor:

| Value | Effect |
|---|---|
| `batch` | Global batch std. The recipes' choice: degenerate groups stay near zero, scale holds across steps. |
| `none` | Dr.GRPO, unscaled. Unbiased, but gradient magnitude tracks the batch's raw reward spread. |
| `group` | Per-group std. TRL's default; on a sparse reward a near-degenerate std → 0 amplifies noise. |

`scale_rewards_std_floor` (default `0`, `0.2` in the code-contests recipes) divides by `max(std, floor)`, so a degenerate batch cannot inflate its noise into full-scale advantages. Set it below the healthy batch std; under `scale_rewards: none` there is no std to floor, and a floor above `0` is refused.

A non-finite reward or advantage fails the step on every rank: under `batch` scaling a single one makes the shared std, and so every advantage of the step, non-finite.

`drop_degenerate_groups` defaults **on** here (off for online GRPO) and judges a group on the reward each environment settled: its grade, its shaping and every external score, without the trainer's [reasoning floor and price](rollouts.md#reasoning-length-reward). Both are length regularizers: judged with them, a group tied on everything else would train on its members' reasoning lengths alone. A group whose members all settled the same reward has no task contrast to learn from, yet its tokens still inflate the DAPO normalizer (`sampling/degenerate_group_frac`). `mask_truncated_completions` is enforced here, not in TRL's generation path; the recipes leave it off.

`balance_token_mass` (default off, on in the code-contests recipes) cancels each generation round's net push on the tokens the policy sampled. Under the token-sum loss a trajectory pulls with its advantage times its trained tokens, each weighted by its truncated IS ratio. A group's advantages sum to zero but their token-weighted sum does not: where failures run longer than solves the round lowers the probability of the sampled tokens and entropy climbs, and where solves run longer it sharpens the policy. Without a KL anchor the sign of that share sets the direction entropy drifts, and a fixed scale on the negatives only moves the drift to the other side. The balance sums the positive and the negative mass of the trainable rows over every rank and shrinks the heavier sign by their ratio, never up, so every row keeps its sign and its order within the sign. The rows of [untrainable turns](#untrainable-turns), which train on a negative advantage only, stay out of both sums and keep their raw advantage: the trainable rows net to zero, and the round's net push is the untrainable rows' own mass, lowering only the tokens they sampled. The scale falls continuously to 0 as the lighter side empties, so a round whose trainable rows carry mass on one sign only (their positives all IS-masked, say) trains nothing on their advantages, and balanced they pull at `1 - |net|` of their unbalanced strength, half at `net = -0.5`. `advantage/net_token_mass` is the trainable rows' share before balancing, `(P - N) / (P + N)`, logged with the knob off too, and `advantage/token_mass_scale` the applied factor; a mass that is not finite raises on every rank, knob on or off. `advantage/negative_only_mass` is the untrainable rows' share of the trained mass, after the balance when it is on, where it is the round's whole net push. The completions record's `advantage` is the scaled value a trajectory's trainable rows carry. The balance needs `loss_type` `cispo`, `dapo` or `dr_grpo`, where every loss token of a step shares one normalizer, and `top_entropy_quantile: 1.0`, and raises otherwise: under `grpo` a completion pulls with its advantage alone, whatever its length, and the entropy mask drops tokens after the balance has weighed them. Online GRPO takes the same knob, and refuses it beside TRL's `off_policy_mask_threshold`, which drops negative sequences inside the loss.

## Untrainable turns

A turn the engine cut off, one the model ended on nothing, and one whose every call named a nonexistent tool, was refused unrun or ran and showed nothing ([Rollouts](rollouts.md#training-on-sampled-tokens)) train only on a negative advantage (strictly below 0), as per-turn rows of their sampled ids; on the single re-tokenized row they train on nothing. Rewarded, the runaway reasoning, the empty stop or the invented call would be reinforced whenever the episode recovers; left out entirely, a failing episode's signal lands on its other turns alone and the over-long reasoning behind a cut grows unchecked. The row's `tool_mask` is cleared before the DAPO normalizer is taken, in train and eval alike, so a dropped row counts in neither the loss nor the normalizer; a forced reasoning close inside a kept row still gets ratio 0. `balance_token_mass` leaves a kept row unscaled and out of its sums ([Advantages](#advantages)). `sampling/untrainable_rows_frac` is the share of rows tagged, `sampling/untrainable_rows_trained_frac` the share of those that reached the loss.

## KL and template protection

`beta > 0` adds TRL's k3 KL term, whose estimator is unbounded where the policy suppresses a token the reference likes. An unconditional clamp (`clamp_ref_logps`) caps the log-ratio at 5 nats, bounding per-token KL at `exp(5) ≈ 148`. Watch `kl_clamp_frac`, the clamped share of the loss tokens the KL term averages over (padding, tool output and dropped rows excluded): persistently non-zero means the policy sits far from the reference on real tokens.

`top_entropy_quantile < 1.0` trains only the highest-entropy tokens. Structural tokens — role markers, tool delimiters, BOS/EOS — are the lowest-entropy ones, so that alone starves the template until tool calls stop parsing. `ProtectedTokenEntropyMixin` unions the tokenizer's special and added tokens back in. A mask whose shape does not match the completion ids stashed by the last log-prob forward **raises** rather than silently dropping the protection. The recipes leave `top_entropy_quantile: 1.0` and lean on a nonzero `tool_error_penalty` (`0.05`) on the reward side.

## Routing replay

`routing_replay` pins the update pass's top-k expert selection to a recorded mask, so the gradient forward trains through the distribution the IS ratios were computed on.

- `recompute` (R2) captures the mask in the trainer's no-grad log-prob pass, so it needs a config that runs that pass: the IS correction, a nonzero `beta`, `num_iterations > 1`, or misaligned gradient accumulation.
- `rollout` (R3) replays the engine's own selection, and needs `train_on_sampled_tokens` plus a capture-capable server.

Both need MoE EP wrappers; Gemma 4 and Zaya are rejected at construction. The mask costs 2 bytes per token per MoE layer per top-k slot; `routing/replay_flip_rate` reports the share of selections the live top-k would have flipped. Under `rollout`, the step fails on every rank when a trainable batch carries no mask, or when no routed row on some rank matches an engine coverage convention (`routing/rollout_unresolved_frac` is the world share, so it can read below 1 then). The gpt-oss ep1 recipes ship `rollout`.

## Batch construction

TRL's `RepeatSampler` delivers each prompt `num_generations` consecutive times, rank-local; the trainer rolls out one trajectory per row. Advantages normalize within each group, so `per_device_train_batch_size × steps_per_generation` must divide by `num_generations` — stricter than TRL's global rule.

![Batch construction at the stage-1 code-contests shape: the sampler gives each of a rank's 3 prompts 8 consecutive rows, one rank's round is 24 rows (per_device_train_batch_size 1 × steps_per_generation 24), and six data-parallel ranks make one optimizer step of 144 rows = 18 prompts × 8; a row is one episode, a group is one prompt's rows and never straddles ranks](../../../assets/diagrams/batch_prompt_expansion.png)

A rollout with no learning signal — a raised episode, an `episode_timeout` cancellation, an invalid one — enters as a zero-masked row, out of its group's baseline (`sampling/invalid_episode_frac`). A step where no episode survived warns once, then **halts the run on the second consecutive one**.

Rows carry two masks. `completion_mask` is attention-valid — every real completion token, tool results and generation-prompt headers included, since they conditioned the sampling — while `tool_mask` is the loss mask, `1` only on assistant spans. The loss and the DAPO normalizer use their intersection. The batch is never packed.

`per_device_train_batch_size` counts rows, that is completions; unique prompts per step are the rollout total ÷ `num_generations` ([online-GRPO count](../online-grpo.md#data-flow-and-batch-construction)). The sizing knobs:

| Knob | Sets | Sizing rule |
|---|---|---|
| `num_generations` | GRPO group size | 8 in most recipes, 12 in the curriculum's stage-2 and stage-3 recipes, 4 in the template and the exam-QA recipe. Trades unique prompts for samples each. |
| `max_concurrent_rollouts` | Per-rank cap on rollouts in flight | Defaults to 4× this rank's actor share; above the round's count it never binds. |
| `num_rollout_workers` | Ray actor pool per rank (default `64`) | The environment's blocking per-episode cost ([Pool sizing](../../../infrastructure/ray.md#pool-sizing)). |

## Learning rate

Async GRPO refines an already tuned policy, so the rate sits near the SFT floor: `2e-7` on the code-contests full fine-tunes, `3e-6` on their LoRA siblings, `1e-6` on the lighter single-answer tasks, `5e-6` on the template. All but the curriculum's stage-2 and stage-3 recipes (`constant_with_warmup`) use a cosine schedule over the run's useful length, not the dataset's ([SFT](../../sft.md#learning-rate-and-global-batch-size)).
