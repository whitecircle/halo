# Memory and Throughput

Async GRPO is rollout-bound at the step level and logits-bound inside the training forward.

## The logits wall

TRL's GRPO computes per-token log-probs from a full `[B, T, vocab]` logits tensor. At gpt-oss's ~201k vocab, a 24k-token trajectory is ~9 GiB of bf16 logits, roughly doubled by the log-softmax saved for the backward.

Neither EP nor TP shrinks the logits: EP and MoE TP leave `lm_head` whole, and dense TP gathers its full-vocab output on every rank. Two levers help:

- [Chunked log-probs](#chunked-log-probs).
- `per_device_train_batch_size: 1`. Both forwards chunk to this many rows, so peak follows the longest single row.

With chunked log-probs off, every GRPO trainer weighs the logits plane against free device memory at construction: `rows × (completion width + 1) × vocab × 4 B`.

- **Rows**: `per_device_train_batch_size`, or `per_device_eval_batch_size` when evaluation runs and it is larger.
- **Width**: `rollout_max_tokens`, or `max_train_row_tokens` under `train_on_sampled_tokens: false` (the context window when unset). Online and offline GRPO read `max_completion_length` ([`use_chunked_grpo_logprobs`](../../reference/configuration-reference.md#offlinegrpoconfig)).
- **Verdict**: a plane that cannot fit raises on every rank; one over half the free memory warns.

The estimate is a floor: it covers one forward chunk, a single-row fallback trajectory is wider than a turn, and a model left on the host (accelerate's own FSDP) is weighed against an empty device.

## Chunked log-probs

`use_chunked_grpo_logprobs: true` (default off) removes the wall: log-probs come from the backbone's `last_hidden_state` through a chunked (sequence × vocab) matmul (`halo/kernels/head_logprobs.py`, driven by `halo/trainers/grpo/logprob_scoring.py`), so peak follows the tile size. Every code-contests recipe sets it; online and offline GRPO take the same flag.

- **Precision.** Logits and log-probs stay fp32 (the backward takes bf16 operands, fp32 accumulation), within 5e-3 nats of an fp64 reference. A caller's autocast is off inside the sweep, so fp32 activations (QLoRA's) take fp32 head GEMMs in both directions, on the CUDA cores. The full path rounds logits to bf16, and so do both engines by default: on one hidden state that is 0.014 to 0.035 nats mean absolute by family (0.006 to 0.016 on tokens above p 0.1), with signed means of at most 0.005, and up to 0.08 to 0.20 nats at the largest logits ([fp32 heads on the engines](../../infrastructure/rollout-servers.md#throughput)).
- **Cost.** A recompute backward at about the full path's cost. Measured on 1.1.0, whose tiles ran eagerly: Qwen3-8B head at 8k tokens, 51 ms forward+backward on a B300; offline GRPO on 8× B300, Qwen3-8B at 8,192 tokens 18,367 tok/s/GPU against 19,266 on full logits, gpt-oss-20b EP8 at 4,096 tokens 9,042 against 8,980. `tests/gpu/profiling/benchmark_head_logprobs.py` times the head alone at the Qwen3-8B, gpt-oss-20b and Gemma 4 26B-A4B shapes, a cold compile included.
- **Support.** FSDP2, ep1/EP and attention-only TP. A multimodal batch takes the full path. A PEFT adapter on `lm_head` raises at the first chunked forward.
- **Compiled tiles.** On CUDA everything a tile does after its GEMM runs as one `torch.compile`d graph per direction: the head transform, the online softmax, the target pick and the entropy forward; the softmax gradient, already in the GEMM dtype, backward. The compile is `fullgraph` with dynamic shapes, in-process on one compile thread, under inductor's `deterministic` mode, so ranks that see the same tiles (TP and ETP replicas) and reruns get the same kernels. A compile failure raises. `TORCHDYNAMO_DISABLE=1` runs the same tile functions eagerly, at a few 256 MiB tiles more peak memory and more memory traffic, and turns off every other `torch.compile` in the process with them, FlexAttention's included.
- **Graphs.** The first chunked step compiles one graph per direction for the head path (bias, logit scale and softcap each present or not), and one more for the entropy-carrying forward. New row counts, vocabularies, temperatures and head scalars reuse them, as do a tile one row or one column wide (run eagerly), targets sliced out of a wider batch, an expanded upstream gradient, a caller's autocast, and a bias of any dtype, trained or frozen. Each of these compiles one more: a second head path or activation dtype in the process, a caller's `inference_mode`, toggling `torch.use_deterministic_algorithms`, and a first tile as tall as it is wide (a vocabulary of at most 4,096). Dynamo allows 8 graphs per tile function; under `fullgraph` the 9th raises.

The sweep applies the family's head transform (Gemma's softcap, Cohere's `logit_scale`, Granite's `logits_scaling`, Inkling's μP division and vocabulary cut), verified against the family's own forward at construction (`halo/models/head_transform.py`). A mismatch is refused; a new family declares its transform ([Adding a Model](../../models/adding-a-model.md#declare-the-head-transform)).

Under FA4, at `per_device_train_batch_size: 1`, or on an offline GRPO [row-position family](../grpo/offline.md#row-positions), rows are trimmed to their real span before the backbone forward, one forward per row, and one head sweep then covers every row's real completion tokens.

One forward per row means one backward recomputes each checkpointed decoder layer once per row. FSDP2's hooks catch only one of those recomputes, so the rest are reduced after the backward, by torch's final callback for a two-layer ZeRO-2 unit and by the trainer for a one-layer group: one extra reduce-scatter per decoder-layer shard group per microbatch, the same size as the one FSDP2 already issues but exposed after the backward, or one per window under `fsdp_defer_grad_sync`. Until the backward returns, every layer it has recomputed twice stays gathered with an unsharded gradient, ZeRO-3 included ([Layers recomputed more than once per backward](../../parallelism/data-parallelism.md#layers-recomputed-more-than-once-per-backward)). With one EP group the experts' divide waits for the optimizer step for the same reason ([Gradient synchronization](../../parallelism/expert-parallelism.md#gradient-synchronization)).

## Context budget

Keep the trained trajectory under the served context. Reasoning survives only on the last turn:

`prompt + last-turn(reasoning + answer) + Σ other-turns(answer + tool output)`

The per-turn reasoning budget counts once, not `max_turns` times (twice under `carry_reasoning`). `prompt + max_turns × rollout_max_tokens` is a loose bound that over-restricts `max_turns`; `rollout_max_episode_tokens` bounds the generated share directly ([Trajectory length](episodes.md#trajectory-length)).

- Cap the longest training row with [`max_train_row_tokens`](episodes.md#training-row-cap) (unset by default; `64000` in the Qwen3.6 curriculum stages).
- Bound tool output with `max_observation_chars` (default `16384`, in `environment_kwargs`), which every tool-use environment applies ([Common knobs](environments/README.md#common-knobs)).
- The usual memory savers are on by default: gradient checkpointing ([reentrant on every MoE](../../optimization/flash-attention.md#usage)) and AdamWBF16 at 6 B/param, which a bf16 run gets from the default `optim` ([BF16 optimizer](../../optimization/bf16-optimizer.md#usage)).

## Throughput levers

Each step is generate → train → NCCL sync, serial on one server unless it prefetches, so idle trainer GPUs during generation are expected there. In priority order:

1. **Prefetch**, the only way to overlap generation with training. It is on by default with two or more servers; a single server needs `enable_prefetch: true`. At the default `prefetch_depth: 1`, **step time = sync + max(round, update)**, at one step of IS-corrected staleness, and the servers drain at each round's tail while its longest episodes finish. `prefetch_depth: 2` collects the next round beside it to fill that tail, at the cost of one more step of staleness and of engine KV ([Prefetch](setup.md#multiple-servers-and-prefetch)).
2. **Saturate the servers.** The optimizer step's row count sets the rollouts in flight; `max_concurrent_rollouts` only throttles them. Grow the batch with `gradient_accumulation_steps` (no per-forward memory cost), then raise the cap to match. `Waiting: > 0` or timeouts means you overshot.
3. **Speed generation.** Server-side levers are on [Rollout Servers](../../infrastructure/rollout-servers.md#throughput). Trainer-side, keep `rollout_max_tokens` and `rollout_max_thinking_tokens` no larger than the task needs; `sync_weights_every_n_steps > 1` amortizes the sync.
4. **Cut `max_turns`.** Turns are sequential, so halving them roughly halves rollout time. If `episode/turns` is pinned at the cap, raise it; well below, lower it.

`nvidia-smi dmon`, `EfficiencyCallback` or `async/prefetch_hit_rate` shows which phase bounds the step ([Prefetch](setup.md#multiple-servers-and-prefetch)).

## Length-ordered micro-batches

Under `train_on_sampled_tokens` a round holds one row per assistant turn, and at `per_device_train_batch_size: 1` each row is its own backbone call: in the loss pass, its checkpoint recompute and the no-grad recompute. Under ZeRO-3 (`fsdp_reshard_after_forward: true`) or an EP dispatch the ranks forward their rows in lock-step, so a row costs the longest row its peers forward with it. `length_ordered_microbatches` ranks every rank's rows longest first and deals them round-robin over the round's `steps_per_generation` micro-batches, in place of TRL's per-rank shuffle: the `i`-th longest rows of all ranks run together, and each micro-batch takes one row per length stratum. The no-grad forwards run in the same order and hand their scores back in batch order. A rank holding more row tokens in total than its peers still sets the pace.

Only the row order changes. Each row is forwarded on its own, so its log-probs, IS ratio and masks are bitwise the shuffled run's wherever a row's forward does not read its peers' (pure DP, FSDP2, TP; an EP dispatch keeps its own run-to-run tolerance) and the head sweep scores one row at a time (`per_device_train_batch_size: 1`); under FA4 at `per_device_train_batch_size > 1` one head sweep covers a micro-batch's rows together, so a different order matches to kernel rounding. The loss and gradients match the shuffled run's to accumulation order. Unset, the deal is on wherever that holds and off otherwise:

- accumulation that straddles a generation round, where the order picks which micro-batches an optimizer step takes;
- `loss_type: bnpo`, `top_entropy_quantile < 1` and the entropy bonus, which read each micro-batch's own tokens;
- a padded batch (`use_chunked_grpo_logprobs: false`, or `per_device_train_batch_size > 1` without FA4), where every row costs the batch's width whatever its order.

`true` raises at construction on those; `false` keeps TRL's shuffle.

On a six-GPU async GRPO test row (1.2.0) the deal takes the update phase from 480.5 to 328.3 s (1.46×), the no-grad pass from 170.7 to 114.0 s (1.50×) and peak memory from 210.25 to 183.35 GiB, the recompute bitwise.

## Packed micro-batches

`packed_microbatch_tokens` replaces one micro-batch per row by token-budget bins. After the round's rewards, advantages and masks exist, each rank plans its real rows into bins of at most that many attended tokens (first-fit-decreasing; a row over the budget keeps its own bin; more bins than `steps_per_generation` are relaxed to that many balanced ones). The world agrees the largest bin count `K` and the widest bin, every rank re-partitions into exactly `K` bins, and the window's last `K` slots train one bin each. The leading `steps_per_generation - K` slots are no-ops that run no forward and no collective, so the window still ends on a real backward and FSDP2's window toggles, DDP's `no_sync` and the optimizer step see the window they see without the knob. The round's tensors stay as built in a round store; TRL's buffer carries only the slot index tensors, and each slot's `[rows_per_bin, width]` view is gathered from the store just before its loss (filler rows are inert: one attended prompt token, a masked completion, ratio 1).

- **Objective.** Row-mean losses (`grpo`, `sapo`, `dr_grpo`, `luspo`) are rescaled by `rows_per_bin × steps_per_generation ÷ round_rows`, so a window sums to the per-row run's loss and gradient; `steps_per_generation`, not `gradient_accumulation_steps`, is the factor, as each of `num_iterations` cycles carries the objective once. `dapo` and `cispo` divide by a round-global token count and need none. `bnpo` divides each micro-batch by its own token count and is refused.
- **No-grad passes.** The recompute and the KL reference run one scoring call per bin on every rank, an empty bin an inert row that routes naturally and is left out of routing capture and replay. Scores come back in batch order; rows no bin holds (the dummy rows that pad a round across ranks) score 0. A bin forwards only its real rows, not the filler rows the loss forward adds to reach `rows_per_bin`: a filler row is two attended tokens, so the loss forward's bin holds at most `2 × (rows_per_bin − 1)` more tokens than the recompute's, a small share of a bin budget in the thousands, and the grouped GEMM sees correspondingly different per-expert counts. That moves accumulation order only (bf16-level, the same class as the varlen kernel's), not the objective; forwarding the filler too would add its tokens to both no-grad passes for no change in the objective. Its size on a real run is not measured.
- **Packed forward.** A bin reaches the backbone as one flattened `[1, N]` varlen call (one DeepEP capacity scope): each row's positions restart at 0, so it sees the RoPE positions the per-row call gives it, and the family's segment markers (`seq_idx`, `cu_seq_lens`) keep the conv, delta-rule and Mamba state per row, as SFT packing emits them. The head sweep then scores each row's real completion tokens. A one-row call carries no markers and runs as the per-row path. Evaluation rounds keep the per-row path, which is why the evaluation batch size must be 1. The result is exact in objective and not bitwise: a varlen kernel and the grouped GEMM's tile choice change accumulation order.
- **Attention and kernels.** The script requests the multi-document-rows default under the knob (FA2 for GatedDeltaNet decoders on Blackwell and under `full_determinism`, the loader's pick otherwise) and keeps the loader's pick for live gpt-oss sinks, which only FA4 carries. A two-document packed forward and backward runs through the scorer at construction on every rank, so the varlen attention and the delta-rule and conv variants load there and not between a dispatch and its combine; the gradients it leaves are dropped and the expert loads it counted are reset.
- **M-RoPE decoders.** Qwen3.5/3.6 (text or composite) pack like any other family: each row's positions restart at 0 and the markers keep the GatedDeltaNet state per row, held by the packed-row gate `tests/gpu/kernels/test_packed_rl_forward_mrope.py`, which scores gradients per parameter group against the per-row run's own bf16 noise (distance from fp32, reordered documents, one backward per document); the per-head scalar group (GatedDeltaNet `A_log` and `dt_bias`, token sums of cancelling terms that sit 10% from fp32 per row against 1-2% for every other group) is judged on its packed/per-row norm ratio, inside 0.8-1.25, its direction only logged.
- **Budget.** It is checked at construction against the DeepEP wire index and, on a group spanning NVLink domains, the proxy-GIN ceiling. A rank whose round tokens exceed `steps_per_generation` bins of that ceiling raises on every rank.
- **Metrics.** TRL's per-micro-step metrics become means over bins, so their logged values shift without any gradient change.

## Weight from the loss forward

On an aligned one-step round every micro-batch's loss forward runs on the weights the round was built on, so the loss forward's own detached log-probs are the log-probs the no-grad pass would return. With `loss_forward_importance_weights` the pass does not run: the build hands the loss the sampling log-probs, the tokens the ratio corrects and the forced closes it zeroes, and the scoring call of each loss forward builds `clamp(exp(logπ − logπ_sampling), max=vllm_importance_sampling_clip_max)` from `logπ.detach()` (a weight with a graph would add `A·logπ·∂w/∂θ` to the gradient) and clamps the reference log-probs of a `beta` run against them. The loss and its gradient equal the pass's for every admitted loss type, per-row or [packed](#packed-micro-batches); they differ only by the numerics between the pass's forward and the loss forward (the entropy-free head graph, no-grad kernel selection, EP's run-to-run dispatch tolerance), a relative change of about the forward's log-prob noise per token.

The knob defaults to `false`. `true` raises at construction naming the first consumer that needs the round's log-probs before its first backward:

- `balance_token_mass`, which needs the round's global importance-weighted token mass;
- the `isr_*` stages, `isr_engine_reference` and `skip_update_masked_frac`, which pool a trajectory's log-ratio over rows that run in different micro-batches or decide before the first optimizer step;
- `routing_replay: recompute`, which pins the update to the pass's routing (`rollout` fuses: the loss forward replays the engine's masks);
- `num_iterations > 1` and accumulation that straddles a generation round, where later micro-batches run on moved weights and the PPO ratio needs the round-start log-probs.

With `early_stop_logratio_gap` set, the logged log-ratio reads loss tokens only (below), a different estimator than a threshold set on the pass's reading. Eval rounds keep the pass. With neither the IS correction nor `beta` there is no pass to remove. `sampling/is_weights_from_loss_forward` reports the path of each round.

The loss forward must score exactly once with entropy, and TRL must read the weight and the clamped reference after that call; a forward that bypasses the hook, or a TRL that moves its reads, raises at the micro-batch naming `loss_forward_importance_weights: false` as the way back. The metrics the pass fed (`logratio_mean`, `is_ratio_*`, `is_ess_frac`, `logps/advantage_cov`, the token masses, `kl_clamp_frac`) accumulate on the device over the round's micro-batches and fold in one collective at its last one, before the optimizer step. They read loss tokens only, where the pass read every corrected token, rows the round drops (degenerate groups, invalid or truncated episodes) included. The zero-gradient tracker counts a token as trained unless a forced close zeroed it; a ratio underflowing to 0 (a log-ratio below about −87) is the only difference.

## Host work per step

The loss and no-grad loops run once per row at `per_device_train_batch_size: 1`, so a host read or collective inside them recurs 32 or more times per optimizer step, each draining the launch queue the next op was queued behind. A sync only costs the overlap between the host launching ahead and the device still running, so the gain is bounded by the smaller of the loop's launch time and its device time; on EP lanes DeepEP's own host reads inside each layer already drain the stream. [`log_phase_times`](monitoring.md#phase-times) shows where a step goes.

- **Row spans.** The dense path trims each row to its attended span, which takes one device-to-host copy of the mask. The no-grad pass reads the spans of its whole batch once, not once per row chunk, and under routing replay the replay layout and the scorer share that read. The round's build also reads them once and attaches them as `row_spans`, a host table riding TRL's shuffle, split and the length-ordered deal with the other row-aligned keys, so no loss forward copies a mask. Every loss forward recomputes its micro-batch's spans on the device and tallies disagreements without a host read; the stash is always on where rows forward densely and no TP or ETP group exists (no knob), and the window's last micro-step reads the tally (one all-reduce and one host read per optimizer step) and raises on every rank if any row read spans built from another mask, naming the stage to rebuild after (`_with_row_spans`). Under TP and ETP each micro-batch is re-broadcast from the group leader, so the table is not attached and each forward reads its own spans (logged once at startup).
- **Image consensus.** A composite's scoring forwards agree across ranks whether any batch carries images, since mixed full and chunked paths would deadlock FSDP2. The environmental trainer's batches are text-only, so its scoring skips that all-reduce and host read (a modality input reaching it raises on that rank alone, a code-invariant bug trap); online GRPO and SDPG keep one per scoring forward.

TRL's loss forward reduces six metrics across the ranks per micro-step (entropy, three clip ratios and two clip extremes), each ending in a host read; Halo leaves that block as TRL has it.

## Sizing a run

For a new model, environment or GPU, size in order:

- **Rollouts in flight per server** = `world_size × min(max_concurrent_rollouts, per_device_train_batch_size × steps_per_generation) ÷ num_servers` per round, peaking at a round's start, with up to `prefetch_depth` rounds in flight at once; an eval round opens with `world_size × eval_rollout_batch_size ÷ num_servers` ([eval rounds](evaluation.md#during-training)). TP and ETP siblings roll out too, though only each group leader's rollouts are kept ([Ray — Pool sizing](../../infrastructure/ray.md#pool-sizing)).
- **Decode speed at that concurrency.** Engine step latency ≈ `a + b × n` (one CPU thread). Measure two concurrencies against a live server with the run's own prompts ([Throughput](../../infrastructure/rollout-servers.md#throughput)).
- **Timeouts from the per-turn cap**, the turn's total ([Enforced and soft budgets](reasoning.md#enforced-and-soft-budgets)): `rollout_max_tokens`, or the engine's reasoning cap plus `rollout_max_answer_tokens` where that is smaller.
    - `request_timeout ≥ 2 × cap ÷ speed(n_peak)` (default `120`).
    - `episode_timeout ≥ min(max_turns × cap, rollout_max_episode_tokens) ÷ speed(n_peak) + tool time` (default `1200`); over the NCCL watchdog it raises at startup.
- **KV capacity.** bf16 KV per token = `2 × full-attention layers × kv_heads × head_dim × 2 B`, 20 KB on Qwen3.6-35B-A3B. Check `n_peak × mean context × that` against the pool the server reports at startup.
- **Step time.** The round is the batch's slowest episode. The update is `per-rank episodes × per-episode train time`, read off a short run, and hides inside the round while shorter.
- **Serving against training.** A trainer GPU absorbs about `3600 ÷ per-episode train time` episodes per hour. Keep the two within 2×: roughly one serving GPU per trainer GPU, each running one TP=1 engine while the model and its KV fit.

Worked shape, the 4-rank code-contests recipe (Qwen3.6-35B-A3B, 2 servers, `gradient_accumulation_steps: 24`, `num_generations: 8`):

| Quantity | Value |
|---|---|
| Episodes per step | 96 (12 prompts), 48 per server; an eval round (`eval_rollout_batch_size: 50`) opens with 100 per server |
| Per-turn cap | The 30,000-token `rollout_max_tokens`. The levels' `thinking_tokens` (8,192–16,384) are soft; the engine closes a normal turn's reasoning only at 21,808, leaving the 8,192-token answer room |
| Timeouts | `request_timeout: 1200`, `episode_timeout: 2700` |
| KV | 48 × 13k × 20 KB ≈ 12 GB at a 13k mean context; 48 × 30k × 20 KB ≈ 29 GB once every request holds a whole turn; about twice both in an eval round |
