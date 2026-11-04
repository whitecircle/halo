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

- **Precision.** Logits and log-probs stay fp32 (the backward takes bf16 operands, fp32 accumulation), within 5e-3 nats of an fp64 reference. A caller's autocast is off inside the sweep, so fp32 activations (QLoRA's) take fp32 head GEMMs in both directions, on the CUDA cores. The full path rounds logits to bf16, about 0.1 nats at |logit| ≈ 20.
- **Cost.** A recompute backward at about the full path's cost (Qwen3-8B head, 8k tokens: 51 ms forward+backward on a B300). Offline GRPO on 8× B300: Qwen3-8B at 8,192 tokens runs 18,367 tok/s/GPU against 19,266 on full logits; gpt-oss-20b EP8 at 4,096 tokens, 9,042 against 8,980. `tests/gpu/profiling/benchmark_head_logprobs.py` times the head alone at the Qwen3-8B, gpt-oss-20b and Gemma 4 26B-A4B shapes, a cold compile included.
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

## Sizing a run

For a new model, environment or GPU, size in order:

- **Rollouts in flight per server** = `world_size × min(max_concurrent_rollouts, per_device_train_batch_size × steps_per_generation) ÷ num_servers` per round, peaking at a round's start, with up to `prefetch_depth` rounds in flight at once; an eval round opens with `world_size × eval_rollout_batch_size ÷ num_servers` ([eval rounds](evaluation.md#during-training)). TP and ETP siblings roll out too, though only each group leader's rollouts are kept ([Ray — Pool sizing](../../infrastructure/ray.md#pool-sizing)).
- **Decode speed at that concurrency.** Engine step latency ≈ `a + b × n` (one CPU thread). Measure two concurrencies against a live server with the run's own prompts ([Throughput](../../infrastructure/rollout-servers.md#throughput)).
- **Timeouts from the per-turn cap**, the turn's total ([Enforced and soft budgets](reasoning.md#enforced-and-soft-budgets)): `rollout_max_tokens`, or the engine's reasoning cap plus `rollout_max_answer_tokens` where that is smaller.
    - `request_timeout ≥ 2 × cap ÷ speed(n_peak)` (default `120`).
    - `episode_timeout ≥ min(max_turns × cap, rollout_max_episode_tokens) ÷ speed(n_peak) + tool time` (default `1200`); over the NCCL watchdog it raises at startup.
- **KV capacity.** bf16 KV per token = `2 × full-attention layers × kv_heads × head_dim × 2 B`, 20 KB on Qwen3.6-35B-A3B. Check `n_peak × mean context × that` against the pool the server reports at startup.
- **Step time.** The round is the batch's slowest episode. The update is `per-rank episodes × per-episode train time` (~19 s per 43k-token episode, 35B-A3B full fine-tune, six B300 ranks) and hides inside the round while shorter.
- **Serving against training.** A trainer GPU absorbs about `3600 ÷ per-episode train time` episodes per hour. Keep the two within 2×: roughly one serving GPU per trainer GPU, each running one TP=1 engine while the model and its KV fit.

Worked shape, the 4-rank code-contests recipe (Qwen3.6-35B-A3B, 2 servers, `gradient_accumulation_steps: 24`, `num_generations: 8`):

| Quantity | Value |
|---|---|
| Episodes per step | 96 (12 prompts), 48 per server; an eval round (`eval_rollout_batch_size: 50`) opens with 100 per server |
| Per-turn cap | The 30,000-token `rollout_max_tokens`. The levels' `thinking_tokens` (8,192–16,384) are soft; the engine closes a normal turn's reasoning only at 21,808, leaving the 8,192-token answer room |
| Timeouts | `request_timeout: 1200`, `episode_timeout: 2700` |
| KV | 48 × 13k × 20 KB ≈ 12 GB at a 13k mean context; 48 × 30k × 20 KB ≈ 29 GB once every request holds a whole turn; about twice both in an eval round |
