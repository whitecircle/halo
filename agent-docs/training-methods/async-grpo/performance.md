# Memory and Throughput

Async GRPO is rollout-bound at the step level and logits-bound inside the training forward.

## The logits wall

TRL's GRPO computes per-token log-probs from a full `[B, T, vocab]` logits tensor. At gpt-oss's ~201k vocab, a 24k-token trajectory is ~9 GiB of bf16 logits, roughly doubled by the log-softmax saved for the backward.

Neither EP nor TP shrinks the logits: EP and MoE TP leave `lm_head` whole, and dense TP gathers its full-vocab output on every rank. Three levers help:

- [Chunked log-probs](#chunked-log-probs).
- `per_device_train_batch_size: 1`. Both forwards chunk to this many rows, so peak follows the longest single row.
- `fp32_output_conversion: false` (the default), keeping logits in bf16.

With chunked log-probs off, every GRPO trainer weighs the logits plane against free device memory at construction: `rows × (completion width + 1) × vocab × 4 B`.

- **Rows**: `per_device_train_batch_size`, or `per_device_eval_batch_size` when evaluation runs and it is larger.
- **Width**: `rollout_max_tokens`, or `max_train_row_tokens` under `train_on_sampled_tokens: false` (the context window when unset). Online and offline GRPO read `max_completion_length` ([`use_chunked_grpo_logprobs`](../../reference/configuration-reference.md#offlinegrpoconfig)).
- **Verdict**: a plane that cannot fit raises on every rank; one over half the free memory warns.

The estimate is a floor: it covers one forward chunk, a single-row fallback trajectory is wider than a turn, and a model left on the host (accelerate's own FSDP) is weighed against an empty device.

## Chunked log-probs

`use_chunked_grpo_logprobs: true` (default off) removes the wall: log-probs come from the backbone's `last_hidden_state` through a chunked (sequence × vocab) matmul, so peak follows the tile size. Every code-contests recipe sets it; online and offline GRPO take the same flag.

- **Precision.** Logits and log-probs stay fp32 (the backward takes bf16 operands, fp32 accumulation), within 5e-3 nats of an fp64 reference. The full path rounds logits to bf16, about 0.1 nats at |logit| ≈ 20.
- **Cost.** A recompute backward at about the full path's cost (Qwen3-8B head, 8k tokens: 51 ms forward+backward on a B300). Offline GRPO on 8× B300: Qwen3-8B at 8,192 tokens runs 18,367 tok/s/GPU against 19,266 on full logits; gpt-oss-20b EP8 at 4,096 tokens, 9,042 against 8,980.
- **Support.** FSDP2, ep1/EP and attention-only TP. A multimodal batch takes the full path. A PEFT adapter on `lm_head` raises at the first chunked forward.

The sweep applies the family's head transform (Gemma's softcap, Cohere's `logit_scale`, Granite's `logits_scaling`, Inkling's μP division and vocabulary cut), verified against the family's own forward at construction (`src/models/head_transform.py`). A mismatch is refused; a new family declares its transform ([Adding a Model](../../models/adding-a-model.md#declare-the-head-transform)).

Under FA4, or at `per_device_train_batch_size: 1`, rows are trimmed to their real span before the backbone forward and the head sweep.

## Context budget

Keep the trained trajectory under the served context. Reasoning survives only on the last turn:

`prompt + last-turn(reasoning + answer) + Σ other-turns(answer + tool output)`

The per-turn reasoning budget counts once, not `max_turns` times (twice under `carry_reasoning`). `prompt + max_turns × rollout_max_tokens` is a loose bound that over-restricts `max_turns`; `rollout_max_episode_tokens` bounds the generated share directly ([Trajectory length](episodes.md#trajectory-length)).

- Cap the longest training row with [`max_train_row_tokens`](episodes.md#trajectory-length) (unset by default; `50000` in the Qwen3.6 curriculum stages).
- Bound tool output with `max_observation_chars` (default `16384`, in `environment_kwargs`), which every tool-use environment applies ([Common knobs](environments/README.md#common-knobs)).
- The usual levers apply: `gradient_checkpointing: true` (`use_reentrant` is forced to `true` on every MoE) and `optim: adamw_torch_fused` → AdamWBF16 at 6 B/param ([BF16 optimizer](../../optimization/bf16-optimizer.md#usage)).

## Throughput levers

Each step is generate → train → NCCL sync, serial on one server, so idle trainer GPUs during generation are expected. In priority order:

1. **Multiple servers plus prefetch**, the only way to overlap generation with training. Prefetch runs one round deep, so **step time = sync + max(round, update)**. It auto-disables on a single server.
2. **Saturate the servers.** The optimizer step's row count sets the rollouts in flight; `max_concurrent_rollouts` only throttles them. Grow the batch with `gradient_accumulation_steps` (no per-forward memory cost), then raise the cap to match. `Waiting: > 0` or timeouts means you overshot.
3. **Speed generation.** Server-side levers are on [Rollout Servers](../../infrastructure/rollout-servers.md#throughput). Trainer-side, keep `rollout_max_tokens` and `rollout_max_thinking_tokens` no larger than the task needs; `sync_weights_every_n_steps > 1` amortizes the sync.
4. **Cut `max_turns`.** Turns are sequential, so halving them roughly halves rollout time. If `episode/turns` is pinned at the cap, raise it; well below, lower it.

`nvidia-smi dmon`, `EfficiencyCallback` or `async/prefetch_hit_rate` shows which phase bounds the step ([Prefetch](setup.md#multiple-servers-and-prefetch)).

## Sizing a run

For a new model, environment or GPU, size in order:

- **Rollouts in flight per server** = `world_size × min(max_concurrent_rollouts, per_device_train_batch_size × steps_per_generation) ÷ num_servers`, peaking at a round's start; an eval round opens with `world_size × eval_rollout_batch_size ÷ num_servers` ([eval rounds](evaluation.md#during-training)). TP and ETP siblings roll out too, though only each group leader's rollouts are kept ([Ray — Pool sizing](../../infrastructure/ray.md#pool-sizing)).
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
