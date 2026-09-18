# Memory and Throughput

Async GRPO is rollout-bound at the step level, logits-bound inside the training forward. Size both.

## The logits wall

TRL's GRPO computes per-token log-probs from a full `[B, T, vocab]` logits tensor: at gpt-oss's ~201k vocab, a 24k-token trajectory is ~9 GiB of bf16 logits, roughly doubled by the log-softmax saved for the backward.

Neither EP nor TP attacks it: EP wraps only the MoE experts and the MoE TP path shards attention alone, so `lm_head` stays dense either way. Three levers do:

- [Chunked log-probs](#chunked-log-probs).
- `per_device_train_batch_size: 1`. Both forwards chunk to this many rows whatever the micro-batch holds, so peak follows the longest single row.
- `fp32_output_conversion: false` (the default), keeping logits in bf16.

## Chunked log-probs

`use_chunked_grpo_logprobs: true` (default off) removes the wall: completion log-probs come from the backbone's `last_hidden_state` through a chunked (sequence × vocab) matmul, so peak follows the tile size, not `B·T·vocab`. Every code-contests recipe sets it; the online and offline GRPO trainers take the same flag.

The objective is identical — log-probs match the full path to bf16 tolerance, the head's `final_logit_softcapping` applied where the family caps its logits (Gemma) — at the cost of a recompute backward. It runs under FSDP2, ep1/EP and attention-only TP. A multimodal batch takes the full path instead, decided on every rank at once; a PEFT adapter on `lm_head` raises at the first chunked forward.

Under FA4, or at `per_device_train_batch_size: 1`, rows are trimmed to their real span before the backbone forward and the head sweep — most of the training pass, since multi-turn row lengths differ by an order of magnitude.

## Context budget

Keep the trained trajectory under the served context. Reasoning survives only on the last turn, so the budget is `prompt + last-turn(reasoning + answer) + Σ other-turns(answer + tool output)` — the per-turn CoT budget counts once, not `max_turns` times, and twice under `carry_reasoning`. `prompt + max_turns × rollout_max_tokens` bounds it loosely, which over-restricts `max_turns` ([Trajectory length](rollouts.md#trajectory-length)).

Cap the longest training row with [`max_train_row_tokens`](rollouts.md#trajectory-length), unset by default; the Qwen3.6 curriculum stages set `50000`.

Bound tool output: every tool-use environment truncates an observation at `max_observation_chars` (default `16384`, set in `environment_kwargs`). The coding grader's `max_output_size` defaults to `1_000_000` bytes for gradeable large-array results — cap it for training.

The usual levers apply otherwise: `gradient_checkpointing: true` (`use_reentrant` is forced to `true` on every MoE, EP or not — `src/trainers/mixins/base.py`) and `optim: adamw_torch_fused` → AdamWBF16 at 6 B/param ([BF16 optimizer](../../../optimization/bf16-optimizer.md#usage)).

## Throughput levers

Each step is generate → train → NCCL sync, serial on one server — trainer GPUs at idle wattage during generation are expected, not a slow step. In priority order:

1. **Multiple servers plus prefetch**, the only way to overlap generation with training. One round deep, so **step time = max(round, update)**. Auto-disabled on a single server.
2. **Saturate the servers.** Rollouts in flight are set by the optimizer step's row count, not `max_concurrent_rollouts` — a semaphore only throttles work that exists. Grow the generation batch with `gradient_accumulation_steps`, which adds no per-forward memory, then raise the cap to match. `Waiting: > 0` or timeouts means you overshot.
3. **Speed generation.** Server-side levers are on [Rollout Servers](../../../infrastructure/rollout-servers.md#throughput); trainer-side, keep `rollout_max_tokens` and `rollout_max_thinking_tokens` no larger than the task needs. `sync_weights_every_n_steps > 1` amortizes the sync.
4. **Cut `max_turns`.** Turns are sequential, so halving them roughly halves rollout time. Watch `episode/turns`: pinned at the cap, raise it; well below, lower it.

To find which phase bounds the step, profile one with `nvidia-smi dmon` or `EfficiencyCallback`; `async/prefetch_hit_rate` reads the same split ([Prefetch](setup.md#multiple-servers-and-prefetch)).

## Sizing a run

In order, for a new model, environment or GPU:

- **Rollouts in flight per server** = `data_parallel_size × per_device_train_batch_size × steps_per_generation ÷ num_servers`, peaking at the start of a round; an eval round holds `data_parallel_size × eval_rollout_batch_size ÷ num_servers` throughout ([eval rounds](monitoring.md#evaluation)).
- **Decode speed at that concurrency.** The engine step is one CPU thread, latency ≈ `a + b × n`; nothing in the config predicts it, so measure two concurrencies against a live server with the run's own prompts ([Throughput](../../../infrastructure/rollout-servers.md#throughput)).
- **Timeouts from the per-turn cap** = thinking budget + the `rollout_max_tokens − rollout_max_thinking_tokens` answer headroom. `request_timeout ≥ 2 × cap ÷ speed(n_peak)` (default `120`); `episode_timeout ≥ max_turns × cap ÷ speed(n_peak) + tool time` (default `1200`), and over the NCCL watchdog it raises at startup.
- **KV capacity.** bf16 KV per token = `2 × full-attention layers × kv_heads × head_dim × 2 B`, 20 KB on Qwen3.6-35B-A3B. Check `n_peak × mean context × that` against the pool the server reports at startup.
- **Step time.** The round is the batch's slowest episode; the update is `per-rank episodes × per-episode train time` (~19 s per 43k-token episode, 35B-A3B full fine-tune, six B300 ranks) and hides inside the round while shorter.
- **Serving against training.** A trainer GPU absorbs about `3600 ÷ per-episode train time` episodes per hour. Keep the two within a factor of two — roughly one serving GPU per trainer GPU, one engine per serving GPU at TP=1 while the model and its KV fit one GPU.

Worked shape, the 4-rank code-contests recipe: Qwen3.6-35B-A3B on 4 ranks and 2 servers, `gradient_accumulation_steps: 24`, `num_generations: 8` — 96 episodes per step (12 prompts), 48 per server; the `high` profile's per-turn cap is 22,384 tokens. It sets `request_timeout: 1200` and `episode_timeout: 2700`; KV is 48 × 13k × 20 KB ≈ 12 GB.
