# Halo vs stock TRL / transformers

Halo 1.2.0 against `trl.SFTTrainer` 1.17.1 on Transformers 5.20.2, 8× B300 and the Blackwell image, bf16, FA4, Liger,
`grouped_mm` experts. Same model, rows and kernels on both sides; only the framework changes. The rates themselves, with
NeMo AutoModel and Megatron Bridge beside them, and the conditions that decide how to read them are in
[Throughput Benchmarks](throughput-benchmarks.md#full-parameter-sft-framework-comparison). This page holds the TRL
side: layouts, ratios, what differs, and the older long-context and convergence runs.

## Halo 1.2.0 against TRL 1.17.1

Short rows (16k and 4k, 131,072 tokens per step), Halo over TRL:

| Shape | Halo 1.2.0 / TRL | TRL layout |
|---|---:|---|
| Qwen3-8B, 16k | 1.10 | ZeRO-2 (ZeRO-3: 16,166 tok/s/GPU) |
| Qwen3-8B, 4k, batch 4 | 1.13 | ZeRO-2 (ZeRO-3: 19,369) |
| gpt-oss-20b, 16k | 2.77 | ZeRO-2, aux coefficient 0 (ZeRO-3: 9,011) |
| gpt-oss-20b, 16k, checkpointing | 2.47 | ZeRO-2 (ZeRO-3: 8,282) |
| gpt-oss-20b, 4k, batch 4 | 3.06 | ZeRO-2 (ZeRO-3: 9,309) |
| Qwen3.5-35B-A3B, 16k | 1.41 | ZeRO-2, FA4, steady state (1.54 against its 10-step mean of 13,137, which includes two slow steps) |
| Qwen3-30B-A3B, 16k | 1.27 | ZeRO-2 (ZeRO-3: 12,750) |
| Qwen3.6-35B-A3B, 16k | 1.59 | ZeRO-2, SDPA (FA4 not run; the gap is likely smaller) |

On packed 64k and 128k rows Halo runs 1.17× TRL on Qwen3.5-9B, 1.47× on Qwen3.6-35B-A3B and 1.50× on
Qwen3.5-122B-A10B. TRL's rows there are pure bf16 parameters and Adam moments; its fp32-master runs are within 3% on the
9B and 35B and do not fit on the 122B. Halo's rows are the median of 3 runs of a chosen setup; TRL's are single runs of its best
layouts.

Halo's lead is largest on gpt-oss-20b (2.47–3.06×) and smallest on dense Qwen3-8B (1.10–1.13×). The first-step loss on the same batch agrees to three decimals in seven of the eight short shapes. TRL's default adds the
router auxiliary loss (coefficient 0.9 on gpt-oss) to its loss, so every throughput row sets it to 0.

## What differs between the two sides

| | TRL 1.17.1 | Halo |
|---|---|---|
| Trainer | `trl.SFTTrainer` | `DistributedSFTTrainer` |
| Precision | bf16 + FA4 bf16 compute | *identical* |
| Optimizer | `adamw_torch_fused` over bf16 weights: bf16 moments, nearest rounding, 6 B/param (fp32 master with mixed precision) | AdamWBF16 + stochastic rounding, 6 B/param |
| Sharding | FSDP2 ZeRO-2 or ZeRO-3 (`reshard_after_forward`) | FSDP2 ZeRO-2 default + Expert Parallelism, ZeRO-3 for the 122B |
| Expert kernel | `grouped_mm` (Transformers default) | `grouped_mm` + EP token distribution |
| Liger / loss | Liger on; fused linear CE in the MoE applier | Liger on, plain CE by default; fused linear CE in the comparison rows that name it |
| Packed rows | pre-packed rows with `seq_lengths`, `padding_free`, FA4 varlen | `DataCollatorWithPacking`, FA4 varlen |

## TRL expert parallelism

TRL 1.17.1 has no expert-parallel option of its own. Transformers' expert-parallel path (`ep_size` 8, experts as
DTensors, 4 of 32 local) switched on through a local patch trains gpt-oss-20b at 16k at 2,184 tok/s/GPU and 117.6 GiB,
4.4× below TRL's ZeRO-2 row and 9× below Halo's EP8 row (19,817). It is untuned and not a TRL trainer feature.

## Gradient checkpointing on vs off

gpt-oss-20b at 16k, batch 1, Halo EP1 with fused linear cross-entropy against TRL ZeRO-2:

| | Checkpointing | No checkpointing | Gain | Peak GiB (on → off) |
|---|---:|---:|---:|---|
| Halo 1.2.0 | 21,559 | 26,612 | +23% | 62.1 → 116.3 |
| TRL 1.17.1 | 8,741 | 9,595 | +10% | 76.6 → 163.5 |

Halo's lead widens without checkpointing (2.47× to 2.77×), and its peak is lower in both columns. TRL on ZeRO-3 with
checkpointing fits in 50.6 GiB (8,282 tok/s/GPU), below Halo's 62.1 GiB.

## Long context on gpt-oss-20b (Halo 1.0.0)

These rows ran on Halo 1.0.0 against the TRL that release pinned (version not recorded), ZeRO-3 with fused linear
cross-entropy, batch 1, gradient checkpointing, tok/s/GPU · peak GiB. Nothing on 1.2.0 replaces the CP rows.

| config | 128k | 256k |
|---|---|---|
| TRL (ZeRO-3) | 5,961 · 93 | 4,463 · 148 |
| **Halo EP1 ZeRO-3 (dense)** | **9,566 · 67** | **5,728 · 112** |
| Halo EP8+CP4 (DP2) | 1,898 · 81 | 2,310 · 142 |
| Halo EP8+CP8 (DP1) | 3,810 · 47 | 2,395 · 77 |
| Halo dense CP-only (ZeRO-3) | 3,279 · **32** | 1,836 · **56** |
| Halo EP1 ZeRO-2, EP8, EP8+CP2, EP8+TP8 | OOM / crash | OOM / crash |

- Dense EP1 ZeRO-3 wins throughput at both lengths and still beats TRL on memory.
- More CP is strictly better past 64k: `EP8+CP8` beats `EP8+CP4` on both axes at 128k.
- Dense CP-only ZeRO-3 is the leanest point (32–56 GiB across 128k→256k) and the only long-context mode with no DeepEP
  dependency. Splitting 4-way instead of 8 trades memory for throughput (256k: CP4 2,257 · 98 vs CP8 1,836 · 56).
- `EP1` ZeRO-2 OOMs (ZeRO-2 keeps ~40 GB of params resident). `EP8`, `EP8+CP2` and `EP8+TP8` put 128k tokens per rank
  through the dispatch, which neither transport trains: elastic EP8 deadlocks at ≥~64k tokens per rank (its combine
  barrier races FSDP2's reduce-scatter) and `ep_buffer_backend: legacy` fails at 128k
  ([DeepEP → Transport backend](../infrastructure/deepep.md#transport-backend),
  [Dispatch wire-index limit](../infrastructure/deepep.md#dispatch-wire-index-limit)).

## Convergence (Halo 1.1.0)

200 steps on the same seeded data, same global batch (16), constant LR, 8× B300. TRL (ZeRO-3), Halo dense (EP1), EP2 and
EP8 reach the same loss, about 0.00205 at step 200 (0.002048–0.002051), within 0.8% of each other by step 100.

## Reproduce

```bash
# TRL baseline, synthetic fixed-length rows (ZeRO-3 FSDP2; --fsdp_sharding shard_grad_op for ZeRO-2)
torchrun --nproc_per_node=8 tests/gpu/profiling/benchmark_trl_baseline.py \
    --model gpt-oss-20b --seq 4096 --batch_size 1
# Halo EP1 / EP2 / EP8 (add --zero3 for ZeRO-3, --no_gc to drop checkpointing)
torchrun --nproc_per_node=8 tests/gpu/profiling/benchmark_sft_ep.py \
    --model gpt-oss-20b --ep 2 --seq 65536 --batch_size 1
# convergence (--ep 0 = TRL, >=1 = Halo)
torchrun --nproc_per_node=8 tests/gpu/profiling/benchmark_convergence.py \
    --ep 0 --seq 2048 --batch_size 2 --steps 200
```

These scripts build `create_benchmark_dataset`: 64 rows of exactly `seq` uniform token ids with `labels = input_ids`, a
shape fixture whose fixed length isolates the framework comparison. Held constant: `learning_rate=2e-5`, seed 42,
`gradient_accumulation_steps=1`, global batch `8 × per_device_train_batch_size`; TRL runs
`fsdp="full_shard auto_wrap"` with `fsdp_config={"version": 2, "reshard_after_forward": …}` (true is ZeRO-3).
