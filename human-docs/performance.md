# Performance

The throughput to expect, the levers that move it, and the ones that do not
help at MoE shapes.

## Rough numbers

Measured on B300 with the Blackwell image: bf16, Liger and grouped GEMM on,
Flash Attention 4 where the family takes it (Qwen3.5 falls back to SDPA).
Tokens per second per GPU:

| Model | Shape | tok/s/GPU |
| --- | --- | --- |
| Qwen3-4B dense | 1 GPU, batch 16 × 2048, no checkpointing | 41,000 |
| GPT-OSS 20B | 8 GPUs, `ep1`, batch 4 × 4096, no checkpointing | 28,700 |
| Qwen3.5-35B-A3B | 8 GPUs, `ep2`, batch 4 × 4096, checkpointing on | 16,400 |
| GPT-OSS 20B | 8 GPUs, `ep8`, batch 4 × 4096, checkpointing on | 13,200 |
| GPT-OSS 20B, 32k context | 8 GPUs, `ep8 + cp8`, checkpointing on | 9,000 |

What the table shows:

- **EP costs throughput.** At batch 4 with checkpointing on for both, `ep1`
  runs about 1.8× `ep8` on GPT-OSS 20B (23,590 vs 13,239 tok/s/GPU). Pick the
  lowest EP that fits, not the largest your GPUs allow.
- **CP buys sequence length, not speed.** It holds memory at 25–42 GiB from 16k
  to 64k tokens.
- **Sharding buys capacity.** A 119B MoE trains on four GPUs at `ep2 + etp2`, at
  a cost in tokens per second.

Before you judge a number, check `M = per_device_train_batch_size × max_length`.
On a B300 MoE run it should be at least ~8k; below that the step is
latency-bound. Set `enable_efficiency_metrics: true` to log tokens/s/GPU in your
own run.

## Levers that help

The defaults already turn most of these on. The ones you set per run are the
first three.

| Lever | Effect |
| --- | --- |
| Bigger `M`: raise `per_device_train_batch_size` or `max_length` | the largest effect: batch 1 → 4 at 4k is 1.2–2.5× on MoE, least at high EP. Fill the global batch with `gradient_accumulation_steps`, not more parallelism |
| `gradient_checkpointing: false` when activations fit | +29% on GPT-OSS `ep8` at 4k, at about 1.8× the peak memory |
| `packing: true` | 14.5× on a corpus whose rows average a quarter of `max_length` (`padding_free: true`: 4.5×); no gain when rows already fill it |
| `use_grouped_gemm: true` (default on SM90+) | one batched expert matmul instead of a loop: 2.45–4.1× on Qwen3-30B-A3B at `ep2` ([details](../agent-docs/optimization/grouped-gemm.md) ↗) |
| Flash Attention 4 (auto on Blackwell) | 1.2× FA2 at 4k, 2.3× at 32k on dense; about +17% on MoE |
| `use_liger_kernel: true` (default) | +39% and 28 GiB less on Qwen3-30B-A3B at `ep2`. Past ~16k tokens, add `liger_kernel_config: {fused_linear_cross_entropy: true}`: on GPT-OSS `ep1` at 16k–32k it saves 14–30 GiB for 7–20% of speed, less at higher EP |
| `AdamWBF16` (automatic with `bf16: true` and the default `adamw_torch` / `adamw_torch_fused`, except under accelerate-managed DDP) | 6 bytes/param of weights and optimizer state instead of 12, at about half the step time of `adamw_torch_fused` |
| `fsdp_defer_grad_sync: true`, `fsdp_reshard_after_backward: false` (with `gradient_accumulation_steps > 1`) | one gradient reduce and one re-gather per optimizer step: +3–12% on one node, +9–13% across two. The deferred sync holds an unsharded gradient copy per GPU (+13 GiB on Qwen3-8B); both are refused under ZeRO-3 ([details](../agent-docs/parallelism/data-parallelism.md) ↗) |
| `use_chunked_grpo_logprobs: true` (GRPO) | completion log-probs in fp32 without the full `[tokens, vocab]` logits, at about full-logits speed. Turn it on when the logits do not fit |

Three more are on by default and need no setting:

- the atomic-free expert permute: +24% on GPT-OSS 20B at `ep8`;
- the fused MoE path (fused GLU and RMSNorm, the gradient clip folded into
  `AdamWBF16`): 1.09–1.25× on 2× B300; `HALO_FUSED_GLU=0` turns off its GLU
  kernels;
- `sdpa_flex_sliding` (FlexAttention on Gemma 4's sliding layers, matmul
  attention on its global ones): 1.34× at 2k and 3.93× at 16k tokens end to
  end against SDPA, on top of the fused MoE path; `HALO_FLEX_SLIDING=0` turns
  it off.

Measurements: [Grouped GEMM](../agent-docs/optimization/grouped-gemm.md) ↗ ·
[Gemma 4](../agent-docs/models/gemma4.md) ↗.

## Levers that do not help here

| Lever | Why not |
| --- | --- |
| fp8 / fp4 compute (`lowp_precision`) | fine-grained experts are bound by weight bandwidth, not FLOPs, so lower matmul precision buys nothing |
| The simulated low-precision backend | an exact QAT oracle, not a fast path: about 8× a bf16 step at mxfp8, 17–19× at fp4 |
| Native DeepGEMM (`HALO_DEEPGEMM_NATIVE=1`) | 0.05–0.07× of bf16 at production shapes; never auto-selected |
| `torch_compile: true` on an EP MoE run | at most +1% with `torch_compile_mode: default`; the `reduce-overhead` mode used when none is set is 7–10% slower ([details](../agent-docs/optimization/torch-compile.md) ↗) |

Low precision helps at export instead. `halo run quantize-to-lowp` cuts a
checkpoint's expert-weight bytes by half (fp8) or three quarters (fp4). That is a
serving-memory lever, and its block-scaled layout still needs mapping onto a
serving engine's loader.

![Roofline for a B300 expert GEMM, showing small-M experts far below the compute ceiling](../agent-docs/assets/diagrams/roofline.png)

An expert GEMM with few tokens sits in the memory-bound region. At 128 token
rows against GPT-OSS's 16 MB expert weight, the tensor cores idle while HBM
streams the weight, and fine-grained MoEs put even fewer tokens on each expert.
The fix is a bigger `M`, not a faster kernel.

## Compared with stock TRL

Halo trains GPT-OSS 20B at 2.6–3.5× stock TRL (dense and `ep2`, short and
mid-length configs) on the same 8 B300s, with the same kernels on both sides.
`ep8` runs at 1.7–2.8× TRL with about half its memory.

Most of the gap is structural. The baseline has no expert parallelism, its FSDP2
`full_shard` re-gathers all parameters every micro-step, and its AdamW keeps 12
bytes per parameter of state where `AdamWBF16` keeps 6. TRL and Halo's dense,
`ep2` and `ep8` runs reach the same loss over 200 seeded steps
([full comparison](../agent-docs/optimization/halo-vs-stock-trl.md) ↗).

## Going deeper

- [GPU Training Theory](../agent-docs/reference/gpu-training-theory.md) ↗: the
  roofline argument in full, and where a step's time goes.
- [Optimization](../agent-docs/optimization/README.md) ↗: one page per lever,
  with the measured tables behind the numbers here. Start with
  [Throughput Benchmarks](../agent-docs/optimization/throughput-benchmarks.md) ↗.

For memory rather than speed, start at [Parallelism](parallelism.md). For a run
slower than these numbers with no obvious cause, turn on the profiler
([Monitoring](monitoring.md)).
