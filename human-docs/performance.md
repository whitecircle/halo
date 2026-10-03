# Performance

What to expect from a run, which levers actually move it, and which famous ones
do nothing at MoE shapes.

## Rough numbers

Measured on B300 (Blackwell), bf16, Liger and grouped GEMM on, Flash Attention 4
where the family takes it (Qwen3.5 falls back to SDPA). Tokens per second per
GPU, so a number scales by the GPU count:

| Model | Shape | tok/s/GPU |
| --- | --- | --- |
| Qwen3-4B dense | 1 GPU, batch 8 × 2048, no checkpointing | 39,900 |
| GPT-OSS 20B | 8 GPUs, `ep1`, batch 4 × 4096, no checkpointing | 24,500 |
| Qwen3.5-35B-A3B | 8 GPUs, `ep2`, batch 4 × 4096, checkpointing on | 12,600 |
| GPT-OSS 20B | 8 GPUs, `ep8`, batch 4 × 4096, checkpointing on | 10,100 |
| GPT-OSS 20B, 32k context | 8 GPUs, `ep8 + cp8`, checkpointing on | 6,000 |

Two things to read out of that table. Expert parallelism costs throughput — at
batch 4, 4k and matched checkpointing `ep1` runs about 2× `ep8` on the same
model — because it trades local parameters for all-to-all traffic, so pick the
*lowest* EP that fits rather than the largest your GPUs allow. And context parallelism is a way to afford a
long sequence, not a way to go faster: it holds memory nearly flat from 16k to
64k tokens and buys no speed. Sharding buys capacity — a 119B MoE trains on four
GPUs at `ep2 + etp2` — and you pay for it in tokens per second.

One rule of thumb before you judge any number: `M = per_device_train_batch_size ×
max_length` should be at least ~8k on a B300 MoE run. Below that the step is
latency-bound and the GPUs are waiting, not computing. Turn on
`enable_efficiency_metrics: true` to get tokens/s/GPU in your own logs.

## Levers that help

| Lever | What it buys |
| --- | --- |
| Bigger `M` — raise `per_device_train_batch_size` or `max_length` | the single largest effect; batch 1 → 4 at 4k is 1.2–2.4× on MoE, least at high EP, and at 16k `ep8` loses throughput under batch (its dispatch grows with tokens per rank). Fill the global batch with `gradient_accumulation_steps`, not more parallelism |
| `gradient_checkpointing: false` when activations fit | +29% on GPT-OSS `ep8` at 4k, at roughly double the peak memory |
| `packing: true` | 9.2× on a corpus averaging a quarter of `max_length` (`padding_free: true`: 2.3×); nothing when rows already fill it |
| `use_grouped_gemm: true` (default on SM90+) | 2.1–3.4× end-to-end at `ep2` — one batched expert matmul instead of a loop |
| Flash Attention 4 (auto on Blackwell) | 1.1× at 4k rising to 2.3× at 32k on dense; ~+13% on MoE, where all-to-all dominates |
| `use_liger_kernel: true` (default) | +40% and 19 GiB on Qwen3-30B-A3B at `ep2` (measured at v1.0.0), +6.6% on Qwen3.5-35B-A3B; add `liger_kernel_config: {fused_linear_cross_entropy: true}` past ~16k tokens, which trades 7–20% of speed for 14–30 GiB on GPT-OSS `ep1` (the cost shrinks as the sequence grows) |
| `AdamWBF16` (automatic with `bf16: true`) | weights and optimizer state in 6 bytes/param where fp32-state AdamW needs 12, and a 17% shorter step than `adamw_torch_fused` |
| `fsdp_defer_grad_sync: true` and `fsdp_reshard_after_backward: false`, with `gradient_accumulation_steps > 1` | one gradient reduce and one parameter re-gather per optimizer step instead of per microstep. On one 8-GPU node the deferred reduce gives +3% on Qwen3-8B (+5–7% with both knobs) and +11.7% on Qwen3-30B-A3B at `ep_size: 1`; across two nodes over EFA it gives +9–13% on its own. Each keeps an unsharded copy per GPU (Qwen3-8B: +13 GB for the gradients). The config refuses both under `fsdp_reshard_after_forward: true` (ZeRO-3). [Details](../agent-docs/parallelism/data-parallelism.md) ↗ |
| The fused MoE path (on by default) | fused GLU, torch's fused RMSNorm on four families, a fused weighted un-permute and a gradient clip folded into `AdamWBF16`'s step, measured together (29cf60ded against 425f04103): 1.24× on Gemma 4 26B-A4B at `ep2` and 2,048 tokens (most of it from the optimizer), 1.09–1.25× on GLM-4.7-Flash, Qwen3-30B-A3B and GPT-OSS 20B at `ep2` and 4,096 tokens (2× B300, same peak memory). `HALO_FUSED_GLU=0` turns the GLU kernels off |
| FlexAttention on Gemma 4's sliding layers (on by default) | 1.34× at 2,048 tokens and 3.93× at 16,384 on Gemma 4 26B-A4B at `ep2` without checkpointing (2× B300), and 12.6 GiB less peak at 16k. `HALO_FLEX_SLIDING=0` turns it off |
| `use_chunked_grpo_logprobs: true` (GRPO) | completion log-probs without the full `[tokens, vocab]` logits, in fp32, at about the full-logits speed: offline GRPO on 8 B300s runs 18,367 vs 19,266 tok/s/GPU on Qwen3-8B at 8,192 tokens and 9,042 vs 8,980 on GPT-OSS 20B `ep8` at 4,096. Turn it on when the logits do not fit |

The defaults already have most of this on. The levers you actually set per run
are the first three.

## Levers that do not help here

| Lever | Why not |
| --- | --- |
| fp8 / fp4 **compute** (`lowp_precision`) | fine-grained experts are weight-bandwidth-bound, not FLOP-bound, so halving the matmul precision buys nothing — bf16 is already at the roofline |
| The simulated low-precision backend | it is an exact QAT oracle, not a fast path: roughly 8× a bf16 step at mxfp8, 17–19× at fp4 |
| Native DeepGEMM (`HALO_DEEPGEMM_NATIVE=1`) | 0.05–0.07× of bf16 at production shapes; the per-token activation quantization never amortizes. Never auto-selected |
| `torch_compile: true` on an EP MoE run | it works, but it targets the same spans Liger already fuses, so stacking adds ~2%, against 2–5 minutes of compile on the first step |

Low precision does earn its keep on the way out: `halo run quantize-to-lowp`
halves (fp8) or quarters (fp4) the expert-weight bytes of a checkpoint you are
about to serve. That is a memory lever, not a training one.

![Roofline for a B300 expert GEMM, showing small-M experts far below the compute ceiling](../agent-docs/assets/diagrams/roofline.png)

Fine-grained MoE experts sit in the shaded memory-bound region: with 128 token
rows against a 16 MB weight, the tensor cores finish and idle while HBM streams,
so the fix is a bigger `M`, not a faster kernel.

## Why not stock TRL

Halo trains GPT-OSS 20B at 2.3–2.8× stock TRL across every short and mid-length
config on the same 8 B300s, with the same kernels and the strongest stock options
enabled on both sides. Most of the gap is structural rather than kernel-level:
the baseline runs no expert parallelism, and its FSDP2 `full_shard` re-gathers
all 20.7B parameters every micro-step, a fixed cost a short step cannot hide,
while its AdamW keeps 12 bytes per parameter of fp32 state where `AdamWBF16`
keeps 6. At the same ZeRO-3 sharding and expert kernel, dense Halo still leads
1.2–2.7×. The `ep8` shape trades some of that speed back for memory:
1.3–2.1× TRL at about half its footprint. None of it costs convergence — TRL,
dense Halo, `ep2` and `ep8` all land within ~1% of the same loss over 200 seeded
steps.

## Going deeper

[GPU Training Theory](../agent-docs/reference/gpu-training-theory.md) ↗ is the
long-form version of the roofline argument above: arithmetic intensity, the ridge
point, where a step's time actually goes. The measured tables behind every number
on this page are in
[Throughput Benchmarks](../agent-docs/optimization/throughput-benchmarks.md) ↗ ·
[Low-Precision MoE](../agent-docs/optimization/low-precision-moe-kernels.md) ↗ ·
[vs Stock TRL](../agent-docs/optimization/halo-vs-stock-trl.md) ↗.

For memory rather than speed, start at [Parallelism](parallelism.md); for a run
that is slower than these numbers without an obvious cause,
[Monitoring](monitoring.md) turns on the profiler.
