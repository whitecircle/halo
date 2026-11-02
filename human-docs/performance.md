# Performance

The throughput to expect, the levers that move it, and the ones that do not
help at MoE shapes.

## Rough numbers

On 8× B300 with the Blackwell image, GPT-OSS 20B trains at about 28,700
tokens/s per GPU at `ep1` (batch 4 × 4096, no checkpointing) and Qwen3.5-35B-A3B
at about 16,400 at `ep2` with checkpointing on. Every measured shape, from one
dense GPU to 32k-token EP + CP, is in
[Throughput Benchmarks](../agent-docs/optimization/throughput-benchmarks.md) ↗.

In short:

- **EP costs throughput.** Pick the lowest expert-parallel size that fits, not
  the largest your GPUs allow.
- **CP buys sequence length, not speed.**
- **Sharding buys capacity**, at a cost in tokens per second.

Before you judge a number, check `M = per_device_train_batch_size × max_length`.
On a B300 MoE run it should be at least ~8k; below that the step is
latency-bound. Set `enable_efficiency_metrics: true` to log tokens/s/GPU in your
own run.

## Levers that help

Set these per run:

- **A bigger `M`**: raise `per_device_train_batch_size` or `max_length`, and
  fill the global batch with `gradient_accumulation_steps`, not more
  parallelism. It is the largest effect on MoE.
- **`gradient_checkpointing: false`** when the activations fit, at much higher
  peak memory. On packed SFT rows `gradient_checkpointing_plan: auto` finds that
  out itself: it measures step 2 and checkpoints only the layers the memory
  needs ([Callbacks](../agent-docs/training-methods/callbacks.md#checkpointingplancallback) ↗).
- **`packing: true`** on a corpus of short rows; `padding_free: true` gains
  less. Where SFT rows share long prefixes (GPT-OSS, Qwen3, Qwen3.5/3.6,
  GLM-5 Next), `tree_packing: true` stores each prefix once, at some extra
  step time on full-attention layers
  ([Tree Packing](../agent-docs/optimization/tree-packing.md) ↗).
- **`liger_kernel_config: {fused_linear_cross_entropy: true}`** past ~16k
  tokens, which trades a little speed for a lot of logits memory. Under
  context parallelism the CP wrapper runs it over each rank's chunk.
- **`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`** in front of a dense
  launch: 13–14% less peak reserved memory at the same speed, which is why the
  dense SFT, DPO and SMPO examples carry it. Under EP it costs throughput, so
  set it there only to fit
  ([Allocator](../agent-docs/optimization/throughput-benchmarks.md#allocator-expandable-segments) ↗).
- **`fsdp_defer_grad_sync: true` with `fsdp_reshard_after_backward: false`**
  under gradient accumulation: one gradient reduce and one re-gather per
  optimizer step, at an unsharded gradient copy per GPU.
- **`use_chunked_grpo_logprobs: true`** on GRPO when the full logits do not fit.

On by default: grouped GEMM for MoE experts, Flash Attention 4 on Blackwell,
Liger, `AdamWBF16` (6 bytes per parameter instead of 12), the atomic-free
expert permute and its sync-free routing sort, the fused MoE path
(`HALO_FUSED_GLU=0` turns off its GLU kernels), TP linears on their local
shards, two decoder layers per FSDP2 shard group under ZeRO-2, one Ulysses
all-to-all for Q, K and V under CP, and `sdpa_flex_sliding` for Gemma 4 and
Step-3.7 (`HALO_FLEX_SLIDING=0`). Each lever's page carries its
measurements: [Optimization](../agent-docs/optimization/README.md) ↗.

## Levers that do not help here

fp8 / fp4 compute (`lowp_precision`), native DeepGEMM and `torch.compile` on an
EP MoE run were each measured and lose at these shapes, where fine-grained
experts are bound by weight bandwidth, not FLOPs
([Levers that do not help](../agent-docs/optimization/README.md#levers-that-do-not-help-at-these-shapes) ↗).

![Roofline for a B300 expert GEMM, showing small-M experts far below the compute ceiling](../agent-docs/assets/diagrams/roofline.png)

An expert GEMM with few tokens sits in the memory-bound region. At 128 token
rows against GPT-OSS's 16 MB expert weight, the tensor cores idle while HBM
streams the weight, and fine-grained MoEs put even fewer tokens on each expert.
The fix is a bigger `M`, not a faster kernel.

## Compared with stock TRL

Halo trains GPT-OSS 20B at 2.6–3.5× stock TRL (dense and `ep2`) on the same 8
B300s, with the same kernels on both sides, and both reach the same loss. Most
of the gap is structural: expert parallelism, no per-micro-step parameter
re-gather, and 6-byte optimizer state
([full comparison](../agent-docs/optimization/halo-vs-stock-trl.md) ↗).

For memory rather than speed, start at [Parallelism](parallelism.md). For a run
slower than these numbers with no obvious cause, turn on the profiler
([Monitoring](monitoring.md)). Why the defaults are what they are:
[GPU Training Theory](../agent-docs/reference/gpu-training-theory.md) ↗.
