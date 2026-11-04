---
name: optimize
description: >-
  Recommend throughput/memory levers to raise tokens/s/GPU or cut peak memory for
  a given Halo training config — and call out the levers that did not pay off on B300
  (low-precision fp8/fp4 compute, the DeepGEMM adapter, torch.compile on EP MoE,
  sub-bf16 master weights) so the user does not
  chase dead-ends. Use when the user asks how to speed up training, fit a longer
  seq / bigger batch, get more tok/s/GPU, lower peak memory / avoid OOM, raise MFU,
  or whether fp8/fp4/torch.compile/DeepGEMM are worth turning on. Every number is
  cited to agent-docs/optimization/*.md — do not invent figures.
allowed-tools: [Read, Grep, Glob, Bash]
---

# Optimize: throughput & memory levers

Recommend levers to raise **tokens/s/GPU** (the headline metric — MFU/achieved-TFLOPS are opt-in
diagnostics from `EfficiencyCallback`, *not* the goal) or cut **peak memory** for a concrete config.
This page carries the decision flow only. Every figure lives on the lever's own page under
`agent-docs/optimization/`, with the shape it was measured at: read it there and quote it from
there, never from memory.

## How to use this skill

1. Pin the **bottleneck**: throughput-bound (want more tok/s/GPU) or memory-bound (OOM /
   want longer seq / bigger batch)? For MoE/EP also ask where the layer's time goes: expert
   work and the wait for the busiest rank, or permute/elementwise work (high top-k, e.g.
   qwen3.6)? When unsure, measure: `tests/gpu/profiling/benchmark_sft_ep.py --comm_profile`
   for the per-rank EP split, or `enable_torch_profiler: true` →
   `python scripts/profiling/trace_report.py` (TraceLens compute/comm/idle split). A rank's
   `ep.combine` includes its wait for the slowest rank, so it is not wire time.
2. Confirm the **shape**: dense vs MoE/EP, short vs long seq, fixed- vs variable-length data.
3. Recommend from the flow below; take the measured effect and the exact YAML/env flag from the
   lever's page. Measure an EP throughput delta at batch ≥ 4 ([why](../../agent-docs/optimization/liger-kernels.md#benchmarks)).
4. Steer away from the [levers that do not help](../../agent-docs/optimization/README.md#levers-that-do-not-help-at-these-shapes) —
   they are the common dead-ends.

## Already on by default (don't "enable" — they're free)

Mention these only to confirm, not as new advice:
[grouped GEMM](../../agent-docs/optimization/grouped-gemm.md) and its atomic-free permute and fused MoE path,
[Liger](../../agent-docs/optimization/liger-kernels.md), [FA4 on Blackwell](../../agent-docs/optimization/flash-attention.md),
[AdamWBF16 with stochastic rounding](../../agent-docs/optimization/bf16-optimizer.md) (auto-off under replicated DDP),
[two-layer FSDP2 shard groups](../../agent-docs/parallelism/data-parallelism.md#shard-groups),
[local-shard TP linears](../../agent-docs/parallelism/tensor-parallelism.md#local-shard-linears),
`CUDA_DEVICE_MAX_CONNECTIONS=1` (baked into the image), and FlexAttention on the masked layers of SDPA runs (Gemma 4, Step-3.7)
([`sdpa_flex_sliding`](../../agent-docs/optimization/flash-attention.md#sliding-window-and-wide-head-layers-on-sdpa)).

## Throughput flow (raise tok/s/GPU)

- **MoE/EP, any shape →** push **seq × batch as high as memory allows** first: it fills the expert
  GEMMs, so batch is a leading EP lever. Read achieved TFLOPS / tok-s, not plain MFU%
  ([Throughput benchmarks](../../agent-docs/optimization/throughput-benchmarks.md#maximizing-throughput-sequence--batch)).
- **MoE/EP, fits without GC →** turn **gradient checkpointing off**, the largest single lever in the EP
  feature A/B, at a much larger activation footprint
  ([the A/B](../../agent-docs/optimization/throughput-benchmarks.md#where-the-ep-steps-time-goes)).
  On packed SFT rows `gradient_checkpointing_plan: auto` measures whether it fits and checkpoints only the
  layers the memory needs ([GradientCheckpointingPlanCallback](../../agent-docs/training-methods/callbacks.md#gradientcheckpointingplancallback)).
- **MoE, small model fits in FSDP →** prefer **FSDP (ep1) over EP** (no all-to-all); use EP only when
  FSDP OOMs ([Maximizing achieved TFLOPS](../../agent-docs/optimization/throughput-benchmarks.md#maximizing-achieved-tflops)).
- **Pure EP on 8 GPUs →** **ep2 (throughput) or ep8 (memory)**; ep4 there is rejected at config time
  ([why](../../agent-docs/parallelism/expert-parallelism.md#single-domain-multi-group-ep-races-and-hangs)).
  For finer expert sharding use **EP+ETP** (`ep4+etp2`); EP+TP does not help.
- **Dense, long context →** FA4's win over FA2 is largest here; confirm it is selected, unless the resolver
  chose FA2 on purpose ([Model-specific handling](../../agent-docs/optimization/flash-attention.md#model-specific-handling)).
- **Variable-length data (avg << max_len) →** **packing** is the big win, padding-free the smaller one
  (when cross-sequence boundaries are undesirable); uniform long rows tie
  ([Padding-free collator](../../agent-docs/optimization/padding-free-collator.md#when-to-use-each)). Rows sharing a long prefix:
  [tree packing](../../agent-docs/optimization/tree-packing.md).
- **Multi-socket host, host-bound rows (TP, CP, small-batch EP) →** `halo launch … --numa-binding node`, torchrun
  launches only ([NUMA binding](../../agent-docs/optimization/throughput-benchmarks.md#numa-binding)).
- **Pipeline parallelism, memory allows →** gradient checkpointing off, on dense stages and under EP alike
  ([Measured throughput](../../agent-docs/parallelism/pipeline-parallelism.md#measured-throughput)).
- **Convergence speed (fewer steps to a target loss) →** `optim: muon` on matrix params, at a costlier
  optimizer step and higher peak memory; a convergence lever, not a per-step one
  ([Muon](../../agent-docs/optimization/muon-optimizer.md)).

## Memory flow (cut peak / fit longer seq / bigger batch)

- **Long-seq OOM at the loss →** **FusedLinearCrossEntropy**
  (`liger_kernel_config: {cross_entropy: false, fused_linear_cross_entropy: true}` — sub-keys of that
  dict, not top-level fields): never materializes the logits plane; SFT-only, no entropy logging, forced
  off under TP and PP, and run per chunk by the CP wrapper under CP
  ([Liger](../../agent-docs/optimization/liger-kernels.md#crossentropy-vs-fusedlinearcrossentropy),
  [CP fused head loss](../../agent-docs/parallelism/context-parallelism.md#fused-head-loss)). GRPO: `use_chunked_grpo_logprobs`
  ([the logits wall](../../agent-docs/training-methods/async-grpo/performance.md#the-logits-wall)).
- **MoE activation OOM →** keep **GC on** and/or **raise the EP degree** (less expert memory per GPU).
- **Fragmentation on a dense run (reserved well above allocated) →**
  `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` in front of the launch, as the dense SFT, DPO and SMPO
  examples' launch comments carry it (13–14% less peak reserved memory). Under EP it costs ~1.5% throughput,
  so set it there only to fit
  ([Allocator](../../agent-docs/optimization/throughput-benchmarks.md#allocator-expandable-segments)).
- **Dense model does not fit one GPU at load →** **TP**: plain FSDP2/DDP loads the whole model onto every
  GPU before sharding ([Data Parallelism](../../agent-docs/parallelism/data-parallelism.md#fsdp2-strategy-by-mode)).
  Past one node, **PP** splits the layers across nodes
  ([Pipeline Parallelism](../../agent-docs/parallelism/pipeline-parallelism.md)).
- **Optimizer-state OOM →** AdamWBF16 (auto) already halves fp32 AdamW's state; beyond it
  `optim: flash_adamw` ([FlashAdamW](../../agent-docs/optimization/flash-adamw.md#when-to-use-which-optimizer)).
- **Exact fp32 on dense params with headroom →** `fp32_non_ep_params: true`
  ([BF16 optimizer](../../agent-docs/optimization/bf16-optimizer.md#master-weight-and-grad-reduce-options)).
- **Fit on a small/consumer GPU →** **QLoRA**, not under EP, TP or PP ([PEFT](../../agent-docs/optimization/peft.md#quantized-training-qlora)).
- **Grad-accumulation runs with headroom →** `fsdp_defer_grad_sync: true` reduces once per step
  ([Deferred gradient reduce](../../agent-docs/parallelism/data-parallelism.md#deferred-gradient-reduce-fsdp_defer_grad_sync)).
- **Many-rank / multi-node grad precision (not memory) →** `fp32_grad_reduce: true`.

## Sources of truth

The `agent-docs/optimization/` pages own every figure and rule above; the code is the **ultimate**
authority when a page, this skill or memory disagrees with the `halo/` implementation or a fresh
benchmark. Find the module behind a lever in the [module map](../../agent-docs/reference/architecture.md).
Related skills: `data` (packing / padding-free / sharding), `checkpoints` (low-precision fp8/fp4 export).
