---
name: optimize
description: >-
  Recommend throughput/memory levers to raise tokens/s/GPU or cut peak memory for
  a given Halo training config — and call out the levers that do NOT help
  at fine-grained MoE shapes (low-precision fp8/fp4 compute, native
  DeepGEMM, torch.compile on EP MoE, sub-bf16 master weights) so the user does not
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
   want longer seq / bigger batch)? For MoE/EP also ask: comm-bound (low top-k, e.g.
   gpt-oss) or permute/elementwise-bound (high top-k, e.g. qwen3.6)? When unsure, measure:
   `enable_torch_profiler: true` → `python scripts/profiling/trace_report.py` (TraceLens
   compute/comm/idle split; the `ep.dispatch`/`ep.combine` trace ranges give the all-to-all
   fraction directly).
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
`CUDA_DEVICE_MAX_CONNECTIONS=1` (baked into the image), and FlexAttention on Gemma 4's sliding layers
([`sdpa_flex_sliding`](../../agent-docs/optimization/flash-attention.md#sliding-window-and-wide-head-layers-on-sdpa)).

## Throughput flow (raise tok/s/GPU)

- **MoE/EP, any shape →** push **seq × batch as high as memory allows** first: EP at low token counts
  is comm-bound, so batch is the dominant EP lever. Read achieved TFLOPS / tok-s, not plain MFU%
  ([Throughput benchmarks](../../agent-docs/optimization/throughput-benchmarks.md#maximizing-throughput-sequence--batch)).
- **MoE/EP, fits without GC →** turn **gradient checkpointing off**, the largest single throughput lever,
  at about twice the activation memory.
- **MoE, small model fits in FSDP →** prefer **FSDP (ep1) over EP** (no all-to-all); use EP only when
  FSDP OOMs ([Maximizing achieved TFLOPS](../../agent-docs/optimization/throughput-benchmarks.md#maximizing-achieved-tflops)).
- **Pure EP on 8 GPUs →** **ep2 (throughput) or ep8 (memory)**; ep4 there is rejected at config time
  ([why](../../agent-docs/parallelism/expert-parallelism.md#single-domain-multi-group-ep-races-and-hangs)).
  For finer expert sharding use **EP+ETP** (`ep4+etp2`); EP+TP does not help.
- **Dense, long context →** FA4's win over FA2 is largest here; confirm it is selected.
- **Variable-length data (avg << max_len) →** **packing** is the big win, padding-free the smaller one
  (when cross-sequence boundaries are undesirable); uniform long rows tie
  ([Padding-free collator](../../agent-docs/optimization/padding-free-collator.md#when-to-use-each)). Rows sharing a long prefix:
  [tree packing](../../agent-docs/optimization/tree-packing.md).
- **Convergence speed (fewer steps to a target loss) →** `optim: muon` on matrix params, at a costlier
  optimizer step and higher peak memory; a convergence lever, not a per-step one
  ([Muon](../../agent-docs/optimization/muon-optimizer.md)).

## Memory flow (cut peak / fit longer seq / bigger batch)

- **Long-seq OOM at the loss →** **FusedLinearCrossEntropy**
  (`liger_kernel_config: {cross_entropy: false, fused_linear_cross_entropy: true}` — sub-keys of that
  dict, not top-level fields): never materializes the logits plane; SFT-only, no entropy logging, not
  CP-compatible ([Liger](../../agent-docs/optimization/liger-kernels.md)). GRPO: `use_chunked_grpo_logprobs`
  ([the logits wall](../../agent-docs/training-methods/async-grpo/performance.md#the-logits-wall)).
- **MoE activation OOM →** keep **GC on** and/or **raise the EP degree** (less expert memory per GPU).
- **Optimizer-state OOM →** AdamWBF16 (auto) already halves fp32 AdamW's state; beyond it
  `optim: flash_adamw` ([FlashAdamW](../../agent-docs/optimization/flash-adamw.md#when-to-use-which-optimizer)).
- **Exact fp32 on dense params with headroom →** `fp32_non_ep_params: true`
  ([BF16 optimizer](../../agent-docs/optimization/bf16-optimizer.md#master-weight-and-grad-reduce-options)).
- **Fit on a small/consumer GPU →** **QLoRA**, not under EP/TP ([PEFT](../../agent-docs/optimization/peft.md#quantized-training-qlora)).
- **Grad-accumulation runs with headroom →** `fsdp_defer_grad_sync: true` reduces once per step
  ([Deferred gradient reduce](../../agent-docs/parallelism/data-parallelism.md#deferred-gradient-reduce-fsdp_defer_grad_sync)).
- **Many-rank / multi-node grad precision (not memory) →** `fp32_grad_reduce: true`.

## Sources of truth

The `agent-docs/optimization/` pages own every figure and rule above; the code is the **ultimate**
authority when a page, this skill or memory disagrees with the `halo/` implementation or a fresh
benchmark. Find the module behind a lever in the [module map](../../agent-docs/reference/architecture.md).
Related skills: `data` (packing / padding-free / sharding), `checkpoints` (low-precision fp8/fp4 export).
