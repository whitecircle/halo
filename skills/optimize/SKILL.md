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

Recommend levers to raise **tokens/s/GPU** (the headline metric — MFU/achieved-TFLOPS
are opt-in diagnostics from `EfficiencyCallback`, *not* the goal) or cut **peak memory**
for a concrete config. **[levers.md](levers.md)** holds every figure with its measured shape, each
row citing the `agent-docs/optimization/` page it comes from. This page carries the flow
only: read levers.md and quote the number from there, never from here.

## How to use this skill
1. Pin the **bottleneck**: throughput-bound (want more tok/s/GPU) or memory-bound (OOM /
   want longer seq / bigger batch)? For MoE/EP also ask: comm-bound (low top-k, e.g.
   gpt-oss) or permute/elementwise-bound (high top-k, e.g. qwen3.6)? When unsure, measure:
   `enable_torch_profiler: true` → `python scripts/profiling/trace_report.py` (TraceLens
   compute/comm/idle split; the `ep.dispatch`/`ep.combine` trace ranges give the all-to-all
   fraction directly).
2. Confirm the **shape**: dense vs MoE/EP, short vs long seq, fixed- vs variable-length data.
3. Recommend from the flow below; take the measured effect + the exact YAML/env flag from
   levers.md. If a lever's win is batch-1-noisy (all EP throughput), say "measure at batch ≥ 4".
4. Steer away from the **negatives** below — they are the common dead-ends.

## Already on by default (don't "enable" — they're free)
These fire automatically; mention them only to confirm, not as new advice. Figures in levers.md.
- **Grouped GEMM** — auto on SM90+ for MoE experts; the win is largest with many local experts per rank
  and narrows as batch (per-expert M) grows.
- **Liger CE + RMSNorm/RoPE/SwiGLU** — `use_liger_kernel: true` default; a double-digit-% throughput win
  and tens of GB, on dense and MoE alike.
- **FA4 on Blackwell** — auto-selected; the win over FA2 grows with sequence length and is small on MoE,
  where the step is expert-GEMM and all-to-all bound. FA2 is the slow outlier on B300.
- **AdamWBF16 + stochastic rounding** — auto from `bf16: true`; half the per-param state of fp32 AdamW at
  a loss curve that tracks the fp32 master. Auto-OFF under replicated DDP.
- **CDMC=1** — baked into the image env; free win on ep8, neutral dense/ep2.
- **Atomic-free expert permute** — auto for high-top_k MoE (`top_k ≥ ep_size`); win grows with sequence
  length. gpt-oss (top-4) stays on the cheaper `index_add_` path.

## Throughput flow (raise tok/s/GPU)
- **MoE/EP, any shape →** push **seq × batch as high as memory allows** first — EP at low token
  counts is comm-bound (fixed all-to-all), so **batch is the dominant EP lever**. Read achieved
  TFLOPS / tok-s, not plain MFU% (sparse MoE can't approach a dense MFU; longer seq + lower EP raise
  the achieved-TFLOPS ceiling).
- **MoE/EP, fits without GC →** turn **gradient checkpointing off** — the largest single throughput
  lever, at ~2× activation memory. Keep GC on only when you need the memory.
- **MoE, small model fits in FSDP →** prefer **FSDP (ep1) over EP** (no all-to-all): it keeps every param
  local and tops the achieved-TFLOPS table; use EP only when FSDP OOMs.
- **Pure EP on 8 GPUs →** use **ep2 (throughput) or ep8 (memory)** only. **ep4 is rejected at config time**;
  ep4 on exactly 4 GPUs is fine. For finer expert sharding use **EP+ETP** (`ep4+etp2`); EP+TP does
  not help. Rule and mechanism: the `parallelism` skill (`matrix.md`, row *Multi-group >2-rank EP on
  one NVLink domain*).
- **Dense, long context →** FA4's win over FA2 is largest here. Confirm it's selected.
- **Variable-length data (avg << max_len) →** **packing** is the big win; padding-free is the smaller
  one — use it when cross-sequence boundaries are undesirable. Uniform/long-seq data → all collators
  tie (~1%), use standard.
- **Convergence speed (fewer steps to target loss) →** `optim: muon` reaches a lower loss in the same
  step budget on matrix params, at a much costlier optimizer step and higher peak memory — far less
  end-to-end, measure on your model. A convergence lever, not a per-step throughput one.

## Memory flow (cut peak / fit longer seq / bigger batch)
- **Long-seq OOM at the loss →** **FusedLinearCrossEntropy**
  (`liger_kernel_config: {cross_entropy: false, fused_linear_cross_entropy: true}` — they are
  sub-keys of that dict, not top-level fields): never materializes the `batch×seq×vocab` logits, for
  tens of GB at 32k at near-CE throughput; SFT-only, disables entropy logging, not CP-compatible.
- **MoE activation OOM →** keep **GC on** (roughly half the GC-off activation footprint, and what makes
  32k fit at all — pure ep8 GC-off OOMs there) and/or **raise EP degree** (more ranks = less expert
  memory/GPU; ep8 keeps ~4.2B local vs ep2 ~11.4B for gpt-oss-20b).
- **Optimizer-state OOM →** AdamWBF16 (auto) is already half of fp32 AdamW's state. For more,
  `optim: flash_adamw` (quantized moments, convergence matches AdamW; needs `flashoptim`) — tens of GB
  at 70B+.
- **Want exact fp32 on dense params with headroom →** `fp32_non_ep_params: true` (dense params fp32,
  experts stay bf16+SR).
- **Fit on a small/consumer GPU →** **QLoRA**: far less memory than full FT and faster than bf16 LoRA
  (the 4-bit base is bandwidth-bound). With FLCE a Qwen3-8B 32k run fits a 24 GB GPU.
  EP→attention-only targets, no QLoRA under EP/TP.
- **Many-rank / multi-node grad precision (not memory) →** `fp32_grad_reduce: true`: a tighter
  grad-reduce at bf16 storage cost, ~2× cost on the reduce collective only.

## What does NOT help here (do not chase these)
The honest negatives — cited in full in levers.md. At fine-grained MoE shapes (EP8,
256–512 tokens/expert, N ≤ 4096) these are dead-ends:
- **Low-precision fp8/fp4 *compute* — no throughput win. Train bf16.** Per-expert GEMM is
  weight-bandwidth-bound at the bf16 roofline (proven roofline + Blackwell HW + measured DeepGEMM).
  Simulated fake-quant runs many times a bf16 step — a *convergence-validation* oracle, not a speed
  path. The real fp8/fp4 win is **inference memory** (`quantize_to_lowp.py`, fp8 = ½ / fp4 = ¼ expert
  bytes) — not training.
- **Native DeepGEMM — net-slower than bf16 at every training shape**; opt-in only
  (`HALO_DEEPGEMM_NATIVE=1`), never auto-selected.
- **torch.compile on EP MoE — reaches Liger's speedup but does NOT stack on top of it.** Compile and
  Liger target the same compilable spans between DeepEP/FA4 graph breaks, so the two stacked land within
  noise of Liger alone — and Liger is already on by default with no warmup cost. Reach for
  `torch_compile: true` only to fuse a span Liger doesn't cover. Not a free extra win.
- **Sub-bf16 master weights — dead.** Params, checkpoint and optimizer state never go below bf16; only
  GEMM operands are cast. bf16 + SR is the floor.
- **FA2 on Blackwell — the slow outlier** (well under half FA4's throughput at 32k). Use FA4 (auto);
  SDPA is the dense fallback.
- **MoE throughput levers measured at batch 1** — comm-bound, run-to-run noisy. Always **batch ≥ 4**.

## Sources of truth
Figures live in [levers.md](levers.md), each cited to its page in `agent-docs/optimization/`:
grouped-gemm.md, liger-kernels.md, flash-attention.md, bf16-optimizer.md, padding-free-collator.md,
throughput-benchmarks.md, low-precision-moe-kernels.md, muon-optimizer.md, flash-adamw.md,
torch-compile.md, peft.md. Read the cited doc page before quoting an exact number — and
the code is the **ultimate** authority: when a doc, this skill, or memory disagrees with the actual
`src/` implementation or a fresh benchmark, or you are unsure, trust the code/measurement and read it
before asserting. (`CLAUDE.md`: docs-first, the code wins.) Related skills: `data` (packing /
padding-free / sharding levers), `checkpoints` (low-precision fp8/fp4 export).
