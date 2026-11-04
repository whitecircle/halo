# Optimization

Memory reduction, throughput, and hardware utilization during training. The [GPU training theory](../reference/gpu-training-theory.md) page explains why each lever works; the pages below toggle them.

---

## Foundations

- **[GPU training theory](../reference/gpu-training-theory.md)** — The roofline, training-step bottlenecks, and why each kernel exists — every number measured on Blackwell (B300).
- **[Throughput benchmarks](throughput-benchmarks.md)** — 1.2.0 against 1.1.0 on twelve shapes, then tokens/s/GPU and achieved TFLOPS on B300 across parallelism modes, sequence lengths, and model sizes.
- **[Halo vs stock TRL](halo-vs-stock-trl.md)** — The same gpt-oss-20b run under upstream `trl.SFTTrainer` with native FSDP, at matched model, data, and kernels.

---

## Memory

- **[PEFT (LoRA)](peft.md)** — Train a few percent of parameters at ~50% less memory, across EP and CP (attention-only under ETP; rejected under TP) — with merge-back to a standalone checkpoint.
- **[BF16 optimizer](bf16-optimizer.md)** — AdamW with bf16 masters and stochastic rounding at 6 bytes/param. Auto-enabled with `bf16: true` on FSDP/EP/TP.
- **[FlashAdamW](flash-adamw.md)** — Quantized 8-bit states + 24-bit masters (~5 bytes/param) with AdamW-matching convergence. Set `optim: flash_adamw`.
- **[Low-precision compute (FP8/FP4)](low-precision-moe-kernels.md)** — FP8/FP4 matmul over bf16 masters and checkpoints — a QAT oracle plus a block-scaled weight export (`scripts/after_training/quantize_to_lowp.py`).
- **[Allocator](throughput-benchmarks.md#allocator-expandable-segments)** — `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` lowers peak reserved memory on dense runs at unchanged throughput; the dense SFT, DPO and SMPO examples launch with it, an EP run sets it only to fit.

---

## Throughput

- **[Padding-free collator](padding-free-collator.md)** — Concatenates sequences into Flash-Attention blocks instead of padding, with boundary masking against cross-sample leakage.
- **[Tree packing](tree-packing.md)** — Rows that share a prefix store it once and branch after it; each example trains as it does packed flat, at fewer tokens.
- **[Flash attention](flash-attention.md)** — Auto-detected per architecture — FA4 on Blackwell (B200/B300), FA3 on Hopper (H100/H200), FA2 where neither imports.
- **[Grouped GEMM](grouped-gemm.md)** — Batched MoE expert matmul via `torch.nn.functional.grouped_mm`, default on SM90+. Home to the atomic-free permute and MoE throughput playbook.
- **[Liger kernels](liger-kernels.md)** — Triton kernels for fused cross-entropy, RMS normalization, and SwiGLU (Qwen3 MoE, GLM-4.7-Flash, GPT-OSS and Gemma 4 run their norms on torch's fused `F.rms_norm`). On by default — set `use_liger_kernel: false` to disable.
- **[torch.compile](torch-compile.md)** — Inductor compilation of the spans between MoE/attention graph breaks (`torch_compile: true`) — at most 1% on EP MoE, and slower in `reduce-overhead` mode.
- **[Checkpointing plan](../training-methods/callbacks.md#gradientcheckpointingplancallback)** — `gradient_checkpointing_plan: auto` measures one fully checkpointed step and checkpoints only the decoder layers the memory budget needs (opt-in, packed SFT rows).
- **[Muon optimizer](muon-optimizer.md)** — Newton-Schulz orthogonalization for matrix params, fused Triton (~5× over upstream's Python-loop step). Set `optim: muon`.
- **[NUMA binding](throughput-benchmarks.md#numa-binding)** — `torchrun --numa-binding node` runs each rank on its GPU's NUMA node, which helps host-bound rows on a multi-socket host. `torchrun` launches only.

---

## Levers that do not help at these shapes

Each of these ran on B300 and did not pay off; its page holds the shapes.

- **Halo's fp8/fp4 paths** — QAT, not a speedup: whole steps run at 0.10–0.54× bf16 at ep8: [Why bf16 is the default](low-precision-moe-kernels.md#why-bf16-is-the-default).
- **DeepGEMM adapter** — whole steps at 0.37–0.54× bf16, since casts and row copies cost more than the kernel saves: [DeepGEMM](low-precision-moe-kernels.md#deepgemm-the-native-fp8fp4-path).
- **torch.compile on EP MoE** — at most 1%, and `reduce-overhead` is slower: [Benchmark results](torch-compile.md#benchmark-results).
- **Sub-bf16 master weights** — parameters, checkpoints and optimizer state never go below bf16; bf16 with stochastic rounding is the floor: [BF16 optimizer](bf16-optimizer.md).
- **FA2 on Blackwell where FA4 runs the model** — the slow outlier against FA4 and SDPA at head_dim 128: [FA4 vs FA2 vs SDPA](flash-attention.md#fa4-vs-fa2-vs-sdpa-on-blackwell). The defaults land on FA2 on purpose for GLM-4.7-Flash, Gemma 2/3 at head_dim 256 and softcapped rows that reach FA4's varlen kernels; leave those on it. Packed head_dim-256 GatedDeltaNet rows (Qwen3.5/3.6, Qwen3-Next) default to FA2 as well, where FA4 gains under 5% ([Model-specific handling](flash-attention.md#model-specific-handling)).
- **MoE throughput measured at batch 1** — noisy from run to run; measure at batch ≥ 4 ([Liger benchmarks](liger-kernels.md#benchmarks)).
