# Optimization

Memory reduction, throughput, and hardware utilization during training. The [GPU training theory](../reference/gpu-training-theory.md) page explains why each lever works; the pages below toggle them.

---

## Foundations

- **[GPU training theory](../reference/gpu-training-theory.md)** — The roofline, training-step bottlenecks, and why each kernel exists — every number measured on Blackwell (B300).
- **[Throughput benchmarks](throughput-benchmarks.md)** — tokens/s/GPU and achieved TFLOPS on B300 across parallelism modes, sequence lengths, and model sizes.
- **[Halo vs stock TRL](halo-vs-stock-trl.md)** — The same gpt-oss-20b run under upstream `trl.SFTTrainer` with native FSDP, at matched model, data, and kernels.

---

## Memory

- **[PEFT (LoRA)](peft.md)** — Train <5% of parameters at ~50% less memory, across EP and CP (attention-only under ETP; rejected under TP and PP) — with merge-back to a standalone checkpoint.
- **[BF16 optimizer](bf16-optimizer.md)** — AdamW with bf16 masters and stochastic rounding at 6 bytes/param. Auto-enabled with `bf16: true` on FSDP/EP/TP.
- **[FlashAdamW](flash-adamw.md)** — Quantized 8-bit states + 24-bit masters (~5 bytes/param) with identical convergence. Set `optim: flash_adamw`.
- **[Low-precision compute (FP8/FP4)](low-precision-moe-kernels.md)** — FP8/FP4 matmul over bf16 masters and checkpoints — a QAT oracle plus inference-memory export (`scripts/after_training/quantize_to_lowp.py`).

---

## Throughput

- **[Padding-free collator](padding-free-collator.md)** — Concatenates sequences into Flash-Attention blocks instead of padding, with boundary masking against cross-sample leakage.
- **[Flash attention](flash-attention.md)** — Auto-detected per architecture — FA4 on Blackwell (B200/B300), FA2 + FA3 on Hopper (H100/H200).
- **[Grouped GEMM](grouped-gemm.md)** — Batched MoE expert matmul via `torch.nn.functional.grouped_mm`, default on SM90+. Home to the atomic-free permute and MoE throughput playbook.
- **[Liger kernels](liger-kernels.md)** — Triton kernels for fused cross-entropy, RMS normalization, and SwiGLU. On by default — set `use_liger_kernel: false` to disable.
- **[torch.compile](torch-compile.md)** — Inductor compilation of the spans between MoE/attention graph breaks (`torch_compile: true`) — about the same gain as Liger; the two overlap, so stacking adds little.
- **[Muon optimizer](muon-optimizer.md)** — Newton-Schulz orthogonalization for matrix params, fused Triton (~5× over upstream's Python-loop step). Set `optim: muon`.
