# Optimization

Memory reduction, throughput, and hardware utilization during training. The [GPU training theory](../reference/gpu-training-theory.md) page explains why each lever works; the pages below toggle them.

---

## Foundations

<!-- markdownlint-disable MD030 -- mkdocs-material grid cards require the 4-space content indent -->

<div class="grid cards" markdown>

-   :material-school:{ .lg .middle } **GPU training theory**

    ---

    The roofline, training-step bottlenecks, and why each kernel exists — every number measured on Blackwell (B300).

    [:octicons-arrow-right-24: GPU training theory](../reference/gpu-training-theory.md)

-   :material-chart-line:{ .lg .middle } **Throughput benchmarks**

    ---

    tokens/s/GPU and achieved TFLOPS on B300 across parallelism modes, sequence lengths, and model sizes.

    [:octicons-arrow-right-24: Throughput benchmarks](throughput-benchmarks.md)

-   :material-scale-balance:{ .lg .middle } **Halo vs stock TRL**

    ---

    The same gpt-oss-20b run under upstream `trl.SFTTrainer` with native FSDP, at matched model, data, and kernels.

    [:octicons-arrow-right-24: Halo vs stock TRL](halo-vs-stock-trl.md)

</div>

---

## Memory

<div class="grid cards" markdown>

-   :material-puzzle:{ .lg .middle } **PEFT (LoRA)**

    ---

    Train <5% of parameters at ~50% less memory, across EP and CP (attention-only under ETP; rejected under TP and PP) — with merge-back to a standalone checkpoint.

    [:octicons-arrow-right-24: PEFT (LoRA)](peft.md)

-   :material-memory:{ .lg .middle } **BF16 optimizer**

    ---

    AdamW with bf16 masters and stochastic rounding at 6 bytes/param. Auto-enabled with `bf16: true` on FSDP/EP/TP.

    [:octicons-arrow-right-24: BF16 optimizer](bf16-optimizer.md)

-   :material-flash:{ .lg .middle } **FlashAdamW**

    ---

    Quantized 8-bit states + 24-bit masters (~5 bytes/param) with identical convergence. Set `optim: flash_adamw`.

    [:octicons-arrow-right-24: FlashAdamW](flash-adamw.md)

-   :material-package-down:{ .lg .middle } **Low-precision compute (FP8/FP4)**

    ---

    FP8/FP4 matmul over bf16 masters and checkpoints — a QAT oracle plus inference-memory export (`scripts/after_training/quantize_to_lowp.py`).

    [:octicons-arrow-right-24: Low-precision compute](low-precision-moe-kernels.md)

</div>

---

## Throughput

<div class="grid cards" markdown>

-   :material-arrow-collapse-horizontal:{ .lg .middle } **Padding-free collator**

    ---

    Concatenates sequences into Flash-Attention blocks instead of padding, with boundary masking against cross-sample leakage.

    [:octicons-arrow-right-24: Padding-free collator](padding-free-collator.md)

-   :material-lightning-bolt:{ .lg .middle } **Flash attention**

    ---

    Auto-detected per architecture — FA4 on Blackwell (B200/B300), FA2 + FA3 on Hopper (H100/H200).

    [:octicons-arrow-right-24: Flash attention](flash-attention.md)

-   :material-matrix:{ .lg .middle } **Grouped GEMM**

    ---

    Batched MoE expert matmul via `torch.nn.functional.grouped_mm`, default on SM90+. Home to the atomic-free permute and MoE throughput playbook.

    [:octicons-arrow-right-24: Grouped GEMM](grouped-gemm.md)

-   :material-chip:{ .lg .middle } **Liger kernels**

    ---

    Triton kernels for fused cross-entropy, RMS normalization, and SwiGLU. On by default — set `use_liger_kernel: false` to disable.

    [:octicons-arrow-right-24: Liger kernels](liger-kernels.md)

-   :material-cog-sync:{ .lg .middle } **torch.compile**

    ---

    Inductor compilation of the spans between MoE/attention graph breaks (`torch_compile: true`) — about the same gain as Liger; the two overlap, so stacking adds little.

    [:octicons-arrow-right-24: torch.compile](torch-compile.md)

-   :material-rotate-orbit:{ .lg .middle } **Muon optimizer**

    ---

    Newton-Schulz orthogonalization for matrix params, fused Triton (~5× over upstream's Python-loop step). Set `optim: muon`.

    [:octicons-arrow-right-24: Muon optimizer](muon-optimizer.md)

</div>
