"""Opt-in low-precision (fp8/fp4) compute: block-scale quantizers, fake-quant linear/GEMM, the native
DeepGEMM backend, and the pre-FSDP apply layer. Masters and checkpoints stay bf16/fp32 — only GEMM
operands are cast, a QAT oracle for quantized serving. See ``agent-docs/optimization/low-precision-moe-kernels.md``.
"""
