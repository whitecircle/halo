"""Grouped GEMM op for MoE expert FFNs across precisions — what the EP and non-EP layers call.

Per-expert GEMMs with variable token counts via cumulative ``offs``, dispatched by precision: bf16 (the
production path) to the :mod:`src.kernels.grouped_mm_autograd` primitive, mxfp8/nvfp4 to the fake-quant
oracle or the opt-in native DeepGEMM under :mod:`src.kernels.lowp`. Masters stay bf16/fp32.
"""

from __future__ import annotations

from enum import Enum

import torch

from src.kernels.grouped_mm_autograd import grouped_mm as _bf16_grouped_mm
from src.kernels.lowp.deepgemm import DEEPGEMM_FORMATS, deepgemm_grouped_gemm, use_deepgemm
from src.kernels.lowp.quantization import cached_fake_quant, fake_quant


class GroupedGemmPrecision(str, Enum):
    """Compute precision for :func:`grouped_gemm` (the master weight stays bf16/fp32)."""

    BF16 = "bf16"
    MXFP8 = "mxfp8"  # e4m3 elements + e8m0 (1x32) block scales
    MXFP4 = "mxfp4"  # e2m1 elements + e8m0 (1x32) block scales — fast fp4
    NVFP4 = "nvfp4"  # e2m1 elements + e4m3 (1x16) block scales — most accurate fp4


def grouped_gemm(
    mat_a: torch.Tensor,
    mat_b: torch.Tensor,
    *,
    offs: torch.Tensor | None = None,
    precision: GroupedGemmPrecision = GroupedGemmPrecision.BF16,
    weight_cacheable: bool = True,
) -> torch.Tensor:
    """Grouped GEMM for MoE experts (EP and non-EP), at the requested precision.

    ``torch.nn.functional.grouped_mm`` over the shapes the MoE layers issue, with a correct
    Blackwell backward. No fused bias: the families carrying expert biases add them through
    ``MoEExpertBiasGather``, whose backward is an atomic-free GEMM rather than a bf16 atomic scatter.

    Args:
        mat_a: ``[total_tokens, K]`` activations (tokens stacked across experts).
        mat_b: ``[num_experts, K, N]`` expert weights.
        offs: ``[num_experts]`` int32 cumulative token counts per expert.
        precision: ``BF16`` (default, production), or ``MXFP8`` / ``NVFP4`` (block-scale fake-quant QAT
            oracle, slower than bf16; native DeepGEMM only when ``HALO_DEEPGEMM_NATIVE=1`` + shape qualifies).
        weight_cacheable: whether ``mat_b``'s quantization may be cached on its version counter. False
            for FSDP2-managed weights, whose unsharded param's counter is pinned by FSDP2 — a cache
            keyed on it would serve the step-0 quantization forever.
    """
    if precision is GroupedGemmPrecision.BF16:
        return _bf16_grouped_mm(mat_a, mat_b, offs=offs)

    fmt = precision.value
    # Native DeepGEMM fp8/nvfp4 — opt-in, never auto-selected (fp4 tolerates K-not-÷256 experts, the
    # adapter zero-pads K internally).
    if (
        fmt in DEEPGEMM_FORMATS
        and offs is not None
        and use_deepgemm(mat_a.shape[0] / max(mat_b.shape[0], 1), mat_b.shape[2])
    ):
        return deepgemm_grouped_gemm(mat_a, mat_b, offs=offs, precision=fmt, weight_cacheable=weight_cacheable)

    a_q = fake_quant(mat_a, fmt, axis=-1)
    quant_b = cached_fake_quant if weight_cacheable else fake_quant
    b_q = quant_b(mat_b, fmt, axis=1)  # mat_b is [E, K, N]; K is axis 1
    return _bf16_grouped_mm(a_q, b_q, offs=offs)
