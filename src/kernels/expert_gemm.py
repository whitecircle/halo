"""Grouped GEMM for MoE expert FFNs, by quantization format: bf16 (the production path) on the
:mod:`src.kernels.grouped_mm` primitive, a block-scaled format on the fake-quant oracle or the opt-in
native DeepGEMM kernel. Master weights stay bf16/fp32.
"""

from __future__ import annotations

import torch

from src.kernels.grouped_mm import grouped_mm
from src.kernels.lowp.deepgemm import DEEPGEMM_FORMATS, deepgemm_grouped_gemm, use_deepgemm
from src.kernels.lowp.quantization import cached_fake_quant, fake_quant
from src.kernels.lowp.spec import QuantFormat


def expert_gemm(
    mat_a: torch.Tensor,
    mat_b: torch.Tensor,
    *,
    offs: torch.Tensor | None = None,
    quant_format: QuantFormat | None = None,
    weight_cacheable: bool = True,
) -> torch.Tensor:
    """``[total_tokens, K]`` activations stacked by expert times ``[num_experts, K, N]`` expert weights,
    ``offs`` the ``[num_experts]`` int32 cumulative token counts.

    ``quant_format`` None is bf16. A block-scaled format fake-quantizes both operands (a QAT oracle,
    slower than bf16), or runs native DeepGEMM under ``HALO_DEEPGEMM_NATIVE=1`` when the shape
    qualifies. ``weight_cacheable`` is False for FSDP2-managed weights: FSDP2 pins the unsharded param's
    version counter, so a cache keyed on it would serve the step-0 quantization forever. Expert biases
    are added by ``MoEExpertBiasGather``, not here.
    """
    if quant_format is None:
        return grouped_mm(mat_a, mat_b, offs=offs)
    if (
        quant_format in DEEPGEMM_FORMATS
        and offs is not None
        and use_deepgemm(mat_a.shape[0] / max(mat_b.shape[0], 1), mat_b.shape[2])
    ):
        return deepgemm_grouped_gemm(
            mat_a, mat_b, offs=offs, quant_format=quant_format, weight_cacheable=weight_cacheable
        )
    quantize_weight = cached_fake_quant if weight_cacheable else fake_quant
    # mat_b is [E, K, N]: the contraction is axis 1.
    return grouped_mm(
        fake_quant(mat_a, quant_format, axis=-1), quantize_weight(mat_b, quant_format, axis=1), offs=offs
    )
