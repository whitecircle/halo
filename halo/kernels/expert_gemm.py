"""Grouped GEMM for MoE expert FFNs, by quantization format: bf16 (the production path), or a block-scaled format on
the fake-quant oracle or the opt-in native DeepGEMM kernel."""

from __future__ import annotations

import torch

from halo.kernels.grouped_mm import grouped_mm
from halo.kernels.lowp.deepgemm import deepgemm_grouped_gemm, use_deepgemm
from halo.kernels.lowp.quantization import cached_fake_quant, fake_quant
from halo.kernels.lowp.spec import QuantFormat


def expert_gemm(
    mat_a: torch.Tensor,
    mat_b: torch.Tensor,
    *,
    offs: torch.Tensor | None = None,
    quant_format: QuantFormat | None = None,
    weight_cacheable: bool = True,
) -> torch.Tensor:
    """``[total_tokens, K]`` activations stacked by expert times ``[num_experts, K, N]`` weights. ``weight_cacheable``
    is False for FSDP2-managed weights, as FSDP2 pins the unsharded param's version counter and a cache keyed on it
    would serve the step-0 quantization forever."""
    if quant_format is None:
        return grouped_mm(mat_a, mat_b, offs=offs)
    if use_deepgemm(quant_format):
        return deepgemm_grouped_gemm(
            mat_a, mat_b, offs=offs, quant_format=quant_format, weight_cacheable=weight_cacheable
        )
    quantize_weight = cached_fake_quant if weight_cacheable else fake_quant
    # mat_b is [E, K, N]: the contraction is axis 1.
    return grouped_mm(
        fake_quant(mat_a, quant_format, axis=-1), quantize_weight(mat_b, quant_format, axis=1), offs=offs
    )
