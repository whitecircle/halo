"""Frozen expert weights kept in a release's block-scaled format (NVFP4 or MXFP8) and dequantized to bf16 per GEMM.

A bank holds one projection for this rank's experts in ``F.linear`` convention (``[E, N, K]``, blocks along ``K``).
The grouped GEMM dequantizes it for the call and again in backward instead of saving the bf16 weight, and gives the
weight no gradient: a bank is the frozen base under LoRA."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch.autograd import Function

from halo.kernels.grouped_mm import grouped_mm_grads
from halo.kernels.lowp.dequant_triton import FUSED_DEQUANT_BLOCK, dequantize_bank_triton
from halo.kernels.lowp.quantization import FORMAT_BLOCK_SIZE, BlockScaledTensor, dequantize
from halo.kernels.lowp.spec import QuantFormat

# ``dequantize`` holds int64 nibble indices and fp32 elements, ~15x its bf16 output, so it runs a few experts at a time.
_DEQUANT_EXPERT_CHUNK = 4


@dataclass(frozen=True)
class FrozenExpertBank:
    """``data``: ``[E, N, K/2]`` uint8 nibbles (NVFP4) or ``[E, N, K]`` e4m3 (MXFP8); ``scales``: one per block along
    ``K``; ``global_scale``: NVFP4's fp32 per-expert scale ``[E]``."""

    quant_format: QuantFormat
    data: torch.Tensor
    scales: torch.Tensor
    global_scale: torch.Tensor | None = None

    def __post_init__(self) -> None:
        if self.quant_format not in ("nvfp4", "mxfp8"):
            raise ValueError(f"frozen expert banks hold nvfp4 or mxfp8, got {self.quant_format!r}")
        expected_scales = (*self.data.shape[:-1], self.in_features // FORMAT_BLOCK_SIZE[self.quant_format])
        expected_global = (self.num_experts,) if self.quant_format == "nvfp4" else None
        actual_global = None if self.global_scale is None else tuple(self.global_scale.shape)
        if tuple(self.scales.shape) != expected_scales or actual_global != expected_global:
            raise ValueError(
                f"{self.quant_format} bank of data {list(self.data.shape)} needs scales {list(expected_scales)} and "
                f"global scale {expected_global}, got {list(self.scales.shape)} and {actual_global}"
            )

    @property
    def num_experts(self) -> int:
        return self.data.shape[0]

    @property
    def out_features(self) -> int:
        return self.data.shape[1]

    @property
    def in_features(self) -> int:
        return self.data.shape[2] * (2 if self.quant_format == "nvfp4" else 1)

    def dequantize_linear(self, start: int = 0, end: int | None = None) -> torch.Tensor:
        """Experts ``[start, end)`` as bf16 ``[E, N, K]``."""
        end = self.num_experts if end is None else end
        # One fused kernel on CUDA, bit-identical to the eager path below.
        if self.data.is_cuda and self.in_features % FUSED_DEQUANT_BLOCK == 0:
            with torch.profiler.record_function(f"frozen_bank.dequantize_fused[{self.quant_format}]"):
                global_scale = None if self.global_scale is None else self.global_scale[start:end]
                return dequantize_bank_triton(
                    self.data[start:end], self.scales[start:end], global_scale, self.quant_format
                )
        packed = self.quant_format == "nvfp4"
        out = torch.empty(
            (end - start, self.out_features, self.in_features), dtype=torch.bfloat16, device=self.data.device
        )
        with torch.profiler.record_function(f"frozen_bank.dequantize[{self.quant_format}]"):
            for lo in range(start, end, _DEQUANT_EXPERT_CHUNK):
                hi = min(lo + _DEQUANT_EXPERT_CHUNK, end)
                chunk = BlockScaledTensor(
                    data=self.data[lo:hi],
                    scales=self.scales[lo:hi],
                    block_size=FORMAT_BLOCK_SIZE[self.quant_format],
                    axis=-1,
                    packed=packed,
                    pow2_scale=not packed,
                    global_scale=None if self.global_scale is None else self.global_scale[lo:hi].view(-1, 1, 1),
                )
                out[lo - start : hi - start] = dequantize(chunk)
        return out


class _FrozenBankGroupedMM(Function):
    @staticmethod
    def forward(ctx, mat_a: torch.Tensor, offs: torch.Tensor | None, bank: FrozenExpertBank) -> torch.Tensor:
        offs = None if offs is None else offs.to(torch.int32)
        ctx.bank = bank
        ctx.save_for_backward(offs)
        return F.grouped_mm(mat_a, bank.dequantize_linear().to(mat_a.dtype).transpose(1, 2), offs=offs)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        (offs,) = ctx.saved_tensors
        weight = ctx.bank.dequantize_linear().to(grad_output.dtype).transpose(1, 2)
        grad_a, _ = grouped_mm_grads(None, weight, grad_output, offs, needs_a=ctx.needs_input_grad[0], needs_b=False)
        return grad_a, None, None


def frozen_bank_grouped_mm(mat_a: torch.Tensor, bank: FrozenExpertBank, *, offs: torch.Tensor | None) -> torch.Tensor:
    """``[total_tokens, K]`` activations stacked by expert times the bank's ``[E, K, N]`` weights."""
    return _FrozenBankGroupedMM.apply(mat_a, offs, bank)
