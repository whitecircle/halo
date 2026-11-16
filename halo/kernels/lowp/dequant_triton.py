"""Fused Triton dequantization of a block-scaled expert bank (NVFP4 or MXFP8) to bf16: one kernel reads the packed
weights and their block scales once and writes the bf16 weights, bit-identical to the eager
:func:`~halo.kernels.lowp.quantization.dequantize`."""

import torch
import triton
import triton.language as tl

from halo.kernels.lowp.fake_quant_triton import mul_rn, pow2
from halo.kernels.lowp.quantization import E8M0_BIAS, E8M0_NAN

# Output elements per program: a multiple of both formats' block sizes (16, 32) that divides 2304 and 4608.
FUSED_DEQUANT_BLOCK = 256
# A kernel reads a module-level value only as a constexpr.
_E8M0_BIAS = tl.constexpr(E8M0_BIAS)
_E8M0_NAN = tl.constexpr(E8M0_NAN)


@triton.jit
def _e2m1(code):
    """fp32 value of each e2m1 code (bit 3 the sign): 0, 0.5, 1, 1.5, 2, 3, 4, 6."""
    magnitude = code & 7
    exponent = magnitude >> 1
    mantissa = (magnitude & 1).to(tl.float32)
    value = tl.where(exponent == 0, mantissa * 0.5, (1.0 + mantissa * 0.5) * pow2(exponent - 1))
    # A multiply by the sign, as eager applies it: code 8 is -0.
    return value * tl.where(code >= 8, -1.0, 1.0)


# One binary per ``K``: a one-expert slice hands in a 1-row count or a global scale off 16-byte alignment, and
# neither compiles another.
@triton.jit(do_not_specialize=["rows_per_expert"], do_not_specialize_on_alignment=["global_ptr"])
def _nvfp4_bank_kernel(
    data_ptr, scales_ptr, global_ptr, out_ptr, rows_per_expert, K: tl.constexpr, BLOCK_K: tl.constexpr
):
    """Row ``row`` of ``[E * N, K]``, columns ``[col0, col0 + BLOCK_K)``: low nibble first, a fp8 scale per 16
    elements times the expert's fp32 global scale, as eager multiplies them."""
    row = tl.program_id(0).to(tl.int64)
    col0 = tl.program_id(1) * BLOCK_K
    packed = tl.load(data_ptr + row * (K // 2) + col0 // 2 + tl.arange(0, BLOCK_K // 2)).to(tl.int32)
    elements = tl.reshape(tl.join(_e2m1(packed & 0x0F), _e2m1(packed >> 4)), [BLOCK_K // 16, 16])
    scales = tl.load(scales_ptr + row * (K // 16) + col0 // 16 + tl.arange(0, BLOCK_K // 16)).to(tl.float32)
    scales = mul_rn(scales, tl.load(global_ptr + row // rows_per_expert))
    out = tl.reshape(mul_rn(elements, scales[:, None]), [BLOCK_K])
    tl.store(out_ptr + row * K + col0 + tl.arange(0, BLOCK_K), out.to(tl.bfloat16))


@triton.jit
def _mxfp8_bank_kernel(data_ptr, scales_ptr, out_ptr, K: tl.constexpr, BLOCK_K: tl.constexpr):
    """Row ``row`` of ``[E * N, K]``: e4m3 elements times an e8m0 power-of-two scale per 32, NaN for code 255."""
    row = tl.program_id(0).to(tl.int64)
    col0 = tl.program_id(1) * BLOCK_K
    elements = tl.load(data_ptr + row * K + col0 + tl.arange(0, BLOCK_K)).to(tl.float32)
    codes = tl.load(scales_ptr + row * (K // 32) + col0 // 32 + tl.arange(0, BLOCK_K // 32)).to(tl.int32)
    scales = tl.where(codes == _E8M0_NAN, float("nan"), pow2(codes - _E8M0_BIAS))
    out = tl.reshape(mul_rn(tl.reshape(elements, [BLOCK_K // 32, 32]), scales[:, None]), [BLOCK_K])
    tl.store(out_ptr + row * K + col0 + tl.arange(0, BLOCK_K), out.to(tl.bfloat16))


def supports_fused_dequant(data: torch.Tensor, in_features: int) -> bool:
    """Whether :func:`dequantize_bank_triton` takes a bank of packed ``data`` and ``in_features`` (``K``) columns:
    on CUDA, at ``K`` a multiple of :data:`FUSED_DEQUANT_BLOCK`."""
    return data.is_cuda and in_features % FUSED_DEQUANT_BLOCK == 0


def dequantize_bank_triton(
    data: torch.Tensor, scales: torch.Tensor, global_scale: torch.Tensor | None, quant_format: str
) -> torch.Tensor:
    """bf16 ``[E, N, K]`` of a bank's contiguous CUDA ``data`` (``[E, N, K/2]`` nibbles or ``[E, N, K]`` e4m3) and
    ``scales`` (one per block along ``K``); NVFP4 takes its per-expert ``global_scale`` ``[E]``. Raises off
    :func:`supports_fused_dequant`."""
    num_experts, rows_per_expert = data.shape[:2]
    in_features = data.shape[2] * (2 if quant_format == "nvfp4" else 1)
    if not supports_fused_dequant(data, in_features):
        raise ValueError(
            f"the fused dequant takes a CUDA bank at K a multiple of {FUSED_DEQUANT_BLOCK}, got K={in_features} on "
            f"{data.device}"
        )
    out = torch.empty((num_experts, rows_per_expert, in_features), dtype=torch.bfloat16, device=data.device)
    grid = (num_experts * rows_per_expert, in_features // FUSED_DEQUANT_BLOCK)
    if quant_format == "nvfp4":
        _nvfp4_bank_kernel[grid](
            data.contiguous(),
            scales.contiguous(),
            global_scale.float().contiguous(),
            out,
            rows_per_expert,
            K=in_features,
            BLOCK_K=FUSED_DEQUANT_BLOCK,
        )
    else:
        _mxfp8_bank_kernel[grid](
            data.contiguous(), scales.contiguous(), out, K=in_features, BLOCK_K=FUSED_DEQUANT_BLOCK
        )
    return out
