"""Fused Triton MX fake quantization: one kernel per format for the quantize-dequantize round trip that
:func:`~halo.kernels.lowp.quantization.fake_quant` takes on CUDA, contiguous, last-axis tensors, bit-identical to
the eager :func:`~halo.kernels.lowp.quantization._block_round_trip`."""

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from triton.runtime.jit import JITFunction

# Blocks per program: fastest on 4 of 5 weight and activation shapes in a 16–256 sweep on B300 (SM103)
_BPP = 128


@triton.jit
def _mul_rn(a, b):
    """IEEE fp32 multiply with subnormals preserved. Triton compiles arithmetic/libdevice with
    flush-to-zero; PyTorch's CUDA kernels don't, so the scaling path must be explicit PTX."""
    return tl.inline_asm_elementwise(
        "mul.rn.f32 $0, $1, $2;", "=r,r,r", [a, b], dtype=tl.float32, is_pure=True, pack=1
    )


@triton.jit
def _pow2(e):
    """int32 e in [-127, 127] -> fp32 2^e, built from bits so 2^-127 (subnormal) is exact."""
    bits = tl.where(e == -127, 0x00400000, (e + 127) << 23)
    return bits.to(tl.float32, bitcast=True)


@triton.jit
def _load_blocks(x_ptr, n_blocks, BPP: tl.constexpr, BLOCK: tl.constexpr):
    """This program's ``BPP`` blocks of ``BLOCK`` elements as fp32 ``[BPP, BLOCK]``, with their offsets and mask."""
    blk = tl.program_id(0).to(tl.int64) * BPP + tl.arange(0, BPP)
    lane = tl.arange(0, BLOCK)
    offs = blk[:, None] * BLOCK + lane[None, :]
    mask = (blk < n_blocks)[:, None] & (lane < BLOCK)[None, :]
    x = tl.load(x_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    return x, offs, mask


@triton.jit
def _e8m0_exponent(x, FMAX: tl.constexpr, TINY: tl.constexpr):
    """Per-block shared exponent, as the eager ``_e8m0_scale``: ``ceil(log2(max(amax, TINY) / FMAX))`` clamped to
    e8m0's [-127, 127]. Torch on CUDA divides by a Python scalar as a multiply by its fp32 reciprocal, which can land
    one ulp from a true division and move the exponent, so the kernel does the same."""
    amax = tl.max(tl.abs(x), axis=1)
    y = _mul_rn(tl.maximum(amax, TINY), tl.math.div_rn(tl.full([], 1.0, tl.float32), FMAX))
    se = libdevice.ceil(libdevice.log2(y))
    return tl.minimum(tl.maximum(se, -127.0), 127.0).to(tl.int32)


@triton.jit(do_not_specialize=["n_blocks"])
def _mxfp8_fake_quant_kernel(
    x_ptr, out_ptr, n_blocks, BPP: tl.constexpr, BLOCK: tl.constexpr, FMAX: tl.constexpr, TINY: tl.constexpr
):
    """Each program takes ``BPP`` blocks: ``e8m0`` exponent, round to ``e4m3``, rescale, all in registers, so the
    tensor is read and written once. NaN blocks and subnormal scales match eager; the GPU test pins that on
    adversarial inputs."""
    x, offs, mask = _load_blocks(x_ptr, n_blocks, BPP, BLOCK)
    has_nan = tl.sum((x != x).to(tl.int32), axis=1) > 0
    se_i = _e8m0_exponent(x, FMAX, TINY)

    v = _mul_rn(x, _pow2(-se_i)[:, None])
    v = tl.minimum(tl.maximum(v, -FMAX), FMAX)
    q = v.to(tl.float8e4nv).to(tl.float32)
    out = _mul_rn(q, _pow2(se_i)[:, None])
    out = tl.where(has_nan[:, None], float("nan"), out)
    tl.store(out_ptr + offs, out.to(tl.bfloat16).to(out_ptr.dtype.element_ty), mask=mask)


@triton.jit(do_not_specialize=["n_blocks"])
def _mxfp4_fake_quant_kernel(
    x_ptr, out_ptr, n_blocks, BPP: tl.constexpr, BLOCK: tl.constexpr, FMAX: tl.constexpr, TINY: tl.constexpr
):
    """As the mxfp8 kernel, rounding to ``e2m1`` instead. e2m1 has one mantissa bit, so a magnitude in
    ``[2^e, 2^(e+1))`` (``e >= 0``, and everything below 1 with ``e = 0``) sits on a grid of step ``2^(e-1)``:
    ``rint`` of the magnitude over that step is round-to-nearest-even, the levels eager reaches with ``bucketize`` and
    its ties-up table. The sign comes from ``scaled < 0`` as in eager, so -0.0 returns +0.0 and a negative that rounds
    to zero returns -0.0."""
    x, offs, mask = _load_blocks(x_ptr, n_blocks, BPP, BLOCK)
    has_nan = tl.sum((x != x).to(tl.int32), axis=1) > 0
    se_i = _e8m0_exponent(x, FMAX, TINY)

    v = _mul_rn(x, _pow2(-se_i)[:, None])
    mag = tl.minimum(tl.abs(v), FMAX)
    e = tl.maximum(((mag.to(tl.int32, bitcast=True) >> 23) & 0xFF) - 127, 0)
    level = _mul_rn(libdevice.rint(_mul_rn(mag, _pow2(1 - e))), _pow2(e - 1))
    # A multiply, as in eager's ``level x sign``: Triton lowers a float negation to ``0 - x``, which loses -0.0.
    q = _mul_rn(level, tl.where(v < 0, -1.0, 1.0))
    out = _mul_rn(q, _pow2(se_i)[:, None])
    out = tl.where(has_nan[:, None], float("nan"), out)
    tl.store(out_ptr + offs, out.to(tl.bfloat16).to(out_ptr.dtype.element_ty), mask=mask)


def _launch(
    kernel: JITFunction, t: torch.Tensor, block_size: int, format_max: float, amax_floor: float
) -> torch.Tensor:
    """Run ``kernel`` over ``t`` in blocks of ``block_size`` into a new tensor of the same shape and dtype."""
    if t.numel() == 0:
        return torch.empty_like(t)
    out = torch.empty_like(t)
    n_blocks = t.numel() // block_size
    kernel[(triton.cdiv(n_blocks, _BPP),)](
        t, out, n_blocks, BPP=_BPP, BLOCK=block_size, FMAX=format_max, TINY=amax_floor
    )
    return out


def fake_quant_mxfp8_triton(t: torch.Tensor, block_size: int, format_max: float, amax_floor: float) -> torch.Tensor:
    """Last-axis, contiguous, CUDA. Caller guarantees ``t.shape[-1] % block_size == 0`` and passes the eager
    quantizer's e4m3 maximum and amax floor, so both paths scale by the same constants."""
    return _launch(_mxfp8_fake_quant_kernel, t, block_size, format_max, amax_floor)


def fake_quant_mxfp4_triton(t: torch.Tensor, block_size: int, format_max: float, amax_floor: float) -> torch.Tensor:
    """As :func:`fake_quant_mxfp8_triton`, with the eager quantizer's e2m1 maximum."""
    return _launch(_mxfp4_fake_quant_kernel, t, block_size, format_max, amax_floor)
