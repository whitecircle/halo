"""Fused Triton fake quantization: one kernel per format for the quantize-dequantize round trip that
:func:`~halo.kernels.lowp.quantization.fake_quant` takes on CUDA, contiguous, last-axis tensors, bit-identical to
the eager :func:`~halo.kernels.lowp.quantization._block_round_trip` on CUDA, launched through
:func:`fake_quant_triton`. NVFP4 takes a tensor abs-max pass first, for its per-tensor global scale."""

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from triton.runtime.jit import JITFunction

# Blocks per program: fastest on 4 of 5 weight and activation shapes in a 16–256 sweep on B300 (SM103)
_BPP = 128
# Elements per program of the NVFP4 tensor abs-max pass, one atomic each.
_AMAX_CHUNK = 8192


@triton.jit
def _mul_rn(a, b):
    """IEEE fp32 multiply with subnormals preserved. Triton compiles arithmetic/libdevice with
    flush-to-zero; PyTorch's CUDA kernels don't, so the scaling path must be explicit PTX."""
    return tl.inline_asm_elementwise(
        "mul.rn.f32 $0, $1, $2;", "=r,r,r", [a, b], dtype=tl.float32, is_pure=True, pack=1
    )


@triton.jit
def _div_rn(a, b):
    """IEEE fp32 division with subnormals preserved, as PyTorch's CUDA kernels divide one tensor by another."""
    return tl.inline_asm_elementwise(
        "div.rn.f32 $0, $1, $2;", "=r,r,r", [a, b], dtype=tl.float32, is_pure=True, pack=1
    )


@triton.jit
def _pow2(e):
    """int32 e in [-127, 127] -> fp32 2^e, built from bits so 2^-127 (subnormal) is exact."""
    bits = tl.where(e == -127, 0x00400000, (e + 127) << 23)
    return bits.to(tl.float32, bitcast=True)


@triton.jit
def _cuda_exp2_of_integer(e):
    """fp32 ``exp2(e)`` for an integral or non-finite ``e``, bit for bit as PyTorch's CUDA ``exp2``: an exact power of
    two from 2^-149 to 2^127, except ``exp2(-127)``, which CUDA returns one ulp low (``0x3FFFFF``), then 0 below, inf
    above and NaN for NaN. Built from bits, as Triton's libdevice ``exp2`` flushes a subnormal result to 0."""
    ei = tl.minimum(tl.maximum(e, -150.0), 128.0).to(tl.int32)
    shift = tl.minimum(tl.maximum(ei + 149, 0), 22)
    sub = tl.where(ei == -127, 0x003FFFFF, tl.where(ei >= -149, 1 << shift, 0))
    out = tl.where(ei >= -126, (ei + 127) << 23, sub).to(tl.float32, bitcast=True)
    out = tl.where(e >= 128.0, float("inf"), out)
    return tl.where(e != e, float("nan"), out)


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


@triton.jit
def _e2m1_level(scaled, FMAX: tl.constexpr):
    """``scaled`` rounded to a signed ``e2m1`` level, as eager's ``bucketize`` with its ties-up table. e2m1 has one
    mantissa bit, so a magnitude in ``[2^e, 2^(e+1))`` (``e >= 0``, and everything below 1 with ``e = 0``) sits on a
    grid of step ``2^(e-1)``: ``rint`` of the magnitude over that step is round-to-nearest-even. The sign comes from
    ``scaled < 0`` as in eager, applied by a multiply: Triton lowers a float negation to ``0 - x``, which loses -0.0."""
    mag = tl.minimum(tl.abs(scaled), FMAX)
    e = tl.maximum(((mag.to(tl.int32, bitcast=True) >> 23) & 0xFF) - 127, 0)
    level = _mul_rn(libdevice.rint(_mul_rn(mag, _pow2(1 - e))), _pow2(e - 1))
    return _mul_rn(level, tl.where(scaled < 0, -1.0, 1.0))


@triton.jit
def _nvfp4_global_scale(amax, FMAX: tl.constexpr, SMAX: tl.constexpr, TINY: tl.constexpr):
    """``exp2(ceil(log2(max(amax, TINY) / (SMAX x FMAX))))``, as eager ``nvfp4_global_scale`` on CUDA: the scalar
    divide is a multiply by its fp32 reciprocal, and a NaN amax gives a NaN scale, as ``clamp`` keeps it. The quotient
    is subnormal below a tensor amax of about 3.2e-35, which libdevice's ``log2`` handles as PyTorch's does."""
    floored = tl.where(amax != amax, amax, tl.maximum(amax, TINY))
    y = _mul_rn(floored, tl.math.div_rn(tl.full([], 1.0, tl.float32), SMAX * FMAX))
    return _cuda_exp2_of_integer(libdevice.ceil(libdevice.log2(y)))


@triton.jit(do_not_specialize=["n_blocks"])
def mxfp8_fake_quant_kernel(
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
def mxfp4_fake_quant_kernel(
    x_ptr, out_ptr, n_blocks, BPP: tl.constexpr, BLOCK: tl.constexpr, FMAX: tl.constexpr, TINY: tl.constexpr
):
    """As the mxfp8 kernel, rounding to ``e2m1`` instead (:func:`_e2m1_level`), so -0.0 returns +0.0 and a negative
    that rounds to zero returns -0.0."""
    x, offs, mask = _load_blocks(x_ptr, n_blocks, BPP, BLOCK)
    has_nan = tl.sum((x != x).to(tl.int32), axis=1) > 0
    se_i = _e8m0_exponent(x, FMAX, TINY)

    q = _e2m1_level(_mul_rn(x, _pow2(-se_i)[:, None]), FMAX)
    out = _mul_rn(q, _pow2(se_i)[:, None])
    out = tl.where(has_nan[:, None], float("nan"), out)
    tl.store(out_ptr + offs, out.to(tl.bfloat16).to(out_ptr.dtype.element_ty), mask=mask)


@triton.jit(do_not_specialize=["n"])
def tensor_abs_max_bits_kernel(x_ptr, amax_ptr, n, CHUNK: tl.constexpr):
    """``max |x|`` over the whole tensor into ``amax_ptr`` (int32, zeroed by the caller) as fp32 bits, by atomic max.
    A non-negative fp32 orders like its bits, and a NaN's bits with the sign cleared sit above inf's, so one NaN
    anywhere wins, as in eager ``amax``. Reads the tensor once and materializes no ``|x|``."""
    offs = tl.program_id(0).to(tl.int64) * CHUNK + tl.arange(0, CHUNK)
    x = tl.load(x_ptr + offs, mask=offs < n, other=0.0).to(tl.float32)
    bits = x.to(tl.int32, bitcast=True) & 0x7FFFFFFF
    tl.atomic_max(amax_ptr, tl.max(bits, axis=0))


@triton.jit(do_not_specialize=["n_blocks"])
def nvfp4_fake_quant_kernel(
    x_ptr,
    out_ptr,
    amax_ptr,
    n_blocks,
    BPP: tl.constexpr,
    BLOCK: tl.constexpr,
    FMAX: tl.constexpr,
    TINY: tl.constexpr,
    SMAX: tl.constexpr,
):
    """NVFP4's round trip after :func:`tensor_abs_max_bits_kernel`, as eager ``_nvfp4_core`` and ``dequantize`` on
    CUDA, each divide the way eager does it: the global scale from the tensor amax (:func:`_nvfp4_global_scale`), the
    block scale ``amax / FMAX / global`` (a reciprocal multiply, then a true division) clamped at ``SMAX`` and rounded
    to ``e4m3``, ``e2m1`` codes from ``x / (scale x global)`` (true division, a zero divisor replaced by 1), and the
    output ``level x scale x global``. A NaN reaches the tensor amax, so it makes the whole tensor NaN."""
    x, offs, mask = _load_blocks(x_ptr, n_blocks, BPP, BLOCK)
    amax_t = tl.load(amax_ptr).to(tl.float32, bitcast=True)
    global_scale = _nvfp4_global_scale(amax_t, FMAX, SMAX, TINY)

    amax = tl.max(tl.abs(x), axis=1)
    has_nan = tl.sum((x != x).to(tl.int32), axis=1) > 0
    amax = tl.where(has_nan, float("nan"), amax)  # Triton's max drops a NaN, eager's amax keeps it
    scale = _div_rn(_mul_rn(amax, tl.math.div_rn(tl.full([], 1.0, tl.float32), FMAX)), global_scale)
    scale = tl.minimum(scale, SMAX, propagate_nan=tl.PropagateNan.ALL)
    scale_f = _mul_rn(scale.to(tl.float8e4nv).to(tl.float32), global_scale)
    divisor = tl.where(scale_f == 0.0, 1.0, scale_f)

    q = _e2m1_level(_div_rn(x, divisor[:, None]), FMAX)
    out = _mul_rn(q, scale_f[:, None])
    tl.store(out_ptr + offs, out.to(tl.bfloat16).to(out_ptr.dtype.element_ty), mask=mask)


def fake_quant_triton(
    kernel: JITFunction,
    t: torch.Tensor,
    block_size: int,
    format_max: float,
    amax_floor: float,
    scale_max: float | None = None,
    out_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """``kernel``'s round trip of ``t`` in blocks of ``block_size``, into a new tensor of the same shape, in ``t``'s
    dtype unless ``out_dtype`` is given. Last-axis, contiguous, CUDA. The caller guarantees
    ``t.shape[-1] % block_size == 0`` and passes the eager quantizer's element maximum and amax floor, so both paths
    scale by the same constants. ``scale_max`` (NVFP4's ``E4M3_MAX``) marks a two-level format: a tensor abs-max pass
    into a device scalar runs first, with no host sync."""
    if t.numel() == 0:
        return torch.empty_like(t, dtype=out_dtype)
    out = torch.empty_like(t, dtype=out_dtype)
    n_blocks = t.numel() // block_size
    grid = (triton.cdiv(n_blocks, _BPP),)
    if scale_max is None:
        kernel[grid](t, out, n_blocks, BPP=_BPP, BLOCK=block_size, FMAX=format_max, TINY=amax_floor)
        return out
    amax_bits = torch.zeros(1, dtype=torch.int32, device=t.device)
    tensor_abs_max_bits_kernel[(triton.cdiv(t.numel(), _AMAX_CHUNK),)](t, amax_bits, t.numel(), CHUNK=_AMAX_CHUNK)
    kernel[grid](
        t, out, amax_bits, n_blocks, BPP=_BPP, BLOCK=block_size, FMAX=format_max, TINY=amax_floor, SMAX=scale_max
    )
    return out
