"""Triton casts of DeepGEMM's grouped operands, bit-identical on finite inputs to its reference casts with UE8M0
scales: ``per_token_cast_to_fp8`` (1x128) and ``per_channel_cast_to_fp8`` (128x1) through a row map that writes the
128-aligned expert layout in the same pass, ``per_block_cast_to_fp8`` (128x128) of the ``[E, K, N]`` weight in its own
layout and ``per_token_cast_to_fp4`` (1x32 MXFP4) of it as the ``[E, N, K]`` 4-bit operand. A gradient is cast without
the reference's 1e-4 floor on a group's largest element (``GRADIENT_AMAX_FLOOR``)."""

import torch
import triton
import triton.language as tl

from halo.kernels.lowp.fake_quant_triton import mul_rn

TILE = 128  # DeepGEMM's fp8 scale granularity along either axis
# Fastest on DeepSeek-V3's EP4 operands on B300 (SM103): 16 rows on one warp (1x128), 64 columns on four (128x1).
_ROWS_PER_PROGRAM, _ROWS_WARPS = 16, 1
_COLS_PER_PROGRAM, _COLS_WARPS = 64, 4
_BLOCK_WARPS = 4  # 0.41 ms on a [64, 7168, 2048] bank on B300 (SM103), 4.6 TB/s
# 1.1 ms on a [64, 7168, 2048] bank on B300: one 32-wide group of 32 rows per warp.
MXFP4_GROUP, _FP4_BK, _FP4_BN, _FP4_WARPS = 32, 32, 32, 1
# The reference casts scale a group by max(amax, 1e-4): right for activations and weights, but a gradient group whose
# largest element is under 1e-7 would then land in e4m3's subnormals, and under 5e-10 round to zero. A mean-reduced
# loss puts expert-output gradients there, so their casts scale by the group's own amax.
REFERENCE_AMAX_FLOOR, GRADIENT_AMAX_FLOOR = 1e-4, 0.0


@triton.jit
def _ue8m0_scale(amax, FLOOR: tl.constexpr, FMAX: tl.constexpr = 448.0):
    """``ceil_to_ue8m0(max(amax, FLOOR) / FMAX)`` and its reciprocal, both powers of two built from the exponent
    bits; the exponent clamp keeps an all-zero group's scale a normal float."""
    bits = tl.math.div_rn(tl.maximum(amax, FLOOR), FMAX).to(tl.int32, bitcast=True)
    exp = ((bits >> 23) & 0xFF) + ((bits & 0x7FFFFF) != 0).to(tl.int32)
    exp = tl.minimum(tl.maximum(exp, 1), 254)
    return (exp << 23).to(tl.float32, bitcast=True), ((254 - exp) << 23).to(tl.float32, bitcast=True)


# The activation casts and the MXFP4 weight cast are specialized like the weight cast below: every shape and stride
# they see is fixed per projection, and the row count is a multiple of 128, so a run compiles each once.
@triton.jit
def _rows_kernel(
    x_ptr,
    src_ptr,
    out_ptr,
    sf_ptr,
    rows,
    k,
    width,
    x_stride,
    ROWS: tl.constexpr,
    TILE: tl.constexpr,
    FLOOR: tl.constexpr,
):
    r = tl.program_id(0) * ROWS + tl.arange(0, ROWS)
    c = tl.program_id(1) * TILE + tl.arange(0, TILE)
    src = tl.load(src_ptr + r, mask=r < rows, other=-1).to(tl.int64)
    x = tl.load(
        x_ptr + src[:, None] * x_stride + c[None, :], mask=(src >= 0)[:, None] & (c < k)[None, :], other=0.0
    ).to(tl.float32)
    scale, inv = _ue8m0_scale(tl.max(tl.abs(x), axis=1), FLOOR)
    q = mul_rn(x, inv[:, None]).to(tl.float8e4nv)
    r64 = r.to(tl.int64)
    tl.store(out_ptr + r64[:, None] * width + c[None, :], q, mask=(r < rows)[:, None] & (c < width)[None, :])
    tl.store(sf_ptr + r64 * tl.num_programs(1) + tl.program_id(1), scale, mask=r < rows)


@triton.jit
def _tiles_kernel(
    x_ptr, src_ptr, out_ptr, sf_ptr, cols, x_stride, COLS: tl.constexpr, TILE: tl.constexpr, FLOOR: tl.constexpr
):
    r = (tl.program_id(0) * TILE + tl.arange(0, TILE)).to(tl.int64)
    c = tl.program_id(1) * COLS + tl.arange(0, COLS)
    src = tl.load(src_ptr + r).to(tl.int64)
    x = tl.load(
        x_ptr + src[:, None] * x_stride + c[None, :], mask=(src >= 0)[:, None] & (c < cols)[None, :], other=0.0
    ).to(tl.float32)
    scale, inv = _ue8m0_scale(tl.max(tl.abs(x), axis=0), FLOOR)
    q = mul_rn(x, inv[None, :]).to(tl.float8e4nv)
    tl.store(out_ptr + r[:, None] * cols + c[None, :], q, mask=(c < cols)[None, :])
    tl.store(sf_ptr + tl.program_id(0).to(tl.int64) * cols + c, scale, mask=c < cols)


# Specialized on its shape and strides, which a weight keeps all run: divisible by 16 and a unit inner stride, the
# loads vectorize, 8.5x faster on a DeepSeek-V3 bank.
@triton.jit
def _blocks_kernel(w_ptr, out_ptr, sf_ptr, k, n, s_e, s_k, s_n, TILE: tl.constexpr, FLOOR: tl.constexpr):
    """One 128x128 scale block of an expert's ``[K, N]`` weight, cast in place: a transposed fp8 store runs at a
    fifth of copy bandwidth, and the GEMMs read either major."""
    e = tl.program_id(2).to(tl.int64)
    rk = tl.program_id(1) * TILE + tl.arange(0, TILE)
    rn = tl.program_id(0) * TILE + tl.arange(0, TILE)
    mask = (rk < k)[:, None] & (rn < n)[None, :]
    w = tl.load(w_ptr + e * s_e + rk[:, None].to(tl.int64) * s_k + rn[None, :] * s_n, mask=mask, other=0.0)
    w = w.to(tl.float32)
    scale, inv = _ue8m0_scale(tl.max(tl.max(tl.abs(w), axis=1), axis=0), FLOOR)
    q = mul_rn(w, tl.broadcast_to(inv, (TILE, TILE))).to(tl.float8e4nv)
    tl.store(out_ptr + e * k * n + rk[:, None].to(tl.int64) * n + rn[None, :], q, mask=mask)
    tl.store(sf_ptr + (e * tl.num_programs(1) + tl.program_id(1)) * tl.num_programs(0) + tl.program_id(0), scale)


@triton.jit
def _e2m1_codes(x):
    """DeepGEMM's e2m1 rounding: ``|x|`` against the midpoints of {0, 0.5, 1, 1.5, 2, 3, 4, 6}, each tie to the even
    code (0.75, 1.75 and 3.5 up, the rest down), and a sign bit on nonzero codes only."""
    ax = tl.abs(x)
    code = (
        (ax > 0.25).to(tl.int32)
        + (ax >= 0.75).to(tl.int32)
        + (ax > 1.25).to(tl.int32)
        + (ax >= 1.75).to(tl.int32)
        + (ax > 2.5).to(tl.int32)
        + (ax >= 3.5).to(tl.int32)
        + (ax > 5.0).to(tl.int32)
    )
    return code | (((x < 0) & (code != 0)).to(tl.int32) << 3)


@triton.jit
def _fp4_rows_kernel(
    w_ptr,
    out_ptr,
    sf_ptr,
    nonfinite_ptr,
    k,
    n,
    k_pad,
    s_e,
    s_k,
    s_n,
    BK: tl.constexpr,
    BN: tl.constexpr,
    GROUP: tl.constexpr,
):
    """``BN`` rows of an expert's ``[N, K_pad]`` operand over ``BK`` of the contraction, read off its ``[K, N]``
    weight; columns past ``K`` are the zero pad. A block holding a NaN or inf sets ``nonfinite``."""
    e = tl.program_id(2).to(tl.int64)
    rk = tl.program_id(1) * BK + tl.arange(0, BK)
    rn = tl.program_id(0) * BN + tl.arange(0, BN)
    w = tl.load(
        w_ptr + e * s_e + rk[:, None].to(tl.int64) * s_k + rn[None, :] * s_n,
        mask=(rk < k)[:, None] & (rn < n)[None, :],
        other=0.0,
    )
    x = tl.reshape(tl.trans(w.to(tl.float32)), (BN, BK // GROUP, GROUP))
    # x - x is NaN exactly where x is NaN or inf; only such blocks touch the flag, so it sees no contention.
    tl.atomic_max(nonfinite_ptr, 1, mask=tl.sum(((x - x) != 0).to(tl.int32)) > 0)
    scale, inv = _ue8m0_scale(tl.max(tl.abs(x), axis=2), 1e-4, 6.0)
    codes = tl.reshape(
        _e2m1_codes(mul_rn(x, tl.broadcast_to(inv[:, :, None], (BN, BK // GROUP, GROUP)))), (BN, BK // 2, 2)
    )
    lo, hi = tl.split(codes)
    packed = ((lo & 0xF) | ((hi & 0xF) << 4)).to(tl.uint8).to(tl.int8, bitcast=True)
    rows = (e * n + rn).to(tl.int64)
    half = tl.program_id(1) * (BK // 2) + tl.arange(0, BK // 2)
    tl.store(out_ptr + rows[:, None] * (k_pad // 2) + half[None, :], packed, mask=(rn < n)[:, None])
    group = tl.program_id(1) * (BK // GROUP) + tl.arange(0, BK // GROUP)
    tl.store(sf_ptr + rows[:, None] * (k_pad // GROUP) + group[None, :], scale, mask=(rn < n)[:, None])


def cast_rows_to_fp8(
    x: torch.Tensor, src: torch.Tensor, width: int, amax_floor: float = REFERENCE_AMAX_FLOOR
) -> tuple[torch.Tensor, torch.Tensor]:
    """Row ``i`` of the ``[len(src), width]`` e4m3 output is row ``src[i]`` of ``x`` ``[T, K]``, zero where ``src[i]``
    is -1 and past ``K``, scaled per 1x128; scales are fp32 ``[len(src), ceil(width / 128)]``."""
    x = x.contiguous()
    rows, k = src.numel(), x.shape[1]
    out = x.new_empty(rows, width, dtype=torch.float8_e4m3fn)
    sf = x.new_empty(rows, triton.cdiv(width, TILE), dtype=torch.float32)
    grid = (triton.cdiv(rows, _ROWS_PER_PROGRAM), sf.shape[1])
    _rows_kernel[grid](
        x, src, out, sf, rows, k, width, x.stride(0),
        ROWS=_ROWS_PER_PROGRAM, TILE=TILE, FLOOR=amax_floor, num_warps=_ROWS_WARPS,
    )  # fmt: skip
    return out, sf


def cast_tiles_to_fp8(
    x: torch.Tensor, src: torch.Tensor, amax_floor: float = REFERENCE_AMAX_FLOOR
) -> tuple[torch.Tensor, torch.Tensor]:
    """The same row map, scaled per 128x1 over each 128-row tile instead; ``len(src)`` must be a multiple of 128.
    Scales are fp32 ``[len(src) / 128, C]``."""
    x = x.contiguous()
    rows, cols = src.numel(), x.shape[1]
    out = x.new_empty(rows, cols, dtype=torch.float8_e4m3fn)
    sf = x.new_empty(rows // TILE, cols, dtype=torch.float32)
    grid = (rows // TILE, triton.cdiv(cols, _COLS_PER_PROGRAM))
    _tiles_kernel[grid](
        x, src, out, sf, cols, x.stride(0),
        COLS=_COLS_PER_PROGRAM, TILE=TILE, FLOOR=amax_floor, num_warps=_COLS_WARPS,
    )  # fmt: skip
    return out, sf


def cast_weight_blocks_to_fp8(w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """``w`` ``[E, K, N]`` (any strides) as e4m3 ``[E, K, N]`` scaled per 128x128 block of each expert, and its fp32
    scales ``[E, ceil(K / 128), ceil(N / 128)]``."""
    experts, k, n = w.shape
    out = w.new_empty(experts, k, n, dtype=torch.float8_e4m3fn)
    sf = w.new_empty(experts, triton.cdiv(k, TILE), triton.cdiv(n, TILE), dtype=torch.float32)
    grid = (sf.shape[2], sf.shape[1], experts)
    _blocks_kernel[grid](w, out, sf, k, n, *w.stride(), TILE=TILE, FLOOR=REFERENCE_AMAX_FLOOR, num_warps=_BLOCK_WARPS)
    return out, sf


def cast_weight_rows_to_mxfp4(w: torch.Tensor, k_pad: int) -> tuple[torch.Tensor, torch.Tensor]:
    """``w`` ``[E, K, N]`` (any strides) as MXFP4 ``[E, N, k_pad / 2]`` int8 (two e2m1 codes a byte, low nibble first)
    over the contraction zero-padded to ``k_pad`` (a multiple of 128), and its fp32 scales ``[E, N, k_pad / 32]``.
    Raises on a NaN or inf weight: e2m1 has no NaN code and a NaN amax becomes a 2^127 scale, so the block would decode
    as finite zeros."""
    experts, k, n = w.shape
    out = w.new_empty(experts, n, k_pad // 2, dtype=torch.int8)
    sf = w.new_empty(experts, n, k_pad // MXFP4_GROUP, dtype=torch.float32)
    nonfinite = torch.zeros(1, device=w.device, dtype=torch.int32)
    grid = (triton.cdiv(n, _FP4_BN), k_pad // _FP4_BK, experts)
    _fp4_rows_kernel[grid](
        w, out, sf, nonfinite, k, n, k_pad, *w.stride(), BK=_FP4_BK, BN=_FP4_BN, GROUP=MXFP4_GROUP,
        num_warps=_FP4_WARPS,
    )  # fmt: skip
    if nonfinite.item():
        raise ValueError(f"Non-finite expert weight {tuple(w.shape)}: DeepGEMM's MXFP4 cast would hide it as zeros.")
    return out, sf
