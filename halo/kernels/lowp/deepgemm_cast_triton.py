"""Triton casts of DeepGEMM's grouped operands, bit-identical on finite inputs to its reference casts with UE8M0
scales: ``per_token_cast_to_fp8`` (1x128) and ``per_channel_cast_to_fp8`` (128x1) read the token rows through a row map,
so the 128-aligned expert layout is written in the same pass; ``per_block_cast_to_fp8`` (128x128) reads the
``[E, K, N]`` expert weight in place and writes its ``[E, N, K]`` operand, with no transposed or padded copy, as
``per_token_cast_to_fp4`` (1x32 MXFP4, two e2m1 codes per byte) does for the 4-bit weight."""

import torch
import triton
import triton.language as tl

from halo.kernels.lowp.fake_quant_triton import mul_rn

TILE = 128  # DeepGEMM's fp8 scale granularity along either axis
# Fastest on a [65536, 2880] bf16 operand on B300 (SM103): 8 rows on one warp, 32 columns on four.
_ROWS_PER_PROGRAM, _ROWS_WARPS = 8, 1
_COLS_PER_PROGRAM, _COLS_WARPS = 32, 4
_BLOCK_WARPS = 4  # 4.6x DeepGEMM's eager cast on a [64, 7168, 4096] bank on B300
# 9x DeepGEMM's eager MXFP4 cast on a [64, 7168, 4096] bank on B300.
MXFP4_GROUP, _FP4_BK, _FP4_BN, _FP4_WARPS = 32, 128, 32, 4


@triton.jit
def _ue8m0_scale(amax, FMAX: tl.constexpr = 448.0):
    """``ceil_to_ue8m0(max(amax, 1e-4) / FMAX)`` and its reciprocal, both powers of two built from the exponent bits."""
    bits = tl.math.div_rn(tl.maximum(amax, 1e-4), FMAX).to(tl.int32, bitcast=True)
    exp = ((bits >> 23) & 0xFF) + ((bits & 0x7FFFFF) != 0).to(tl.int32)
    exp = tl.minimum(tl.maximum(exp, 1), 254)
    return (exp << 23).to(tl.float32, bitcast=True), ((254 - exp) << 23).to(tl.float32, bitcast=True)


@triton.jit(do_not_specialize=["rows", "k", "width", "x_stride"])
def _rows_kernel(x_ptr, src_ptr, out_ptr, sf_ptr, rows, k, width, x_stride, ROWS: tl.constexpr, TILE: tl.constexpr):
    r = tl.program_id(0) * ROWS + tl.arange(0, ROWS)
    c = tl.program_id(1) * TILE + tl.arange(0, TILE)
    src = tl.load(src_ptr + r, mask=r < rows, other=-1).to(tl.int64)
    x = tl.load(
        x_ptr + src[:, None] * x_stride + c[None, :], mask=(src >= 0)[:, None] & (c < k)[None, :], other=0.0
    ).to(tl.float32)
    scale, inv = _ue8m0_scale(tl.max(tl.abs(x), axis=1))
    q = mul_rn(x, inv[:, None]).to(tl.float8e4nv)
    r64 = r.to(tl.int64)
    tl.store(out_ptr + r64[:, None] * width + c[None, :], q, mask=(r < rows)[:, None] & (c < width)[None, :])
    tl.store(sf_ptr + r64 * tl.num_programs(1) + tl.program_id(1), scale, mask=r < rows)


@triton.jit(do_not_specialize=["cols", "x_stride"])
def _tiles_kernel(x_ptr, src_ptr, out_ptr, sf_ptr, cols, x_stride, COLS: tl.constexpr, TILE: tl.constexpr):
    r = (tl.program_id(0) * TILE + tl.arange(0, TILE)).to(tl.int64)
    c = tl.program_id(1) * COLS + tl.arange(0, COLS)
    src = tl.load(src_ptr + r).to(tl.int64)
    x = tl.load(
        x_ptr + src[:, None] * x_stride + c[None, :], mask=(src >= 0)[:, None] & (c < cols)[None, :], other=0.0
    ).to(tl.float32)
    scale, inv = _ue8m0_scale(tl.max(tl.abs(x), axis=0))
    q = mul_rn(x, inv[None, :]).to(tl.float8e4nv)
    tl.store(out_ptr + r[:, None] * cols + c[None, :], q, mask=(c < cols)[None, :])
    tl.store(sf_ptr + tl.program_id(0).to(tl.int64) * cols + c, scale, mask=c < cols)


@triton.jit(do_not_specialize=["k", "n", "s_e", "s_k", "s_n"])
def _blocks_kernel(w_ptr, out_ptr, sf_ptr, k, n, s_e, s_k, s_n, TILE: tl.constexpr):
    """One 128x128 scale block of an expert's ``[K, N]`` weight, cast and written transposed."""
    e = tl.program_id(2).to(tl.int64)
    rk = tl.program_id(1) * TILE + tl.arange(0, TILE)
    rn = tl.program_id(0) * TILE + tl.arange(0, TILE)
    mask = (rk < k)[:, None] & (rn < n)[None, :]
    w = tl.load(w_ptr + e * s_e + rk[:, None].to(tl.int64) * s_k + rn[None, :] * s_n, mask=mask, other=0.0)
    w = w.to(tl.float32)
    scale, inv = _ue8m0_scale(tl.max(tl.max(tl.abs(w), axis=1), axis=0))
    q = tl.trans(mul_rn(w, tl.broadcast_to(inv, (TILE, TILE))).to(tl.float8e4nv))
    tl.store(out_ptr + e * n * k + rn[:, None].to(tl.int64) * k + rk[None, :], q, mask=tl.trans(mask))
    tl.store(sf_ptr + (e * tl.num_programs(0) + tl.program_id(0)) * tl.num_programs(1) + tl.program_id(1), scale)


@triton.jit
def _e2m1_codes(x):
    """DeepGEMM's e2m1 rounding: ``|x|`` against the midpoints of {0, 0.5, 1, 1.5, 2, 3, 4, 6}, every tie going down, and
    a sign bit on nonzero codes only."""
    ax = tl.abs(x)
    code = (
        (ax > 0.25).to(tl.int32)
        + (ax > 0.75).to(tl.int32)
        + (ax > 1.25).to(tl.int32)
        + (ax > 1.75).to(tl.int32)
        + (ax > 2.5).to(tl.int32)
        + (ax > 3.5).to(tl.int32)
        + (ax > 5.0).to(tl.int32)
    )
    return code | (((x < 0) & (code != 0)).to(tl.int32) << 3)


@triton.jit(do_not_specialize=["k", "n", "k_pad", "s_e", "s_k", "s_n"])
def _fp4_rows_kernel(
    w_ptr, out_ptr, sf_ptr, k, n, k_pad, s_e, s_k, s_n, BK: tl.constexpr, BN: tl.constexpr, GROUP: tl.constexpr
):
    """``BN`` rows of an expert's ``[N, K_pad]`` operand over ``BK`` of the contraction, read off its ``[K, N]``
    weight; columns past ``K`` are the zero pad."""
    e = tl.program_id(2).to(tl.int64)
    rk = tl.program_id(1) * BK + tl.arange(0, BK)
    rn = tl.program_id(0) * BN + tl.arange(0, BN)
    w = tl.load(
        w_ptr + e * s_e + rk[:, None].to(tl.int64) * s_k + rn[None, :] * s_n,
        mask=(rk < k)[:, None] & (rn < n)[None, :],
        other=0.0,
    )
    x = tl.reshape(tl.trans(w.to(tl.float32)), (BN, BK // GROUP, GROUP))
    scale, inv = _ue8m0_scale(tl.max(tl.abs(x), axis=2), 6.0)
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


def cast_rows_to_fp8(x: torch.Tensor, src: torch.Tensor, width: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Row ``i`` of the ``[len(src), width]`` e4m3 output is row ``src[i]`` of ``x`` ``[T, K]``, zero where ``src[i]``
    is -1 and past ``K``, scaled per 1x128; scales are fp32 ``[len(src), ceil(width / 128)]``."""
    x = x.contiguous()
    rows, k = src.numel(), x.shape[1]
    out = x.new_empty(rows, width, dtype=torch.float8_e4m3fn)
    sf = x.new_empty(rows, triton.cdiv(width, TILE), dtype=torch.float32)
    grid = (triton.cdiv(rows, _ROWS_PER_PROGRAM), sf.shape[1])
    _rows_kernel[grid](
        x, src, out, sf, rows, k, width, x.stride(0), ROWS=_ROWS_PER_PROGRAM, TILE=TILE, num_warps=_ROWS_WARPS
    )
    return out, sf


def cast_tiles_to_fp8(x: torch.Tensor, src: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """The same row map, scaled per 128x1 over each 128-row tile instead; ``len(src)`` must be a multiple of 128.
    Scales are fp32 ``[len(src) / 128, C]``."""
    x = x.contiguous()
    rows, cols = src.numel(), x.shape[1]
    out = x.new_empty(rows, cols, dtype=torch.float8_e4m3fn)
    sf = x.new_empty(rows // TILE, cols, dtype=torch.float32)
    grid = (rows // TILE, triton.cdiv(cols, _COLS_PER_PROGRAM))
    _tiles_kernel[grid](x, src, out, sf, cols, x.stride(0), COLS=_COLS_PER_PROGRAM, TILE=TILE, num_warps=_COLS_WARPS)
    return out, sf


def cast_weight_blocks_to_fp8(w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """``w`` ``[E, K, N]`` (any strides) as e4m3 ``[E, N, K]`` scaled per 128x128 block of each expert, and its fp32
    scales ``[E, ceil(N / 128), ceil(K / 128)]``."""
    experts, k, n = w.shape
    out = w.new_empty(experts, n, k, dtype=torch.float8_e4m3fn)
    sf = w.new_empty(experts, triton.cdiv(n, TILE), triton.cdiv(k, TILE), dtype=torch.float32)
    grid = (sf.shape[1], sf.shape[2], experts)
    _blocks_kernel[grid](w, out, sf, k, n, *w.stride(), TILE=TILE, num_warps=_BLOCK_WARPS)
    return out, sf


def cast_weight_rows_to_mxfp4(w: torch.Tensor, k_pad: int) -> tuple[torch.Tensor, torch.Tensor]:
    """``w`` ``[E, K, N]`` (any strides) as MXFP4 ``[E, N, k_pad / 2]`` int8 (two e2m1 codes a byte, low nibble first)
    over the contraction zero-padded to ``k_pad`` (a multiple of 128), and its fp32 scales ``[E, N, k_pad / 32]``."""
    experts, k, n = w.shape
    out = w.new_empty(experts, n, k_pad // 2, dtype=torch.int8)
    sf = w.new_empty(experts, n, k_pad // MXFP4_GROUP, dtype=torch.float32)
    grid = (triton.cdiv(n, _FP4_BN), k_pad // _FP4_BK, experts)
    _fp4_rows_kernel[grid](
        w, out, sf, k, n, k_pad, *w.stride(), BK=_FP4_BK, BN=_FP4_BN, GROUP=MXFP4_GROUP, num_warps=_FP4_WARPS
    )
    return out, sf
