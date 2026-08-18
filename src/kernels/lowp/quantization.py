"""Block-scaled quantization for the low-precision GEMMs — the single source of the quant math, plus
the per-step :class:`WeightVersionCache` its weight paths and :mod:`~src.kernels.lowp.deepgemm` share.

A format is low-precision elements plus one shared scale per contraction-axis block: MXFP8 (``e4m3`` +
``e8m0``/32), MXFP4 (``e2m1`` + ``e8m0``/32), NVFP4 (``e2m1`` + ``e4m3``/16 + a per-tensor fp32 global
scale — two-level, the most accurate fp4). Storage form is a :class:`BlockScaledTensor`.
"""

from __future__ import annotations

import logging
import weakref
from dataclasses import dataclass

import torch

from src.env import env_flag

logger = logging.getLogger(__name__)

E4M3_MAX = 448.0  # largest finite value of float8_e4m3fn
E2M1_MAX = 6.0  # largest magnitude of e2m1
_E2M1_LEVELS = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)  # e2m1 magnitudes, codes 0..7
# RNE bucket edges; a plain tuple (lru_cache'd tensors defeat compile) keeps _round_trip exact.
_E2M1_BOUNDARIES = tuple((a + b) / 2 for a, b in zip(_E2M1_LEVELS[:-1], _E2M1_LEVELS[1:], strict=True))
# Midpoints whose lower level has an odd mantissa code; hardware RNE resolves these ties UP (0.75→1.0)
# while bucketize resolves every tie down, and exact ties are routine at bf16 over power-of-two scales.
_E2M1_TIES_UP = _E2M1_BOUNDARIES[1::2]
_E8M0_BIAS = 127  # e8m0 stores (exponent + 127) as uint8
_E8M0_NAN = 255  # OCP MX: biased exponent 0xFF is e8m0's only non-finite code, and means NaN
# Keeps log2/divide finite on an all-zero block. The smallest fp32 NORMAL, not a round epsilon: a
# floor above a tensor's real amax rescales every block and quantizes it to zeros.
_AMAX_FLOOR = torch.finfo(torch.float32).tiny
# Tensor amax NVFP4's global scale normalizes to, so the largest block's e4m3 scale reaches E4M3_MAX.
_NVFP4_GLOBAL_SCALE_DIVISOR = E4M3_MAX * E2M1_MAX

# Elements per shared scale, per format. The checkpoint exporter sizes its contraction axis by it
# rather than restating the block layout this module's math defines.
FORMAT_BLOCK_SIZE = {"mxfp8": 32, "mxfp4": 32, "nvfp4": 16}


@dataclass(frozen=True)
class BlockScaledTensor:
    """A block-scaled quantized tensor.

    ``data`` holds ``float8_e4m3fn`` elements, or two ``e2m1`` nibbles per byte when ``packed``; ``scales``
    one entry per ``block_size``-element block along ``axis``, an ``e8m0`` biased exponent when
    ``pow2_scale`` (MX) and an ``e4m3`` value otherwise (NVFP4). An element is
    ``code x scales x global_scale``, so dropping NVFP4's per-tensor ``global_scale`` (``None`` for the
    single-level MX formats) reads the tensor rescaled by up to ``E4M3_MAX * E2M1_MAX``.
    """

    data: torch.Tensor
    scales: torch.Tensor
    block_size: int
    axis: int
    packed: bool
    pow2_scale: bool
    global_scale: torch.Tensor | None = None


def _move_axis_last(t: torch.Tensor, axis: int) -> tuple[torch.Tensor, int]:
    axis = axis % t.ndim
    if axis != t.ndim - 1:
        t = t.transpose(axis, -1).contiguous()
    return t, axis


def _restore_axis(t: torch.Tensor, axis: int, ndim: int) -> torch.Tensor:
    if axis != ndim - 1:
        t = t.transpose(axis, -1).contiguous()
    return t


def _e8m0_scale(amax: torch.Tensor, format_max: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-block ``e8m0`` scale for an MX format: ``(fp32 divisor, biased-exponent uint8)``.

    A NaN block takes the spec's ``0xFF`` NaN code: ``e2m1`` has no NaN encoding and a NaN exponent casts
    to ``uint8`` 0, which would decode the poison as a finite near-zero block.
    """
    scale_exp = torch.ceil(torch.log2(amax.clamp(min=_AMAX_FLOOR) / format_max)).clamp(-_E8M0_BIAS, _E8M0_BIAS)
    biased = (scale_exp + _E8M0_BIAS).to(torch.uint8).masked_fill(amax.isnan(), _E8M0_NAN)
    return torch.exp2(scale_exp), biased


def _e8m0_to_float(scales: torch.Tensor) -> torch.Tensor:
    """Biased ``e8m0`` exponents → fp32 multipliers, decoding the ``0xFF`` NaN code as NaN."""
    return torch.exp2(scales.float() - _E8M0_BIAS).masked_fill(scales == _E8M0_NAN, float("nan"))


def _mxfp8_core(blocks: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """``blocks`` → (``e4m3`` data, ``e8m0`` biased-exponent scale ``[*, n_block, 1]`` uint8)."""
    divisor, scales = _e8m0_scale(blocks.abs().amax(dim=-1, keepdim=True), E4M3_MAX)
    data = (blocks / divisor).clamp(-E4M3_MAX, E4M3_MAX).to(torch.float8_e4m3fn)
    return data, scales


def _e2m1_codes(blocks: torch.Tensor, divisor: torch.Tensor) -> torch.Tensor:
    """fp32 ``blocks`` → e2m1 codes (``uint8``: ``sign << 3 | magnitude-level``) given a per-block
    ``divisor``. Shared by MXFP4/NVFP4 — only the scale differs.

    Only an exactly-zero divisor is substituted, and only to avoid the division (such a block rounds to
    code 0 regardless); an epsilon floor would corrupt every tensor whose real scales sit below it.
    """
    scaled = blocks / divisor.masked_fill(divisor == 0, 1.0)
    sign = (scaled < 0).to(torch.uint8)
    mag = scaled.abs().clamp(max=E2M1_MAX)
    boundaries = torch.tensor(_E2M1_BOUNDARIES, device=blocks.device, dtype=mag.dtype)
    codes = torch.bucketize(mag, boundaries).to(torch.uint8)
    ties_up = torch.isin(mag, torch.tensor(_E2M1_TIES_UP, device=blocks.device, dtype=mag.dtype))
    return (codes + ties_up.to(torch.uint8)) | (sign << 3)


def _mxfp4_core(blocks: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, None]:
    """``blocks`` → (e2m1 code ``[*, n_block, block_size]`` uint8, ``e8m0`` scale ``[*, n_block, 1]`` uint8,
    no global scale — MX is single-level by definition)."""
    divisor, scales = _e8m0_scale(blocks.abs().amax(dim=-1, keepdim=True), E2M1_MAX)
    return _e2m1_codes(blocks, divisor), scales, None


def _nvfp4_core(blocks: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """``blocks`` → (e2m1 code ``[*, n_block, block_size]`` uint8, ``e4m3`` scale ``[*, n_block, 1]``,
    per-tensor fp32 global scale).

    Two-level, as the format defines it: the global scale lifts the tensor amax to ``E4M3_MAX`` so each
    block's ``e4m3`` scale is RELATIVE to the tensor. A single absolute scale rounds to zero — the whole
    block dequantizing to zeros — for any block amax below ``E2M1_MAX x`` half of e4m3's smallest
    subnormal (5.86e-3), the band residual-scaled ``down_proj`` init (``0.02/sqrt(2L)``) sits in.

    Rounding the global scale UP to a power of two keeps e4m3 rounding invariant, so an EP rank's expert
    shard and the gathered tensor dequantize identically despite their differing amax — the QAT forward
    and the exported checkpoint agree — at a cost of one bit of the second level's ~4.6 decades.
    """
    tensor_amax = blocks.abs().amax().clamp(min=_AMAX_FLOOR)
    global_scale = torch.exp2(torch.ceil(torch.log2(tensor_amax / _NVFP4_GLOBAL_SCALE_DIVISOR)))
    amax = blocks.abs().amax(dim=-1, keepdim=True)
    # Redundant by construction (no block amax exceeds the tensor amax); kept so fp rounding can never
    # push the scale past e4m3's range, where it would cast to NaN.
    scale = (amax / E2M1_MAX / global_scale).clamp(max=E4M3_MAX).to(torch.float8_e4m3fn)
    code = _e2m1_codes(blocks, scale.float() * global_scale)
    return code, scale, global_scale


def _as_blocks(t: torch.Tensor, axis: int, block_size: int) -> tuple[torch.Tensor, int, tuple, int]:
    """Move ``axis`` last, upcast to fp32, split into blocks. Returns (blocks, axis, lead_shape, k)."""
    if t.shape[axis] % block_size != 0:
        raise ValueError(f"axis {axis} size {t.shape[axis]} not divisible by block_size {block_size}")
    work, axis = _move_axis_last(t.float(), axis)
    lead, k = work.shape[:-1], work.shape[-1]
    return work.reshape(*lead, k // block_size, block_size), axis, lead, k


def quantize_mxfp8(t: torch.Tensor, axis: int = -1, block_size: int = 32) -> BlockScaledTensor:
    """Quantize ``t`` to MXFP8 (``e4m3`` data + ``e8m0`` scales) along ``axis`` (storage form)."""
    blocks, axis, lead, k = _as_blocks(t, axis, block_size)
    quant, scales = _mxfp8_core(blocks)
    data = _restore_axis(quant.reshape(*lead, k), axis, t.ndim)
    scales = _restore_axis(scales.squeeze(-1), axis, t.ndim)
    return BlockScaledTensor(data.contiguous(), scales.contiguous(), block_size, axis, packed=False, pow2_scale=True)


def _quantize_fp4(t: torch.Tensor, axis: int, block_size: int, core, pow2_scale: bool) -> BlockScaledTensor:
    """Shared fp4 storage path (MXFP4 / NVFP4): per-block e2m1 codes, two nibbles packed per byte."""
    blocks, axis, lead, k = _as_blocks(t, axis, block_size)
    code, scale, global_scale = core(blocks)
    code = code.reshape(*lead, k)
    packed = (code[..., 0::2] | (code[..., 1::2] << 4)).to(torch.uint8)
    data = _restore_axis(packed, axis, t.ndim)
    scales = _restore_axis(scale.squeeze(-1), axis, t.ndim)
    return BlockScaledTensor(
        data.contiguous(),
        scales.contiguous(),
        block_size,
        axis,
        packed=True,
        pow2_scale=pow2_scale,
        global_scale=global_scale,
    )


def quantize_mxfp4(t: torch.Tensor, axis: int = -1, block_size: int = 32) -> BlockScaledTensor:
    """Quantize ``t`` to MXFP4 (packed ``e2m1`` data + ``e8m0`` power-of-two scales) along ``axis``."""
    return _quantize_fp4(t, axis, block_size, _mxfp4_core, pow2_scale=True)


def quantize_nvfp4(t: torch.Tensor, axis: int = -1, block_size: int = 16) -> BlockScaledTensor:
    """Quantize ``t`` to NVFP4 (packed ``e2m1`` data + ``e4m3`` scales) along ``axis`` (storage form)."""
    return _quantize_fp4(t, axis, block_size, _nvfp4_core, pow2_scale=False)


def dequantize(q: BlockScaledTensor) -> torch.Tensor:
    """Reconstruct a bf16 tensor from a :class:`BlockScaledTensor` — the export tool's round-trip
    error check, and the reference every round-trip test measures against."""
    work, axis = _move_axis_last(q.data, q.axis)
    scales, _ = _move_axis_last(q.scales, q.axis)
    if q.packed:
        low = (work & 0x0F).to(torch.long)
        high = (work >> 4).to(torch.long)
        levels = torch.tensor(_E2M1_LEVELS, device=q.data.device)
        mag = torch.stack([levels[low & 0x7], levels[high & 0x7]], dim=-1)
        sgn = torch.stack([1 - 2 * (low >> 3), 1 - 2 * (high >> 3)], dim=-1)
        elems = (mag * sgn).reshape(*work.shape[:-1], work.shape[-1] * 2)
    else:
        elems = work.float()
    scale_f = _e8m0_to_float(scales) if q.pow2_scale else scales.float()
    if q.global_scale is not None:
        scale_f = scale_f * q.global_scale
    lead, k = elems.shape[:-1], elems.shape[-1]
    out = (elems.reshape(*lead, k // q.block_size, q.block_size) * scale_f.unsqueeze(-1)).reshape(*lead, k)
    return _restore_axis(out, axis, q.data.ndim).to(torch.bfloat16)


_QUANTIZERS = {"mxfp8": quantize_mxfp8, "mxfp4": quantize_mxfp4, "nvfp4": quantize_nvfp4}


def _block_round_trip(t: torch.Tensor, fmt: str, axis: int) -> torch.Tensor:
    quantizer = _QUANTIZERS.get(fmt)
    if quantizer is None:
        raise ValueError(f"unknown block-scaled format {fmt!r} (expected 'mxfp8', 'mxfp4' or 'nvfp4')")
    return dequantize(quantizer(t, axis=axis))


# Only e8m0 (power-of-two) scales survive compile's reciprocal-multiply exactly; NVFP4's e4m3 scale flips
# ~0.18% of elements, desyncing the QAT forward from the eager checkpoint quantizer. Weights only.
_COMPILABLE = ("mxfp8", "mxfp4")
_compiled_round_trip = None
_compile_failed = False


def _warn_compile_disabled(exc: Exception) -> None:
    """Report the one-way drop to eager quantization (slower per-step, same numerics)."""
    logger.warning(
        "Low-precision quantization could not run compiled (%s: %s) — falling back to eager "
        "block-scale quantization for the rest of the run. Set HALO_LOWP_COMPILE=0 to silence.",
        type(exc).__name__,
        exc,
    )


def _round_trip(t: torch.Tensor, fmt: str, axis: int) -> torch.Tensor:
    """Quantize→dequantize ``t``, compiled for the power-of-two-scale formats (fixed-shape weights).
    Eager for nvfp4 / when compilation is off or unavailable."""
    global _compiled_round_trip, _compile_failed
    if fmt not in _COMPILABLE or _compile_failed or not env_flag("HALO_LOWP_COMPILE", default=True):
        return _block_round_trip(t, fmt, axis)
    if _compiled_round_trip is None:
        try:
            _compiled_round_trip = torch.compile(_block_round_trip, dynamic=True)
        except Exception as exc:  # no torch.compile backend → eager
            _compile_failed = True
            _warn_compile_disabled(exc)
            return _block_round_trip(t, fmt, axis)
    try:
        return _compiled_round_trip(t, fmt, axis)
    except Exception as exc:  # compile/runtime failure → eager for the rest of the run
        _compile_failed = True
        _warn_compile_disabled(exc)
        return _block_round_trip(t, fmt, axis)


class _FakeQuant(torch.autograd.Function):
    """Straight-through block-scaled quantization (QAT estimator): forward quantize→dequantize so the
    matmul sees low-precision numerics; backward is identity to the bf16/fp32 master."""

    @staticmethod
    def forward(ctx, t: torch.Tensor, fmt: str, axis: int) -> torch.Tensor:
        # Eager: activation path (and dense weights) — shapes vary, so compiling thrashes.
        return _block_round_trip(t, fmt, axis).to(t.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None, None


def fake_quant(t: torch.Tensor, fmt: str, axis: int = -1) -> torch.Tensor:
    """Differentiable block-scaled fake quantization (straight-through estimator), eager.

    Quantizes ``t`` to ``fmt`` along ``axis`` and back, identity backward; for weights use
    :func:`cached_fake_quant`. The quantized axis must divide by the block size (32 mx, 16 nvfp4).
    """
    return _FakeQuant.apply(t, fmt, axis)


class WeightVersionCache:
    """``weight -> value`` cache keyed by tensor identity plus a caller fingerprint (``(_version, fmt)``
    and the like), for tensors derived once per optimizer step. A ``weakref`` finalizer evicts an entry
    when its weight is collected, so a freed weight's reused ``id`` can never serve a stale hit."""

    def __init__(self) -> None:
        self._entries: dict[int, tuple[weakref.ref, tuple, object]] = {}

    def get(self, weight: torch.Tensor, fingerprint: tuple) -> object | None:
        entry = self._entries.get(id(weight))
        if entry is None:
            return None
        ref, cached_fingerprint, value = entry
        if ref() is not weight or cached_fingerprint != fingerprint:
            return None
        return value

    def put(self, weight: torch.Tensor, fingerprint: tuple, value: object) -> None:
        wid = id(weight)
        entries = self._entries
        ref = weakref.ref(weight, lambda _ref, wid=wid: entries.pop(wid, None))
        entries[wid] = (ref, fingerprint, value)

    def __len__(self) -> int:
        return len(self._entries)


# Correct only for the EP expert weights it is used on: plain 3D params whose `_version` tracks the
# optimizer step exactly, unlike a dense MLP's all-gathered FSDP2 view.
_WEIGHT_DEQUANT_CACHE = WeightVersionCache()  # weight -> dequantized_detached, keyed (_version, fmt, axis)


class _CachedFakeQuant(torch.autograd.Function):
    """Straight-through wrapper reusing a cached dequant weight (fresh clone per call so the shared cache
    entry never collides with autograd's one-grad_fn-per-tensor rule). Identity backward."""

    @staticmethod
    def forward(ctx, weight: torch.Tensor, cached_dequant: torch.Tensor) -> torch.Tensor:
        return cached_dequant.clone()

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None


def cached_fake_quant(weight: torch.Tensor, fmt: str, axis: int = -1) -> torch.Tensor:
    """``fake_quant`` for a (quasi-static) EP expert weight, quantized once per optimizer step.

    The round-trip is keyed on ``weight._version`` and reused across grad-accum; exact and
    straight-through either way. Only leaf params are cached — a non-leaf input (a fp32 master's
    ``weight.to(bf16)`` copy) churns its ``id`` while ``_version`` stays 0. Disable with
    ``HALO_LOWP_WEIGHT_CACHE=0``."""
    if not weight.is_leaf or not env_flag("HALO_LOWP_WEIGHT_CACHE", default=True):
        return fake_quant(weight, fmt, axis)
    fingerprint = (weight._version, fmt, axis)
    dequant = _WEIGHT_DEQUANT_CACHE.get(weight, fingerprint)
    if dequant is None:
        with torch.no_grad():
            dequant = _round_trip(weight, fmt, axis).to(weight.dtype)
        _WEIGHT_DEQUANT_CACHE.put(weight, fingerprint, dequant)
    return _CachedFakeQuant.apply(weight, dequant)
