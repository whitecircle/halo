"""Native fp8/fp4 grouped MoE GEMM on DeepGEMM's ``m_indices`` kernel: opt-in (``HALO_DEEPGEMM_NATIVE=1``), since its
activation-quantization pass outweighs the low-precision matmul at training shapes. The forward pads each expert's
tokens to a 128-row segment; the backward is bf16 ``F.grouped_mm`` on the unpadded tokens."""

from __future__ import annotations

import functools
import logging

import torch
import torch.nn.functional as F

from halo.env import env_flag
from halo.kernels.grouped_mm import grouped_mm_grads
from halo.kernels.lowp.quantization import WeightVersionCache
from halo.kernels.lowp.spec import QuantFormat
from halo.log import warn_once

logger = logging.getLogger(__name__)

# MXFP4 has no native DeepGEMM grouped kernel, so it is simulated only.
DEEPGEMM_FORMATS: tuple[QuantFormat, ...] = ("mxfp8", "nvfp4")

_MIN_TOKENS_PER_EXPERT = 1024
_MIN_N = 4096
_ALIGN = 128  # the contiguous layout pads each expert's row (M) segment to this
_FP4_K_ALIGN = 256  # the fp4 kernel's contraction (K) multiple, two e2m1 per byte

_weight_cache = WeightVersionCache()  # w -> (data, scales), re-quantized once per optimizer step
_warned_fallback_shapes: set[tuple] = set()  # (fmt, N, K, E): bounded by the distinct expert shapes


@functools.lru_cache(maxsize=1)
def _deep_gemm():
    """None when the package (Blackwell image only) does not import."""
    try:
        import deep_gemm  # noqa: PLC0415
        from deep_gemm import utils  # noqa: PLC0415
        from deep_gemm.utils import per_block_cast_to_fp8  # noqa: PLC0415, F401  (the public weight cast)
    except Exception:
        return None
    utils.set_mk_alignment_for_contiguous_layout(_ALIGN)
    return deep_gemm, utils


def deepgemm_available() -> bool:
    return _deep_gemm() is not None


@functools.lru_cache(maxsize=1)
def _warn_absent_once() -> None:
    logger.warning(
        "HALO_DEEPGEMM_NATIVE=1 but the deep_gemm package is not importable — the native fp8/fp4 "
        "kernel is unavailable and the simulated fake-quant path runs instead. The wheel ships in "
        "the Blackwell image only (its kernels are SM100 block-scaled); install it in the container "
        "to opt in elsewhere."
    )


def use_deepgemm(tokens_per_expert: float, n: int) -> bool:
    if not env_flag("HALO_DEEPGEMM_NATIVE"):
        return False
    if not deepgemm_available():
        _warn_absent_once()
        return False
    return tokens_per_expert >= _MIN_TOKENS_PER_EXPERT and n >= _MIN_N


def _warn_fallback_once(shape_key: tuple, detail: str, exc: Exception) -> None:
    """``shape_key`` is the weight's ``(fmt, N, K, E)``, as token count and exception text change every forward."""
    warn_once(
        logger,
        _warned_fallback_shapes,
        shape_key,
        "DeepGEMM native path rejected %s (%s: %s) — falling back to the bf16 grouped GEMM for this "
        "weight shape (warned once per shape).",
        detail,
        type(exc).__name__,
        exc,
    )


def _quant_weight(w: torch.Tensor, fmt: QuantFormat, k_pad: int, cacheable: bool = True):
    """Cached per optimizer step unless ``cacheable`` is False, as FSDP2's pinned version counter would serve the
    step-0 quantization forever."""
    _, dg_utils = _deep_gemm()

    # A non-leaf `.to(bf16)` copy churns its id at _version 0.
    cacheable = cacheable and w.is_leaf
    fingerprint = (w._version, fmt)
    if cacheable:
        cached = _weight_cache.get(w, fingerprint)
        if cached is not None:
            return cached
    b = w.transpose(-2, -1).contiguous()  # [E, N, K]
    if k_pad != b.shape[2]:
        b = F.pad(b, (0, k_pad - b.shape[2]))  # zero-pad the contraction; result unchanged
    cast = (
        (lambda t: dg_utils.per_token_cast_to_fp4(t, use_ue8m0=True, gran_k=32))
        if fmt == "nvfp4"
        else (lambda t: dg_utils.per_block_cast_to_fp8(t, use_ue8m0=True, gran_k=_ALIGN))
    )
    data0, sf0 = cast(b[0])
    data = b.new_empty((b.shape[0], *data0.shape), dtype=data0.dtype)
    sf = sf0.new_empty((b.shape[0], *sf0.shape))
    data[0], sf[0] = data0, sf0
    for i in range(1, b.shape[0]):
        data[i], sf[i] = cast(b[i])
    operand = (data, sf)
    if cacheable:
        _weight_cache.put(w, fingerprint, operand)
    return operand


def _deepgemm_forward(x, w, offs, fmt, weight_cacheable):
    """bf16 ``[T, N]``, or None for a shape the kernel rejects. Both kernels take an fp8 activation."""
    deep_gemm, dg_utils = _deep_gemm()
    align = dg_utils.align
    E, K, N = w.shape
    T = x.shape[0]
    ends = offs.tolist()
    # Rows outside ``[starts[g], ends[g])`` of the uninitialized ``out`` stay unwritten, so ``ends[-1]`` must be ``T``.
    covers_all_tokens = bool(ends) and len(ends) == E and ends[0] >= 0 and ends[-1] == T
    if not covers_all_tokens or any(b < a for a, b in zip(ends, ends[1:], strict=False)):
        raise ValueError(
            f"offs must be [{E}] non-decreasing cumulative token counts starting at >= 0 and ending "
            f"at T={T}; got {ends}. Rows outside the covered span are never written and would be "
            f"returned uninitialized."
        )
    starts = [0] + ends[:-1]
    counts = [ends[g] - starts[g] for g in range(E)]
    aligned = [align(c, _ALIGN) for c in counts]
    M = sum(aligned)

    xp = x.new_zeros(M, K)  # padded, expert-contiguous
    m_idx = torch.full((M,), -1, device=x.device, dtype=torch.int32)
    dst = 0
    for g in range(E):
        if counts[g]:
            xp[dst : dst + counts[g]] = x[starts[g] : ends[g]]
            m_idx[dst : dst + counts[g]] = g
        dst += aligned[g]

    # fp4 packs two e2m1 per byte → K must be ÷256; zero-pad the contraction (result unchanged).
    k_pad = align(K, _FP4_K_ALIGN) if fmt == "nvfp4" else K
    if k_pad != K:
        xp = F.pad(xp, (0, k_pad - K))
    a_op = dg_utils.per_token_cast_to_fp8(xp, use_ue8m0=True, gran_k=_ALIGN)
    b_op = _quant_weight(w, fmt, k_pad, cacheable=weight_cacheable)
    d = x.new_empty(M, N, dtype=torch.bfloat16)
    try:
        if fmt == "nvfp4":
            deep_gemm.m_grouped_fp8_fp4_gemm_nt_contiguous(
                a_op, b_op, d, m_idx, recipe=None, recipe_a=(1, 128), recipe_b=(1, 32)
            )
        else:
            deep_gemm.m_grouped_fp8_gemm_nt_contiguous(a_op, b_op, d, m_idx, recipe=None, recipe_a=None, recipe_b=None)
    except RuntimeError as exc:  # shape the kernel can't handle (e.g. narrow N) -> bf16 fallback
        _warn_fallback_once((fmt, N, K, E), f"m_grouped {fmt} GEMM on M={M} N={N} K={K} E={E}", exc)
        return None

    out = x.new_empty(T, N, dtype=torch.bfloat16)
    dst = 0
    for g in range(E):
        if counts[g]:
            out[starts[g] : ends[g]] = d[dst : dst + counts[g]]
        dst += aligned[g]
    return out


class _DeepGEMMGroupedFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w, offs, fmt, weight_cacheable):
        ctx.save_for_backward(x, w, offs)
        out = _deepgemm_forward(x, w, offs, fmt, weight_cacheable)
        if out is None:
            out = F.grouped_mm(x.to(torch.bfloat16), w.to(torch.bfloat16), offs=offs)
        return out.to(x.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        x, w, offs = ctx.saved_tensors
        grad_x, grad_w = grouped_mm_grads(
            x.to(torch.bfloat16),
            w.to(torch.bfloat16),
            grad_output.to(torch.bfloat16),
            offs,
            ctx.needs_input_grad[0],
            ctx.needs_input_grad[1],
        )
        if grad_x is not None:
            grad_x = grad_x.to(x.dtype)
        if grad_w is not None:
            grad_w = grad_w.to(w.dtype)
        return grad_x, grad_w, None, None, None


def deepgemm_grouped_gemm(x, w, *, offs, quant_format: QuantFormat, weight_cacheable=True):
    """Native forward in ``quant_format``, bf16 backward. ``mxfp4`` has no native kernel and raises."""
    if quant_format not in DEEPGEMM_FORMATS:
        raise ValueError(
            f"deepgemm_grouped_gemm supports only {DEEPGEMM_FORMATS}, got {quant_format!r} (mxfp4 is simulated-only)"
        )
    if not deepgemm_available():
        raise ImportError("deepgemm_grouped_gemm needs the deep_gemm package (Blackwell image only)")
    return _DeepGEMMGroupedFn.apply(x, w, offs.to(torch.int32), quant_format, weight_cacheable)
