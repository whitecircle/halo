"""Native fp8 / fp4 grouped MoE GEMM via DeepGEMM's on-device ``m_indices`` kernel: opt-in
(``HALO_DEEPGEMM_NATIVE=1``), never auto-selected, net-slower than bf16 at every measured shape.
Forward pads each expert's tokens to a 128-row segment (``m_indices`` maps rows to experts, ``-1`` =
pad); backward is bf16 ``F.grouped_mm`` on unpadded tokens. Needs the Blackwell image's ``deep_gemm``.
"""

from __future__ import annotations

import functools
import logging

import torch
import torch.nn.functional as F

from src.env import env_flag, env_int
from src.kernels.grouped_mm_autograd import grouped_mm_grads
from src.kernels.lowp.quantization import WeightVersionCache
from src.log import warn_once

logger = logging.getLogger(__name__)

# Block formats with a native DeepGEMM grouped kernel. MXFP4 is absent: simulated-only.
DEEPGEMM_FORMATS = ("mxfp8", "nvfp4")

_MIN_TOKENS_PER_EXPERT = env_int("HALO_DEEPGEMM_MIN_TOKENS_PER_EXPERT", 1024)
_MIN_N = env_int("HALO_DEEPGEMM_MIN_N", 4096)
_ALIGN = 128  # DeepGEMM contiguous layout: each expert's row segment padded to this boundary
# nvfp4 packs two e2m1 values per byte, so the CONTRACTION (K) dim must be a multiple of this many
# elements. Distinct from _ALIGN, which pads the M (row) dim — never unify the two.
_FP4_K_ALIGN = 256

_WEIGHT_CACHE = WeightVersionCache()  # w -> (w_fp8, w_sf), keyed (_version, fmt) — re-quant once per step

# Weight shapes ``(fmt, N, K, E)`` the kernel rejected; bounded by the model's distinct expert shapes.
_WARNED_FALLBACK_SHAPES: set[tuple] = set()


@functools.lru_cache(maxsize=1)
def deepgemm_available() -> bool:
    """True if the ``deep_gemm`` kernel package is importable (Blackwell image only)."""
    try:
        import deep_gemm  # noqa: PLC0415, F401
        from deep_gemm.utils import per_block_cast_to_fp8  # noqa: PLC0415, F401  (public weight-cast)
    except Exception:
        return False
    return True


def use_deepgemm(tokens_per_expert: float, n: int) -> bool:
    """Whether to route fp8/fp4 to native DeepGEMM: opt-in and only above the shape floor."""
    if not env_flag("HALO_DEEPGEMM_NATIVE"):
        return False
    if not deepgemm_available():
        _warn_absent_once()
        return False
    return tokens_per_expert >= _MIN_TOKENS_PER_EXPERT and n >= _MIN_N


@functools.lru_cache(maxsize=1)
def _warn_absent_once() -> None:
    """Say so when the opt-in cannot be honored: silence would leave the user believing they measure the
    native fp8/fp4 kernel while the simulated fake-quant path runs — a different speed AND a different
    numerical regime."""
    logger.warning(
        "HALO_DEEPGEMM_NATIVE=1 but the deep_gemm package is not importable — the native fp8/fp4 "
        "kernel is unavailable and the simulated fake-quant path runs instead. The wheel ships in "
        "the Blackwell image only (its kernels are SM100 block-scaled); install it in the container "
        "to opt in elsewhere."
    )


def _warn_fallback_once(shape_key: tuple, detail: str, exc: Exception) -> None:
    """Warn once per rejected weight shape that the opt-in native path reverted to bf16.

    ``shape_key`` is the *weight* identity ``(fmt, N, K, E)``, fixed per layer. The routed token count and
    the exception text stay out of it: both change every forward, so a key carrying either grows without
    bound and re-warns every step while claiming to warn once.
    """
    warn_once(
        logger,
        _WARNED_FALLBACK_SHAPES,
        shape_key,
        "DeepGEMM native path rejected %s (%s: %s) — falling back to the bf16 grouped GEMM for this "
        "weight shape (warned once per shape).",
        detail,
        type(exc).__name__,
        exc,
    )


@functools.lru_cache(maxsize=1)
def _deep_gemm():
    """Resolve ``deep_gemm`` once and set the process-global contiguous-layout alignment."""
    import deep_gemm  # noqa: PLC0415 — opt-in native backend, not in the image
    from deep_gemm import utils  # noqa: PLC0415 — opt-in native backend, not in the image

    utils.set_mk_alignment_for_contiguous_layout(_ALIGN)
    return deep_gemm, utils


def _quant_weight(w: torch.Tensor, fmt: str, k_pad: int, cacheable: bool = True):
    """Per-expert quantized weight operand (DeepGEMM B = ``[E, N, K]``), cached per optimizer step.

    ``w`` is ``[E, K, N]`` master; ``k_pad`` ≥ K zero-pads the contraction to meet the fp4 kernel's
    ``K/2 % 128 == 0`` constraint. ``cacheable=False`` for FSDP2-managed weights, whose pinned version
    counter would make the cache serve the step-0 quantization forever."""
    _, dg_utils = _deep_gemm()

    # Leaf params only: a non-leaf `.to(bf16)` copy churns its id at _version 0 → stale id-keyed hits.
    cacheable = cacheable and w.is_leaf
    fingerprint = (w._version, fmt)
    if cacheable:
        cached = _WEIGHT_CACHE.get(w, fingerprint)
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
        _WEIGHT_CACHE.put(w, fingerprint, operand)
    return operand


def _deepgemm_forward(x, w, offs, fmt, weight_cacheable):
    """DeepGEMM contiguous grouped forward → bf16 ``[T, N]`` (``None`` if the kernel can't run this shape).

    Both fp8/fp4 kernels take an fp8 activation; only the weight differs."""
    deep_gemm, dg_utils = _deep_gemm()
    align = dg_utils.align
    E, K, N = w.shape
    T = x.shape[0]
    ends = offs.tolist()
    # ``ends[-1] == T`` matters as much as length and order: ``out`` is uninitialized and only rows
    # inside ``[starts[g], ends[g])`` are written, so a short offs returns garbage as activations.
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
    a_op = dg_utils.per_token_cast_to_fp8(xp, use_ue8m0=True, gran_k=_ALIGN)  # fp8 activation for both kernels
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

    out = x.new_empty(T, N, dtype=torch.bfloat16)  # gather valid rows back to unpadded layout
    dst = 0
    for g in range(E):
        if counts[g]:
            out[starts[g] : ends[g]] = d[dst : dst + counts[g]]
        dst += aligned[g]
    return out


class _DeepGEMMGroupedFn(torch.autograd.Function):
    """DeepGEMM fp8/fp4 grouped forward; bf16 grouped backward (Wgrad-in-HP)."""

    @staticmethod
    def forward(ctx, x, w, offs, fmt, weight_cacheable):
        ctx.save_for_backward(x, w, offs)
        out = _deepgemm_forward(x, w, offs, fmt, weight_cacheable)
        if out is None:  # unsupported shape -> bf16 forward
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


def deepgemm_grouped_gemm(x, w, *, offs, precision, weight_cacheable=True):
    """Grouped MoE GEMM, native DeepGEMM fp8/fp4 forward + bf16 backward (master weights unchanged).

    Args:
        x: ``[T, K]`` activations (tokens stacked across experts), bf16/fp32 master.
        w: ``[E, K, N]`` expert weights; offs: ``[E]`` int32 cumulative token counts.
        precision: ``"mxfp8"`` or ``"nvfp4"`` — the only two DeepGEMM grouped formats (``mxfp4`` is
            simulated-only; raise rather than silently downgrade).
        weight_cacheable: False for FSDP2-managed weights, whose pinned version counter would make
            the per-step weight-quant cache serve the step-0 quantization forever.
    """
    if precision not in DEEPGEMM_FORMATS:
        raise ValueError(
            f"deepgemm_grouped_gemm supports only 'mxfp8' / 'nvfp4', got {precision!r} (mxfp4 is simulated-only)"
        )
    offs_i32 = offs.to(torch.int32) if offs.dtype != torch.int32 else offs
    return _DeepGEMMGroupedFn.apply(x, w, offs_i32, precision, weight_cacheable)
