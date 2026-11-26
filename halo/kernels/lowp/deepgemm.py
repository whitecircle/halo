"""Native grouped MoE GEMM on DeepGEMM's ``m_indices`` kernel: FP8 activations (1x128) times FP8 (128x128) or MXFP4
(1x32) weights, opt-in through ``HALO_DEEPGEMM_NATIVE=1``. The forward pads each expert's tokens to a 128-row segment;
the backward is bf16 ``F.grouped_mm`` on the unpadded tokens."""

from __future__ import annotations

import functools
import logging

import torch
import torch.nn.functional as F

from halo.env import env_flag
from halo.kernels.grouped_mm import grouped_mm_grads
from halo.kernels.lowp.quantization import WeightVersionCache
from halo.kernels.lowp.spec import QuantFormat

logger = logging.getLogger(__name__)

# The kernel's weight scales are UE8M0, so NVFP4's e4m3 ones have no native path.
DEEPGEMM_FORMATS: tuple[QuantFormat, ...] = ("mxfp8", "mxfp4")

_ALIGN = 128  # the contiguous layout pads each expert's row (M) segment to this
_FP4_K_ALIGN = 256  # the fp4 kernel's contraction (K) multiple, two e2m1 per byte
_CAST_CHUNK_BYTES = 256 << 20  # the fp32 temporaries of one weight-cast call, which are the size of its input

_weight_cache = WeightVersionCache()  # w -> (data, scales), re-quantized once per optimizer step


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
        "HALO_DEEPGEMM_NATIVE=1 but the deep_gemm package is not importable — the native fp8/mxfp4 "
        "kernel is unavailable and the simulated fake-quant path runs instead. The wheel ships in "
        "the Blackwell image only (its kernels are SM100 block-scaled); install it in the container "
        "to opt in elsewhere."
    )


@functools.cache
def _warn_no_native_format_once(quant_format: QuantFormat) -> None:
    logger.warning(
        "HALO_DEEPGEMM_NATIVE=1 has no kernel for %s, so it runs simulated; the native formats are %s.",
        quant_format,
        DEEPGEMM_FORMATS,
    )


def use_deepgemm(quant_format: QuantFormat) -> bool:
    """The run's one native decision. It reads no shape or token count, as a per-call gate would flip an expert's
    numerics between forwards."""
    if not env_flag("HALO_DEEPGEMM_NATIVE"):
        return False
    if quant_format not in DEEPGEMM_FORMATS:
        _warn_no_native_format_once(quant_format)
        return False
    if not deepgemm_available():
        _warn_absent_once()
        return False
    return True


def _cast_experts(b: torch.Tensor, cast, align, block_scaled: bool):
    """``(data, scales)`` of ``b`` ``[E, N, K]``, cast over chunks of experts that stay under ``_CAST_CHUNK_BYTES``.

    The MXFP4 cast scales row by row, so experts flatten as they are. The FP8 weight cast scales ``_ALIGN`` x ``_ALIGN``
    blocks, so each expert's N is zero-padded to a block boundary first: a block then never spans two experts, and the
    padding is the zeros the per-expert cast would add itself. Outputs are allocated dense and filled by copy, as
    DeepGEMM asserts the weight's expert stride is ``N * K``, which a slice of the padded layout does not keep at E=1."""
    E, N, K = b.shape
    n_cast = align(N, _ALIGN) if block_scaled else N
    step = max(1, _CAST_CHUNK_BYTES // (4 * n_cast * K))

    def cast_chunk(start: int):
        chunk = b[start : start + step]
        count = chunk.shape[0]
        if n_cast != N:
            chunk = F.pad(chunk, (0, 0, 0, n_cast - N))
        data_c, sf_c = cast(chunk.reshape(count * n_cast, K))
        return data_c.reshape(count, n_cast, *data_c.shape[1:])[:, :N], sf_c.reshape(count, -1, *sf_c.shape[1:])

    data_c, sf_c = cast_chunk(0)
    data = data_c.new_empty((E, N, *data_c.shape[2:]))
    sf = sf_c.new_empty((E, *sf_c.shape[1:]))
    for start in range(0, E, step):
        if start:
            data_c, sf_c = cast_chunk(start)
        data[start : start + len(data_c)] = data_c
        sf[start : start + len(sf_c)] = sf_c
        del data_c, sf_c  # the next cast starts with no chunk of the quantized weight held beside the full outputs
    return data, sf


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
    # e2m1 has no NaN code and the cast turns a NaN amax into a 2^127 scale, so the block would decode as finite zeros.
    if fmt == "mxfp4" and not w.isfinite().all():
        raise ValueError(f"Non-finite expert weight {tuple(w.shape)}: DeepGEMM's MXFP4 cast would hide it as zeros.")
    b = w.transpose(-2, -1).contiguous()  # [E, N, K]
    if k_pad != b.shape[2]:
        b = F.pad(b, (0, k_pad - b.shape[2]))  # zero-pad the contraction; result unchanged
    cast = (
        (lambda t: dg_utils.per_token_cast_to_fp4(t, use_ue8m0=True, gran_k=32))
        if fmt == "mxfp4"
        else (lambda t: dg_utils.per_block_cast_to_fp8(t, use_ue8m0=True, gran_k=_ALIGN))
    )
    operand = _cast_experts(b, cast, dg_utils.align, block_scaled=fmt == "mxfp8")
    if cacheable:
        _weight_cache.put(w, fingerprint, operand)
    return operand


def _deepgemm_forward(x, w, offs, fmt, weight_cacheable):
    """bf16 ``[T, N]``. Both kernels take an fp8 activation."""
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
    k_pad = align(K, _FP4_K_ALIGN) if fmt == "mxfp4" else K
    if k_pad != K:
        xp = F.pad(xp, (0, k_pad - K))
    a_op = dg_utils.per_token_cast_to_fp8(xp, use_ue8m0=True, gran_k=_ALIGN)
    b_op = _quant_weight(w, fmt, k_pad, cacheable=weight_cacheable)
    d = x.new_empty(M, N, dtype=torch.bfloat16)
    try:
        if fmt == "mxfp4":
            deep_gemm.m_grouped_fp8_fp4_gemm_nt_contiguous(
                a_op, b_op, d, m_idx, recipe=None, recipe_a=(1, 128), recipe_b=(1, 32)
            )
        else:
            deep_gemm.m_grouped_fp8_gemm_nt_contiguous(a_op, b_op, d, m_idx, recipe=None, recipe_a=None, recipe_b=None)
    except torch.cuda.OutOfMemoryError:
        raise
    except RuntimeError as exc:
        raise RuntimeError(
            f"DeepGEMM rejected the {fmt} grouped GEMM on M={M} N={N} K={K} E={E}: {exc}. A native run has no bf16 "
            f"fallback, which would change this expert's numerics mid-run; unset HALO_DEEPGEMM_NATIVE to run simulated."
        ) from exc

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
        return _deepgemm_forward(x, w, offs, fmt, weight_cacheable).to(x.dtype)

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
    """Native forward in ``quant_format``, bf16 backward. ``nvfp4`` has no native kernel and raises."""
    if quant_format not in DEEPGEMM_FORMATS:
        raise ValueError(
            f"deepgemm_grouped_gemm supports only {DEEPGEMM_FORMATS}, got {quant_format!r} (nvfp4 is simulated-only)"
        )
    if not deepgemm_available():
        raise ImportError("deepgemm_grouped_gemm needs the deep_gemm package (Blackwell image only)")
    if offs is None:
        raise ValueError("deepgemm_grouped_gemm needs offs: the kernel's m_indices are built from the group ends")
    return _DeepGEMMGroupedFn.apply(x, w, offs.to(torch.int32), quant_format, weight_cacheable)
