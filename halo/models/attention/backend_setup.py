"""Process-wide attention backend fixes (flash varlen seqlens, FA4 head_dim-256 strides, wide-head SDPA, flex
compile) and the FA4 warm-up."""

from __future__ import annotations

import functools
import logging

import torch
from transformers import PreTrainedModel
from transformers import modeling_flash_attention_utils as flash_utils
from transformers.integrations import sdpa_attention as _sdpa_mod
from transformers.integrations.flex_attention import WrappedFlexAttention, flex_attention

from halo.models.attention.fa4_entry_points import wrap_fa4_entry_points
from halo.models.attention.gpt_oss_sinks import uses_gpt_oss_sinks
from halo.models.attention.layout import attention_calls
from halo.models.attention.selection import (
    FA4_DEDICATED_HEAD_DIM,
    FLASH_MAX_HEAD_DIM,
    effective_attn_implementation,
    flash_attention_deterministic,
    flash_kernel,
)
from halo.runtime.ranks import is_global_main_process

logger = logging.getLogger(__name__)

# Warm-up row length: the floor keeps a kernel on a realistic tile count, the margin pushes the row past a
# sliding window so the windowed variant is the one that compiles.
WARMUP_MIN_SEQ_LEN = 512
_WARMUP_WINDOW_MARGIN = 8

# Marker on the wrapped ``flash_attn.cute`` entry points.
_FA4_CONTIGUOUS_ATTR = "_halo_fa4_contiguous_dedicated_head"


def patch_transformers_flash_varlen_int_seqlen() -> None:
    """Make transformers pass varlen ``max_seqlen_{q,k}`` to the flash kernel as python ints (idempotent).

    It forwards a fresh CUDA tensor each step, and FA4's varlen-backward compile key hashes it by
    identity, so the backward kernel would recompile on every call.
    """
    fn = flash_utils._process_flash_attention_kwargs
    if getattr(fn, "_halo_int_seqlen", False):
        return

    @functools.wraps(fn)
    def wrapped(*args, **kwargs):
        flash_kwargs = fn(*args, **kwargs)
        mq = flash_kwargs.get("max_seqlen_q")
        mk = flash_kwargs.get("max_seqlen_k")
        if torch.is_tensor(mq):
            flash_kwargs["max_seqlen_q"] = int(mq)
            # transformers aliases max_seqlen_k to the same tensor to save a sync; keep the alias.
            if mk is mq:
                flash_kwargs["max_seqlen_k"] = flash_kwargs["max_seqlen_q"]
                return flash_kwargs
        if torch.is_tensor(mk):
            flash_kwargs["max_seqlen_k"] = int(mk)
        return flash_kwargs

    wrapped._halo_int_seqlen = True
    flash_utils._process_flash_attention_kwargs = wrapped


def fa4_contiguous_dedicated_head_inputs(fn):
    """``fn`` (an FA4 entry point) with q, k and v made contiguous at :data:`FA4_DEDICATED_HEAD_DIM`; every other
    call passes through untouched."""

    @functools.wraps(fn)
    def wrapped(q, k, v, *args, **kwargs):
        if q.shape[-1] == v.shape[-1] == FA4_DEDICATED_HEAD_DIM:
            q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
        return fn(q, k, v, *args, **kwargs)

    return wrapped


def install_fa4_contiguous_dedicated_head_inputs() -> bool:
    """Hand FA4's dedicated head_dim-256 kernel contiguous q/k/v (idempotent).

    flash-attn-4 4.0.0b16 builds that kernel's forward layouts with packed strides and repairs only ``stride(-1)``,
    so a head-major q/k/v is read scrambled: a finite but wrong output and wrong gradients, NaN in training. Qwen3.5's
    partial-rotary ``torch.cat`` leaves q/k head-major ahead of the flash transpose, and a KV-cache forward leaves
    k/v so on every family. Fixed upstream from 4.0.0b19 (Dao-AILab/flash-attention#2666, #2670); drop this with
    that bump. Installed wherever FA4 runs, since the copy is exact on every FA4 kernel. False when
    ``flash_attn.cute`` is not importable.
    """
    return wrap_fa4_entry_points(
        _FA4_CONTIGUOUS_ATTR,
        lambda fn, _varlen: fa4_contiguous_dedicated_head_inputs(fn),
        f"contiguous q/k/v for the dedicated head_dim-{FA4_DEDICATED_HEAD_DIM} kernel",
    )


def patch_sdpa_for_wide_heads() -> None:
    """Force SDPA to its mem-efficient kernel, process-wide, for heads past :data:`FLASH_MAX_HEAD_DIM`.

    FA2 and cuDNN SDPA reject such a head and the math kernel OOMs on the full score matrix. GQA is
    forced to the manual KV repeat: native ``enable_gqa`` under mem-efficient-only is unverified on the
    narrower sliding heads, the repeat is the measured path. Idempotent.
    """
    if is_global_main_process():
        logger.warning(
            "Wide-head SDPA patch: forcing mem-efficient SDPA and use_gqa_in_sdpa=False PROCESS-GLOBALLY — "
            "every other sdpa model in this process (e.g. a frozen teacher beside a narrower-headed policy) "
            "takes the manual-KV-repeat path too (numerics identical; perf/memory only)."
        )
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_cudnn_sdp(False)
    torch.backends.cuda.enable_math_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(True)

    if not getattr(_sdpa_mod.use_gqa_in_sdpa, "_wide_head_patched", False):

        def _no_gqa_in_sdpa(attention_mask, key, value):
            return False

        _no_gqa_in_sdpa._wide_head_patched = True
        _sdpa_mod.use_gqa_in_sdpa = _no_gqa_in_sdpa
    if is_global_main_process():
        logger.info(
            f"Forced mem-efficient SDPA + manual KV repeat for heads wider than {FLASH_MAX_HEAD_DIM} "
            "(unlocks seq>20k by avoiding the math kernel's score matrix)"
        )


def patch_flex_attention_compile(reason: str):
    """Run raw flex_attention instead of transformers' compiled wrapper; call before the first flex
    forward. The wrapper's seq-length recompiles deadlock EP's all-to-all, and its backward NaNs under
    FSDP2 with GptOss sinks."""
    WrappedFlexAttention._is_flex_compiled = False
    WrappedFlexAttention._compiled_flex_attention = None

    @torch.compiler.disable(recursive=False)
    def _safe_init(self, training):
        if not self._is_flex_compiled or training != self.training:
            self.training = training
            self._compiled_flex_attention = flex_attention
            self._is_flex_compiled = True

    WrappedFlexAttention.__init__ = _safe_init
    if is_global_main_process():
        logger.info(f"Patched FlexAttention: disabled torch.compile wrapper ({reason})")


def warm_fa4_kernels(model: PreTrainedModel, *, dtype: torch.dtype, device: torch.device | None = None) -> None:
    """Pre-compile this model's FA4 forward and backward kernels, dense and varlen, for each attention layer's
    own geometry (:func:`~halo.models.attention.layout.attention_calls`) at the run's ``dtype``, in transformers'
    deterministic mode as set at the load (:func:`flash_attention_deterministic`), which FA4's backward keys on. A
    softcapped layer warms its dense call only: FA4's varlen backward takes no softcap, and its varlen forward with
    one does not finish on B300, so no varlen call is launched for it.

    Rank-local and best-effort: a first-use JIT compile on one rank lets its peers race into the next
    collective, so the caller fences every rank around it and nothing here may gate a collective.
    """
    try:
        config = getattr(model, "config", None)
        if config is None:
            return
        if effective_attn_implementation(config) != "flash_attention_4":
            return

        # Each layer's own geometry, both with and without its window: a row no longer than the window
        # dispatches the windowless kernel.
        windows: dict[tuple[int, int, int, float | None], set[int | None]] = {}
        for call in attention_calls(model):
            windows.setdefault((call.heads, call.kv_heads, call.head_dim, call.softcap), {None}).add(call.window)
        if not windows:
            raise ValueError(f"{type(model).__name__} holds no attention module to read the kernel geometry from")
        if device is None:
            device = torch.device(f"cuda:{torch.cuda.current_device()}")
        longest_window = max((w for ws in windows.values() for w in ws if w), default=0)
        seq_len = max(WARMUP_MIN_SEQ_LEN, longest_window + _WARMUP_WINDOW_MARGIN)
        sinks = uses_gpt_oss_sinks(config)
        deterministic = flash_attention_deterministic()

        def _warm(flash_fn, q_shape, kv_shape, **kw) -> None:
            q = torch.randn(*q_shape, device=device, dtype=dtype, requires_grad=True)
            k = torch.randn(*kv_shape, device=device, dtype=dtype, requires_grad=True)
            v = torch.randn(*kv_shape, device=device, dtype=dtype, requires_grad=True)
            out = flash_fn(q, k, v, softmax_scale=q_shape[-1] ** -0.5, causal=True, deterministic=deterministic, **kw)
            (out[0] if isinstance(out, tuple) else out).sum().backward()  # fwd+bwd both compile

        # Two documents, so the varlen kernel compiles against a real cu_seqlens.
        cu_seqlens = torch.arange(0, 3 * seq_len, seq_len, device=device, dtype=torch.int32)
        varlen_kwargs = {
            "cu_seqlens_q": cu_seqlens,
            "cu_seqlens_k": cu_seqlens,
            "max_seqlen_q": seq_len,
            "max_seqlen_k": seq_len,
        }
        dense_fn, varlen_fn = flash_kernel("flash_attention_4", varlen=False), flash_kernel("flash_attention_4")
        # Every dispatched variant (geometry x dense/varlen x full/sliding x sink): a missed one compiles mid-run.
        variants = 0
        for (heads, kv_heads, head_dim, softcap), layer_windows in sorted(windows.items(), key=str):
            layer_kwargs = {"softcap": softcap} if softcap else {}
            if sinks:
                layer_kwargs["learnable_sink"] = torch.randn(heads, device=device, dtype=dtype)
            for window in sorted(layer_windows, key=lambda w: w or 0):
                kwargs = {**layer_kwargs, **({"window_size": (window - 1, window - 1)} if window else {})}
                _warm(dense_fn, (1, seq_len, heads, head_dim), (1, seq_len, kv_heads, head_dim), **kwargs)
                variants += 1
                if not softcap:
                    q_shape, kv_shape = (2 * seq_len, heads, head_dim), (2 * seq_len, kv_heads, head_dim)
                    _warm(varlen_fn, q_shape, kv_shape, **varlen_kwargs, **kwargs)
                    variants += 1
        torch.cuda.synchronize()
        if is_global_main_process():
            logger.info(
                f"FA4 kernel warmup done ((heads, kv_heads, head_dim, softcap)={sorted(windows, key=str)}, dtype={dtype}, "
                f"sliding_window={longest_window or None}, sinks={sinks}, deterministic={deterministic}, "
                f"variants={variants})"
            )
    except Exception as e:
        # Every rank: the skip is rank-local, and a rank without the warm-up desyncs later.
        logger.warning(f"FA4 kernel warmup skipped (non-fatal): {type(e).__name__}: {e}")
