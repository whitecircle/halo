"""Process-wide attention backend fixes (flash varlen seqlens, wide-head SDPA, flex compile) and the FA4 warm-up."""

from __future__ import annotations

import functools
import logging

import torch
from transformers import PreTrainedModel
from transformers import modeling_flash_attention_utils as flash_utils
from transformers.integrations import sdpa_attention as _sdpa_mod
from transformers.integrations.flex_attention import WrappedFlexAttention, flex_attention

from src.models.attention.gpt_oss_sinks import uses_gpt_oss_sinks
from src.models.attention.layout import resolve_head_dim, resolve_num_key_value_heads
from src.models.attention.selection import FLASH_MAX_HEAD_DIM, effective_attn_implementation, flash_kernel
from src.models.config_access import text_config
from src.runtime.ranks import is_global_main_process

logger = logging.getLogger(__name__)

# Warm-up row length: the floor keeps a kernel on a realistic tile count, the margin pushes the row past a
# sliding window so the windowed variant is the one that compiles.
WARMUP_MIN_SEQ_LEN = 512
_WARMUP_WINDOW_MARGIN = 8


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
    """Pre-compile this model's FA4 forward and backward kernels, dense and varlen, at the run's ``dtype``.

    Rank-local and best-effort: a first-use JIT compile on one rank lets its peers race into the next
    collective, so the caller fences every rank around it and nothing here may gate a collective.
    """
    try:
        config = getattr(model, "config", None)
        if config is None:
            return
        text_cfg = text_config(config)
        if effective_attn_implementation(config) != "flash_attention_4":
            return

        head_dim = resolve_head_dim(config)
        n_heads = text_cfg.num_attention_heads
        n_kv_heads = resolve_num_key_value_heads(config)
        sliding_window = getattr(text_cfg, "sliding_window", None)
        if device is None:
            device = torch.device(f"cuda:{torch.cuda.current_device()}")
        seq_len = max(WARMUP_MIN_SEQ_LEN, (sliding_window or 0) + _WARMUP_WINDOW_MARGIN)
        scale = head_dim**-0.5
        sink = torch.randn(n_heads, device=device, dtype=dtype) if uses_gpt_oss_sinks(config) else None

        def _warm(flash_fn, q_shape, kv_shape, **kw) -> None:
            q = torch.randn(*q_shape, device=device, dtype=dtype, requires_grad=True)
            k = torch.randn(*kv_shape, device=device, dtype=dtype, requires_grad=True)
            v = torch.randn(*kv_shape, device=device, dtype=dtype, requires_grad=True)
            out = flash_fn(q, k, v, softmax_scale=scale, causal=True, **kw)
            (out[0] if isinstance(out, tuple) else out).sum().backward()  # fwd+bwd both compile

        # Two documents, so the varlen kernel compiles against a real cu_seqlens.
        cu_seqlens = torch.arange(0, 3 * seq_len, seq_len, device=device, dtype=torch.int32)
        entry_points = (
            (
                flash_kernel("flash_attention_4", varlen=False),
                (1, seq_len, n_heads, head_dim),
                (1, seq_len, n_kv_heads, head_dim),
                {},
            ),
            (
                flash_kernel("flash_attention_4"),
                (2 * seq_len, n_heads, head_dim),
                (2 * seq_len, n_kv_heads, head_dim),
                {
                    "cu_seqlens_q": cu_seqlens,
                    "cu_seqlens_k": cu_seqlens,
                    "max_seqlen_q": seq_len,
                    "max_seqlen_k": seq_len,
                },
            ),
        )
        # Every dispatched variant (dense/varlen x full/sliding x sink): a missed one compiles mid-run.
        window_variants = [{}] + (
            [{"window_size": (sliding_window - 1, sliding_window - 1)}] if sliding_window else []
        )
        sink_kwargs = {} if sink is None else {"learnable_sink": sink}
        for flash_fn, q_shape, kv_shape, shape_kwargs in entry_points:
            for window_kwargs in window_variants:
                _warm(flash_fn, q_shape, kv_shape, **shape_kwargs, **window_kwargs, **sink_kwargs)
        torch.cuda.synchronize()
        if is_global_main_process():
            logger.info(
                f"FA4 kernel warmup done (head_dim={head_dim}, dtype={dtype}, sliding_window={sliding_window}, "
                f"sinks={sink is not None}, variants={len(entry_points) * len(window_variants)})"
            )
    except Exception as e:
        # Every rank: the skip is rank-local, and a rank without the warm-up desyncs later.
        logger.warning(f"FA4 kernel warmup skipped (non-fatal): {type(e).__name__}: {e}")
