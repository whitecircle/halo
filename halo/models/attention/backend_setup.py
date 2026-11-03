"""Process-wide attention backend fixes (flash varlen seqlens and unpad memo, FA4's varlen softcap refusal,
wide-head SDPA, flex compile) and the FA4 warm-up."""

from __future__ import annotations

import functools
import inspect
import logging

import torch
from transformers import PreTrainedModel, masking_utils
from transformers import modeling_flash_attention_utils as flash_utils
from transformers.integrations import sdpa_attention as _sdpa_mod
from transformers.integrations.flex_attention import WrappedFlexAttention, flex_attention

from halo.models.attention.fa4_entry_points import install_fa4_entry_point_wrapper
from halo.models.attention.gpt_oss_sinks import uses_gpt_oss_sinks
from halo.models.attention.layout import attention_calls
from halo.models.attention.selection import (
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

_FA4_VARLEN_SOFTCAP_ATTR = "_halo_fa4_varlen_softcap_refusal"
# On the padding mask, as ``(version, set)``.
_UNPAD_MEMO_ATTR = "_halo_unpad_data"


def patch_transformers_flash_varlen_int_seqlen() -> None:
    """Pass varlen ``max_seqlen_{q,k}`` as python ints. FA4 compiles its head_dim-256 varlen kernels one way for a
    host-int bound and another for a tensor one; the warm-up warms the int variant, at one ``.item()`` per call."""
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


def patch_transformers_flash_unpad_memo() -> None:
    """Derive a padding mask's unpad set once per mask instead of once per layer, each derivation being a ``nonzero``
    and a host read. Kept on the mask tensor, keyed on its version counter; a single-entry ``TensorMemo`` would derive
    again in every layer of a model that alternates sliding and full layers."""
    fn = flash_utils._get_unpad_data
    if getattr(fn, "_halo_unpad_memo", False):
        return
    if "_get_unpad_data" not in flash_utils._upad_input.__code__.co_names:
        raise RuntimeError(
            "transformers' _upad_input no longer derives its unpad set through the module-level _get_unpad_data, "
            "so patch_transformers_flash_unpad_memo cannot reach it; move the memo to the function _upad_input calls."
        )

    @torch.compiler.disable
    @functools.wraps(fn)
    def memoized(attention_mask):
        if attention_mask.is_inference():
            return fn(attention_mask)
        memo = getattr(attention_mask, _UNPAD_MEMO_ATTR, None)
        if memo is None or memo[0] != attention_mask._version:
            memo = (attention_mask._version, fn(attention_mask))
            setattr(attention_mask, _UNPAD_MEMO_ATTR, memo)
        return memo[1]

    memoized._halo_unpad_memo = True
    flash_utils._get_unpad_data = memoized


def _with_varlen_softcap_refusal(fn, varlen: bool):
    """The dense entry point, which trains a softcap, is returned as is."""
    if not varlen:
        return fn

    @functools.wraps(fn)
    def wrapped(*args, **kwargs):
        if kwargs.get("softcap"):
            raise RuntimeError(
                "A softcapped attention call reached FlashAttention-4's varlen kernels, which are not validated with "
                "attention logit softcapping. A packed, padding-free or padded batch, training or evaluation, reaches "
                "them. Leave attn_implementation unset or set flash_attention_2; FA4 trains a softcapped model only "
                "on one unpadded row per batch."
            )
        return fn(*args, **kwargs)

    return wrapped


def install_fa4_varlen_softcap_refusal() -> bool:
    return install_fa4_entry_point_wrapper(
        _FA4_VARLEN_SOFTCAP_ATTR, _with_varlen_softcap_refusal, "the softcap refusal on the varlen kernel"
    )


def patch_sdpa_for_wide_heads() -> None:
    """FA2 and cuDNN SDPA reject a head past :data:`FLASH_MAX_HEAD_DIM` and the math kernel OOMs on the full score
    matrix. GQA takes the manual KV repeat, the measured path, as native ``enable_gqa`` under mem-efficient-only is
    unverified on the narrower sliding heads. Process-wide and idempotent."""
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


def patch_transformers_packed_sequence_probe() -> None:
    """Skip the packed-sequence probe for a ``flash_attention_mask`` model, as it reads a device flag back at the start
    of each forward and the flash path finds the documents in ``position_ids`` itself. Other backends keep it."""
    fn = masking_utils._preprocess_mask_arguments
    if getattr(fn, "_halo_packed_probe_skip", False):
        return

    @functools.wraps(fn)
    def preprocess(
        config, inputs_embeds, attention_mask, past_key_values, position_ids, layer_idx, encoder_hidden_states=None
    ):
        interfaces = masking_utils.ALL_MASK_ATTENTION_FUNCTIONS
        impl = config._attn_implementation
        # An implementation outside the global table exits before the probe, and indexing it would raise.
        if impl in interfaces._global_mapping and interfaces[impl] is masking_utils.flash_attention_mask:
            position_ids = None
        return fn(
            config, inputs_embeds, attention_mask, past_key_values, position_ids, layer_idx, encoder_hidden_states
        )

    expected, found = inspect.signature(preprocess).parameters, inspect.signature(fn).parameters
    if list(found) != list(expected) or found["encoder_hidden_states"].default is not None:
        raise RuntimeError(
            f"transformers' _preprocess_mask_arguments takes ({', '.join(found)}), not the "
            f"({', '.join(expected)}) patch_transformers_packed_sequence_probe forwards; update its replacement."
        )
    preprocess._halo_packed_probe_skip = True
    masking_utils._preprocess_mask_arguments = preprocess


def patch_flex_attention_compile(reason: str):
    """Call before the first flex forward. The compiled wrapper's seq-length recompiles deadlock EP's all-to-all, and
    its backward NaNs under FSDP2 with GptOss sinks."""
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
    """Rank-local and best-effort. A first-use JIT compile on one rank lets its peers race into the next collective, so
    the caller fences every rank around it and nothing here may gate a collective. A softcapped layer warms its dense
    call only, as a softcap stays off FA4's varlen kernels."""
    try:
        config = getattr(model, "config", None)
        if config is None:
            return
        if effective_attn_implementation(config) != "flash_attention_4":
            return

        # With and without the window, as a row no longer than it dispatches the windowless kernel. transformers' flash
        # path pads the value up to the query/key width, so the value width keys no kernel of its own.
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
            (out[0] if isinstance(out, tuple) else out).sum().backward()

        # Two documents, so the varlen kernel compiles against a real cu_seqlens.
        cu_seqlens = torch.arange(0, 3 * seq_len, seq_len, device=device, dtype=torch.int32)
        varlen_kwargs = {
            "cu_seqlens_q": cu_seqlens,
            "cu_seqlens_k": cu_seqlens,
            "max_seqlen_q": seq_len,
            "max_seqlen_k": seq_len,
        }
        dense_fn, varlen_fn = flash_kernel("flash_attention_4", varlen=False), flash_kernel("flash_attention_4")
        # Geometry x dense/varlen x full/sliding x sink; a missed variant compiles mid-run.
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
        # On every rank, as the skip is rank-local and a rank without the warm-up desyncs later.
        logger.warning(f"FA4 kernel warmup skipped (non-fatal): {type(e).__name__}: {e}")
