"""Sliding-window layers on compiled FlexAttention, SDPA-hostile global layers on matmul attention.

Under SDPA a sliding-window layer receives transformers' dense boolean mask, so the kernel computes every
masked-out tile of the window. FlexAttention takes the same mask as a block-sparse ``BlockMask`` and skips
the tiles outside the window (and across packed-document boundaries). The block mask is derived from the
mask transformers already built, so every masking rule (causality, the window, per-document isolation of
packed rows, padding) is inherited unchanged, and one block mask per forward is shared by every sliding
layer.

A causal global layer runs plain matmul-softmax-matmul attention where SDPA has only its mem-efficient
kernel (a head dim beyond flash's 256, or the flash backend disabled for the process) and the saved score
matrices fit :data:`EAGER_GLOBAL_BUDGET_BYTES`; mem-efficient SDPA serves it beyond that budget. On Gemma 4
(head dim 512 on the global layers, whose loader pins mem-efficient SDPA process-wide) matmul attention is
3.3x faster than mem-efficient SDPA at 2,048 tokens on B300 (16 heads at 2,048 tokens stay well inside the
budget; at 8,192 tokens they exceed it and the layer runs mem-efficient SDPA). FlexAttention does not
compete there: at head dim 512 only its smallest tiles fit shared memory, and those are slower still.

Registered as the attention implementation :data:`FLEX_SLIDING` with SDPA's mask builder, and chosen by
:func:`resolve_flex_sliding_attn_implementation` for any model whose run resolved to ``sdpa`` and that has
sliding layers or heads wider than flash supports. Every call outside those two cases, and every call
carrying something FlexAttention does not model here (attention sinks, a logit softcap, a position bias,
dropout, a float mask), goes to the implementation registered as ``sdpa``.
"""

from __future__ import annotations

import logging
import weakref

import torch
from torch.nn.attention.flex_attention import BlockMask, create_block_mask, flex_attention
from transformers.masking_utils import ALL_MASK_ATTENTION_FUNCTIONS, sdpa_mask
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

from src.env import env_flag
from src.models.loading.config_levels import text_config
from src.models.patches.attention import model_has_sinks

logger = logging.getLogger(__name__)

FLEX_SLIDING = "sdpa_flex_sliding"

# The widest head flash SDPA (and FlashAttention 2) runs; wider heads leave SDPA only its mem-efficient kernel.
FLASH_SDPA_MAX_HEAD_DIM = 256

# Attention-function keywords FlexAttention is not given here; a call carrying any of them keeps SDPA.
_SDPA_ONLY_KWARGS = ("s_aux", "softcap", "position_bias")

# Saved-activation bytes (fp32 softmax output + bf16 probabilities, 6 B per score) one global layer may
# spend on matmul attention before falling back to mem-efficient SDPA. 2 GiB covers 16 heads at 4,096
# tokens per micro-batch row.
EAGER_GLOBAL_BUDGET_BYTES = 2 * 2**30

# FlexAttention tiles for sliding layers of head_dim 256 on SM100+, as measured on Gemma 4's sliding layers
# (window 1,024). Fwd+bwd on B300 at 2,048 / 8,192 / 32,768 tokens: 0.46 / 1.46 / 6.07 ms against 0.63 /
# 2.66 / 9.92 ms with the default tiles (and 0.78 / 1.74 / 6.98 ms for FlashAttention 2's local window).
# The choice ignores the sequence length on purpose: a length-dependent choice is a guard the compiled
# function recompiles on whenever a batch crosses it.
_SM100_SLIDING_KERNEL_OPTIONS = {
    "BLOCK_M": 128,
    "BLOCK_N": 64,
    "BLOCK_M1": 32,
    "BLOCK_N1": 64,
    "BLOCK_M2": 64,
    "BLOCK_N2": 32,
    "num_warps": 8,
    "num_stages": 2,
}
_TUNED_HEAD_DIM = 256


def _sliding_kernel_options(query: torch.Tensor) -> dict | None:
    """The tuned tiles where they were measured (16-bit inputs, head_dim 256, SM100+), else the defaults."""
    if (
        query.dtype not in (torch.bfloat16, torch.float16)
        or query.shape[-1] != _TUNED_HEAD_DIM
        or torch.cuda.get_device_capability(query.device)[0] < 10
    ):
        return None
    return _SM100_SLIDING_KERNEL_OPTIONS


# Shape-dynamic from the first call: one compilation per process serves every batch and sequence length.
# Packed batches change length every step; a static compile would recompile on each new length until
# Dynamo's recompile limit, then run FlexAttention unfused with the full score matrix. Under expert
# parallelism a mid-run recompile on one rank also stalls the others inside EP's NCCL all-to-all, which is
# why the loader disables transformers' own compiled flex_attention there; with a single compilation at the
# first step on every rank, this one has no mid-run recompile to stall on.
_compiled_flex = torch.compile(flex_attention, dynamic=True)

# The last block mask built and what it was built from. Every sliding layer of one forward (and of its
# gradient-checkpoint recompute) receives the same mask tensor, so they share one BlockMask. The mask is
# matched by identity through a weak reference: a later step's mask can reuse the freed storage (same
# data_ptr, version 0), so address-based keys would serve a stale mask.
_block_mask_cache: tuple[weakref.ref | None, tuple, BlockMask] | None = None


def sliding_block_mask(
    attention_mask: torch.Tensor | None, batch: int, q_len: int, kv_len: int, window: int, device: torch.device
) -> BlockMask:
    """The ``BlockMask`` equal to transformers' dense sliding-window mask.

    ``attention_mask`` is SDPA's boolean ``[B, 1, Q, K]`` mask (``True`` = attend), or ``None`` when
    transformers proved the mask purely causal (no padding, no packing, length within the window), in
    which case the causal window is built directly.
    """
    global _block_mask_cache
    shape_key = (batch, q_len, kv_len, window, device)
    if _block_mask_cache is not None:
        mask_ref, cached_shape, cached = _block_mask_cache
        same_mask = attention_mask is None if mask_ref is None else mask_ref() is attention_mask
        if same_mask and cached_shape == shape_key:
            return cached
    if attention_mask is None:
        offset = kv_len - q_len

        def mask_mod(b, h, q_idx, kv_idx):
            q_pos = q_idx + offset
            return (kv_idx <= q_pos) & (q_pos - kv_idx < window)

        block_mask = create_block_mask(mask_mod, None, None, q_len, kv_len, device=device)
    else:
        dense = attention_mask.to(device=device, dtype=torch.bool)
        heads_broadcast = dense.shape[1] == 1

        def mask_mod(b, h, q_idx, kv_idx):
            return dense[b, 0 if heads_broadcast else h, q_idx, kv_idx]

        block_mask = create_block_mask(
            mask_mod, dense.shape[0], None if heads_broadcast else dense.shape[1], q_len, kv_len, device=device
        )
    _block_mask_cache = (None if attention_mask is None else weakref.ref(attention_mask), shape_key, block_mask)
    return block_mask


def _matmul_global_attention(
    query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, attention_mask: torch.Tensor | None, scaling: float
) -> torch.Tensor:
    """Causal (or ``attention_mask``-masked) attention as two matmuls and an fp32 softmax.

    GQA without a KV copy: the query heads of each KV group are stacked on the sequence axis, so each KV
    head is multiplied once. Masked scores take the dtype minimum rather than ``-inf`` (transformers' eager
    convention), so no row can softmax to NaN. Returns ``[B, S, H, D]``.
    """
    batch, heads, q_len, dim = query.shape
    kv_heads, kv_len = key.shape[1], key.shape[-2]
    group = heads // kv_heads
    q = query.reshape(batch, kv_heads, group * q_len, dim)
    scores = torch.matmul(q, key.transpose(-1, -2)).view(batch, kv_heads, group, q_len, kv_len)
    if scaling != 1.0:
        scores = scores * scaling
    if attention_mask is None:
        allowed = torch.ones(q_len, kv_len, dtype=torch.bool, device=query.device).tril(kv_len - q_len)
    else:
        allowed = (
            attention_mask[:, :, None]
            if attention_mask.shape[1] == 1
            else attention_mask.view(batch, kv_heads, group, q_len, kv_len)
        )
    scores = scores.masked_fill(~allowed, torch.finfo(scores.dtype).min)
    probs = torch.softmax(scores, dim=-1, dtype=torch.float32).to(query.dtype)
    out = torch.matmul(probs.view(batch, kv_heads, group * q_len, kv_len), value)
    return out.view(batch, heads, q_len, dim).transpose(1, 2).contiguous()


def _fits_eager_budget(query: torch.Tensor, kv_len: int) -> bool:
    batch, heads, q_len, _ = query.shape
    return batch * heads * q_len * kv_len * 6 <= EAGER_GLOBAL_BUDGET_BYTES


def _sdpa_is_mem_efficient_only(head_dim: int) -> bool:
    """Whether SDPA would run this head on its mem-efficient kernel alone."""
    return head_dim > FLASH_SDPA_MAX_HEAD_DIM or not torch.backends.cuda.flash_sdp_enabled()


def flex_sliding_attention(
    module: torch.nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    dropout: float = 0.0,
    scaling: float | None = None,
    sliding_window: int | None = None,
    **kwargs,
) -> tuple[torch.Tensor, None]:
    """Sliding layers on FlexAttention, SDPA-hostile causal global layers on matmul attention within the
    budget, every other call on the implementation registered as ``sdpa`` (see the module docstring)."""
    boolean_mask = attention_mask is None or attention_mask.dtype == torch.bool
    eligible = (
        query.is_cuda
        and not dropout
        and not kwargs.get("output_attentions", False)
        and boolean_mask
        and getattr(module, "is_causal", False)
        and all(kwargs.get(name) is None for name in _SDPA_ONLY_KWARGS)
    )
    if (
        eligible
        and sliding_window is None
        and query.shape[-1] == value.shape[-1]
        and _sdpa_is_mem_efficient_only(query.shape[-1])
        and _fits_eager_budget(query, key.shape[-2])
    ):
        scale = query.shape[-1] ** -0.5 if scaling is None else scaling
        return _matmul_global_attention(query, key, value, attention_mask, scale), None
    if sliding_window is None or not eligible:
        return ALL_ATTENTION_FUNCTIONS["sdpa"](
            module, query, key, value, attention_mask, dropout=dropout, scaling=scaling, **kwargs
        )
    batch, _, q_len, _ = query.shape
    kv_len = key.shape[-2]
    block_mask = sliding_block_mask(attention_mask, batch, q_len, kv_len, sliding_window, query.device)
    out = _compiled_flex(
        query,
        key,
        value,
        block_mask=block_mask,
        scale=scaling,
        enable_gqa=query.shape[1] != key.shape[1],
        kernel_options=_sliding_kernel_options(query),
    )
    return out.transpose(1, 2).contiguous(), None


def register_flex_sliding_attention() -> str:
    """Register :data:`FLEX_SLIDING` (idempotent) and return its name for ``attn_implementation``."""
    if FLEX_SLIDING not in ALL_ATTENTION_FUNCTIONS:
        ALL_ATTENTION_FUNCTIONS.register(FLEX_SLIDING, flex_sliding_attention)
        ALL_MASK_ATTENTION_FUNCTIONS.register(FLEX_SLIDING, sdpa_mask)
        logger.info(
            f"Registered attention '{FLEX_SLIDING}': sliding-window layers on compiled FlexAttention "
            "(block-sparse window), mem-efficient-only global layers on matmul attention, the rest on SDPA."
        )
    return FLEX_SLIDING


def model_has_sliding_layers(model_config) -> bool:
    """Whether any decoder layer attends through a sliding window (per ``layer_types``, or a window every
    layer uses unless ``use_sliding_window`` turns it off)."""
    config = text_config(model_config)
    if not getattr(config, "sliding_window", None):
        return False
    layer_types = getattr(config, "layer_types", None)
    if layer_types:
        return "sliding_attention" in layer_types
    return bool(getattr(config, "use_sliding_window", True))


def model_has_wide_heads(model_config) -> bool:
    """Whether any attention head is wider than flash SDPA runs."""
    config = text_config(model_config)
    if "head_dim" in (getattr(config, "per_layer_attributes", None) or ()):
        # A heterogeneous config (Gemma 4: 256 on sliding layers, 512 on global ones) has no single
        # `head_dim`; reading it raises. Its per-layer view carries each layer's width.
        widths = [getattr(layer, "head_dim", None) for layer in config.per_layer_config]
    else:
        widths = [getattr(config, name, None) for name in ("head_dim", "global_head_dim")]
    return any(width and width > FLASH_SDPA_MAX_HEAD_DIM for width in widths)


def resolve_flex_sliding_attn_implementation(model_config, attn_implementation: str) -> str:
    """The implementation a model is built with: :data:`FLEX_SLIDING` where the run resolved to ``sdpa`` on
    CUDA and the model has sliding layers or wide heads, the resolved value otherwise. A sinks model keeps
    ``sdpa``: every one of its calls carries the sinks, which FlexAttention is not given here.
    ``HALO_FLEX_SLIDING=0`` keeps plain SDPA."""
    if (
        attn_implementation != "sdpa"
        or not torch.cuda.is_available()
        or model_has_sinks(model_config)
        or not env_flag("HALO_FLEX_SLIDING", True)
        or not (model_has_sliding_layers(model_config) or model_has_wide_heads(model_config))
    ):
        return attn_implementation
    return register_flex_sliding_attention()
