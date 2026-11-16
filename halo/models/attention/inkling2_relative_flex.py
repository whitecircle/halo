"""The ``inkling2_relative_flex`` attention: Inkling 2's attention layers on compiled FlexAttention, the relative
logits read per score at their compact ``[batch, heads, q_len, rel_extent]`` size, the sinks applied from the softmax
normalizer, and causality, the sliding window, padding and packed documents in one block mask per forward."""

from __future__ import annotations

import logging

import torch
from torch.nn.attention.flex_attention import AuxRequest, BlockMask, create_block_mask, flex_attention
from transformers.masking_utils import ALL_MASK_ATTENTION_FUNCTIONS, flash_attention_mask
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

logger = logging.getLogger(__name__)

INKLING2_RELATIVE_FLEX = "inkling2_relative_flex"

# Set on the forward's ``position_ids``, so every layer and the checkpoint recompute share the masks and they are freed
# with the forward.
_BLOCK_MASKS_ATTR = "_inkling2_relative_flex_block_masks"

# Backward tiles of the 16-bit head_dim-128 kernels: the defaults, with a score_mod reading a captured tensor, exceed
# SM100's 227 KiB of shared memory per block. fp32 inputs keep the defaults, whose fp32 accumulation these tiles drop.
_KERNEL_OPTIONS = {"BLOCK_M1": 32, "BLOCK_N1": 64, "BLOCK_M2": 64, "BLOCK_N2": 32}

# One shape family per call signature; dynamic, so each micro-batch's length reuses the graph.
_compiled_flex = torch.compile(flex_attention, dynamic=True)


def _documents(position_ids: torch.Tensor) -> torch.Tensor:
    """``[batch, seq]`` document index of each token: a new document starts wherever the position restarts at 0."""
    return (position_ids == 0).cumsum(dim=-1)


def _block_mask(
    position_ids: torch.Tensor, padding_mask: torch.Tensor | None, window: int | None, batch: int, length: int
) -> BlockMask:
    """``position_ids`` is ``[batch, seq]`` or one ``[1, seq]`` row every batch row shares."""
    cache = getattr(position_ids, _BLOCK_MASKS_ATTR, None)
    if cache is None:
        cache = {}
        setattr(position_ids, _BLOCK_MASKS_ATTR, cache)
    key = (window, batch, length)
    entry = cache.get(key)
    if entry is not None and entry[0] is padding_mask:
        return entry[1]

    documents = _documents(position_ids.expand(batch, -1))
    valid = None if padding_mask is None else padding_mask.bool()

    def mask_mod(b, h, q_idx, kv_idx):
        keep = (q_idx >= kv_idx) & (documents[b, q_idx] == documents[b, kv_idx])
        if valid is not None:
            keep = keep & valid[b, kv_idx]
        if window is not None:
            keep = keep & (q_idx - kv_idx < window)
        return keep

    block_mask = create_block_mask(mask_mod, B=batch, H=None, Q_LEN=length, KV_LEN=length, device=documents.device)
    cache[key] = (padding_mask, block_mask)
    return block_mask


def inkling2_relative_flex_attention(
    module: torch.nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    scaling: float,
    dropout: float = 0.0,
    sliding_window: int | None = None,
    position_bias: torch.Tensor | None = None,
    sinks: torch.Tensor | None = None,
    position_ids: torch.Tensor | None = None,
    **kwargs,
) -> tuple[torch.Tensor, None]:
    """``query`` ``[B, H, Q, D]``, ``key``/``value`` ``[B, KV, Q, D]``, ``position_bias`` the compact
    ``[B, H, Q, rel_extent]`` logits, ``sinks`` ``[B, H, Q, 1]``, ``attention_mask`` the 2D padding mask or None.

    The ``score_mod`` adds the bias entry at each score's distance, none at or past ``rel_extent``, so no
    ``[q_len, kv_len]`` bias or score plane is built. The sink column, logit ``sink`` and value zero, takes the share
    ``exp(sink) / (exp(lse) + exp(sink))`` of the softmax mass, so the output is the sink-free attention scaled by
    ``sigmoid(lse - sink)``."""
    batch, _, length, _ = query.shape
    if key.shape[2] != length:
        raise ValueError(
            f"{INKLING2_RELATIVE_FLEX} runs full-sequence forwards (training and scoring); a forward with a KV cache "
            f"({length} queries over {key.shape[2]} keys) runs on attn_implementation='eager'."
        )
    if dropout:
        raise ValueError(f"{INKLING2_RELATIVE_FLEX} takes no attention dropout, got {dropout}.")
    if position_bias is None or sinks is None:
        raise ValueError(
            f"{INKLING2_RELATIVE_FLEX} serves Inkling 2's attention layers, which pass both a relative bias and sinks."
        )
    extent = position_bias.shape[-1]
    if position_bias.shape != (batch, query.shape[1], length, extent):
        raise ValueError(
            f"{INKLING2_RELATIVE_FLEX} reads the compact [batch, heads, q_len, rel_extent] relative logits, got "
            f"{tuple(position_bias.shape)}: the relative-logits patch is not active on this layer."
        )
    if position_ids is None:
        position_ids = torch.arange(length, device=query.device).unsqueeze(0)
    if attention_mask is not None and attention_mask.dim() != 2:
        raise ValueError(f"{INKLING2_RELATIVE_FLEX} takes a 2D padding mask, got shape {tuple(attention_mask.shape)}.")
    block_mask = _block_mask(position_ids, attention_mask, sliding_window, batch, length)

    def score_mod(score, b, h, q_idx, kv_idx):
        distance = q_idx - kv_idx
        relative = position_bias[b, h, q_idx, torch.clamp(distance, 0, extent - 1)]
        return torch.where(distance < extent, score + relative, score)

    out, aux = _compiled_flex(
        query,
        key,
        value,
        score_mod=score_mod,
        block_mask=block_mask,
        scale=scaling,
        enable_gqa=key.shape[1] != query.shape[1],
        return_aux=AuxRequest(lse=True),
        kernel_options=None if query.dtype == torch.float32 else _KERNEL_OPTIONS,
    )
    # A query with no key in reach has lse -inf, so a zero share: it attends to the sink alone, as on the eager path.
    sink_share = torch.sigmoid(aux.lse - sinks.squeeze(-1).float())
    out = (out.float() * sink_share.unsqueeze(-1)).to(query.dtype)
    return out.transpose(1, 2).contiguous(), None


def register_inkling2_relative_flex() -> str:
    if INKLING2_RELATIVE_FLEX not in ALL_ATTENTION_FUNCTIONS:
        ALL_ATTENTION_FUNCTIONS.register(INKLING2_RELATIVE_FLEX, inkling2_relative_flex_attention)
        # The kernel reads the 2D padding mask and the documents off position_ids, as a varlen flash kernel does.
        ALL_MASK_ATTENTION_FUNCTIONS.register(INKLING2_RELATIVE_FLEX, flash_attention_mask)
        logger.info(
            f"Registered attention '{INKLING2_RELATIVE_FLEX}': Inkling 2's relative logits and sinks on compiled "
            "block-sparse FlexAttention."
        )
    return INKLING2_RELATIVE_FLEX
