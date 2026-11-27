"""The ``inkling2_relative_flex`` attention: Inkling 2's attention layers on compiled FlexAttention, the relative
logits read per score at their compact ``[batch, heads, q_len, rel_extent]`` size, the sinks applied from the softmax
normalizer, and causality, the sliding window, padding and packed documents in one block mask per row and forward. A
KV-cache decode step runs the eager attention."""

from __future__ import annotations

import logging
from typing import NamedTuple

import torch
import transformers.models.inkling2.modeling_inkling2 as inkling2_modeling
from torch.nn.attention.flex_attention import AuxRequest, BlockMask, create_block_mask, flex_attention
from transformers.masking_utils import ALL_MASK_ATTENTION_FUNCTIONS, eager_mask, flash_attention_mask
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

logger = logging.getLogger(__name__)

INKLING2_RELATIVE_FLEX = "inkling2_relative_flex"

# Set on the forward's ``position_ids``, so every layer and the checkpoint recompute share the masks and they are freed
# with the forward.
_BLOCK_MASKS_ATTR = "_inkling2_relative_flex_block_masks"

_ATTENTION_CLASSES = (inkling2_modeling.Inkling2SlidingAttention, inkling2_modeling.Inkling2FullAttention)

# Backward tiles of the 16-bit head_dim-128 kernels: the defaults, with a score_mod reading a captured tensor, exceed
# SM100's 227 KiB of shared memory per block. fp32 inputs keep the defaults, whose fp32 accumulation these tiles drop.
_KERNEL_OPTIONS = {"BLOCK_M1": 32, "BLOCK_N1": 64, "BLOCK_M2": 64, "BLOCK_N2": 32}

# FlexAttention's block-sparsity tile.
_TILE = 128
# Dynamo specializes an axis of size 1, so every call is one row of more than one tile: a batch runs row by row, and a
# shorter row is padded to this length.
_MIN_ROW_LENGTH = _TILE + 1

# Every call has one shape family per attention signature (heads, widths, relative extent and scaling of the layer; all
# inputs requiring grad or none), so it compiles twice per signature, a training and a forward-only graph, both at
# warm-up. Dynamic, so each micro-batch's length reuses the graph.
_compiled_flex = torch.compile(flex_attention, dynamic=True)
# Compiled, the mask function fuses into the block reduction, so no dense ``[q_len, kv_len]`` mask is built.
_compiled_block_mask = torch.compile(create_block_mask, dynamic=True)


class _WarmupCall(NamedTuple):
    """What the compiled kernel's guards key on."""

    heads: int
    kv_heads: int
    head_dim: int
    extent: int
    scaling: float


def _first_keys(position_ids: torch.Tensor, window: int | None, batch: int, length: int) -> torch.Tensor:
    """``[batch, seq]`` int32, the first key each query reaches: the start of its document (the last token at or before
    it whose position is 0), or the edge of the sliding window where that is later."""
    index = torch.arange(length, device=position_ids.device, dtype=torch.int32)
    first = torch.where(position_ids.expand(batch, -1) == 0, index, 0).cummax(dim=-1).values
    if window is not None:
        first = torch.maximum(first, index - (window - 1))
    return first


def _row_block_mask(first: torch.Tensor, valid: torch.Tensor) -> BlockMask:
    """``first`` and ``valid`` are int32 ``[seq]``. One mask function for every call, so the window and an absent
    padding mask are no guard of either compiled graph. ``valid`` is no bool tensor: with one captured, the fp32
    kernel's products round to about 1e-3 on a row of several tiles."""
    length = first.shape[0]

    def mask_mod(b, h, q_idx, kv_idx):
        return (kv_idx <= q_idx) & (kv_idx >= first[q_idx]) & (valid[kv_idx] > 0)

    with torch.no_grad(), torch.autocast(first.device.type, enabled=False):
        return _compiled_block_mask(mask_mod, B=1, H=None, Q_LEN=length, KV_LEN=length, device=first.device)


def _row_block_masks(
    position_ids: torch.Tensor, padding_mask: torch.Tensor | None, window: int | None, batch: int, length: int
) -> list[BlockMask]:
    """One mask per batch row, each over ``max(length, _MIN_ROW_LENGTH)`` tokens: a padding token past ``length``
    reaches itself alone. ``position_ids`` is ``[batch, seq]`` or one ``[1, seq]`` row every batch row shares."""
    cache = getattr(position_ids, _BLOCK_MASKS_ATTR, None)
    if cache is None:
        cache = {}
        setattr(position_ids, _BLOCK_MASKS_ATTR, cache)
    key = (window, batch, length)
    entry = cache.get(key)
    if entry is not None and entry[0] is padding_mask:
        return entry[1]

    first = _first_keys(position_ids, window, batch, length)
    if padding_mask is None:
        valid = torch.ones(batch, length, dtype=torch.int32, device=first.device)
    else:
        valid = padding_mask.bool().to(torch.int32)
    if length < _MIN_ROW_LENGTH:
        tail = torch.arange(length, _MIN_ROW_LENGTH, device=first.device, dtype=first.dtype)
        first = torch.cat([first, tail.expand(batch, -1)], dim=-1)
        valid = torch.nn.functional.pad(valid, (0, _MIN_ROW_LENGTH - length), value=1)
    # Each row's bounds as tensors of their own: the compiled graph guards on the batch size of a view's base.
    block_masks = [
        _row_block_mask(row_first.clone(), row_valid.clone())
        for row_first, row_valid in zip(first, valid, strict=True)
    ]
    cache[key] = (padding_mask, block_masks)
    return block_masks


def _relative_score_mod(position_bias: torch.Tensor):
    """Adds the ``[1, heads, q_len, rel_extent]`` logits' entry at each score's distance, none at or past
    ``rel_extent``."""
    extent = position_bias.shape[-1]

    def score_mod(score, b, h, q_idx, kv_idx):
        distance = q_idx - kv_idx
        relative = position_bias[b, h, q_idx, torch.clamp(distance, 0, extent - 1)]
        return torch.where(distance < extent, score + relative, score)

    return score_mod


def inkling2_relative_flex_mask(batch_size: int, q_length: int, kv_length: int, **kwargs) -> torch.Tensor | None:
    """The 2D padding mask the kernel reads on a full-sequence forward, as a varlen flash kernel takes it (the documents
    come off ``position_ids``); the eager mask on a KV-cache decode step, which the eager attention runs."""
    mask = flash_attention_mask if q_length == kv_length else eager_mask
    return mask(batch_size=batch_size, q_length=q_length, kv_length=kv_length, **kwargs)


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
    ``sigmoid(lse - sink)``.

    A KV-cache decode step (fewer queries than keys) arrives with the eager mask and the dense relative bias, and runs
    the eager attention. So does the composite's vision tower, whose config carries the same implementation name and
    whose attention passes neither a relative bias nor sinks."""
    if position_bias is None and sinks is None:
        return inkling2_modeling.eager_attention_forward(
            module, query, key, value, attention_mask, scaling, dropout=dropout, **kwargs
        )
    batch, _, length, _ = query.shape
    if key.shape[2] != length:
        return inkling2_modeling.eager_attention_forward(
            module,
            query,
            key,
            value,
            attention_mask,
            scaling,
            dropout=dropout,
            position_bias=position_bias,
            sinks=sinks,
        )
    if dropout:
        raise ValueError(f"{INKLING2_RELATIVE_FLEX} takes no attention dropout, got {dropout}.")
    if position_bias is None or sinks is None:
        raise ValueError(
            f"{INKLING2_RELATIVE_FLEX} serves Inkling 2's text attention layers, which pass both a relative bias and "
            f"sinks; this call passes one of them."
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
    block_masks = _row_block_masks(position_ids, attention_mask, sliding_window, batch, length)
    padding = max(_MIN_ROW_LENGTH - length, 0)
    if padding:
        query, key, value = (torch.nn.functional.pad(t, (0, 0, 0, padding)) for t in (query, key, value))
        # Padded in the layout the relative-logits patch emits, a transposed [batch, q_len, heads, rel_extent].
        position_bias = torch.nn.functional.pad(position_bias.transpose(1, 2), (0, 0, 0, 0, 0, padding)).transpose(
            1, 2
        )
    else:
        # The compiled graph guards on strides, and the layers hand over transposed views and copies alike.
        query, key, value = query.contiguous(), key.contiguous(), value.contiguous()
    records_grad = torch.is_grad_enabled() and any(t.requires_grad for t in (query, key, value, position_bias))
    if records_grad:
        # A frozen input joins as a detached leaf, so the graph never depends on which projections train.
        query, key, value, position_bias = (
            t if t.requires_grad else t.detach().requires_grad_() for t in (query, key, value, position_bias)
        )
    outputs, normalizers = [], []
    with torch.set_grad_enabled(records_grad), torch.autocast(query.device.type, enabled=False):
        for row, block_mask in enumerate(block_masks):
            out, aux = _compiled_flex(
                query[row : row + 1],
                key[row : row + 1],
                value[row : row + 1],
                score_mod=_relative_score_mod(position_bias[row : row + 1]),
                block_mask=block_mask,
                scale=scaling,
                enable_gqa=key.shape[1] != query.shape[1],
                return_aux=AuxRequest(lse=True),
                kernel_options=None if query.dtype == torch.float32 else _KERNEL_OPTIONS,
            )
            outputs.append(out)
            normalizers.append(aux.lse)
    out = torch.cat(outputs)[:, :, :length]
    lse = torch.cat(normalizers)[:, :, :length]
    # A query with no key in reach has lse -inf, so a zero share: it attends to the sink alone, as on the eager path.
    sink_share = torch.sigmoid(lse - sinks.squeeze(-1).float())
    out = (out.float() * sink_share.unsqueeze(-1)).to(query.dtype)
    return out.transpose(1, 2).contiguous(), None


def register_inkling2_relative_flex() -> str:
    if INKLING2_RELATIVE_FLEX not in ALL_ATTENTION_FUNCTIONS:
        ALL_ATTENTION_FUNCTIONS.register(INKLING2_RELATIVE_FLEX, inkling2_relative_flex_attention)
        ALL_MASK_ATTENTION_FUNCTIONS.register(INKLING2_RELATIVE_FLEX, inkling2_relative_flex_mask)
        logger.info(
            f"Registered attention '{INKLING2_RELATIVE_FLEX}': Inkling 2's relative logits and sinks on compiled "
            "block-sparse FlexAttention."
        )
    return INKLING2_RELATIVE_FLEX


def _warmup_calls(model: torch.nn.Module) -> set[_WarmupCall]:
    return {
        _WarmupCall(
            module.num_heads,
            module.num_key_value_heads,
            module.head_dim,
            module.rel_extent,
            module.scaling,
        )
        for module in model.modules()
        if isinstance(module, _ATTENTION_CLASSES) and module.config._attn_implementation == INKLING2_RELATIVE_FLEX
    }


def warm_inkling2_relative_flex_kernels(
    model: torch.nn.Module, *, dtype: torch.dtype, device: torch.device | None = None
) -> None:
    """Compile the block-mask graph and, per attention signature of the model, the training and forward-only graphs,
    so no rank compiles one mid-run. Rank-local; the caller fences it."""
    calls = _warmup_calls(model)
    if not calls:
        return
    if device is None:
        device = torch.device(f"cuda:{torch.cuda.current_device()}")
    for call in sorted(calls, key=str):
        span = max(call.extent, _TILE)
        seq = 2 * span + 3  # past the relative extent (a sliding layer's window), off the tile edge; two documents
        position_ids = torch.cat([torch.arange(span), torch.arange(seq - span)]).to(device).unsqueeze(0)
        for records_grad in (True, False):

            def zeros(*shape, records_grad=records_grad) -> torch.Tensor:
                return torch.zeros(shape, device=device, dtype=dtype, requires_grad=records_grad)

            out, _ = inkling2_relative_flex_attention(
                None,
                zeros(1, call.heads, seq, call.head_dim),
                zeros(1, call.kv_heads, seq, call.head_dim),
                zeros(1, call.kv_heads, seq, call.head_dim),
                None,
                call.scaling,
                # Laid out as the relative-logits patch emits it: the compiled graph guards on the strides.
                position_bias=zeros(1, seq, call.heads, call.extent).transpose(1, 2),
                sinks=zeros(1, call.heads, seq, 1),
                position_ids=position_ids,
            )
            if records_grad:
                out.sum().backward()
    logger.info(
        f"Compiled the {INKLING2_RELATIVE_FLEX} FlexAttention kernels of {len(calls)} call(s) (training and "
        "forward-only)"
    )
