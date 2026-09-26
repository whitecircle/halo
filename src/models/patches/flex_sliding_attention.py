"""Sliding-window layers on compiled FlexAttention, SDPA-hostile global layers on matmul attention.

Under SDPA a sliding-window layer receives transformers' dense boolean mask, so the kernel computes every
masked-out tile of the window. FlexAttention takes the same mask as a block-sparse ``BlockMask`` and skips
the tiles outside the window (and across packed-document boundaries). The block mask is derived from the
mask transformers already built, so every masking rule (causality, the window, per-document isolation of
packed rows, padding) is inherited unchanged, and one set of block masks per forward is shared by every
sliding layer.

A causal global layer runs plain matmul-softmax-matmul attention where SDPA has only its mem-efficient
kernel (a head dim beyond flash's 256, or the flash backend disabled for the process) and the saved score
matrices fit :data:`EAGER_GLOBAL_BUDGET_BYTES`; mem-efficient SDPA serves it beyond that budget. On Gemma 4
(head dim 512 on the global layers, whose loader pins mem-efficient SDPA process-wide) matmul attention is
faster than mem-efficient SDPA while the scores fit the budget (at 8,192 tokens they exceed it and the layer
runs mem-efficient SDPA). FlexAttention does not compete there: at head dim 512 only its smallest tiles fit
shared memory, and those are slower still.

Registered as the attention implementation :data:`FLEX_SLIDING` with SDPA's mask builder, and chosen by
:func:`resolve_flex_sliding_attn_implementation` for a model whose run resolved to ``sdpa`` and that has
heads wider than flash supports (Gemma 4). Every call outside those two cases (a sliding call no longer
than its window included), and every call carrying something FlexAttention does not model here (attention
sinks, a logit softcap, a position bias, dropout, a float mask), goes to the implementation registered as
``sdpa``.
"""

from __future__ import annotations

import logging

import torch
from torch.nn.attention.flex_attention import BlockMask, flex_attention
from transformers.masking_utils import ALL_MASK_ATTENTION_FUNCTIONS, sdpa_mask
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

from src.env import env_flag
from src.hardware import is_blackwell_gpu
from src.models.patches.attention import (
    FLASH_MAX_HEAD_DIM,
    effective_attn_implementation,
    head_dim_exceeds_flash,
    model_has_sinks,
)

logger = logging.getLogger(__name__)

FLEX_SLIDING = "sdpa_flex_sliding"

# Attention-function keywords FlexAttention is not given here; a call carrying any of them keeps SDPA.
_SDPA_ONLY_KWARGS = ("s_aux", "softcap", "position_bias")

# Bytes matmul attention saves for backward per score: the fp32 softmax output and the 16-bit
# probabilities.
_SAVED_BYTES_PER_SCORE = 6
# Saved-activation bytes one global layer may spend on matmul attention before falling back to
# mem-efficient SDPA. The budget is per layer: a model with several global layers and no gradient
# checkpointing holds it once per layer. 2 GiB covers 16 heads at 4,096 tokens per micro-batch row.
EAGER_GLOBAL_BUDGET_BYTES = 2 * 2**30

# FlexAttention tiles for sliding layers of head_dim 256 on SM100+, tuned on Gemma 4's sliding layers
# (window 1,024), where they beat the default tiles at every length. The choice ignores the sequence length
# on purpose: a length-dependent choice is a guard the compiled function recompiles on whenever a batch
# crosses it.
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
    """The tuned tiles where they were tuned (16-bit inputs, head_dim 256, SM100+), else the defaults."""
    if (
        query.dtype not in (torch.bfloat16, torch.float16)
        or query.shape[-1] != _TUNED_HEAD_DIM
        or not is_blackwell_gpu()
    ):
        return None
    return _SM100_SLIDING_KERNEL_OPTIONS


# Every compiled call has one shape family, so the kernel compiles at most twice (a training graph and a
# forward-only one), both at load (:func:`warmup_flex_sliding_kernels`). Dynamo guards on a batch of one,
# on a length that divides the tile, on grad mode and each input's ``requires_grad``, on the autocast
# state and on whether a block mask carries a mask tensor; a mid-run compile stalls the other ranks in the
# next collective, and past Dynamo's recompile limit (8) FlexAttention runs unfused with the full score
# matrix. So each batch row runs alone, padded to a multiple of the tile, always under a block mask built
# from a dense mask, outside autocast (its inputs are already in the compute dtype), and a call either
# records a graph with every input requiring grad or runs under ``no_grad``. Shape-dynamic from the first
# call, so a new length reuses the graph. Deterministic-algorithms mode is a guard too, which the
# warm-up does not normalize (see ``warmup_flex_sliding_kernels``).
_compiled_flex = torch.compile(flex_attention, dynamic=True)

# FlexAttention's block-sparsity tile.
_TILE = 128
# Query rows of a dense mask reduced to key runs per pass (see ``key_intervals``).
_INTERVAL_ROWS = 256

# The attribute a dense mask carries its per-row block masks on (an empty list for a mask SDPA serves).
# Every sliding layer of one forward (and of its gradient-checkpoint recompute) receives the same mask
# tensor, so they share the block masks, which are freed with the mask. Held on the tensor rather than
# keyed by its address, which a later step's mask can reuse.
_BLOCK_MASKS_ATTR = "_flex_sliding_block_masks"


def _pad_to_tile(length: int) -> int:
    return -length % _TILE


def key_intervals(allowed: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Per query, the first and last key of a boolean ``[Q, K]`` mask (``True`` = attend), as int32
    ``[Q]`` tensors (an empty row gets ``first = K``, ``last = -1``), or ``None`` when some query's allowed
    keys are not one contiguous run.

    Every mask transformers builds for a sliding layer (causality, the window, packed-document isolation,
    padding) allows each query one run of keys. The mask is read in chunks of :data:`_INTERVAL_ROWS`
    query rows, so the temporaries stay a small multiple of that many rows.
    """
    q_len, kv_len = allowed.shape
    first = torch.empty(q_len, dtype=torch.int32, device=allowed.device)
    last = torch.empty_like(first)
    contiguous = torch.ones((), dtype=torch.bool, device=allowed.device)
    for start in range(0, q_len, _INTERVAL_ROWS):
        rows = allowed[start : start + _INTERVAL_ROWS]
        count = rows.sum(dim=-1, dtype=torch.int32)
        lo = rows.view(torch.uint8).argmax(dim=-1)  # the first allowed key (0 for an empty row)
        runs = rows[:, 0].to(torch.int32) + (rows[:, 1:] & ~rows[:, :-1]).sum(dim=-1, dtype=torch.int32)
        contiguous &= (runs <= 1).all()
        empty = count == 0
        first[start : start + rows.shape[0]] = torch.where(empty, kv_len, lo)
        last[start : start + rows.shape[0]] = torch.where(empty, -1, lo + count - 1)
    if not contiguous.item():  # one host sync per mask, which every sliding layer of a forward shares
        return None
    return first, last


def interval_block_mask(first: torch.Tensor, last: torch.Tensor, kv_len: int) -> BlockMask:
    """The ``BlockMask`` of per-query key runs ``first[q] <= kv <= last[q]``, both sides padded to whole
    :data:`_TILE` tiles (a padded query attends nothing).

    A tile is a full block where every row's run covers it and a partial block where some row's run may
    reach it (from each query tile's extreme bounds, so a few partial blocks may turn out empty). The mask
    function compares against the two ``[Q]`` bound tensors: nothing indexes a ``[Q, K]`` grid, whose
    flattened offset outgrows int32 past about 46k tokens, and ``create_block_mask``'s index tensors over
    that grid are never built.
    """
    q_len = first.shape[0]
    q_pad, kv_pad = q_len + _pad_to_tile(q_len), kv_len + _pad_to_tile(kv_len)
    first = torch.nn.functional.pad(first, (0, q_pad - q_len), value=kv_pad)
    last = torch.nn.functional.pad(last, (0, q_pad - q_len), value=-1)
    tile_first, tile_last = first.view(-1, _TILE), last.view(-1, _TILE)
    starts = torch.arange(0, kv_pad, _TILE, device=first.device, dtype=torch.int32)
    ends = starts + _TILE - 1
    full = (tile_first.amax(dim=1)[:, None] <= starts) & (tile_last.amin(dim=1)[:, None] >= ends)
    reached = (tile_first.amin(dim=1)[:, None] <= ends) & (tile_last.amax(dim=1)[:, None] >= starts)
    partial = reached & ~full

    def blocks(selected: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        counts = selected.sum(dim=-1, dtype=torch.int32)
        indices = torch.argsort(selected.to(torch.int8), dim=-1, descending=True, stable=True).to(torch.int32)
        return counts[None, None], indices[None, None]

    def mask_mod(b, h, q_idx, kv_idx):
        return (kv_idx >= first[q_idx]) & (kv_idx <= last[q_idx])

    return BlockMask.from_kv_blocks(
        *blocks(partial), *blocks(full), BLOCK_SIZE=_TILE, mask_mod=mask_mod, seq_lengths=(q_pad, kv_pad)
    )


def sliding_block_masks(attention_mask: torch.Tensor) -> list[BlockMask] | None:
    """One block mask per row of SDPA's boolean ``[B, 1, Q, K]`` mask, or ``None`` when a row allows some
    query a non-contiguous set of keys (which SDPA then serves). Built once per mask tensor."""
    masks = getattr(attention_mask, _BLOCK_MASKS_ATTR, None)
    if masks is not None:
        return masks or None
    kv_len = attention_mask.shape[-1]
    masks = []
    for row in attention_mask.to(torch.bool):
        intervals = key_intervals(row[0])
        if intervals is None:
            masks = []
            break
        masks.append(interval_block_mask(*intervals, kv_len))
    setattr(attention_mask, _BLOCK_MASKS_ATTR, masks)
    return masks or None


def _padded_rows(tensor: torch.Tensor) -> list[torch.Tensor]:
    """Each batch row of ``[B, H, S, D]`` as a contiguous ``[1, H, S', D]``, ``S'`` padded to whole tiles."""
    padding = _pad_to_tile(tensor.shape[-2])
    return [
        torch.nn.functional.pad(row, (0, 0, 0, padding)) if padding else row.contiguous() for row in tensor.split(1)
    ]


def _sliding_flex_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    block_masks: list[BlockMask],
    scaling: float | None,
) -> torch.Tensor:
    """FlexAttention over each batch row under its block mask. Returns ``[B, S, H, D]``."""
    q_len = query.shape[-2]
    records_grad = torch.is_grad_enabled() and any(t.requires_grad for t in (query, key, value))
    if records_grad:
        # An input left frozen (a LoRA that adapts q_proj alone) joins the graph as a detached leaf whose
        # gradient is dropped, so the graph never depends on which projections are trained.
        query, key, value = (t if t.requires_grad else t.detach().requires_grad_() for t in (query, key, value))
    kernel_options = _sliding_kernel_options(query)
    rows = zip(_padded_rows(query), _padded_rows(key), _padded_rows(value), block_masks, strict=True)
    with torch.set_grad_enabled(records_grad), torch.autocast(query.device.type, enabled=False):
        out = torch.cat(
            [
                _compiled_flex(
                    q,
                    k,
                    v,
                    block_mask=block_mask,
                    scale=scaling,
                    enable_gqa=query.shape[1] != key.shape[1],
                    kernel_options=kernel_options,
                )
                for q, k, v, block_mask in rows
            ]
        )
    return out[:, :, :q_len].transpose(1, 2).contiguous()


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
    return batch * heads * q_len * kv_len * _SAVED_BYTES_PER_SCORE <= EAGER_GLOBAL_BUDGET_BYTES


def _sdpa_is_mem_efficient_only(head_dim: int) -> bool:
    """Whether SDPA would run this head on its mem-efficient kernel alone."""
    return head_dim > FLASH_MAX_HEAD_DIM or not torch.backends.cuda.flash_sdp_enabled()


def _takes_flex(query: torch.Tensor, attention_mask: torch.Tensor | None, sliding_window: int) -> bool:
    """Whether a sliding call has the one shape family the compiled kernel serves: a per-row boolean mask
    over a query longer than the window and than one tile. A query within the window never reaches its
    edge, so SDPA computes no tile the window would skip; a single-tile query would give the block mask
    size-1 dimensions, which Dynamo specializes on."""
    return (
        attention_mask is not None
        and attention_mask.shape[0] == query.shape[0]
        and attention_mask.shape[1] == 1
        and query.shape[-2] > max(sliding_window, _TILE)
    )


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
    if eligible and sliding_window is not None and _takes_flex(query, attention_mask, sliding_window):
        block_masks = sliding_block_masks(attention_mask)
        if block_masks is not None:
            return _sliding_flex_attention(query, key, value, block_masks, scaling), None
    return ALL_ATTENTION_FUNCTIONS["sdpa"](
        module, query, key, value, attention_mask, dropout=dropout, scaling=scaling, **kwargs
    )


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


def resolve_flex_sliding_attn_implementation(model_config, attn_implementation: str) -> str:
    """The implementation a model is built with: :data:`FLEX_SLIDING` where the run resolved to ``sdpa`` on
    CUDA and a head is wider than flash runs (Gemma 4), the resolved value otherwise. A sinks model keeps
    ``sdpa``: every one of its calls carries the sinks, which FlexAttention is not given here.
    ``HALO_FLEX_SLIDING=0`` keeps plain SDPA."""
    if (
        attn_implementation != "sdpa"
        or not torch.cuda.is_available()
        or model_has_sinks(model_config)
        or not env_flag("HALO_FLEX_SLIDING", True)
        or not head_dim_exceeds_flash(model_config)
    ):
        return attn_implementation
    return register_flex_sliding_attention()


def sliding_attention_calls(model: torch.nn.Module) -> set[tuple[int, int, int, int, float | None]]:
    """``(heads, kv_heads, head_dim, window, scaling)`` of every distinct sliding-window attention module in
    ``model``: what each passes the attention function, which the compiled kernel's guards key on (the
    ``scaling`` included, Gemma 4's being 1.0 rather than ``head_dim ** -0.5``)."""
    calls = set()
    for module in model.modules():
        window = getattr(module, "sliding_window", None)
        if not window or not getattr(module, "is_causal", False) or not hasattr(module, "num_key_value_groups"):
            continue
        heads = module.config.num_attention_heads
        calls.add((heads, heads // module.num_key_value_groups, module.head_dim, window, module.scaling))
    return calls


def warmup_flex_sliding_kernels(model: torch.nn.Module, *, dtype: torch.dtype) -> None:
    """Compile the sliding layers' FlexAttention graphs (training and forward-only) for each sliding
    attention call ``model`` makes, so no rank compiles one mid-run. Rank-local; the caller fences it with
    a barrier.

    Warmed under the default deterministic-algorithms mode: a run that turns it on after the load
    (``full_determinism``) compiles each graph again at its first sliding call.
    """
    if effective_attn_implementation(model.config) != FLEX_SLIDING:
        return
    device = torch.device(f"cuda:{torch.cuda.current_device()}")
    for heads, kv_heads, head_dim, window, scaling in sorted(sliding_attention_calls(model), key=str):
        seq = 2 * max(window, _TILE) + 3  # past the window and a tile, off the tile edge (the padded path)
        positions = torch.arange(seq, device=device)
        document = positions >= window  # two documents
        mask = (
            (positions[None] <= positions[:, None])
            & (positions[:, None] - positions[None] < window)
            & (document[None] == document[:, None])
        )[None, None]
        module = torch.nn.Module()
        module.is_causal = True
        module.num_key_value_groups = heads // kv_heads
        for records_grad in (True, False):
            q = torch.zeros(1, heads, seq, head_dim, device=device, dtype=dtype, requires_grad=records_grad)
            k = torch.zeros(1, kv_heads, seq, head_dim, device=device, dtype=dtype, requires_grad=records_grad)
            v = torch.zeros(1, kv_heads, seq, head_dim, device=device, dtype=dtype, requires_grad=records_grad)
            out, _ = flex_sliding_attention(module, q, k, v, mask.clone(), scaling=scaling, sliding_window=window)
            if records_grad:
                out.sum().backward()
    logger.info(f"Compiled the {FLEX_SLIDING} sliding-layer kernels (training and forward-only)")
