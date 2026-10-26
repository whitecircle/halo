"""The ``sdpa_flex_sliding`` attention: sliding layers on compiled FlexAttention, wide-head global layers on matmul attention.

FlexAttention skips the tiles outside the window that SDPA's dense mask computes, under a block mask
derived from the mask transformers built. Every other call goes to the implementation registered as
``sdpa``; agent-docs/optimization/flash-attention.md has the routing and the compile contract.
"""

from __future__ import annotations

import logging

import torch
from torch.nn.attention.flex_attention import BlockMask, flex_attention
from transformers.masking_utils import ALL_MASK_ATTENTION_FUNCTIONS, sdpa_mask
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

from halo.env import env_flag
from halo.hardware import is_blackwell_gpu
from halo.models.attention.gpt_oss_sinks import uses_gpt_oss_sinks
from halo.models.attention.selection import (
    FLASH_MAX_HEAD_DIM,
    effective_attn_implementation,
    has_head_dim_beyond_flash,
)

logger = logging.getLogger(__name__)

SDPA_FLEX_SLIDING = "sdpa_flex_sliding"

# Attention-function keywords FlexAttention is not given here; a call carrying any of them keeps SDPA.
_SDPA_ONLY_KWARGS = ("s_aux", "softcap", "position_bias")

# Bytes matmul attention saves for backward per score: the fp32 softmax output and the 16-bit weights.
_SAVED_BYTES_PER_SCORE = 6
# Saved scores one global layer may hold on matmul attention before mem-efficient SDPA takes over:
# 16 heads at 4,096 tokens per row.
MATMUL_GLOBAL_BUDGET_BYTES = 2 * 2**30

# Tiles tuned on Gemma 4's head_dim-256 sliding layers on SM100+. Length-independent on purpose: a
# length-dependent choice is a guard the compiled kernel would recompile on.
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

# FlexAttention's block-sparsity tile.
_TILE = 128
# Query rows of a dense mask reduced to key runs per pass (see ``key_intervals``).
_INTERVAL_ROWS = 256

# The attribute a dense mask carries its per-row block masks on (an empty list for a mask SDPA serves),
# so every sliding layer of a forward shares them and they are freed with the mask.
_BLOCK_MASKS_ATTR = "_flex_sliding_block_masks"

# Every call has one shape family (one padded row, a block mask, autocast off, all inputs requiring grad
# or none), so it compiles twice, a training and a forward-only graph, both at warm-up.
_compiled_flex = torch.compile(flex_attention, dynamic=True)

# The deterministic-algorithms mode the warm-up compiled under, ``None`` in a process that ran none.
_warmed_deterministic: bool | None = None


def _pad_to_tile(length: int) -> int:
    return -length % _TILE


def key_intervals(allowed: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Per query, the first and last key of a boolean ``[Q, K]`` mask as int32 ``[Q]`` tensors (an empty
    row gets ``first = K``, ``last = -1``), or ``None`` when some query's keys are not one contiguous run.
    Read :data:`_INTERVAL_ROWS` query rows at a time to bound the temporaries.
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

    The mask function compares against the two ``[Q]`` bounds rather than indexing a ``[Q, K]`` grid,
    whose flattened offset outgrows int32 past about 46k tokens.
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


def _to_autocast_dtype(*tensors: torch.Tensor) -> tuple[torch.Tensor, ...]:
    """``tensors`` in the autocast dtype where autocast is on for their device, as autocast casts SDPA's
    inputs; the kernels here run outside autocast's reach."""
    device_type = tensors[0].device.type
    if not torch.is_autocast_enabled(device_type):
        return tensors
    dtype = torch.get_autocast_dtype(device_type)
    return tuple(t.to(dtype) for t in tensors)


def _sliding_flex_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    block_masks: list[BlockMask],
    scaling: float | None,
) -> torch.Tensor:
    """FlexAttention over each batch row under its block mask. Returns ``[B, S, H, D]``."""
    q_len = query.shape[-2]
    # fp32 rotary output beside 16-bit values would otherwise be a graph the warm-up never compiled.
    query, key, value = _to_autocast_dtype(query, key, value)
    records_grad = torch.is_grad_enabled() and any(t.requires_grad for t in (query, key, value))
    if records_grad:
        # A frozen input joins as a detached leaf, so the graph never depends on which projections train.
        query, key, value = (t if t.requires_grad else t.detach().requires_grad_() for t in (query, key, value))
    # The tuned tiles only where they were tuned (16-bit inputs, head_dim 256, SM100+).
    tuned = (
        query.dtype in (torch.bfloat16, torch.float16) and query.shape[-1] == _TUNED_HEAD_DIM and is_blackwell_gpu()
    )
    kernel_options = _SM100_SLIDING_KERNEL_OPTIONS if tuned else None
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


class _Fp32ScoreMatmul(torch.autograd.Function):
    """``query @ key_t`` over ``[N, M, D] @ [N, D, K]``, returned in at least fp32 at 16-bit GEMM speed
    (upcast inputs would run an fp32 GEMM, ~18x slower at Gemma 4's shapes). ``bmm``'s ``out_dtype``
    form has no derivative, hence the explicit backward.
    """

    @staticmethod
    def forward(ctx, query: torch.Tensor, key_t: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(query, key_t)
        return torch.bmm(query, key_t, out_dtype=torch.promote_types(query.dtype, torch.float32))

    @staticmethod
    def backward(ctx, grad: torch.Tensor) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        query, key_t = ctx.saved_tensors
        grad = grad.to(query.dtype)
        grad_query = torch.bmm(grad, key_t.transpose(1, 2)) if ctx.needs_input_grad[0] else None
        grad_key_t = torch.bmm(query.transpose(1, 2), grad) if ctx.needs_input_grad[1] else None
        return grad_query, grad_key_t


def _matmul_global_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    scaling: float,
    is_causal: bool,
) -> torch.Tensor:
    """Attention as two matmuls and a softmax over fp32 logits, as mem-efficient SDPA keeps them, masked as
    SDPA masks the call. Each KV group's query heads stack on the sequence axis, so no KV copy is made;
    masked scores take the dtype minimum, so no row softmaxes to NaN. Returns ``[B, S, H, D]``.
    """
    batch, heads, q_len, dim = query.shape
    kv_heads = key.shape[1]
    group = heads // kv_heads
    causal = attention_mask is None and is_causal and q_len > 1
    if causal:
        key, value = key[:, :, :q_len], value[:, :, :q_len]
    kv_len = key.shape[-2]
    q = query.reshape(batch * kv_heads, group * q_len, dim)
    k = key.reshape(batch * kv_heads, kv_len, dim)
    q, k = _to_autocast_dtype(q, k)  # bmm's out_dtype form falls through autocast
    scores = _Fp32ScoreMatmul.apply(q, k.transpose(1, 2))
    if scaling != 1.0:
        scores.mul_(scaling)
    scores = scores.view(batch, kv_heads, group, q_len, kv_len)
    if causal:
        allowed = torch.ones(q_len, kv_len, dtype=torch.bool, device=query.device).tril()
    elif attention_mask is not None:
        allowed = (
            attention_mask[:, :, None]
            if attention_mask.shape[1] == 1
            else attention_mask.view(batch, kv_heads, group, q_len, kv_len)
        )
    else:
        allowed = None
    if allowed is not None:
        scores = scores.masked_fill(~allowed, torch.finfo(scores.dtype).min)
    # Rebound, so the fp32 logits are freed before the 16-bit copy of the weights is made.
    scores = torch.softmax(scores, dim=-1)
    out = torch.matmul(scores.to(query.dtype).view(batch, kv_heads, group * q_len, kv_len), value)
    return out.view(batch, heads, q_len, dim).transpose(1, 2).contiguous()


def flex_sliding_attention(
    module: torch.nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    dropout: float = 0.0,
    scaling: float | None = None,
    sliding_window: int | None = None,
    is_causal: bool | None = None,
    **kwargs,
) -> tuple[torch.Tensor, None]:
    """The registered attention function: sliding calls on FlexAttention, mem-efficient-only causal global
    calls within the budget on matmul attention, the rest on ``sdpa``. An explicit ``is_causal`` overrides
    the module's, as it does for SDPA."""
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
        # SDPA would run this head on its mem-efficient kernel alone.
        and (query.shape[-1] > FLASH_MAX_HEAD_DIM or not torch.backends.cuda.flash_sdp_enabled())
        and query.shape[:-1].numel() * key.shape[-2] * _SAVED_BYTES_PER_SCORE <= MATMUL_GLOBAL_BUDGET_BYTES
    ):
        scale = query.shape[-1] ** -0.5 if scaling is None else scaling
        causal = module.is_causal if is_causal is None else is_causal
        return _matmul_global_attention(query, key, value, attention_mask, scale, causal), None
    # A query within the window skips no tile; a single-tile one gives the block mask size-1 dims,
    # which Dynamo specializes on.
    if (
        eligible
        and sliding_window is not None
        and attention_mask is not None
        and attention_mask.shape[0] == query.shape[0]
        and attention_mask.shape[1] == 1
        and query.shape[-2] > max(sliding_window, _TILE)
    ):
        block_masks = sliding_block_masks(attention_mask)
        if block_masks is not None:
            return _sliding_flex_attention(query, key, value, block_masks, scaling), None
    return ALL_ATTENTION_FUNCTIONS["sdpa"](
        module, query, key, value, attention_mask, dropout=dropout, scaling=scaling, is_causal=is_causal, **kwargs
    )


def register_flex_sliding_attention() -> str:
    """Register :data:`SDPA_FLEX_SLIDING` (idempotent) and return its name for ``attn_implementation``."""
    if SDPA_FLEX_SLIDING not in ALL_ATTENTION_FUNCTIONS:
        ALL_ATTENTION_FUNCTIONS.register(SDPA_FLEX_SLIDING, flex_sliding_attention)
        ALL_MASK_ATTENTION_FUNCTIONS.register(SDPA_FLEX_SLIDING, sdpa_mask)
        logger.info(
            f"Registered attention '{SDPA_FLEX_SLIDING}': sliding-window layers on compiled FlexAttention "
            "(block-sparse window), mem-efficient-only global layers on matmul attention, the rest on SDPA."
        )
    return SDPA_FLEX_SLIDING


def resolve_flex_sliding_attn_implementation(model_config, attn_implementation: str) -> str:
    """:data:`SDPA_FLEX_SLIDING` where the run resolved to ``sdpa`` on CUDA and a head is wider than flash
    runs (Gemma 4), else ``attn_implementation``. A sinks model, whose every call carries the sinks, and
    ``HALO_FLEX_SLIDING=0`` keep plain SDPA."""
    if (
        attn_implementation != "sdpa"
        or not torch.cuda.is_available()
        or uses_gpt_oss_sinks(model_config)
        or not env_flag("HALO_FLEX_SLIDING", True)
        or not has_head_dim_beyond_flash(model_config)
    ):
        return attn_implementation
    return register_flex_sliding_attention()


def sliding_attention_calls(model: torch.nn.Module) -> set[tuple[int, int, int, int, float | None]]:
    """``(heads, kv_heads, head_dim, window, scaling)`` of every distinct sliding-window attention module in
    ``model``: the call arguments the compiled kernel's guards key on."""
    calls = set()
    for module in model.modules():
        window = getattr(module, "sliding_window", None)
        if not window or not getattr(module, "is_causal", False) or not hasattr(module, "num_key_value_groups"):
            continue
        heads = module.config.num_attention_heads
        calls.add((heads, heads // module.num_key_value_groups, module.head_dim, window, module.scaling))
    return calls


def warm_flex_sliding_kernels(model: torch.nn.Module, *, dtype: torch.dtype) -> None:
    """Compile the training and forward-only FlexAttention graphs of each sliding call ``model`` makes, so
    no rank compiles one mid-run, and record the deterministic mode they compiled under. Rank-local; the
    caller fences it."""
    global _warmed_deterministic
    if effective_attn_implementation(model.config) != SDPA_FLEX_SLIDING:
        return
    calls = sliding_attention_calls(model)
    if not calls:  # nothing to compile, so no graph for a later mode to miss
        return
    device = torch.device(f"cuda:{torch.cuda.current_device()}")
    for heads, kv_heads, head_dim, window, scaling in sorted(calls, key=str):
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
    _warmed_deterministic = torch.are_deterministic_algorithms_enabled()
    logger.info(f"Compiled the {SDPA_FLEX_SLIDING} sliding-layer kernels (training and forward-only)")


def reject_full_determinism_after_warmup(full_determinism: bool) -> None:
    """Refuse ``full_determinism`` once the warm-up compiled the sliding graphs without deterministic mode;
    call before ``Trainer.__init__`` turns it on. Dynamo guards on the mode, so each rank would recompile
    at its own first long sliding call while its peers wait in the next collective."""
    if full_determinism and _warmed_deterministic is False:
        raise ValueError(
            f"full_determinism: true cannot run with the {SDPA_FLEX_SLIDING} attention this model was built with: "
            "its sliding-layer FlexAttention graphs were compiled at load without deterministic mode, which "
            "full_determinism turns on afterwards, so each rank would recompile them at its own first long "
            "sliding call while its peers wait in the next collective (a DeepEP dispatch times out). Set "
            "HALO_FLEX_SLIDING=0 to build the model on plain SDPA, or drop full_determinism."
        )
