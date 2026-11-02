"""The ``sdpa_flex_sliding`` attention: masked causal layers on compiled FlexAttention, wide-head global layers on
matmul attention or, on packed rows past its budget, on mem-efficient attention per document, each skipping the
tiles the mask excludes; every other call goes to ``sdpa``. agent-docs/optimization/flash-attention.md has the
routing and the compile contract."""

from __future__ import annotations

import logging

import torch
from torch.nn.attention.flex_attention import BlockMask, flex_attention
from transformers.masking_utils import ALL_MASK_ATTENTION_FUNCTIONS, sdpa_mask
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

from halo.env import env_flag
from halo.hardware import is_blackwell_gpu
from halo.kernels.fp32_matmul import bmm_fp32
from halo.log import warn_once
from halo.models.attention.gpt_oss_sinks import uses_gpt_oss_sinks
from halo.models.attention.layout import AttentionCall, attention_call, attention_calls, has_sliding_window_layers
from halo.models.attention.selection import (
    FLASH_MAX_HEAD_DIM,
    declared_model_class,
    effective_attn_implementation,
    has_attention_logit_softcap,
    has_head_dim_beyond_flash,
)
from halo.models.model_classes import is_remote_code_class

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
# The narrowest query/key or value head Inductor's FlexAttention kernel takes (``tl.dot``'s minimum); a narrower
# one keeps SDPA, as a head wider than FLASH_MAX_HEAD_DIM does.
FLEX_MIN_HEAD_DIM = 16
# Query rows of a dense mask reduced to key runs per pass (see ``key_intervals``).
_INTERVAL_ROWS = 256

# The attribute a dense mask carries its per-row block masks on (an empty list for a mask SDPA serves),
# so every layer of a forward that gets the mask shares them and they are freed with it.
_BLOCK_MASKS_ATTR = "_flex_sliding_block_masks"
# The attribute a dense mask carries its packed documents on (``None`` for a mask that is not document-causal).
_DOCUMENTS_ATTR = "_flex_sliding_documents"
# ``custom_mask_type`` of the mem-efficient kernel: causal from each document's first query and key.
_CAUSAL_FROM_TOP_LEFT = 1

# Every call has one shape family per attention signature (one padded row, a block mask, autocast off, all
# inputs requiring grad or none), so it compiles twice, a training and a forward-only graph, both at warm-up.
_compiled_flex = torch.compile(flex_attention, dynamic=True)

# The deterministic-algorithms mode the warm-up compiled under, ``None`` in a process that ran none.
_warmed_deterministic: bool | None = None

_warned_unwarmed_calls: set[tuple] = set()


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


def row_block_masks(attention_mask: torch.Tensor) -> list[BlockMask] | None:
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


def causal_documents(attention_mask: torch.Tensor) -> tuple[torch.Tensor, int] | None:
    """The packed documents of SDPA's boolean ``[B, 1, S, S]`` mask when each query attends exactly its own
    document, from the document's first token up to itself: ``(cu_seqlens, max_len)``, the int32 boundaries of
    every row's documents laid end to end and the longest one's length. ``None`` for any other mask (a padded
    query, a bidirectional span, a window), which SDPA then serves. Built once per mask tensor."""
    if hasattr(attention_mask, _DOCUMENTS_ATTR):
        return getattr(attention_mask, _DOCUMENTS_ATTR)
    batch, _, seq, _ = attention_mask.shape
    positions = torch.arange(seq, dtype=torch.int32, device=attention_mask.device)
    causal = torch.ones((), dtype=torch.bool, device=attention_mask.device)
    starts, documents = [], None
    for row in attention_mask:
        intervals = key_intervals(row[0])
        if intervals is None:
            break
        first, last = intervals
        begins = first == positions
        document_start = torch.where(begins, positions, 0).cummax(dim=0).values
        causal &= ((last == positions) & (first == document_start)).all()
        starts.append(positions[begins] + len(starts) * seq)
    else:
        if causal.item():
            cu_seqlens = torch.cat([*starts, positions.new_tensor([batch * seq])])
            documents = cu_seqlens, int(cu_seqlens.diff().max())
    setattr(attention_mask, _DOCUMENTS_ATTR, documents)
    return documents


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


def _block_sparse_attention(
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
    (upcast inputs would run an fp32 GEMM, ~18x slower at Gemma 4's shapes). :func:`bmm_fp32` has no
    derivative on that path, hence the explicit backward.
    """

    @staticmethod
    def forward(ctx, query: torch.Tensor, key_t: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(query, key_t)
        return bmm_fp32(query, key_t)

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


class _VarlenCausalAttention(torch.autograd.Function):
    """Mem-efficient attention over ``[1, T, H, D]`` inputs, causal within each ``cu_seqlens`` document: the
    kernel SDPA runs these heads on, over the documents' own tiles only."""

    @staticmethod
    def forward(ctx, query, key, value, cu_seqlens: torch.Tensor, max_len: int, scale: float) -> torch.Tensor:
        records_grad = any(ctx.needs_input_grad[:3])
        out, logsumexp, seed, offset, *_ = torch.ops.aten._efficient_attention_forward(
            query,
            key,
            value,
            None,
            cu_seqlens,
            cu_seqlens,
            max_len,
            max_len,
            0.0,
            _CAUSAL_FROM_TOP_LEFT,
            compute_log_sumexp=records_grad,
            scale=scale,
        )
        if records_grad:
            ctx.save_for_backward(query, key, value, out, logsumexp, cu_seqlens, seed, offset)
            ctx.max_len, ctx.scale = max_len, scale
        return out

    @staticmethod
    def backward(ctx, grad: torch.Tensor):
        query, key, value, out, logsumexp, cu_seqlens, seed, offset = ctx.saved_tensors
        grad_query, grad_key, grad_value, _ = torch.ops.aten._efficient_attention_backward(
            grad.contiguous(),
            query,
            key,
            value,
            None,
            out,
            cu_seqlens,
            cu_seqlens,
            ctx.max_len,
            ctx.max_len,
            logsumexp,
            0.0,
            seed,
            offset,
            _CAUSAL_FROM_TOP_LEFT,
            False,
            scale=ctx.scale,
        )
        return grad_query, grad_key, grad_value, None, None, None


def _varlen_causal_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    documents: tuple[torch.Tensor, int],
    scaling: float,
) -> torch.Tensor:
    """Causal attention within each packed document of ``documents`` (:func:`causal_documents`), the KV heads
    repeated per query group as SDPA repeats them. Returns ``[B, S, H, D]``."""
    batch, heads, seq, dim = query.shape
    query, key, value = _to_autocast_dtype(query, key, value)

    def tokens(tensor: torch.Tensor) -> torch.Tensor:
        tokens_first, group = tensor.transpose(1, 2), heads // tensor.shape[1]
        if group > 1:
            tokens_first = tokens_first.repeat_interleave(group, dim=2)
        return tokens_first.reshape(1, batch * seq, heads, dim)

    with torch.autocast(query.device.type, enabled=False):
        out = _VarlenCausalAttention.apply(tokens(query), tokens(key), tokens(value), *documents, scaling)
    return out.view(batch, seq, heads, dim)


def _flex_widths(head_dim: int, v_head_dim: int) -> bool:
    """Whether FlexAttention runs heads this wide: query/key and value at least FLEX_MIN_HEAD_DIM, and query/key
    no wider than FLASH_MAX_HEAD_DIM."""
    return min(head_dim, v_head_dim) >= FLEX_MIN_HEAD_DIM and head_dim <= FLASH_MAX_HEAD_DIM


def _flex_runs(call: AttentionCall) -> bool:
    """Whether FlexAttention serves ``call``: heads of a width it runs (:func:`_flex_widths`), and no softcap,
    which keeps SDPA."""
    return _flex_widths(call.head_dim, call.v_head_dim) and call.softcap is None


def _flex_call(module: torch.nn.Module) -> AttentionCall | None:
    """The call ``module`` makes on FlexAttention (:func:`~halo.models.attention.layout.attention_call`), or
    ``None`` for one FlexAttention serves no call of."""
    call = attention_call(module)
    return call if call is not None and _flex_runs(call) else None


def _flex_serves(
    module: torch.nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    window: int | None,
    scaling: float | None,
) -> bool:
    """Whether FlexAttention takes a masked call: one mask row per batch row over exactly the call's queries
    and keys, heads of a width it runs (:func:`_flex_widths`), more queries than the window
    and one tile (below either no tile is skipped, and a single-tile row gives the block mask size-1 dims,
    which Dynamo specializes on), and the heads, widths and scaling of the call the warm-up compiles for the
    module, so no call compiles mid-run. Tensor parallelism hands a call the rank's share of the heads, which
    the module does not declare, so such a call keeps SDPA."""
    if not (
        attention_mask is not None
        and attention_mask.shape == (query.shape[0], 1, query.shape[-2], key.shape[-2])
        and _flex_widths(query.shape[-1], value.shape[-1])
        and query.shape[-2] > max(window or 0, _TILE)
    ):
        return False
    call = _flex_call(module)
    expected = None if call is None else (call.heads, call.kv_heads, call.head_dim, call.v_head_dim, call.scaling)
    made = (query.shape[1], key.shape[1], query.shape[-1], value.shape[-1], scaling)
    if made == expected:
        return True
    warn_once(
        logger,
        _warned_unwarmed_calls,
        (type(module).__name__, made),
        "%s: a %s call (query / KV heads, query / value widths, scaling) %s is not the call the FlexAttention "
        "warm-up compiles for its module, %s, so it runs SDPA; under tensor parallelism a call carries the "
        "rank's share of the heads",
        SDPA_FLEX_SLIDING,
        type(module).__name__,
        made,
        expected,
    )
    return False


def _varlen_serves(query: torch.Tensor, key: torch.Tensor, attention_mask: torch.Tensor | None) -> bool:
    """Whether per-document mem-efficient attention may take a masked call FlexAttention does not (a head wider
    than FLASH_MAX_HEAD_DIM): one mask row per batch row over the call's queries as its keys."""
    return (
        attention_mask is not None
        and query.shape[-2] == key.shape[-2]
        and attention_mask.shape == (query.shape[0], 1, query.shape[-2], key.shape[-2])
        and query.shape[-1] > FLASH_MAX_HEAD_DIM
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
    is_causal: bool | None = None,
    **kwargs,
) -> tuple[torch.Tensor, None]:
    """The registered attention function: mem-efficient-only causal global calls within the budget on matmul
    attention, masked causal calls (sliding windows, packed documents, padding) on FlexAttention, wide-head ones
    past the budget on packed rows on mem-efficient attention per document, the rest on ``sdpa``. An explicit
    ``is_causal`` overrides the module's, as it does for SDPA."""
    boolean_mask = attention_mask is None or attention_mask.dtype == torch.bool
    eligible = (
        query.is_cuda
        and not dropout
        and not kwargs.get("output_attentions", False)
        and boolean_mask
        and getattr(module, "is_causal", False)
        and all(kwargs.get(name) is None for name in _SDPA_ONLY_KWARGS)
    )
    mem_efficient_only = (
        eligible
        and sliding_window is None
        and query.shape[-1] == value.shape[-1]
        # SDPA would run this head on its mem-efficient kernel alone.
        and (query.shape[-1] > FLASH_MAX_HEAD_DIM or not torch.backends.cuda.flash_sdp_enabled())
    )
    scale = query.shape[-1] ** -0.5 if scaling is None else scaling
    if (
        mem_efficient_only
        and query.shape[:-1].numel() * key.shape[-2] * _SAVED_BYTES_PER_SCORE <= MATMUL_GLOBAL_BUDGET_BYTES
    ):
        causal = module.is_causal if is_causal is None else is_causal
        return _matmul_global_attention(query, key, value, attention_mask, scale, causal), None
    if eligible and _flex_serves(module, query, key, value, attention_mask, sliding_window, scaling):
        block_masks = row_block_masks(attention_mask)
        if block_masks is not None:
            return _block_sparse_attention(query, key, value, block_masks, scaling), None
    if mem_efficient_only and _varlen_serves(query, key, attention_mask):
        documents = causal_documents(attention_mask)
        if documents is not None:
            return _varlen_causal_attention(query, key, value, documents, scale), None
    return ALL_ATTENTION_FUNCTIONS["sdpa"](
        module, query, key, value, attention_mask, dropout=dropout, scaling=scaling, is_causal=is_causal, **kwargs
    )


def register_flex_sliding_attention() -> str:
    """Register :data:`SDPA_FLEX_SLIDING` (idempotent) and return its name for ``attn_implementation``."""
    if SDPA_FLEX_SLIDING not in ALL_ATTENTION_FUNCTIONS:
        ALL_ATTENTION_FUNCTIONS.register(SDPA_FLEX_SLIDING, flex_sliding_attention)
        ALL_MASK_ATTENTION_FUNCTIONS.register(SDPA_FLEX_SLIDING, sdpa_mask)
        logger.info(
            f"Registered attention '{SDPA_FLEX_SLIDING}': masked causal layers (sliding windows, packed "
            "documents) on compiled block-sparse FlexAttention, mem-efficient-only global layers on matmul "
            "attention or, on packed rows past its budget, per-document mem-efficient attention, the rest on SDPA."
        )
    return SDPA_FLEX_SLIDING


def resolve_flex_sliding_attn_implementation(model_config, attn_implementation: str) -> str:
    """:data:`SDPA_FLEX_SLIDING` where the run resolved to ``sdpa`` on CUDA for a model with sliding-window
    layers (Step-3.7, Gemma 4, a Mistral on a pinned ``sdpa``) or a head wider than flash runs, else
    ``attn_implementation``.

    Plain SDPA stays for a model whose every call carries what the kernels here are not given (GptOss's
    sinks, Gemma 2's logit softcap), for a class transformers does not declare backend-compatible (Inkling,
    whose calls carry a dense relative bias), for remote code (unmapped, or mapped by an earlier load), whose
    attention need not dispatch through the registry, and under ``HALO_FLEX_SLIDING=0``.
    """
    if (
        attn_implementation != "sdpa"
        or not torch.cuda.is_available()
        or not env_flag("HALO_FLEX_SLIDING", True)
        or uses_gpt_oss_sinks(model_config)
        or has_attention_logit_softcap(model_config)
    ):
        return attn_implementation
    model_class = declared_model_class(model_config)
    if (
        model_class is None
        or is_remote_code_class(model_class)
        or not model_class.is_backend_compatible()
        or not (has_sliding_window_layers(model_config) or has_head_dim_beyond_flash(model_config))
    ):
        return attn_implementation
    return register_flex_sliding_attention()


def flex_attention_calls(model: torch.nn.Module) -> set[AttentionCall]:
    """The distinct calls ``model``'s causal attention modules make on FlexAttention (a sliding layer, or a full
    layer, with heads of a width it runs and no softcap): the call arguments the compiled kernel's guards key on."""
    return {call for call in attention_calls(model) if _flex_runs(call)}


def warm_flex_sliding_kernels(
    model: torch.nn.Module, *, dtype: torch.dtype, device: torch.device | None = None
) -> None:
    """Compile the training and forward-only FlexAttention graphs of each masked call ``model`` makes, on ``device``
    (the current CUDA device by default), so no rank compiles one mid-run, and record the deterministic mode they
    compiled under. Rank-local; the caller fences it."""
    global _warmed_deterministic
    if effective_attn_implementation(model.config) != SDPA_FLEX_SLIDING:
        return
    calls = flex_attention_calls(model)
    if not calls:  # nothing to compile, so no graph for a later mode to miss
        return
    if device is None:
        device = torch.device(f"cuda:{torch.cuda.current_device()}")
    for call in sorted(calls, key=str):
        span = max(call.window or 0, _TILE)
        seq = 2 * span + 3  # past the window and a tile, off the tile edge (the padded path)
        positions = torch.arange(seq, device=device)
        document = positions >= span  # two documents
        mask = (positions[None] <= positions[:, None]) & (document[None] == document[:, None])
        if call.window:
            mask &= positions[:, None] - positions[None] < call.window
        block_masks = row_block_masks(mask[None, None])
        q_shape, k_shape = (1, call.heads, seq, call.head_dim), (1, call.kv_heads, seq, call.head_dim)
        v_shape = (1, call.kv_heads, seq, call.v_head_dim)
        for records_grad in (True, False):
            q = torch.zeros(q_shape, device=device, dtype=dtype, requires_grad=records_grad)
            k = torch.zeros(k_shape, device=device, dtype=dtype, requires_grad=records_grad)
            v = torch.zeros(v_shape, device=device, dtype=dtype, requires_grad=records_grad)
            out = _block_sparse_attention(q, k, v, block_masks, call.scaling)
            if records_grad:
                out.sum().backward()
    _warmed_deterministic = torch.are_deterministic_algorithms_enabled()
    logger.info(
        f"Compiled the {SDPA_FLEX_SLIDING} FlexAttention kernels of {len(calls)} call(s) (training and forward-only)"
    )


def reject_full_determinism_after_warmup(full_determinism: bool) -> None:
    """Refuse ``full_determinism`` once the warm-up compiled the FlexAttention graphs without deterministic
    mode; call before ``Trainer.__init__`` turns it on. Dynamo guards on the mode, so each rank would recompile
    at its own first long masked call while its peers wait in the next collective."""
    if full_determinism and _warmed_deterministic is False:
        raise ValueError(
            f"full_determinism: true cannot run with the {SDPA_FLEX_SLIDING} attention this model was built with: "
            "its FlexAttention graphs were compiled at load without deterministic mode, which full_determinism "
            "turns on afterwards, so each rank would recompile them at its own first long masked call while its "
            "peers wait in the next collective (a DeepEP dispatch times out). Set HALO_FLEX_SLIDING=0 to build "
            "the model on plain SDPA, or drop full_determinism."
        )
