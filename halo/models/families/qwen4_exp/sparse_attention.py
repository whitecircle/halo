"""Qwen sparse attention (QSA) for uncached ``qwen4_exp`` training rows, memory linear in the row length, selecting the
blocks transformers' dense indexer selects. Attention runs on the block-sparse kernel or, as reference and fallback, on
masked SDPA tiles. A checkpoint's recompute attends the first pass's selection."""

import logging
import math
from dataclasses import dataclass
from typing import Literal, get_args

import torch
import torch.nn.functional as F
import transformers.models.qwen4_exp.modeling_qwen4_exp as qwen4_exp_modeling
from torch import nn
from torch.autograd.function import once_differentiable
from triton.errors import TritonError

from halo.env import env_positive_int, env_str
from halo.kernels.block_sparse_attention import (
    SUPPORTED_DTYPES,
    block_sparse_attention,
    block_sparse_unsupported,
    warm_block_sparse_kernels,
)
from halo.kernels.fp32_matmul import to_autocast_dtype
from halo.log import warn_once
from halo.models.attention.selection import flash_attention_deterministic
from halo.models.gradient_checkpointing import active_recompute_scope
from halo.models.row_documents import RowDocument, row_documents
from halo.runtime.ranks import is_global_main_process

logger = logging.getLogger(__name__)

QUERY_TILE = 128
_SELECTION_SLOT = "qsa_selection"
QSAKernelMode = Literal["auto", "on", "off"]
_QSA_KERNEL_DEFAULT: QSAKernelMode = "off"
_logged_tiled_reasons: set[str] = set()
# Layouts whose kernels failed to build at warm-up under ``auto``, with the error: they run the tiled path.
_failed_warmups: dict[tuple, str] = {}


@dataclass(frozen=True)
class QSASelection:
    """``blocks[row, query]`` holds block indices counted from the document's first key, ``-1`` in unused slots; the
    keys after the last complete block are always attended. ``spans[row, query]`` is ``(first, visible)``: the
    document's first unpadded key and how many keys the query sees from it, which the kernel reads in place of
    ``documents``. A query's valid slots are a prefix of ``blocks``, ``min(visible // ratio, block_topk)`` long."""

    documents: tuple[RowDocument, ...]
    blocks: torch.Tensor
    spans: torch.Tensor
    compress_ratio: int


def _block_scores(queries: torch.Tensor, pooled: torch.Tensor) -> torch.Tensor:
    """The head-summed ReLU of the fp32 dot products, in transformers' op order."""
    scores = torch.matmul(queries.float(), pooled.float().transpose(-1, -2)).transpose(-1, -2)
    return torch.relu(scores).sum(dim=-1) / math.sqrt(pooled.shape[-1])


def _rescore_like_transformers(query: torch.Tensor, pooled: torch.Tensor, count: int, block_topk: int) -> torch.Tensor:
    """One query against its ``count`` complete blocks, so a tie at the cutoff resolves in transformers' order."""
    return _block_scores(query, pooled[:count]).topk(min(block_topk, count), dim=0).indices


def _indexer_chunk(blocks: int, index_heads: int) -> int:
    """Queries scored per matmul: whole tiles whose fp32 per-head products, their ReLU and the scores fit
    ``HALO_QSA_INDEXER_MIB``."""
    per_query = 4 * blocks * (2 * index_heads + 1)
    budget = env_positive_int("HALO_QSA_INDEXER_MIB", 256) << 20
    return QUERY_TILE * max(1, budget // (QUERY_TILE * per_query))


def _select_tile(
    queries: torch.Tensor, pooled: torch.Tensor, complete: torch.Tensor, blocks: int, block_topk: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """``[tile, block_topk]`` selected block ids, ``-1`` unused, and per query whether its k-th and next block tie."""
    columns = torch.arange(block_topk, device=queries.device)
    every_block = torch.where(columns < complete.unsqueeze(-1), columns, -1)
    if blocks <= block_topk:
        return every_block, torch.zeros_like(complete, dtype=torch.bool)
    scores = _block_scores(queries, pooled[:blocks]).masked_fill(
        torch.arange(blocks, device=queries.device) >= complete.unsqueeze(-1), -torch.inf
    )
    values, top = scores.topk(block_topk + 1, dim=-1)
    sparse = complete > block_topk
    selected = torch.where(sparse.unsqueeze(-1), top[:, :block_topk], every_block)
    return selected, sparse & (values[:, block_topk - 1] == values[:, block_topk])


@torch.no_grad()
def _compute_selection(
    indexer: nn.Module,
    hidden_states: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
    seq_idx: torch.Tensor | None,
    attention_mask: torch.Tensor | None,
) -> QSASelection:
    batch, length, _ = hidden_states.shape
    head_dim, ratio, block_topk = indexer.index_head_dim, indexer.compress_ratio, indexer.block_topk
    cos, sin = position_embeddings
    queries, keys = torch.split(
        indexer.index_qk_proj(hidden_states),
        [indexer.index_n_heads * head_dim, indexer.index_kv_heads * head_dim],
        dim=-1,
    )
    queries = indexer.q_layernorm(queries.reshape(batch, length, -1, head_dim))
    queries = qwen4_exp_modeling.apply_rotary_pos_emb(queries, cos=cos, sin=sin, unsqueeze_dim=2)
    keys = keys.reshape(batch, length, -1, head_dim).squeeze(2)

    documents = row_documents(seq_idx, attention_mask, batch, length, hidden_states.device)
    most_blocks = max((doc.key_stop - doc.key_start) // ratio for doc in documents)
    index_dtype = torch.int16 if most_blocks <= torch.iinfo(torch.int16).max else torch.int32
    blocks = torch.full((batch, length, block_topk), -1, dtype=index_dtype, device=hidden_states.device)
    spans = torch.zeros(batch, length, 2, dtype=torch.int32, device=hidden_states.device)
    for doc in documents:
        spans[doc.row, doc.start : doc.stop, 0] = doc.key_start
        spans[doc.row, doc.start : doc.stop, 1] = doc.visible_keys(doc.start, doc.stop, hidden_states.device)
        count = (doc.key_stop - doc.key_start) // ratio
        if count == 0:
            continue
        block_starts = doc.key_start + ratio * torch.arange(count, device=hidden_states.device)
        pooled = keys[doc.row, doc.key_start : doc.key_start + count * ratio].contiguous().view(count, ratio, head_dim)
        pooled = indexer.k_layernorm(pooled.float().mean(dim=1).to(keys.dtype))
        pooled = qwen4_exp_modeling.apply_rotary_pos_emb(
            pooled.unsqueeze(1), cos=cos[doc.row, block_starts], sin=sin[doc.row, block_starts]
        ).squeeze(1)
        # Read back once per document, as a host read per tile would stall the device per tile.
        tied = torch.zeros(doc.stop - doc.start, dtype=torch.bool, device=hidden_states.device)
        chunk = _indexer_chunk(count, queries.shape[2])
        for lo in range(doc.start, doc.stop, chunk):
            hi = min(lo + chunk, doc.stop)
            chunk_blocks = doc.tile_keys(hi) // ratio
            if chunk_blocks == 0:
                continue
            complete = doc.visible_keys(lo, hi, hidden_states.device) // ratio
            selected, chunk_tied = _select_tile(queries[doc.row, lo:hi], pooled, complete, chunk_blocks, block_topk)
            blocks[doc.row, lo:hi] = selected.to(index_dtype)
            tied[lo - doc.start : hi - doc.start] = chunk_tied
        for query in (tied.nonzero().flatten() + doc.start).tolist():
            complete = doc.tile_keys(query + 1) // ratio
            blocks[doc.row, query] = _rescore_like_transformers(queries[doc.row, query], pooled, complete, block_topk)
    return QSASelection(documents=documents, blocks=blocks, spans=spans, compress_ratio=ratio)


def select_qsa_keys(
    indexer: nn.Module,
    hidden_states: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
    seq_idx: torch.Tensor | None = None,
    attention_mask: torch.Tensor | None = None,
) -> QSASelection:
    """Inside a gradient-driven checkpoint the first pass keeps its selection and the recompute reads it back, so the
    backward attends the keys the forward did."""
    scope = active_recompute_scope()
    if scope is None or not scope.grad_enabled:
        return _compute_selection(indexer, hidden_states, position_embeddings, seq_idx, attention_mask)
    cache = scope.slot(indexer, _SELECTION_SLOT)
    if not scope.is_recompute:
        cache["selection"] = _compute_selection(indexer, hidden_states, position_embeddings, seq_idx, attention_mask)
    selection = cache.get("selection")
    if selection is None or selection.blocks.shape[:2] != hidden_states.shape[:2]:
        raise RuntimeError(
            f"{type(indexer).__name__}: the checkpoint recompute found no QSA selection for its "
            f"{tuple(hidden_states.shape[:2])} row set from the forward it replays."
        )
    return selection


def _tile_mask(blocks: torch.Tensor, visible: torch.Tensor, keys: int, ratio: int) -> torch.Tensor:
    """Each query's selected blocks and the keys after its last complete block, up to the keys it sees."""
    block_count = -(-keys // ratio)
    chosen = torch.zeros(blocks.shape[0], block_count + 1, dtype=torch.bool, device=blocks.device)
    chosen.scatter_(1, torch.where(blocks >= 0, blocks.long(), block_count), True)
    chosen = chosen[:, :block_count].repeat_interleave(ratio, dim=1)[:, :keys]
    positions = torch.arange(keys, device=blocks.device)
    tail = positions >= (visible // ratio * ratio).unsqueeze(-1)
    return (chosen | tail) & (positions < visible.unsqueeze(-1))


def _attend_tile(
    query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, mask: torch.Tensor, scale: float
) -> torch.Tensor:
    """Each K/V head's query group is folded into the query axis, so K/V are never repeated. A query whose mask row is
    empty reads zeros: SDPA's output for a fully masked row is undefined, so such a row attends every key and is zeroed."""
    heads, tile, dim = query.shape
    kv_heads, keys, _ = key.shape
    groups = heads // kv_heads
    live = mask.any(dim=-1)
    output = F.scaled_dot_product_attention(
        query.reshape(1, kv_heads, groups * tile, dim),
        key.unsqueeze(0),
        value.unsqueeze(0),
        attn_mask=(mask | ~live.unsqueeze(-1)).expand(groups, tile, keys).reshape(1, 1, groups * tile, keys),
        scale=scale,
    )
    return output.reshape(heads, tile, dim) * live.unsqueeze(-1).to(output.dtype)


def _tiles(selection: QSASelection, device: torch.device):
    """Each query tile's row, query span, K/V slice and mask, one setup for both directions."""
    for doc in selection.documents:
        for lo in range(doc.start, doc.stop, QUERY_TILE):
            hi = min(lo + QUERY_TILE, doc.stop)
            keys = doc.tile_keys(hi)
            if keys:
                visible = doc.visible_keys(lo, hi, device)
                mask = _tile_mask(selection.blocks[doc.row, lo:hi], visible, keys, selection.compress_ratio)
                yield doc.row, lo, hi, slice(doc.key_start, doc.key_start + keys), mask


class _TiledQSAttention(torch.autograd.Function):
    """One query tile at a time; saves Q/K/V and the selection only."""

    @staticmethod
    def forward(ctx, query, key, value, selection, scale):
        ctx.save_for_backward(query, key, value)
        ctx.selection, ctx.scale = selection, scale
        output = torch.zeros_like(query)
        with torch.autocast(query.device.type, enabled=False):
            for row, lo, hi, kv, mask in _tiles(selection, query.device):
                output[row, :, lo:hi] = _attend_tile(
                    query[row, :, lo:hi], key[row, :, kv], value[row, :, kv], mask, scale
                )
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        query, key, value = ctx.saved_tensors
        selection, scale = ctx.selection, ctx.scale
        needs = ctx.needs_input_grad[:3]
        accumulate = torch.float32 if key.dtype in (torch.float16, torch.bfloat16) else key.dtype
        grad_query = torch.zeros_like(query) if needs[0] else None
        grad_key = torch.zeros_like(key, dtype=accumulate) if needs[1] else None
        grad_value = torch.zeros_like(value, dtype=accumulate) if needs[2] else None
        for row, lo, hi, kv, mask in _tiles(selection, query.device):
            inputs = [
                tensor.detach().requires_grad_(need)
                for tensor, need in zip((query[row, :, lo:hi], key[row, :, kv], value[row, :, kv]), needs, strict=True)
            ]
            with torch.enable_grad(), torch.autocast(query.device.type, enabled=False):
                output = _attend_tile(*inputs, mask, scale)
                grads = iter(
                    torch.autograd.grad(output, [t for t in inputs if t.requires_grad], grad_output[row, :, lo:hi])
                )
            if needs[0]:
                grad_query[row, :, lo:hi] = next(grads)
            if needs[1]:
                grad_key[row, :, kv] += next(grads)
            if needs[2]:
                grad_value[row, :, kv] += next(grads)
        return (
            grad_query,
            grad_key.to(key.dtype) if needs[1] else None,
            grad_value.to(value.dtype) if needs[2] else None,
            None,
            None,
        )


def tiled_qsa_attention(
    query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, selection: QSASelection, scale: float
) -> torch.Tensor:
    """Returns ``[batch, length, heads, dim]``. A query with no visible key reads zeros, as SDPA gives a masked row."""
    if query.shape[1] % key.shape[1]:
        raise ValueError(f"{query.shape[1]} query heads do not group over {key.shape[1]} K/V heads")
    # Before the Function, so the tiles, which run with autocast off, see what SDPA under autocast would, and the
    # backward recompute matches whatever autocast state it runs under.
    query, key, value = to_autocast_dtype(query, key, value)
    output = _TiledQSAttention.apply(query, key, value, selection, scale)
    return output.transpose(1, 2)


def qsa_kernel_mode() -> QSAKernelMode:
    mode = env_str("HALO_QSA_KERNEL", _QSA_KERNEL_DEFAULT)
    if mode not in get_args(QSAKernelMode):
        raise ValueError(f"HALO_QSA_KERNEL={mode!r} is not one of {get_args(QSAKernelMode)}")
    return mode


def qsa_kernel_refusal(
    dtype: torch.dtype, device: torch.device, head_dim: int, heads: int, kv_heads: int, ratio: int
) -> str | None:
    """Why the block-sparse kernel cannot serve this layer under this run, else ``None``."""
    if torch.are_deterministic_algorithms_enabled() or flash_attention_deterministic():
        return (
            "the kernel accumulates dK and dV with fp32 atomics, which deterministic mode (full_determinism) forbids"
        )
    failure = _failed_warmups.get((dtype, device, head_dim, heads, kv_heads, ratio))
    if failure is not None:
        return f"the kernel failed to build for this layer at load: {failure}"
    return block_sparse_unsupported(dtype, device, head_dim, heads, kv_heads, ratio)


def resolve_qsa_kernel(
    dtype: torch.dtype, device: torch.device, head_dim: int, heads: int, kv_heads: int, ratio: int
) -> bool:
    """Whether attention runs on the block-sparse kernel: ``HALO_QSA_KERNEL=on`` raises ``ValueError`` where it cannot,
    ``auto`` takes the tiled path for the reason it logs once on rank 0, ``off`` always takes it."""
    mode = qsa_kernel_mode()
    if mode == "off":
        return False
    reason = qsa_kernel_refusal(dtype, device, head_dim, heads, kv_heads, ratio)
    if reason is None:
        return True
    if mode == "on":
        raise ValueError(
            f"HALO_QSA_KERNEL=on cannot run: {reason}. Set HALO_QSA_KERNEL=auto to fall back to the tiled path, or off."
        )
    if is_global_main_process():
        warn_once(logger, _logged_tiled_reasons, reason, "QSA runs on the tiled path: %s.", reason)
    return False


def qsa_attention(
    query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, selection: QSASelection, scale: float
) -> torch.Tensor:
    """``[batch, length, heads, dim]`` from the block-sparse kernel or, where :func:`resolve_qsa_kernel` says so, the
    tiled path."""
    query, key, value = to_autocast_dtype(query, key, value)
    if resolve_qsa_kernel(
        query.dtype, query.device, query.shape[-1], query.shape[1], key.shape[1], selection.compress_ratio
    ):
        return block_sparse_attention(
            query, key, value, selection.blocks, selection.spans, selection.compress_ratio, scale
        ).transpose(1, 2)
    return tiled_qsa_attention(query, key, value, selection, scale)


def qsa_layouts(model: nn.Module) -> set[tuple[torch.device, int, int, int, int]]:
    """``(device, query heads, K/V heads, head_dim, compress_ratio)`` of each QSA layer of ``model``."""
    return {
        (
            next(module.parameters()).device,
            module.config.num_attention_heads,
            module.config.num_key_value_heads,
            module.head_dim,
            module.indexer.compress_ratio,
        )
        for module in model.modules()
        if isinstance(module, qwen4_exp_modeling.Qwen4ExpTextAttention)
    }


def warm_qsa_kernels(model: nn.Module) -> None:
    """Compile the block-sparse kernels the model's QSA layers run, so no step compiles one. Rank-local, no collective;
    bf16 and fp16 are both warmed because the run dtype is not known here. A layout the kernel cannot build runs the
    tiled path under ``HALO_QSA_KERNEL=auto``, logged once, and raises ``RuntimeError`` under ``on``."""
    if qsa_kernel_mode() == "off" or not torch.cuda.is_available():
        return
    for device, heads, kv_heads, head_dim, ratio in sorted(qsa_layouts(model), key=str):
        if device.type != "cuda":
            continue
        for dtype in SUPPORTED_DTYPES:
            if not resolve_qsa_kernel(dtype, device, head_dim, heads, kv_heads, ratio):
                continue
            try:
                warm_block_sparse_kernels(device, dtype, heads, kv_heads, head_dim, ratio)
            except TritonError as error:
                if qsa_kernel_mode() == "on":
                    raise RuntimeError(
                        f"HALO_QSA_KERNEL=on: the block-sparse kernel failed to build for {dtype} with {heads} query "
                        f"heads, {kv_heads} K/V heads and head_dim {head_dim}: {error}. Set HALO_QSA_KERNEL=auto to "
                        "fall back to the tiled path, or off."
                    ) from error
                _failed_warmups[(dtype, device, head_dim, heads, kv_heads, ratio)] = str(error)


def qsa_attention_forward(
    module: nn.Module,
    hidden_states: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
    attention_mask: torch.Tensor | None = None,
    past_key_values=None,
    seq_idx: torch.Tensor | None = None,
    **kwargs,
) -> tuple[torch.Tensor, None]:
    """``attention_mask`` is the 2-D padding mask or ``None`` and ``seq_idx`` marks the documents."""
    if past_key_values is not None:
        raise ValueError("QSA serves uncached rows; a call with past_key_values takes the stock forward.")
    if module.training and module.attention_dropout:
        raise ValueError(f"QSA applies no attention dropout, got attention_dropout={module.attention_dropout}")
    if attention_mask is not None and attention_mask.ndim != 2:
        raise ValueError(
            f"QSA reads documents from seq_idx and padding from a 2-D mask, got a {attention_mask.ndim}-D "
            "attention_mask: pass the model the per-layer-type mask mapping."
        )
    selection = select_qsa_keys(module.indexer, hidden_states, position_embeddings, seq_idx, attention_mask)
    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, module.head_dim)
    query, gate = torch.chunk(module.q_proj(hidden_states).view(*input_shape, -1, module.head_dim * 2), 2, dim=-1)
    gate = gate.reshape(*input_shape, -1)
    query = module.q_norm(query.view(hidden_shape)).transpose(1, 2)
    key = module.k_norm(module.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
    value = module.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    cos, sin = position_embeddings
    query, key = qwen4_exp_modeling.apply_rotary_pos_emb(query, key, cos, sin)
    output = qsa_attention(query, key, value, selection, module.scaling)
    output = output.reshape(*input_shape, -1) * torch.sigmoid(gate)
    return module.o_proj(output), None
