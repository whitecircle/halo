"""Tiled Qwen sparse attention (QSA) for uncached ``qwen4_exp`` training rows, memory linear in the row length,
selecting the blocks transformers' dense indexer selects. A checkpoint's recompute attends the first pass's."""

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
import transformers.models.qwen4_exp.modeling_qwen4_exp as qwen4_exp_modeling
from torch import nn
from torch.autograd.function import once_differentiable

from halo.models.gradient_checkpointing import active_recompute_scope
from halo.models.segment_markers import document_boundaries

QUERY_TILE = 128
_SELECTION_SLOT = "qsa_selection"


@dataclass(frozen=True)
class QSADocument:
    row: int
    start: int
    stop: int
    key_start: int
    key_stop: int

    def visible_keys(self, lo: int, hi: int, device: torch.device) -> torch.Tensor:
        """Per query of ``[lo, hi)``, how many of the document's keys it sees, those up to itself."""
        return (torch.arange(lo, hi, device=device) - self.key_start + 1).clamp(0, self.key_stop - self.key_start)

    def tile_keys(self, hi: int) -> int:
        """The span every tile's K/V slice covers."""
        return min(max(hi - self.key_start, 0), self.key_stop - self.key_start)


@dataclass(frozen=True)
class QSASelection:
    """``blocks[row, query]`` holds block indices counted from the document's first key, ``-1`` in unused slots; the
    keys after the last complete block are always attended."""

    documents: tuple[QSADocument, ...]
    blocks: torch.Tensor
    compress_ratio: int


def qsa_documents(
    seq_idx: torch.Tensor | None, attention_mask: torch.Tensor | None, batch: int, length: int, device: torch.device
) -> tuple[QSADocument, ...]:
    """In row-major order. A document's unpadded tokens must be contiguous, as left or right padding leaves them."""
    if attention_mask is not None and attention_mask.shape != (batch, length):
        raise ValueError(
            f"Tiled QSA takes a 2-D [batch, sequence] = {(batch, length)} padding mask, got "
            f"{tuple(attention_mask.shape)}"
        )
    boundaries = document_boundaries(seq_idx, batch, length, device)
    rows, starts = boundaries.nonzero(as_tuple=True)
    document = boundaries.flatten().cumsum(0) - 1
    last_of_row = torch.ones_like(rows, dtype=torch.bool)
    last_of_row[:-1] = rows[1:] != rows[:-1]
    stops = torch.where(last_of_row, length, starts.roll(-1))
    valid = (
        torch.ones(batch * length, dtype=torch.bool, device=device)
        if attention_mask is None
        else attention_mask.to(device=device, dtype=torch.bool).flatten()
    )
    positions = torch.arange(length, device=device).repeat(batch)
    count = torch.zeros_like(starts).scatter_add_(0, document, valid.long())
    first = starts.clone().scatter_reduce_(0, document[valid], positions[valid], "amin", include_self=False)
    last = (starts - 1).scatter_reduce_(0, document[valid], positions[valid], "amax", include_self=False)
    if bool(((last - first + 1) != count).any()):
        raise ValueError(
            "Tiled QSA needs each document's unpadded tokens to be contiguous (left or right padding); the padding "
            "mask interleaves padding inside a document."
        )
    spans = torch.stack([rows, starts, stops, first, first + count]).T.tolist()
    return tuple(QSADocument(*span) for span in spans)


def _block_scores(queries: torch.Tensor, pooled: torch.Tensor) -> torch.Tensor:
    """The head-summed ReLU of the fp32 dot products, in transformers' op order."""
    scores = torch.matmul(queries.float(), pooled.float().transpose(-1, -2)).transpose(-1, -2)
    return torch.relu(scores).sum(dim=-1) / math.sqrt(pooled.shape[-1])


def _rescore_like_transformers(query: torch.Tensor, pooled: torch.Tensor, count: int, block_topk: int) -> torch.Tensor:
    """One query against its ``count`` complete blocks, so a tie at the cutoff resolves in transformers' order."""
    return _block_scores(query, pooled[:count]).topk(min(block_topk, count), dim=0).indices


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

    documents = qsa_documents(seq_idx, attention_mask, batch, length, hidden_states.device)
    most_blocks = max((doc.key_stop - doc.key_start) // ratio for doc in documents)
    index_dtype = torch.int16 if most_blocks <= torch.iinfo(torch.int16).max else torch.int32
    blocks = torch.full((batch, length, block_topk), -1, dtype=index_dtype, device=hidden_states.device)
    for doc in documents:
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
        for lo in range(doc.start, doc.stop, QUERY_TILE):
            hi = min(lo + QUERY_TILE, doc.stop)
            tile_blocks = doc.tile_keys(hi) // ratio
            if tile_blocks == 0:
                continue
            complete = doc.visible_keys(lo, hi, hidden_states.device) // ratio
            selected, tile_tied = _select_tile(queries[doc.row, lo:hi], pooled, complete, tile_blocks, block_topk)
            blocks[doc.row, lo:hi] = selected.to(index_dtype)
            tied[lo - doc.start : hi - doc.start] = tile_tied
        for query in (tied.nonzero().flatten() + doc.start).tolist():
            complete = doc.tile_keys(query + 1) // ratio
            blocks[doc.row, query] = _rescore_like_transformers(queries[doc.row, query], pooled, complete, block_topk)
    return QSASelection(documents=documents, blocks=blocks, compress_ratio=ratio)


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
    """Each K/V head's query group is folded into the query axis, so K/V are never repeated."""
    heads, tile, dim = query.shape
    kv_heads, keys, _ = key.shape
    groups = heads // kv_heads
    output = F.scaled_dot_product_attention(
        query.reshape(1, kv_heads, groups * tile, dim),
        key.unsqueeze(0),
        value.unsqueeze(0),
        attn_mask=mask.expand(groups, tile, keys).reshape(1, 1, groups * tile, keys),
        scale=scale,
    )
    return output.reshape(heads, tile, dim)


def _tiles(selection: QSASelection, blocks: torch.Tensor, device: torch.device):
    """Each query tile's row, query span, K/V slice and mask, one setup for both directions."""
    for doc in selection.documents:
        for lo in range(doc.start, doc.stop, QUERY_TILE):
            hi = min(lo + QUERY_TILE, doc.stop)
            keys = doc.tile_keys(hi)
            if keys:
                visible = doc.visible_keys(lo, hi, device)
                mask = _tile_mask(blocks[doc.row, lo:hi], visible, keys, selection.compress_ratio)
                yield doc.row, lo, hi, slice(doc.key_start, doc.key_start + keys), mask


class _TiledQSAttention(torch.autograd.Function):
    """One query tile at a time; saves Q/K/V and the block ids only."""

    @staticmethod
    def forward(ctx, query, key, value, blocks, selection, scale):
        ctx.save_for_backward(query, key, value, blocks)
        ctx.selection, ctx.scale = selection, scale
        output = torch.zeros_like(query)
        with torch.autocast(query.device.type, enabled=False):
            for row, lo, hi, kv, mask in _tiles(selection, blocks, query.device):
                output[row, :, lo:hi] = _attend_tile(
                    query[row, :, lo:hi], key[row, :, kv], value[row, :, kv], mask, scale
                )
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        query, key, value, blocks = ctx.saved_tensors
        selection, scale = ctx.selection, ctx.scale
        needs = ctx.needs_input_grad[:3]
        accumulate = torch.float32 if key.dtype in (torch.float16, torch.bfloat16) else key.dtype
        grad_query = torch.zeros_like(query) if needs[0] else None
        grad_key = torch.zeros_like(key, dtype=accumulate) if needs[1] else None
        grad_value = torch.zeros_like(value, dtype=accumulate) if needs[2] else None
        for row, lo, hi, kv, mask in _tiles(selection, blocks, query.device):
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
            None,
        )


def tiled_qsa_attention(
    query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, selection: QSASelection, scale: float
) -> torch.Tensor:
    """Returns ``[batch, length, heads, dim]``. A query with no visible key reads zeros, as SDPA gives a masked row."""
    if query.shape[1] % key.shape[1]:
        raise ValueError(f"{query.shape[1]} query heads do not group over {key.shape[1]} K/V heads")
    device_type = query.device.type
    if torch.is_autocast_enabled(device_type):
        # Cast before the Function so the tiles, which run with autocast off, see what SDPA under autocast would,
        # and the backward recompute matches whatever autocast state it runs under.
        dtype = torch.get_autocast_dtype(device_type)
        query, key, value = (t.to(dtype) if t.dtype != torch.float64 else t for t in (query, key, value))
    output = _TiledQSAttention.apply(query, key, value, selection.blocks, selection, scale)
    return output.transpose(1, 2)


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
        raise ValueError("Tiled QSA serves uncached rows; a call with past_key_values takes the stock forward.")
    if module.training and module.attention_dropout:
        raise ValueError(f"Tiled QSA applies no attention dropout, got attention_dropout={module.attention_dropout}")
    if attention_mask is not None and attention_mask.ndim != 2:
        raise ValueError(
            f"Tiled QSA reads documents from seq_idx and padding from a 2-D mask, got a {attention_mask.ndim}-D "
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
    output = tiled_qsa_attention(query, key, value, selection, module.scaling)
    output = output.reshape(*input_shape, -1) * torch.sigmoid(gate)
    return module.o_proj(output), None
