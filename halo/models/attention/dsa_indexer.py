"""GLM-5 Next's DSA key selection on a padded batch, scored a bounded number of one row's queries at a time. Upstream
scores every query against every pool in one call: at the release's sizes three ``[B, S, heads, S / pool]`` tensors of
96 bytes a squared token outside autocast (an SFT step runs none), 48 under bf16 autocast, and past one row a 128-byte
broadcast copy of the pool keys, which pass the rest of the step's memory near 32k tokens."""

import torch
import torch.nn.functional as F
from transformers.models.glm5_next import modeling_glm5_next

from halo.models.attention import tree_sparse
from halo.models.gradient_checkpointing import patch_layer_forward

_stock_forward = modeling_glm5_next.Glm5NextTextIndexer.forward


@torch.no_grad()
def select_keys_in_query_chunks(
    indexer, hidden_states: torch.Tensor, q_resid: torch.Tensor, attention_mask: torch.Tensor
) -> torch.Tensor:
    """The stock indexer's ``[B, S, index_topk (+ index_kpool - 1)]`` int32 key rows (``-1`` pads) of a forward with
    no cache, through the same projections, pools, visibility and tail."""
    batch_size, seq_len = hidden_states.shape[:2]
    query = indexer.wq_b(q_resid).view(batch_size, seq_len, -1, indexer.head_dim)
    key = indexer.k_norm(indexer.wk(hidden_states))
    gate_scores = F.linear(hidden_states, indexer.index_kpool_compress_gate)
    packed_states = torch.cat([key, gate_scores, attention_mask.to(key.dtype)[..., None]], dim=-1)
    valid_keys = packed_states[..., -1].bool()
    visible_tokens = indexer.get_visible_tokens(valid_keys=valid_keys, q_length=seq_len, current_length=seq_len)
    pool_keys, pool_indices, pool_valid = indexer.get_pooled_states(packed_states=packed_states)
    weights = indexer.weights_proj(hidden_states.to(indexer.weights_proj.weight.dtype)).float() * indexer.n_heads**-0.5

    pools = pool_keys.shape[1]
    pool_keys = pool_keys.transpose(-1, -2).float().unsqueeze(1)
    pool_end = pool_indices[..., -1].clamp(0, seq_len - 1)
    select_k = min(indexer.index_topk // indexer.index_kpool, pools)
    # One row a call: rows scored together broadcast the pool keys to a [rows, queries, head_dim, pools] copy,
    # head_dim / heads times the scores.
    chunk = max(1, tree_sparse.INDEXER_SCORES_PER_CALL // (indexer.n_heads * max(pools, 1)))
    rows = []
    for row in range(batch_size):
        # A one-row batch dim keeps upstream's per-query products: a squeezed one folds into a single wide matmul.
        one = slice(row, row + 1)
        row_keys = []
        for start in range(0, seq_len, chunk):
            queries = slice(start, start + chunk)
            scores = torch.matmul(query[one, queries].float(), pool_keys[one]).mul_(indexer.softmax_scale).relu_()
            scores = torch.matmul(weights[one, queries].unsqueeze(-2), scores).squeeze(-2)
            # A pool is selectable only if its final token is visible to the query.
            visible = visible_tokens[one, queries].gather(-1, pool_end[one, None, :].expand(-1, scores.shape[1], -1))
            candidates = visible & pool_valid[one, None]
            selected = scores.masked_fill(~candidates, torch.finfo(scores.dtype).min).topk(select_k, dim=-1).indices
            keys = pool_indices[row][selected].masked_fill(~candidates.gather(-1, selected)[..., None], -1)
            row_keys.append(keys.flatten(-2))
        rows.append(torch.cat(row_keys, dim=1))
    topk_indices = torch.cat(rows)

    output_width = indexer.index_topk
    if indexer.index_kpool_always_select_tail:
        topk_indices = indexer.append_visible_tail(topk_indices, visible_tokens, valid_keys)
        output_width += indexer.index_kpool - 1
    topk_indices = F.pad(topk_indices, (0, output_width - topk_indices.shape[-1]), value=-1)[..., :output_width]
    return topk_indices.masked_fill(~attention_mask[..., None], -1).to(torch.int32)


def _forward(indexer, hidden_states, q_resid, attention_mask, past_key_values):
    """A cache holds keys of earlier calls and a static one its own key length: upstream's forward keeps those."""
    if past_key_values is not None:
        return _stock_forward(indexer, hidden_states, q_resid, attention_mask, past_key_values)
    return select_keys_in_query_chunks(indexer, hidden_states, q_resid, attention_mask)


def patch_glm5_next_indexer() -> None:
    """Idempotent."""
    patch_layer_forward(modeling_glm5_next.Glm5NextTextIndexer, _forward)
