"""Tree attention for GLM-5 Next's DeepSeek sparse attention: every query selects its keys along its own path and
attends only its selected MLA latent rows, with the key projection folded into the query."""

import functools

import torch
import torch.nn.functional as F
from peft.tuners.tuners_utils import BaseTunerLayer
from peft.utils.constants import INCLUDE_LINEAR_LAYERS_SHORTHAND

from halo.kernels.fp32_matmul import bmm_fp32
from halo.models.attention.tree_plan import TREE_ATTENTION_KEY
from halo.models.config_access import text_config

# fp32 indexer scores one selection call holds (queries x candidate pools x indexer heads): 1 GiB.
INDEXER_SCORES_PER_CALL = 1 << 28
# Latent rows one attention call gathers (queries x selected keys): 1 GiB at bf16 and kv_lora_rank 512
# in the forward; the backward also holds two fp32 products of that shape, about 8 GiB in all.
ATTENTION_KEYS_PER_CALL = 1 << 20

# PEFT matches a list entry of its targets against the end of a module's path.
_FOLDED_PROJECTION_PATH = "self_attn.kv_b_proj"
_FOLDED_ADAPTER_REFUSAL = (
    "tree packing runs GLM-5 Next's sparse attention on the MLA latent and folds kv_b_proj's weight into the "
    "query, so an adapter on kv_b_proj would never be applied. Leave kv_b_proj out of lora_target_modules "
    "(all-linear includes it)."
)
_QUANTIZED_REFUSAL = (
    "tree packing runs GLM-5 Next's sparse attention on the MLA latent and folds kv_b_proj's weight into the "
    "query, which a quantized load does not hold as a plain matrix. Train tree rows on unquantized weights: drop "
    "load_in_4bit/load_in_8bit (plain LoRA), and convert the fp8 hub release to bf16 first "
    "(scripts/before_training/convert_glm5_bf16.py)."
)


def _path_back(predecessor: torch.Tensor, tokens: torch.Tensor, steps: int) -> torch.Tensor:
    """Each token and its ``steps - 1`` nearest predecessors on its path, ``-1`` past a root."""
    chain = [tokens]
    for _ in range(steps - 1):
        previous = chain[-1]
        chain.append(torch.where(previous >= 0, predecessor[previous.clamp_min(0)], previous))
    return torch.stack(chain, dim=-1)


def _pool_ends(links: dict, pool: int) -> dict[str, torch.Tensor]:
    """Kept on ``links``, built once per plan, as finding the ends is a device sync."""
    key = ("pool_ends", pool)
    if key not in links:
        ends = (links["position"] % pool == pool - 1).nonzero().squeeze(-1)
        links[key] = {
            "ends": ends,
            "members": _path_back(links["predecessor"], ends, pool).flip(-1),
            "end_bound": links["subtree_end"][ends],
        }
    return links[key]


def _query_chunk_pools(links: dict, pool: int, tokens: int, chunk: int) -> list[tuple[int, int, int, int]]:
    """``(start, stop, low, high)`` as host ints, built once with one ``searchsorted`` and one host read, which every
    layer and recompute shares. ``[low, high)`` are the pools of the chunk's trees that end before the chunk does."""
    key = ("query_chunk_pools", pool, chunk)
    if key not in links:
        ends = _pool_ends(links, pool)["ends"]
        starts = torch.arange(0, tokens, chunk, device=ends.device)
        bounds = torch.stack([links["tree_start"][starts], (starts + chunk).clamp_max(tokens)])
        lows, highs = torch.searchsorted(ends, bounds).tolist()
        links[key] = [
            (start, min(start + chunk, tokens), low, high)
            for start, low, high in zip(range(0, tokens, chunk), lows, highs, strict=True)
        ]
    return links[key]


@torch.no_grad()
def tree_key_selection(indexer, hidden_states: torch.Tensor, q_resid: torch.Tensor, links: dict) -> torch.Tensor:
    """The stock indexer's ``[1, T, index_topk (+ index_kpool - 1)]`` int32 key rows (``-1`` pads), per path."""
    hidden_states, q_resid = hidden_states[0], q_resid[0]
    tokens, pool = hidden_states.shape[0], indexer.index_kpool
    query = indexer.wq_b(q_resid).view(tokens, indexer.n_heads, indexer.head_dim)
    key = indexer.k_norm(indexer.wk(hidden_states))
    gate = F.linear(hidden_states, indexer.index_kpool_compress_gate)
    weights = indexer.weights_proj(hidden_states.to(indexer.weights_proj.weight.dtype)).float() * indexer.n_heads**-0.5
    predecessor, position = links["predecessor"], links["position"]

    pools = _pool_ends(links, pool)
    ends, members, end_bound = pools["ends"], pools["members"], pools["end_bound"]
    logits = gate[members].float() + indexer.index_kpool_compress_ape.float()
    pool_keys = (torch.nan_to_num(logits.softmax(dim=1)).to(key.dtype) * key[members]).sum(dim=1)

    select = min(indexer.index_topk // pool, ends.numel())
    tail = pool - 1 if indexer.index_kpool_always_select_tail else 0
    width = indexer.index_topk + tail
    chunk = max(1, INDEXER_SCORES_PER_CALL // (indexer.n_heads * max(ends.numel(), 1)))
    rows = []
    for start, stop, low, high in _query_chunk_pools(links, pool, tokens, chunk):
        queries = torch.arange(start, stop, device=hidden_states.device)
        scores = torch.matmul(query[start:stop].float(), pool_keys[low:high].float().T)
        scores = scores.mul_(indexer.softmax_scale).relu_()
        scores = torch.matmul(weights[start:stop].unsqueeze(-2), scores).squeeze(-2)
        visible = (ends[low:high] <= queries[:, None]) & (queries[:, None] < end_bound[low:high])
        scores = scores.masked_fill(~visible, torch.finfo(scores.dtype).min)
        picked = scores.sort(dim=-1, descending=True, stable=True).indices[:, : min(select, high - low)]
        keys = members[low:high][picked].masked_fill(~visible.gather(-1, picked)[..., None], -1).flatten(-2)
        if tail:
            count = (position[start:stop] + 1) % pool  # the tokens past the last complete pool, itself included
            behind = _path_back(predecessor, queries, tail)
            keys = torch.cat(
                [keys, behind.masked_fill(torch.arange(tail, device=keys.device) >= count[:, None], -1)], -1
            )
        rows.append(F.pad(keys, (0, width - keys.shape[-1]), value=-1)[:, :width])
    return torch.cat(rows)[None].to(torch.int32)


def _wide(dtype: torch.dtype) -> torch.dtype:
    return torch.promote_types(dtype, torch.float32)


def _query_chunks(keys: torch.Tensor):
    step = max(1, ATTENTION_KEYS_PER_CALL // keys.shape[1])
    for start in range(0, keys.shape[0], step):
        yield slice(start, min(start + step, keys.shape[0]))


def _chunk_scores(query, latent, rows, scale) -> tuple[torch.Tensor, torch.Tensor]:
    """The chunk's selected latent rows and their masked fp32 scores, one setup for both directions, whose mismatch
    would give wrong gradients."""
    gathered = latent[rows.clamp_min(0)]
    scores = (bmm_fp32(query, gathered.transpose(1, 2)) * scale).masked_fill((rows < 0)[:, None], float("-inf"))
    return gathered, scores


class _SparseLatentAttention(torch.autograd.Function):
    """Every query's softmax over its own selected latent rows. Scores and the softmax are fp32 and the two products
    run at the inputs' precision with fp32 results, so it is applied outside autocast. A query with no key gets a zero
    output, as SDPA's safe softmax gives it. The backward sums a latent row's gradient over its queries in fp32."""

    @staticmethod
    def forward(ctx, query, latent, keys, scale):
        out = torch.empty_like(query)
        lse = torch.empty(query.shape[:2], dtype=_wide(query.dtype), device=query.device)
        for chunk in _query_chunks(keys):
            gathered, scores = _chunk_scores(query[chunk], latent, keys[chunk], scale)
            lse[chunk] = torch.logsumexp(scores, dim=-1).nan_to_num(neginf=0.0)
            probs = torch.exp(scores - lse[chunk][..., None])
            out[chunk] = bmm_fp32(probs.to(latent.dtype), gathered).to(out.dtype)
        ctx.save_for_backward(query, latent, keys, out, lse)
        ctx.scale = scale
        return out

    @staticmethod
    def backward(ctx, grad):
        query, latent, keys, out, lse = ctx.saved_tensors
        d_query = torch.empty_like(query)
        d_latent = torch.zeros(latent.shape, dtype=_wide(latent.dtype), device=latent.device)
        for chunk in _query_chunks(keys):
            rows = keys[chunk]
            gathered, scores = _chunk_scores(query[chunk], latent, rows, ctx.scale)
            probs = torch.exp(scores - lse[chunk][..., None])
            upstream = grad[chunk]
            delta = (upstream.to(lse.dtype) * out[chunk].to(lse.dtype)).sum(-1, keepdim=True)
            d_scores = (probs * (bmm_fp32(upstream, gathered.transpose(1, 2)) - delta) * ctx.scale).to(query.dtype)
            d_query[chunk] = bmm_fp32(d_scores, gathered).to(query.dtype)
            d_gathered = bmm_fp32(d_scores.transpose(1, 2), query[chunk]) + bmm_fp32(
                probs.to(upstream.dtype).transpose(1, 2), upstream
            )
            # Every slot is added, sparing a host read of the selected count; a pad slot adds an exact zero to a row
            # spread by its flat slot index, which keeps the pads' atomics off any one row.
            selected = rows >= 0
            spread = torch.arange(rows.numel(), device=rows.device).view_as(rows) % latent.shape[0]
            d_gathered.masked_fill_(~selected[..., None], 0.0)
            d_latent.index_add_(0, torch.where(selected, rows, spread).flatten(), d_gathered.flatten(0, 1))
        return d_query, d_latent.to(latent.dtype), None, None


def _dropout_refusal(attention_dropout: float) -> str:
    return f"tree packing's sparse attention takes no dropout, but attention_dropout={attention_dropout}. Set it to 0."


def _adapts_folded_projection(lora_target_modules: str | list[str]) -> bool:
    """Read without the model. A regex, or an entry pinned to one layer's path, resolves only on the loaded tree."""
    if isinstance(lora_target_modules, str):
        return lora_target_modules.lower() == INCLUDE_LINEAR_LAYERS_SHORTHAND
    entries = [str(entry) for entry in lora_target_modules]
    if [entry.lower() for entry in entries] == [INCLUDE_LINEAR_LAYERS_SHORTHAND]:
        return True
    return any(entry == _FOLDED_PROJECTION_PATH or _FOLDED_PROJECTION_PATH.endswith(f".{entry}") for entry in entries)


def reject_unfoldable_tree_options(
    model_config, lora_target_modules: str | list[str] | None, quantization_config=None
) -> None:
    """Refuse, before the load, what the fold would skip or misread without a word: adapter targets that wrap
    ``kv_b_proj``, a quantized ``kv_b_proj`` and a training-time ``attention_dropout``."""
    if lora_target_modules is not None and _adapts_folded_projection(lora_target_modules):
        raise ValueError(_FOLDED_ADAPTER_REFUSAL)
    if quantization_config is not None or getattr(model_config, "quantization_config", None) is not None:
        raise ValueError(_QUANTIZED_REFUSAL)
    if attention_dropout := getattr(text_config(model_config), "attention_dropout", 0.0):
        raise ValueError(_dropout_refusal(attention_dropout))


def _reject_unfoldable_layer(attention) -> None:
    """Checked on every tree batch, as the trainer wraps its adapters after tree attention is installed and only the
    loaded module tree resolves every target spelling."""
    if isinstance(attention.kv_b_proj, BaseTunerLayer):
        raise ValueError(_FOLDED_ADAPTER_REFUSAL)
    if attention.training and attention.attention_dropout:
        raise ValueError(_dropout_refusal(attention.attention_dropout))


def tree_sparse_forward(forward):
    @functools.wraps(forward)
    def with_tree_plan(self, hidden_states, attention_mask, past_key_values=None, prev_topk_indices=None, **kwargs):
        plan = kwargs.pop(TREE_ATTENTION_KEY, None)
        if plan is None:
            return forward(
                self,
                hidden_states,
                attention_mask,
                past_key_values=past_key_values,
                prev_topk_indices=prev_topk_indices,
                **kwargs,
            )
        if past_key_values is not None or hidden_states.shape[0] != 1:
            raise ValueError("a tree row is one flattened training row, run without a KV cache")
        _reject_unfoldable_layer(self)
        tokens, heads = hidden_states.shape[1], self.num_heads
        q_resid = self.q_a_layernorm(self.q_a_proj(hidden_states))
        query = self.q_b_proj(q_resid).view(tokens, heads, self.qk_head_dim)
        latent = self.kv_a_layernorm(self.kv_a_proj_with_mqa(hidden_states)[..., : self.kv_lora_rank])[0]
        if self.indexer is None:
            if prev_topk_indices is None:
                raise ValueError("Shared DSA layers require top-k indices from a previous full indexer layer.")
            selected = prev_topk_indices
        else:
            selected = tree_key_selection(self.indexer, hidden_states, q_resid, plan["sparse"])
        weight = self.kv_b_proj.weight.view(heads, self.qk_nope_head_dim + self.v_head_dim, self.kv_lora_rank)
        key_weight, value_weight = weight.split([self.qk_nope_head_dim, self.v_head_dim], dim=1)
        folded = torch.einsum("thd,hdr->thr", query, key_weight)
        # Under autocast the fold comes out of its matmul in the autocast dtype and the latent out of its norm in the
        # parameters' dtype: the attention reads both in the former and keeps its own precision outside autocast.
        with torch.autocast(latent.device.type, enabled=False):
            out = _SparseLatentAttention.apply(folded, latent.to(folded.dtype), selected[0].long(), self.scaling)
        out = torch.einsum("thr,hvr->thv", out, value_weight).reshape(1, tokens, heads * self.v_head_dim)
        return self.o_proj(out), None, selected if self.next_skip_topk else None

    return with_tree_plan
