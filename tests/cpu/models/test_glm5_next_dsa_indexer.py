#!/usr/bin/env python
"""GLM-5 Next's DSA indexer on a padded batch selects the keys upstream's forward selects, bit for bit, while it
scores a bounded number of one row's queries at a time and copies no operand (``halo/models/attention/dsa_indexer.py``),
and the family patches install it."""

import pytest
import torch
from torch.utils._python_dispatch import TorchDispatchMode
from transformers import Glm5NextConfig, Glm5NextTextConfig
from transformers.models.glm5_next.modeling_glm5_next import Glm5NextTextIndexer

from halo.model_loading.model_preparation import apply_family_patches
from halo.models.attention import dsa_indexer, tree_sparse
from halo.models.attention.dsa_indexer import patch_glm5_next_indexer, select_keys_in_query_chunks
from tests.common.tiny_models import TINY_MOE_FAMILIES, tiny_family_model
from tests.common.trees import randomize_indexer_pools

# Eight indexer heads: with fewer, a random model scores many pools exactly zero and the top-k's order among those
# ties is unspecified. A 4-token pool and two pools per selection, so rows this short still select.
OVERRIDES = {"index_n_heads": 8, "index_topk": 8, "index_kpool": 4, "index_kpool_always_select_tail": True}
SEQ_LEN = 61
# (real tokens of each row, padding side): pooling starts at a row's first real token, so both sides are covered.
BATCHES = {
    "one-row": ((SEQ_LEN,), "right"),
    "right-padded": ((SEQ_LEN, 38, 3), "right"),
    "left-padded": ((SEQ_LEN, 38, 3), "left"),
}


class _ScoringOps(TorchDispatchMode):
    """The element count of every batched product's result, and the shape of every copy an op makes of its input
    (a broadcast operand a reshape cannot view)."""

    def __init__(self):
        super().__init__()
        self.products = []
        self.copies = []

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        result = func(*args, **(kwargs or {}))
        if func is torch.ops.aten.bmm.default:
            self.products.append(result.numel())
        elif func is torch.ops.aten.clone.default:
            self.copies.append(tuple(result.shape))
        return result


def _pool_key_copies(recorded: _ScoringOps, indexer) -> list[tuple[int, ...]]:
    return [shape for shape in recorded.copies if shape[-2:] == (indexer.head_dim, SEQ_LEN // indexer.index_kpool)]


@pytest.fixture(scope="module")
def attention():
    torch.manual_seed(0)
    model = tiny_family_model(TINY_MOE_FAMILIES["glm5_next"], overrides=OVERRIDES).eval()
    randomize_indexer_pools(model)
    return next(m for m in model.modules() if isinstance(getattr(m, "indexer", None), Glm5NextTextIndexer))


def _inputs(attention, name: str):
    lengths, side = BATCHES[name]
    generator = torch.Generator().manual_seed(1)
    hidden = torch.randn(len(lengths), SEQ_LEN, attention.hidden_size, generator=generator)
    positions = torch.arange(SEQ_LEN)
    real = torch.tensor(lengths)[:, None]
    mask = positions < real if side == "right" else positions >= SEQ_LEN - real
    with torch.no_grad():
        q_resid = attention.q_a_layernorm(attention.q_a_proj(hidden))
    return hidden, q_resid, mask


@pytest.mark.parametrize("name", sorted(BATCHES))
@pytest.mark.parametrize("queries_per_call", [1, 5, 37, SEQ_LEN])
def test_the_chunked_selection_is_upstreams(attention, monkeypatch, name, queries_per_call):
    indexer = attention.indexer
    hidden, q_resid, mask = _inputs(attention, name)
    stock = dsa_indexer._stock_forward(indexer, hidden, q_resid, mask, None)
    pools = SEQ_LEN // indexer.index_kpool
    budget = indexer.n_heads * pools * queries_per_call
    monkeypatch.setattr(tree_sparse, "INDEXER_SCORES_PER_CALL", budget)
    with _ScoringOps() as recorded:
        chunked = select_keys_in_query_chunks(indexer, hidden, q_resid, mask)
    assert torch.equal(chunked, stock)
    assert (stock >= 0).any(), "no query selected a key"
    assert max(recorded.products) <= budget, "a call scored more than its budget"
    calls = hidden.shape[0] * -(-SEQ_LEN // queries_per_call)
    assert len(recorded.products) == 2 * calls, "the scores and their head sum, per row and chunk"
    assert not _pool_key_copies(recorded, indexer), "the pool keys were broadcast into copies"


@pytest.mark.parametrize("name", sorted(name for name, (lengths, _) in BATCHES.items() if len(lengths) > 1))
def test_the_recorder_catches_upstreams_pool_key_copy(attention, name):
    """Premise of the no-copy assertion: past one row, upstream's forward broadcasts the pool keys into a copy."""
    hidden, q_resid, mask = _inputs(attention, name)
    with _ScoringOps() as recorded:
        dsa_indexer._stock_forward(attention.indexer, hidden, q_resid, mask, None)
    assert _pool_key_copies(recorded, attention.indexer)


def test_padding_queries_and_keys_select_nothing(attention):
    hidden, q_resid, mask = _inputs(attention, "right-padded")
    selected = select_keys_in_query_chunks(attention.indexer, hidden, q_resid, mask)
    assert (selected[~mask] == -1).all()
    for row, real in zip(selected, mask.sum(-1).tolist(), strict=True):
        assert int(row.max()) < real


def test_the_patch_keeps_upstreams_forward_for_a_cached_call(attention, monkeypatch):
    monkeypatch.setattr(Glm5NextTextIndexer, "forward", Glm5NextTextIndexer.forward)
    patch_glm5_next_indexer()
    patch_glm5_next_indexer()
    indexer = attention.indexer
    hidden, q_resid, mask = _inputs(attention, "right-padded")
    stock = dsa_indexer._stock_forward(indexer, hidden, q_resid, mask, None)
    assert torch.equal(indexer(hidden, q_resid, mask, None), stock)

    cached_calls = []
    monkeypatch.setattr(dsa_indexer, "_stock_forward", lambda *args: cached_calls.append(args) or "stock")
    cache = object()
    assert indexer(hidden, q_resid, mask, cache) == "stock"
    assert cached_calls == [(indexer, hidden, q_resid, mask, cache)]
    monkeypatch.setattr(dsa_indexer, "select_keys_in_query_chunks", lambda *args: "chunked")
    assert indexer(hidden, q_resid, mask, None) == "chunked"


@pytest.mark.parametrize("config", [Glm5NextConfig(), Glm5NextTextConfig()], ids=["composite", "text"])
def test_the_family_patches_install_the_chunked_selection(monkeypatch, config):
    monkeypatch.setattr(Glm5NextTextIndexer, "forward", dsa_indexer._stock_forward)
    apply_family_patches(config, None)
    assert Glm5NextTextIndexer.forward is dsa_indexer._forward


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
