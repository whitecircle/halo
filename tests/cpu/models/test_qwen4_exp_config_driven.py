#!/usr/bin/env python
"""The ``qwen4_exp`` family code follows its config: a tiny model whose QSA indexer, hyper-connections, per-layer
embeddings, n-gram table and eos id all differ from the default tiny config computes transformers' logits."""

import pytest
import torch

import halo.models.families.qwen4_exp.sparse_attention as qsa
from halo.models.external_tables import named_external_tables
from tests.common.qwen4_exp import (
    build_tiny_qwen4_exp_checkpoint,
    load_host_table_model,
    packed_markers,
    packed_positions,
    transformers_forwards,
)

# Every family-read field away from TINY_QWEN4_EXP_CONFIG's value, each still valid for the config's own checks.
OVERRIDES = {
    "hc_count": 3,
    "hc_lowrank": 8,
    "indexer_n_heads": 3,
    "indexer_head_dim": 32,
    "indexer_budget": 6,
    "indexer_compress_ratio": 2,
    "ple_layer_ids": [1, 3],
    "ple_embed_dim": 54,
    "ple_conv_kernel_size": 3,
    "ngram_size": 4,
    "heads_per_ngram": 3,
    "ngram_vocab_size_base": 509,
    "split_ngram_parts": 2,
    "eos_token_id": 5,
}
TOLERANCE = 1e-6
# Past the 3-block budget and the 4-token n-gram context, with room for a short and a long document.
DOCUMENTS = [7, 70, 31]


@pytest.fixture(scope="module")
def model(tmp_path_factory):
    directory = str(tmp_path_factory.mktemp("tiny") / "qwen4_exp")
    build_tiny_qwen4_exp_checkpoint(directory, overrides=OVERRIDES)
    loaded = load_host_table_model(directory)
    torch.manual_seed(0)
    with torch.no_grad():
        for name, parameter in loaded.named_parameters():
            if "norm" in name:
                parameter.normal_(std=0.3)
    return loaded.eval()


def _ids(length: int, batch: int = 1) -> torch.Tensor:
    ids = torch.randint(6, 512, (batch, length), generator=torch.Generator().manual_seed(1))
    ids[:, length // 3] = OVERRIDES["eos_token_id"]
    return ids


def _logits(model, ids: torch.Tensor, **kwargs) -> torch.Tensor:
    with torch.no_grad():
        return model(input_ids=ids, use_cache=False, **kwargs).logits


def test_the_built_model_carries_the_overridden_fields(model):
    text = model.config
    assert [layer.ple is not None for layer in model.model.layers] == [True, False, True, False]
    assert model.model.layers[0].attn_hyper_connection.hc_count == OVERRIDES["hc_count"]
    tables = [table for _, table in named_external_tables(model)]
    assert [table.part_count for table in tables] == [OVERRIDES["split_ngram_parts"]] * 2
    embedding = model.model.layers[0].ple.ple_embedding
    assert embedding.eos_token_id == OVERRIDES["eos_token_id"]
    assert embedding.ngram_heads == (text.ngram_size - 1) * text.heads_per_ngram


def test_one_document_rows_compute_transformers_logits(model):
    ids = _ids(110, batch=2)
    patched = _logits(model, ids)
    with transformers_forwards():
        stock = _logits(model, ids)
    assert (patched - stock).abs().max() <= TOLERANCE


def test_padded_rows_compute_transformers_logits(model):
    ids = _ids(110, batch=2)
    mask = torch.ones_like(ids)
    mask[0, 90:] = 0
    mask[1, :15] = 0
    patched = _logits(model, ids, attention_mask=mask)
    with transformers_forwards():
        stock = _logits(model, ids, attention_mask=mask)
    assert (patched - stock)[mask.bool()].abs().max() <= TOLERANCE


def test_packed_documents_compute_their_logits_alone(model):
    ids = _ids(sum(DOCUMENTS))
    positions = packed_positions([DOCUMENTS])
    packed = _logits(model, ids, position_ids=positions, **packed_markers(model.config, positions))
    bounds = torch.tensor([0, *DOCUMENTS]).cumsum(0).tolist()
    for lo, hi in zip(bounds[:-1], bounds[1:], strict=True):
        assert (packed[:, lo:hi] - _logits(model, ids[:, lo:hi])).abs().max() <= TOLERANCE, (lo, hi)


def test_the_selection_follows_the_indexer_config(model, monkeypatch):
    selections = []
    original = qsa.select_qsa_keys
    monkeypatch.setattr(qsa, "select_qsa_keys", lambda *a, **k: selections.append(original(*a, **k)) or selections[-1])
    _logits(model, _ids(110))
    (selection,) = selections
    assert selection.compress_ratio == OVERRIDES["indexer_compress_ratio"]
    assert selection.blocks.shape[-1] == OVERRIDES["indexer_budget"] // OVERRIDES["indexer_compress_ratio"]
    assert bool((selection.blocks >= 0).any()), "no query selected a block, so the row never went sparse"


def test_a_multi_head_indexer_key_is_refused(model):
    indexer = next(m for m in model.modules() if isinstance(m, type(model.model.layers[3].self_attn.indexer)))
    original = indexer.index_kv_heads
    indexer.index_kv_heads = 2
    try:
        with pytest.raises(ValueError, match="indexer_kv_heads=1"):
            _logits(model, _ids(40))
    finally:
        indexer.index_kv_heads = original


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
