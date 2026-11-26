#!/usr/bin/env python
"""Tiled QSA against transformers' ``Qwen4ExpTextAttention``, whose indexer selects each query's key blocks in a Python
loop over a dense ``[B, 1, T, T]`` mask."""

import functools

import pytest
import torch
import transformers.models.qwen4_exp.modeling_qwen4_exp as qwen4_exp_modeling
from torch.utils.checkpoint import checkpoint
from transformers.integrations.sdpa_attention import sdpa_attention_forward
from transformers.masking_utils import create_causal_mask

from halo.models.families.qwen4_exp import sparse_attention
from halo.models.families.qwen4_exp.sparse_attention import (
    QUERY_TILE,
    qsa_attention_forward,
    select_qsa_keys,
    tiled_qsa_attention,
)
from halo.models.gradient_checkpointing import active_recompute_scope, scoped_checkpoint_func
from tests.common.qwen4_exp import QSA_LAYER_IDX, LargestTensor, selection_keys, tiny_qwen4_exp_text_config
from tests.common.qwen4_exp_qsa import ROWS, Inputs, Rows, build_inputs

SEED = 1234
TOLERANCE = 1e-6
# Wide enough that the tiled path's largest tensor (a projection of the row) stays well under T x T.
ALLOCATION_ROW = 2048


@pytest.fixture(scope="module")
def config():
    return tiny_qwen4_exp_text_config()


@pytest.fixture(scope="module")
def attention(config):
    """The QSA layer with random norm weights: the stock zero init (``1 + weight``) would hide a skipped norm."""
    torch.manual_seed(SEED)
    module = qwen4_exp_modeling.Qwen4ExpTextAttention(config, QSA_LAYER_IDX)
    with torch.no_grad():
        for name, parameter in module.named_parameters():
            if "norm" in name:
                parameter.normal_(std=0.3)
    return module


def _transformers_keys(attention, inputs: Inputs) -> torch.Tensor:
    """``[batch, length, length]``: the keys transformers' indexer and mask let each query attend."""
    with torch.no_grad():
        selected = attention.indexer(inputs.hidden, inputs.position_embeddings, inputs.dense_mask, None)
    return (selected & inputs.dense_mask)[:, 0]


def _assert_same_keys(expected: torch.Tensor, actual: torch.Tensor) -> None:
    differing = (expected != actual).any(dim=-1).nonzero().tolist()
    assert not differing, f"{len(differing)} queries attend other keys than transformers', first {differing[:5]}"


@pytest.mark.parametrize("rows", ROWS.values(), ids=ROWS.keys())
def test_selection_matches_transformers_per_query(attention, config, rows):
    inputs = build_inputs(config, rows)
    batch, length = inputs.hidden.shape[:2]
    selection = select_qsa_keys(
        attention.indexer, inputs.hidden, inputs.position_embeddings, inputs.seq_idx, inputs.padding_mask
    )
    assert (selection.blocks >= 0).sum(-1).max() == attention.indexer.block_topk, "no query crossed the budget"
    _assert_same_keys(_transformers_keys(attention, inputs), selection_keys(selection, batch, length))


def test_tie_rows_resolve_in_transformers_order(attention, config, monkeypatch):
    """Blocks of zero hidden states score exactly 0 for every query, and a zero query scores every block 0, so most
    rows tie at their k-th block. transformers' ``topk`` order on its one-query row decides them."""
    rows = Rows(((96, 160),))
    torch.manual_seed(SEED + 2)
    hidden = torch.zeros(1, 256, config.hidden_size)
    for block in (3, 9, 30, 41):
        hidden[0, 4 * block : 4 * block + 4] = torch.randn(4, config.hidden_size)
    inputs = build_inputs(config, rows, hidden=hidden)
    rescored = []
    rescore = sparse_attention._rescore_like_transformers

    def counting_rescore(*args):
        rescored.append(args[2])
        return rescore(*args)

    monkeypatch.setattr(sparse_attention, "_rescore_like_transformers", counting_rescore)
    selection = select_qsa_keys(attention.indexer, inputs.hidden, inputs.position_embeddings, inputs.seq_idx)
    assert len(rescored) > 100, f"only {len(rescored)} tied rows: the inputs no longer produce ties"
    _assert_same_keys(_transformers_keys(attention, inputs), selection_keys(selection, 1, 256))


def _max_difference(expected, actual) -> float:
    return max((e - a).abs().max().item() for e, a in zip(expected, actual, strict=True))


def _max_relative_difference(expected, actual) -> float:
    """Over each tensor's own scale: a parameter gradient sums the row's tokens, so its rounding grows with it."""
    return max(
        ((e - a).abs().max() / e.abs().max().clamp_min(1.0)).item() for e, a in zip(expected, actual, strict=True)
    )


@pytest.mark.parametrize("rows", ROWS.values(), ids=ROWS.keys())
def test_attention_core_matches_sdpa_under_the_dense_mask(attention, config, rows):
    inputs = build_inputs(config, rows)
    batch, length = inputs.hidden.shape[:2]
    selection = select_qsa_keys(
        attention.indexer, inputs.hidden, inputs.position_embeddings, inputs.seq_idx, inputs.padding_mask
    )
    mask = _transformers_keys(attention, inputs).unsqueeze(1)
    torch.manual_seed(SEED + 3)
    shapes = [
        (batch, config.num_attention_heads, length, config.head_dim),
        (batch, config.num_key_value_heads, length, config.head_dim),
        (batch, config.num_key_value_heads, length, config.head_dim),
    ]
    query, key, value = (torch.randn(shape) for shape in shapes)
    upstream = torch.randn(batch, length, config.num_attention_heads, config.head_dim)

    dense_inputs = [t.clone().requires_grad_() for t in (query, key, value)]
    dense, _ = sdpa_attention_forward(attention, *dense_inputs, mask, scaling=attention.scaling)
    dense_grads = torch.autograd.grad(dense, dense_inputs, upstream)
    tiled_inputs = [t.clone().requires_grad_() for t in (query, key, value)]
    tiled = tiled_qsa_attention(*tiled_inputs, selection, attention.scaling)
    tiled_grads = torch.autograd.grad(tiled, tiled_inputs, upstream)

    assert (tiled - dense).abs().max() <= TOLERANCE
    assert _max_difference(dense_grads, tiled_grads) <= TOLERANCE


@pytest.mark.parametrize("rows", ROWS.values(), ids=ROWS.keys())
def test_attention_forward_matches_transformers(attention, config, rows):
    """The whole layer, gradients included; the indexer gets none in either."""
    inputs = build_inputs(config, rows)
    parameters = list(attention.parameters())
    torch.manual_seed(SEED + 4)
    upstream = torch.randn_like(inputs.hidden)

    dense_hidden = inputs.hidden.clone().requires_grad_()
    dense, _ = attention(dense_hidden, inputs.position_embeddings, inputs.dense_mask)
    dense_grads = torch.autograd.grad(dense, [dense_hidden, *parameters], upstream, allow_unused=True)
    tiled_hidden = inputs.hidden.clone().requires_grad_()
    tiled, _ = qsa_attention_forward(
        attention, tiled_hidden, inputs.position_embeddings, inputs.padding_mask, seq_idx=inputs.seq_idx
    )
    tiled_grads = torch.autograd.grad(tiled, [tiled_hidden, *parameters], upstream, allow_unused=True)

    assert (tiled - dense).abs().max() <= TOLERANCE
    assert [g is None for g in tiled_grads] == [g is None for g in dense_grads]
    dense_grads, tiled_grads = ([g for g in grads if g is not None] for grads in (dense_grads, tiled_grads))
    assert _max_relative_difference(dense_grads, tiled_grads) <= TOLERANCE


def test_each_tile_reads_its_document_keys_up_to_the_tile_end(attention, config, monkeypatch):
    rows = ROWS["packed"]
    inputs = build_inputs(config, rows)
    tiles = []
    attend = sparse_attention._attend_tile

    def recording_attend(query, key, value, mask, scale):
        tiles.append((query.shape[1], key.shape[1]))
        return attend(query, key, value, mask, scale)

    monkeypatch.setattr(sparse_attention, "_attend_tile", recording_attend)
    with torch.no_grad():
        qsa_attention_forward(attention, inputs.hidden, inputs.position_embeddings, seq_idx=inputs.seq_idx)
    expected = []
    for length in rows.documents[0]:
        for lo in range(0, length, QUERY_TILE):
            hi = min(lo + QUERY_TILE, length)
            expected.append((hi - lo, hi))
    assert tiles == expected


def test_no_tensor_grows_with_the_square_of_the_row(attention, config):
    inputs = build_inputs(config, Rows(((ALLOCATION_ROW,),)))
    hidden = inputs.hidden.clone().requires_grad_()
    with LargestTensor() as tiled:
        output, _ = qsa_attention_forward(attention, hidden, inputs.position_embeddings, seq_idx=inputs.seq_idx)
        output.sum().backward()
    assert tiled.numel < ALLOCATION_ROW * ALLOCATION_ROW, f"tiled QSA allocated {tiled.shape}"

    # The detector sees transformers' own [B, 1, T, T] masks.
    with torch.no_grad(), LargestTensor() as dense:
        mask = create_causal_mask(
            config=config,
            inputs_embeds=inputs.hidden,
            attention_mask=None,
            past_key_values=None,
            position_ids=None,
            allow_is_causal_skip=False,
        )
        attention(inputs.hidden, inputs.position_embeddings, mask)
    assert dense.numel >= ALLOCATION_ROW * ALLOCATION_ROW


def test_recompute_attends_the_forward_selection(attention, config, monkeypatch):
    """A recompute whose input drifted (an inexact replay) still attends the keys the forward selected."""
    inputs = build_inputs(config, ROWS["packed"])
    torch.manual_seed(SEED + 5)
    drift = 3.0 * torch.randn_like(inputs.hidden)
    attended = []
    attend = sparse_attention.tiled_qsa_attention

    def recording_attention(query, key, value, selection, scale):
        attended.append(selection.blocks.clone())
        return attend(query, key, value, selection, scale)

    monkeypatch.setattr(sparse_attention, "tiled_qsa_attention", recording_attention)

    def layer(hidden):
        if active_recompute_scope().is_recompute:
            hidden = hidden + drift
        return qsa_attention_forward(attention, hidden, inputs.position_embeddings, seq_idx=inputs.seq_idx)[0]

    scoped = scoped_checkpoint_func(functools.partial(checkpoint, use_reentrant=True))
    scoped(layer, inputs.hidden.clone().requires_grad_()).sum().backward()

    drifted = select_qsa_keys(
        attention.indexer, inputs.hidden + drift, inputs.position_embeddings, inputs.seq_idx
    ).blocks
    assert len(attended) == 2
    assert not torch.equal(drifted, attended[0]), "the drift no longer moves the selection"
    assert torch.equal(attended[1], attended[0])


def test_a_no_grad_pass_through_a_checkpoint_keeps_no_selection(attention, config):
    inputs = build_inputs(config, ROWS["one-document"])
    scoped = scoped_checkpoint_func(functools.partial(checkpoint, use_reentrant=True))
    kept = []

    def layer(hidden):
        kept.append(active_recompute_scope())
        return qsa_attention_forward(attention, hidden, inputs.position_embeddings, seq_idx=inputs.seq_idx)[0]

    with torch.no_grad():
        scoped(layer, inputs.hidden)
    assert kept[0]._slots == {}


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ("dense-mask", "2-D mask"),
        ("cache", "past_key_values"),
        ("interleaved-padding", "contiguous"),
        ("seq-idx-shape", "integer"),
        ("dropout", "dropout"),
    ],
)
def test_refusals(attention, config, change, message):
    inputs = build_inputs(config, ROWS["one-document"])
    kwargs = {"attention_mask": None, "seq_idx": inputs.seq_idx}
    module = attention
    if change == "dense-mask":
        kwargs["attention_mask"] = inputs.dense_mask
    elif change == "cache":
        kwargs["past_key_values"] = object()
    elif change == "interleaved-padding":
        padding = torch.ones(1, 300, dtype=torch.long)
        padding[0, 100:120] = 0
        kwargs["attention_mask"] = padding
    elif change == "seq-idx-shape":
        kwargs["seq_idx"] = inputs.seq_idx[:, :-1]
    else:
        module = qwen4_exp_modeling.Qwen4ExpTextAttention(tiny_qwen4_exp_text_config(attention_dropout=0.1), 3)
    with pytest.raises(ValueError, match=message):
        qsa_attention_forward(module, inputs.hidden, inputs.position_embeddings, **kwargs)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
