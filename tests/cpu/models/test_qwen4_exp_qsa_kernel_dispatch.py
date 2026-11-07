#!/usr/bin/env python
"""What the block-sparse QSA kernel reads and when it runs: its key set from ``(blocks, spans)`` against the tiled path
and transformers, the fp64 gather reference against the tiled path, the indexer's chunking and the ``HALO_QSA_KERNEL``
dispatch. The kernel itself runs in ``tests/gpu/kernels/test_block_sparse_attention.py``."""

import pytest
import torch
import transformers.models.qwen4_exp.modeling_qwen4_exp as qwen4_exp_modeling

from halo.kernels import block_sparse_attention as kernel
from halo.models.attention.selection import FLASH_ATTENTION_DETERMINISTIC_ENV
from halo.models.families.qwen4_exp import sparse_attention
from halo.models.families.qwen4_exp.sparse_attention import (
    QUERY_TILE,
    qsa_attention,
    qsa_attention_forward,
    qsa_kernel_mode,
    resolve_qsa_kernel,
    select_qsa_keys,
    tiled_qsa_attention,
)
from tests.common.qwen4_exp import QSA_LAYER_IDX, selection_keys, tiny_qwen4_exp_text_config
from tests.common.qwen4_exp_qsa import (
    ROWS,
    SHORT_DOCUMENT_ROWS,
    Rows,
    attention_operands,
    build_inputs,
    error_ratio,
    gather_reference,
    key_set_from_spans,
    layout_config,
)

SEED = 77
FP32_RATIO = 2e-5
# A group of one, the tiny config's two, and one past what a head count of 12 over one K/V head serves.
GROUPED_LAYOUTS = {"group-1": (2, 2), "group-2": (4, 2), "group-12": (12, 1)}
# (compress ratio, token budget): sparse at 2 blocks, sparse at 8 blocks, and dense, every block fitting.
REGIMES = {"sparse-2": (4, 8), "sparse-8": (2, 16), "dense": (4, 4096)}
BF16_CPU = (torch.bfloat16, torch.device("cpu"))
ALL_ROWS = {**ROWS, "short-documents": SHORT_DOCUMENT_ROWS}


def _attention(config, seed=SEED):
    torch.manual_seed(seed)
    module = qwen4_exp_modeling.Qwen4ExpTextAttention(config, QSA_LAYER_IDX)
    with torch.no_grad():
        for name, parameter in module.named_parameters():
            if "norm" in name:
                parameter.normal_(std=0.3)
    return module


def _selection(config, rows):
    inputs = build_inputs(config, rows)
    attention = _attention(config)
    selection = select_qsa_keys(
        attention.indexer, inputs.hidden, inputs.position_embeddings, inputs.seq_idx, inputs.padding_mask
    )
    return attention, inputs, selection


def _expected_spans(selection):
    """Per query its document's first key and the keys it sees, by plain loops over the documents."""
    spans = torch.zeros_like(selection.spans)
    for doc in selection.documents:
        for query in range(doc.start, doc.stop):
            spans[doc.row, query, 0] = doc.key_start
            spans[doc.row, query, 1] = min(max(query - doc.key_start + 1, 0), doc.key_stop - doc.key_start)
    return spans


@pytest.mark.parametrize("rows", ALL_ROWS.values(), ids=ALL_ROWS.keys())
def test_spans_hold_each_queries_document_start_and_visible_keys(rows):
    _, _, selection = _selection(tiny_qwen4_exp_text_config(), rows)
    assert torch.equal(selection.spans, _expected_spans(selection))


def test_padding_leaves_the_tiled_paths_visibility():
    """A left-padded query sees no key, and a right-padded one sees its whole document, as the tiled path attends it."""
    _, _, left = _selection(tiny_qwen4_exp_text_config(), ROWS["left-padded"])
    assert (left.spans[0, :80, 1] == 0).all()
    assert left.spans[0, 80, 1] == 1
    _, _, right = _selection(tiny_qwen4_exp_text_config(), ROWS["right-padded"])
    assert (right.spans[0, 220:, 1] == 220).all()


@pytest.mark.parametrize("rows", ALL_ROWS.values(), ids=ALL_ROWS.keys())
@pytest.mark.parametrize("regime", REGIMES.values(), ids=REGIMES.keys())
def test_kernel_key_set_equals_the_tiled_mask_and_transformers(rows, regime):
    ratio, budget = regime
    config = tiny_qwen4_exp_text_config(indexer_compress_ratio=ratio, indexer_budget=budget)
    attention, inputs, selection = _selection(config, rows)
    batch, length = inputs.hidden.shape[:2]
    from_spans = key_set_from_spans(selection.blocks.cpu(), selection.spans.cpu(), ratio)
    assert torch.equal(from_spans, selection_keys(selection, batch, length))
    for doc in selection.documents:
        for lo in range(doc.start, doc.stop, QUERY_TILE):
            hi = min(lo + QUERY_TILE, doc.stop)
            keys = doc.tile_keys(hi)
            if keys:
                visible = doc.visible_keys(lo, hi, torch.device("cpu"))
                tiled = sparse_attention._tile_mask(selection.blocks[doc.row, lo:hi], visible, keys, ratio)
                assert torch.equal(from_spans[doc.row, lo:hi, doc.key_start : doc.key_start + keys], tiled)
    with torch.no_grad():
        dense = attention.indexer(inputs.hidden, inputs.position_embeddings, inputs.dense_mask, None)
    transformers_keys = (dense & inputs.dense_mask)[:, 0]
    # transformers attends nothing for a padded query row it masks entirely; compare the rows it does select.
    attended = inputs.dense_mask[:, 0].any(-1)
    assert torch.equal(from_spans[attended], transformers_keys[attended])


@pytest.mark.parametrize("layout", GROUPED_LAYOUTS.values(), ids=GROUPED_LAYOUTS.keys())
@pytest.mark.parametrize(
    "rows", [ROWS["packed"], ROWS["right-padded"], SHORT_DOCUMENT_ROWS], ids=lambda r: str(r)[:30]
)
def test_gather_reference_matches_the_tiled_path(layout, rows):
    heads, kv_heads = layout
    config = tiny_qwen4_exp_text_config(num_attention_heads=heads, num_key_value_heads=kv_heads)
    _, inputs, selection = _selection(config, rows)
    batch, length = inputs.hidden.shape[:2]
    query, key, value, upstream = attention_operands(config, batch, length, SEED)
    scale = config.head_dim**-0.5
    expected = gather_reference(query, key, value, selection.blocks, selection.spans, selection.compress_ratio, scale)
    reference_inputs = [t.double().requires_grad_() for t in (query, key, value)]
    reference = gather_reference(*reference_inputs, selection.blocks, selection.spans, selection.compress_ratio, scale)
    expected_grads = torch.autograd.grad(reference, reference_inputs, upstream.double().transpose(1, 2))
    tiled_inputs = [t.requires_grad_() for t in (query, key, value)]
    tiled = tiled_qsa_attention(*tiled_inputs, selection, scale).transpose(1, 2)
    tiled_grads = torch.autograd.grad(tiled, tiled_inputs, upstream.transpose(1, 2))
    assert error_ratio(tiled, expected) < FP32_RATIO
    for actual, wanted in zip(tiled_grads, expected_grads, strict=True):
        assert error_ratio(actual, wanted) < FP32_RATIO


def test_the_gather_reference_fails_on_a_wrong_key_set():
    config = tiny_qwen4_exp_text_config()
    _, inputs, selection = _selection(config, ROWS["packed"])
    batch, length = inputs.hidden.shape[:2]
    query, key, value, _ = attention_operands(config, batch, length, SEED)
    scale = config.head_dim**-0.5
    tiled = tiled_qsa_attention(query, key, value, selection, scale).transpose(1, 2)
    shifted = selection.spans.clone()
    shifted[..., 1] = (shifted[..., 1] - 1).clamp(min=0)
    wrong = gather_reference(query, key, value, selection.blocks, shifted, selection.compress_ratio, scale)
    assert error_ratio(tiled, wrong) > 100 * FP32_RATIO


@pytest.mark.parametrize("chunk", [1, 8, 64, 10**6])
def test_selection_does_not_depend_on_the_indexer_chunk(chunk, monkeypatch):
    """Blocks of zero hidden states score exactly 0 for every query, so most rows tie at their k-th block and the
    per-query re-score settles them."""
    config = tiny_qwen4_exp_text_config()
    attention = _attention(config)
    hidden = torch.zeros(1, 256, config.hidden_size)
    torch.manual_seed(SEED)
    for block in (3, 9, 30, 41):
        hidden[0, 4 * block : 4 * block + 4] = torch.randn(4, config.hidden_size)
    inputs = build_inputs(config, Rows(((96, 160),)), hidden=hidden)

    def chosen():
        return select_qsa_keys(attention.indexer, inputs.hidden, inputs.position_embeddings, inputs.seq_idx).blocks

    baseline = chosen()
    monkeypatch.setattr(sparse_attention, "_indexer_chunk", lambda blocks, index_heads: chunk)
    assert torch.equal(chosen(), baseline)


def test_indexer_chunk_follows_the_memory_budget(monkeypatch):
    monkeypatch.setenv("HALO_QSA_INDEXER_MIB", "1")
    small = sparse_attention._indexer_chunk(blocks=200, index_heads=4)
    monkeypatch.setenv("HALO_QSA_INDEXER_MIB", "64")
    large = sparse_attention._indexer_chunk(blocks=200, index_heads=4)
    assert small % QUERY_TILE == large % QUERY_TILE == 0
    assert QUERY_TILE <= small < large
    per_query = 4 * 200 * (2 * 4 + 1)
    assert large * per_query <= 64 << 20
    assert sparse_attention._indexer_chunk(blocks=10**7, index_heads=4) == QUERY_TILE, "a tile is the floor"
    assert sparse_attention._indexer_chunk(blocks=200, index_heads=8) < large, "more index heads shrink the chunk"


def test_an_unknown_kernel_mode_raises(monkeypatch):
    monkeypatch.setenv("HALO_QSA_KERNEL", "fast")
    with pytest.raises(ValueError, match="HALO_QSA_KERNEL.*auto"):
        qsa_kernel_mode()


def test_the_kernel_mode_defaults_to_the_documented_constant(monkeypatch):
    monkeypatch.delenv("HALO_QSA_KERNEL", raising=False)
    assert qsa_kernel_mode() == "off"


def _resolve(**overrides):
    arguments = {"dtype": torch.bfloat16, "device": torch.device("cpu"), "head_dim": 64, "heads": 4, "kv_heads": 2}
    return resolve_qsa_kernel(**{**arguments, "ratio": 4, **overrides})


def test_off_takes_the_tiled_path(monkeypatch):
    monkeypatch.setenv("HALO_QSA_KERNEL", "off")
    assert _resolve() is False


def test_auto_falls_back_where_the_kernel_cannot_run(monkeypatch):
    monkeypatch.setenv("HALO_QSA_KERNEL", "auto")
    assert _resolve() is False


def test_on_names_the_remedy_where_the_kernel_cannot_run(monkeypatch):
    monkeypatch.setenv("HALO_QSA_KERNEL", "on")
    with pytest.raises(ValueError, match=r"HALO_QSA_KERNEL=on.*needs a CUDA device.*HALO_QSA_KERNEL=auto"):
        _resolve()


@pytest.mark.parametrize("mode", ["on", "auto"])
def test_full_determinism_refuses_the_kernel(mode, monkeypatch):
    """The dK/dV atomics are not deterministic: ``on`` raises and ``auto`` takes the tiled path."""
    monkeypatch.setenv("HALO_QSA_KERNEL", mode)
    monkeypatch.setenv(FLASH_ATTENTION_DETERMINISTIC_ENV, "1")
    if mode == "auto":
        assert _resolve() is False
    else:
        with pytest.raises(ValueError, match="full_determinism"):
            _resolve()


def test_deterministic_algorithms_refuse_the_kernel(monkeypatch):
    monkeypatch.setenv("HALO_QSA_KERNEL", "on")
    torch.use_deterministic_algorithms(True)
    try:
        with pytest.raises(ValueError, match="full_determinism"):
            _resolve()
    finally:
        torch.use_deterministic_algorithms(False)


@pytest.mark.parametrize(
    ("arguments", "reason"),
    [
        ({"dtype": torch.float32}, "bf16 or fp16"),
        ({"head_dim": 48}, "power-of-two head_dim"),
        ({"head_dim": 8}, "power-of-two head_dim"),
        ({"heads": 5, "kv_heads": 2}, "do not group"),
        ({"heads": 600, "kv_heads": 2}, "accumulator"),
        ({"ratio": 3}, "key tile"),
        ({"ratio": 64}, "key tile"),
        ({}, "CUDA device"),
    ],
)
def test_unsupported_shapes_name_their_reason(arguments, reason):
    base = {"dtype": torch.bfloat16, "device": torch.device("cpu"), "head_dim": 64, "heads": 4, "kv_heads": 2}
    refusal = kernel.block_sparse_unsupported(**{**base, "ratio": 4, **arguments})
    assert refusal is not None
    assert reason in refusal


def test_every_launch_config_keeps_its_ratio_room():
    """A key tile holds whole blocks and the tail, which is under one block."""
    for pair in (*kernel.LAUNCH_CONFIGS.values(), kernel.DEFAULT_LAUNCH_CONFIG):
        for config in pair:
            assert config.keys % 4 == 0 and config.keys > 4
            assert config.keys >= kernel.MIN_DOT_EXTENT


@pytest.mark.parametrize("groups", [1, 2, 3, 12, 16, 17, 128])
def test_group_tile_fits_the_group_in_a_dot_tile(groups):
    tile = kernel.group_tile(groups)
    assert tile >= max(groups, kernel.MIN_DOT_EXTENT)
    assert tile & (tile - 1) == 0
    assert tile < 2 * max(groups, kernel.MIN_DOT_EXTENT)


def test_shapes_come_from_the_config_not_the_release():
    """Heads, K/V heads, head_dim, ratio and top-k all differ from Flash-Next's 24/2/256/4/512."""
    config = layout_config(6, 2, 32, indexer_compress_ratio=2, indexer_budget=16)
    attention = _attention(config)
    inputs = build_inputs(config, ROWS["packed"])
    dense, _ = attention(inputs.hidden, inputs.position_embeddings, inputs.dense_mask)
    tiled, _ = qsa_attention_forward(
        attention, inputs.hidden, inputs.position_embeddings, inputs.padding_mask, seq_idx=inputs.seq_idx
    )
    assert (tiled - dense).abs().max() < 1e-5
    assert kernel.block_sparse_unsupported(torch.bfloat16, torch.device("cpu"), 32, 6, 2, 2) == (
        "the block-sparse kernel needs a CUDA device, got cpu"
    )


def test_the_dispatcher_hands_the_kernel_the_selections_blocks_and_spans(monkeypatch):
    config = tiny_qwen4_exp_text_config()
    _, inputs, selection = _selection(config, ROWS["packed"])
    batch, length = inputs.hidden.shape[:2]
    query, key, value, _ = attention_operands(config, batch, length, SEED)
    seen = []

    def recording_kernel(*arguments):
        seen.append(arguments)
        return torch.zeros_like(arguments[0])

    monkeypatch.setattr(sparse_attention, "resolve_qsa_kernel", lambda *arguments: True)
    monkeypatch.setattr(sparse_attention, "block_sparse_attention", recording_kernel)
    output = qsa_attention(query, key, value, selection, 0.5)
    ((_, _, _, blocks, spans, ratio, scale),) = seen
    assert blocks is selection.blocks and spans is selection.spans
    assert (ratio, scale) == (selection.compress_ratio, 0.5)
    assert output.shape == (batch, length, config.num_attention_heads, config.head_dim)


def test_the_dispatcher_keeps_the_tiled_path_when_the_kernel_is_off(monkeypatch):
    config = tiny_qwen4_exp_text_config()
    _, inputs, selection = _selection(config, ROWS["one-document"])
    query, key, value, _ = attention_operands(config, 1, 300, SEED)
    monkeypatch.setenv("HALO_QSA_KERNEL", "off")

    def unreachable(*arguments):
        raise AssertionError("the kernel ran with HALO_QSA_KERNEL=off")

    monkeypatch.setattr(sparse_attention, "block_sparse_attention", unreachable)
    assert torch.equal(
        qsa_attention(query, key, value, selection, 0.125), tiled_qsa_attention(query, key, value, selection, 0.125)
    )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
