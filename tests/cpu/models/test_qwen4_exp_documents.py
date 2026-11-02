#!/usr/bin/env python
"""Document-aware ``qwen4_exp`` PLE (``halo.models.qwen4_exp.documents``) against transformers' own PLE layer.

Stock PLE resets its n-gram history only after eos and never resets its dilated conv, so a packed row leaks each
document into the next. The document-aware pieces must equal the stock layer on a one-document row (eos resets
included), keep every packed document equal to the same document alone, and refuse a packed row whose documents
no ``seq_idx`` marks.

    python tests/cpu/models/test_qwen4_exp_documents.py
"""

import pytest
import torch
import transformers.models.qwen4_exp.modeling_qwen4_exp as qwen4_exp_modeling
from torch import nn

from halo.models.qwen4_exp.documents import (
    document_ngram_ids,
    document_ple_forward,
    document_short_conv,
    reject_unmarked_documents,
    shift_within_documents,
)
from halo.models.segment_markers import document_starts
from tests.common.qwen4_exp import EOS_TOKEN_ID, packed_positions, packed_seq_idx, tiny_qwen4_exp_text_config

SEED = 1234
TOLERANCE = 1e-6
# Lengths of the packed documents: the middle one spans the conv's whole 9-token reach and the n-gram context.
DOCUMENTS = [9, 40, 23]
# A stock leak below this would leave the isolation checks unable to tell a leaking path from an isolated one.
LEAK_FLOOR = 1e-3


@pytest.fixture(scope="module")
def config():
    return tiny_qwen4_exp_text_config()


@pytest.fixture(scope="module")
def ple(config):
    """Layer 2's PLE with random weights everywhere: the stock init zeroes the conv and centres the norms."""
    torch.manual_seed(SEED)
    layer = qwen4_exp_modeling.Qwen4ExpTextPLELayer(config, layer_idx=1, ple_layer_index=0)
    with torch.no_grad():
        for parameter in layer.parameters():
            parameter.normal_(std=0.5)
    return layer.eval()


def _token_ids(length: int, batch: int = 1, generator_seed: int = SEED) -> torch.Tensor:
    """Random ids with eos inside the row, never as a document's last token."""
    generator = torch.Generator().manual_seed(generator_seed)
    ids = torch.randint(8, 512, (batch, length), generator=generator)
    ids[:, 5] = EOS_TOKEN_ID
    ids[:, 30] = EOS_TOKEN_ID
    return ids


def _hidden(config, length: int, batch: int = 1) -> torch.Tensor:
    torch.manual_seed(SEED + 1)
    return torch.randn(batch, length, config.hc_count * config.hidden_size)


def _document_slices() -> list[slice]:
    bounds = torch.tensor([0, *DOCUMENTS]).cumsum(0).tolist()
    return [slice(lo, hi) for lo, hi in zip(bounds[:-1], bounds[1:], strict=True)]


def test_document_starts_mark_each_documents_first_token():
    seq_idx = torch.tensor([[0, 0, 0, 1, 1, 2], [4, 4, 4, 4, 4, 4]], dtype=torch.int32)
    assert document_starts(seq_idx, 2, 6, seq_idx.device).tolist() == [[0, 0, 0, 3, 3, 5], [0] * 6]
    assert document_starts(None, 2, 6, seq_idx.device).tolist() == [[0] * 6] * 2
    with pytest.raises(ValueError, match="integer"):
        document_starts(seq_idx.float(), 2, 6, seq_idx.device)


def test_shift_matches_stock_on_one_document_rows(ple, config):
    ids = _token_ids(64, batch=2)
    starts = torch.zeros_like(ids)
    for shift in range(config.ngram_size):
        expected = ple.ple_embedding._shift_right_ignore_eos(ids, shift)
        assert torch.equal(shift_within_documents(ids, shift, EOS_TOKEN_ID, starts), expected), shift


def test_ngram_history_restarts_at_every_document(ple):
    ids = _token_ids(sum(DOCUMENTS))
    starts = document_starts(packed_seq_idx(packed_positions([DOCUMENTS])), 1, ids.shape[1], ids.device)
    packed = document_ngram_ids(ple.ple_embedding, ids, starts)
    leaked = document_ngram_ids(ple.ple_embedding, ids, torch.zeros_like(ids))
    for document in _document_slices():
        alone = document_ngram_ids(ple.ple_embedding, ids[:, document], torch.zeros_like(ids[:, document]))
        assert torch.equal(packed[:, document], alone), document
    assert not torch.equal(packed, leaked), "the row's documents never start without an eos before them"


def test_ple_matches_stock_on_one_document_rows(ple, config):
    ids = _token_ids(64, batch=2)
    hidden = _hidden(config, 64, batch=2)
    with torch.no_grad():
        expected = ple(hidden, ids, None)
        assert (document_ple_forward(ple, hidden, ids, None) - expected).abs().max() <= TOLERANCE
        one_document = torch.zeros_like(ids, dtype=torch.int32)
        assert (document_ple_forward(ple, hidden, ids, one_document) - expected).abs().max() <= TOLERANCE


def test_ple_matches_stock_on_padded_rows(ple, config):
    """The model hands PLE eos at padding and a 2-D conv mask; the padded tail must not move the real tokens."""
    ids = _token_ids(64, batch=2)
    hidden = _hidden(config, 64, batch=2)
    conv_mask = torch.ones_like(ids)
    conv_mask[0, 50:] = 0
    ids = torch.where(conv_mask.bool(), ids, EOS_TOKEN_ID)
    with torch.no_grad():
        expected = ple(hidden, ids, None, conv_mask=conv_mask)
        assert (document_ple_forward(ple, hidden, ids, None, conv_mask) - expected).abs().max() <= TOLERANCE


def test_conv_taps_stop_at_every_document_start(ple, config):
    hidden = _hidden(config, sum(DOCUMENTS))
    starts = document_starts(packed_seq_idx(packed_positions([DOCUMENTS])), 1, hidden.shape[1], hidden.device)
    with torch.no_grad():
        packed = document_short_conv(ple.conv1d, hidden, starts)
        leaked = ple._short_conv(hidden, None)
        for document in _document_slices():
            alone = ple._short_conv(hidden[:, document], None)
            assert (packed[:, document] - alone).abs().max() <= TOLERANCE, document
    assert (packed - leaked).abs().max() > LEAK_FLOOR


def test_packed_ple_computes_each_document_as_alone(ple, config):
    ids = _token_ids(sum(DOCUMENTS))
    hidden = _hidden(config, sum(DOCUMENTS))
    seq_idx = packed_seq_idx(packed_positions([DOCUMENTS]))
    with torch.no_grad():
        packed = document_ple_forward(ple, hidden, ids, seq_idx)
        stock = ple(hidden, ids, None)
        for document in _document_slices():
            alone = ple(hidden[:, document], ids[:, document], None)
            assert (packed[:, document] - alone).abs().max() <= TOLERANCE, document
    assert (stock - packed).abs().max() > LEAK_FLOOR, "stock PLE no longer leaks across this row's documents"


def test_ple_gradients_stay_inside_their_document(ple, config):
    ids = _token_ids(sum(DOCUMENTS))
    hidden = _hidden(config, sum(DOCUMENTS)).requires_grad_()
    seq_idx = packed_seq_idx(packed_positions([DOCUMENTS]))
    last, earlier = _document_slices()[-1], slice(0, _document_slices()[-1].start)
    output = document_ple_forward(ple, hidden, ids, seq_idx)
    (gradient,) = torch.autograd.grad(output[:, last].sum(), hidden)
    assert gradient[:, earlier].abs().max() == 0
    assert gradient[:, last].abs().max() > 0


@pytest.mark.parametrize(
    "conv",
    [
        nn.Conv1d(8, 8, kernel_size=4, groups=8, dilation=3, bias=True),
        nn.Conv1d(8, 8, kernel_size=4, groups=4, dilation=3, bias=False),
        nn.Sequential(nn.Conv1d(8, 8, kernel_size=4, groups=8, dilation=3, bias=False)),
    ],
    ids=["biased", "grouped", "wrapped"],
)
def test_conv_refuses_a_conv_it_cannot_reproduce(conv):
    with pytest.raises(TypeError, match="lora_target_modules"):
        document_short_conv(conv, torch.randn(1, 12, 8), torch.zeros(1, 12, dtype=torch.long))


@pytest.mark.parametrize(
    ("position_ids", "seq_idx", "message"),
    [
        (packed_positions([DOCUMENTS]), None, "without seq_idx"),
        (packed_positions([DOCUMENTS]), packed_seq_idx(packed_positions([[9, 63]])), "different documents"),
        (None, packed_seq_idx(packed_positions([DOCUMENTS])), "different documents"),
        (packed_positions([[72]]), None, None),
        (packed_positions([DOCUMENTS]), packed_seq_idx(packed_positions([DOCUMENTS])), None),
        (None, None, None),
    ],
    ids=["packed-unmarked", "packed-mismarked", "marked-without-positions", "one-document", "packed-marked", "none"],
)
def test_reject_unmarked_documents(position_ids, seq_idx, message):
    if message is None:
        reject_unmarked_documents(position_ids, seq_idx)
        return
    with pytest.raises(ValueError, match=message):
        reject_unmarked_documents(position_ids, seq_idx)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
