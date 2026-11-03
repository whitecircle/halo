#!/usr/bin/env python
"""A training row's documents as spans, and the upstream guard on a forward written against one release."""

import pytest
import torch

from halo.models.row_documents import RowDocument, row_documents
from halo.models.upstream_guard import verify_upstream_parameters

CPU = torch.device("cpu")


def _spans(seq_idx, mask, batch, length) -> list[tuple[int, int, int, int, int]]:
    return [
        (doc.row, doc.start, doc.stop, doc.key_start, doc.key_stop)
        for doc in row_documents(seq_idx, mask, batch, length, CPU)
    ]


def test_each_row_is_one_document_without_markers():
    assert _spans(None, None, 2, 5) == [(0, 0, 5, 0, 5), (1, 0, 5, 0, 5)]


def test_a_packed_row_splits_at_every_marker_change():
    seq_idx = torch.tensor([[0, 0, 0, 1, 1, 2]])
    assert _spans(seq_idx, None, 1, 6) == [(0, 0, 3, 0, 3), (0, 3, 5, 3, 5), (0, 5, 6, 5, 6)]


@pytest.mark.parametrize(
    ("mask", "expected"),
    [
        ([1, 1, 1, 0, 0], (0, 0, 5, 0, 3)),
        ([0, 0, 1, 1, 1], (0, 0, 5, 2, 5)),
        ([0, 0, 0, 0, 0], (0, 0, 5, 0, 0)),
    ],
    ids=["right_padded", "left_padded", "all_padding"],
)
def test_padding_narrows_the_keys_not_the_span(mask, expected):
    assert _spans(None, torch.tensor([mask]), 1, 5) == [expected]


def test_padding_only_moves_the_document_it_sits_in():
    seq_idx = torch.tensor([[0, 0, 1, 1, 1], [0, 0, 0, 0, 0]])
    mask = torch.tensor([[1, 1, 0, 1, 1], [1, 1, 1, 0, 0]])
    assert _spans(seq_idx, mask, 2, 5) == [(0, 0, 2, 0, 2), (0, 2, 5, 3, 5), (1, 0, 5, 0, 3)]


def test_padding_inside_a_document_is_refused():
    """Its keys would no longer be one slice of the row."""
    with pytest.raises(ValueError, match="contiguous"):
        row_documents(None, torch.tensor([[1, 0, 1]]), 1, 3, CPU)


def test_a_mask_of_another_shape_is_refused():
    with pytest.raises(ValueError, match="2-D"):
        row_documents(None, torch.ones(1, 1, 3, 3), 1, 3, CPU)


def test_keys_count_from_the_document_not_the_row():
    doc = RowDocument(row=0, start=4, stop=10, key_start=5, key_stop=9)
    assert doc.visible_keys(4, 10, CPU).tolist() == [0, 1, 2, 3, 4, 4]
    assert [doc.tile_keys(hi) for hi in (4, 5, 7, 9, 10)] == [0, 0, 2, 4, 4]


def test_the_guard_passes_the_signature_it_was_written_against():
    def forward(self, input_ids, attention_mask=None):
        return input_ids

    verify_upstream_parameters(forward, ("self", "input_ids", "attention_mask"), "patch_forward")


@pytest.mark.parametrize(
    "expected",
    [("self", "input_ids"), ("self", "attention_mask", "input_ids")],
    ids=["a_parameter_was_added", "the_order_moved"],
)
def test_the_guard_refuses_a_signature_that_moved(expected):
    def forward(self, input_ids, attention_mask=None):
        return input_ids

    with pytest.raises(RuntimeError, match="patch_forward.*re-derive"):
        verify_upstream_parameters(forward, expected, "patch_forward")


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
