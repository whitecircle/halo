#!/usr/bin/env python
"""``TensorMemo``: one derivation for every reader of the same tensor contents, and never one tensor's derivation for
another's.

Every attention layer of a forward reads the same ``position_ids``, and a reentrant checkpoint's recompute reads
detached copies of it, which share its storage and version counter. A later micro-batch's row is a new tensor that
the allocator can place at the freed address of the last one, with the same shape and version: a memo keyed on the
address alone would hand it the earlier row's document boundaries.

    python tests/cpu/models/test_tensor_memo.py
"""

import pytest
import torch

from halo.models.attention.tensor_memo import TensorMemo
from halo.models.segment_markers import packed_row_varlen_kwargs

# Trials of a freed row followed by another row of the same shape: the CPU allocator returns the freed block to
# the next one in nearly every trial, so a memo that kept no reference would answer most of them stale.
TRIALS = 50


def _row(*lengths: int) -> torch.Tensor:
    return torch.cat([torch.arange(length) for length in lengths])[None]


def _counted():
    calls = []

    def derive(tensor):
        calls.append(tensor.shape)
        return packed_row_varlen_kwargs(tensor)

    return TensorMemo(derive), calls


def test_a_row_and_its_detached_copies_derive_once():
    memo, calls = _counted()
    row = _row(3, 4, 5)
    first = memo(row)
    assert memo(row) is first
    assert memo(row.detach()) is first
    assert len(calls) == 1
    assert first["cu_seq_lens_q"].tolist() == [0, 3, 7, 12]


def test_an_in_place_write_derives_again():
    memo, calls = _counted()
    row = _row(3, 4, 5)
    memo(row)
    row[0, 3:7] = torch.arange(4) + 3
    assert memo(row)["cu_seq_lens_q"].tolist() == [0, 7, 12]
    assert len(calls) == 2


def test_rows_with_other_documents_never_read_an_earlier_derivation():
    # No counting derive here: a list of the rows it saw would keep them alive, the memo's job under test.
    memo = TensorMemo(packed_row_varlen_kwargs)
    for _ in range(TRIALS):
        earlier = _row(9, 7, 5)
        memo(earlier)
        del earlier
        later = _row(4, 12, 5)
        assert memo(later)["cu_seq_lens_q"].tolist() == [0, 4, 16, 21]


def test_an_inference_tensor_derives_on_every_call():
    memo, calls = _counted()
    with torch.inference_mode():
        row = _row(3, 4)
    memo(row)
    memo(row)
    assert len(calls) == 2


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
