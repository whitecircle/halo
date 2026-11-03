#!/usr/bin/env python
"""The packing collator warns once per instance when rows carry no ``seq_lengths``: each row then collates as one
document and concatenated documents attend across each other, which wrapped pretraining may intend but never
silently."""

import warnings
from unittest.mock import MagicMock

import pytest
import torch

from halo.data.collators.packing import DataCollatorWithPacking


def make_tokenizer() -> MagicMock:
    tok = MagicMock()
    tok.pad_token_id = 0
    tok.eos_token_id = 1
    tok.padding_side = "right"
    tok.model_max_length = 4096

    def _pad(features, **kwargs):
        max_len = max(len(f["input_ids"]) for f in features)
        out = {"input_ids": [], "attention_mask": []}
        for f in features:
            ids = list(f["input_ids"])
            pad = max_len - len(ids)
            out["input_ids"].append(ids + [tok.pad_token_id] * pad)
            out["attention_mask"].append([1] * len(ids) + [0] * pad)
        return {key: torch.tensor(value) for key, value in out.items()}

    tok.pad.side_effect = _pad
    return tok


def test_rows_without_seq_lengths_warn_once():
    collator = DataCollatorWithPacking(tokenizer=make_tokenizer())
    rows = [{"input_ids": [5, 6, 7], "attention_mask": [1, 1, 1]}]

    with pytest.warns(UserWarning, match="seq_lengths"):
        collator.torch_call(rows)

    # A per-step warning would flood the log.
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        collator.torch_call(rows)

    # A fresh collator warns again; a class-level latch would silence every instance after the first.
    with pytest.warns(UserWarning, match="seq_lengths"):
        DataCollatorWithPacking(tokenizer=make_tokenizer()).torch_call(rows)


def test_packed_rows_do_not_warn():
    collator = DataCollatorWithPacking(tokenizer=make_tokenizer())
    rows = [{"input_ids": [5, 6, 7, 8], "attention_mask": [1, 1, 1, 1], "seq_lengths": [2, 2]}]

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        batch = collator.torch_call(rows)
    assert batch["position_ids"].tolist() == [[0, 1, 0, 1]]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
