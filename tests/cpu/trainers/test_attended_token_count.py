#!/usr/bin/env python
"""The token count behind ``num_input_tokens_seen`` (and so tokens/s) holds the batch's real tokens, padding out:
SMPO and embedding batches, and pipeline-shaped batches the collator padded to the frozen length."""

import types
from unittest.mock import MagicMock

import pytest
import torch

from halo.data.collators.fixed_shape import FixedShapeCollator
from halo.data.collators.packing import DataCollatorWithPacking
from halo.data.collators.smpo import DataCollatorForSMPO
from halo.distributed.pipeline_parallel.runner import resolve_batch_pad_values
from halo.trainers.token_metrics import TokenMetricsMixin, count_attended_tokens

PAD = 0
MAX_LENGTH = 32


def _tokenizer() -> MagicMock:
    tokenizer = MagicMock()
    tokenizer.pad_token_id = PAD
    tokenizer.padding_side = "right"

    def pad(encoded, **kwargs):
        width = max(len(row["input_ids"]) for row in encoded)
        return {
            key: torch.tensor([row[key] + [fill] * (width - len(row[key])) for row in encoded])
            for key, fill in (("input_ids", PAD), ("attention_mask", 0))
            if key in encoded[0]
        }

    tokenizer.pad = pad
    return tokenizer


def _smpo_examples(lengths):
    return [
        {
            "prompt_input_ids": list(range(1, p + 1)),
            "chosen_input_ids": list(range(1, c + 1)),
            "rejected_input_ids": list(range(1, r + 1)),
        }
        for p, c, r in lengths
    ]


def test_smpo_batch_counts_each_groups_real_tokens_not_its_padded_shape():
    lengths = [(3, 5, 2), (6, 1, 4), (2, 2, 7)]
    batch = DataCollatorForSMPO(pad_token_id=PAD)(_smpo_examples(lengths))

    padded = sum(batch[f"{group}_input_ids"].numel() for group in ("prompt", "chosen", "rejected"))
    real = sum(map(sum, lengths))
    assert padded > real
    assert int(count_attended_tokens(batch)) == real


def test_embedding_batch_counts_every_text_group_by_its_own_mask():
    anchor_mask = torch.tensor([[1, 1, 1, 0], [1, 1, 0, 0]])
    positive_mask = torch.tensor([[1, 0, 0, 0], [1, 1, 1, 1]])
    batch = {
        "anchor_input_ids": torch.ones(2, 4, dtype=torch.long),
        "anchor_attention_mask": anchor_mask,
        "positive_input_ids": torch.ones(2, 4, dtype=torch.long),
        "positive_attention_mask": positive_mask,
    }
    assert int(count_attended_tokens(batch)) == 5 + 5


def test_packed_embedding_group_counts_to_its_last_offset():
    batch = {"anchor_input_ids": torch.ones(1, 9, dtype=torch.long), "anchor_cu_seq_lens_q": torch.tensor([0, 4, 9])}
    assert int(count_attended_tokens(batch)) == 9
    padded = {**batch, "anchor_input_ids": torch.ones(1, 12, dtype=torch.long)}
    assert int(count_attended_tokens(padded)) == 9


def test_pp_batch_padded_to_the_frozen_length_counts_its_attention_mask():
    inner = lambda features: {  # noqa: E731
        "input_ids": torch.ones(2, 6, dtype=torch.long),
        "labels": torch.ones(2, 6, dtype=torch.long),
        "attention_mask": torch.tensor([[1] * 6, [1] * 2 + [0] * 4]),
    }
    batch = FixedShapeCollator(inner, MAX_LENGTH, resolve_batch_pad_values(PAD))([{}, {}])
    assert batch["input_ids"].numel() == 2 * MAX_LENGTH
    assert int(count_attended_tokens(batch)) == 8


def test_pp_packed_batch_counts_the_packed_rows_real_lengths():
    inner = DataCollatorWithPacking(tokenizer=_tokenizer())
    inner.flatten_to_single_row = False
    rows = [
        {"input_ids": list(range(1, 8)), "seq_lengths": [3, 4]},
        {"input_ids": list(range(1, 6)), "seq_lengths": [2, 3]},
    ]
    batch = FixedShapeCollator(inner, MAX_LENGTH, resolve_batch_pad_values(PAD))(rows)
    assert batch["input_ids"].numel() == 2 * MAX_LENGTH and "attention_mask" not in batch
    assert int(count_attended_tokens(batch)) == 7 + 5


def test_flattened_packed_row_counts_all_its_tokens():
    inner = DataCollatorWithPacking(tokenizer=_tokenizer())
    batch = inner([{"input_ids": list(range(1, 8)), "seq_lengths": [3, 4]}, {"input_ids": [1, 2], "seq_lengths": [2]}])
    assert batch["input_ids"].shape == (1, 9)
    assert int(count_attended_tokens(batch)) == 9


def test_a_batch_with_no_token_group_counts_zero():
    assert count_attended_tokens({"prompt": ["a", "b"]}) == 0


def test_all_mode_accumulates_attended_tokens_on_the_device():
    trainer = types.SimpleNamespace(
        args=types.SimpleNamespace(include_num_input_tokens_seen="all"), _unsummed_input_tokens=0
    )
    mask = torch.tensor([[1] * 6, [1] * 2 + [0] * 4])
    batch = {"input_ids": torch.ones(2, 6, dtype=torch.long), "attention_mask": mask}
    for _ in range(2):
        TokenMetricsMixin._track_num_input_tokens(trainer, batch)
    assert isinstance(trainer._unsummed_input_tokens, torch.Tensor)
    assert int(trainer._unsummed_input_tokens) == 16


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
