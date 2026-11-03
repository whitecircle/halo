#!/usr/bin/env python3
"""The classification collator pads only model inputs: ``remove_unused_columns`` defaults to False, so raw ``text``
and string-label columns reach it, and ``tokenizer.pad`` cannot tensorize a string. Requires the Qwen3 tokenizer
in the HF cache."""

import pytest
import torch

from halo.data.collators.classification import ClassificationDataCollatorWithPadding
from tests.common.models import QWEN3_0_6B
from tests.common.tokenizers import load_cached_tokenizer


def _features(tokenizer):
    """Two rows as the classification map leaves them: tokenized inputs plus the source columns."""
    rows = []
    for text, label in [("the first sentence", 0), ("a much longer second sentence here", 1)]:
        tokenized = tokenizer(text)
        rows.append(
            {
                "input_ids": tokenized["input_ids"],
                "attention_mask": tokenized["attention_mask"],
                "label": label,
                "text": text,
                "prompt": [{"role": "user", "content": text}],
            }
        )
    return rows


def test_collator_drops_source_columns():
    tokenizer = load_cached_tokenizer(QWEN3_0_6B, trust_remote_code=True)
    collator = ClassificationDataCollatorWithPadding(tokenizer, max_length=64)

    batch = collator(_features(tokenizer))

    assert set(batch) == {"input_ids", "attention_mask", "labels"}, f"unexpected batch keys: {sorted(batch)}"
    assert batch["input_ids"].shape[0] == 2
    assert batch["input_ids"].shape == batch["attention_mask"].shape
    assert torch.equal(batch["labels"], torch.tensor([0, 1]))


def test_multi_label_targets_survive():
    """Multi-label rows carry a float vector in `label`; it must reach the batch as `labels`."""
    tokenizer = load_cached_tokenizer(QWEN3_0_6B, trust_remote_code=True)
    collator = ClassificationDataCollatorWithPadding(tokenizer, max_length=64)

    features = _features(tokenizer)
    for feature, target in zip(features, ([1.0, 0.0], [0.0, 1.0]), strict=True):
        feature["label"] = target

    batch = collator(features)

    assert torch.equal(batch["labels"], torch.tensor([[1.0, 0.0], [0.0, 1.0]]))


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
