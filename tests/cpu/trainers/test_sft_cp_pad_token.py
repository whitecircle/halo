#!/usr/bin/env python
"""CP's divisibility padding must fail loud without a pad token instead of inventing id 0.

``DistributedSFTTrainer.compute_loss`` pads the batch up to a multiple of ``cp_size`` before the
Ulysses split. A pad id falling back to ``0`` whenever the tokenizer has none pads every sequence
with whatever real vocabulary token happens to sit at index 0 — indistinguishable, in the logs and
in the loss, from a correctly padded run.

Run: python tests/cpu/trainers/test_sft_cp_pad_token.py
"""

from types import SimpleNamespace
from unittest import mock

import pytest
import torch
from transformers import Trainer

from src.data.spans import LABEL_IGNORE_INDEX
from src.trainers.sft import DistributedSFTTrainer


def _inputs(seq_len):
    ids = torch.arange(1, seq_len + 1).unsqueeze(0)
    return {
        "input_ids": ids.clone(),
        "labels": ids.clone(),
        "attention_mask": torch.ones_like(ids),
    }


def _trainer(processing_class, cp_size=2):
    """Fake ``self`` carrying only what the CP branch of ``compute_loss`` reads."""
    return SimpleNamespace(
        _validate_inputs=lambda inputs: None,
        is_cp_mode=True,
        cp_size=cp_size,
        cp_config=SimpleNamespace(cp_rank=0),
        processing_class=processing_class,
        _compute_cp_metrics=lambda *args, **kwargs: None,
    )


def _run(processing_class, seq_len, cp_size=2):
    """Drive the real ``compute_loss`` CP branch; returns the inputs the base Trainer received."""
    captured = {}

    def recorder(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        captured.update(inputs)
        return torch.zeros(()), SimpleNamespace(logits=None)

    with mock.patch.object(Trainer, "compute_loss", recorder):
        DistributedSFTTrainer.compute_loss(_trainer(processing_class, cp_size), None, _inputs(seq_len))
    return captured


def test_missing_pad_token_raises_instead_of_padding_with_token_zero():
    tokenizer = SimpleNamespace(pad_token_id=None)
    with pytest.raises(ValueError, match="pad_token_id"):
        _run(tokenizer, seq_len=3)


def test_a_processor_without_a_pad_token_raises_too():
    """The VLM path hands the trainer a processor; the gate must read through to its tokenizer."""
    processor = SimpleNamespace(tokenizer=SimpleNamespace(pad_token_id=None))
    with pytest.raises(ValueError, match="pad_token_id"):
        _run(processor, seq_len=3)


def test_a_real_pad_token_still_pads_the_batch():
    """Anti-vacuity: with a pad id the padding contract is unchanged — pad id on input_ids, ignore
    index on labels, zeros on the attention mask, all widened to a multiple of cp_size."""
    captured = _run(SimpleNamespace(pad_token_id=7), seq_len=3)
    assert captured["input_ids"].tolist() == [[1, 2, 3, 7]]
    assert captured["labels"].tolist() == [[1, 2, 3, LABEL_IGNORE_INDEX]]
    assert captured["attention_mask"].tolist() == [[1, 1, 1, 0]]


def test_pad_token_id_zero_is_a_valid_pad_token():
    """``or 0`` collapsed 'no pad token' and 'pad token is id 0' into one branch; id 0 is a real
    tokenizer setting (Llama-family pad) and must go through, not trip the new raise."""
    captured = _run(SimpleNamespace(pad_token_id=0), seq_len=3)
    assert captured["input_ids"].tolist() == [[1, 2, 3, 0]]


def test_a_divisible_batch_never_consults_the_tokenizer():
    """The raise belongs to the padding branch only: a run whose sequences already divide evenly
    must not be refused for a tokenizer setting it never uses."""

    class _Exploding:
        @property
        def pad_token_id(self):
            raise AssertionError("pad_token_id read on a batch that needs no padding")

    captured = _run(_Exploding(), seq_len=4)
    assert captured["input_ids"].tolist() == [[1, 2, 3, 4]]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
