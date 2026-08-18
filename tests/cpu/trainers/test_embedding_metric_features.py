#!/usr/bin/env python
"""Tests that the embedding trainer's metrics-encoding pass survives every collated feature shape.

``EmbeddingTrainer._encode_and_compute_metrics`` re-encodes each text group of the collated batch
under ``no_grad``. Sentence-transformers hands it a padded group for an SDPA backbone but a
packed/varlen group for a flash-attention one (``cu_seq_lens_q`` offsets over a single flattened row,
plus a scalar ``max_length_q`` and a string ``modality``). Slicing every value positionally crashes on
the scalars and desynchronizes the offsets from the tokens.

Run: pytest tests/cpu/trainers/test_embedding_metric_features.py
"""

import pytest
import torch

from src.trainers.embedding.trainer import _METRIC_MAX_SAMPLES, EmbeddingTrainer

_CAP = _METRIC_MAX_SAMPLES


class _RecordingModel:
    """Stands in for the SentenceTransformer forward, recording the features it was handed."""

    def __init__(self, embedding_dim: int = 4):
        self.embedding_dim = embedding_dim
        self.seen: list[dict] = []

    def __call__(self, features: dict) -> dict:
        self.seen.append(features)
        samples = features["cu_seq_lens_q"].numel() - 1 if "cu_seq_lens_q" in features else len(features["input_ids"])
        return {
            "sentence_embedding": torch.arange(samples * self.embedding_dim, dtype=torch.float32).reshape(
                samples, self.embedding_dim
            )
        }


def _trainer() -> EmbeddingTrainer:
    """The trainer's metrics pass reads only class state, so skip the (model-loading) __init__."""
    return EmbeddingTrainer.__new__(EmbeddingTrainer)


def _packed_group(prefix: str, samples: int, tokens: int) -> dict:
    """One flash-attention (packed/varlen) text group as sentence-transformers collates it."""
    return {
        f"{prefix}input_ids": torch.ones(1, tokens, dtype=torch.long),
        f"{prefix}position_ids": torch.arange(tokens).unsqueeze(0),
        f"{prefix}seq_idx": torch.zeros(1, tokens, dtype=torch.long),
        f"{prefix}cu_seq_lens_q": torch.arange(samples + 1, dtype=torch.int32),
        f"{prefix}cu_seq_lens_k": torch.arange(samples + 1, dtype=torch.int32),
        f"{prefix}max_length_q": tokens,
        f"{prefix}max_length_k": tokens,
        f"{prefix}modality": "message",
    }


def _padded_group(prefix: str, samples: int, length: int) -> dict:
    return {
        f"{prefix}input_ids": torch.ones(samples, length, dtype=torch.long),
        f"{prefix}attention_mask": torch.ones(samples, length, dtype=torch.long),
        f"{prefix}modality": "message",
    }


def test_packed_group_is_encoded_verbatim():
    """A packed group has no sliceable sample axis: it must reach the model exactly as collated."""
    inputs = _packed_group("query_", samples=8, tokens=123) | _packed_group("answer_", samples=8, tokens=456)
    model = _RecordingModel()

    metrics = _trainer()._encode_and_compute_metrics(model, inputs)

    assert len(model.seen) == 2
    for features, tokens in zip(model.seen, (123, 456), strict=True):
        assert features["max_length_q"] == tokens
        assert features["modality"] == "message"
        assert features["cu_seq_lens_q"].numel() == 9
        assert features["input_ids"].shape == (1, tokens)
    assert "embed/cos_sim" in metrics


def test_padded_group_over_the_cap_is_sliced_on_the_sample_axis():
    inputs = _padded_group("anchor_", samples=_CAP + 16, length=7)
    model = _RecordingModel()

    _trainer()._encode_and_compute_metrics(model, inputs)

    features = model.seen[0]
    assert features["input_ids"].shape == (_CAP, 7)
    assert features["attention_mask"].shape == (_CAP, 7)
    assert features["modality"] == "message"  # a string must not be sliced into "mess…"


def test_padded_group_under_the_cap_is_untouched():
    inputs = _padded_group("anchor_", samples=4, length=7)
    model = _RecordingModel()

    _trainer()._encode_and_compute_metrics(model, inputs)

    assert model.seen[0]["input_ids"].shape == (4, 7)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
