#!/usr/bin/env python
"""Gemma4ForSequenceClassification: a tiny model from a multimodal config forwards on CPU and pools pad-aware.
Registration is pinned by ``tests/cpu/models/test_seq_cls_head_registration.py``."""

import pytest
import torch
from transformers import AutoModelForSequenceClassification
from transformers.models.gemma4 import Gemma4Config

import halo.models.upstream.seq_cls_heads  # noqa: F401  (registers the head as an import side effect)


def _tiny_config(num_labels: int = 1) -> Gemma4Config:
    config = Gemma4Config(
        text_config={
            "hidden_size": 32,
            "intermediate_size": 64,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 8,
            "global_head_dim": 8,
            "vocab_size": 256,
            "max_position_embeddings": 512,
            "pad_token_id": 0,
            "sliding_window": 16,
        },
        vision_config=None,
        audio_config=None,
    )
    config.num_labels = num_labels
    return config


def test_forward_pools_pad_aware():
    torch.manual_seed(0)
    model = AutoModelForSequenceClassification.from_config(_tiny_config()).eval()
    # The Generic surface the RM trainers dispatch on.
    assert hasattr(model, "model") and hasattr(model, "score")
    ids = torch.randint(1, 250, (1, 7))
    mask = torch.ones_like(ids)
    with torch.no_grad():
        alone = model(input_ids=ids, attention_mask=mask).logits
        pad = torch.zeros((1, 5), dtype=torch.long)
        padded = model(
            input_ids=torch.cat([ids, pad], dim=1),
            attention_mask=torch.cat([mask, torch.zeros_like(pad)], dim=1),
        ).logits
    assert alone.shape == (1, 1) and torch.isfinite(alone).all()
    # The pooled position must not move with right padding.
    assert torch.allclose(alone, padded, atol=1e-5)


def test_classification_head_num_labels():
    model = AutoModelForSequenceClassification.from_config(_tiny_config(num_labels=3))
    ids = torch.randint(1, 250, (2, 5))
    logits = model(input_ids=ids, attention_mask=torch.ones_like(ids)).logits
    assert logits.shape == (2, 3)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
