#!/usr/bin/env python
"""Tests that the embedding config's pipeline knobs reach a hub-loaded SentenceTransformer.

``pooling_mode`` / ``normalize_embeddings`` / ``max_length`` build the Pooling and Normalize modules
only when the parallel loader's pipeline wraps a bare backbone. Every other pipeline comes from the
checkpoint, on sentence-transformers' own load as on the parallel loader's rebuild, so the same knobs
must be re-applied there or they parse and do nothing — a bare causal LM training mean pooling while
the YAML says ``lasttoken``. These tests fail if that alignment drops.

Run: pytest tests/cpu/config/test_embedding_pipeline_alignment.py
"""

import pytest
from sentence_transformers import SentenceTransformer
from sentence_transformers.models import Normalize, Pooling

from halo.configs.embedding import EmbeddingConfig
from scripts.training.embedding import align_st_pipeline_to_config

_CPU_OK = {"output_dir": "/tmp/test_embedding_alignment", "bf16": False}


def _pipeline(pooling_mode: str, *, normalize: bool) -> SentenceTransformer:
    """A module-only ST pipeline (no weights): enough to exercise the alignment contract."""
    modules = [Pooling(word_embedding_dimension=8, pooling_mode=pooling_mode)]
    if normalize:
        modules.append(Normalize())
    return SentenceTransformer(modules=modules)


def test_pooling_mode_overrides_the_checkpoints_own_pooling():
    model = _pipeline("mean", normalize=True)
    config = EmbeddingConfig(pooling_mode="lasttoken", normalize_embeddings=True, max_length=512, **_CPU_OK)

    align_st_pipeline_to_config(model, config)

    assert [m for m in model if isinstance(m, Pooling)][0].pooling_mode == "lasttoken"


def test_max_length_overrides_the_checkpoints_window():
    model = _pipeline("mean", normalize=True)
    model.max_seq_length = 32768  # what a hub checkpoint ships
    config = EmbeddingConfig(pooling_mode="mean", normalize_embeddings=True, max_length=512, **_CPU_OK)

    align_st_pipeline_to_config(model, config)

    assert model.max_seq_length == 512


def test_normalize_module_is_appended_when_missing():
    model = _pipeline("mean", normalize=False)
    config = EmbeddingConfig(pooling_mode="mean", normalize_embeddings=True, max_length=128, **_CPU_OK)

    align_st_pipeline_to_config(model, config)

    assert any(isinstance(m, Normalize) for m in model)


def test_normalize_false_against_a_normalizing_checkpoint_raises():
    """Silently keeping the Normalize module would train a model whose embeddings are not what the
    config describes; silently dropping it would change the checkpoint's similarity scale."""
    model = _pipeline("mean", normalize=True)
    config = EmbeddingConfig(pooling_mode="mean", normalize_embeddings=False, max_length=128, **_CPU_OK)

    with pytest.raises(ValueError, match="normalize_embeddings=False"):
        align_st_pipeline_to_config(model, config)


def test_pipeline_without_pooling_raises_instead_of_dropping_the_knob():
    model = SentenceTransformer(modules=[Normalize()])
    config = EmbeddingConfig(pooling_mode="lasttoken", normalize_embeddings=True, max_length=128, **_CPU_OK)

    with pytest.raises(ValueError, match="no Pooling module"):
        align_st_pipeline_to_config(model, config)


def test_dataset_num_proc_is_not_a_field():
    """Declared, it would never be read (the ST path runs no dataset map) and a YAML setting it would
    look honored. Off the dataclass, the parser rejects the key instead."""
    assert "dataset_num_proc" not in EmbeddingConfig.__dataclass_fields__


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
