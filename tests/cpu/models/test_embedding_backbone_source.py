#!/usr/bin/env python
"""The default embedding path's backbone loads at the checkpoint source the ranks agreed on.

sentence-transformers loads that backbone itself, outside ``load_distributed_model``, so
``build_sentence_transformer`` runs ``resolve_model_source`` on its own and must hand the agreed
commit on — not the configured revision, which each node's Hub cache would resolve for itself.

    python tests/cpu/models/test_embedding_backbone_source.py
"""

from types import SimpleNamespace

import pytest
from trl import ModelConfig

import scripts.training.embedding as embedding_script
from src.configs.embedding_config import EmbeddingConfig

REPO = "org/model"
COMMIT_A = "a" * 40


class _BackboneBuilt(Exception):
    """Stops the embedding build once sentence-transformers was asked for the backbone."""


def test_the_embedding_backbone_loads_at_the_agreed_revision(monkeypatch, tmp_path):
    seen = {}

    def resolve(model_name_or_path, revision, *, tag):
        seen["resolve"] = (model_name_or_path, revision, tag)
        return COMMIT_A

    def build(model_name_or_path, *, revision, **kwargs):
        seen["load"] = (model_name_or_path, revision)
        raise _BackboneBuilt

    monkeypatch.setattr(embedding_script, "resolve_model_source", resolve)
    monkeypatch.setattr(embedding_script, "SentenceTransformer", build)
    runtime = SimpleNamespace(
        parallelism_config=SimpleNamespace(is_ep_mode=False, is_tp_mode=False, max_concurrent_loading=None),
        model_source=REPO,
    )
    with pytest.raises(_BackboneBuilt):
        embedding_script.build_sentence_transformer(
            runtime,
            EmbeddingConfig(output_dir=str(tmp_path), bf16=False, use_cpu=True),
            ModelConfig(model_name_or_path=REPO, model_revision="v1"),
            SimpleNamespace(reset_sinks=True, train_sinks=False),
        )
    assert seen == {"resolve": (REPO, "v1", "embedding_model"), "load": (REPO, COMMIT_A)}


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
