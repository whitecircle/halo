#!/usr/bin/env python
"""Every ``loss_type`` the embedding config accepts builds a loss, and every registered loss is accepted.

The parser refuses a value outside ``EmbeddingConfig.loss_type``'s choices, so ``build_embedding_loss``
looks the registry up without a miss branch: a choice added to the config without a registry entry
would fail only at a run's first construction, and a registry entry the config omits is unreachable.

Run: pytest tests/cpu/trainers/test_embedding_loss_registry.py
"""

from typing import get_args, get_type_hints

import pytest

from src.configs.embedding import EmbeddingConfig
from src.trainers.embedding.trainer import LOSS_REGISTRY


def test_the_registry_holds_exactly_the_configs_loss_choices():
    assert set(LOSS_REGISTRY) == set(get_args(get_type_hints(EmbeddingConfig)["loss_type"]))


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
