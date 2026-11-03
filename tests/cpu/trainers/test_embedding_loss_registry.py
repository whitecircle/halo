#!/usr/bin/env python
"""Every ``loss_type`` the embedding config accepts builds a loss, and every registered loss is accepted.
A choice with no registry entry fails only at a run's first construction; an entry the config omits is unreachable."""

from typing import get_args, get_type_hints

import pytest

from halo.configs.embedding import EmbeddingConfig
from halo.trainers.embedding.trainer import LOSS_REGISTRY


def test_the_registry_holds_exactly_the_configs_loss_choices():
    assert set(LOSS_REGISTRY) == set(get_args(get_type_hints(EmbeddingConfig)["loss_type"]))


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
