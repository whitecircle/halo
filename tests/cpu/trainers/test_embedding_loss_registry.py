#!/usr/bin/env python
"""Every ``loss_type`` the embedding config accepts builds a loss, and every registered loss is accepted.
A choice with no registry entry fails only at a run's first construction; an entry the config omits is unreachable."""

import inspect
from typing import get_args, get_type_hints

import pytest

from halo.configs.embedding import SCALED_LOSSES, EmbeddingConfig
from halo.trainers.embedding.trainer import LOSS_REGISTRY

_CPU_OK = {"output_dir": "/tmp/test_embedding_loss_registry", "bf16": False, "use_cpu": True}
_REFUSED = "does not support these config fields"


def test_the_registry_holds_exactly_the_configs_loss_choices():
    assert set(LOSS_REGISTRY) == set(get_args(get_type_hints(EmbeddingConfig)["loss_type"]))


def test_the_scaled_losses_are_the_ones_whose_constructor_takes_a_scale():
    assert {
        name for name, loss in LOSS_REGISTRY.items() if "scale" in inspect.signature(loss.__init__).parameters
    } == SCALED_LOSSES


@pytest.mark.parametrize("loss_type", sorted(LOSS_REGISTRY))
def test_loss_scale_is_refused_on_every_loss_that_does_not_read_it(loss_type):
    if loss_type in SCALED_LOSSES:
        assert EmbeddingConfig(loss_type=loss_type, loss_scale=5.0, **_CPU_OK).loss_scale == 5.0
    else:
        with pytest.raises(ValueError, match=rf"loss_type={loss_type} {_REFUSED}.*loss_scale"):
            EmbeddingConfig(loss_type=loss_type, loss_scale=5.0, **_CPU_OK)


@pytest.mark.parametrize("loss_type", sorted(LOSS_REGISTRY))
def test_the_mini_batch_size_is_refused_on_every_loss_but_cached_mnrl(loss_type):
    if loss_type == "cached_mnrl":
        config = EmbeddingConfig(loss_type=loss_type, cached_mnrl_mini_batch_size=8, **_CPU_OK)
        assert config.cached_mnrl_mini_batch_size == 8
    else:
        with pytest.raises(ValueError, match=rf"loss_type={loss_type} {_REFUSED}.*cached_mnrl_mini_batch_size"):
            EmbeddingConfig(loss_type=loss_type, cached_mnrl_mini_batch_size=8, **_CPU_OK)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
