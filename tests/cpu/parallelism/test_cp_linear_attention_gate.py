#!/usr/bin/env python
"""Ulysses CP's config-only gate refuses exactly the models declaring linear-attention layers, read through
``has_linear_attention_layers``. A meta shell instantiates no mixer; swept over every transformers config."""

import pytest
import torch.nn as nn
import transformers
from transformers import CONFIG_MAPPING

from halo.distributed.context_parallel.config import CPConfigError
from halo.distributed.context_parallel.model_rules import reject_unsupported_ulysses_model
from halo.models.attention.layout import has_linear_attention_layers
from halo.models.config_access import get_config_field

COMPOSITE_RELEASES = ("qwen3_5", "qwen3_5_moe", "glm5_next", "qwen4_exp")


class _ConfigOnlyShell(nn.Module):
    """The config and nothing else."""

    def __init__(self, config):
        super().__init__()
        self.config = config


def _default_configs():
    """That build with defaults and name one text config."""
    transformers.logging.set_verbosity_error()
    configs = {}
    for model_type in sorted(CONFIG_MAPPING.keys()):
        try:
            config = CONFIG_MAPPING[model_type]()
            config.get_text_config()
        except Exception:
            continue
        configs[model_type] = config
    return configs


def _refuses_linear_attention(config) -> bool:
    try:
        reject_unsupported_ulysses_model(_ConfigOnlyShell(config), 2)
    except CPConfigError as error:
        return "``linear_attention``" in str(error)
    return False


def test_the_gate_refuses_exactly_where_the_predicate_holds():
    configs = _default_configs()
    declaring = {name for name, config in configs.items() if has_linear_attention_layers(config)}
    assert set(COMPOSITE_RELEASES) <= declaring, "premise: the composite releases are in the sweep"
    refused = {name for name, config in configs.items() if _refuses_linear_attention(config)}
    assert refused == declaring


@pytest.mark.parametrize("model_type", COMPOSITE_RELEASES)
def test_a_composite_release_is_refused_with_its_text_towers_count(model_type):
    config = _default_configs()[model_type]
    layer_types = get_config_field(config, "layer_types")
    count = sum(layer_type == "linear_attention" for layer_type in layer_types)
    with pytest.raises(CPConfigError, match=rf"{count} of {len(layer_types)} layers are ``linear_attention``"):
        reject_unsupported_ulysses_model(_ConfigOnlyShell(config), 2)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
