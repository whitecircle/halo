#!/usr/bin/env python
"""CPU test: one predicate reads whether a model declares linear-attention layers.

``has_linear_attention_layers`` reads ``layer_types`` through a VLM wrapper, where the composite releases
(Qwen3.5/3.6, GLM-5 Next, Qwen3.8-Flash-Next) keep it: their top-level config declares none. The GatedDeltaNet
predicate and the refusal of linear attention no segment marker reaches both build on it, so a composite
checkpoint resolves exactly as its text tower does. Every transformers config declaring such layers is swept.

    python tests/cpu/models/test_linear_attention_layers.py
"""

import pytest
import transformers
from transformers import CONFIG_MAPPING

from halo.models.attention.layout import (
    declares_linear_layer_groups,
    has_gated_delta_net_layers,
    has_linear_attention_layers,
)
from halo.models.segment_markers import find_unmarked_sequence_mixer, segment_markers_for

# Composite configs whose text sub-config alone carries the linear-attention layer types.
COMPOSITE_RELEASES = ("qwen3_5", "qwen3_5_moe", "glm5_next", "qwen4_exp")
GATED_DELTA_NET = ("qwen3_5", "qwen3_5_moe", "qwen3_5_text", "qwen3_5_moe_text", "qwen3_next", "qwen4_exp")
UNMARKED = ("glm5_next", "glm5_next_text", "kimi_linear", "bailing_hybrid", "minimax", "jamba")


def _default_config(model_type):
    transformers.logging.set_verbosity_error()
    return CONFIG_MAPPING[model_type]()


@pytest.mark.parametrize("model_type", COMPOSITE_RELEASES)
def test_a_composite_release_declares_its_text_towers_layers(model_type):
    config = _default_config(model_type)
    assert "linear_attention" not in (getattr(config, "layer_types", None) or ()), "premise: none at the top level"
    assert has_linear_attention_layers(config)
    assert has_linear_attention_layers(config) == has_linear_attention_layers(config.get_text_config())


@pytest.mark.parametrize("model_type", ("llama", "qwen3", "gpt_oss", "lfm2", "gemma3_text"))
def test_softmax_and_conv_families_declare_none(model_type):
    assert not has_linear_attention_layers(_default_config(model_type))


def test_no_config_declares_none():
    assert not has_linear_attention_layers(None)


@pytest.mark.parametrize("model_type", GATED_DELTA_NET + UNMARKED)
def test_both_consumers_resolve_from_the_shared_predicate(model_type):
    config = _default_config(model_type)
    assert has_linear_attention_layers(config)
    gated_delta_net = model_type in GATED_DELTA_NET
    assert has_gated_delta_net_layers(config) is gated_delta_net
    assert segment_markers_for(config).cu_seq_lens is gated_delta_net
    assert (find_unmarked_sequence_mixer(config) is None) is gated_delta_net


def test_hybrid_layer_groups_are_unmarked_without_layer_types():
    """A group-declared hybrid with no ``layer_types`` reaches the refusal through its groups alone."""
    config = transformers.PretrainedConfig(layer_group_size=4)
    assert declares_linear_layer_groups(config) and not has_linear_attention_layers(config)
    assert find_unmarked_sequence_mixer(config) == "linear-attention layers"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
