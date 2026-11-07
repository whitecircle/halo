"""FSDP2 shards something for every backbone shape. Zero ``fully_shard`` calls still set ``_fsdp_wrapped``,
suppressing the DDP fallback, so replicas diverge (``SentenceTransformer`` reaches this)."""

import math
from unittest.mock import patch

import pytest
import torch.nn as nn
from torch.distributed.fsdp import MixedPrecisionPolicy
from transformers import GenerationMixin, PretrainedConfig, PreTrainedModel

import halo.distributed.fsdp as fsdp
from halo.distributed.fsdp import IdentityParamSet
from halo.models.introspection.module_tree import DECODER_LAYER_LIST_ATTRS


class _ProbeConfig(PretrainedConfig):
    model_type = "fsdp_probe"


class _SentenceTransformerLike(nn.Sequential):
    """The shape that sharded nothing."""

    def __init__(self):
        super().__init__(nn.Linear(8, 8), nn.Linear(8, 8))


class _UnreachableLayerListDecoder(PreTrainedModel, GenerationMixin):
    """Spells its layer list ``.blocks``, invisible to the probe."""

    config_class = _ProbeConfig

    def __init__(self):
        super().__init__(_ProbeConfig())
        self.blocks = nn.ModuleList([nn.Linear(8, 8), nn.Linear(8, 8)])


class _BertLikeClassifier(PreTrainedModel):
    """``encoder.layer`` is out of reach by design."""

    config_class = _ProbeConfig

    def __init__(self):
        super().__init__(_ProbeConfig())
        self.encoder = nn.Module()
        self.encoder.layer = nn.ModuleList([nn.Linear(8, 8)])
        self.classifier = nn.Linear(8, 2)


class _PipelineStageLike(nn.Module):
    """Mirrors ``can_generate`` but is no ``PreTrainedModel``; its slice may hold no layer."""

    def __init__(self):
        super().__init__()
        self.head = nn.Linear(8, 8)

    def can_generate(self) -> bool:
        return True


class _Gpt2Like(nn.Module):
    def __init__(self):
        super().__init__()
        self.transformer = nn.Module()
        self.transformer.h = nn.ModuleList([nn.Linear(8, 8), nn.Linear(8, 8)])


class _CausalLmLike(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([nn.Linear(8, 8), nn.Linear(8, 8)])


def _shard_calls(model):
    calls = []
    with (
        patch.object(fsdp, "fully_shard", side_effect=lambda mod, **kw: calls.append(mod)),
        patch.object(fsdp, "_install_unit_hooks"),
    ):
        count = fsdp.apply_fsdp2_per_layer(
            model,
            dp_mesh=None,
            mp_policy=MixedPrecisionPolicy(),
            reshard_after_forward=False,
            ignored_params=IdentityParamSet(),
        )
    return count, calls


@pytest.mark.parametrize(
    "factory,label",
    [
        (_SentenceTransformerLike, "SentenceTransformer-like (no layer list)"),
        (_Gpt2Like, "GPT-2-like (.h)"),
        (_CausalLmLike, "CausalLM-like (.layers)"),
        (_BertLikeClassifier, "BERT-family classification backbone"),
        (_PipelineStageLike, "PP stage holding no decoder layer"),
    ],
)
def test_every_backbone_shape_gets_at_least_one_shard_group(factory, label):
    count, calls = _shard_calls(factory())
    assert count >= 1, f"{label}: sharded nothing — replicas would never sync gradients"
    assert len(calls) == count, f"{label}: reported {count} groups but issued {len(calls)} fully_shard calls"


def test_layer_list_backbones_shard_in_layer_units_not_monolithically():
    """One root group all-gathers the whole model."""
    for factory, attr in ((_CausalLmLike, ".layers"), (_Gpt2Like, ".h")):
        model = factory()
        count, calls = _shard_calls(model)
        layer_list = getattr(model, "model", None) or model.transformer
        layers = getattr(layer_list, "layers", None)
        if layers is None:
            layers = layer_list.h
        units = math.ceil(len(layers) / fsdp.layers_per_shard_group(reshard_after_forward=False))
        assert count == units + 2, f"{attr}: expected {units} layer unit(s) + backbone + root, got {count}"
        wrapped = [module for call in calls for module in (call if isinstance(call, list) else [call])]
        for layer in layers:
            assert any(module is layer for module in wrapped), f"{attr}: decoder layer never sharded in a unit"


def test_a_model_with_no_layers_shards_its_root():
    """Anti-vacuity: pins which module the fallback shards."""
    model = _SentenceTransformerLike()
    _, calls = _shard_calls(model)
    assert calls == [model], f"expected the root itself to be the shard group, got {calls}"


def test_a_decoder_whose_layer_list_is_unreachable_is_refused():
    """The message names the class and probed attributes, since the fix is adding this spelling to
    ``DECODER_LAYER_LIST_ATTRS``."""
    model = _UnreachableLayerListDecoder()
    with pytest.raises(RuntimeError) as excinfo:
        _shard_calls(model)
    message = str(excinfo.value)
    assert "_UnreachableLayerListDecoder" in message
    assert "DECODER_LAYER_LIST_ATTRS" in message
    for attr in DECODER_LAYER_LIST_ATTRS:
        assert repr(attr) in message or attr in message


def test_the_refusal_fires_before_anything_is_wrapped():
    """No fully_shard call may have run."""
    calls = []
    with patch.object(fsdp, "fully_shard", side_effect=lambda mod, **kw: calls.append(mod)):
        with pytest.raises(RuntimeError):
            fsdp.apply_fsdp2_per_layer(
                _UnreachableLayerListDecoder(),
                dp_mesh=None,
                mp_policy=MixedPrecisionPolicy(),
                reshard_after_forward=False,
                ignored_params=IdentityParamSet(),
            )
    assert calls == [], f"wrapped {len(calls)} module(s) before refusing"


def test_reachable_layer_lists_and_non_decoders_are_never_refused():
    """Gated on generative decoder rather than on no layer list."""
    for factory in (_BertLikeClassifier, _PipelineStageLike, _SentenceTransformerLike, _CausalLmLike):
        count, _ = _shard_calls(factory())
        assert count >= 1


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
