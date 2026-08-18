"""FSDP2 must shard something for every backbone shape, and say so.

``apply_fsdp2_per_layer`` resolves the decoder layer list to build per-layer shard groups. Finding
none must not issue ZERO ``fully_shard`` calls while the caller still returns success — that sets
``_fsdp_wrapped``, which suppresses the DDP fallback. The result is a data-parallel run with no
gradient synchronization on either path: replicas diverge from step one and whichever rank writes
the checkpoint wins. Reachable from the documented ``scripts/training/embedding.py`` launch,
because ``SentenceTransformer`` is an ``nn.Sequential`` with no ``.layers``/``.h``/``.model``, and
from any BERT-family reward/classification backbone.

A second shape is merely wasteful: ``backbone_with_layers`` accepts ``.h`` (GPT-2 style), so a
function checking only ``.layers`` gives those models one monolithic root group, all-gathered for
the whole forward.

The remaining trap is a DECODER whose layer list this probe cannot reach: one root group, the whole
model all-gathered for the entire forward, reported as success. That one is refused
(``_reject_unreachable_decoder_layers``) — but only for generative ``PreTrainedModel`` decoders, so
the layer-less shapes above keep working.

    python tests/cpu/parallelism/test_fsdp2_shards_every_backbone.py
"""

import sys
from unittest.mock import patch

import pytest
import torch.nn as nn
from transformers import GenerationMixin, PretrainedConfig, PreTrainedModel

import src.distributed.fsdp as fsdp
from src.distributed.fsdp import IdentityParamSet
from src.models.structure import DECODER_LAYER_LIST_ATTRS


class _ProbeConfig(PretrainedConfig):
    model_type = "fsdp_probe"


class _SentenceTransformerLike(nn.Sequential):
    """No ``.layers``/``.h``/``.model``/``.transformer`` — the shape that sharded nothing."""

    def __init__(self):
        super().__init__(nn.Linear(8, 8), nn.Linear(8, 8))


class _UnreachableLayerListDecoder(PreTrainedModel, GenerationMixin):
    """A generative decoder spelling its layer list ``.blocks`` — invisible to the probe."""

    config_class = _ProbeConfig

    def __init__(self):
        super().__init__(_ProbeConfig())
        self.blocks = nn.ModuleList([nn.Linear(8, 8), nn.Linear(8, 8)])


class _BertLikeClassifier(PreTrainedModel):
    """A PreTrainedModel that is NOT generative: ``encoder.layer`` is out of reach by design."""

    config_class = _ProbeConfig

    def __init__(self):
        super().__init__(_ProbeConfig())
        self.encoder = nn.Module()
        self.encoder.layer = nn.ModuleList([nn.Linear(8, 8)])
        self.classifier = nn.Linear(8, 2)


class _PipelineStageLike(nn.Module):
    """A PP stage mirrors ``can_generate`` but is no ``PreTrainedModel``; its slice may hold no layer."""

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
    with patch.object(fsdp, "fully_shard", side_effect=lambda mod, **kw: calls.append(mod)):
        count = fsdp.apply_fsdp2_per_layer(
            model,
            dp_mesh=None,
            mp_policy=None,
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
    """Zero shard groups is the failure that reports itself as success."""
    count, calls = _shard_calls(factory())
    assert count >= 1, f"{label}: sharded nothing — replicas would never sync gradients"
    assert len(calls) == count, f"{label}: reported {count} groups but issued {len(calls)} fully_shard calls"


def test_layer_list_backbones_shard_per_layer_not_monolithically():
    """Per-layer groups are the point of FSDP2; one root group all-gathers the whole model.

    Both ``.layers`` and ``.h`` resolve through the same rule as ``backbone_with_layers``, so both
    must produce a group per layer plus the backbone plus the root.
    """
    for factory, attr in ((_CausalLmLike, ".layers"), (_Gpt2Like, ".h")):
        model = factory()
        count, calls = _shard_calls(model)
        assert count == 4, f"{attr}: expected 2 layers + backbone + root, got {count}"
        layer_list = getattr(model, "model", None) or model.transformer
        layers = getattr(layer_list, "layers", None)
        if layers is None:
            layers = layer_list.h
        for layer in layers:
            assert any(c is layer for c in calls), f"{attr}: decoder layer never sharded"


def test_a_model_with_no_layers_shards_its_root():
    """Anti-vacuity: pin WHICH module the fallback shards, so a future edit cannot satisfy the
    count above by sharding some arbitrary submodule."""
    model = _SentenceTransformerLike()
    _, calls = _shard_calls(model)
    assert calls == [model], f"expected the root itself to be the shard group, got {calls}"


def test_a_decoder_whose_layer_list_is_unreachable_is_refused():
    """One root group for a decoder is the silent memory cliff — it must not wrap at all.

    The message has to name the class and the probed attributes: the fix is adding this family's
    spelling to ``DECODER_LAYER_LIST_ATTRS``, and nothing else in the run says so.
    """
    model = _UnreachableLayerListDecoder()
    with pytest.raises(RuntimeError) as excinfo:
        _shard_calls(model)
    message = str(excinfo.value)
    assert "_UnreachableLayerListDecoder" in message
    assert "DECODER_LAYER_LIST_ATTRS" in message
    for attr in DECODER_LAYER_LIST_ATTRS:
        assert repr(attr) in message or attr in message


def test_the_refusal_fires_before_anything_is_wrapped():
    """A half-wrapped model is worse than an unwrapped one: no fully_shard call may have run."""
    calls = []
    with patch.object(fsdp, "fully_shard", side_effect=lambda mod, **kw: calls.append(mod)):
        with pytest.raises(RuntimeError):
            fsdp.apply_fsdp2_per_layer(
                _UnreachableLayerListDecoder(),
                dp_mesh=None,
                mp_policy=None,
                reshard_after_forward=False,
                ignored_params=IdentityParamSet(),
            )
    assert calls == [], f"wrapped {len(calls)} module(s) before refusing"


def test_reachable_layer_lists_and_non_decoders_are_never_refused():
    """Anti-false-positive: the refusal is gated on 'generative decoder', not on 'no layer list'.

    A BERT-family classification backbone and a PP stage with no decoder layer are supported shapes
    that legitimately reach the root-only wrap; a decoder with a reachable list never gets there.
    """
    for factory in (_BertLikeClassifier, _PipelineStageLike, _SentenceTransformerLike, _CausalLmLike):
        count, _ = _shard_calls(factory())
        assert count >= 1


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
