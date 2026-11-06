#!/usr/bin/env python
"""A weight push carries what the engine's reload needs beyond the trainer's parameters. vLLM's layerwise reload counts
every tensor of a module and rebuilds one that received only part of them from uninitialized storage, so a Ling push
that leaves ``gate.expert_bias`` behind overwrites the served router bias while ``/update_weights`` answers 200."""

from __future__ import annotations

import json
import os
import re

import pytest
import torch
from peft import LoraConfig, get_peft_model

from halo.checkpoint.parallel.ep import save_ep_gathered_checkpoint
from halo.distributed.expert_parallel.moe_layer.gather import EPGatherMixin
from halo.distributed.expert_parallel.moe_layer.layer import named_ep_layers
from halo.distributed.expert_parallel.parallelize import apply_ep_layers
from halo.models.moe.family_specs import MOE_FAMILY_SPECS
from halo.trainers.grpo.rollout import weight_push
from halo.trainers.grpo.rollout.weight_push import (
    gather_and_send_weights,
    reject_unsyncable_model,
    sync_weights_to_client,
)
from halo.weight_sync.registry import resolve_weight_sync_client, rollout_backends
from tests.common.checkpoint_io import save_context, written_keys
from tests.common.lora_sync_exactness import syncable_moe_families
from tests.common.parallelism import single_process_ep_config
from tests.common.repo import REPO_ROOT
from tests.common.tiny_models import TINY_MOE_FAMILIES, tiny_family_model
from tests.common.weight_sync import RecordingSender, Wire, offline_client

_BIAS = "gate.expert_bias"
_PARITY_FIXTURES = os.path.join(REPO_ROOT, "docker", "vllm", "parity", "fixtures")
# What each syncable family's gathered checkpoint holds that a push leaves out, and why the served tensor keeps its
# launch value. vLLM's reload skips ``e_score_correction_bias`` by name wherever it sits, and a module that receives
# nothing keeps its kernel tensors; SGLang's loaders copy name by name, so an unsent tensor is untouched.
_LEFT_OUT_OF_THE_PUSH = {
    # Its decoder-layer module holds no other direct tensor.
    "gemma4_text": {"model.layers.N.layer_scalar"},
    "glm4_moe_lite": {"model.layers.N.mlp.gate.e_score_correction_bias"},
    # vLLM maps the hub key onto the gate's ``e_score_correction_bias``.
    "lfm2_moe": {"model.layers.N.feed_forward.expert_bias"},
    # ``router_bias`` on vLLM's block module, which holds no other direct tensor.
    "step3p7": {"model.layers.N.moe.router_bias"},
}


@pytest.fixture(autouse=True)
def _gather_on_the_host(monkeypatch):
    """The push asks for its experts on the GPU, which the CPU tier does not have."""
    start = EPGatherMixin.start_gather_expert_state_dict

    def on_host(self, *, device="cpu", **kwargs):
        return start(self, device="cpu", **kwargs)

    monkeypatch.setattr(EPGatherMixin, "start_gather_expert_state_dict", on_host)


def _ep_model(family: str, overrides: dict | None = None):
    torch.manual_seed(0)
    model = tiny_family_model(TINY_MOE_FAMILIES[family], overrides=overrides).to(torch.bfloat16)
    apply_ep_layers(model, single_process_ep_config())
    assert named_ep_layers(model), f"{family}: no EP layer to gather"
    return model


def _pushed(model) -> dict[str, torch.Tensor]:
    sender = RecordingSender(keep_values=True)
    gather_and_send_weights(model, sender)
    assert len(sender.names) == len(set(sender.names)), "a tensor was forwarded twice"
    return {param.name: param.value for param in sender.params}


def _distinct_biases(model) -> dict[str, torch.Tensor]:
    """Filled per layer, so a tensor sent under another layer's name or the zeros of a fresh buffer cannot match."""
    biases = {}
    for index, (name, layer) in enumerate(named_ep_layers(model)):
        bias = layer.get_buffer(_BIAS)
        bias.copy_(torch.linspace(-0.25, 0.25, bias.numel()) + index / 8)
        biases[f"{name}.{_BIAS}"] = bias
    return biases


def _per_layer_spelling(names) -> set[str]:
    return {re.sub(r"\.\d+(?=\.)", ".N", name) for name in names}


def _laguna_with_bias(value: float):
    model = _ep_model("laguna")
    for _, layer in named_ep_layers(model):
        layer.get_submodule("gate").e_score_correction_bias.data.fill_(value)
    return model


def _bias_warnings(caplog) -> list[str]:
    return [record.getMessage() for record in caplog.records if "nonzero router bias" in record.getMessage()]


def test_a_trained_laguna_bias_warns_that_the_engine_routes_without_it(caplog):
    """The cookbook's SFT with bias balancing ships the bias, its GRPO step starts from that checkpoint, and vLLM's
    loader drops the key: routing then differs on every prompt, which no sync test would see."""
    model = _laguna_with_bias(0.25)
    with caplog.at_level("WARNING"):
        reject_unsyncable_model(model, "vllm")
    (warning,) = _bias_warnings(caplog)
    assert "routing_replay: rollout" in warning and str(len(named_ep_layers(model))) in warning


def test_a_laguna_at_its_zero_release_bias_is_silent(caplog):
    with caplog.at_level("WARNING"):
        reject_unsyncable_model(_laguna_with_bias(0.0), "vllm")
    assert not _bias_warnings(caplog)


def test_a_family_whose_engine_loads_its_bias_is_silent(caplog):
    model = _ep_model("glm4_moe_lite")
    for _, layer in named_ep_layers(model):
        layer.get_submodule("gate").e_score_correction_bias.data.fill_(0.25)
    with caplog.at_level("WARNING"):
        reject_unsyncable_model(model, "vllm")
    assert not _bias_warnings(caplog)


def test_a_ling_push_carries_each_gate_bias_at_the_value_the_trainer_routes_with():
    model = _ep_model("bailing_moe")
    biases = _distinct_biases(model)
    pushed = _pushed(model)

    assert len(biases) == sum(1 for name in pushed if name.endswith("mlp.gate.weight")) > 0
    for name, bias in biases.items():
        assert torch.equal(pushed[name], bias), f"{name} was not pushed at the trainer's value"


def test_a_ling_checkpoint_without_the_bias_sends_none():
    """Both engines create the tensor on this flag and vLLM's loader raises on the name without it, while the release's
    gate registers the buffer either way."""
    model = _ep_model("bailing_moe", {"moe_router_enable_expert_bias": False})
    assert all(layer.get_buffer(_BIAS) is not None for _, layer in named_ep_layers(model))

    pushed = _pushed(model)

    assert not [name for name in pushed if name.endswith(_BIAS)]
    assert any(name.endswith("mlp.gate.weight") for name in pushed)


def test_a_lora_push_carries_the_bias_under_its_hub_name():
    model = _ep_model("bailing_moe")
    biases = _distinct_biases(model)
    peft_model = get_peft_model(model, LoraConfig(r=2, target_modules=["query_key_value"]))

    pushed = _pushed(peft_model)

    for name, bias in biases.items():
        assert torch.equal(pushed[name], bias)


@pytest.mark.parametrize("backend", rollout_backends())
def test_a_gate_under_modules_to_save_pushes_the_bias_it_routes_with(backend):
    """``lora_modules_to_save: [gate]`` is a documented target: it puts PEFT's wrapper between the layer and the buffer,
    and the trainable copy behind it is the one that routes."""
    model = _ep_model("bailing_moe")
    names = list(_distinct_biases(model))
    peft_model = get_peft_model(model, LoraConfig(r=2, target_modules=["query_key_value"], modules_to_save=["gate"]))
    routed = {}
    for name, (_, layer) in zip(names, named_ep_layers(peft_model), strict=True):
        routed[name] = layer.gate.modules_to_save["default"].get_buffer("expert_bias")
        routed[name].add_(1.0)

    reject_unsyncable_model(peft_model, backend)
    pushed = _pushed(peft_model)

    for name, bias in routed.items():
        assert torch.equal(pushed[name], bias)
        assert name.replace(_BIAS, "gate.weight") in pushed


@pytest.mark.parametrize("backend", rollout_backends())
def test_the_bias_reaches_the_wire_of_both_engines(backend):
    """SGLang needs it less, as its loader copies name by name, and takes it as a parameter of the same name."""
    model = _ep_model("bailing_moe")
    biases = _distinct_biases(model)
    wire = Wire()
    client = wire.attach(offline_client(resolve_weight_sync_client(backend)))

    sync_weights_to_client(model, client, is_main=True, is_tp_main=True)

    sent = dict(wire.sent)
    for name, bias in biases.items():
        assert torch.equal(sent[name], bias)


@pytest.mark.parametrize("family", syncable_moe_families())
def test_a_push_covers_the_gathered_checkpoint_on_every_syncable_family(family, tmp_path):
    """Every tensor a served checkpoint holds is pushed under its own name, but for the ones a family's engine keeps
    apart. A new buffer on any family lands here until someone reads what the engine does with it."""
    model = _ep_model(family)
    save_ep_gathered_checkpoint(save_context(model), str(tmp_path))
    checkpoint, pushed = written_keys(str(tmp_path)), set(_pushed(model))

    assert _per_layer_spelling(checkpoint - pushed) == _LEFT_OUT_OF_THE_PUSH.get(family, set())
    assert pushed <= checkpoint


def test_the_left_out_table_names_only_syncable_families():
    assert set(_LEFT_OUT_OF_THE_PUSH) <= set(syncable_moe_families())


@pytest.mark.parametrize("backend", rollout_backends())
def test_a_declared_buffer_the_layer_lost_is_refused_at_construction(backend):
    """An upstream rename would otherwise raise mid-push, after the engine already took a chunk."""
    model = _ep_model("bailing_moe")
    reject_unsyncable_model(model, backend)

    (_, layer), *_ = named_ep_layers(model)
    del layer.gate._buffers["expert_bias"]

    with pytest.raises(ValueError, match="engine_loaded_buffers"):
        reject_unsyncable_model(model, backend)


def test_a_declared_buffer_that_became_a_parameter_is_refused_at_construction():
    """The dense push sends a parameter already, and two copies of one name stall vLLM's count and fail SGLang's chunk."""
    model = _ep_model("bailing_moe")
    (_, layer), *_ = named_ep_layers(model)
    bias = layer.gate._buffers.pop("expert_bias")
    layer.gate.register_parameter("expert_bias", torch.nn.Parameter(bias))

    with pytest.raises(ValueError, match="engine_loaded_buffers"):
        reject_unsyncable_model(model, "vllm")


def test_a_rank_with_no_sender_walks_the_same_buffers_in_the_same_order(monkeypatch):
    """A sharded buffer would be a collective, which every rank must join in one order."""
    model = _ep_model("bailing_moe")
    biases = {id(bias) for bias in _distinct_biases(model).values()}
    walked: list[int | None] = []
    materialize = weight_push.materialize_dtensor

    def recording(data):
        # A parameter's ``.data`` is a fresh object per read, so only its place in the walk is compared.
        walked.append(id(data) if id(data) in biases else None)
        return materialize(data)

    monkeypatch.setattr(weight_push, "materialize_dtensor", recording)

    gather_and_send_weights(model, RecordingSender())
    forwarding, walked[:] = list(walked), []
    gather_and_send_weights(model, None)

    assert walked == forwarding
    assert biases <= set(walked)


@pytest.mark.parametrize("spec", [spec for spec in MOE_FAMILY_SPECS if spec.engine_loaded_buffers])
def test_a_declared_flag_is_the_one_the_served_config_turns_on(spec):
    """A misspelled flag would silently leave the buffer out of every push."""
    with open(os.path.join(_PARITY_FIXTURES, spec.model_types[0], "config.json")) as handle:
        config = json.load(handle)

    assert config[spec.engine_loaded_buffers.config_flag] is True


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
