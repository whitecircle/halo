#!/usr/bin/env python
"""load_expert_lora_adapters matches every saved key and shape to the live expert adapters before it loads any.
A partial miss would resume the missed layer's adapters from zero-init."""

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from halo.checkpoint.adapters import is_expert_lora_key
from halo.distributed.expert_parallel.moe_layer.expert_lora import load_expert_lora_adapters
from tests.common.ep_stubs import StubEPLayerBase


class _StubEPLoraLayer(StubEPLayerBase):
    """Expert LoRA on ``gate_up_proj`` for its two experts; records what the apply walk hands it."""

    def __init__(self):
        super().__init__()
        self.expert_start, self.expert_end = 0, 2
        self.ep_config = SimpleNamespace(ep_size=1)
        self.gate_up_proj_lora_A = nn.Parameter(torch.zeros(2, 4, 2))
        self.gate_up_proj_lora_B = nn.Parameter(torch.zeros(2, 2, 8))
        self._expert_lora_attrs = frozenset({"gate_up_proj"})
        self.loaded_state = None

    def load_expert_lora_state_dict(self, layer_state):
        self.loaded_state = dict(layer_state)


def _model_with_ep_layers() -> nn.Module:
    """Two EP layers with expert LoRA at ``model.layers.{0,1}.mlp``, an HF-shaped tree."""
    root = nn.Module()
    backbone = nn.Module()
    layers = nn.ModuleList()
    for _ in range(2):
        layer = nn.Module()
        layer.mlp = _StubEPLoraLayer()
        layers.append(layer)
    backbone.layers = layers
    root.model = backbone
    return root


def _adapter_keys(layer_idx: int) -> dict[str, torch.Tensor]:
    prefix = f"model.layers.{layer_idx}.mlp.experts.gate_up_proj"
    return {f"{prefix}.lora_A": torch.randn(2, 4, 2), f"{prefix}.lora_B": torch.randn(2, 2, 8)}


def test_stock_peft_expert_keys_are_not_claimed_as_native():
    """Bailing/Ling keys carry ``.experts.`` too; a substring predicate would lose them to the EP loader."""
    peft_key = "base_model.model.model.layers.0.mlp.experts.3.gate_proj.lora_A.default.weight"
    assert not is_expert_lora_key(peft_key)
    assert not is_expert_lora_key("base_model.model.model.layers.0.mlp.experts.3.gate_proj.lora_B.weight")
    # The native namespace ends at the adapter, with no tuner suffix.
    assert is_expert_lora_key("model.layers.0.mlp.experts.gate_up_proj.lora_A")
    assert is_expert_lora_key("model.layers.0.mlp.experts.down_proj.lora_B")


def test_expert_keys_with_no_ep_layer_to_receive_them_raise():
    """The walk finds no layer to miss, so without this the deltas are dropped and the resume reports success."""
    model = _model_with_ep_layers()
    for layer in model.model.layers:
        layer.mlp._expert_lora_attrs = frozenset()  # a run that builds no expert adapters

    with pytest.raises(RuntimeError, match="no live tensor takes"):
        load_expert_lora_adapters(model, _adapter_keys(0))


def test_all_layers_matched_loads_every_layer():
    model = _model_with_ep_layers()
    load_expert_lora_adapters(model, {**_adapter_keys(0), **_adapter_keys(1)})
    for layer in model.model.layers:
        assert layer.mlp.loaded_state is not None
        assert set(layer.mlp.loaded_state) == {"experts.gate_up_proj.lora_A", "experts.gate_up_proj.lora_B"}


def test_full_miss_raises():
    """Keys under a foreign path (an unstripped wrapper prefix) match no layer."""
    model = _model_with_ep_layers()
    foreign = {f"wrapper.{k}": v for k, v in _adapter_keys(0).items()}
    with pytest.raises(RuntimeError, match=r"missing from the file .*wrapper\."):
        load_expert_lora_adapters(model, foreign)


def test_partial_miss_raises_naming_missed_layer():
    """The missed layer's adapters would resume from zero-init; nothing may load partially."""
    model = _model_with_ep_layers()
    with pytest.raises(RuntimeError, match=r"model\.layers\.1\.mlp"):
        load_expert_lora_adapters(model, _adapter_keys(0))  # layer 1 gets zero keys
    for layer in model.model.layers:
        assert layer.mlp.loaded_state is None, "a miss must not leave the model partially updated"


def test_no_expert_tensors_for_a_run_that_trains_expert_lora_raise():
    """An attention-only save resumed by a mixed run would resume every expert adapter from zero-init."""
    model = _model_with_ep_layers()
    with pytest.raises(RuntimeError, match=r"model\.layers\.0\.mlp\.experts\.gate_up_proj\.lora_A"):
        load_expert_lora_adapters(model, {})
    for layer in model.model.layers:
        assert layer.mlp.loaded_state is None


def test_no_expert_tensors_for_a_run_without_expert_lora_is_a_noop():
    model = _model_with_ep_layers()
    for layer in model.model.layers:
        layer.mlp._expert_lora_attrs = frozenset()
    load_expert_lora_adapters(model, {})
    for layer in model.model.layers:
        assert layer.mlp.loaded_state is None


def test_a_reshaped_adapter_raises_before_any_layer_loads():
    model = _model_with_ep_layers()
    state = {**_adapter_keys(0), **_adapter_keys(1)}
    key = "model.layers.1.mlp.experts.gate_up_proj.lora_A"
    state[key] = torch.randn(2, 4, 3)
    with pytest.raises(RuntimeError, match="reshaped"):
        load_expert_lora_adapters(model, state)
    for layer in model.model.layers:
        assert layer.mlp.loaded_state is None


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
