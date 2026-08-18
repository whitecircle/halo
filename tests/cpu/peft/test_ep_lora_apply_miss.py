#!/usr/bin/env python
"""apply_ep_lora_adapters miss detection: full AND partial adapter-key misses must fail loud.

A total-consumed check catches only the case where NO saved key matches any EP layer. A PARTIAL
miss — one layer's adapters unmatched (wrapper-prefixed save path, renamed layer) while other
layers match — slips through it: the missed layer gets an empty state dict and its adapters resume
from zero-init, which the real ``load_expert_lora_state_dict`` surfaces only as an opaque
``KeyError`` that never names the mismatch. Any layer with expert LoRA matching zero keys must
raise, naming the layers, before anything loads.

Run: pytest tests/cpu/peft/test_ep_lora_apply_miss.py
"""

import sys

import pytest
import torch
import torch.nn as nn

from src.checkpoint.adapters import is_expert_lora_key
from src.distributed.expert_parallel.base_layer import EPMoELayerBase
from src.distributed.expert_parallel.expert_weights import apply_ep_lora_adapters


class _StubEPLoraLayer(EPMoELayerBase):
    """Minimal EP layer with expert LoRA; records what the apply walk hands it."""

    def __init__(self):
        nn.Module.__init__(self)  # skip EPMoELayerBase.__init__ (needs live EP groups)
        self._expert_lora_attrs = frozenset({"gate_up_proj"})
        self.loaded_state = None

    def forward(self, hidden_states, **kwargs):  # pragma: no cover - never called
        return hidden_states

    def load_expert_lora_state_dict(self, layer_state):
        self.loaded_state = dict(layer_state)


def _model_with_ep_layers() -> nn.Module:
    """``model.layers.{0,1}.mlp`` — two EP layers with expert LoRA under an HF-shaped tree."""
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
    """Per-expert ``nn.ModuleList`` families (Bailing/Ling) let stock PEFT adapt the experts directly
    when no EP wrapper is built. Those keys carry ``.experts.`` too, so a substring predicate would
    route them into the EP loader — which has nowhere to put them, and which strips them from the
    attention state so PEFT never sees them either."""
    peft_key = "base_model.model.model.layers.0.mlp.experts.3.gate_proj.lora_A.default.weight"
    assert not is_expert_lora_key(peft_key)
    assert not is_expert_lora_key("base_model.model.model.layers.0.mlp.experts.3.gate_proj.lora_B.weight")
    # The native namespace ends AT the adapter — no tuner suffix.
    assert is_expert_lora_key("model.layers.0.mlp.experts.gate_up_proj.lora_A")
    assert is_expert_lora_key("model.layers.0.mlp.experts.down_proj.lora_B")


def test_expert_keys_with_no_ep_layer_to_receive_them_raise():
    """A checkpoint whose expert adapters nothing can consume must fail loudly: the walk below finds no
    layer to miss, so without this the deltas are dropped and the resume reports success."""
    model = _model_with_ep_layers()
    for layer in model.model.layers:
        layer.mlp._expert_lora_attrs = frozenset()  # a run that builds no expert adapters

    with pytest.raises(RuntimeError, match="no EP layer with expert LoRA"):
        apply_ep_lora_adapters(model, _adapter_keys(0))


def test_all_layers_matched_loads_every_layer():
    model = _model_with_ep_layers()
    apply_ep_lora_adapters(model, {**_adapter_keys(0), **_adapter_keys(1)})
    for layer in model.model.layers:
        assert layer.mlp.loaded_state is not None
        assert set(layer.mlp.loaded_state) == {"experts.gate_up_proj.lora_A", "experts.gate_up_proj.lora_B"}


def test_full_miss_raises():
    """Keys under a foreign path (e.g. an unstripped wrapper prefix) match no layer → raise."""
    model = _model_with_ep_layers()
    foreign = {f"wrapper.{k}": v for k, v in _adapter_keys(0).items()}
    with pytest.raises(RuntimeError, match="matched none of the"):
        apply_ep_lora_adapters(model, foreign)


def test_partial_miss_raises_naming_missed_layer():
    """One layer matched, the other not: must raise (naming the missed layer) instead of silently
    resuming the missed layer's adapters from zero-init — and must not partially load the model."""
    model = _model_with_ep_layers()
    with pytest.raises(RuntimeError, match=r"model\.layers\.1\.mlp"):
        apply_ep_lora_adapters(model, _adapter_keys(0))  # layer 1 gets zero keys
    for layer in model.model.layers:
        assert layer.mlp.loaded_state is None, "a miss must not leave the model partially updated"


def test_empty_adapter_state_is_a_noop():
    model = _model_with_ep_layers()
    apply_ep_lora_adapters(model, {})
    for layer in model.model.layers:
        assert layer.mlp.loaded_state is None


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
