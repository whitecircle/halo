#!/usr/bin/env python
"""Laguna's hub checkpoint spells two modules differently from the live tree. vLLM silently skips unknown names, so
an unrewritten gather loses the router correction bias at serving."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModelForCausalLM
from transformers.conversion_mapping import get_checkpoint_conversion_mapping
from transformers.core_model_loading import WeightRenaming

from halo.distributed.expert_parallel.families.laguna import EPLagunaMoELayer
from halo.distributed.expert_parallel.moe_layer.layer import EPMoELayerBase
from halo.model_loading.lazy_safetensors.conversion_selection import family_key_mapping
from halo.models.moe.family_specs import MoEFamilySpec

LAGUNA_MOE_LAYER = 1  # Layer 0 is dense (first_k_dense_replace).


def _tiny_laguna():
    config = AutoConfig.for_model("laguna")
    for field, value in (
        ("hidden_size", 64),
        ("intermediate_size", 64),
        ("moe_intermediate_size", 32),
        ("num_hidden_layers", 2),
        ("num_attention_heads", 4),
        ("num_key_value_heads", 2),
        ("head_dim", 16),
        ("vocab_size", 64),
        ("num_experts", 4),
        ("num_experts_per_tok", 2),
        ("shared_expert_intermediate_size", 32),
        ("tie_word_embeddings", False),
    ):
        if hasattr(config, field):
            setattr(config, field, value)
    return AutoModelForCausalLM.from_config(config)


class _StubLagunaLayer(EPMoELayerBase):
    """``EPLagunaMoELayer.__init__`` needs live EP groups; the renames are read off the class attribute either way."""

    FAMILY = MoEFamilySpec(model_types=(), export_key_renames=EPLagunaMoELayer.FAMILY.export_key_renames)

    def __init__(self):
        nn.Module.__init__(self)
        self._expert_lora_attrs = frozenset()
        self.gate_up_proj = nn.Parameter(torch.zeros(2, 4, 6))
        self.gate = nn.Linear(6, 2, bias=False)
        self.gate.register_buffer("e_score_correction_bias", torch.zeros(2), persistent=True)
        self.shared_experts = nn.Linear(6, 6, bias=False)

    def forward(self, hidden_states, **kwargs):  # pragma: no cover - never called
        return hidden_states

    def expert_named_params(self):
        return [("gate_up_proj", self.gate_up_proj)]

    def gather_expert_state_dict(self, device: str = "cpu", merge_lora: bool = False, retain: bool = True) -> dict:
        return {"experts.gate_up_proj": torch.zeros(4, 4, 6, device=device)} if retain else {}


def test_declaration_matches_transformers():
    """Drift here silently reopens both gaps."""
    declared = {hub: module for module, hub in EPLagunaMoELayer.FAMILY.export_key_renames}
    upstream = {
        entry.source_patterns[0].split("mlp.", 1)[-1]: entry.target_patterns[0].split("mlp.", 1)[-1]
        for entry in get_checkpoint_conversion_mapping("laguna")
        if isinstance(entry, WeightRenaming)
    }
    assert upstream, "transformers no longer declares laguna renamings — re-derive the declaration"
    assert declared == upstream, f"declared {declared} != transformers {upstream}"


def test_the_two_spellings_really_differ():
    """Anti-vacuity: the live tree must use the names the gather rewrites."""
    keys = set(_tiny_laguna().state_dict())
    prefix = f"model.layers.{LAGUNA_MOE_LAYER}.mlp"
    assert f"{prefix}.gate.e_score_correction_bias" in keys
    assert any(key.startswith(f"{prefix}.shared_experts.") for key in keys)
    assert not any(key.startswith(f"{prefix}.shared_expert.") for key in keys)


def test_gather_emits_hub_spelling():
    gathered = set(_StubLagunaLayer().gather_layer_state_dict("model.layers.1.mlp", retain=True))
    assert "model.layers.1.mlp.experts.e_score_correction_bias" in gathered, gathered
    assert "model.layers.1.mlp.shared_expert.weight" in gathered, gathered
    assert "model.layers.1.mlp.gate.e_score_correction_bias" not in gathered
    assert "model.layers.1.mlp.shared_experts.weight" not in gathered
    # The renames are no blanket rewrite.
    assert "model.layers.1.mlp.experts.gate_up_proj" in gathered


def test_a_family_without_renames_is_untouched():
    class _Plain(_StubLagunaLayer):
        FAMILY = MoEFamilySpec(model_types=())

    gathered = set(_Plain().gather_layer_state_dict("model.layers.1.mlp", retain=True))
    assert "model.layers.1.mlp.gate.e_score_correction_bias" in gathered
    assert "model.layers.1.mlp.shared_experts.weight" in gathered


def test_lazy_loader_maps_hub_keys_onto_the_live_module():
    """The planner silently skips an unmapped key, leaving the module randomly initialized."""
    model = _tiny_laguna()
    prefix = f"model.layers.{LAGUNA_MOE_LAYER}.mlp"
    hub_keys = [
        f"{prefix}.experts.e_score_correction_bias",
        f"{prefix}.shared_expert.gate_proj.weight",
        f"{prefix}.shared_expert.up_proj.weight",
        f"{prefix}.shared_expert.down_proj.weight",
        f"{prefix}.gate.weight",
    ]
    mapping, _ = family_key_mapping(model, hub_keys)
    live = set(model.state_dict())
    for hub_key in hub_keys:
        assert mapping[hub_key] in live, f"{hub_key} -> {mapping[hub_key]} matches no live tensor"
    assert mapping[f"{prefix}.experts.e_score_correction_bias"] == f"{prefix}.gate.e_score_correction_bias"
    assert mapping[f"{prefix}.shared_expert.gate_proj.weight"] == f"{prefix}.shared_experts.gate_proj.weight"
    assert mapping[f"{prefix}.gate.weight"] == f"{prefix}.gate.weight"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
