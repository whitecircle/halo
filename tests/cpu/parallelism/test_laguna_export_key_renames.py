#!/usr/bin/env python
"""Laguna's hub checkpoint spells two modules differently from the live module tree.

``transformers`` declares the difference as two ``WeightRenaming`` entries and applies them inside
``from_pretrained`` only. Both of the toolkit's own key paths therefore have to carry it:

- the EP **gather** writes the live module names, so without the rewrite a gathered/merged checkpoint
  and the online-GRPO weight sync ship ``mlp.shared_experts.*`` and
  ``mlp.gate.e_score_correction_bias``. vLLM keys on hub names and silently skips unknown ones, so
  serving loses the router correction bias;
- the EP **lazy loader** reads safetensors directly and never sees transformers' mapping, so the hub's
  ``mlp.shared_expert.*`` matches no live key and is dropped — the shared expert loads as random
  weights (the shipped ``examples/sft/laguna/laguna-s-2.1-ultrachat-ep.yaml`` is that shape).

``EPMoELayerBase._EXPORT_KEY_RENAMES``-driven rewriting closes both, and
``test_declaration_matches_transformers`` keeps the declaration honest.

    python tests/cpu/parallelism/test_laguna_export_key_renames.py
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModelForCausalLM
from transformers.conversion_mapping import get_checkpoint_conversion_mapping
from transformers.core_model_loading import WeightRenaming

from src.distributed.expert_parallel.base_layer import EPMoELayerBase
from src.distributed.expert_parallel.expert_weights import (
    gather_ep_layer_weights,
    hub_to_module_key_renames,
    to_hub_layer_key,
)
from src.distributed.expert_parallel.layers.laguna import EPLagunaMoELayer
from src.distributed.expert_parallel.lazy_loader import build_family_key_mapping

LAGUNA_MOE_LAYER = 1  # layer 0 is dense (first_k_dense_replace)


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
    """A gatherable layer carrying Laguna's own rename declaration and its replicated module names.

    Built off the base rather than off ``EPLagunaMoELayer`` because that class's ``__init__`` needs
    live EP process groups; the renames under test are read off the class attribute either way.
    """

    _EXPORT_KEY_RENAMES = EPLagunaMoELayer._EXPORT_KEY_RENAMES
    _PER_EXPERT_UNFUSED_KEYS = None  # this stub does its own (trivial) gather

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

    @classmethod
    def merge_shards_to_hf(cls, prefix: str, params: dict) -> dict:  # pragma: no cover - paired override
        return {}


def test_declaration_matches_transformers():
    """The class declaration is a mirror of transformers' ``laguna`` renamings — drift here silently
    reopens both gaps."""
    declared = {hub: module for module, hub in EPLagunaMoELayer._EXPORT_KEY_RENAMES}
    upstream = {
        entry.source_patterns[0].split("mlp.", 1)[-1]: entry.target_patterns[0].split("mlp.", 1)[-1]
        for entry in get_checkpoint_conversion_mapping("laguna")
        if isinstance(entry, WeightRenaming)
    }
    assert upstream, "transformers no longer declares laguna renamings — re-derive the declaration"
    assert declared == upstream, f"declared {declared} != transformers {upstream}"


def test_the_two_spellings_really_differ():
    """Anti-vacuity: the live module tree must actually use the names the gather has to rewrite."""
    keys = set(_tiny_laguna().state_dict())
    prefix = f"model.layers.{LAGUNA_MOE_LAYER}.mlp"
    assert f"{prefix}.gate.e_score_correction_bias" in keys
    assert any(key.startswith(f"{prefix}.shared_experts.") for key in keys)
    assert not any(key.startswith(f"{prefix}.shared_expert.") for key in keys)


def test_gather_emits_hub_spelling():
    """The export side: vLLM and ``from_pretrained`` both key on the hub names."""
    gathered = set(gather_ep_layer_weights("model.layers.1.mlp", _StubLagunaLayer(), retain=True))
    assert "model.layers.1.mlp.experts.e_score_correction_bias" in gathered, gathered
    assert "model.layers.1.mlp.shared_expert.weight" in gathered, gathered
    assert "model.layers.1.mlp.gate.e_score_correction_bias" not in gathered
    assert "model.layers.1.mlp.shared_experts.weight" not in gathered
    # The expert tensors themselves are unaffected — the renames are not a blanket rewrite.
    assert "model.layers.1.mlp.experts.gate_up_proj" in gathered


def test_a_family_without_renames_is_untouched():
    """Anti-over-rejection: every other family's gather keys must be byte-identical to before."""

    class _Plain(_StubLagunaLayer):
        _EXPORT_KEY_RENAMES = ()

    gathered = set(gather_ep_layer_weights("model.layers.1.mlp", _Plain(), retain=True))
    assert "model.layers.1.mlp.gate.e_score_correction_bias" in gathered
    assert "model.layers.1.mlp.shared_experts.weight" in gathered
    assert hub_to_module_key_renames("qwen3_moe") == ()
    assert hub_to_module_key_renames("gpt_oss") == ()


def test_lazy_loader_maps_hub_keys_onto_the_live_module():
    """The load side: an unmapped key is silently skipped by the planner, so the module it belongs to
    is randomly initialized rather than loaded."""
    model = _tiny_laguna()
    prefix = f"model.layers.{LAGUNA_MOE_LAYER}.mlp"
    hub_keys = [
        f"{prefix}.experts.e_score_correction_bias",
        f"{prefix}.shared_expert.gate_proj.weight",
        f"{prefix}.shared_expert.up_proj.weight",
        f"{prefix}.shared_expert.down_proj.weight",
        f"{prefix}.gate.weight",
    ]
    mapping, _ = build_family_key_mapping(model, hub_keys)
    live = set(model.state_dict())
    for hub_key in hub_keys:
        assert mapping[hub_key] in live, f"{hub_key} -> {mapping[hub_key]} matches no live tensor"
    assert mapping[f"{prefix}.experts.e_score_correction_bias"] == f"{prefix}.gate.e_score_correction_bias"
    assert mapping[f"{prefix}.shared_expert.gate_proj.weight"] == f"{prefix}.shared_experts.gate_proj.weight"
    # An already-resolving key must not be rewritten by the fallback.
    assert mapping[f"{prefix}.gate.weight"] == f"{prefix}.gate.weight"


def test_export_and_load_renames_are_inverses():
    for module_spelling, hub_spelling in EPLagunaMoELayer._EXPORT_KEY_RENAMES:
        assert to_hub_layer_key(f"{module_spelling}x", EPLagunaMoELayer) == f"{hub_spelling}x"
    assert dict(hub_to_module_key_renames("laguna")) == {
        hub: module for module, hub in EPLagunaMoELayer._EXPORT_KEY_RENAMES
    }


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
