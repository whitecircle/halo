#!/usr/bin/env python
"""Adapter refusals read off a live model with EP layers: each fires on its own condition and names its mechanism."""

import pytest
import torch
from peft import LoraConfig, get_peft_model
from torch import nn

from halo.models.moe.expert_lora_spec import ExpertLoraSpec
from halo.trainers.parallelism.adapter_rules import (
    reject_expert_lora_peft_conflicts,
    reject_peft_lora_in_ep_layers,
    reject_unrealized_expert_lora,
)
from tests.common.ep_stubs import StubEPLayerBase

HIDDEN, EXPERTS, RANK = 8, 2, 2
_SPEC = ExpertLoraSpec(r=RANK, alpha=2 * RANK)


class _MoEHost(nn.Module):
    """An attention projection beside an EP layer's shared-expert ``nn.Linear``, with or without native adapters."""

    def __init__(self, *, native_adapters: bool):
        super().__init__()
        self.q_proj = nn.Linear(HIDDEN, HIDDEN, bias=False)
        self.mlp = StubEPLayerBase()
        self.mlp.shared_expert = nn.Linear(HIDDEN, HIDDEN, bias=False)
        if native_adapters:
            self.mlp._expert_lora_attrs = frozenset({"gate_up_proj"})
            self.mlp.gate_up_proj_lora_A = nn.Parameter(torch.zeros(EXPERTS, HIDDEN, RANK))
            self.mlp.gate_up_proj_lora_B = nn.Parameter(torch.zeros(EXPERTS, RANK, 2 * HIDDEN))

    def forward(self, x):
        return self.q_proj(x)


def _peft(targets: list[str], *, native_adapters: bool = True, **lora):
    return get_peft_model(_MoEHost(native_adapters=native_adapters), LoraConfig(target_modules=targets, **lora))


def test_the_supported_adapter_layouts_pass():
    """Native adapters beside a PEFT attention adapter of a matching recipe, an expert-only run, and no expert LoRA."""
    attention_only = _peft(["q_proj"], r=RANK, lora_alpha=2 * RANK)
    reject_peft_lora_in_ep_layers(attention_only)
    reject_expert_lora_peft_conflicts(attention_only, _SPEC)
    reject_unrealized_expert_lora(attention_only, _SPEC)

    expert_only = _MoEHost(native_adapters=True)
    reject_expert_lora_peft_conflicts(expert_only, _SPEC)
    reject_unrealized_expert_lora(expert_only, _SPEC)
    reject_unrealized_expert_lora(_MoEHost(native_adapters=False), None)


@pytest.mark.parametrize(
    ("refusal", "match"),
    [
        (
            lambda: reject_peft_lora_in_ep_layers(_peft(["shared_expert"])),
            r"PEFT-wrapped LoRA adapters detected inside EP-wrapped MoE layers"
            r"(?s:.*)mlp\.shared_expert\.lora_A\.default\.weight",
        ),
        (
            lambda: reject_expert_lora_peft_conflicts(_peft(["q_proj"], use_rslora=True), _SPEC),
            r"settings the native grouped expert adapters cannot reproduce(?s:.*)use_rslora=True",
        ),
        (
            lambda: reject_expert_lora_peft_conflicts(_peft(["q_proj"], use_dora=True), _SPEC),
            r"settings the native grouped expert adapters cannot reproduce(?s:.*)use_dora=True",
        ),
        (
            lambda: reject_unrealized_expert_lora(_MoEHost(native_adapters=False), _SPEC),
            r"routed to native EP grouped-LoRA .* no EP/grouped MoE layer built the adapters",
        ),
    ],
    ids=["peft-lora-in-ep-layer", "rslora-mismatch", "dora", "unrealized-expert-lora"],
)
def test_each_adapter_refusal_names_its_mechanism(refusal, match):
    with pytest.raises(ValueError, match=match):
        refusal()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
