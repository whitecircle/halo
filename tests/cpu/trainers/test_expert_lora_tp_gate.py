#!/usr/bin/env python
"""Under TP, every kind of adapter is refused — native EP expert LoRA included.

``_validate_lora_tp_compatibility`` must not test only the adapters PEFT owns — an ``isinstance``
check plus ``has_non_expert_lora``, which excludes EP expert params **by identity** precisely so the
TP replicated-grad sweep can skip them. EP's native grouped expert adapters pass every such gate:
not PEFT-wrapped, not counted as non-expert LoRA, excluded from the TP grad sweep — an expert-only
LoRA run would then train and export on a path with no equivalence gate, no save/merge test and no
doc claiming it works. It is refused at construction, on the allowlist discipline the parallelism
axes use.

    python tests/cpu/trainers/test_expert_lora_tp_gate.py
"""

import types

import pytest
import torch.nn as nn
from peft import LoraConfig, PeftModel, get_peft_model

import src.trainers.mixins.validation as validation_mod
from src.trainers.mixins.base import DistributedTrainerMixin
from src.trainers.mixins.validation import has_non_expert_lora

_validate = DistributedTrainerMixin._validate_lora_tp_compatibility


class _TinyLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = nn.Linear(8, 8, bias=False)


def _fake_self(model):
    # Borrowed, not reimplemented: a stub unwrap could pass while the real one is broken.
    me = types.SimpleNamespace(model=model)
    me._top_level_model = types.MethodType(DistributedTrainerMixin._top_level_model, me)
    return me


def test_native_expert_lora_is_rejected_under_tp(monkeypatch):
    model = _TinyLM()
    # Premise: this is exactly the shape both existing predicates wave through.
    assert not isinstance(model, PeftModel)
    assert not has_non_expert_lora(model)
    monkeypatch.setattr(validation_mod, "has_ep_lora", lambda m: True)

    with pytest.raises(ValueError, match="expert LoRA is not supported with Tensor Parallelism"):
        _validate(_fake_self(model))


def test_a_full_finetune_under_tp_is_not_refused(monkeypatch):
    """No adapters of either kind: the gate must stay silent, or every EP+TP full fine-tune dies."""
    monkeypatch.setattr(validation_mod, "has_ep_lora", lambda m: False)

    assert _validate(_fake_self(_TinyLM())) is None


def test_attention_lora_under_tp_still_names_its_own_mechanism(monkeypatch):
    """The pre-existing arm, unchanged: a PEFT-wrapped run is rejected for the DTensor-graph reason,
    not swallowed by the new expert-LoRA branch."""
    monkeypatch.setattr(validation_mod, "has_ep_lora", lambda m: False)
    peft_model = get_peft_model(_TinyLM(), LoraConfig(r=4, target_modules=["q_proj"]))

    with pytest.raises(ValueError, match="LoRA/PEFT adapters are not supported with Tensor Parallelism"):
        _validate(_fake_self(peft_model))


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
