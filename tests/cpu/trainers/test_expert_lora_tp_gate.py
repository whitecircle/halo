#!/usr/bin/env python
"""Under TP, every kind of adapter is refused — native EP expert LoRA included.

``reject_lora_under_tp`` must not test only the adapters PEFT owns — an ``isinstance``
check plus the tuner layers' own parameters (``tuner_adapter_param_ids``), and the TP replicated-grad
sweep skips EP expert params **by identity**. EP's native grouped expert adapters pass every such
gate: not PEFT-wrapped, no tuner layer, excluded from the TP grad sweep — an expert-only LoRA run
would then train and export on a path with no equivalence gate, no save/merge test and no doc
claiming it works. It is refused at construction, on the allowlist discipline the parallelism axes
use, and the trainer's construction spine runs the refusal before any wrap. Adapters injected in place
count by their tuner layers, not by a ``lora_`` name: a backbone's own ``lora_*`` parameter is a base
weight a TP full fine-tune may train.

    python tests/cpu/trainers/test_expert_lora_tp_gate.py
"""

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
import torch.nn as nn
from peft import IA3Config, LoraConfig, PeftModel, get_peft_model, inject_adapter_in_model

import halo.trainers.parallelism.adapter_rules as validation_mod
from halo.models.peft_adapters import tuner_adapter_param_ids
from halo.trainers.distributed_trainer import DistributedTrainerMixin
from halo.trainers.parallelism.adapter_rules import reject_lora_under_tp
from tests.common.parallelism import make_parallelism_config


class _TinyLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = nn.Linear(8, 8, bias=False)


class _NativeLoraBackbone(_TinyLM):
    """A remote-code backbone whose own weights are named like adapters (jina's LoRA parametrizations)."""

    def __init__(self):
        super().__init__()
        self.lora_A = nn.Parameter(torch.zeros(4, 8))
        self.lora_B = nn.Parameter(torch.zeros(8, 4))


def test_native_expert_lora_is_rejected_under_tp(monkeypatch):
    model = _TinyLM()
    # Premise: this is exactly the shape both existing predicates wave through.
    assert not isinstance(model, PeftModel)
    assert not tuner_adapter_param_ids(model)
    monkeypatch.setattr(validation_mod, "model_has_expert_lora", lambda m: True)

    with pytest.raises(ValueError, match="expert LoRA is not supported with Tensor Parallelism"):
        reject_lora_under_tp(model)


def test_a_full_finetune_under_tp_is_not_refused(monkeypatch):
    """No adapters of either kind: the gate must stay silent, or every EP+TP full fine-tune dies."""
    monkeypatch.setattr(validation_mod, "model_has_expert_lora", lambda m: False)

    assert reject_lora_under_tp(_TinyLM()) is None


def test_attention_lora_under_tp_still_names_its_own_mechanism(monkeypatch):
    """A PEFT-wrapped run is rejected for the DTensor-graph reason, not swallowed by the expert-LoRA
    branch."""
    monkeypatch.setattr(validation_mod, "model_has_expert_lora", lambda m: False)
    peft_model = get_peft_model(_TinyLM(), LoraConfig(r=4, target_modules=["q_proj"]))

    with pytest.raises(ValueError, match="LoRA/PEFT adapters are not supported with Tensor Parallelism"):
        reject_lora_under_tp(peft_model)


def test_a_backbones_own_lora_named_weights_are_not_adapters(monkeypatch):
    """A TP full fine-tune of a backbone carrying native ``lora_*`` parameters is not a LoRA run."""
    monkeypatch.setattr(validation_mod, "model_has_expert_lora", lambda m: False)
    model = _NativeLoraBackbone()
    assert any("lora_" in name for name, _param in model.named_parameters()), "premise: adapter-like names"

    assert reject_lora_under_tp(model) is None


@pytest.mark.parametrize(
    "config",
    [LoraConfig(r=4, target_modules=["q_proj"]), IA3Config(target_modules=["q_proj"], feedforward_modules=[])],
    ids=["lora", "ia3"],
)
def test_adapters_injected_in_place_are_refused_under_tp(monkeypatch, config):
    """The embedding path injects adapters without a PeftModel; their tuner layers still count, whatever
    their parameters are named ((IA)3's ``ia3_l`` carries no ``lora_``)."""
    monkeypatch.setattr(validation_mod, "model_has_expert_lora", lambda m: False)
    model = inject_adapter_in_model(config, _TinyLM())
    assert not isinstance(model, PeftModel), "premise: injected in place, not wrapped"

    with pytest.raises(ValueError, match="LoRA/PEFT adapters are not supported with Tensor Parallelism"):
        reject_lora_under_tp(model)


class _PastTpGuard(Exception):
    """Carries control out of the construction spine at the step after the TP adapter refusal."""


class _ConstructionHost(DistributedTrainerMixin):
    """The mixin's real post-``__init__`` construction spine over a bare model, stopped one step past the refusal."""

    def __init__(self, model, **parallelism):
        self.model = model
        self.args = SimpleNamespace(optim="adamw_torch")
        self._bf16_optimizer = False
        with patch("halo.distributed.parallelism_config.validate_nvlink_domain_against_fabric"):
            self.parallelism_config = make_parallelism_config(world_size=2, gpus_per_node=2, **parallelism)

    def _restore_trl_overrides(self):
        """No TRL constructor ran, so there is nothing to take back."""

    def _disable_inert_router_logits(self):
        raise _PastTpGuard


@pytest.mark.parametrize(
    ("parallelism", "error", "fragment"),
    [
        ({"tp_size": 2}, ValueError, "LoRA/PEFT adapters are not supported with Tensor Parallelism"),
        ({}, _PastTpGuard, None),
    ],
    ids=["tp", "dp"],
)
def test_trainer_construction_refuses_lora_under_tp_only(parallelism, error, fragment):
    """The pins above hold only while the trainer calls the refusal: a LoRA model under TP is refused before
    any wrap, and the same model on plain data parallelism passes it."""
    host = _ConstructionHost(get_peft_model(_TinyLM(), LoraConfig(r=4, target_modules=["q_proj"])), **parallelism)

    with patch("halo.trainers.distributed_trainer.agree_rank_uniform_env"), pytest.raises(error, match=fragment):
        DistributedTrainerMixin._configure_distributed_modes(host)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
