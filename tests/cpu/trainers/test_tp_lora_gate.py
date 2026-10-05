"""Native TP adapters are an SFT-only declaration, with early config and live-layout checks."""

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model

import src.trainers.mixins.validation as validation
from src.trainers.distillation.self_distillation import DistributedSelfDistillationTrainer
from src.trainers.mixins.base import DistributedTrainerMixin
from src.trainers.mixins.validation import ParallelismValidationMixin
from src.trainers.sft import DistributedSFTTrainer


class Gate(ParallelismValidationMixin):
    _supports_tp_lora = True

    def __init__(self, model, **axes):
        self.model = model
        self.parallelism_config = SimpleNamespace(
            **{
                "is_tp_mode": True,
                "tp_size": 2,
                "data_parallel_size": 1,
                "is_ep_mode": False,
                "is_expert_tp_mode": False,
                "is_cp_mode": False,
                "is_pp_mode": False,
                **axes,
            }
        )

    def _top_level_model(self):
        return self.model


def model(*, moe=False):
    base = nn.Module()
    base.q_proj = nn.Linear(8, 8, bias=False)
    base.config = SimpleNamespace(num_experts=4 if moe else 0)
    return base


def test_only_sft_declares_native_tp_lora():
    assert DistributedTrainerMixin._supports_tp_lora is False
    assert DistributedSFTTrainer._supports_tp_lora is True
    assert DistributedSelfDistillationTrainer._supports_tp_lora is False


@pytest.mark.parametrize(
    "axes",
    [
        {"data_parallel_size": 2},
        {"is_ep_mode": True},
        {"is_expert_tp_mode": True},
        {"is_cp_mode": True},
        {"is_pp_mode": True},
    ],
)
def test_native_request_rejects_other_axes_before_injection(axes, monkeypatch):
    called = []
    monkeypatch.setattr(
        validation, "validate_native_tp_lora_config", lambda *args, **kwargs: called.append((args, kwargs))
    )
    gate = Gate(model(), **axes)
    with pytest.raises(ValueError, match="Tensor Parallelism.*tp_size > 1"):
        gate._validate_tp_lora_request(gate.model, LoraConfig(target_modules=["q_proj"]))
    assert not called


def test_native_request_rejects_moe_before_injection():
    gate = Gate(model(moe=True))
    with pytest.raises(ValueError, match="dense SFT"):
        gate._validate_tp_lora_request(gate.model, LoraConfig(target_modules=["q_proj"]))


@pytest.mark.parametrize("postflight", [False, True])
@pytest.mark.parametrize(
    "quantization", ["is_loaded_in_4bit", "is_loaded_in_8bit", "is_quantized", "hf_quantizer", "config", "weights"]
)
def test_native_scope_rejects_quantized_base_with_plain_targets(postflight, quantization, monkeypatch):
    base = model()
    if postflight:
        del base.config
        wrapped = get_peft_model(base, LoraConfig(target_modules=["q_proj"]))
        base.config = SimpleNamespace(num_experts=0)
    else:
        wrapped = base
    if quantization == "config":
        base.config.quantization_config = {"quant_method": "test"}
    elif quantization == "weights":
        base.register_parameter("packed_base", nn.Parameter(torch.zeros(8, dtype=torch.uint8), requires_grad=False))
    else:
        setattr(base, quantization, True)
    assert type(base.q_proj.get_base_layer() if postflight else base.q_proj) is nn.Linear
    monkeypatch.setattr(
        validation, "validate_native_tp_lora_config", lambda *args, **kwargs: pytest.fail("gate bypassed")
    )
    monkeypatch.setattr(
        validation, "validate_native_tp_lora_model", lambda *args, **kwargs: pytest.fail("gate bypassed")
    )
    gate = Gate(wrapped)
    with pytest.raises(ValueError, match="does not support a quantized base"):
        if postflight:
            gate._validate_lora_tp_compatibility()
        else:
            gate._validate_tp_lora_request(base, LoraConfig(target_modules=["q_proj"]))


def test_native_request_validates_config_before_injection(monkeypatch):
    gate = Gate(model())
    config = LoraConfig(target_modules=["q_proj"])
    called = []
    monkeypatch.setattr(
        validation, "validate_native_tp_lora_config", lambda *args, **kwargs: called.append((args, kwargs))
    )
    gate._validate_tp_lora_request(gate.model, config)
    assert called == [((gate.model, config), {"tp_size": 2})]


def test_native_post_wrap_uses_live_layout_validator(monkeypatch):
    base = model()
    del base.config  # PEFT's minimal nn.Module fixture does not need a Transformers config.
    wrapped = get_peft_model(base, LoraConfig(target_modules=["q_proj"]))
    gate = Gate(wrapped)
    called = []
    monkeypatch.setattr(
        validation, "validate_native_tp_lora_model", lambda *args, **kwargs: called.append((args, kwargs))
    )
    gate._validate_lora_tp_compatibility()
    assert called == [((wrapped,), {"tp_size": 2})]
    assert gate._native_tp_lora is True


def test_other_trainer_refusal_keeps_required_wording(monkeypatch):
    base = model()
    del base.config
    gate = Gate(get_peft_model(base, LoraConfig(target_modules=["q_proj"])))
    gate._supports_tp_lora = False
    monkeypatch.setattr(validation, "validate_native_tp_lora_model", lambda *args: pytest.fail("gate bypassed"))
    with pytest.raises(ValueError, match="LoRA/PEFT adapters are not supported with Tensor Parallelism.*tp_size > 1"):
        gate._validate_lora_tp_compatibility()


def test_non_tp_request_does_not_change_lora_support(monkeypatch):
    gate = Gate(model(), is_tp_mode=False)
    monkeypatch.setattr(validation, "validate_native_tp_lora_config", lambda *args: pytest.fail("unexpected TP gate"))
    gate._validate_tp_lora_request(gate.model, LoraConfig(lora_dropout=0.1, target_modules=["q_proj"]))


def test_native_request_needs_preloaded_tp_model():
    gate = Gate("model-id")
    with pytest.raises(ValueError, match="preloaded HF-native TP model"):
        gate._validate_tp_lora_request(gate.model, LoraConfig(target_modules=["q_proj"]))


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
