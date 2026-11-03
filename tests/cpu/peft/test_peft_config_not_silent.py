#!/usr/bin/env python
"""PEFT knobs the native EP expert adapters cannot honour must fail loud, never half-apply.
``ExpertLoraSpec`` carries only r/alpha/dropout/projections/use_rslora; any other knob would miss the expert half."""

import math
import types
from contextlib import contextmanager
from unittest.mock import patch

import pytest
import torch.nn as nn
from peft import LoraConfig

from halo.configs.script_arguments import ModelArguments
from halo.model_loading.parallel import model_source
from halo.model_loading.parallel import peft_setup as model_info
from halo.model_loading.parallel.peft_setup import configure_adapter_training, split_expert_lora_targets
from halo.models.moe.expert_lora_spec import ExpertLoraSpec
from tests.common.ep_stubs import StubEPLayerBase


@contextmanager
def _as_moe(is_moe: bool = True):
    """Stub the AutoConfig load the peel gates on, so these stay pure logic."""
    with (
        patch.object(model_source.AutoConfig, "from_pretrained", return_value=object()),
        patch.object(model_info, "has_routed_experts", return_value=is_moe),
    ):
        yield


def _model_config(**kwargs) -> ModelArguments:
    return ModelArguments(model_name_or_path="dummy/moe", use_peft=True, **kwargs)


def _args():
    return types.SimpleNamespace(unfreeze_layers_patterns=None, freeze_layers_patterns=None)


class _Dense(nn.Module):
    """A model with no EP layers, so ``model_has_expert_lora`` is False."""

    def __init__(self):
        super().__init__()
        self.q_proj = nn.Linear(4, 4)

    def forward(self, x):  # pragma: no cover - never called
        return self.q_proj(x)


def _model_with_ep_layer() -> nn.Module:
    model = _Dense()
    model.mlp = StubEPLayerBase()
    return model


@pytest.mark.parametrize("spelling", [["experts"], "experts"], ids=["one-entry-list", "bare-string"])
def test_a_lone_expert_target_peels(spelling):
    """Skipping it would leave no expert LoRA, no realized-check to catch that, and a PEFT target matching nothing."""
    config = _model_config(lora_target_modules=spelling)

    with _as_moe():
        spec = split_expert_lora_targets(config)

    assert spec is not None
    assert spec.projections == frozenset({"gate", "up", "down"})
    assert config.lora_target_modules == [], "the peel must consume the lone target"


@pytest.mark.parametrize("target", ["all-linear", "o_proj"])
def test_a_lone_non_expert_target_is_left_to_peft(target):
    """Each survives the peel as the one-entry list given, which PEFT adapts by name."""
    config = _model_config(lora_target_modules=[target])

    with _as_moe():
        assert split_expert_lora_targets(config) is None
    assert config.lora_target_modules == [target]


def test_rslora_reaches_the_expert_scaling():
    """rsLoRA is alpha/sqrt(r); applying it to attention only would run two scalings in one adapter."""
    config = _model_config(lora_target_modules=["q_proj", "gate_up_proj"], lora_r=64, lora_alpha=128, use_rslora=True)

    with _as_moe():
        spec = split_expert_lora_targets(config)

    assert spec.use_rslora is True
    assert spec.scaling == pytest.approx(128 / math.sqrt(64))
    assert spec.scaling != pytest.approx(128 / 64), "rsLoRA must not fall back to alpha/r"


def test_rslora_off_keeps_alpha_over_r():
    config = _model_config(lora_target_modules=["q_proj", "gate_up_proj"], lora_r=64, lora_alpha=128)

    with _as_moe():
        spec = split_expert_lora_targets(config)

    assert spec.use_rslora is False
    assert spec.scaling == pytest.approx(2.0)


def test_dora_with_expert_targets_raises():
    """DoRA has no grouped implementation; it would decompose the attention half alone."""
    config = _model_config(lora_target_modules=["q_proj", "gate_up_proj"], use_dora=True)

    with _as_moe(), pytest.raises(ValueError, match="use_dora"):
        split_expert_lora_targets(config)


def test_dora_without_expert_targets_is_allowed():
    config = _model_config(lora_target_modules=["q_proj", "k_proj"], use_dora=True)

    with _as_moe():
        assert split_expert_lora_targets(config) is None


def test_lora_target_parameters_raises_under_ep():
    """PEFT wraps the EP layer owning the parameter, which lands outside the EP gradient sync and both validators."""
    config = _model_config(lora_target_parameters=["gate_up_proj"])

    with pytest.raises(ValueError, match="lora_target_parameters"):
        configure_adapter_training(_args(), _model_with_ep_layer(), config)


def test_lora_target_parameters_allowed_without_ep_layers():
    """Without EP layers (ep_size<=1 + use_grouped_gemm: false) the experts are ordinary modules."""
    config = _model_config(lora_target_modules=["q_proj"], lora_target_parameters=["gate_up_proj"])

    assert configure_adapter_training(_args(), _Dense(), config) is not None


def test_use_peft_with_no_resulting_adapter_raises():
    """Returning with every param trainable is a silent full fine-tune at the LoRA learning rate."""
    config = _model_config(lora_target_modules=[])
    model = _Dense()

    with pytest.raises(ValueError, match="no adapter would be created"):
        configure_adapter_training(_args(), model, config)


def test_use_peft_false_still_full_finetunes():
    config = ModelArguments(model_name_or_path="dummy/moe", use_peft=False, lora_target_modules=[])
    model = _Dense()

    assert configure_adapter_training(_args(), model, config) is None
    assert all(p.requires_grad for p in model.parameters())


def test_expert_only_run_rejects_modules_to_save():
    """No PeftModel exists in an expert-only run, so modules_to_save would leave the router frozen."""
    config = _model_config(lora_target_modules=[], lora_modules_to_save=["router"])
    model = _Dense()

    with (
        patch.object(model_info, "model_has_expert_lora", return_value=True),
        pytest.raises(ValueError, match="lora_modules_to_save"),
    ):
        configure_adapter_training(_args(), model, config)


def test_expert_only_run_without_modules_to_save_is_fine():
    config = _model_config(lora_target_modules=[])
    model = _Dense()

    with patch.object(model_info, "model_has_expert_lora", return_value=True):
        assert configure_adapter_training(_args(), model, config) is None
    assert not any(p.requires_grad for p in model.parameters()), "the base must be frozen"


# A custom peft_config never passes through the peel.


def test_spec_reports_rslora_split_against_a_live_peft_config():
    """The knob most likely set globally, and the one that splits worst (8x at r=64)."""
    spec = ExpertLoraSpec(r=64, alpha=128)  # peeled without use_rslora
    conflicts = spec.peft_config_conflicts(LoraConfig(r=64, lora_alpha=128, use_rslora=True))

    assert any("use_rslora" in c for c in conflicts)


def test_spec_reports_knobs_with_no_grouped_implementation():
    spec = ExpertLoraSpec(r=8, alpha=16)

    assert any("use_dora" in c for c in spec.peft_config_conflicts(LoraConfig(r=8, use_dora=True)))
    assert any("lora_bias" in c for c in spec.peft_config_conflicts(LoraConfig(r=8, lora_bias=True)))
    assert any(
        "init_lora_weights" in c for c in spec.peft_config_conflicts(LoraConfig(r=8, init_lora_weights="gaussian"))
    )


def test_spec_accepts_a_matching_config():
    spec = ExpertLoraSpec(r=64, alpha=128, dropout=0.05, use_rslora=True)
    matching = LoraConfig(r=64, lora_alpha=128, lora_dropout=0.05, use_rslora=True)

    assert spec.peft_config_conflicts(matching) == []


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
