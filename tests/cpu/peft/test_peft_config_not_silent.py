#!/usr/bin/env python
"""PEFT knobs the native EP expert adapters cannot honour must fail loud, never half-apply.

``ExpertLoraSpec`` carries r/alpha/dropout/projections/use_rslora and nothing else, so every other
``LoraConfig`` knob would reach the attention adapters of a mixed run and silently miss the expert
half. These tests drive the REAL trl ``ModelConfig`` (not a stand-in) because its ``__post_init__``
collapses a one-element ``lora_target_modules`` list to a bare string — the exact input a mock
dataclass cannot reproduce, and the one a list-only peel skips entirely.

Run: pytest tests/cpu/peft/test_peft_config_not_silent.py
"""

import math
import types
from contextlib import contextmanager
from unittest.mock import patch

import pytest
import torch.nn as nn
from peft import LoraConfig
from trl import ModelConfig

from src.distributed.expert_parallel.config import ExpertLoraSpec
from src.distributed.loading import peft_setup as model_info
from src.distributed.loading.peft_setup import setup_peft_model, split_expert_lora_targets
from tests.common.ep_stubs import StubEPLayerBase


@contextmanager
def _as_moe(is_moe: bool = True):
    """The peel loads the model's AutoConfig to gate on MoE — stub it so these stay pure-logic."""
    with (
        patch.object(model_info.AutoConfig, "from_pretrained", return_value=object()),
        patch.object(model_info, "config_has_experts", return_value=is_moe),
    ):
        yield


def _model_config(**kwargs) -> ModelConfig:
    return ModelConfig(model_name_or_path="dummy/moe", use_peft=True, **kwargs)


def _args():
    return types.SimpleNamespace(unfreeze_layers_patterns=None, freeze_layers_patterns=None)


class _Dense(nn.Module):
    """A model with no EP layers, so ``has_ep_lora`` is False."""

    def __init__(self):
        super().__init__()
        self.q_proj = nn.Linear(4, 4)

    def forward(self, x):  # pragma: no cover - never called
        return self.q_proj(x)


def _model_with_ep_layer() -> nn.Module:
    model = _Dense()
    model.mlp = StubEPLayerBase()
    return model


# The one-element list trl collapses to a string


def test_lone_expert_target_still_peels_despite_trl_string_collapse():
    """``lora_target_modules: [experts]`` arrives as the string ``"experts"`` — it must still peel.

    Skipping it left the spec None (no expert LoRA, and no realized-check to catch that) and handed
    PEFT a regex that fullmatches no module."""
    config = _model_config(lora_target_modules=["experts"])
    assert config.lora_target_modules == "experts", "trl no longer collapses; this test's premise moved"

    with _as_moe():
        spec = split_expert_lora_targets(config)

    assert spec is not None
    assert spec.projections == frozenset({"gate", "up", "down"})
    assert config.lora_target_modules == [], "the peel must consume the lone target"


def test_all_linear_sentinel_is_left_alone():
    """``all-linear`` is a PEFT sentinel, not an expert name — it must survive the peel untouched."""
    config = _model_config(lora_target_modules=["all-linear"])

    with _as_moe():
        assert split_expert_lora_targets(config) is None
    assert config.lora_target_modules == "all-linear"


# Knobs that would apply to attention only


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
    """Attention-only DoRA is stock PEFT and must keep working."""
    config = _model_config(lora_target_modules=["q_proj", "k_proj"], use_dora=True)

    with _as_moe():
        assert split_expert_lora_targets(config) is None


def test_lora_target_parameters_raises_under_ep():
    """PEFT wraps the module OWNING the parameter — the EP layer itself, so it lands outside the EP
    gradient sync and outside both EP validators."""
    config = _model_config(lora_target_parameters=["gate_up_proj"])

    with pytest.raises(ValueError, match="lora_target_parameters"):
        setup_peft_model(_args(), _model_with_ep_layer(), config)


def test_lora_target_parameters_allowed_without_ep_wrappers():
    """Without EP wrappers (ep_size<=1 + use_grouped_gemm: false, the accelerate route) the experts are
    ordinary modules and this is stock PEFT working as designed — rejecting it would be a regression."""
    config = _model_config(lora_target_modules=["q_proj"], lora_target_parameters=["gate_up_proj"])

    assert setup_peft_model(_args(), _Dense(), config) is not None


# setup_peft_model: no adapter is an error, not a full fine-tune


def test_use_peft_with_no_resulting_adapter_raises():
    """``use_peft: true`` + an empty target list must raise: returning with every param trainable is
    a full fine-tune at the LoRA learning rate, silently."""
    config = _model_config(lora_target_modules=[])
    model = _Dense()

    with pytest.raises(ValueError, match="no adapter would be created"):
        setup_peft_model(_args(), model, config)


def test_use_peft_false_still_full_finetunes():
    """The raise must not leak into deliberate full fine-tuning."""
    config = ModelConfig(model_name_or_path="dummy/moe", use_peft=False, lora_target_modules=[])
    model = _Dense()

    assert setup_peft_model(_args(), model, config) is None
    assert all(p.requires_grad for p in model.parameters())


def test_expert_only_run_rejects_modules_to_save():
    """No PeftModel exists in an expert-only run, so modules_to_save would leave the router frozen."""
    config = _model_config(lora_target_modules=[], lora_modules_to_save=["router"])
    model = _Dense()

    with (
        patch.object(model_info, "has_ep_lora", return_value=True),
        pytest.raises(ValueError, match="lora_modules_to_save"),
    ):
        setup_peft_model(_args(), model, config)


def test_expert_only_run_without_modules_to_save_is_fine():
    config = _model_config(lora_target_modules=[])
    model = _Dense()

    with patch.object(model_info, "has_ep_lora", return_value=True):
        assert setup_peft_model(_args(), model, config) is None
    assert not any(p.requires_grad for p in model.parameters()), "the base must be frozen"


# The live-config gate: a custom peft_config never passes through the peel


def test_spec_reports_rslora_split_against_a_live_peft_config():
    """The knob a user is most likely to set globally, and the one that splits worst (8x at r=64)."""
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
    """r/alpha/dropout/use_rslora are exactly what the grouped adapters express — no false positives."""
    spec = ExpertLoraSpec(r=64, alpha=128, dropout=0.05, use_rslora=True)
    matching = LoraConfig(r=64, lora_alpha=128, lora_dropout=0.05, use_rslora=True)

    assert spec.peft_config_conflicts(matching) == []


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
