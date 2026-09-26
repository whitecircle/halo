#!/usr/bin/env python
"""An adapter run refuses ``unfreeze_layers_patterns`` / ``freeze_layers_patterns`` instead of dropping them.

The patterns are applied only on a full or partial fine-tune. An adapter run freezes every base
parameter and trains the adapters alone, so a pattern there would select nothing; an expert-only
LoRA run used to return before looking at them, and an attention-PEFT run only warned. Both must
raise at setup, and the no-adapter path must still apply them.

Run: pytest tests/cpu/peft/test_peft_layer_patterns_refused.py
"""

import types
from unittest.mock import patch

import pytest
import torch.nn as nn
from trl import ModelConfig

from src.distributed.loading import peft_setup
from src.distributed.loading.peft_setup import setup_peft_model

_PATTERN_ARGS = ("unfreeze_layers_patterns", "freeze_layers_patterns")


class _Dense(nn.Module):
    """No EP layers, so ``has_ep_lora`` is False unless a test patches it."""

    def __init__(self):
        super().__init__()
        self.q_proj = nn.Linear(4, 4)

    def forward(self, x):  # pragma: no cover - never called
        return self.q_proj(x)


def _args(**patterns):
    return types.SimpleNamespace(**{name: patterns.get(name) for name in _PATTERN_ARGS})


@pytest.mark.parametrize("pattern_arg", _PATTERN_ARGS)
def test_attention_lora_refuses_layer_patterns(pattern_arg):
    config = ModelConfig(model_name_or_path="dummy/dense", use_peft=True, lora_target_modules=["q_proj"])

    with pytest.raises(ValueError, match=f"{pattern_arg} cannot combine with adapters"):
        setup_peft_model(_args(**{pattern_arg: ["q_proj*"]}), _Dense(), config)


@pytest.mark.parametrize("pattern_arg", _PATTERN_ARGS)
def test_expert_only_lora_refuses_layer_patterns(pattern_arg):
    config = ModelConfig(model_name_or_path="dummy/moe", use_peft=True, lora_target_modules=[])

    with (
        patch.object(peft_setup, "has_ep_lora", return_value=True),
        pytest.raises(ValueError, match=f"{pattern_arg} cannot combine with adapters"),
    ):
        setup_peft_model(_args(**{pattern_arg: ["q_proj*"]}), _Dense(), config)


def test_full_finetune_still_applies_the_patterns():
    config = ModelConfig(model_name_or_path="dummy/dense", use_peft=False)
    model = _Dense()

    assert setup_peft_model(_args(freeze_layers_patterns=["q_proj.weight"]), model, config) is None
    assert not model.q_proj.weight.requires_grad
    assert model.q_proj.bias.requires_grad


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
