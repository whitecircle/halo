"""The zeroed-LoRA-dropout warning gives advice that silences it.

Trainers with ``disable_dropout`` on zero PEFT's ``lora_dropout`` after the adapter wrap. Leaving
``lora_dropout`` out of the YAML does not help: TRL's ``ModelConfig`` defaults it above zero, so
the only config that stops expecting inert regularization is an explicit ``lora_dropout: 0.0``.

    python tests/cpu/peft/test_lora_dropout_live_warning.py
"""

from types import SimpleNamespace

import pytest
import torch.nn as nn
from peft import LoraConfig, get_peft_model
from trl import ModelConfig
from trl.trainer.utils import disable_dropout_in_model

import src.trainers.mixins.validation as validation
from src.trainers.mixins.validation import ParallelismValidationMixin


class _Base(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = nn.Linear(4, 4)

    def forward(self, x):
        return self.q_proj(x)


class _Trainer(ParallelismValidationMixin):
    def __init__(self, lora_dropout: float):
        self.model = get_peft_model(_Base(), LoraConfig(r=2, target_modules=["q_proj"], lora_dropout=lora_dropout))
        disable_dropout_in_model(self.model)
        self.parallelism_config = SimpleNamespace(expert_lora=None)

    def _top_level_model(self):
        return self.model


@pytest.fixture
def warnings(monkeypatch) -> list[str]:
    recorded: list[str] = []
    monkeypatch.setattr(validation, "logger", SimpleNamespace(warning=recorded.append))
    return recorded


def test_omitting_lora_dropout_keeps_a_nonzero_default():
    """Premise of the advice: a recipe that leaves the key out still configures dropout."""
    assert ModelConfig(model_name_or_path="m").lora_dropout > 0


def test_a_zeroed_dropout_warns_with_the_setting_that_silences_it(warnings):
    _Trainer(lora_dropout=0.05)._validate_lora_dropout_live()
    assert len(warnings) == 1
    assert "lora_dropout: 0.0" in warnings[0], warnings[0]


def test_the_advised_setting_silences_the_warning(warnings):
    _Trainer(lora_dropout=0.0)._validate_lora_dropout_live()
    assert warnings == []


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
