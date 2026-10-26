#!/usr/bin/env python
"""``prepare_policy_model``: the PEFT wrap and dropout switch the policy-holding trainers share, and the
input-grad hook it leaves to gradient checkpointing.

Under reentrant checkpointing a frozen embedding's output carries no grad path into a checkpointed layer,
so the LoRA weights inside it would get no gradient. ``gradient_checkpointing_enable`` installs the hook
that gives the embedding output ``requires_grad`` on a model whose main input is ``input_ids``, which is
why no trainer installs one of its own.

Run: python tests/cpu/peft/test_policy_preparation.py
"""

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
from peft import LoraConfig, PeftModel
from transformers import Qwen3Config, Qwen3ForCausalLM

from src.model_loading.parallel.peft_setup import prepare_policy_model
from tests.common.models import TINY_QWEN3_CONFIG

LORA_DROPOUT = 0.1


def _model() -> Qwen3ForCausalLM:
    torch.manual_seed(0)
    return Qwen3ForCausalLM(Qwen3Config(**TINY_QWEN3_CONFIG))


def _args() -> SimpleNamespace:
    return SimpleNamespace(gradient_checkpointing=True, bf16=False)


def _lora() -> LoraConfig:
    return LoraConfig(r=2, lora_alpha=4, lora_dropout=LORA_DROPOUT, target_modules=["q_proj", "v_proj"])


def _dropout_rates(model: nn.Module) -> set[float]:
    return {module.p for module in model.modules() if isinstance(module, nn.Dropout)}


@pytest.mark.parametrize("disable_dropout", [True, False])
def test_the_adapter_wrap_comes_before_the_dropout_switch(disable_dropout):
    """Dropout is disabled after the wrap, so the adapters' own dropout goes with the model's."""
    model, casted = prepare_policy_model(_model(), _lora(), _args(), disable_dropout=disable_dropout)

    assert isinstance(model, PeftModel)
    assert casted is False, "only a 4-bit model under bf16 casts its adapters"
    assert _dropout_rates(model) == ({0.0} if disable_dropout else {LORA_DROPOUT})


def test_without_a_peft_config_the_model_is_returned_unwrapped():
    base = _model()

    model, casted = prepare_policy_model(base, None, _args(), disable_dropout=True)

    assert model is base and casted is False


def test_a_peft_model_is_kept_when_existing_adapters_are_not_merged():
    adapted, _ = prepare_policy_model(_model(), _lora(), _args(), disable_dropout=False)

    model, _ = prepare_policy_model(adapted, _lora(), _args(), disable_dropout=False, merge_existing=False)

    assert model is adapted


def _checkpointed_loss(*, keep_input_grad_hook: bool) -> tuple[nn.Module, torch.Tensor]:
    """A LoRA policy's training loss under reentrant gradient checkpointing."""
    model, _ = prepare_policy_model(_model(), _lora(), _args(), disable_dropout=True)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": True})
    if not keep_input_grad_hook:
        model.disable_input_require_grads()
    model.train()
    input_ids = torch.randint(0, TINY_QWEN3_CONFIG["vocab_size"], (2, 8))
    return model, model(input_ids=input_ids, labels=input_ids).loss


def test_gradient_checkpointing_enable_installs_the_hook_the_adapters_train_through():
    model, loss = _checkpointed_loss(keep_input_grad_hook=True)
    loss.backward()
    grads = [param.grad for name, param in model.named_parameters() if "lora_B" in name]

    assert grads and all(grad is not None and grad.abs().sum() > 0 for grad in grads)


def test_without_the_hook_reentrant_checkpointing_leaves_the_loss_no_grad_path():
    """Anti-vacuity: the hook is what the adapters train through."""
    with pytest.warns(UserWarning, match="requires_grad"):
        _, loss = _checkpointed_loss(keep_input_grad_hook=False)

    assert not loss.requires_grad


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
