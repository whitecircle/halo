#!/usr/bin/env python
"""``prepare_policy_model``: the PEFT wrap, the dropout switch, and the input-grad hook left to gradient checkpointing.
Under reentrant checkpointing LoRA weights inside a checkpointed layer take their gradient with the hook or without. A
held adapter merges before the new wrap, except one splitting a tied embedding pair, which merging would fold into
both halves."""

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import Qwen3Config, Qwen3ForCausalLM

from halo.model_loading.parallel.peft_setup import prepare_policy_model
from halo.models.peft_adapters import adapter_base_module
from tests.common.adapters import randomize_adapters
from tests.common.models import TINY_QWEN3_CONFIG

LORA_DROPOUT = 0.1
# Each makes the two halves of a tied embedding pair compute with different weights.
HELD_ON_A_HALF = {
    "lora_head": {"target_modules": ["lm_head"]},
    "lora_embedding": {"target_modules": ["embed_tokens"]},
    "full_head": {"target_modules": ["q_proj"], "modules_to_save": ["lm_head"]},
}


def _model(*, tied: bool = False) -> Qwen3ForCausalLM:
    torch.manual_seed(0)
    return Qwen3ForCausalLM(Qwen3Config(**{**TINY_QWEN3_CONFIG, "tie_word_embeddings": tied}))


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


def _held(adapter: str, *, tied: bool) -> PeftModel:
    """A policy holding a trained ``adapter``: adapters and full copies moved off their init."""
    held = get_peft_model(_model(tied=tied), LoraConfig(r=2, lora_alpha=4, **HELD_ON_A_HALF[adapter]))
    randomize_adapters(held, std=0.1)
    with torch.no_grad():
        for name, param in held.named_parameters():
            if "modules_to_save" in name:
                param.add_(torch.randn(param.shape, generator=torch.Generator().manual_seed(1)))
    return held.eval()


def _logits(model: nn.Module) -> torch.Tensor:
    with torch.no_grad():
        return model(input_ids=torch.arange(16)[None]).logits


@pytest.mark.parametrize("adapter", sorted(HELD_ON_A_HALF))
def test_merging_a_held_adapter_that_splits_a_tied_pair_is_refused(adapter):
    held = _held(adapter, tied=True)
    shared = adapter_base_module(held.get_base_model().model.embed_tokens).weight
    before = shared.detach().clone()

    with pytest.raises(ValueError, match="merge_and_unload would fold it"):
        prepare_policy_model(held, _lora(), _args(), disable_dropout=False)
    assert torch.equal(shared, before), "the refused merge wrote the shared weight"


@pytest.mark.parametrize("adapter", sorted(HELD_ON_A_HALF))
def test_a_held_adapter_on_an_untied_base_merges_before_the_new_wrap(adapter):
    held = _held(adapter, tied=False)
    trained = _logits(held)

    model, _ = prepare_policy_model(held, _lora(), _args(), disable_dropout=False)

    assert isinstance(model, PeftModel) and set(model.peft_config) == {"default"}
    # The new adapter starts at a zero lora_B, so the wrapped model computes the merged policy.
    torch.testing.assert_close(_logits(model.eval()), trained, rtol=1e-5, atol=1e-4)


def test_a_held_adapter_on_a_tied_base_is_kept_when_existing_adapters_are_not_merged():
    held = _held("lora_head", tied=True)

    model, _ = prepare_policy_model(held, _lora(), _args(), disable_dropout=False, merge_existing=False)

    assert model is held


def _checkpointed_loss(*, keep_input_grad_hook: bool) -> tuple[nn.Module, torch.Tensor]:
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


def test_without_the_hook_the_adapters_train_through_their_layers_own_input():
    model, loss = _checkpointed_loss(keep_input_grad_hook=False)
    loss.backward()
    grads = [param.grad for name, param in model.named_parameters() if "lora_B" in name]

    assert grads and all(grad is not None and grad.abs().sum() > 0 for grad in grads)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
