"""FSDP2 swaps a wrapped module's class for a dynamic subclass, which transformers' output capture, keyed on the class,
does not know. The trainer registers each one; without it a router-logits capture comes back empty and the aux loss is
the int ``load_balancing_loss_func`` returns for no logits."""

import pytest
import torch
from transformers import Qwen3MoeConfig, Qwen3MoeForCausalLM

from halo.trainers.parallelism.mode_setup import ModeSetupMixin
from tests.common.models import TINY_QWEN3_MOE_CONFIG


class _Trainer(ModeSetupMixin):
    def __init__(self, model):
        self._model = model

    def _backbone_model(self):
        return self._model


def _tiny_model_as_fsdp2_wraps_it(tag: str) -> Qwen3MoeForCausalLM:
    """A tiny Qwen3-MoE whose base model carries the class swap ``fully_shard`` makes: an ``FSDP<class>`` subclass."""
    model = Qwen3MoeForCausalLM(Qwen3MoeConfig(**TINY_QWEN3_MOE_CONFIG, pad_token_id=0, eos_token_id=1))
    base = type(model.model)
    model.model.__class__ = type(f"FSDP{tag}{base.__name__}", (base,), {})
    return model


def test_a_class_fsdp2_swapped_in_captures_router_logits_once_registered():
    token_ids = torch.randint(2, TINY_QWEN3_MOE_CONFIG["vocab_size"], (1, 8))
    unregistered = _tiny_model_as_fsdp2_wraps_it("Unregistered")
    assert unregistered(input_ids=token_ids, output_router_logits=True).router_logits is None

    model = _tiny_model_as_fsdp2_wraps_it("Registered")
    _Trainer(model)._register_output_capturing_for_fsdp()
    outputs = model(input_ids=token_ids, labels=token_ids, output_router_logits=True)
    assert len(outputs.router_logits) == TINY_QWEN3_MOE_CONFIG["num_hidden_layers"]
    assert isinstance(outputs.aux_loss, torch.Tensor)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
