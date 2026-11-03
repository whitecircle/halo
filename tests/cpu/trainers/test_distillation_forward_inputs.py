#!/usr/bin/env python
"""The distillation forwards must return logits under Liger on the HF trainer's eval step.
``Trainer.prediction_step`` injects ``skip_logits=True``, which Liger's fused forward refuses without labels."""

from __future__ import annotations

import pytest
import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

from halo.trainers.distillation.divergences import logits_forward_inputs
from tests.common.models import TINY_QWEN3_CONFIG


class _LigerContractModel(Qwen3ForCausalLM):
    def forward(self, *args, skip_logits: bool | None = None, **kwargs):
        if skip_logits and kwargs.get("labels") is None:
            raise ValueError("skip_logits is True, but labels and shift_labels are None")
        return super().forward(*args, **kwargs)


def test_the_forward_inputs_drop_the_loss_only_keys_and_pin_the_cache_off():
    inputs = {"input_ids": torch.zeros(1, 4, dtype=torch.long), "labels": torch.zeros(1, 4), "skip_logits": True}
    forwarded = logits_forward_inputs(inputs)
    assert set(forwarded) == {"input_ids", "use_cache"}
    assert forwarded["use_cache"] is False
    assert forwarded["input_ids"] is inputs["input_ids"]
    assert "skip_logits" in inputs  # the trainer's own dict is left alone


def test_the_eval_step_injection_reaches_a_liger_forward_as_a_logits_forward():
    model = _LigerContractModel(Qwen3Config(**TINY_QWEN3_CONFIG)).eval()
    inputs = {
        "input_ids": torch.randint(0, model.config.vocab_size, (1, 6)),
        "attention_mask": torch.ones(1, 6, dtype=torch.long),
        "labels": torch.randint(0, model.config.vocab_size, (1, 6)),
        "skip_logits": True,
    }
    with pytest.raises(ValueError, match="skip_logits"):
        model(**{k: v for k, v in inputs.items() if k != "labels"})
    with torch.no_grad():
        outputs = model(**logits_forward_inputs(inputs))
    assert outputs.logits.shape == (1, 6, model.config.vocab_size)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
