#!/usr/bin/env python
"""The distillation forwards must return logits under Liger on the HF trainer's eval step.

transformers' ``Trainer.prediction_step`` injects ``skip_logits=True`` into loss-only eval inputs
whenever ``use_liger_kernel`` is on; the distillation objectives build their own loss from
``outputs.logits`` and forward the inputs WITHOUT labels, where Liger's fused linear-cross-entropy
forward refuses ``skip_logits`` ("labels and shift_labels are None"). Both trainers derive their
forward inputs from :func:`logits_forward_inputs`, which is pinned here against that contract.

Run: ``pytest -m cpu tests/cpu/trainers/test_distillation_forward_inputs.py``
"""

from __future__ import annotations

import pytest
import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

from src.trainers.distillation.losses import logits_forward_inputs
from tests.common.models import TINY_QWEN3_CONFIG


class _LigerContractModel(Qwen3ForCausalLM):
    """A forward with Liger's fused-LCE contract: ``skip_logits`` without labels is an error."""

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
