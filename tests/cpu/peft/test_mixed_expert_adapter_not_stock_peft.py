#!/usr/bin/env python
"""A mixed attention+expert LoRA adapter must never load as a stock PEFT adapter.

``PeftAdapterSaver._save_with_expert_lora`` writes ONE adapter file holding stock-PEFT attention
tensors alongside native EP grouped expert tensors (``<layer>.experts.<attr>.lora_{A,B}``), which
PEFT cannot wrap. ``PeftModel.load_adapter`` filters ``missing_keys`` and never reads
``unexpected_keys``, so under a ``peft_type: LORA`` label stock PEFT loads the attention half,
discards every expert delta, and returns a half-adapted model with no warning — verified against
peft 0.18 by ``test_stock_peft_drops_expert_keys_without_warning`` below.

The label is what prevents that: :data:`MIXED_EXPERT_LORA_PEFT_TYPE` is absent from PEFT's config
mapping, so the load raises instead. These tests fail if the writer falls back to
``peft_config.save_pretrained``, if the marker gains a PEFT-known spelling, or if the merge guard
stops recognizing the shape.

Run: pytest tests/cpu/peft/test_mixed_expert_adapter_not_stock_peft.py
"""

from __future__ import annotations

import json
import os
import warnings

import pytest
import torch
from peft import LoraConfig, PeftModel, get_peft_model
from safetensors.torch import load_file, save_file
from torch import nn

from src.checkpoint.adapters import (
    EXPERT_LORA_PEFT_TYPE,
    MIXED_EXPERT_LORA_PEFT_TYPE,
    assert_no_expert_lora_adapter,
    is_expert_lora_key,
)
from src.checkpoint.format import ADAPTER_CONFIG_FILE, ADAPTER_SAFETENSORS_FILE
from src.distributed.checkpoint.peft import PeftAdapterSaver, expert_lora_config_fields
from src.distributed.expert_parallel.config import ExpertLoraSpec

# Exactly the keys gather_ep_lora_adapters emits under a PeftModel: the layer path it walks is
# PEFT-prefixed (unwrap_model stops at the PeftModel), and the key ends at lora_A/lora_B because the
# whole grouped tensor is stored — no tuner suffix. is_expert_lora_key is asserted on them below.
EXPERT_KEY_A = "base_model.model.model.layers.0.mlp.experts.gate_up_proj.lora_A"
EXPERT_KEY_B = "base_model.model.model.layers.0.mlp.experts.gate_up_proj.lora_B"

SPEC = ExpertLoraSpec(r=8, alpha=16, dropout=0.05, projections=frozenset({"gate", "up"}), use_rslora=True)


class _TinyLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = nn.Linear(8, 8, bias=False)

    def forward(self, x):
        return self.q_proj(x)


def _write_mixed_adapter(directory, *, spec=SPEC, stock_label: bool = False) -> dict:
    """Write a mixed adapter dir the way the saver does, and return its config dict."""
    peft_model = get_peft_model(_TinyLM(), LoraConfig(r=8, lora_alpha=16, target_modules=["q_proj"]))
    peft_config = peft_model.peft_config["default"]
    if stock_label:
        peft_config.save_pretrained(str(directory))  # the stock-PEFT writer, labelling the dir ``LORA``
    else:
        PeftAdapterSaver._write_mixed_adapter_config(peft_config, spec, str(directory))

    state = {name: param.detach() for name, param in peft_model.state_dict().items() if "lora_" in name}
    state[EXPERT_KEY_A] = torch.randn(4, 8, 2)
    state[EXPERT_KEY_B] = torch.randn(4, 2, 8)
    save_file(state, os.path.join(directory, ADAPTER_SAFETENSORS_FILE))

    with open(os.path.join(directory, ADAPTER_CONFIG_FILE)) as fh:
        return json.load(fh)


def test_expert_keys_are_classified_as_expert_adapters():
    """The premise of every assertion below: these keys are the native-expert namespace."""
    assert is_expert_lora_key(EXPERT_KEY_A) and is_expert_lora_key(EXPERT_KEY_B)


def test_stock_peft_cannot_load_the_mixed_adapter(tmp_path):
    """The load raises instead of silently returning an attention-only model."""
    _write_mixed_adapter(tmp_path)
    with pytest.raises(KeyError, match=MIXED_EXPERT_LORA_PEFT_TYPE):
        PeftModel.from_pretrained(_TinyLM(), str(tmp_path))


def test_stock_peft_drops_expert_keys_without_warning(tmp_path):
    """Canary for the failure being prevented — PEFT's own behavior under a stock ``LORA`` label.

    If a future PEFT starts rejecting or warning about unexpected adapter keys, this fails and the
    marker can be reconsidered. Until then it is the reason the marker exists.
    """
    _write_mixed_adapter(tmp_path, stock_label=True)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        PeftModel.from_pretrained(_TinyLM(), str(tmp_path))  # accepted: no raise on the grouped keys
    noticed = [w for w in caught if any(t in str(w.message).lower() for t in ("expert", "unexpected", "ignor"))]
    assert not noticed, [str(w.message) for w in noticed]


def test_mixed_config_keeps_every_lora_field(tmp_path):
    """Re-typing must not cost the LoRA fields — the directory stays self-describing."""
    config = _write_mixed_adapter(tmp_path)
    assert config["r"] == 8
    assert config["lora_alpha"] == 16
    assert set(config["target_modules"]) == {"q_proj"}
    assert config["peft_type"] == MIXED_EXPERT_LORA_PEFT_TYPE


def test_mixed_config_records_the_expert_half(tmp_path):
    """The expert adapters have their own r/alpha/rslora/projections; scaling depends on all of them."""
    config = _write_mixed_adapter(tmp_path)
    assert config["ep_expert_lora"] == expert_lora_config_fields(SPEC)
    assert config["ep_expert_lora"]["use_rslora"] is True
    assert config["ep_expert_lora"]["expert_projections"] == ["gate", "up"]


def test_merge_guard_reports_the_mixed_shape(tmp_path):
    """The message must name the shape AND the save that folds it.

    ``merge_expert_lora_on_save`` covers the mixed shape too — ``save_model`` routes it past
    ``PeftAdapterSaver`` to ``save_ep_checkpoint``, which folds the expert deltas in the family
    gather and the attention deltas via ``merged_adapters``
    (``tests/cpu/peft/test_merge_expert_lora_save_guard.py`` pins that routing). Calling the shape
    resume-only and sending the user back to a retrain with expert-only ``lora_target_modules``
    costs a full run for what a re-save produces.
    """
    _write_mixed_adapter(tmp_path)
    with pytest.raises(ValueError, match="MIXED attention\\+expert") as excinfo:
        assert_no_expert_lora_adapter(str(tmp_path))
    message = str(excinfo.value)
    assert "merge_expert_lora_on_save=True" in message, message
    assert "resume-only" not in message, message
    assert "lora_target_modules" not in message, message


def test_merge_guard_falls_back_to_keys_for_an_unmarked_mixed_adapter(tmp_path):
    """An adapter whose config carries a stock peft_type is still classified by SHAPE, not waved
    through: the marker is a fast path, the expert-key scan is the verdict."""
    _write_mixed_adapter(tmp_path, stock_label=True)
    with pytest.raises(ValueError, match="MIXED attention\\+expert"):
        assert_no_expert_lora_adapter(str(tmp_path))


def test_merge_guard_keeps_the_expert_only_remedy(tmp_path):
    """Same remedy for the expert-only shape; only the marker and the shape description differ."""
    save_file({EXPERT_KEY_A: torch.randn(4, 8, 2)}, os.path.join(tmp_path, ADAPTER_SAFETENSORS_FILE))
    with pytest.raises(ValueError, match="merge_expert_lora_on_save=True"):
        assert_no_expert_lora_adapter(str(tmp_path))
    with open(os.path.join(tmp_path, ADAPTER_CONFIG_FILE), "w") as fh:
        json.dump({"peft_type": EXPERT_LORA_PEFT_TYPE}, fh)
    with pytest.raises(ValueError, match=EXPERT_LORA_PEFT_TYPE):
        assert_no_expert_lora_adapter(str(tmp_path))


def test_attention_only_adapter_still_passes_the_guard(tmp_path):
    """The guard must not start refusing ordinary LoRA runs."""
    peft_model = get_peft_model(_TinyLM(), LoraConfig(target_modules=["q_proj"]))
    peft_model.save_pretrained(str(tmp_path))
    assert not any(is_expert_lora_key(k) for k in load_file(os.path.join(tmp_path, ADAPTER_SAFETENSORS_FILE)))
    assert_no_expert_lora_adapter(str(tmp_path))
    PeftModel.from_pretrained(_TinyLM(), str(tmp_path))


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
