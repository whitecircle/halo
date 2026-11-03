#!/usr/bin/env python
"""A mixed attention+expert LoRA adapter must never load as a stock PEFT adapter.
Under a ``LORA`` label PEFT silently drops every expert delta; :data:`MIXED_EXPERT_LORA_PEFT_TYPE` makes it raise."""

from __future__ import annotations

import json
import os
import warnings

import pytest
import torch
from peft import LoraConfig, PeftModel, get_peft_model
from safetensors.torch import load_file, save_file
from torch import nn

from halo.checkpoint.adapters import (
    EXPERT_LORA_PEFT_TYPE,
    EXPERT_LORA_SCALING_FIELDS,
    MIXED_EXPERT_LORA_PEFT_TYPE,
    expert_lora_adapter_config,
    expert_lora_config_fields,
    is_expert_lora_key,
    lora_scaling_mismatch,
    read_adapter_config,
    reject_expert_lora_adapter,
    write_adapter_config,
    write_mixed_adapter_config,
)
from halo.checkpoint.format import ADAPTER_CONFIG_FILE, ADAPTER_SAFETENSORS_FILE
from halo.models.moe.expert_lora_spec import ExpertLoraSpec

# The keys gather_expert_lora_adapters emits under a PeftModel: a PEFT-prefixed path, ending at lora_A/lora_B with no
# tuner suffix because the whole grouped tensor is stored.
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
        write_mixed_adapter_config(peft_config, spec, str(directory))

    state = {name: param.detach() for name, param in peft_model.state_dict().items() if "lora_" in name}
    state[EXPERT_KEY_A] = torch.randn(4, 8, 2)
    state[EXPERT_KEY_B] = torch.randn(4, 2, 8)
    save_file(state, os.path.join(directory, ADAPTER_SAFETENSORS_FILE))

    with open(os.path.join(directory, ADAPTER_CONFIG_FILE)) as fh:
        return json.load(fh)


def test_expert_keys_are_classified_as_expert_adapters():
    assert is_expert_lora_key(EXPERT_KEY_A) and is_expert_lora_key(EXPERT_KEY_B)


def test_stock_peft_cannot_load_the_mixed_adapter(tmp_path):
    _write_mixed_adapter(tmp_path)
    with pytest.raises(KeyError, match=MIXED_EXPERT_LORA_PEFT_TYPE):
        PeftModel.from_pretrained(_TinyLM(), str(tmp_path))


def test_stock_peft_drops_expert_keys_without_warning(tmp_path):
    """Canary: if a future PEFT rejects or warns about unexpected adapter keys, this fails and the marker can go."""
    _write_mixed_adapter(tmp_path, stock_label=True)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        PeftModel.from_pretrained(_TinyLM(), str(tmp_path))  # accepted: no raise on the grouped keys
    noticed = [w for w in caught if any(t in str(w.message).lower() for t in ("expert", "unexpected", "ignor"))]
    assert not noticed, [str(w.message) for w in noticed]


def test_mixed_config_keeps_every_lora_field(tmp_path):
    """Re-typing must not cost the LoRA fields; the directory stays self-describing."""
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
    """A re-save covers the mixed shape; sending the user to a retrain would cost a full run."""
    _write_mixed_adapter(tmp_path)
    with pytest.raises(ValueError, match="MIXED attention\\+expert") as excinfo:
        reject_expert_lora_adapter(str(tmp_path))
    message = str(excinfo.value)
    assert "merge_expert_lora_on_save=True" in message, message
    assert "resume-only" not in message, message
    assert "lora_target_modules" not in message, message


def test_merge_guard_falls_back_to_keys_for_an_unmarked_mixed_adapter(tmp_path):
    """The marker is a fast path; the expert-key scan is the verdict."""
    _write_mixed_adapter(tmp_path, stock_label=True)
    with pytest.raises(ValueError, match="MIXED attention\\+expert"):
        reject_expert_lora_adapter(str(tmp_path))


def test_merge_guard_keeps_the_expert_only_remedy(tmp_path):
    """Same remedy for the expert-only shape; only the marker and the shape description differ."""
    save_file({EXPERT_KEY_A: torch.randn(4, 8, 2)}, os.path.join(tmp_path, ADAPTER_SAFETENSORS_FILE))
    with pytest.raises(ValueError, match="merge_expert_lora_on_save=True"):
        reject_expert_lora_adapter(str(tmp_path))
    with open(os.path.join(tmp_path, ADAPTER_CONFIG_FILE), "w") as fh:
        json.dump({"peft_type": EXPERT_LORA_PEFT_TYPE}, fh)
    with pytest.raises(ValueError, match=EXPERT_LORA_PEFT_TYPE):
        reject_expert_lora_adapter(str(tmp_path))


def test_the_expert_config_records_every_field_a_resume_holds_to_the_live_run():
    assert set(EXPERT_LORA_SCALING_FIELDS) <= set(expert_lora_config_fields(SPEC))


def test_an_expert_only_adapter_config_round_trips_through_the_resume_and_merge_readers(tmp_path):
    write_adapter_config(expert_lora_adapter_config(SPEC, "org/base"), str(tmp_path))

    assert read_adapter_config(str(tmp_path)) == {
        "peft_type": EXPERT_LORA_PEFT_TYPE,
        "base_model_name_or_path": "org/base",
        **expert_lora_config_fields(SPEC),
    }
    live = {"lora_alpha": SPEC.alpha, "use_rslora": SPEC.use_rslora}
    assert lora_scaling_mismatch(str(tmp_path), None, live) is None
    assert "lora_alpha" in lora_scaling_mismatch(str(tmp_path), None, {**live, "lora_alpha": 2 * SPEC.alpha})
    with pytest.raises(ValueError, match=EXPERT_LORA_PEFT_TYPE):
        reject_expert_lora_adapter(str(tmp_path))


def test_attention_only_adapter_still_passes_the_guard(tmp_path):
    peft_model = get_peft_model(_TinyLM(), LoraConfig(target_modules=["q_proj"]))
    peft_model.save_pretrained(str(tmp_path))
    assert not any(is_expert_lora_key(k) for k in load_file(os.path.join(tmp_path, ADAPTER_SAFETENSORS_FILE)))
    reject_expert_lora_adapter(str(tmp_path))
    PeftModel.from_pretrained(_TinyLM(), str(tmp_path))


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
