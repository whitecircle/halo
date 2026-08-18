#!/usr/bin/env python
"""Adapter saves must carry the buffers of ``modules_to_save`` clones.

``PeftAdapterSaver._resolve_adapter_state`` feeds PEFT a pre-resolved state dict filtered to
adapter-relevant names. PEFT serializes a ``modules_to_save`` clone via its WHOLE state dict —
parameters and buffers — so a router/gate carrying a balancing buffer (``e_score_correction_bias``,
an adopted GptOss ``router.bias``) would KeyError the lookup at the first ``save_steps`` if the
filter fed parameters only. Pinned on a real CPU PeftModel, no GPU.

    python tests/cpu/peft/test_adapter_state_includes_buffers.py
"""

import pytest
import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model

from src.checkpoint.adapters import cast_adapter_state_to_save_dtype
from src.distributed.checkpoint.peft import PeftAdapterSaver


class _Gate(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(8, 4, bias=False)
        self.register_buffer("e_score_correction_bias", torch.full((4,), 0.5))
        # Non-persistent cache: recomputes on load, must never reach the adapter file.
        self.register_buffer("score_cache", torch.zeros(4), persistent=False)

    def forward(self, x):
        return self.proj(x) + self.e_score_correction_bias


class _Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = nn.Linear(8, 8, bias=False)
        self.gate = _Gate()

    def forward(self, x):
        return self.gate(self.attn(x))


def test_modules_to_save_buffer_survives_adapter_state():
    peft_model = get_peft_model(_Tiny(), LoraConfig(target_modules=["attn"], modules_to_save=["gate"], r=4))
    state = PeftAdapterSaver._resolve_adapter_state(peft_model)
    buffer_keys = [k for k in state if k.endswith("e_score_correction_bias")]
    assert buffer_keys, f"modules_to_save buffer missing from adapter state: {sorted(state)}"
    assert torch.equal(state[buffer_keys[0]], torch.full((4,), 0.5))
    assert any("lora_" in k for k in state), "LoRA tensors must still be present"


def test_non_persistent_buffers_stay_out_of_adapter_state():
    """A non-persistent cache recomputes on load; serializing it would reload it stale on resume —
    the filter feeds PEFT persistent buffers only, like every other checkpoint writer."""
    peft_model = get_peft_model(_Tiny(), LoraConfig(target_modules=["attn"], modules_to_save=["gate"], r=4))
    state = PeftAdapterSaver._resolve_adapter_state(peft_model)
    leaked = [k for k in state if "score_cache" in k]
    assert not leaked, f"non-persistent buffer leaked into the adapter file: {leaked}"


def test_adapter_cast_keeps_balancing_tensors_at_trained_dtype():
    """The balancing export contract reaches the ADAPTER file too: a ``modules_to_save`` router's
    balancing bias stays at trained fp32 while ordinary adapter tensors cast to the save dtype —
    a bf16-rounded bias flips near-tie top-k picks when the exported slot serves."""
    bias_key = "base_model.model.model.layers.0.mlp.gate.modules_to_save.default.e_score_correction_bias"
    state = {
        bias_key: torch.full((4,), 0.5, dtype=torch.float32),
        "attn.lora_A.weight": torch.randn(4, 8, dtype=torch.float32),
    }
    cast = cast_adapter_state_to_save_dtype(state)
    assert cast[bias_key].dtype == torch.float32
    assert cast["attn.lora_A.weight"].dtype == torch.bfloat16, (
        "non-balancing tensor was not cast — the contract test would be vacuous"
    )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
