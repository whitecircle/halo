#!/usr/bin/env python
"""Adapter saves must carry the buffers of ``modules_to_save`` clones.
PEFT serializes a clone's whole state dict, so a parameters-only filter would KeyError at the first ``save_steps``."""

import pytest
import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model

import halo.checkpoint.parallel.adapters as adapters_mod
import halo.runtime.consensus as consensus_mod
import halo.runtime.dtensor as dtensor_mod
from halo.checkpoint.parallel.adapters import gather_adapter_state
from halo.checkpoint.writing import adapter_save_dtype_caster
from halo.models.moe.balancing import is_balancing_state_key
from tests.common.tiny_models import TINY_DENSE_FAMILIES, TINY_MOE_FAMILIES, TINY_MOE_VLM_FAMILIES, tiny_family_model

ALL_FAMILIES = {**TINY_MOE_FAMILIES, **TINY_MOE_VLM_FAMILIES, **TINY_DENSE_FAMILIES}


class _Gate(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(8, 4, bias=False)
        self.register_buffer("e_score_correction_bias", torch.full((4,), 0.5))
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
    state = gather_adapter_state(peft_model)
    buffer_keys = [k for k in state if k.endswith("e_score_correction_bias")]
    assert buffer_keys, f"modules_to_save buffer missing from adapter state: {sorted(state)}"
    assert torch.equal(state[buffer_keys[0]], torch.full((4,), 0.5))
    assert any("lora_" in k for k in state), "LoRA tensors must still be present"


def test_non_persistent_buffers_stay_out_of_adapter_state():
    """Serializing a non-persistent cache would reload it stale on resume."""
    peft_model = get_peft_model(_Tiny(), LoraConfig(target_modules=["attn"], modules_to_save=["gate"], r=4))
    state = gather_adapter_state(peft_model)
    leaked = [k for k in state if "score_cache" in k]
    assert not leaked, f"non-persistent buffer leaked into the adapter file: {leaked}"


def test_off_the_writer_the_state_joins_every_gather_and_keeps_nothing(monkeypatch):
    """A resolve per tensor is a mesh collective; a host copy per local rank is what ``retain`` saves."""
    peft_model = get_peft_model(_Tiny(), LoraConfig(target_modules=["attn"], modules_to_save=["gate"], r=4))
    gathers = {"retained": 0, "dropped": 0}
    real_resolve = dtensor_mod.full_cpu_tensor

    def resolve(tensor):
        gathers["retained"] += 1
        return real_resolve(tensor)

    def materialize(tensor):
        gathers["dropped"] += 1
        return tensor

    monkeypatch.setattr(dtensor_mod, "full_cpu_tensor", resolve)
    monkeypatch.setattr(dtensor_mod, "materialize_dtensor", materialize)
    assert gather_adapter_state(peft_model)
    assert gather_adapter_state(peft_model, retain=False) == {}
    assert gathers["retained"] == gathers["dropped"] > 0


def test_a_failing_peft_lookup_reaches_the_peers_through_a_join(monkeypatch):
    """Only the writer runs the lookup: a bare raise there strands the others in the save's next collective."""
    peft_model = get_peft_model(_Tiny(), LoraConfig(target_modules=["attn"], modules_to_save=["gate"], r=4))
    joined = []
    monkeypatch.setattr(consensus_mod, "reject_across_ranks", lambda reason, *a, **k: joined.append(reason))

    def missing_key(*_args, **_kwargs):
        raise KeyError("gate.modules_to_save.default.bias")

    monkeypatch.setattr(adapters_mod, "get_peft_model_state_dict", missing_key)
    gather_adapter_state(peft_model)
    assert joined == ["KeyError: 'gate.modules_to_save.default.bias'"]
    gather_adapter_state(peft_model, retain=False)
    assert joined[-1] is None, "a peer enters the same join"


def test_adapter_cast_keeps_balancing_tensors_at_trained_dtype():
    """A bf16-rounded balancing bias flips near-tie top-k picks when the exported slot serves."""
    bias_key = "base_model.model.model.layers.0.mlp.gate.modules_to_save.default.e_score_correction_bias"
    state = {
        bias_key: torch.full((4,), 0.5, dtype=torch.float32),
        "attn.lora_A.weight": torch.randn(4, 8, dtype=torch.float32),
    }
    caster = adapter_save_dtype_caster()
    cast = {key: caster(key, t) for key, t in state.items()}
    assert cast[bias_key].dtype == torch.float32
    assert cast["attn.lora_A.weight"].dtype == torch.bfloat16, (
        "non-balancing tensor was not cast — the contract test would be vacuous"
    )


def test_an_adapter_file_keeps_exactly_the_balancing_tensors_a_full_checkpoint_keeps():
    cast = adapter_save_dtype_caster()
    kept_families = set()
    for name, family in ALL_FAMILIES.items():
        torch.manual_seed(0)
        for key, tensor in tiny_family_model(family).state_dict().items():
            module, _, leaf = key.rpartition(".")
            stored = cast(f"base_model.model.{module}.modules_to_save.default.{leaf}", tensor.float())
            expected = torch.float32 if is_balancing_state_key(key) else torch.bfloat16
            assert stored.dtype == expected, (name, key)
            assert cast(f"base_model.model.{module}.lora_A.weight", tensor.float()).dtype == torch.bfloat16
            if expected == torch.float32:
                kept_families.add(name)
    assert {"gpt_oss", "glm4_moe_lite", "laguna", "zaya"} <= kept_families, kept_families


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
