#!/usr/bin/env python
"""EP families with no validated sync into either engine are rejected at trainer construction by
``reject_unsyncable_model``. Otherwise ``sync_weights_to_client`` would fail, or land nowhere, mid-run."""

from __future__ import annotations

from types import SimpleNamespace
from typing import get_args

import pytest
import torch.nn as nn
from transformers import Qwen3Config

import scripts.training.grpo.environmental as environmental_script
from halo.distributed.expert_parallel.families.roster import MOE_LAYER_MAP
from halo.distributed.expert_parallel.moe_layer.layer import EPMoELayerBase
from halo.environments.rollout_config import RolloutBackend
from halo.models.moe.family_specs import MoEFamilySpec
from halo.trainers.grpo.rollout.weight_push import (
    base_push_refusal,
    pushes_base,
    reject_unsyncable_config,
    reject_unsyncable_model,
    serves_family_as_adapter,
)

# Families refused on every engine; each reason is stated at the class declaration. What one engine
# alone cannot serve is that engine client's ``UNSERVABLE_MODEL_TYPES``
# (test_rollout_backend_selection.py).
EXPECTED_UNSUPPORTED = {
    "EPCohere2MoELayer",
    "EPInklingMoELayer",
    "EPInkling2MoELayer",
    "EPNemotronHMoELayer",
    "EPQwen3NextMoELayer",
}


class _EPModuleStub(EPMoELayerBase):
    """Subclasses the real base, the gate's predicate. The base ``__init__`` needs a live process group, so only the
    ``nn.Module`` half is initialized."""

    FAMILY = MoEFamilySpec(model_types=())

    def __init__(self):
        nn.Module.__init__(self)
        self.ep_config = object()

    def forward(self, hidden_states, **kwargs):
        raise NotImplementedError


class _UnsupportedEPModuleStub(_EPModuleStub):
    FAMILY = MoEFamilySpec(model_types=(), weight_sync_refusal="no sync into this stub family is validated")


class _PeftStyleWrapper(nn.Module):
    """PEFT's ``ModulesToSaveWrapper`` forwards ``__getattr__`` to the module it wraps, so a duck-typed ``ep_config``
    probe matches the wrapper. Reachable via ``lora_modules_to_save`` naming an EP-wrapped MoE block."""

    def __init__(self, wrapped: nn.Module):
        super().__init__()
        self.original_module = wrapped

    def __getattr__(self, item):
        try:
            return super().__getattr__(item)
        except AttributeError:
            return getattr(self.original_module, item)


def test_unsupported_family_roster_is_pinned():
    """Read off the EP layer registry: a new family that cannot sync, or a flag cleared on an existing one, must be a
    deliberate edit here."""
    unsupported = {cls.__name__ for cls in set(MOE_LAYER_MAP.values()) if cls.FAMILY.weight_sync_refusal is not None}
    assert unsupported == EXPECTED_UNSUPPORTED


@pytest.mark.parametrize("backend", ["vllm", "sglang"])
def test_gate_rejects_unsupported_ep_family(backend):
    model = nn.Module()
    model.mlp = _UnsupportedEPModuleStub()
    with pytest.raises(ValueError, match="does not support weight sync"):
        reject_unsyncable_model(model, backend)


class _AdapterValidatedEPModuleStub(_EPModuleStub):
    FAMILY = MoEFamilySpec(
        model_types=(),
        weight_sync_refusal="no sync into this stub family is validated",
        adapter_sync_backends=("vllm",),
    )


def test_adapter_sync_passes_a_refused_family_only_on_the_engine_validated_for_it():
    """No base is pushed under adapter sync, so the family's refusal moves to the base push it would need."""
    model = nn.Module()
    model.mlp = _AdapterValidatedEPModuleStub()
    reject_unsyncable_model(model, "vllm", adapter_sync=True)
    assert base_push_refusal(model) == _AdapterValidatedEPModuleStub.FAMILY.weight_sync_refusal
    assert not pushes_base(model, adapter_sync=True) and pushes_base(model, adapter_sync=False)
    with pytest.raises(ValueError, match="does not support weight sync.*rollout_backend: vllm"):
        reject_unsyncable_model(model, "sglang", adapter_sync=True)
    with pytest.raises(ValueError, match="does not support weight sync"):
        reject_unsyncable_model(model, "vllm")
    model.mlp = _EPModuleStub()
    assert base_push_refusal(model) is None and pushes_base(model, adapter_sync=True)


@pytest.mark.parametrize("backend", ["vllm", "sglang"])
def test_adapter_sync_does_not_lift_the_refusal_of_a_family_no_run_validated(backend):
    model = nn.Module()
    model.mlp = _UnsupportedEPModuleStub()
    with pytest.raises(ValueError, match="does not support weight sync"):
        reject_unsyncable_model(model, backend, adapter_sync=True)


def test_every_adapter_sync_backend_a_family_names_is_a_rollout_backend():
    """The gate compares these names with the client's ``BACKEND_KEY``, so a misspelt one refuses the family."""
    named = {backend for cls in set(MOE_LAYER_MAP.values()) for backend in cls.FAMILY.adapter_sync_backends}
    assert named <= set(get_args(RolloutBackend)), (
        f"not rollout backends: {sorted(named - set(get_args(RolloutBackend)))}"
    )


def test_adapter_sync_is_validated_for_inkling2_on_vllm_alone():
    """Read off the registry: a family or engine added here ran its adapter sync against the engine."""
    validated = {
        cls.__name__: cls.FAMILY.adapter_sync_backends
        for cls in set(MOE_LAYER_MAP.values())
        if cls.FAMILY.adapter_sync_backends
    }
    assert validated == {"EPInkling2MoELayer": ("vllm",)}
    for model_type in ("inkling2", "inkling2_text"):
        config = SimpleNamespace(model_type=model_type)
        assert serves_family_as_adapter(config, "vllm") and not serves_family_as_adapter(config, "sglang")
        reject_unsyncable_config(config, "vllm", adapter_sync=True)
        for backend, adapter_sync in (("sglang", True), ("vllm", False)):
            with pytest.raises(ValueError, match="does not support weight sync"):
                reject_unsyncable_config(config, backend, adapter_sync=adapter_sync)
    assert not serves_family_as_adapter(SimpleNamespace(model_type="qwen3_moe"), "vllm")
    assert not serves_family_as_adapter(None, "vllm")


def test_gate_reports_the_offending_module_path():
    model = nn.Module()
    model.layers = nn.ModuleList([_EPModuleStub(), _UnsupportedEPModuleStub()])
    with pytest.raises(ValueError, match=r"layers\.1"):
        reject_unsyncable_model(model, "vllm")


def test_gate_passes_supported_ep_family():
    model = nn.Module()
    model.mlp = _EPModuleStub()
    reject_unsyncable_model(model, "vllm")


def test_gate_ignores_non_ep_modules_declaring_the_flag():
    model = nn.Module()
    model.dense = nn.Linear(4, 4)
    model.dense.FAMILY = _UnsupportedEPModuleStub.FAMILY
    reject_unsyncable_model(model, "vllm")


def test_gate_survives_a_peft_wrapper_around_a_supported_family():
    """A duck-typed probe would raise AttributeError on a supported family: the wrapper has no family flag."""
    model = nn.Module()
    model.mlp = _PeftStyleWrapper(_EPModuleStub())
    reject_unsyncable_model(model, "vllm")


def test_gate_still_rejects_an_unsupported_family_behind_a_peft_wrapper():
    """The wrapped layer is its own child module."""
    model = nn.Module()
    model.mlp = _PeftStyleWrapper(_UnsupportedEPModuleStub())
    with pytest.raises(ValueError, match="does not support weight sync"):
        reject_unsyncable_model(model, "vllm")


@pytest.mark.parametrize(("adapter_sync", "pushed"), [(True, False), (False, True)])
def test_the_script_asks_servers_for_a_push_format_only_where_a_base_is_pushed(monkeypatch, adapter_sync, pushed):
    """Over a base no push carries, a quantized server and a repacking MoE backend are what the run serves from."""
    monkeypatch.setattr(environmental_script, "efforts_by_level", lambda environment, **caps: {})
    model = nn.Module()
    model.mlp = _AdapterValidatedEPModuleStub()
    model.config = Qwen3Config()
    async_config = SimpleNamespace(
        rollout_backend="vllm",
        rollout_adapter_sync=adapter_sync,
        rollout_max_tokens=64,
        rollout_max_thinking_tokens=None,
        rollout_max_answer_tokens=None,
        rollout_max_episode_tokens=None,
        reasoning_floor=0.0,
        routing_replay="off",
        train_on_sampled_tokens=False,
        rollout_weight_precision="bf16",
        score_centering_top_k=None,
    )
    environment = SimpleNamespace(carry_reasoning=False, get_tools_schema=list, TOOLS_ARRIVE_ON_RESET=False)

    assert environmental_script.server_requirements(async_config, environment, model).pushes_base is pushed


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
