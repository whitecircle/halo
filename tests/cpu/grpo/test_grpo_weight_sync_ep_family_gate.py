#!/usr/bin/env python
"""EP families with no validated sync into either engine are rejected at trainer construction by
``reject_unsyncable_model``. Otherwise ``sync_weights_to_client`` would fail, or land nowhere, mid-run."""

from __future__ import annotations

import pytest
import torch.nn as nn

from halo.distributed.expert_parallel.families.roster import MOE_LAYER_MAP
from halo.distributed.expert_parallel.moe_layer.layer import EPMoELayerBase
from halo.models.moe.family_specs import MoEFamilySpec
from halo.trainers.grpo.rollout.weight_push import reject_unsyncable_model

# Families refused on every engine; each reason is stated at the class declaration. What one engine
# alone cannot serve is that engine client's ``UNSERVABLE_MODEL_TYPES``
# (test_rollout_backend_selection.py).
EXPECTED_UNSUPPORTED = {
    "EPCohere2MoELayer",
    "EPGlm5NextMoELayer",
    "EPInklingMoELayer",
    "EPQwen3NextMoELayer",
    "EPQwen4ExpMoELayer",
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


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
