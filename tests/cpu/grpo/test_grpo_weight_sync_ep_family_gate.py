#!/usr/bin/env python
"""EP families no engine can load must be rejected at trainer construction, not at the first sync.

``sync_weights_to_client`` forwards trainer parameter names straight into the engine's
``model.load_weights``, so a family whose served implementation uses a different checkpoint
namespace than the HuggingFace module tree the trainer trains has nowhere to put the weights.
Inkling is that family: its hub keeps Thinking Machines' namespace (``model.llm.*``, ``wq_du``,
interleaved ``w13_weight``) that only a from_pretrained conversion maps onto the module tree, so
the module-spelled names the sync sends land nowhere on either engine.
``validate_weight_sync_support`` is the construction gate.

Run: ``python tests/cpu/grpo/test_grpo_weight_sync_ep_family_gate.py`` (or ``pytest -m cpu``).
"""

from __future__ import annotations

import pytest
import torch.nn as nn

from src.distributed.expert_parallel.base_layer import EPMoELayerBase
from src.distributed.expert_parallel.patching import MOE_LAYER_MAP
from src.trainers.grpo.rollout.weight_sync import validate_weight_sync_support

# Families whose gathered export no engine can load under the names the sync sends; each reason is
# stated at the class declaration. What one engine alone cannot serve is that engine client's
# ``UNSERVABLE_MODEL_TYPES`` (test_rollout_backend_selection.py); Step-3.7 sends hub names
# (test_weight_sync_hub_namespace.py) and is not here.
EXPECTED_UNSUPPORTED = {
    "EPCohere2MoELayer",
    "EPGlm5NextMoELayer",
    "EPInklingMoELayer",
}


class _EPModuleStub(EPMoELayerBase):
    """Stands in for an EP-wrapped MoE block. Subclasses the real base (the gate's predicate) and
    takes the family flag from its type, exactly as a real wrapper does; the base ``__init__`` needs
    a live process group, so only the ``nn.Module`` half is initialized."""

    def __init__(self):
        nn.Module.__init__(self)
        self.ep_config = object()

    def forward(self, hidden_states, **kwargs):
        raise NotImplementedError


class _UnsupportedEPModuleStub(_EPModuleStub):
    _supports_weight_sync = False


class _PeftStyleWrapper(nn.Module):
    """PEFT's ``ModulesToSaveWrapper`` forwards ``__getattr__`` to the module it wraps, so a
    duck-typed ``ep_config`` probe matches the WRAPPER while the flag resolves off the wrapper's own
    type. Reachable via ``lora_modules_to_save`` naming an EP-wrapped MoE block."""

    def __init__(self, wrapped: nn.Module):
        super().__init__()
        self.original_module = wrapped

    def __getattr__(self, item):
        try:
            return super().__getattr__(item)
        except AttributeError:
            return getattr(self.original_module, item)


def test_unsupported_family_roster_is_pinned():
    """Read off the EP wrapper registry, not a hand-maintained list: a new family that cannot sync,
    or a flag cleared on an existing one, must be a deliberate edit here with the evidence in the
    declaring class."""
    unsupported = {cls.__name__ for cls in set(MOE_LAYER_MAP.values()) if cls._supports_weight_sync is False}
    assert unsupported == EXPECTED_UNSUPPORTED


@pytest.mark.parametrize("backend", ["vllm", "sglang"])
def test_gate_rejects_unsupported_ep_family(backend):
    """The family flag is engine-independent: what no gather can spell lands on neither engine."""
    model = nn.Module()
    model.mlp = _UnsupportedEPModuleStub()
    with pytest.raises(ValueError, match="does not support weight sync"):
        validate_weight_sync_support(model, backend)


def test_gate_reports_the_offending_module_path():
    """The message must name where the module sits — a multi-layer model gives no other clue."""
    model = nn.Module()
    model.layers = nn.ModuleList([_EPModuleStub(), _UnsupportedEPModuleStub()])
    with pytest.raises(ValueError, match=r"layers\.1"):
        validate_weight_sync_support(model, "vllm")


def test_gate_passes_supported_ep_family():
    model = nn.Module()
    model.mlp = _EPModuleStub()
    validate_weight_sync_support(model, "vllm")  # must not raise


def test_gate_ignores_non_ep_modules_declaring_the_flag():
    """A plain module that happens to carry the attribute is not an EP wrapper and must not trip the
    gate."""
    model = nn.Module()
    model.dense = nn.Linear(4, 4)
    model.dense._supports_weight_sync = False
    validate_weight_sync_support(model, "vllm")  # must not raise


def test_gate_survives_a_peft_wrapper_around_a_supported_family():
    """The wrapper forwards ``ep_config`` but its own type has no family flag: a duck-typed probe
    would raise AttributeError on a family that IS supported."""
    model = nn.Module()
    model.mlp = _PeftStyleWrapper(_EPModuleStub())
    validate_weight_sync_support(model, "vllm")  # must not raise


def test_gate_still_rejects_an_unsupported_family_behind_a_peft_wrapper():
    """Skipping the wrapper must not lose the rejection — the wrapped layer is its own child module."""
    model = nn.Module()
    model.mlp = _PeftStyleWrapper(_UnsupportedEPModuleStub())
    with pytest.raises(ValueError, match="does not support weight sync"):
        validate_weight_sync_support(model, "vllm")


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
