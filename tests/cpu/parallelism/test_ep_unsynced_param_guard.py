#!/usr/bin/env python
"""CPU tests for the mixin's unsynced-EP-param guard
(``DistributedTrainerMixin._validate_ep_peft_trainable_params_synced``).

EP modules are FSDP-ignored, so a trainable param the family's grad-sync hooks don't cover drifts
silently across DP ranks. The guard must scan EVERY EP run — full-finetune included, not only PEFT
(a family wrapper forgetting to declare a new weight in ``expert_named_params`` /
``replicated_named_params`` is exactly the non-PEFT failure mode) — and raise naming the offender.

Uses a stub trainer (only ``model`` + ``_find_ep_modules``) around a stub EP layer that exercises
the REAL ``synced_trainable_param_ids`` logic on the in-backward hook path.

Run: ``python tests/cpu/parallelism/test_ep_unsynced_param_guard.py`` (or ``pytest -m cpu``).
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from src.trainers.mixins.base import DistributedTrainerMixin
from tests.common.ep_stubs import StubEPLayerBase

E, H, M = 2, 4, 8


class _StubEPLayer(StubEPLayerBase):
    """Concrete EP layer on the single-node in-backward sync path, with the REAL
    ``synced_trainable_param_ids`` / ``expert_named_params`` machinery."""

    def __init__(self, rogue: bool = False):
        super().__init__()
        # In-backward hook path: not the ep1-FSDP-sharded state, not the deferred post-backward sweep.
        self.ep_config = SimpleNamespace(
            fsdp_shard_ep1_experts=False, ep_group_size=2, experts_fsdp_managed=False, defer_grad_sync=False
        )
        self.gate = nn.Linear(H, E, bias=False)  # the base's default _ROUTER_ATTR
        self.gate_proj = nn.Parameter(torch.randn(E, H, M))
        self.down_proj = nn.Parameter(torch.randn(E, M, H))
        if rogue:
            # A trainable weight the family never declared — no hook will ever sync it.
            self.rogue_scale = nn.Parameter(torch.randn(H))


class _StubTrainer:
    """Only what the guard reads: ``model`` (non-PEFT plain module) and ``_find_ep_modules``."""

    def __init__(self, layers):
        self.model = nn.Module()
        self._layers = layers

    def _find_ep_modules(self):
        return self._layers

    _validate_ep_peft_trainable_params_synced = DistributedTrainerMixin._validate_ep_peft_trainable_params_synced


def test_guard_raises_on_undeclared_trainable_param_without_peft():
    """FULL-FT run (no PeftModel anywhere): an undeclared trainable EP param must still raise —
    the PEFT-only early-out hid exactly this family-wrapper bug."""
    trainer = _StubTrainer([_StubEPLayer(rogue=True)])
    with pytest.raises(RuntimeError, match="rogue_scale"):
        trainer._validate_ep_peft_trainable_params_synced()


def test_guard_passes_on_fully_declared_layer():
    trainer = _StubTrainer([_StubEPLayer(rogue=False)])
    trainer._validate_ep_peft_trainable_params_synced()  # must not raise


def test_guard_passes_when_rogue_param_is_frozen():
    """Only TRAINABLE undeclared params are a drift risk — a frozen one must not trip the guard."""
    layer = _StubEPLayer(rogue=True)
    layer.rogue_scale.requires_grad_(False)
    _StubTrainer([layer])._validate_ep_peft_trainable_params_synced()  # must not raise


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
