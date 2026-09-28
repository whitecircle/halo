#!/usr/bin/env python
"""CPU tests for the mixin's unsynced-trainable-param guard
(``DistributedTrainerMixin._reject_unsynced_trainable_params``).

EP modules are FSDP-ignored, so a trainable param the family's grad-sync hooks don't cover drifts
silently across DP ranks. The guard must scan EVERY EP run — full-finetune included, not only PEFT
(a family wrapper forgetting to declare a new weight in ``expert_named_params`` /
``replicated_named_params`` is exactly the non-PEFT failure mode) — and raise naming the offender.
FSDP-managed EP layers register no hooks, so they cover nothing left out of the shard groups; the
deferred post-backward sweep covers everything.

Uses a stub trainer (only ``model`` + ``_find_ep_modules``) around a stub EP layer that exercises
the REAL ``synced_trainable_param_ids`` logic on the in-backward hook path. The guard runs once, at
construction after the mode's wrap: the per-mode test drives the real ``_setup_distributed_modes``
through every axis set whose wrap leaves an EP layer out of FSDP2 and holds each one to the refusal.

Run: ``python tests/cpu/parallelism/test_ep_unsynced_param_guard.py`` (or ``pytest -m cpu``).
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from torch import nn

from src.trainers.mixins.base import DistributedTrainerMixin
from tests.common.ep_stubs import StubEPLayerBase
from tests.common.parallelism import make_parallelism_config

E, H, M = 2, 4, 8

# Every axis set whose wrap leaves EP layers out of FSDP2, paired with the mode setup
# _setup_distributed_modes dispatches it to: ep_group_size > 1, or plain DP with fsdp_shard_ep1_experts
# off. CP and TP refuse that flag at config time, so their wraps shard every ep1 expert.
_EP_WRAPPER_MODES = {
    "dp": ({"fsdp_shard_ep1_experts": False}, "_setup_ep_only"),
    "ep": ({"ep_size": 8}, "_setup_ep_only"),
    "etp": ({"expert_tp_size": 2}, "_setup_ep_only"),
    "ep_etp": ({"ep_size": 4, "expert_tp_size": 2}, "_setup_ep_only"),
    "ep_cp": ({"ep_size": 8, "cp_size": 2}, "_setup_ep_cp"),
    "ep_tp": ({"ep_size": 8, "tp_size": 2}, "_setup_ep_tp"),
}
_MODE_SETUPS = (
    "_setup_pipeline_parallel",
    "_setup_ep_tp",
    "_setup_ep_cp",
    "_setup_cp_only",
    "_setup_tp_only",
    "_setup_ep_only",
    "_setup_standard_data_parallel",
)
# Everything _setup_distributed_modes runs ahead of the guard that needs a live Trainer or process group.
_PRE_GUARD_STEPS = (
    "_log_parallelism_config",
    "_validate_lora_tp_compatibility",
    "_validate_router_aux_loss_consumable",
    "_validate_load_best_model_reloadable",
    "_cast_peft_params_to_compute_dtype",
    "_upcast_non_ep_params_to_fp32",
    "_validate_lora_ep_compatibility",
)


class _StubEPLayer(StubEPLayerBase):
    """Concrete EP layer on the single-node in-backward sync path, with the REAL
    ``synced_trainable_param_ids`` / ``expert_named_params`` machinery."""

    def __init__(self, rogue: bool = False, managed: bool = False):
        super().__init__()
        # In-backward hook path unless managed: not the deferred post-backward sweep.
        self.ep_config = SimpleNamespace(
            fsdp_shard_ep1_experts=managed,
            ep_group_size=1 if managed else 2,
            experts_fsdp_managed=managed,
            defer_grad_sync=False,
        )
        self.gate = nn.Linear(H, E, bias=False)  # the base's default _ROUTER_ATTR
        self.gate_proj = nn.Parameter(torch.randn(E, H, M))
        self.down_proj = nn.Parameter(torch.randn(E, M, H))
        if rogue:
            # A trainable weight the family never declared — no hook will ever sync it.
            self.rogue_scale = nn.Parameter(torch.randn(H))


class _StubTrainer:
    """Only what the guard reads: ``model`` (non-PEFT plain module holding the layers), ``_ep_config``
    and ``_find_ep_modules``."""

    def __init__(self, layers, *, deferred: bool = False):
        self.model = nn.ModuleList(layers)
        self._layers = layers
        self._ep_config = SimpleNamespace(defer_grad_sync=deferred)

    def _find_ep_modules(self):
        return self._layers

    def check(self):
        """The guard over what an EP run leaves out of FSDP2: every EP layer's parameters."""
        candidates = [param for layer in self._layers for param in layer.parameters()]
        DistributedTrainerMixin._reject_unsynced_trainable_params(self, self.model, candidates)


class _ModeHost(DistributedTrainerMixin):
    """The mixin's real mode dispatch and guard over a stub model; each mode's wrap only records itself."""

    _loss_is_own_mean = False
    _deferred_liger_kernel = False
    _loss_outside_model_forward = False
    _accelerate_manages_fsdp = False
    _accelerate_manages_ddp = False

    def __init__(self, config, layers):
        self.parallelism_config = config
        self.model = nn.ModuleList(layers)
        self._ep_config = None
        self.ran_setups: list[str] = []


for _name in _PRE_GUARD_STEPS:
    setattr(_ModeHost, _name, lambda self: None)
for _name in _MODE_SETUPS:
    setattr(_ModeHost, _name, lambda self, _setup=_name: self.ran_setups.append(_setup))


def test_guard_raises_on_undeclared_trainable_param_without_peft():
    """FULL-FT run (no PeftModel anywhere): an undeclared trainable EP param must still raise —
    the PEFT-only early-out hid exactly this family-wrapper bug."""
    with pytest.raises(RuntimeError, match="rogue_scale"):
        _StubTrainer([_StubEPLayer(rogue=True)]).check()


def test_guard_passes_on_fully_declared_layer():
    _StubTrainer([_StubEPLayer(rogue=False)]).check()  # must not raise


def test_guard_passes_when_rogue_param_is_frozen():
    """Only TRAINABLE undeclared params are a drift risk — a frozen one must not trip the guard."""
    layer = _StubEPLayer(rogue=True)
    layer.rogue_scale.requires_grad_(False)
    _StubTrainer([layer]).check()  # must not raise


def test_an_fsdp_managed_layer_covers_nothing_left_out_of_the_shard_groups():
    """Under ``experts_fsdp_managed`` the EP layer registers no hooks: FSDP2 is its only sync, so even
    a declared weight left out of the shard groups is unsynced."""
    with pytest.raises(RuntimeError, match="gate_proj"):
        _StubTrainer([_StubEPLayer(managed=True)]).check()


def test_the_deferred_sweep_covers_every_trainable_param():
    _StubTrainer([_StubEPLayer(rogue=True)], deferred=True).check()  # must not raise


@pytest.mark.parametrize("mode", sorted(_EP_WRAPPER_MODES))
def test_construction_refuses_an_unsynced_param_in_every_ep_wrapper_mode(mode):
    """The construction check is the only one over what the wraps leave out, so it must refuse the
    undeclared parameter in every mode that wraps around an EP layer."""
    kwargs, expected_setup = _EP_WRAPPER_MODES[mode]
    with patch("src.distributed.parallelism_config.validate_nvlink_domain_against_fabric"):
        config = make_parallelism_config(world_size=8, gpus_per_node=8, **kwargs)
    assert config.needs_ep_wrappers
    host = _ModeHost(config, [_StubEPLayer(rogue=True)])

    with patch("src.trainers.mixins.base.verify_rank_uniform_env"), pytest.raises(RuntimeError, match="rogue_scale"):
        DistributedTrainerMixin._setup_distributed_modes(host)

    assert host.ran_setups == [expected_setup]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
