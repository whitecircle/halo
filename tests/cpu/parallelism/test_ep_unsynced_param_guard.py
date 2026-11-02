#!/usr/bin/env python
"""CPU tests for the trainer's unsynced-trainable-param guard (``reject_unsynced_trainable_params``).

EP modules are FSDP-ignored, so a trainable param the family's grad-sync hooks don't cover drifts
silently across DP ranks. The guard must scan EVERY EP run — full-finetune included, not only PEFT
(a family wrapper forgetting to declare a new weight in ``expert_named_params`` /
``replicated_named_params`` is exactly the non-PEFT failure mode) — and raise naming the offender.
FSDP-managed EP layers register no hooks, so they cover nothing left out of the shard groups; the
deferred post-backward sweep covers everything.

Uses a stub trainer (only ``model`` + its EP layers) around a stub EP layer that exercises
the REAL ``synced_trainable_param_ids`` logic on the in-backward hook path. The guard runs once, at
construction after the mode's wrap: the per-mode test drives the real ``_configure_distributed_modes`` and
mode ladder through every axis set whose wrap leaves an EP layer out of FSDP2 and holds each one to the refusal,
refuses EP layers a hand-built config does not declare, and passes plain DP, CP and TP, whose frozen
dtype exclusions and FSDP-managed ep1 experts leave no trainable parameter out.

Run: ``python tests/cpu/parallelism/test_ep_unsynced_param_guard.py`` (or ``pytest -m cpu``).
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from torch import nn

from halo.trainers.distributed_trainer import DistributedTrainerMixin
from halo.trainers.parallelism.mode_setup import reject_unsynced_trainable_params
from tests.common.ep_stubs import StubEPLayerBase
from tests.common.parallelism import make_parallelism_config

E, H, M = 2, 4, 8
_FABRIC_PROBE = "halo.distributed.parallelism_config.validate_nvlink_domain_against_fabric"
_RANK_UNIFORM_ENV = "halo.trainers.distributed_trainer.agree_rank_uniform_env"

# Every axis set whose wrap leaves EP layers out of FSDP2, paired with the wrap _wrap_for_parallel_mode
# dispatches it to: ep_group_size > 1, or plain DP with fsdp_shard_ep1_experts off. CP and TP refuse that
# flag at config time, so their wraps shard every ep1 expert. The PP rows span two nodes, one pipeline
# stage each.
_EP_LAYER_MODES = {
    "dp": ({"fsdp_shard_ep1_experts": False}, "_apply_ep_fsdp2"),
    "ep": ({"ep_size": 8}, "_apply_ep_fsdp2"),
    "etp": ({"expert_tp_size": 2}, "_apply_ep_fsdp2"),
    "ep_etp": ({"ep_size": 4, "expert_tp_size": 2}, "_apply_ep_fsdp2"),
    "ep_cp": ({"ep_size": 8, "cp_size": 2}, "_apply_ep_fsdp2"),
    "ep_tp": ({"ep_size": 8, "tp_size": 2}, "_apply_ep_tp_fsdp2"),
    "pp_ep": ({"world_size": 16, "pp_size": 2, "ep_size": 8}, "_configure_pipeline_parallel"),
    "pp_etp": ({"world_size": 16, "pp_size": 2, "expert_tp_size": 2}, "_configure_pipeline_parallel"),
}
# The wraps the ladder dispatches to; each stub records itself.
_WRAPS = (
    "_configure_pipeline_parallel",
    "_apply_ep_tp_fsdp2",
    "_apply_ep_fsdp2",
    "_apply_cp_fsdp2",
    "_apply_tp_fsdp2",
    "_apply_dp_fsdp2",
    "_install_qlora_grad_sync",
)
# Axis sets with no EP distribution; ep1 experts there are FSDP-managed (CP and TP refuse otherwise).
_NON_EP_MODES = {"dp": {}, "cp": {"cp_size": 2}, "tp": {"tp_size": 2}}
# Everything _configure_distributed_modes runs ahead of the guard that needs a live Trainer or process group.
_PRE_GUARD_STEPS = (
    "_cast_peft_params_to_compute_dtype",
    "_upcast_non_ep_params_to_fp32",
    "_install_ep_clip",
    "_install_tp_clip",
    "_adopt_cp_wrapper",
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
            sweep_reduces_across_replicas=False,
            uses_backward_grad_hooks=not managed,
        )
        self.gate = nn.Linear(H, E, bias=False)  # the base's default _ROUTER_ATTR
        self.gate_proj = nn.Parameter(torch.randn(E, H, M))
        self.down_proj = nn.Parameter(torch.randn(E, M, H))
        if rogue:
            # A trainable weight the family never declared — no hook will ever sync it.
            self.rogue_scale = nn.Parameter(torch.randn(H))


class _StubTrainer:
    """Only what the guard reads: ``model`` (non-PEFT plain module holding the layers), its EP layers and
    the captured ``_ep_config``."""

    def __init__(self, layers, *, deferred: bool = False):
        self.model = nn.ModuleList(layers)
        self._layers = layers
        self._ep_config = SimpleNamespace(sweep_reduces_across_replicas=deferred)

    def check(self):
        """The guard over what an EP run leaves out of FSDP2: every EP layer's parameters."""
        candidates = [param for layer in self._layers for param in layer.parameters()]
        reject_unsynced_trainable_params(self.model, candidates, self._layers, self._ep_config)


class _PastGuard(Exception):
    """Carries control out of the spine once the guard has passed."""


class _FrozenFp32Block(nn.Module):
    """A frozen fp32 parameter beside trainable bf16 ones: a dtype exclusion every FSDP2 wrap leaves out."""

    def __init__(self):
        super().__init__()
        self.dt_bias = nn.Parameter(torch.zeros(H), requires_grad=False)
        self.proj = nn.Linear(H, H, bias=False).to(torch.bfloat16)


class _ModeHost(DistributedTrainerMixin):
    """The mixin's real mode dispatch and guard over a stub model; each mode's wrap only records itself."""

    _deferred_liger_kernel = False
    _accelerate_manages_fsdp = False
    _accelerate_manages_ddp = False

    def __init__(self, config, layers):
        self.parallelism_config = config
        self.args = SimpleNamespace(load_best_model_at_end=False, optim="adamw_torch")
        self._bf16_optimizer = False
        self.model = nn.ModuleList(layers)
        self.model._device_mesh = object()  # what the TP loaders attach; the EP+TP wrap requires it
        self._ep_config = None
        self.ran_setups: list[str] = []

    def _warn_on_zeroed_lora_dropout(self):
        raise _PastGuard


for _name in _PRE_GUARD_STEPS:
    setattr(_ModeHost, _name, lambda self: None)
for _name in _WRAPS:
    setattr(_ModeHost, _name, lambda self, *_args, _wrap=_name, **_kwargs: self.ran_setups.append(_wrap))


def _config(world_size: int = 8, **kwargs):
    with patch(_FABRIC_PROBE):
        return make_parallelism_config(world_size=world_size, gpus_per_node=8, **kwargs)


def test_guard_raises_on_undeclared_trainable_param_without_peft():
    """FULL-FT run (no PeftModel anywhere): an undeclared trainable EP param must still raise —
    a PEFT-only early-out would hide exactly this family-wrapper bug."""
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


def test_expert_tp_implies_an_ep_group():
    """The ladder sends ETP down the EP branches on this alone (ep_group_size = ep_size * expert_tp_size), so a
    grouped-GEMM fallback reached under ETP would mis-shard FSDP2."""
    config = _config(expert_tp_size=2)
    assert config.is_expert_tp_mode and config.has_ep_group


@pytest.mark.parametrize("mode", sorted(_EP_LAYER_MODES))
def test_construction_refuses_an_unsynced_param_in_every_ep_layer_mode(mode):
    """The construction check is the only one over what the wraps leave out, so it must refuse the
    undeclared parameter in every mode that wraps around an EP layer."""
    kwargs, expected_setup = _EP_LAYER_MODES[mode]
    config = _config(**kwargs)
    assert config.needs_ep_layers
    host = _ModeHost(config, [_StubEPLayer(rogue=True)])

    with patch(_RANK_UNIFORM_ENV), pytest.raises(RuntimeError, match="rogue_scale"):
        DistributedTrainerMixin._configure_distributed_modes(host)

    assert host.ran_setups == [expected_setup]


def test_construction_refuses_ep_layers_a_hand_built_config_does_not_declare():
    """A model carrying EP layers under a config without EP layers: the plain-DP wrap leaves them out
    of FSDP2 (fsdp_shard_ep1_experts off) and nothing else syncs the undeclared parameter."""
    config = _config(use_grouped_gemm=False, fsdp_shard_ep1_experts=False)
    assert not config.needs_ep_layers, "premise: the config declares no EP layers"
    host = _ModeHost(config, [_StubEPLayer(rogue=True)])

    with patch(_RANK_UNIFORM_ENV), pytest.raises(RuntimeError, match="rogue_scale"):
        DistributedTrainerMixin._configure_distributed_modes(host)

    assert host.ran_setups == ["_apply_dp_fsdp2"]


@pytest.mark.parametrize("grouped_gemm", [False, True], ids=["plain", "grouped-gemm"])
@pytest.mark.parametrize("mode", sorted(_NON_EP_MODES))
def test_construction_passes_where_every_trainable_param_is_sharded(mode, grouped_gemm):
    """No false refusal: the frozen fp32 exclusion is not trainable, and a grouped-GEMM ep1 layer's
    parameters, the undeclared one included, are FSDP-managed."""
    config = _config(use_grouped_gemm=grouped_gemm, **_NON_EP_MODES[mode])
    ep_layers = [_StubEPLayer(rogue=True, managed=True).to(torch.bfloat16)] if grouped_gemm else []
    layers = [_FrozenFp32Block(), *ep_layers]
    host = _ModeHost(config, layers)
    assert config.experts_fsdp_managed and host._dtype_excluded_params(), "premise: an exclusion, managed experts"

    with patch(_RANK_UNIFORM_ENV), pytest.raises(_PastGuard):
        DistributedTrainerMixin._configure_distributed_modes(host)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
