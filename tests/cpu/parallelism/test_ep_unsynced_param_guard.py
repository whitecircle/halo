#!/usr/bin/env python
"""``reject_unsynced_trainable_params``: EP modules are FSDP-ignored, so a param the grad-sync hooks miss drifts
silently across DP ranks."""

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

# ep_group_size > 1, or plain DP with fsdp_shard_ep1_experts off; CP and TP refuse that flag. The PP rows span two
# nodes, one stage each.
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
# Each stub records itself.
_WRAPS = (
    "_configure_pipeline_parallel",
    "_apply_ep_tp_fsdp2",
    "_apply_ep_fsdp2",
    "_apply_cp_fsdp2",
    "_apply_tp_fsdp2",
    "_apply_dp_fsdp2",
    "_install_qlora_grad_sync",
)
# Ep1 experts there are FSDP-managed (CP and TP refuse otherwise).
_NON_EP_MODES = {"dp": {}, "cp": {"cp_size": 2}, "tp": {"tp_size": 2}}
# What needs a live Trainer or process group.
_PRE_GUARD_STEPS = (
    "_cast_peft_params_to_compute_dtype",
    "_upcast_non_ep_params_to_fp32",
    "_install_ep_clip",
    "_install_tp_clip",
    "_adopt_cp_wrapper",
)


class _StubEPLayer(StubEPLayerBase):
    def __init__(self, rogue: bool = False, managed: bool = False):
        super().__init__()
        # In-backward hook path unless managed.
        self.ep_config = SimpleNamespace(
            fsdp_shard_ep1_experts=managed,
            ep_group_size=1 if managed else 2,
            experts_fsdp_managed=managed,
            sweep_reduces_across_replicas=False,
            uses_backward_grad_hooks=not managed,
        )
        self.gate = nn.Linear(H, E, bias=False)  # The base's default _ROUTER_ATTR.
        self.gate_proj = nn.Parameter(torch.randn(E, H, M))
        self.down_proj = nn.Parameter(torch.randn(E, M, H))
        if rogue:
            # A weight the family never declared, so no hook will sync it.
            self.rogue_scale = nn.Parameter(torch.randn(H))


class _StubTrainer:
    def __init__(self, layers, *, deferred: bool = False):
        self.model = nn.ModuleList(layers)
        self._layers = layers
        self._ep_config = SimpleNamespace(sweep_reduces_across_replicas=deferred)

    def check(self):
        candidates = [param for layer in self._layers for param in layer.parameters()]
        reject_unsynced_trainable_params(self.model, candidates, self._layers, self._ep_config)


class _PastGuard(Exception):
    """Carries control out of the spine once the guard has passed."""


class _FrozenFp32Block(nn.Module):
    """A dtype exclusion every FSDP2 wrap leaves out."""

    def __init__(self):
        super().__init__()
        self.dt_bias = nn.Parameter(torch.zeros(H), requires_grad=False)
        self.proj = nn.Linear(H, H, bias=False).to(torch.bfloat16)


class _ModeHost(DistributedTrainerMixin):
    _deferred_liger_kernel = False
    _accelerate_manages_fsdp = False
    _accelerate_manages_ddp = False

    def __init__(self, config, layers):
        self.parallelism_config = config
        self.args = SimpleNamespace(load_best_model_at_end=False, optim="adamw_torch", per_device_train_batch_size=1)
        self.label_smoother = None
        self.compute_loss_func = None
        self._bf16_optimizer = False
        self.model = nn.ModuleList(layers)
        self.model._device_mesh = object()  # What the TP loaders attach; the EP+TP wrap requires it.
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
    """A PEFT-only early-out would hide this family-wrapper bug."""
    with pytest.raises(RuntimeError, match="rogue_scale"):
        _StubTrainer([_StubEPLayer(rogue=True)]).check()


def test_guard_passes_on_fully_declared_layer():
    _StubTrainer([_StubEPLayer(rogue=False)]).check()


def test_guard_passes_when_rogue_param_is_frozen():
    layer = _StubEPLayer(rogue=True)
    layer.rogue_scale.requires_grad_(False)
    _StubTrainer([layer]).check()


def test_an_fsdp_managed_layer_covers_nothing_left_out_of_the_shard_groups():
    """FSDP2 is its only sync, so even a declared weight left out of the shard groups is unsynced."""
    with pytest.raises(RuntimeError, match="gate_proj"):
        _StubTrainer([_StubEPLayer(managed=True)]).check()


def test_the_deferred_sweep_covers_every_trainable_param():
    _StubTrainer([_StubEPLayer(rogue=True)], deferred=True).check()


def test_expert_tp_implies_an_ep_group():
    """The ladder sends ETP down the EP branches on this alone; a grouped-GEMM fallback under ETP would mis-shard
    FSDP2."""
    config = _config(expert_tp_size=2)
    assert config.is_expert_tp_mode and config.has_ep_group


@pytest.mark.parametrize("mode", sorted(_EP_LAYER_MODES))
def test_construction_refuses_an_unsynced_param_in_every_ep_layer_mode(mode):
    """The only check over what the wraps leave out."""
    kwargs, expected_setup = _EP_LAYER_MODES[mode]
    config = _config(**kwargs)
    assert config.needs_ep_layers
    host = _ModeHost(config, [_StubEPLayer(rogue=True)])

    with patch(_RANK_UNIFORM_ENV), pytest.raises(RuntimeError, match="rogue_scale"):
        DistributedTrainerMixin._configure_distributed_modes(host)

    assert host.ran_setups == [expected_setup]


def test_construction_refuses_ep_layers_a_hand_built_config_does_not_declare():
    """The plain-DP wrap leaves them out and nothing else syncs the undeclared parameter."""
    config = _config(use_grouped_gemm=False, fsdp_shard_ep1_experts=False)
    assert not config.needs_ep_layers, "premise: the config declares no EP layers"
    host = _ModeHost(config, [_StubEPLayer(rogue=True)])

    with patch(_RANK_UNIFORM_ENV), pytest.raises(RuntimeError, match="rogue_scale"):
        DistributedTrainerMixin._configure_distributed_modes(host)

    assert host.ran_setups == ["_apply_dp_fsdp2"]


@pytest.mark.parametrize("grouped_gemm", [False, True], ids=["plain", "grouped-gemm"])
@pytest.mark.parametrize("mode", sorted(_NON_EP_MODES))
def test_construction_passes_where_every_trainable_param_is_sharded(mode, grouped_gemm):
    """The frozen fp32 exclusion is not trainable, and a grouped-GEMM ep1 layer's parameters are FSDP-managed."""
    config = _config(use_grouped_gemm=grouped_gemm, **_NON_EP_MODES[mode])
    ep_layers = [_StubEPLayer(rogue=True, managed=True).to(torch.bfloat16)] if grouped_gemm else []
    layers = [_FrozenFp32Block(), *ep_layers]
    host = _ModeHost(config, layers)
    assert config.experts_fsdp_managed and host._dtype_excluded_params(), "premise: an exclusion, managed experts"

    with patch(_RANK_UNIFORM_ENV), pytest.raises(_PastGuard):
        DistributedTrainerMixin._configure_distributed_modes(host)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
