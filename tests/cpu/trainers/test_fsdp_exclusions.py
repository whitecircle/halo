#!/usr/bin/env python
"""What an FSDP2 wrap leaves out of its shard groups, and the wrap-time guard on it — CPU-only.

A parameter outside every ``fully_shard`` group gets no reduce-scatter. Frozen, that is harmless; trainable,
it keeps its own rank's gradient unless an EP layer's hooks or the deferred sweep average it, and the
ranks drift apart while every loss stays finite. So the dtype exclusion is taken per parameter (a
frozen fp32 parameter must not take the trainable adapter under the same module with it), and
``_reject_unsynced_fsdp_exclusions`` refuses any trainable exclusion nothing else syncs.

Run: python tests/cpu/trainers/test_fsdp_exclusions.py  (or pytest)
"""

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from src.trainers.mixins.base import DistributedTrainerMixin


class _ForgetGate(nn.Module):
    """A frozen fp32 parameter owned directly by a module whose child carries a trainable adapter."""

    def __init__(self):
        super().__init__()
        self.dt_bias = nn.Parameter(torch.zeros(4), requires_grad=False)
        self.proj = nn.Linear(4, 4, bias=False).to(torch.bfloat16).requires_grad_(False)
        self.lora_B = nn.Linear(4, 4, bias=False).to(torch.bfloat16)


def _trainer(model: nn.Module, *, ep_modules=(), ep_config=None) -> SimpleNamespace:
    """Only what the exclusion walk and the guard read."""
    trainer = SimpleNamespace(
        model=model,
        parallelism_config=SimpleNamespace(experts_fsdp_managed=False),
        _ep_config=ep_config,
        _find_ep_modules=lambda: list(ep_modules),
    )
    trainer._dtype_excluded_params = lambda: DistributedTrainerMixin._dtype_excluded_params(trainer)
    return trainer


def _ids(params) -> set[int]:
    return {id(p) for p in params}


def test_a_trainable_child_of_a_module_owning_a_frozen_fp32_param_stays_sharded():
    gate = _ForgetGate()

    exclusions = DistributedTrainerMixin._fsdp_exclusions(_trainer(gate))

    assert _ids(exclusions.params) == {id(gate.dt_bias)}


def test_a_uniform_model_excludes_nothing():
    gate = _ForgetGate()
    gate.dt_bias.data = gate.dt_bias.data.to(torch.bfloat16)

    assert DistributedTrainerMixin._fsdp_exclusions(_trainer(gate)).params == []


def test_a_trainable_parameter_excluded_with_no_other_sync_raises():
    gate = _ForgetGate()

    with pytest.raises(RuntimeError, match=r"lora_B\.weight"):
        DistributedTrainerMixin._reject_unsynced_fsdp_exclusions(_trainer(gate), gate, [gate.lora_B.weight])


def test_frozen_exclusions_pass():
    gate = _ForgetGate()

    DistributedTrainerMixin._reject_unsynced_fsdp_exclusions(_trainer(gate), gate, [gate.dt_bias, gate.proj.weight])


def _ep_module(*synced: nn.Parameter, managed: bool) -> SimpleNamespace:
    return SimpleNamespace(
        ep_config=SimpleNamespace(experts_fsdp_managed=managed),
        synced_trainable_param_ids=lambda: _ids(synced),
    )


def test_a_trainable_parameter_an_ep_layer_syncs_passes():
    gate = _ForgetGate()
    trainer = _trainer(gate, ep_modules=[_ep_module(gate.lora_B.weight, managed=False)])

    DistributedTrainerMixin._reject_unsynced_fsdp_exclusions(trainer, gate, [gate.lora_B.weight])


def test_an_fsdp_managed_ep_layer_covers_nothing_it_leaves_out():
    """Under ``experts_fsdp_managed`` the EP layer registers no hooks: FSDP2 is its only sync."""
    gate = _ForgetGate()
    trainer = _trainer(gate, ep_modules=[_ep_module(gate.lora_B.weight, managed=True)])

    with pytest.raises(RuntimeError, match=r"lora_B\.weight"):
        DistributedTrainerMixin._reject_unsynced_fsdp_exclusions(trainer, gate, [gate.lora_B.weight])


def test_the_deferred_sweep_covers_every_trainable_exclusion():
    gate = _ForgetGate()
    trainer = _trainer(gate, ep_config=SimpleNamespace(defer_grad_sync=True))

    DistributedTrainerMixin._reject_unsynced_fsdp_exclusions(trainer, gate, [gate.lora_B.weight])


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
