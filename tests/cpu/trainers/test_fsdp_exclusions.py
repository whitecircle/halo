#!/usr/bin/env python
"""What an FSDP2 wrap leaves out of its shard groups — CPU-only.

A parameter outside every ``fully_shard`` group gets no reduce-scatter. Frozen, that is harmless; trainable,
it keeps its own rank's gradient unless an EP layer's hooks or the deferred sweep average it, and the
ranks drift apart while every loss stays finite. So the dtype exclusion is taken per parameter: a frozen
fp32 parameter must not take the trainable adapter under the same module with it. The guard on what is
left out is ``tests/cpu/parallelism/test_ep_unsynced_param_guard.py``'s.

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


def _trainer(model: nn.Module) -> SimpleNamespace:
    """Only what the exclusion walk reads."""
    trainer = SimpleNamespace(
        model=model,
        parallelism_config=SimpleNamespace(experts_fsdp_managed=False),
        _find_ep_modules=list,
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


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
