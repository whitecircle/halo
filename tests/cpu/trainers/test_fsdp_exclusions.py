#!/usr/bin/env python
"""A trainable parameter left out of FSDP2's shard groups keeps its rank's gradient, so exclusion is per parameter.
The guard on what is left out is in ``tests/cpu/parallelism/test_ep_unsynced_param_guard.py``."""

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from halo.trainers.distributed_trainer import DistributedTrainerMixin


class _ForgetGate(nn.Module):
    """A frozen fp32 parameter owned directly by a module whose child carries a trainable adapter."""

    def __init__(self):
        super().__init__()
        self.dt_bias = nn.Parameter(torch.zeros(4), requires_grad=False)
        self.proj = nn.Linear(4, 4, bias=False).to(torch.bfloat16).requires_grad_(False)
        self.lora_B = nn.Linear(4, 4, bias=False).to(torch.bfloat16)


def _trainer(model: nn.Module) -> SimpleNamespace:
    trainer = SimpleNamespace(
        model=model,
        parallelism_config=SimpleNamespace(experts_fsdp_managed=False),
        _ep_layers=list,
    )
    trainer._dtype_excluded_params = lambda: DistributedTrainerMixin._dtype_excluded_params(trainer)
    return trainer


def _ids(params) -> set[int]:
    return {id(p) for p in params}


def test_a_trainable_child_of_a_module_owning_a_frozen_fp32_param_stays_sharded():
    gate = _ForgetGate()

    assert _ids(DistributedTrainerMixin._fsdp_excluded_params(_trainer(gate))) == {id(gate.dt_bias)}


def test_a_uniform_model_excludes_nothing():
    gate = _ForgetGate()
    gate.dt_bias.data = gate.dt_bias.data.to(torch.bfloat16)

    assert DistributedTrainerMixin._fsdp_excluded_params(_trainer(gate)) == []


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
