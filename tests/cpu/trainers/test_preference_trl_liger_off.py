#!/usr/bin/env python
"""DPO and KTO switch TRL's ``use_liger_kernel`` off before the distributed setup reads it.
On TRL 1.14 the flag selects a chunked log-prob path that refuses the router aux loss and skips the head transform."""

import types

import pytest
import torch.nn as nn
from transformers import PretrainedConfig

from halo.distributed.parallelism_config import ParallelismConfig
from halo.trainers.distributed_trainer import DistributedTrainerMixin
from halo.trainers.preference.dpo import DistributedDPOTrainer
from halo.trainers.preference.kto import DistributedKTOTrainer


class _SetupReached(Exception):
    pass


@pytest.fixture
def flag_at_setup(monkeypatch):
    seen = []

    def record(self, kwargs, *args, training_args=None, ctor_args=(), ctor_positions=None, **extra):
        config = kwargs.get("args") if "args" in kwargs else ctor_args[ctor_positions["args"]]
        seen.append(config.use_liger_kernel)
        raise _SetupReached

    monkeypatch.setattr(DistributedTrainerMixin, "_init_distributed_config", record)
    return seen


def _training_args():
    return types.SimpleNamespace(use_liger_kernel=True)


def _model():
    model = nn.Linear(2, 2)
    model.config = PretrainedConfig(num_experts=8, output_router_logits=True, router_aux_loss_coef=0.01)
    return model


@pytest.mark.parametrize("trainer_cls", [DistributedDPOTrainer, DistributedKTOTrainer], ids=lambda c: c.__name__)
@pytest.mark.parametrize("positional", [False, True], ids=["keyword", "positional"])
def test_trls_liger_flag_is_off_before_the_distributed_setup(flag_at_setup, trainer_cls, positional):
    model, training_args = _model(), _training_args()
    call = ((model, None, training_args), {}) if positional else ((), {"model": model, "args": training_args})
    with pytest.raises(_SetupReached):
        trainer_cls(*call[0], parallelism_config=ParallelismConfig(), **call[1])
    assert flag_at_setup == [False]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
