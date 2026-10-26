#!/usr/bin/env python
"""DPO and KTO switch TRL's ``use_liger_kernel`` off before the distributed setup reads it.

On TRL 1.14 the flag selects TRL's chunked log-prob path, which refuses the router aux loss the
trainers hand TRL under an aux balancing mode (a ValueError in TRL's constructor) and scores through
the output embedding without the family's head transform; under FSDP2 the mixin's Liger-loss guard
would also refuse it, finding none of TRL's 1.6 loss objects. The flag must already be off when
``_init_distributed_config`` runs, whether the config arrives by keyword or positionally.

    python tests/cpu/trainers/test_preference_trl_liger_off.py
"""

import types

import pytest
import torch.nn as nn
from accelerate import PartialState
from transformers import PretrainedConfig

from src.distributed.parallelism_config import ParallelismConfig
from src.trainers.mixins.base import DistributedTrainerMixin
from src.trainers.preference.dpo import DistributedDPOTrainer
from src.trainers.preference.kto import DistributedKTOTrainer

PartialState()  # the trainers' accelerate logger requires an initialized state


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
