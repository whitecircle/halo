#!/usr/bin/env python
"""The SFT trainers score through the model's own forward, never TRL's ``chunked_nll`` patch.

TRL 1.14 defaults an unset ``loss_type`` to ``chunked_nll`` whenever ``use_liger_kernel`` is off, and
that type swaps the model's forward for one that projects ``lm_head`` itself: no head transform, no
router aux loss, no logits for a subclass that reads them. The trainer turns it into ``nll`` before
the distributed setup, whether the config arrives by keyword or positionally.

    python tests/cpu/trainers/test_sft_loss_type.py
"""

import types

import pytest
import torch.nn as nn
from accelerate import PartialState
from transformers import PretrainedConfig
from trl import SFTConfig

from src.distributed.parallelism_config import ParallelismConfig
from src.trainers.distillation.self_distillation import DistributedSelfDistillationTrainer
from src.trainers.distributed_trainer import DistributedTrainerMixin
from src.trainers.sft import DistributedSFTTrainer

PartialState()  # the trainers' accelerate logger requires an initialized state

REQUIRED_KWARGS = {
    DistributedSelfDistillationTrainer: {
        "reference_kl_coef": 0.0,
        "reference_kl_loss": "unnormalized_kl",
        "confidence_weight_opd": True,
        "opd_exclude_eos": True,
    },
}


class _SetupReached(Exception):
    pass


@pytest.fixture(autouse=True)
def stop_at_setup(monkeypatch):
    def stop(self, kwargs, *args, **extra):
        raise _SetupReached

    monkeypatch.setattr(DistributedTrainerMixin, "_init_distributed_config", stop)


def _model():
    model = nn.Linear(2, 2)
    model.config = PretrainedConfig()
    return model


def test_trls_default_for_a_liger_off_run_is_chunked_nll():
    """Premise: what the trainer guards against is TRL's own default."""
    assert SFTConfig(output_dir="unused", use_liger_kernel=False, bf16=False, use_cpu=True).loss_type == "chunked_nll"


@pytest.mark.parametrize(
    "trainer_cls", [DistributedSFTTrainer, DistributedSelfDistillationTrainer], ids=lambda c: c.__name__
)
@pytest.mark.parametrize("positional", [False, True], ids=["keyword", "positional"])
def test_chunked_nll_becomes_nll_before_the_setup(trainer_cls, positional):
    training_args = types.SimpleNamespace(loss_type="chunked_nll")
    call = ((_model(), training_args), {}) if positional else ((), {"model": _model(), "args": training_args})
    with pytest.raises(_SetupReached):
        trainer_cls(
            *call[0], parallelism_config=ParallelismConfig(), **REQUIRED_KWARGS.get(trainer_cls, {}), **call[1]
        )
    assert training_args.loss_type == "nll"


def test_another_loss_type_is_left_alone():
    training_args = types.SimpleNamespace(loss_type="dft")
    with pytest.raises(_SetupReached):
        DistributedSFTTrainer(model=_model(), args=training_args, parallelism_config=ParallelismConfig())
    assert training_args.loss_type == "dft"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
