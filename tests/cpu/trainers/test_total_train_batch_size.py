"""``get_total_train_batch_size`` must count the toolkit's DP replicas, not HF's TP-blind world.
The base reads ``model._tp_size`` and an accelerate ``parallelism_config``, neither set, so every rank is a replica."""

from types import SimpleNamespace

import pytest
import torch.nn as nn
from transformers import Trainer

from halo.trainers.distributed_trainer import DistributedTrainerMixin

PER_DEVICE_BATCH = 4
GRAD_ACCUM = 8


class _Trainer(DistributedTrainerMixin, Trainer):
    pass


def _trainer(data_parallel_size: int) -> _Trainer:
    trainer = _Trainer.__new__(_Trainer)
    trainer._train_batch_size = PER_DEVICE_BATCH
    trainer.parallelism_config = SimpleNamespace(data_parallel_size=data_parallel_size)
    trainer.model = nn.Linear(2, 2)  # no ``_tp_size``, like every tp_plan model transformers builds
    trainer.is_deepspeed_enabled = False
    trainer.accelerator = SimpleNamespace(parallelism_config=None)
    return trainer


@pytest.mark.parametrize(
    ("data_parallel_size", "world_size"),
    [(1, 2), (2, 8), (1, 8)],
    ids=["tp2", "tp4-dp2", "cp8-or-pp8"],
)
def test_counts_data_parallel_replicas_only(data_parallel_size, world_size):
    args = SimpleNamespace(gradient_accumulation_steps=GRAD_ACCUM, world_size=world_size)
    trainer = _trainer(data_parallel_size)

    assert trainer.get_total_train_batch_size(args) == PER_DEVICE_BATCH * GRAD_ACCUM * data_parallel_size
    # Premise: the base derivation still counts the non-DP ranks as replicas.
    assert Trainer.get_total_train_batch_size(trainer, args) == PER_DEVICE_BATCH * GRAD_ACCUM * world_size


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
