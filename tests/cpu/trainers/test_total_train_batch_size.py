"""``get_total_train_batch_size`` must count the toolkit's DP replicas, not HF's TP-blind world.

The base Trainer derives its DP width as ``args.world_size // get_tp_size() // get_cp_size() //
get_sp_size()``. ``get_tp_size`` reads ``model._tp_size``, which transformers never sets for its own
``tp_plan`` TP, and the CP/SP terms read an accelerate ``parallelism_config`` the toolkit never
builds — so every TP/CP/ETP/PP rank counts as a data-parallel replica in the "Total train batch
size" banner and in ``num_train_samples`` / ``train_samples_per_second`` under ``max_steps``.

    python tests/cpu/trainers/test_total_train_batch_size.py
"""

from types import SimpleNamespace

import pytest
import torch.nn as nn
from transformers import Trainer

from src.trainers.mixins.base import DistributedTrainerMixin

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
    # Premise: the base derivation still counts the non-DP ranks as replicas, so the override is load-bearing.
    assert Trainer.get_total_train_batch_size(trainer, args) == PER_DEVICE_BATCH * GRAD_ACCUM * world_size


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
