#!/usr/bin/env python
"""Two seams that keep packed-batch inputs intact on their way into a trainer.

1. Column pruning: ``seq_lengths`` is a collator input, not a model-forward parameter, so HF's
   signature-based pruning drops it unless the trainer's signature set names it. TRL's SFT trainer
   happens to hard-code it; every other base does not. ``DataParallelDataLoaderMixin`` unions the
   collator's declared ``required_dataset_columns`` so the pin cannot drift from the collator.
2. The CP × packing rejection must see a collator passed POSITIONALLY (at TRL's own data_collator
   slot), not only via keyword — a packed collator slipping past the gate under CP attends across
   documents silently.

Run: python tests/cpu/trainers/test_packing_column_and_cp_seams.py
"""

import types
from unittest import mock

import pytest
from accelerate import PartialState
from trl import SFTTrainer

from src.data.collators.packing import DataCollatorWithPacking
from src.trainers.mixins.dataloader import DataParallelDataLoaderMixin
from src.trainers.sft import _CTOR_POSITIONS, DistributedSFTTrainer

PartialState()  # the trainer's accelerate logger requires an initialized state


class _Base:
    """Stands in for the HF Trainer base: sets the model-signature columns when unset."""

    def __init__(self):
        self._signature_columns = None
        self.data_collator = None

    def _set_signature_columns_if_needed(self):
        if self._signature_columns is None:
            self._signature_columns = ["input_ids", "labels"]


class _Trainer(DataParallelDataLoaderMixin, _Base):
    pass


def test_collator_columns_survive_signature_pruning():
    trainer = _Trainer()
    trainer.data_collator = types.SimpleNamespace(required_dataset_columns=("seq_lengths",))
    trainer._set_signature_columns_if_needed()
    assert "seq_lengths" in trainer._signature_columns

    # Idempotent: a second call must not duplicate the column.
    trainer._set_signature_columns_if_needed()
    assert trainer._signature_columns.count("seq_lengths") == 1


def test_collator_without_declared_columns_changes_nothing():
    trainer = _Trainer()
    trainer._set_signature_columns_if_needed()
    assert trainer._signature_columns == ["input_ids", "labels"]


def test_packing_collator_declares_seq_lengths():
    assert "seq_lengths" in DataCollatorWithPacking.required_dataset_columns


def test_union_override_wins_over_trl_base():
    """TRL's SFT base defines the method too; the mixin's union must be first in the MRO, or the
    collator's declaration silently stops reaching the pruning set."""
    assert (
        DistributedSFTTrainer._set_signature_columns_if_needed
        is DataParallelDataLoaderMixin._set_signature_columns_if_needed
    )


def test_positional_collator_reaches_cp_gate():
    def _init_cfg(self, kwargs, **_):
        self.parallelism_config = types.SimpleNamespace(is_cp_mode=True)
        return kwargs

    # ``object.__new__``: isinstance is all the gate reads, and the real ctor wants a tokenizer.
    collator = object.__new__(DataCollatorWithPacking)
    # Placed at TRL's own data_collator slot, as a signature-following caller would pass it.
    ctor_args: list = [None] * (_CTOR_POSITIONS["data_collator"] + 1)
    ctor_args[_CTOR_POSITIONS["data_collator"]] = collator
    with (
        mock.patch.object(DistributedSFTTrainer, "_init_distributed_config", _init_cfg),
        mock.patch.object(SFTTrainer, "__init__", return_value=None),
        mock.patch.object(DistributedSFTTrainer, "_setup_distributed_modes", return_value=None),
    ):
        with pytest.raises(ValueError, match="context parallelism"):
            DistributedSFTTrainer(*ctor_args)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
