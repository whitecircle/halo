#!/usr/bin/env python
"""
CPU contract test for the KTO trainer's TRL integration.

Guards the wiring that GPU tests can't cheaply cover:
  - the distributed subclass really composes DistributedTrainerMixin with the
    correct TRL base, and declares the documented parallelism support flags;
  - the script-args dataclass constructs with sane defaults;
  - the pipeline loss reads TRL's own unpaired-preference collator batch wrapped to the pipeline's
    fixed shape.

Run: python tests/cpu/trainers/test_kto_trainer.py
"""

from types import SimpleNamespace

import pytest
from trl import KTOTrainer
from trl.trainer.kto_trainer import DataCollatorForUnpairedPreference

from halo.configs.preference import KTOScriptArguments
from halo.data.collators.fixed_shape import FixedShapeCollator
from halo.distributed.pipeline_parallel.runner import resolve_batch_pad_values
from halo.kernels.logprobs import LABEL_IGNORE_INDEX
from halo.trainers.distributed_trainer import DistributedTrainerMixin
from halo.trainers.preference.kto import DistributedKTOTrainer


def test_kto_trainer_mro_and_flags():
    assert issubclass(DistributedKTOTrainer, DistributedTrainerMixin)
    assert issubclass(DistributedKTOTrainer, KTOTrainer)
    # KTO mirrors DPO: CP no (full-sequence log-prob pooling + KL ref),
    # PP yes (apo_zero_unpaired only, precompute-only — gated in _reject_unsupported_pp_options).
    assert DistributedKTOTrainer.SUPPORTS_CP is False
    assert DistributedKTOTrainer.PP_REFUSAL is None
    assert "kto" in DistributedKTOTrainer._tag_names


def test_kto_args_defaults():
    args = KTOScriptArguments()
    assert args.completion_field == "completion"
    assert args.label_field == "label"


def test_the_pipeline_batch_transform_reads_the_fixed_shape_collators_batch():
    """Under PP the trainer's collator is wrapped to pad the batch to ``max_length``; the adapter's
    ``pad_spec`` must carry ``completion_mask`` with ``input_ids``, or the transform sees two widths. The
    scored tokens are the completion's, at the positions TRL's completion mask marks; the reference and the
    label ride along per row."""
    rows = [
        {"prompt_ids": [5, 6, 7], "completion_ids": [8, 9], "label": True, "ref_logps": -1.5},
        {"prompt_ids": [5], "completion_ids": [10, 11, 12], "label": False, "ref_logps": -2.5},
    ]
    trainer = DistributedKTOTrainer.__new__(DistributedKTOTrainer)
    trainer._tokenizer = SimpleNamespace(pad_token_id=0)
    trainer.args = SimpleNamespace(max_length=8)
    adapter = trainer._pp_loss_adapter()
    collator = FixedShapeCollator(
        DataCollatorForUnpairedPreference(pad_token_id=0), 8, resolve_batch_pad_values(0, adapter.pad_spec)
    )

    out = adapter.batch_transform(collator(rows))

    assert out["input_ids"].shape == out["labels"].shape == (2, 8)
    assert [labels[labels != LABEL_IGNORE_INDEX].tolist() for labels in out["labels"]] == [[8, 9], [10, 11, 12]]
    assert out["attention_mask"].sum(dim=1).tolist() == [5, 4]
    assert out["ref_logps"].tolist() == [-1.5, -2.5]
    assert out["label"].tolist() == [1.0, 0.0]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
