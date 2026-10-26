#!/usr/bin/env python
"""
CPU contract test for the KTO trainer's TRL integration.

Guards the wiring that GPU tests can't cheaply cover:
  - the distributed subclass really composes DistributedTrainerMixin with the
    correct TRL base, and declares the documented parallelism support flags;
  - the script-args dataclass constructs with sane defaults;
  - the pipeline loss reads TRL's own unpaired-preference collator batch.

Run: python tests/cpu/trainers/test_kto_trainer.py
"""

from types import SimpleNamespace

import pytest
from trl import KTOTrainer
from trl.trainer.kto_trainer import DataCollatorForUnpairedPreference

from src.configs.preference import KTOScriptArguments
from src.kernels.logprobs import LABEL_IGNORE_INDEX
from src.trainers.mixins.base import DistributedTrainerMixin
from src.trainers.preference.kto import DistributedKTOTrainer


def test_kto_trainer_mro_and_flags():
    assert issubclass(DistributedKTOTrainer, DistributedTrainerMixin)
    assert issubclass(DistributedKTOTrainer, KTOTrainer)
    # KTO mirrors DPO: CP no (full-sequence log-prob pooling + KL ref),
    # PP yes (apo_zero_unpaired only, precompute-only — gated in _validate_pp_mode).
    assert DistributedKTOTrainer.SUPPORTS_CP is False
    assert DistributedKTOTrainer.PP_REFUSAL is None
    assert "kto" in DistributedKTOTrainer._tag_names


def test_kto_args_defaults():
    args = KTOScriptArguments()
    assert args.completion_field == "completion"
    assert args.label_field == "label"


def test_the_pipeline_batch_transform_reads_trls_collator_output():
    """The scored tokens are the completion's, at the positions TRL's completion mask marks, padded to
    the pipeline's fixed ``max_length``; the reference and the label ride along per row."""
    rows = [
        {"prompt_ids": [5, 6, 7], "completion_ids": [8, 9], "label": True, "ref_logps": -1.5},
        {"prompt_ids": [5], "completion_ids": [10, 11, 12], "label": False, "ref_logps": -2.5},
    ]
    batch = DataCollatorForUnpairedPreference(pad_token_id=0)(rows)
    trainer = DistributedKTOTrainer.__new__(DistributedKTOTrainer)
    trainer._tokenizer = SimpleNamespace(pad_token_id=0)
    trainer.args = SimpleNamespace(max_length=8)

    out = trainer._pp_kto_batch_transform(batch)

    assert out["input_ids"].shape == out["labels"].shape == (2, 8)
    assert [labels[labels != LABEL_IGNORE_INDEX].tolist() for labels in out["labels"]] == [[8, 9], [10, 11, 12]]
    assert out["attention_mask"].sum(dim=1).tolist() == [5, 4]
    assert out["ref_logps"].tolist() == [-1.5, -2.5]
    assert out["label"].tolist() == [1.0, 0.0]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
