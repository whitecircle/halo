"""Offline GRPO's training checkpoint and evaluation-row identity belong to its trainer."""

import datetime
import hashlib
import os

import numpy as np
import pytest
from datasets import Dataset, Features, Sequence, Value

from src.data.collators.offline_grpo import REF_PER_TOKEN_LOGPS_COLUMN
from src.trainers.grpo.reference_lifecycle import OfflineGRPOReferenceLifecycleMixin, _token_row_keys
from tests.common.gloo import run_gloo_ranks
from tests.common.offline_grpo_reference import (
    ReferenceStorageTrainer,
    attach_reference,
    reference_dataset,
    reference_rows,
    restore_reference,
)


class _TrainingBase:
    def train(self, resume_from_checkpoint=None):
        return resume_from_checkpoint


class _Trainer(OfflineGRPOReferenceLifecycleMixin, ReferenceStorageTrainer, _TrainingBase):
    def __init__(self, output_dir, *, checkpoint=None):
        super().__init__(output_dir, checkpoint=checkpoint)
        self.beta = 0.2
        self._precompute_reference = True


def test_reference_tracking_keeps_the_attached_dataset_not_a_second_token_table(tmp_path):
    trainer = _Trainer(tmp_path)
    attached = attach_reference(trainer, reference_dataset(), "train", reference_rows())
    assert trainer._reference_dataset_by_split["train"] is attached
    assert set(trainer._reference_logps_by_split["train"]) == {"num_rows", "token_digests", "settings"}
    assert not trainer._resumed_reference_logps


def test_noncp_resume_preserves_run_start_scores_and_requires_its_declared_checkpoint(tmp_path):
    trainer = _Trainer(tmp_path)
    attached = attach_reference(trainer, reference_dataset(), "train", reference_rows())
    trainer.save_checkpoint()
    checkpoint = str(tmp_path / "checkpoint-1")
    resumed = _Trainer(tmp_path, checkpoint=checkpoint)
    restored = restore_reference(resumed, reference_dataset(), "train")
    assert restored[REF_PER_TOKEN_LOGPS_COLUMN] == attached[REF_PER_TOKEN_LOGPS_COLUMN]
    with pytest.raises(ValueError, match="same checkpoint"):
        resumed.train()
    assert resumed.train(resume_from_checkpoint=checkpoint) == checkpoint


def test_late_or_different_resume_cannot_follow_a_fresh_reference_sweep(tmp_path):
    fresh = _Trainer(tmp_path)
    assert not fresh._reference_logps_by_split
    assert fresh.train() is None
    with pytest.raises(ValueError, match="before constructing"):
        fresh.train(resume_from_checkpoint=str(tmp_path / "checkpoint-1"))
    resumed = _Trainer(tmp_path, checkpoint=str(tmp_path / "checkpoint-1"))
    with pytest.raises(ValueError, match="same checkpoint"):
        resumed.train()
    with pytest.raises(ValueError, match="same checkpoint"):
        resumed.train(resume_from_checkpoint=str(tmp_path / "checkpoint-2"))
    assert resumed.train(resume_from_checkpoint=str(tmp_path / "checkpoint-1")) == str(tmp_path / "checkpoint-1")


def _ranked_late_resume(rank, root):
    trainer = _Trainer(root)
    with pytest.raises(ValueError, match="before constructing"):
        trainer.train(resume_from_checkpoint=os.path.join(root, "checkpoint-1") if rank == 1 else None)


def test_late_resume_on_one_rank_is_rejected_on_every_rank(tmp_path):
    run_gloo_ranks(_ranked_late_resume, 2, str(tmp_path), pg_timeout=datetime.timedelta(seconds=15))


@pytest.mark.parametrize("dtype", ["int32", "int64"])
def test_token_row_identity_matches_content_across_arrow_widths_and_chunks(dtype):
    dataset = Dataset.from_dict(
        {"prompt_input_ids": [[11, 12], [21, 22, 23], [31]], "completion_input_ids": [[13, 14], [24], []]},
        features=Features({name: Sequence(Value(dtype)) for name in ("prompt_input_ids", "completion_input_ids")}),
    )
    expected = []
    for prompt, completion in zip(dataset["prompt_input_ids"], dataset["completion_input_ids"], strict=True):
        expected.append(
            hashlib.sha256(
                np.asarray([len(prompt), *prompt, len(completion), *completion], dtype=np.int64).tobytes()
            ).digest()
        )
    assert list(_token_row_keys(dataset)) == expected
    assert list(_token_row_keys(dataset.select([1, 0, 1, 2]))) == [expected[1], expected[0], expected[1], expected[2]]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
