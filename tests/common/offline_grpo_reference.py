"""Mapped-reference fixtures shared by offline GRPO storage and trainer tests."""

import os
from types import SimpleNamespace

import torch
from accelerate import PartialState
from datasets import Dataset

from src.distributed.runtime import DeferredRankFailure, fs_aware_save_rank
from src.trainers.grpo.reference_cache import ReferenceScoreCache
from src.trainers.grpo.reference_logps import OfflineGRPOReferenceLogpsMixin

SETTINGS = {"max_prompt_length": 8, "max_completion_length": 6, "pad_token_id": 0}


def reference_dataset() -> Dataset:
    return Dataset.from_dict(
        {
            "prompt_input_ids": [[11, 12], [21, 22, 23], [31]],
            "completion_input_ids": [[13, 14], [24], []],
            "group_id": [0, 1, 2],
        }
    )


def reference_rows() -> list[torch.Tensor]:
    return [torch.tensor([-0.25, -1.5]), torch.tensor([-0.75]), torch.empty(0)]


def mapped_scores(output_dir, dataset, rows):
    cache = ReferenceScoreCache(output_dir, dp_size=1)
    guard = DeferredRankFailure("Preparing fixture reference rows", exc_type=ValueError)
    try:
        if fs_aware_save_rank():
            guard.run(lambda: cache.append_rows(0, rows))
        guard.reject()
        return cache.finish(dataset)
    except BaseException:
        cache.discard()
        raise


def attach_reference(trainer, dataset, split, rows, *, settings=SETTINGS):
    scores = mapped_scores(trainer.output_dir, dataset, rows)
    return trainer._attach_scored_reference_logps(dataset, split, scores, settings=settings)


class _CheckpointBase:
    def _persist_trainer_sidecars(self, checkpoint_dir):
        pass

    def _rotate_checkpoints_after_sidecars(self, trial):
        self.rotated = True


class ReferenceStorageTrainer(OfflineGRPOReferenceLogpsMixin, _CheckpointBase):
    def __init__(self, output_dir, *, checkpoint=None, step=1):
        PartialState()
        self.output_dir = output_dir
        self.state = SimpleNamespace(global_step=step)
        self.rotated = False
        self._init_reference_logps(resume_checkpoint=checkpoint)

    def save_checkpoint(self):
        self._persist_trainer_sidecars(os.path.join(str(self.output_dir), f"checkpoint-{self.state.global_step}"))
        self._rotate_checkpoints_after_sidecars(None)

    def _get_output_dir(self, trial):
        return str(self.output_dir)
