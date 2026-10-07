#!/usr/bin/env python
"""Offline GRPO activation checks that require the real trainer and grouped loader."""

import datetime
from types import SimpleNamespace

import pytest
from accelerate import PartialState
from datasets import Dataset
from torch import nn

import src.trainers.grpo.offline as offline_module
from src.trainers.grpo.offline import OfflineGRPOTrainer
from src.trainers.mixins.checkpointing import CheckpointingMixin
from src.trainers.mixins.reference_logps import ReferenceLogpsCheckpointMixin
from tests.common.gloo import run_gloo_ranks


def _real_loader_trainer():
    PartialState(cpu=True)
    trainer = OfflineGRPOTrainer.__new__(OfflineGRPOTrainer)
    trainer.eval_dataset = Dataset.from_dict({"row_id": [0, 1, 2, 3, 4, 5], "group_id": [0, 0, 1, 1, 2, 2]})
    trainer._cached_eval_group_ids = list(trainer.eval_dataset["group_id"])
    trainer._eval_dataloaders = {}
    trainer.dp_shard_geometry = lambda: (1, 0)
    trainer.data_collator = lambda rows: rows
    trainer._prepare_dataloader = lambda loader, **kwargs: loader
    trainer.args = SimpleNamespace(
        per_device_eval_batch_size=1,
        dataloader_num_workers=1,
        dataloader_persistent_workers=True,
        dataloader_pin_memory=False,
        dataloader_prefetch_factor=1,
    )
    return trainer


def test_dynamic_evaluation_builds_its_own_real_grouped_loader_and_persistent_cache():
    trainer = _real_loader_trainer()
    original = trainer.get_eval_dataloader()
    dynamic = trainer.eval_dataset.select([4, 0, 5, 1])
    current = trainer.get_eval_dataloader(dynamic)
    indices = list(current.batch_sampler.sampler)
    assert [current.dataset[index]["row_id"] for index in indices] == [4, 5, 0, 1]
    assert max(indices) < len(dynamic), "the constructor's longer group IDs indexed beyond the new rows"
    assert current is not original
    assert trainer.get_eval_dataloader(dynamic) is current
    assert trainer.get_eval_dataloader() is original
    assert len(trainer._eval_dataloaders) == 2


def test_reference_cache_constructor_failure_preserves_the_live_training_mode(tmp_path, monkeypatch):
    trainer = OfflineGRPOTrainer.__new__(OfflineGRPOTrainer)
    trainer.model = nn.Linear(1, 1).train()
    trainer.args = SimpleNamespace(per_device_train_batch_size=2, output_dir=str(tmp_path))
    trainer.dp_shard_geometry = lambda: (1, 0)
    trainer._data_parallel_rank_by_global_rank = lambda: [0]
    trainer.data_collator = lambda rows: rows

    def fail_cache(*args, **kwargs):
        raise OSError("cannot create reference cache")

    monkeypatch.setattr(offline_module, "ReferenceScoreCache", fail_cache)
    dataset = Dataset.from_dict(
        {"prompt_input_ids": [[1], [2], [3]], "completion_input_ids": [[4, 5], [6], []], "group_id": [0, 1, 2]}
    )
    with pytest.raises(OSError, match="cannot create reference cache"):
        trainer._sweep_reference_logps(dataset, "evaluation")
    assert trainer.model.training


def test_trainer_sidecar_hook_is_shared_and_does_not_override_checkpoint_rotation():
    assert OfflineGRPOTrainer._persist_trainer_sidecars is ReferenceLogpsCheckpointMixin._persist_trainer_sidecars
    assert (
        OfflineGRPOTrainer._rotate_checkpoints_after_sidecars is CheckpointingMixin._rotate_checkpoints_after_sidecars
    )


def _ranked_dynamic_loader(rank: int) -> None:
    trainer = _real_loader_trainer()
    builds = []
    trainer._prepare_dataloader = lambda loader, **kwargs: builds.append(loader) or loader
    dynamic = trainer.eval_dataset.select([4, 0, 5, 1])
    # A transform the fingerprinter cannot pickle leaves each process its own random fingerprint.
    dynamic._fingerprint = f"process-local-{rank}"
    first = trainer.get_eval_dataloader(dynamic)
    assert trainer.get_eval_dataloader(dynamic) is first, "the persistent loader was rebuilt"
    assert len(builds) == 1
    if rank == 1:
        trainer._eval_dataloaders.clear()
    after_split = trainer.get_eval_dataloader(dynamic)
    assert len(builds) == 2, "a rank skipped the rebuild (and its collectives) a peer entered"
    # The rank still holding its loader keeps it: its persistent workers are already running.
    assert (after_split is first) == (rank == 0), "the cache was displaced, or a rank reused nothing"
    assert trainer.get_eval_dataloader(dynamic) is after_split, "the split left a rank without a cached loader"
    assert len(builds) == 2


def test_a_process_local_fingerprint_neither_refuses_nor_splits_the_persistent_loader_cache():
    run_gloo_ranks(_ranked_dynamic_loader, 2, pg_timeout=datetime.timedelta(seconds=30))


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
