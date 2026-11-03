#!/usr/bin/env python
"""A per-rank shard save writes no ``optimizer.pt``, the same bytes a second time with a full device-to-host copy,
and still writes ``scheduler.pt``."""

import os
from types import SimpleNamespace

import pytest
import torch
from transformers import Trainer, TrainerState

from halo.checkpoint.parallel.optimizer import OptimizerShardStore
from halo.trainers.distributed_trainer import DistributedTrainerMixin
from tests.common.base_save import BaseTrainerSave
from tests.common.checkpoint_io import load_context

# Spelled literally: importing the writer's constants would make a rename invisible here.
BASE_OPTIMIZER_FILES = ("optimizer.pt", "optimizer.bin")
SCHEDULER_FILE = "scheduler.pt"
SHARD_FILE = "optimizer_shard_00000.pt"
META_FILE = "optimizer_meta.pt"
SCHEDULER_STEPS = 3


class _HFOptimizerSave(BaseTrainerSave):
    _save_optimizer_and_scheduler = Trainer._save_optimizer_and_scheduler


class _Trainer(DistributedTrainerMixin, _HFOptimizerSave):
    def __init__(self, run_dir: str, *, sharded: bool, save_only_model: bool):
        self.run_dir = run_dir
        self.args = SimpleNamespace(
            save_total_limit=None,
            save_only_model=save_only_model,
            should_save=True,
            push_to_hub=False,
            include_num_input_tokens_seen="no",
        )
        self.state = TrainerState(global_step=2)
        self.parallelism_config = SimpleNamespace(is_tp_mode=False, merge_expert_lora_on_save=False)
        self.is_deepspeed_enabled = self.is_fsdp_enabled = False
        model = torch.nn.Linear(4, 4)
        self.optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lambda step: 0.5**step)
        model.weight.grad = torch.zeros_like(model.weight)
        for _ in range(SCHEDULER_STEPS):
            self.optimizer.step()
            self.lr_scheduler.step()
        self._store = OptimizerShardStore(
            load_context(model, optimizer=self.optimizer, lr_scheduler=self.lr_scheduler, fsdp_wrapped=sharded)
        )

    def save_model(self, output_dir=None, _internal_call=False):
        self._mark_model_save_collectives_done()

    def _optimizer_store(self) -> OptimizerShardStore:
        return self._store


def _save(tmp_path, monkeypatch, *, sharded: bool, save_only_model: bool = False) -> tuple[list[str], str]:
    """Every file name ``torch.save`` wrote, and the checkpoint directory."""
    written = []
    real_save = torch.save

    def recording_save(obj, f, *args, **kwargs):
        written.append(os.path.basename(f))
        return real_save(obj, f, *args, **kwargs)

    monkeypatch.setattr(torch, "save", recording_save)
    trainer = _Trainer(str(tmp_path), sharded=sharded, save_only_model=save_only_model)
    assert trainer._store.writes_shards is sharded, "premise: the store's mode"
    trainer._save_checkpoint(model=None, trial=None)
    return written, os.path.join(str(tmp_path), "checkpoint-2")


def _scheduler_epoch(checkpoint: str) -> int:
    return torch.load(os.path.join(checkpoint, SCHEDULER_FILE), weights_only=False)["last_epoch"]


def test_a_sharded_save_never_writes_the_base_optimizer_file(tmp_path, monkeypatch):
    written, checkpoint = _save(tmp_path, monkeypatch, sharded=True)

    assert not set(written) & set(BASE_OPTIMIZER_FILES), f"the save rank's state was written twice: {written}"
    assert not set(os.listdir(checkpoint)) & set(BASE_OPTIMIZER_FILES)
    assert written.count(SHARD_FILE) == 1 and os.path.isfile(os.path.join(checkpoint, META_FILE)), written
    assert written.count(SCHEDULER_FILE) == 1, f"scheduler.pt must land once: {written}"
    assert _scheduler_epoch(checkpoint) == SCHEDULER_STEPS


def test_an_unsharded_save_keeps_the_base_optimizer_and_scheduler_files(tmp_path, monkeypatch):
    written, checkpoint = _save(tmp_path, monkeypatch, sharded=False)

    assert written.count("optimizer.pt") == 1, f"the base save is the optimizer state here: {written}"
    assert written.count(SCHEDULER_FILE) == 1, f"scheduler.pt must land once: {written}"
    assert SHARD_FILE not in written and META_FILE not in written
    assert _scheduler_epoch(checkpoint) == SCHEDULER_STEPS


@pytest.mark.parametrize("sharded", [True, False], ids=["sharded", "unsharded"])
def test_a_weights_only_save_still_writes_the_scheduler(tmp_path, monkeypatch, sharded):
    written, checkpoint = _save(tmp_path, monkeypatch, sharded=sharded, save_only_model=True)

    assert not set(written) & {*BASE_OPTIMIZER_FILES, SHARD_FILE, META_FILE}, written
    assert written.count(SCHEDULER_FILE) == 1, written
    assert _scheduler_epoch(checkpoint) == SCHEDULER_STEPS


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
