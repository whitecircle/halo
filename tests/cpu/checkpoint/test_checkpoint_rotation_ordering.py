#!/usr/bin/env python
"""Checkpoint rotation must run only after the toolkit's optimizer sidecars are complete.

HF Trainer's ``_save_checkpoint`` ends by rotating old checkpoints (an rmtree, honoring
``save_total_limit``); the mixin then writes its own sidecars — scheduler.pt under
``save_only_model``, and the per-rank optimizer shards that REPLACE the base's rank-0-only
optimizer.pt. With ``save_total_limit: 1`` that ordering opens a window where the previous
checkpoint is already deleted and the new one has no optimizer state yet — a preemption there
leaves exactly one checkpoint that warm-restarts the optimizer of a multi-day run. The mixin
therefore neutralizes the base's rotation (``save_total_limit=None`` for the duration of the
``super()`` call), deletes the stale optimizer.pt/.bin only after its replacement shards are on
disk, and rotates itself as the true last step — and not at all when the save failed, because
after a failed save the old checkpoints are the only good ones.

The trainer here is a stub: ``_RecordingBase`` stands in for HF's Trainer (writes the checkpoint
directory + optimizer.pt/.bin, then rotates — rotation-last being the hazard under test), and a
recording shard writer stands in for ``OptimizerShardStore.save``.

    python tests/cpu/checkpoint/test_checkpoint_rotation_ordering.py
"""

import os
import time
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
from transformers.trainer_utils import rotate_checkpoints

import src.distributed.checkpoint.optimizer as optimizer_mod
import src.trainers.mixins.checkpointing as checkpointing_mod
from src.distributed.checkpoint.context import CheckpointLoadContext
from src.distributed.checkpoint.optimizer import OptimizerShardStore
from src.trainers.mixins.base import DistributedTrainerMixin

OPTIMIZER_FILES = ("optimizer.pt", "optimizer.bin")


class _RecordingBase:
    """Stands in for HF's Trainer._save_checkpoint: write the checkpoint directory and the
    (rank-0-only, wrong-under-sharding) optimizer files, then rotate — exactly the base's last
    step, via the real transformers ``rotate_checkpoints`` so a non-neutralized limit really
    deletes the old checkpoint the way production would."""

    def _save_checkpoint(self, model, trial):
        self.events.append(("base_save", self.args.save_total_limit))
        ckpt = os.path.join(self.run_dir, f"checkpoint-{self.state.global_step}")
        os.makedirs(ckpt, exist_ok=True)
        for name in OPTIMIZER_FILES:
            with open(os.path.join(ckpt, name), "wb") as fh:
                fh.write(b"rank0-view")
        rotate_checkpoints(
            output_dir=self.run_dir,
            save_total_limit=self.args.save_total_limit,
            best_model_checkpoint=self.state.best_model_checkpoint,
            use_mtime=True,
        )


class _RecordingShardWriter:
    """Stands in for OptimizerShardStore.save: records what the world looked like at
    shard-write time (the preemption window under test), then writes the shard + meta files."""

    def __init__(self, trainer, fail=False):
        self.trainer = trainer
        self.fail = fail

    def save(self, output_dir):
        if self.fail:
            raise RuntimeError("optimizer shard write failed (simulated preemption)")
        self.trainer.events.append(
            (
                "shards_written",
                os.path.exists(os.path.join(output_dir, "optimizer.pt")),
                os.path.isdir(os.path.join(self.trainer.run_dir, "checkpoint-1")),
            )
        )
        with open(os.path.join(output_dir, "optimizer_shard_00000.pt"), "wb") as fh:
            fh.write(b"shard")
        with open(os.path.join(output_dir, "optimizer_meta.pt"), "wb") as fh:
            fh.write(b"meta")


class _Trainer(DistributedTrainerMixin, _RecordingBase):
    """The mixin's _save_checkpoint over the recording base — no heavy trainer construction."""

    def __init__(self, run_dir, *, fsdp_wrapped=True, fail_shards=False, store=None):
        self.run_dir = run_dir
        self.events = []
        self.args = SimpleNamespace(save_total_limit=1, save_only_model=False, should_save=True)
        self.state = SimpleNamespace(global_step=2, best_model_checkpoint=None)
        self.parallelism_config = SimpleNamespace(is_tp_mode=False)
        self._fsdp_wrapped = fsdp_wrapped
        self.lr_scheduler = None
        # ``store``: the REAL OptimizerShardStore, for the cases whose verdict is its own.
        self._shard_writer = store or _RecordingShardWriter(self, fail=fail_shards)

    def _get_output_dir(self, trial=None):
        return self.run_dir

    def _optimizer_store(self):
        return self._shard_writer


def _plant_previous_checkpoint(run_dir) -> str:
    """A complete back-dated checkpoint-1 — the good checkpoint rotation is allowed to remove only
    once its successor is whole."""
    previous = os.path.join(run_dir, "checkpoint-1")
    os.makedirs(previous)
    for name in ("model.safetensors", "optimizer_shard_00000.pt", "optimizer_meta.pt", "scheduler.pt"):
        with open(os.path.join(previous, name), "wb") as fh:
            fh.write(b"x")
    stamp = time.time() - 60
    for name in (previous, *(os.path.join(previous, f) for f in os.listdir(previous))):
        os.utime(name, (stamp, stamp))
    return previous


def _record_rotation(monkeypatch, events):
    """Route the mixin's deferred rotation through a recorder that still performs the real thing.
    raising=False: where the symbol is absent the patch is a no-op and the ordering asserts fail."""

    def recording_rotate(**kwargs):
        events.append(("rotate", kwargs["save_total_limit"]))
        rotate_checkpoints(**kwargs)

    monkeypatch.setattr(checkpointing_mod, "rotate_checkpoints", recording_rotate, raising=False)


def test_rotation_runs_after_the_optimizer_shards_are_on_disk(tmp_path, monkeypatch):
    trainer = _Trainer(str(tmp_path))
    previous = _plant_previous_checkpoint(str(tmp_path))
    _record_rotation(monkeypatch, trainer.events)

    trainer._save_checkpoint(model=None, trial=None)

    # The whole window, in order: the base saved with rotation NEUTRALIZED; the shards were written
    # while optimizer.pt was still in place AND the previous checkpoint still existed (nothing to
    # lose at any preemption point); only then did rotation run, at the caller's real limit.
    assert trainer.events == [("base_save", None), ("shards_written", True, True), ("rotate", 1)]
    assert trainer.args.save_total_limit == 1, "the caller's limit must be restored"

    new_ckpt = os.path.join(str(tmp_path), "checkpoint-2")
    assert not os.path.isdir(previous), "rotation must still happen once the checkpoint is whole"
    assert os.path.isfile(os.path.join(new_ckpt, "optimizer_shard_00000.pt"))
    assert os.path.isfile(os.path.join(new_ckpt, "optimizer_meta.pt"))
    for name in OPTIMIZER_FILES:
        assert not os.path.exists(os.path.join(new_ckpt, name)), (
            f"{name} is rank 0's view alone and must be replaced by the shards"
        )


def test_a_failed_shard_write_keeps_the_previous_checkpoint(tmp_path, monkeypatch):
    """A failure where the preemption would land: the shard write dies. Rotation must NOT run —
    the previous checkpoint is the only one with optimizer state, and rotating would delete it in
    favor of the incomplete newcomer."""
    trainer = _Trainer(str(tmp_path), fail_shards=True)
    previous = _plant_previous_checkpoint(str(tmp_path))
    _record_rotation(monkeypatch, trainer.events)

    with pytest.raises(RuntimeError, match="simulated preemption"):
        trainer._save_checkpoint(model=None, trial=None)

    assert os.path.isdir(previous), "a failed save must never cost the previous checkpoint"
    assert ("rotate", 1) not in trainer.events
    # The base's optimizer.pt survives too: it was not deleted ahead of shards that never came.
    assert os.path.isfile(os.path.join(str(tmp_path), "checkpoint-2", "optimizer.pt"))


def test_non_sharded_modes_still_rotate_after_the_super_call(tmp_path, monkeypatch):
    """The early-return branch (no FSDP2/pure-TP): the base's rotation was neutralized, so the
    deferred rotation must run there too — and the base's optimizer.pt stays, it IS the artifact."""
    trainer = _Trainer(str(tmp_path), fsdp_wrapped=False)
    previous = _plant_previous_checkpoint(str(tmp_path))
    _record_rotation(monkeypatch, trainer.events)

    trainer._save_checkpoint(model=None, trial=None)

    assert trainer.events == [("base_save", None), ("rotate", 1)]
    assert not os.path.isdir(previous)
    assert os.path.isfile(os.path.join(str(tmp_path), "checkpoint-2", "optimizer.pt"))


def _real_store_over_a_stepped_optimizer() -> OptimizerShardStore:
    """The production writer over a live model+optimizer — no stub in the path under test."""
    model = nn.Linear(4, 4)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    model.weight.grad = torch.zeros_like(model.weight)
    optimizer.step()
    noop = lambda *args, **kwargs: None  # noqa: E731 — the base-Trainer fallbacks are never reached here
    return OptimizerShardStore(
        CheckpointLoadContext(
            model=model,
            optimizer=optimizer,
            lr_scheduler=None,
            parallelism_config=None,
            is_pp_mode=False,
            is_cp_mode=False,
            is_tp_mode=False,
            has_ep_layers=False,
            fsdp_wrapped=True,
            tp_rank=0,
            tp_size=1,
            super_load_from_checkpoint=noop,
            super_load_optimizer_and_scheduler=noop,
        )
    )


def test_an_unproducible_optimizer_state_keeps_the_previous_checkpoint(tmp_path, monkeypatch):
    """The other half of the same window, and the silent one: the shard save does not fail, it finds
    it has NOTHING to write (``get_optimizer_state_dict`` refuses the live sharding — FlashAdamW on
    unevenly-sharded DTensors does this on every save). Skipping the write and returning lets this
    rotation delete the last checkpoint that HAD optimizer state, in favour of one that has none, at
    exit code 0. It must fail the checkpoint instead."""

    def refuse(*args, **kwargs):
        raise RuntimeError("FlashAdamW: unevenly-sharded DTensors have no state_dict")

    monkeypatch.setattr(optimizer_mod, "get_optimizer_state_dict", refuse)
    trainer = _Trainer(str(tmp_path), store=_real_store_over_a_stepped_optimizer())
    previous = _plant_previous_checkpoint(str(tmp_path))
    _record_rotation(monkeypatch, trainer.events)

    with pytest.raises(RuntimeError, match="save_only_model"):
        trainer._save_checkpoint(model=None, trial=None)

    assert os.path.isdir(previous), "the last checkpoint with optimizer state was rotated away"
    assert ("rotate", 1) not in trainer.events
    assert os.path.isfile(os.path.join(str(tmp_path), "checkpoint-2", "optimizer.pt")), (
        "the base's optimizer.pt was deleted ahead of shards that were never written"
    )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
