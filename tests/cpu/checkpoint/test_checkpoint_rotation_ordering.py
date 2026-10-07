#!/usr/bin/env python
"""Checkpoint rotation must run only after the toolkit's optimizer sidecars are complete.

HF Trainer's ``_save_checkpoint`` ends by rotating old checkpoints (an rmtree, honoring
``save_total_limit``); the mixin then writes its own sidecars — scheduler.pt under
``save_only_model``, the trainer's own (``_persist_trainer_sidecars``: the precomputed DPO/KTO
reference log-probs), and the per-rank optimizer shards that REPLACE the base's rank-0-only
optimizer.pt. With ``save_total_limit: 1`` that ordering opens a window where the previous
checkpoint is already deleted and the new one has no optimizer state yet — a preemption there
leaves exactly one checkpoint that warm-restarts the optimizer of a multi-day run. The mixin
therefore neutralizes the base's rotation (``save_total_limit=None`` for the duration of the
``super()`` call), deletes the stale optimizer.pt/.bin only after its replacement shards are on
disk, and rotates itself as the true last step — and not at all when the save failed, because
after a failed save the old checkpoints are the only good ones. A ``merge_expert_lora_on_save``
checkpoint's resume adapter is one more such sidecar. The base's checkpoint push to the Hub, a
background upload of the directory as it stands, waits for the complete checkpoint the same way.

The trainer here is a stub over ``BaseTrainerSave``, HF's save in its own order (the checkpoint
directory, optimizer.pt/.bin, the trainer state, then the real rotation — rotation-last being the
hazard under test), and a recording shard writer stands in for ``OptimizerShardStore.save``.

    python tests/cpu/checkpoint/test_checkpoint_rotation_ordering.py
"""

import os
import time
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
from transformers import TrainerState
from transformers.trainer_utils import rotate_checkpoints

import src.distributed.checkpoint.optimizer as optimizer_mod
import src.trainers.mixins.checkpointing as checkpointing_mod
from src.distributed.checkpoint.context import CheckpointLoadContext
from src.distributed.checkpoint.optimizer import OptimizerShardStore
from src.trainers.mixins.base import DistributedTrainerMixin
from tests.common.base_save import BaseTrainerSave

OPTIMIZER_FILES = ("optimizer.pt", "optimizer.bin")


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


class _Trainer(DistributedTrainerMixin, BaseTrainerSave):
    """The mixin's _save_checkpoint over HF's save — no heavy trainer construction."""

    def __init__(self, run_dir, *, fsdp_wrapped=True, fail_shards=False, store=None, merge_expert_lora_on_save=False):
        self.run_dir = run_dir
        self.events = []
        self.args = SimpleNamespace(save_total_limit=1, save_only_model=False, should_save=True, push_to_hub=False)
        self.state = TrainerState(global_step=2)
        self.parallelism_config = SimpleNamespace(
            is_tp_mode=False, merge_expert_lora_on_save=merge_expert_lora_on_save
        )
        self._fsdp_wrapped = fsdp_wrapped
        self.lr_scheduler = None
        # ``store``: the REAL OptimizerShardStore, for the cases whose verdict is its own.
        self._shard_writer = store or _RecordingShardWriter(self, fail=fail_shards)

    def save_model(self, output_dir=None, _internal_call=False):
        """The base save's first step, recording the rotation limit the base save runs under."""
        self.events.append(("base_save", self.args.save_total_limit))
        self._mark_model_save_collectives_done()

    def _save_optimizer_and_scheduler(self, output_dir):
        """The base's rank-0-only optimizer view, wrong under sharding, in both spellings."""
        for name in OPTIMIZER_FILES:
            with open(os.path.join(output_dir, name), "wb") as fh:
                fh.write(b"rank0-view")

    def _push_from_checkpoint(self, checkpoint_folder):
        """Stands in for the base's Hub push, which uploads the directory in the background as it
        stands when called: records that listing."""
        self.events.append(("push", sorted(os.listdir(checkpoint_folder))))

    def _persist_trainer_sidecars(self, checkpoint_dir):
        """Where the trainer-sidecar hook lands in the window: which directory it was handed, and
        whether the previous checkpoint was still on disk when it wrote."""
        self.events.append(
            (
                "trainer_sidecars",
                os.path.relpath(checkpoint_dir, self.run_dir),
                os.path.isdir(os.path.join(self.run_dir, "checkpoint-1")),
            )
        )

    def _optimizer_store(self):
        return self._shard_writer

    def _checkpoint_context(self):
        return "checkpoint context"


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
    assert trainer.events == [
        ("base_save", None),
        ("trainer_sidecars", "checkpoint-2", True),
        ("shards_written", True, True),
        ("rotate", 1),
    ]
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

    assert trainer.events == [("base_save", None), ("trainer_sidecars", "checkpoint-2", True), ("rotate", 1)]
    assert not os.path.isdir(previous)
    assert os.path.isfile(os.path.join(str(tmp_path), "checkpoint-2", "optimizer.pt"))


@pytest.mark.parametrize(
    ("fsdp_wrapped", "save_only_model"),
    [(True, False), (False, False), (True, True)],
    ids=["sharded-optimizer", "base-optimizer", "save_only_model"],
)
def test_the_checkpoint_push_uploads_the_complete_checkpoint(tmp_path, monkeypatch, fsdp_wrapped, save_only_model):
    """``hub_strategy: checkpoint`` uploads the checkpoint directory from the base save, before the mixin
    commits the trainer state and writes the optimizer shards. The push must start once, after the
    commit and before rotation, on the directory as a resume would read it."""
    trainer = _Trainer(str(tmp_path), fsdp_wrapped=fsdp_wrapped)
    trainer.args.push_to_hub = True
    trainer.args.save_only_model = save_only_model
    _plant_previous_checkpoint(str(tmp_path))
    _record_rotation(monkeypatch, trainer.events)

    trainer._save_checkpoint(model=None, trial=None)

    pushes = [event for event in trainer.events if event[0] == "push"]
    assert len(pushes) == 1, f"pushed {len(pushes)} times: {trainer.events}"
    assert trainer.events.index(pushes[0]) == len(trainer.events) - 2, "the push must precede rotation"
    expected = {"trainer_state.json"}
    if not save_only_model:
        expected |= {"optimizer_shard_00000.pt", "optimizer_meta.pt"} if fsdp_wrapped else set(OPTIMIZER_FILES)
    assert set(pushes[0][1]) == expected, f"the push saw {pushes[0][1]}"
    assert trainer.args.push_to_hub is True, "the caller's push_to_hub must be restored"


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


@pytest.mark.parametrize("save_only_model", [False, True], ids=["exact-resume", "save_only_model"])
def test_a_merged_checkpoint_gets_its_resume_adapter_before_rotation(tmp_path, monkeypatch, save_only_model):
    """A ``merge_expert_lora_on_save`` checkpoint resumes from its unmerged adapters, a sidecar like
    the optimizer shards: written into the new checkpoint while the previous one still exists, and
    under ``save_only_model`` too, where the adapters are still the only exact trained weights. The
    save first unmarks the step's directory, before the base save writes anything into it: a resumed
    run saving a step it already saved rewrites that checkpoint, whose old marker would otherwise
    vouch for the old adapter until the new one lands."""
    trainer = _Trainer(str(tmp_path), merge_expert_lora_on_save=True)
    trainer.args.save_only_model = save_only_model
    previous = _plant_previous_checkpoint(str(tmp_path))
    _record_rotation(monkeypatch, trainer.events)

    def recording_unmark(checkpoint_dir):
        trainer.events.append(("unmark", checkpoint_dir))

    def recording_resume_adapter(ctx, checkpoint_dir):
        trainer.events.append(("resume_adapter", ctx, checkpoint_dir, os.path.isdir(previous)))

    monkeypatch.setattr(checkpointing_mod, "remove_stale_completion_markers", recording_unmark)
    monkeypatch.setattr(checkpointing_mod, "save_resume_adapter", recording_resume_adapter)

    trainer._save_checkpoint(model=None, trial=None)

    new_ckpt = os.path.join(str(tmp_path), "checkpoint-2")
    unmark = ("unmark", new_ckpt)
    resume_adapter = ("resume_adapter", "checkpoint context", new_ckpt, True)
    trainer_sidecars = ("trainer_sidecars", "checkpoint-2", True)
    shards = [] if save_only_model else [("shards_written", True, True)]
    assert trainer.events == [unmark, ("base_save", None), resume_adapter, trainer_sidecars, *shards, ("rotate", 1)]


def test_a_torn_resave_of_a_saved_step_does_not_vouch_for_the_abandoned_save(tmp_path, monkeypatch):
    """A run resumed from an earlier checkpoint saves a step it saved before, in place. Stopped after
    the base save but before its own optimizer shards, the directory holds this save's weights beside
    the abandoned save's trainer state, shard and meta. Those markers must be gone before anything is
    written: left in place, the resume detection takes the directory for complete and the optimizer
    restore loads the abandoned run's moments under this run's weights."""
    run_dir = str(tmp_path)
    abandoned = os.path.join(run_dir, "checkpoint-2")
    os.makedirs(abandoned)
    _real_store_over_a_stepped_optimizer().save(abandoned)
    with open(os.path.join(abandoned, "trainer_state.json"), "w") as fh:
        fh.write('{"global_step": 2}')
    trainer = _Trainer(run_dir, fail_shards=True)
    _record_rotation(monkeypatch, trainer.events)

    with pytest.raises(RuntimeError, match="simulated preemption"):
        trainer._save_checkpoint(model=None, trial=None)

    left = sorted(os.listdir(abandoned))
    for marker in ("trainer_state.json", "optimizer_meta.pt"):
        assert marker not in left, f"the abandoned save's {marker} still vouches for the torn re-save: {left}"
    with pytest.raises(RuntimeError, match="torn set of an interrupted save"):
        _real_store_over_a_stepped_optimizer().load(abandoned)


def test_an_unmerged_run_writes_no_resume_adapter(tmp_path, monkeypatch):
    trainer = _Trainer(str(tmp_path))
    written = []
    monkeypatch.setattr(checkpointing_mod, "save_resume_adapter", lambda *args: written.append(args))

    trainer._save_checkpoint(model=None, trial=None)

    assert not written, "the non-merged save already writes the adapter as the checkpoint itself"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
