#!/usr/bin/env python
"""Checkpoint rotation runs only after the toolkit's optimizer sidecars are complete. HF rotates before the mixin
writes them, so with ``save_total_limit: 1`` a preemption there warm-restarts a multi-day run's optimizer."""

import os
import time
from types import SimpleNamespace

import pytest
from transformers import TrainerState
from transformers.trainer_utils import rotate_checkpoints

import halo.checkpoint.parallel.optimizer as optimizer_mod
import halo.trainers.loop.checkpointing as checkpointing_mod
from halo.checkpoint.parallel.optimizer import OptimizerShardStore
from halo.trainers.distributed_trainer import DistributedTrainerMixin
from tests.common.base_save import BaseTrainerSave
from tests.common.checkpoint_resave import stepped_store_context

OPTIMIZER_FILES = ("optimizer.pt", "optimizer.bin")


class _RecordingStore(OptimizerShardStore):
    """Records what the world looked like when its write began, the preemption window under test."""

    def __init__(self, trainer, *, fsdp_wrapped: bool, fail: bool):
        super().__init__(stepped_store_context(fsdp_wrapped=fsdp_wrapped))
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
        super().save(output_dir)


class _BaseOptimizerSave(BaseTrainerSave):
    """Both spellings: rank 0's whole optimizer view."""

    def _save_optimizer_and_scheduler(self, output_dir):
        for name in OPTIMIZER_FILES:
            with open(os.path.join(output_dir, name), "wb") as fh:
                fh.write(b"rank0-view")


class _Trainer(DistributedTrainerMixin, _BaseOptimizerSave):
    def __init__(self, run_dir, *, fsdp_wrapped=True, fail_shards=False, store=None, merge_expert_lora_on_save=False):
        self.run_dir = run_dir
        self.events = []
        self.args = SimpleNamespace(
            save_total_limit=1,
            save_only_model=False,
            should_save=True,
            push_to_hub=False,
            include_num_input_tokens_seen="no",
        )
        self.state = TrainerState(global_step=2)
        self.parallelism_config = SimpleNamespace(
            is_tp_mode=False, merge_expert_lora_on_save=merge_expert_lora_on_save
        )
        self._fsdp_wrapped = fsdp_wrapped
        self.lr_scheduler = None
        self._shard_store = store or _RecordingStore(self, fsdp_wrapped=fsdp_wrapped, fail=fail_shards)

    def save_model(self, output_dir=None, _internal_call=False):
        """Records the rotation limit the base save runs under."""
        self.events.append(("base_save", self.args.save_total_limit))
        self._mark_model_save_collectives_done()

    def _push_from_checkpoint(self, checkpoint_folder):
        """Records the listing the base's background upload would send."""
        self.events.append(("push", sorted(os.listdir(checkpoint_folder))))

    def _persist_trainer_sidecars(self, checkpoint_dir):
        """Records which directory it was handed and whether the previous checkpoint was still on disk."""
        self.events.append(
            (
                "trainer_sidecars",
                os.path.relpath(checkpoint_dir, self.run_dir),
                os.path.isdir(os.path.join(self.run_dir, "checkpoint-1")),
            )
        )

    def _optimizer_store(self):
        return self._shard_store

    def _checkpoint_context(self):
        return "checkpoint context"


def _plant_previous_checkpoint(run_dir) -> str:
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
    """raising=False: where the symbol is absent the patch is a no-op and the ordering asserts fail."""

    def recording_rotate(**kwargs):
        events.append(("rotate", kwargs["save_total_limit"]))
        rotate_checkpoints(**kwargs)

    monkeypatch.setattr(checkpointing_mod, "rotate_checkpoints", recording_rotate, raising=False)


def test_rotation_runs_after_the_optimizer_shards_are_on_disk(tmp_path, monkeypatch):
    trainer = _Trainer(str(tmp_path))
    previous = _plant_previous_checkpoint(str(tmp_path))
    _record_rotation(monkeypatch, trainer.events)

    trainer._save_checkpoint(model=None, trial=None)

    # Base save with rotation neutralized, shards while the previous checkpoint exists, then rotation at the real
    # limit.
    assert trainer.events == [
        ("base_save", None),
        ("trainer_sidecars", "checkpoint-2", True),
        ("shards_written", False, True),
        ("rotate", 1),
    ]
    assert trainer.args.save_total_limit == 1, "the caller's limit must be restored"

    new_ckpt = os.path.join(str(tmp_path), "checkpoint-2")
    assert not os.path.isdir(previous), "rotation must still happen once the checkpoint is whole"
    assert os.path.isfile(os.path.join(new_ckpt, "optimizer_shard_00000.pt"))
    assert os.path.isfile(os.path.join(new_ckpt, "optimizer_meta.pt"))
    for name in OPTIMIZER_FILES:
        assert not os.path.exists(os.path.join(new_ckpt, name)), (
            f"{name} is rank 0's own state written a second time beside its shard"
        )


def test_a_failed_shard_write_keeps_the_previous_checkpoint(tmp_path, monkeypatch):
    """The previous checkpoint is the only one with optimizer state."""
    trainer = _Trainer(str(tmp_path), fail_shards=True)
    previous = _plant_previous_checkpoint(str(tmp_path))
    _record_rotation(monkeypatch, trainer.events)

    with pytest.raises(RuntimeError, match="simulated preemption"):
        trainer._save_checkpoint(model=None, trial=None)

    assert os.path.isdir(previous), "a failed save must never cost the previous checkpoint"
    assert ("rotate", 1) not in trainer.events


def test_non_sharded_modes_still_rotate_after_the_super_call(tmp_path, monkeypatch):
    """The base's rotation was neutralized, so the deferred one must run here too; optimizer.pt is the artifact."""
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
    """``hub_strategy: checkpoint`` uploads from the base save, before the commit and shards; the push must start
    after the commit and before rotation."""
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
    return OptimizerShardStore(stepped_store_context())


def test_an_unproducible_optimizer_state_keeps_the_previous_checkpoint(tmp_path, monkeypatch):
    """FlashAdamW on uneven DTensors finds nothing to write on every save, which would rotate away the last
    checkpoint with state at exit code 0."""

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


@pytest.mark.parametrize("save_only_model", [False, True], ids=["exact-resume", "save_only_model"])
def test_a_merged_checkpoint_gets_its_resume_adapter_before_rotation(tmp_path, monkeypatch, save_only_model):
    """The save unmarks the directory first: a re-saved step's old marker would vouch for the old adapter until the
    new one lands."""
    trainer = _Trainer(str(tmp_path), merge_expert_lora_on_save=True)
    trainer.args.save_only_model = save_only_model
    previous = _plant_previous_checkpoint(str(tmp_path))
    _record_rotation(monkeypatch, trainer.events)

    def recording_unmark(checkpoint_dir):
        trainer.events.append(("unmark", checkpoint_dir))

    def recording_resume_adapter(ctx, checkpoint_dir):
        trainer.events.append(("resume_adapter", ctx, checkpoint_dir, os.path.isdir(previous)))

    monkeypatch.setattr(checkpointing_mod, "prepare_checkpoint_rewrite", recording_unmark)
    monkeypatch.setattr(checkpointing_mod, "save_resume_adapter", recording_resume_adapter)

    trainer._save_checkpoint(model=None, trial=None)

    new_ckpt = os.path.join(str(tmp_path), "checkpoint-2")
    unmark = ("unmark", new_ckpt)
    resume_adapter = ("resume_adapter", "checkpoint context", new_ckpt, True)
    trainer_sidecars = ("trainer_sidecars", "checkpoint-2", True)
    shards = [] if save_only_model else [("shards_written", False, True)]
    assert trainer.events == [unmark, ("base_save", None), resume_adapter, trainer_sidecars, *shards, ("rotate", 1)]


def test_a_torn_resave_of_a_saved_step_does_not_vouch_for_the_abandoned_save(tmp_path, monkeypatch):
    """Left in place, resume detection takes the directory as complete and restores the abandoned run's moments
    under this run's weights."""
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
    for name in ("trainer_state.json", "optimizer_meta.pt", "optimizer_shard_00000.pt", *OPTIMIZER_FILES):
        assert name not in left, f"{name} outlived the torn re-save: {left}"


def test_an_unmerged_run_writes_no_resume_adapter(tmp_path, monkeypatch):
    trainer = _Trainer(str(tmp_path))
    written = []
    monkeypatch.setattr(checkpointing_mod, "save_resume_adapter", lambda *args: written.append(args))

    trainer._save_checkpoint(model=None, trial=None)

    assert not written, "the non-merged save already writes the adapter as the checkpoint itself"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
