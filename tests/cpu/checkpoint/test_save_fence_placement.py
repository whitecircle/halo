#!/usr/bin/env python
"""Where the checkpoint fence may start: after the save's collectives, never around them.

``DeferredRankFailure`` turns a rank-local write failure into a uniform raise by recording it and
closing the region with an ``all_gather_object``. That is only sound for a region a rank can skip
without skipping a collective. HF's ``_save_checkpoint`` is not such a region: it calls this
toolkit's own ``save_model`` first — the strategy gathers, the barriers, PP's index exchange — and
only then makes its writer-local tail writes (rank-0 ``optimizer.pt``, RNG, ``trainer_state.json``).
Fencing the whole call means a rank whose disk fills in the FIRST half jumps straight to the guard's
gather while its peers block in an expert gather it never entered: mismatched collectives and a
watchdog timeout, which is the exact failure the fence exists to prevent.

So the mixin splits it: ``save_model`` records that its collectives are behind it, and
``_save_checkpoint`` defers only failures that carry that mark. Anything earlier is re-raised here
and now, rank-locally — the launcher tears the job down instead of the world timing out one rank at
a time. Both halves are pinned below, plus the mark on both of the mixin ``save_model``'s exits and
a structural sweep over every ``save_model`` in the trainer tree — the two that replace the mixin's
gather just as hard.

    python tests/cpu/checkpoint/test_save_fence_placement.py
"""

import ast
import os
import pathlib
from types import SimpleNamespace

import pytest
import torch.nn as nn

import src.trainers.mixins.checkpointing as checkpointing_mod
from src.distributed.checkpoint.context import CheckpointContext
from src.distributed.runtime import DeferredRankFailure
from src.trainers.mixins.base import DistributedTrainerMixin


class _BaseWithTailWrites:
    """Stands in for HF's ``_save_checkpoint``: the toolkit's ``save_model`` (world collectives),
    then its own writer-local tail writes — the shape the fence has to split."""

    def _save_checkpoint(self, model, trial):
        output_dir = os.path.join(self.run_dir, f"checkpoint-{self.state.global_step}")
        os.makedirs(output_dir, exist_ok=True)
        self.save_model(output_dir, _internal_call=True)
        if self.tail_write_fails:
            raise OSError(28, "No space left on device")


class _Trainer(DistributedTrainerMixin, _BaseWithTailWrites):
    """The mixin's ``_save_checkpoint`` over that base. ``save_model`` is stubbed to the contract the
    real one holds: it sets the collectives-done mark at its end, and raising before that stands for
    a failure inside the gather region."""

    def __init__(self, run_dir, *, save_model_fails=False, tail_write_fails=False):
        self.run_dir = run_dir
        self.save_model_fails = save_model_fails
        self.tail_write_fails = tail_write_fails
        self.args = SimpleNamespace(save_total_limit=1, save_only_model=False, should_save=True)
        self.state = SimpleNamespace(global_step=2, best_model_checkpoint=None)
        self.parallelism_config = SimpleNamespace(is_tp_mode=False)
        self._fsdp_wrapped = True
        self.lr_scheduler = None

    def _get_output_dir(self, trial=None):
        return self.run_dir

    def save_model(self, output_dir, _internal_call=False):
        if self.save_model_fails:
            raise OSError(28, "No space left on device")
        self._model_save_collectives_done = True


def _recording_guards(monkeypatch) -> list:
    """Every guard the save builds, each recording whether its COLLECTIVE rejection was reached."""
    created = []

    class _Guard(DeferredRankFailure):
        def __init__(self, what, **kwargs):
            super().__init__(what, **kwargs)
            self.rejected = False
            created.append(self)

        def reject(self):
            self.rejected = True
            super().reject()

    monkeypatch.setattr(checkpointing_mod, "DeferredRankFailure", _Guard)
    return created


def test_a_failure_inside_the_collective_region_raises_immediately(tmp_path, monkeypatch):
    guards = _recording_guards(monkeypatch)
    trainer = _Trainer(str(tmp_path), save_model_fails=True)

    with pytest.raises(RuntimeError, match="before the save's collectives completed"):
        trainer._save_checkpoint(model=None, trial=None)

    assert len(guards) == 1
    assert not guards[0].rejected, (
        "the failure was deferred to the guard's all_gather while the peers were still inside the "
        "save's gathers — mismatched collectives instead of the diagnostic the fence promises"
    )
    assert "No space left on device" in str(guards[0].reason), "the real cause must survive"


def test_a_failure_in_the_writer_local_tail_is_made_uniform(tmp_path, monkeypatch):
    """The half the fence is FOR: an ENOSPC in the base's own rank-local writes reaches every rank
    through the collective rejection instead of raising on the writer alone."""
    guards = _recording_guards(monkeypatch)
    trainer = _Trainer(str(tmp_path), tail_write_fails=True)

    with pytest.raises(RuntimeError, match="No space left on device"):
        trainer._save_checkpoint(model=None, trial=None)

    assert guards[0].rejected, "a writer-local tail failure must go through the uniform rejection"


class _SaveModelTrainer(DistributedTrainerMixin):
    """The REAL ``save_model``, with only its trainer-state factories stubbed."""

    def __init__(self, ctx):
        self.ctx = ctx
        self.parallelism_config = SimpleNamespace(is_pp_mode=False)

    def _top_level_model(self):
        return self.ctx.model

    def _checkpoint_context(self):
        return self.ctx


def _context(model) -> CheckpointContext:
    return CheckpointContext(
        model=model,
        parallelism_config=SimpleNamespace(is_pp_mode=False),
        is_pp_mode=False,
        is_cp_mode=False,
        is_tp_mode=False,
        is_ep_tp_mode=False,
        has_ep_layers=False,
        fsdp_wrapped=True,
        accelerate_manages_fsdp=False,
        is_save_rank=True,
        max_shard_size="5GB",
        save_sharded_ep=False,
        has_expert_lora=False,
        merge_expert_lora_on_save=False,
        cp_wrapper=None,
        tokenizer=None,
    )


def test_save_model_marks_its_collectives_done_on_the_strategy_exit(tmp_path, monkeypatch):
    monkeypatch.setattr(checkpointing_mod, "save_checkpoint", lambda ctx, out: True)
    trainer = _SaveModelTrainer(_context(nn.Linear(4, 4)))

    trainer.save_model(str(tmp_path))

    assert trainer._model_save_collectives_done


def test_save_model_marks_its_collectives_done_on_the_adapter_exit(tmp_path, monkeypatch):
    """The PEFT branch returns before the strategy ladder; unmarked, every adapter run's saves would
    take the immediate-raise path even for a failure the fence could have made uniform."""
    saved = []
    monkeypatch.setattr(checkpointing_mod, "find_peft_model", lambda model: object())
    monkeypatch.setattr(
        checkpointing_mod,
        "PeftAdapterSaver",
        lambda: SimpleNamespace(save=lambda ctx, peft_model, out: saved.append(out)),
    )
    trainer = _SaveModelTrainer(_context(nn.Linear(4, 4)))

    trainer.save_model(str(tmp_path))

    assert saved == [str(tmp_path)], "premise: the adapter exit must have been the one taken"
    assert trainer._model_save_collectives_done


def test_every_save_model_marks_its_collectives_done():
    """Structural, over the whole trainer tree: a ``save_model`` that replaces the mixin's (the
    embedding trainer does) still gathers, and an unmarked one silently
    downgrades every deferred tail failure of that trainer to a rank-local raise."""
    trainers = pathlib.Path(__file__).resolve().parents[3] / "src" / "trainers"
    assert trainers.is_dir(), f"premise: {trainers} must exist, else this test scans nothing"
    unmarked = []
    for path in trainers.rglob("*.py"):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if not (isinstance(node, ast.FunctionDef) and node.name == "save_model"):
                continue
            marks = any(
                isinstance(call.func, ast.Attribute) and call.func.attr == "_mark_model_save_collectives_done"
                for call in ast.walk(node)
                if isinstance(call, ast.Call)
            )
            if not marks:
                unmarked.append(f"{path}:{node.lineno}")
    assert not unmarked, f"save_model implementations that never mark their collectives done: {unmarked}"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
