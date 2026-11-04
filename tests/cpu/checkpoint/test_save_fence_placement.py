#!/usr/bin/env python
"""The checkpoint fence starts after the save's collectives, since a rank skipping one for the guard's gather times
the world out. Earlier failures re-raise rank-locally."""

import ast
from types import SimpleNamespace

import pytest
import torch.nn as nn
from transformers import TrainerState

import halo.checkpoint.parallel.save as save_mod
import halo.trainers.checkpointing as checkpointing_mod
from halo.runtime.consensus import DeferredRankFailure
from halo.trainers.distributed_trainer import DistributedTrainerMixin
from tests.common.base_save import BaseTrainerSave
from tests.common.checkpoint_io import save_context
from tests.common.repo import REPO_ROOT


class _Trainer(DistributedTrainerMixin, BaseTrainerSave):
    """``save_model`` is stubbed to the real contract: it sets the collectives-done mark at its end, and a raise
    before that is a failure inside the gather region."""

    def __init__(self, run_dir, *, save_model_fails=False, tail_write_fails=False):
        self.run_dir = run_dir
        self.save_model_fails = save_model_fails
        self.tail_write_fails = tail_write_fails
        self.args = SimpleNamespace(
            save_total_limit=1,
            save_only_model=False,
            should_save=True,
            push_to_hub=False,
            include_num_input_tokens_seen="no",
        )
        self.state = TrainerState(global_step=2)
        self.parallelism_config = SimpleNamespace(is_tp_mode=False, merge_expert_lora_on_save=False)
        self._fsdp_wrapped = True
        self.lr_scheduler = None

    def save_model(self, output_dir, _internal_call=False):
        if self.save_model_fails:
            raise OSError(28, "No space left on device")
        self._model_save_collectives_done = True

    def _save_optimizer_and_scheduler(self, output_dir):
        if self.tail_write_fails:
            raise OSError(28, "No space left on device")


def _recording_guards(monkeypatch) -> list:
    """Each records whether its COLLECTIVE rejection was reached."""
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
    """An ENOSPC in the base's rank-local writes reaches every rank."""
    guards = _recording_guards(monkeypatch)
    trainer = _Trainer(str(tmp_path), tail_write_fails=True)

    with pytest.raises(RuntimeError, match="No space left on device"):
        trainer._save_checkpoint(model=None, trial=None)

    assert guards[0].rejected, "a writer-local tail failure must go through the uniform rejection"


class _SaveModelTrainer(DistributedTrainerMixin):
    """Only its trainer-state factories are stubbed."""

    def __init__(self, ctx):
        self.ctx = ctx
        self.parallelism_config = SimpleNamespace(is_pp_mode=False)

    def _top_level_model(self):
        return self.ctx.model

    def _checkpoint_context(self):
        return self.ctx


def _context(model):
    return save_context(model, toolkit_fsdp2=True, is_save_rank=True)


def test_save_model_marks_its_collectives_done_on_the_strategy_exit(tmp_path, monkeypatch):
    monkeypatch.setattr(checkpointing_mod, "save_checkpoint", lambda ctx, out: True)
    trainer = _SaveModelTrainer(_context(nn.Linear(4, 4)))

    trainer.save_model(str(tmp_path))

    assert trainer._model_save_collectives_done


def test_save_model_marks_its_collectives_done_on_the_adapter_exit(tmp_path, monkeypatch):
    """Unmarked, every adapter run's saves take the immediate-raise path."""
    saved = []
    monkeypatch.setattr(save_mod, "find_peft_model", lambda model: object())
    monkeypatch.setattr(save_mod, "save_peft_adapters", lambda ctx, out: saved.append(out))
    trainer = _SaveModelTrainer(_context(nn.Linear(4, 4)))

    trainer.save_model(str(tmp_path))

    assert saved == [str(tmp_path)], "premise: the adapter exit must have been the one taken"
    assert trainer._model_save_collectives_done


def test_a_payload_override_keeps_the_shared_steps_around_it(tmp_path, monkeypatch):
    """The balancing sidecar before the payload and the mark after it stay the mixin's, on every rank."""
    order = []
    monkeypatch.setattr(checkpointing_mod, "save_checkpoint", lambda ctx, out: pytest.fail("ran the mixin's ladder"))
    monkeypatch.setattr(checkpointing_mod, "save_balancing_biases", lambda *args, **kwargs: order.append("sidecar"))

    class _OwnLayout(_SaveModelTrainer):
        def _write_model_payload(self, ctx, output_dir, _internal_call):
            order.append(("payload", output_dir, _internal_call, self._model_save_collectives_done))

    trainer = _OwnLayout(_context(nn.Linear(4, 4)))
    trainer._model_save_collectives_done = False

    trainer.save_model(str(tmp_path), _internal_call=True)

    assert order == ["sidecar", ("payload", str(tmp_path), True, False)]
    assert trainer._model_save_collectives_done, "the mark must follow the overridden payload"


def test_every_save_model_marks_its_collectives_done():
    """An unmarked override silently downgrades every deferred tail failure to a rank-local raise."""
    trainers = REPO_ROOT / "halo" / "trainers"
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
