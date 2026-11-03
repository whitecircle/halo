#!/usr/bin/env python
"""Saving or restoring a stateless optimizer's shards runs no grad-sync step-pre-hook. Torch's zero-LR step on such
an optimizer would sweep the finished step, and the deferred sweep raises "ran twice"."""

import os
from types import SimpleNamespace

import pytest
import torch
from transformers import TrainerState

from halo.checkpoint.parallel.optimizer import OptimizerShardStore
from halo.trainers.distributed_trainer import DistributedTrainerMixin
from halo.trainers.parallelism.grad_sync import claim_optimizer_step
from tests.common.base_save import BaseTrainerSave
from tests.common.checkpoint_io import load_context

# Spelled literally: importing the writer's constant would make a rename invisible here.
SHARD_FILE = "optimizer_shard_00000.pt"


class _Trainer(DistributedTrainerMixin, BaseTrainerSave):
    """Plain SGD (momentum 0, so no per-param state); the sweep records the step it ran at and claims it as the real
    one does."""

    def __init__(self, run_dir: str):
        self.run_dir = run_dir
        self.args = SimpleNamespace(
            save_total_limit=None,
            save_only_model=False,
            should_save=True,
            push_to_hub=False,
            include_num_input_tokens_seen="no",
        )
        self.state = TrainerState(global_step=0)
        self.parallelism_config = SimpleNamespace(is_tp_mode=False, merge_expert_lora_on_save=False)
        self.is_deepspeed_enabled = self.is_fsdp_enabled = False
        torch.manual_seed(0)
        self.model = torch.nn.Linear(4, 4)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.1)
        self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lambda step: 1.0)
        self._ep_config = SimpleNamespace(runs_post_backward_sweep=True)
        self.sweeps: list[int] = []
        self._register_grad_sync_step_hooks()

    def _run_deferred_grad_sweep(self) -> None:
        self.sweeps.append(self.state.global_step)
        if not claim_optimizer_step(self, "_deferred_sweep_last_step"):
            raise RuntimeError(f"the deferred sweep ran twice at global_step={self.state.global_step}")

    def train_step(self) -> None:
        for param in self.model.parameters():
            param.grad = torch.ones_like(param)
        self._run_deferred_grad_sweep()
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)
        self.state.global_step += 1

    def save_model(self, output_dir=None, _internal_call=False):
        self._mark_model_save_collectives_done()

    def _optimizer_store(self) -> OptimizerShardStore:
        return OptimizerShardStore(
            load_context(self.model, optimizer=self.optimizer, lr_scheduler=self.lr_scheduler, fsdp_wrapped=True)
        )


def _trained_and_saved(run_dir: str) -> _Trainer:
    trainer = _Trainer(run_dir)
    assert trainer._deferred_ep_grad_sync_hook_registered, "premise: the sweep's backstop hook is registered"
    trainer.train_step()
    assert trainer.sweeps == [0] and not trainer.optimizer.state, "premise: one swept step, and SGD keeps no state"
    trainer._save_checkpoint(model=None, trial=None)
    return trainer


def test_a_shard_save_of_a_stateless_optimizer_runs_no_sweep(tmp_path):
    trainer = _trained_and_saved(str(tmp_path))

    assert os.path.isfile(os.path.join(str(tmp_path), "checkpoint-1", SHARD_FILE)), "premise: the shard save ran"
    assert trainer.sweeps == [0], "the save's zero-LR step ran the sweep for the step just finished"
    trainer.train_step()
    assert trainer.sweeps == [0, 1], "the next step's clip did not sweep its own step"


def test_a_shard_restore_of_a_stateless_optimizer_runs_no_sweep(tmp_path):
    _trained_and_saved(str(tmp_path))
    resumed = _Trainer(str(tmp_path))
    resumed.state.global_step = 1

    resumed._load_optimizer_and_scheduler(os.path.join(str(tmp_path), "checkpoint-1"))

    assert resumed.sweeps == [], "the restore's zero-LR step ran the sweep for the resumed step"
    resumed.train_step()
    assert resumed.sweeps == [1], "the first resumed step's clip did not sweep its own step"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
