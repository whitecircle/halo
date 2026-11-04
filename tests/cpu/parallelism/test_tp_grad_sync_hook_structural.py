#!/usr/bin/env python
"""The TP pre-step replicated-grad sync runs structurally: grad presence is rank-local, so a grad-gated call desyncs TP
ranks into a hang. The sync itself agrees presence over the TP group, keeping the collective count uniform."""

from __future__ import annotations

import contextlib
from types import MethodType, SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch
import torch.nn as nn

import halo.trainers.parallelism.grad_sync as grad_sync_mod
from halo.trainers.distributed_trainer import DistributedTrainerMixin


def _trainer(max_grad_norm: float = 0.0):
    model = nn.Linear(4, 4)
    trainer = SimpleNamespace(
        model=model,
        optimizer=torch.optim.SGD(model.parameters(), lr=0.1),
        args=SimpleNamespace(max_grad_norm=max_grad_norm),
        parallelism_config=SimpleNamespace(is_tp_mode=True),
        _ep_config=None,
        _get_sharded_expert_param_ids=lambda: set(),
        _sync_tp_replicated_grads=Mock(),
        _sync_qlora_grads=Mock(),
    )
    trainer._non_expert_params = MethodType(DistributedTrainerMixin._non_expert_params, trainer)
    return trainer


def test_pre_step_sync_runs_even_when_all_grads_are_none():
    trainer = _trainer(max_grad_norm=0.0)
    DistributedTrainerMixin._register_grad_sync_step_hooks(trainer)
    assert trainer._tp_grad_sync_hook_registered is True

    # Another TP rank may have grads, so the sync collective must still be issued here.
    assert all(p.grad is None for p in trainer.model.parameters())
    trainer.optimizer.step()

    trainer._sync_tp_replicated_grads.assert_called_once()
    (synced_params,) = trainer._sync_tp_replicated_grads.call_args.args
    assert set(map(id, synced_params)) == set(map(id, trainer.model.parameters()))


def test_pre_step_sync_skipped_when_clip_path_owns_it():
    trainer = _trainer(max_grad_norm=1.0)  # Clipping enabled, so the TP clip syncs instead.
    DistributedTrainerMixin._register_grad_sync_step_hooks(trainer)
    trainer.optimizer.step()
    trainer._sync_tp_replicated_grads.assert_not_called()


def _syncing_trainer(global_step: int):
    """A bare TP mesh whose group has peers under :func:`_tp_sync`."""
    trainer = _trainer(max_grad_norm=0.0)
    trainer.state = SimpleNamespace(global_step=global_step)
    trainer.parallelism_config = SimpleNamespace(is_tp_mode=True, fp32_grad_reduce=False)
    trainer._device_mesh = SimpleNamespace(get_group=lambda: object())
    trainer._tp_sharded_plain_param_ids = lambda: set()
    trainer._tp_per_head_norm_param_ids = lambda: set()
    return trainer


@contextlib.contextmanager
def _tp_sync(reduce_calls: Mock):
    with (
        patch.object(grad_sync_mod.dist, "get_world_size", return_value=2),
        patch.object(grad_sync_mod.dist, "all_reduce"),
        patch.object(grad_sync_mod, "reduce_grads_bucketed", reduce_calls),
    ):
        yield


def test_sync_is_idempotent_within_one_step():
    """At ``max_grad_norm == 0`` transformers still reaches the patched clip, so the hook is a second call; the
    per-head SUM would double, hence the marker."""
    trainer = _syncing_trainer(global_step=7)
    params = list(trainer.model.parameters())

    reduce_calls = Mock()
    with _tp_sync(reduce_calls):
        DistributedTrainerMixin._sync_tp_replicated_grads(trainer, params)
        assert grad_sync_mod.same_optimizer_step(trainer._tp_sync_last_step, trainer.state), (
            "the sync did not mark its step"
        )
        first_round = reduce_calls.call_count
        assert first_round > 0, "nothing was reduced — the rest of this test would be vacuous"

        DistributedTrainerMixin._sync_tp_replicated_grads(trainer, params)
        assert reduce_calls.call_count == first_round, "re-reduced within one optimizer step"

        trainer.state.global_step = 8
        DistributedTrainerMixin._sync_tp_replicated_grads(trainer, params)
        assert reduce_calls.call_count == 2 * first_round, "the next step did not sync"


def test_each_run_syncs_its_own_first_step():
    """``global_step`` restarts at 0 with every ``train()``; a stale mark would leave TP replicas drifting apart."""
    trainer = _syncing_trainer(global_step=0)
    params = list(trainer.model.parameters())

    reduce_calls = Mock()
    with _tp_sync(reduce_calls):
        DistributedTrainerMixin._sync_tp_replicated_grads(trainer, params)
        first_round = reduce_calls.call_count
        assert first_round > 0, "nothing was reduced — the rest of this test would be vacuous"

        trainer.state = SimpleNamespace(global_step=0)
        DistributedTrainerMixin._sync_tp_replicated_grads(trainer, params)
        assert reduce_calls.call_count == 2 * first_round, "the new run's first step was taken as already synced"
        DistributedTrainerMixin._sync_tp_replicated_grads(trainer, params)
        assert reduce_calls.call_count == 2 * first_round, "the new run synced its first step twice"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
