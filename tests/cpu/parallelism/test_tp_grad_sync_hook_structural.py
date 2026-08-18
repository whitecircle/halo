#!/usr/bin/env python
"""CPU test: the TP pre-step replicated-grad sync must run STRUCTURALLY, not grad-gated.

``_register_tp_replicated_grad_sync_hook``'s pre-step hook issues a TP-group collective
(``_sync_tp_replicated_grads``). Grad presence is rank-local (sparse routing / idle VLM tower can
leave every grad ``None`` on one rank only), so gating the call on ``any(p.grad is not None)``
desyncs the TP ranks into a hang — the codebase's structural-collective invariant (see
``tp_clip_grad_norm_``). The hook must call the sync unconditionally; the sync itself zero-fills
missing grads so the collective count stays uniform.

Run: ``python tests/cpu/parallelism/test_tp_grad_sync_hook_structural.py`` (or ``pytest -m cpu``).
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch
import torch.nn as nn

import src.trainers.mixins.grad_sync as grad_sync_mod
from src.trainers.mixins.base import DistributedTrainerMixin


def _trainer(max_grad_norm: float = 0.0):
    model = nn.Linear(4, 4)
    trainer = SimpleNamespace(
        model=model,
        optimizer=torch.optim.SGD(model.parameters(), lr=0.1),
        args=SimpleNamespace(max_grad_norm=max_grad_norm),
        parallelism_config=SimpleNamespace(is_tp_mode=True),
        _get_sharded_expert_param_ids=lambda: set(),
        _sync_tp_replicated_grads=Mock(),
    )
    return trainer


def test_pre_step_sync_runs_even_when_all_grads_are_none():
    trainer = _trainer(max_grad_norm=0.0)
    DistributedTrainerMixin._register_tp_replicated_grad_sync_hook(trainer)
    assert trainer._tp_grad_sync_hook_registered is True

    # No backward ran on this rank — every grad is None. Another TP rank may have grads, so the
    # sync collective must still be issued here (an any(grad) gate skips it → hang).
    assert all(p.grad is None for p in trainer.model.parameters())
    trainer.optimizer.step()

    trainer._sync_tp_replicated_grads.assert_called_once()
    (synced_params,) = trainer._sync_tp_replicated_grads.call_args.args
    assert set(map(id, synced_params)) == set(map(id, trainer.model.parameters()))


def test_pre_step_sync_skipped_when_clip_path_owns_it():
    trainer = _trainer(max_grad_norm=1.0)  # clipping enabled → tp_clip_grad_norm_ syncs instead
    DistributedTrainerMixin._register_tp_replicated_grad_sync_hook(trainer)
    trainer.optimizer.step()
    trainer._sync_tp_replicated_grads.assert_not_called()


def test_sync_is_idempotent_within_one_step():
    """At ``max_grad_norm == 0`` transformers still reaches the patched clip (``_get_grad_norm``
    calls it with ``inf``), so the pre-step hook is a SECOND call in the same step. The AVG bucket
    survives that; the per-head-norm SUM would double (S → tp_size·S), so the sync itself must be
    marker-gated rather than the hook guessing when transformers skips clipping."""
    trainer = _trainer(max_grad_norm=0.0)
    trainer.state = SimpleNamespace(global_step=7)
    trainer.parallelism_config = SimpleNamespace(is_tp_mode=True, fp32_grad_reduce=False)
    trainer._get_tp_process_group = lambda: object()  # a group with peers, per the patch below
    trainer._tp_sharded_plain_param_ids = lambda: set()
    trainer._tp_per_head_norm_param_ids = lambda: set()
    params = list(trainer.model.parameters())

    reduce_calls = Mock()
    with (
        patch.object(grad_sync_mod.dist, "get_world_size", return_value=2),
        patch.object(grad_sync_mod, "reduce_grads_bucketed", reduce_calls),
    ):
        DistributedTrainerMixin._sync_tp_replicated_grads(trainer, params)
        assert trainer._tp_sync_last_step == 7, "the sync did not record the step it ran on"
        first_round = reduce_calls.call_count
        assert first_round > 0, "nothing was reduced — the rest of this test would be vacuous"

        DistributedTrainerMixin._sync_tp_replicated_grads(trainer, params)
        assert reduce_calls.call_count == first_round, "re-reduced within one optimizer step"

        trainer.state.global_step = 8
        DistributedTrainerMixin._sync_tp_replicated_grads(trainer, params)
        assert reduce_calls.call_count == 2 * first_round, "the next step did not sync"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
