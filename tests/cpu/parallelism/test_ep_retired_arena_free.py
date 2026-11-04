#!/usr/bin/env python
"""The DeepEP arenas a grow retires are freed once a step's backwards are done, not held until teardown. Freeing is a
collective, and every rank of a group retired the same buffers since capacity is the group's all-reduced max."""

from types import SimpleNamespace

import pytest
import torch

from halo.distributed.expert_parallel.deepep import backends as backends_mod
from halo.distributed.expert_parallel.deepep.backends import (
    _ElasticArena,
    free_all_ep_buffers,
    free_retired_ep_buffers,
)
from halo.trainers import distributed_trainer
from halo.trainers.distributed_trainer import DistributedTrainerMixin

CAPACITIES = (64, 128, 256, 512)
GRAD_ACCUM = 3


class _Buffer:
    """Numbered in build order, logging each destroy."""

    built: list = []
    destroyed: list = []

    def __init__(self, group, *, num_max_tokens_per_rank: int, **kwargs):
        self.capacity = num_max_tokens_per_rank
        _Buffer.built.append(self)

    def destroy(self) -> None:
        _Buffer.destroyed.append(self)


@pytest.fixture(autouse=True)
def recorded_deep_ep(monkeypatch):
    _Buffer.built, _Buffer.destroyed = [], []
    monkeypatch.setattr(backends_mod, "deep_ep", lambda: SimpleNamespace(ElasticBuffer=_Buffer))
    monkeypatch.setattr(backends_mod._Arena, "_registry", {})
    monkeypatch.setattr(backends_mod, "EP_CAPACITY_DEDUP", True)
    monkeypatch.setattr(backends_mod, "has_process_group", lambda: False)


def _grown_arena(group) -> _ElasticArena:
    arena = _ElasticArena.acquire(group, 256, 2, 0, False)
    for capacity in CAPACITIES:
        assert arena.ensure(capacity)
    return arena


def test_a_step_boundary_frees_every_retired_buffer_once_and_keeps_the_live_one():
    arena = _grown_arena(object())
    live = arena.buffer
    assert live is _Buffer.built[-1] and live.capacity == CAPACITIES[-1]

    assert free_retired_ep_buffers() == len(CAPACITIES) - 1
    assert _Buffer.destroyed == _Buffer.built[:-1], "the retired buffers, in the order they were built"
    assert arena.buffer is live and arena.capacity == CAPACITIES[-1]
    assert not arena.ensure(CAPACITIES[-1]), "the live buffer still serves the capacity it was built for"

    assert free_retired_ep_buffers() == 0, "a retired buffer is freed exactly once"
    free_all_ep_buffers()
    assert _Buffer.destroyed == _Buffer.built, "teardown frees the live buffer, and only it"


def test_every_arena_is_swept_in_build_order():
    first, second = _grown_arena(object()), _grown_arena(object())
    retired = [*_Buffer.built[: len(CAPACITIES) - 1], *_Buffer.built[len(CAPACITIES) : -1]]
    assert free_retired_ep_buffers() == len(retired)
    assert _Buffer.destroyed == retired
    assert first.buffer is _Buffer.built[len(CAPACITIES) - 1] and second.buffer is _Buffer.built[-1]


class _Backward:
    """The inherited step: one backward per call."""

    def training_step(self, model, inputs, num_items_in_batch=None):
        return torch.zeros(())


class _Trainer(DistributedTrainerMixin, _Backward):
    """The per-batch accounting around the step is stubbed."""

    def __init__(self, *, pipeline: bool):
        self.accelerator = SimpleNamespace(sync_gradients=True)
        self._pp_runner = object() if pipeline else None
        self._pp_training_step = lambda inputs: torch.zeros(())
        self._accumulate_unmasked_output_tokens = lambda count: None
        self._extract_output_token_count = lambda inputs: 0
        self._accumulate_attention_flops = lambda inputs: None
        self._warn_once_on_thin_memory_margin = lambda: None
        self._set_window_end = lambda last: None
        self._finish_fsdp2_backward = lambda model: None


@pytest.fixture
def frees(monkeypatch) -> list[int]:
    calls = []
    monkeypatch.setattr(distributed_trainer, "free_retired_ep_buffers", lambda: calls.append(1) or 0)
    return calls


def test_the_trainer_frees_after_each_windows_last_backward(frees):
    trainer = _Trainer(pipeline=False)
    for step in range(2):
        for micro in range(GRAD_ACCUM):
            trainer.accelerator.sync_gradients = micro == GRAD_ACCUM - 1
            trainer.training_step(None, {})
            assert len(frees) == step + (micro == GRAD_ACCUM - 1), (step, micro)


def test_the_trainer_frees_after_every_pipeline_step(frees):
    trainer = _Trainer(pipeline=True)
    for step in range(3):
        trainer.training_step(None, {})
        assert len(frees) == step + 1


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
