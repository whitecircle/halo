#!/usr/bin/env python
"""QLoRA's post-backward gradient sync: membership in the collective must be identical on every rank.
A rank with no vision grad on a text-only microbatch would skip a collective its peers enter and hang the job.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from halo.trainers.distributed_trainer import DistributedTrainerMixin


class _Model(torch.nn.Module):
    """Two adapters: one every microbatch touches, one only a multimodal batch does."""

    def __init__(self):
        super().__init__()
        self.text_lora = torch.nn.Linear(4, 4, bias=False)
        self.vision_lora = torch.nn.Linear(4, 4, bias=False)
        self.frozen = torch.nn.Linear(4, 4, bias=False)
        self.frozen.requires_grad_(False)


def _trainer(model, *, fp32=False):
    stub = SimpleNamespace(
        _qlora_grad_sync=True,
        _qlora_sweep_last_step=None,
        state=SimpleNamespace(global_step=0),
        parallelism_config=SimpleNamespace(fp32_grad_reduce=fp32),
        model=model,
    )
    stub._top_level_model = lambda: model
    return stub


def _run_sweep(trainer, *, present_on_some_rank):
    """Supplies the peer's presence mask; ``all_reduce(MAX)`` over the mask is what makes membership structural."""
    reduced: list[list[torch.Tensor]] = []

    def fake_all_reduce(tensor, op=None, group=None):
        tensor.copy_(torch.tensor(present_on_some_rank, dtype=tensor.dtype))

    with (
        patch("halo.trainers.parallelism.grad_sync.dist.all_reduce", side_effect=fake_all_reduce),
        patch(
            "halo.trainers.parallelism.grad_sync.reduce_grads_bucketed",
            side_effect=lambda grads, **kw: reduced.append(grads),
        ),
    ):
        DistributedTrainerMixin._sync_qlora_grads(trainer)
    return reduced[0] if reduced else []


def test_a_param_another_rank_has_a_grad_for_still_enters_the_collective():
    model = _Model()
    model.text_lora.weight.grad = torch.ones(4, 4)
    # vision_lora.grad stays None: a text-only microbatch on this rank.
    trainer = _trainer(model)

    reduced = _run_sweep(trainer, present_on_some_rank=[1, 1])

    assert model.vision_lora.weight.grad is not None, (
        "a param a peer rank reduced was left out: that rank enters a collective this one skips"
    )
    assert torch.equal(model.vision_lora.weight.grad, torch.zeros(4, 4)), (
        "the absent contribution must be zero, not the parameter's stale grad"
    )
    assert len(reduced) == 2, f"both trainable params must be reduced, got {len(reduced)}"


def test_a_param_no_rank_touched_is_left_alone():
    """Zero-filling everywhere would hand the optimizer weight decay and momentum on a parameter it should skip."""
    model = _Model()
    model.text_lora.weight.grad = torch.ones(4, 4)
    trainer = _trainer(model)

    reduced = _run_sweep(trainer, present_on_some_rank=[1, 0])

    assert model.vision_lora.weight.grad is None, "no rank produced this grad; it must stay None"
    assert len(reduced) == 1


def test_frozen_params_never_enter():
    model = _Model()
    model.text_lora.weight.grad = torch.ones(4, 4)
    trainer = _trainer(model)

    _run_sweep(trainer, present_on_some_rank=[1, 1])

    assert model.frozen.weight.grad is None


def test_the_sweep_runs_once_per_optimizer_step():
    """Both callers (the patched clip, the step-pre-hook) are reachable in one step."""
    model = _Model()
    model.text_lora.weight.grad = torch.ones(4, 4)
    trainer = _trainer(model)

    first = _run_sweep(trainer, present_on_some_rank=[1, 1])
    second = _run_sweep(trainer, present_on_some_rank=[1, 1])

    assert first and not second, "the second call in one step must be a no-op"

    trainer.state.global_step = 1
    assert _run_sweep(trainer, present_on_some_rank=[1, 1]), "a new step must sweep again"


def test_each_run_sweeps_its_own_first_step():
    """``global_step`` restarts per ``train()``; an earlier first sweep must not leave the new run un-averaged."""
    model = _Model()
    model.text_lora.weight.grad = torch.ones(4, 4)
    trainer = _trainer(model)
    assert _run_sweep(trainer, present_on_some_rank=[1, 1])

    trainer.state = SimpleNamespace(global_step=0)
    assert _run_sweep(trainer, present_on_some_rank=[1, 1]), "the new run's first step was taken as already swept"
    assert not _run_sweep(trainer, present_on_some_rank=[1, 1]), "the new run swept its first step twice"


def test_disabled_when_the_run_is_not_qlora():
    model = _Model()
    model.text_lora.weight.grad = torch.ones(4, 4)
    trainer = _trainer(model)
    trainer._qlora_grad_sync = False

    assert not _run_sweep(trainer, present_on_some_rank=[1, 1])


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
