#!/usr/bin/env python
"""CPU tests for the QLoRA post-backward gradient sync.

QLoRA cannot use FSDP2 (quantized base weights conflict with DTensor inputs), so the DP average is
the toolkit's own. The property that matters is **structural membership**: which parameters enter
the collective must be identical on every rank, because grad presence is not. A VLM with vision
adapters in ``lora_target_modules`` and a text-only microbatch on one rank alone is enough — the
rank with no vision grad skips a collective its peers enter, and the job hangs until the watchdog.

Run: ``pytest -m cpu tests/cpu/trainers/test_qlora_grad_sync.py``
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from src.trainers.mixins.base import DistributedTrainerMixin


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
    """Drive the sweep with the presence mask a peer rank would have contributed.

    ``all_reduce(MAX)`` over the mask is what makes membership structural, so the test supplies the
    peer's answer rather than mocking the decision away.
    """
    reduced: list[list[torch.Tensor]] = []

    def fake_all_reduce(tensor, op=None, group=None):
        tensor.copy_(torch.tensor(present_on_some_rank, dtype=tensor.dtype))

    with (
        patch("src.trainers.mixins.grad_sync.dist.all_reduce", side_effect=fake_all_reduce),
        patch("src.trainers.mixins.grad_sync.current_device", return_value=torch.device("cpu")),
        patch(
            "src.trainers.mixins.grad_sync.reduce_grads_bucketed",
            side_effect=lambda grads, **kw: reduced.append(grads),
        ),
    ):
        DistributedTrainerMixin._sync_qlora_grads(trainer)
    return reduced[0] if reduced else []


def test_a_param_another_rank_has_a_grad_for_still_enters_the_collective():
    """The defect: this rank produced no vision grad, its peer did. Skipping it here desyncs them."""
    model = _Model()
    model.text_lora.weight.grad = torch.ones(4, 4)
    # vision_lora.grad stays None — a text-only microbatch on this rank.
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
    """Anti-over-reach: zero-filling everywhere would hand the optimizer weight decay and momentum
    on a parameter that should have been skipped entirely."""
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


def test_disabled_when_the_run_is_not_qlora():
    model = _Model()
    model.text_lora.weight.grad = torch.ones(4, 4)
    trainer = _trainer(model)
    trainer._qlora_grad_sync = False

    assert not _run_sweep(trainer, present_on_some_rank=[1, 1])


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
