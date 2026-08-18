#!/usr/bin/env python
"""CPU test: the gradient norm must not be read back to the host on every step.

The toolkit's clip paths (EP / TP / PP / FSDP) return the norm as a DEVICE tensor on purpose: a
``.item()`` on the per-step path is a full host sync that serializes the whole rank against the
copy engine, and at 512 ranks it is paid once per rank per step for a number only the logging steps
consume. HF reads it back inside ``_maybe_log_save_evaluate``'s ``should_log`` branch, and
``_get_grad_norm`` itself converts only under DeepSpeed — this pins both, so a transformers bump
that starts reading it back every step fails here instead of showing up as a throughput regression.

    python tests/cpu/trainers/test_grad_norm_host_sync.py
"""

from types import SimpleNamespace

import pytest
import torch
from transformers import Trainer

from src.trainers.mixins.grad_sync import GradientSyncMixin
from src.trainers.mixins.pipeline import PipelineTrainerMixin


class _NoHostSync(torch.Tensor):
    """A tensor that refuses to be read back to the host."""

    def item(self):
        raise AssertionError("the gradient norm was read back to the host on a non-logging step")


class _Accelerator:
    distributed_type = "MULTI_GPU"

    def __init__(self):
        self.clip_calls = 0

    def clip_grad_norm_(self, parameters, max_norm, norm_type=2):
        self.clip_calls += 1
        return torch.tensor(2.0).as_subclass(_NoHostSync)


class _Trainer:
    """Stands in for the trainer ``Trainer._get_grad_norm`` is a method of."""

    def __init__(self):
        self.accelerator = _Accelerator()
        self.model = torch.nn.Linear(2, 2)


def test_get_grad_norm_keeps_the_norm_on_device():
    """The clip's own by-product is handed straight back — no conversion on the step path."""
    me = _Trainer()

    norm = Trainer._get_grad_norm(me, me.model, grad_norm=torch.tensor(3.0).as_subclass(_NoHostSync))

    assert isinstance(norm, torch.Tensor), "a float here means a per-step host sync"
    assert me.accelerator.clip_calls == 0, "an already-computed norm must not trigger a second pass"


def test_get_grad_norm_without_clipping_computes_one_but_still_does_not_sync():
    """At max_grad_norm <= 0 HF computes an unclipped norm for logging; it must stay on device."""
    me = _Trainer()

    norm = Trainer._get_grad_norm(me, me.model, grad_norm=None)

    assert isinstance(norm, torch.Tensor) and me.accelerator.clip_calls == 1


class _CountingAccelerator:
    """Counts the unclipped-norm pass HF asks for at ``max_grad_norm <= 0``."""

    distributed_type = "MULTI_GPU"

    def __init__(self):
        self.norm_calls = 0

    def clip_grad_norm_(self, parameters, max_norm, norm_type=2):
        self.norm_calls += 1
        return torch.tensor(2.0)


class _HFBase:
    """The base the mixin's ``_get_grad_norm`` delegates to — HF's own implementation."""

    _get_grad_norm = Trainer._get_grad_norm


class _Gated(GradientSyncMixin, _HFBase):
    """The composition every distributed trainer has: the mixin ahead of the Trainer base."""


def _gated_trainer(global_step, **arg_overrides):
    me = object.__new__(_Gated)
    me.accelerator = _CountingAccelerator()
    me.model = torch.nn.Linear(2, 2)
    me.control = SimpleNamespace(should_log=False)
    me.state = SimpleNamespace(global_step=global_step, logging_steps=10, max_steps=100)
    me.args = SimpleNamespace(**{"logging_strategy": "steps", "logging_first_step": False, **arg_overrides})
    return me


def test_the_unclipped_norm_is_not_computed_on_a_non_logging_step():
    """The whole point: a foreach_norm plus an all-reduce per rank per step for a discarded number."""
    me = _gated_trainer(global_step=4)

    assert me._get_grad_norm(me.model, grad_norm=None) is None
    assert me.accelerator.norm_calls == 0, "the norm collective ran on a step that logs nothing"


@pytest.mark.parametrize(
    ("global_step", "overrides"),
    [
        pytest.param(9, {}, id="logging_steps_boundary"),
        pytest.param(0, {"logging_first_step": True}, id="logging_first_step"),
        pytest.param(99, {}, id="final_step"),
        pytest.param(4, {"logging_strategy": "epoch"}, id="epoch_strategy_is_unpredictable"),
    ],
)
def test_the_norm_is_still_computed_on_every_step_that_logs_it(global_step, overrides):
    """Anti-vacuity: a gate that always skipped would drop the ``grad_norm`` series entirely."""
    me = _gated_trainer(global_step, **overrides)

    assert me._get_grad_norm(me.model, grad_norm=None) is not None
    assert me.accelerator.norm_calls == 1


def test_a_clipped_run_is_untouched():
    """With clipping on, the clip's own by-product is passed in — the gate must not intercept it."""
    me = _gated_trainer(global_step=4)

    norm = me._get_grad_norm(me.model, grad_norm=torch.tensor(3.0))

    assert norm.item() == 3.0 and me.accelerator.norm_calls == 0


def test_a_callback_that_already_raised_should_log_keeps_the_metric():
    me = _gated_trainer(global_step=4)
    me.control.should_log = True

    assert me._get_grad_norm(me.model, grad_norm=None) is not None


def test_the_pipeline_clip_returns_a_device_tensor(monkeypatch):
    """The PP clip aggregates the whole pipeline's squared norms in one all-reduce and applies the
    coefficient unconditionally — no ``if norm > max_norm``, which would be a host sync per step."""

    class _Dist:
        ReduceOp = torch.distributed.ReduceOp

        def __init__(self):
            self.all_reduces = 0

        def all_reduce(self, tensor, op=None, group=None):
            self.all_reduces += 1

    fake = _Dist()
    monkeypatch.setattr("src.trainers.mixins.pipeline.dist", fake)
    monkeypatch.setattr("src.trainers.mixins.pipeline.torch.cuda.current_device", lambda: "cpu")
    monkeypatch.setattr(torch.Tensor, "item", _NoHostSync.item, raising=True)

    params = [torch.nn.Parameter(torch.ones(4)) for _ in range(2)]
    for param in params:
        param.grad = torch.full((4,), 3.0)

    norm = PipelineTrainerMixin._pp_clip_grad_norm(object.__new__(PipelineTrainerMixin), params, max_norm=1.0)

    assert isinstance(norm, torch.Tensor) and norm.device.type == "cpu"
    assert fake.all_reduces == 1, "one collective per step, over the whole pipeline"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
