#!/usr/bin/env python
"""The clips and transformers keep the gradient norm on the device, since an ``.item()`` host-syncs the whole rank.
HF reads the norm only in ``_maybe_log_save_evaluate``'s ``should_log`` branch; a bump adding a read fails here."""

import pytest
import torch
from transformers import Trainer

from halo.trainers.parallelism.grad_clipping import GradientClippingMixin, pp_clip_grad_norm


class _NoHostSync(torch.Tensor):
    def item(self):
        raise AssertionError("the gradient norm was read back to the host")


class _Accelerator:
    distributed_type = "MULTI_GPU"

    def __init__(self):
        self.clip_calls = 0

    def clip_grad_norm_(self, parameters, max_norm, norm_type=2):
        self.clip_calls += 1
        return torch.tensor(2.0).as_subclass(_NoHostSync)


class _Trainer:
    _has_mixed_mesh_grads = GradientClippingMixin._has_mixed_mesh_grads

    def __init__(self):
        self.accelerator = _Accelerator()
        self.model = torch.nn.Linear(2, 2)


def test_get_grad_norm_keeps_the_norm_on_device():
    me = _Trainer()

    norm = Trainer._get_grad_norm(me, me.model, grad_norm=torch.tensor(3.0).as_subclass(_NoHostSync))

    assert isinstance(norm, torch.Tensor), "a float here means a per-step host sync"
    assert me.accelerator.clip_calls == 0, "an already-computed norm must not trigger a second pass"


def test_get_grad_norm_without_clipping_computes_one_but_still_does_not_sync():
    """At max_grad_norm <= 0 HF still computes an unclipped norm for the log and the step guard."""
    me = _Trainer()

    norm = Trainer._get_grad_norm(me, me.model, grad_norm=None)

    assert isinstance(norm, torch.Tensor) and me.accelerator.clip_calls == 1


def test_the_pipeline_clip_returns_a_device_tensor(monkeypatch):
    """The coefficient applies unconditionally: ``if norm > max_norm`` would be a host sync per step."""

    class _Dist:
        ReduceOp = torch.distributed.ReduceOp

        def __init__(self):
            self.all_reduces = 0

        def all_reduce(self, tensor, op=None, group=None):
            self.all_reduces += 1

    fake = _Dist()
    monkeypatch.setattr("halo.trainers.parallelism.grad_clipping.dist", fake)
    monkeypatch.setattr("halo.trainers.parallelism.grad_clipping.torch.cuda.current_device", lambda: "cpu")
    monkeypatch.setattr(torch.Tensor, "item", _NoHostSync.item, raising=True)

    params = [torch.nn.Parameter(torch.ones(4)) for _ in range(2)]
    for param in params:
        param.grad = torch.full((4,), 3.0)

    norm = pp_clip_grad_norm(params, max_norm=1.0)

    assert isinstance(norm, torch.Tensor) and norm.device.type == "cpu"
    assert fake.all_reduces == 1, "one collective per step, over the whole pipeline"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
