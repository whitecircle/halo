#!/usr/bin/env python
"""CPU test: the clips and transformers keep the gradient norm on the device.

The toolkit's clip paths (EP / TP / PP / FSDP) return the norm as a DEVICE tensor on purpose: a
``.item()`` is a full host sync that serializes the whole rank against the copy engine. The one
per-step read is the non-finite step guard's verdict, taken once per optimizer step
(``tests/cpu/trainers/test_nonfinite_step_guard.py``). HF reads the norm back inside
``_maybe_log_save_evaluate``'s ``should_log`` branch, and ``_get_grad_norm`` itself converts only under
DeepSpeed — this pins both, so a transformers bump that adds a second read fails here instead of
showing up as a throughput regression.

    python tests/cpu/trainers/test_grad_norm_host_sync.py
"""

import pytest
import torch
from transformers import Trainer

from halo.trainers.parallelism.grad_clipping import GradientClippingMixin, pp_clip_grad_norm


class _NoHostSync(torch.Tensor):
    """A tensor that refuses to be read back to the host."""

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
    """Stands in for the trainer ``Trainer._get_grad_norm`` is a method of."""

    _has_mixed_mesh_grads = GradientClippingMixin._has_mixed_mesh_grads

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
    """At max_grad_norm <= 0 HF computes an unclipped norm (for the log and the step guard); it stays on device."""
    me = _Trainer()

    norm = Trainer._get_grad_norm(me, me.model, grad_norm=None)

    assert isinstance(norm, torch.Tensor) and me.accelerator.clip_calls == 1


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
