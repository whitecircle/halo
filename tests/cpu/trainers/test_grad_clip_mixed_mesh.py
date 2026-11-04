#!/usr/bin/env python
"""HF's mixed-mesh gradient norm must never replace the toolkit's clip.
It would skip the global norm over EP/TP/FSDP groups, the deferred EP sweep and the TP-replicated sync."""

from types import SimpleNamespace

import pytest
import torch
import transformers.trainer
from transformers import Trainer

from halo.trainers.parallelism.grad_clipping import GradientClippingMixin


class _Accelerator:
    distributed_type = "MULTI_GPU"

    def __init__(self):
        self.clip_calls = []

    def clip_grad_norm_(self, parameters, max_norm, norm_type=2):
        self.clip_calls.append(max_norm)
        return torch.tensor(2.0)


class _HFBase:
    _clip_grad_norm = Trainer._clip_grad_norm
    _get_grad_norm = Trainer._get_grad_norm
    _has_mixed_mesh_grads = Trainer._has_mixed_mesh_grads
    _mixed_mesh_grad_norm = Trainer._mixed_mesh_grad_norm


class _Composed(GradientClippingMixin, _HFBase):
    pass


def _trainer(cls):
    me = object.__new__(cls)
    me.accelerator = _Accelerator()
    me.model = torch.nn.Linear(2, 2)
    for param in me.model.parameters():
        param.grad = torch.ones_like(param)
    me._mixed_mesh_grads = None
    me.args = SimpleNamespace(max_grad_norm=1.0)
    return me


@pytest.fixture
def mixed_grads(monkeypatch):
    monkeypatch.setattr(transformers.trainer, "has_mixed_dtensor", lambda tensors: True)


def test_the_clip_goes_through_the_patched_accelerator_clip(mixed_grads):
    me = _trainer(_Composed)

    me._clip_grad_norm(me.model)

    assert me.accelerator.clip_calls == [1.0], "HF's per-mesh norm ran instead of the toolkit clip"


def test_the_unclipped_norm_goes_through_the_patched_accelerator_clip(mixed_grads):
    me = _trainer(_Composed)

    me._get_grad_norm(me.model, grad_norm=None)

    assert me.accelerator.clip_calls == [float("inf")], "HF's per-mesh norm ran instead of the toolkit clip"


def test_premise_hf_alone_takes_its_own_branch_on_mixed_grads(mixed_grads):
    me = _trainer(_HFBase)

    me._clip_grad_norm(me.model)

    assert me.accelerator.clip_calls == []


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
