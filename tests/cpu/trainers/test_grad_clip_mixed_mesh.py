#!/usr/bin/env python
"""CPU test: HF's mixed-mesh gradient norm must never replace the toolkit's clip.

transformers sends gradients that mix DTensor and plain tensors (EP, TP-replicated params, EP+LoRA)
to its own per-mesh norm, in both ``_clip_grad_norm`` and the unclipped ``_get_grad_norm``. Every
toolkit clip replaces ``accelerator.clip_grad_norm_`` (the global norm over EP/TP/FSDP groups, the
deferred EP sweep, the TP-replicated sync), so on those runs HF's branch would skip all of it. The
grads are reported mixed here by patching HF's own detector, which is what a real EP run trips.

    python tests/cpu/trainers/test_grad_clip_mixed_mesh.py
"""

from types import SimpleNamespace

import pytest
import torch
import transformers.trainer
from transformers import Trainer

from src.trainers.parallelism.grad_clipping import GradientClippingMixin


class _Accelerator:
    distributed_type = "MULTI_GPU"

    def __init__(self):
        self.clip_calls = []

    def clip_grad_norm_(self, parameters, max_norm, norm_type=2):
        self.clip_calls.append(max_norm)
        return torch.tensor(2.0)


class _HFBase:
    """HF's own clip and norm, which the mixin sits in front of in every distributed trainer."""

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
    me.control = SimpleNamespace(should_log=True)
    me.state = SimpleNamespace(global_step=0, logging_steps=1, max_steps=1)
    me.args = SimpleNamespace(max_grad_norm=1.0, logging_strategy="steps", logging_first_step=True)
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
    """Anti-vacuity: without the mixin's answer the patched detector does divert HF's clip."""
    me = _trainer(_HFBase)

    me._clip_grad_norm(me.model)

    assert me.accelerator.clip_calls == []


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
