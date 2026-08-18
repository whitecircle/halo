#!/usr/bin/env python
"""
Tests for DistributedSFTTrainer's eval-mode skip_logits injection.

Liger's ``lce_forward`` only takes the fused linear-cross-entropy path when
``self.training`` is True, so evaluation would materialize the full
``[B, S, vocab]`` fp32 logits and OOM at long context (~43 GiB at 75k tokens
x 155k vocab). ``DistributedSFTTrainer.prediction_step`` injects
``skip_logits=True`` for loss-only eval batches when the model's forward is
FLCE-patched. These tests cover the injection gating logic; the end-to-end
fused-eval-loss test is tests/gpu/trainers/sft/test_sft_eval_flce.py.

Run: python tests/cpu/trainers/test_skip_logits_injection.py
"""

import inspect
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
from transformers import Trainer

from src.kernels.liger.lce_forward import build_lce_forward
from src.trainers.sft import DistributedSFTTrainer


class FlcePatchedModel(nn.Module):
    """Forward signature of a Liger lce_forward-patched model."""

    def forward(self, input_ids=None, labels=None, skip_logits=None, **kwargs):
        return None


class UnpatchedModel(nn.Module):
    """Standard transformers forward — no skip_logits parameter."""

    def forward(self, input_ids=None, labels=None, **kwargs):
        return None


def make_trainer(model=None, use_liger_kernel=True, is_cp_mode=False):
    """Bare DistributedSFTTrainer with only the state prediction_step reads."""
    trainer = object.__new__(DistributedSFTTrainer)
    trainer.model = model if model is not None else FlcePatchedModel()
    trainer.args = SimpleNamespace(use_liger_kernel=use_liger_kernel)
    trainer.parallelism_config = SimpleNamespace(is_cp_mode=is_cp_mode)
    return trainer


def captured_inputs(trainer, inputs, prediction_loss_only=True):
    """Run prediction_step with the base Trainer step stubbed to a recorder."""
    captured = {}
    original = Trainer.prediction_step

    def recorder(self, model, inputs, prediction_loss_only, ignore_keys=None):
        captured.update(inputs)
        return (None, None, None)

    Trainer.prediction_step = recorder
    try:
        trainer.prediction_step(trainer.model, inputs, prediction_loss_only)
    finally:
        Trainer.prediction_step = original
    return captured


LABELS = torch.tensor([[1, 2, 3]])


def test_injects_for_loss_only_eval():
    captured = captured_inputs(make_trainer(), {"input_ids": LABELS, "labels": LABELS})
    assert captured.get("skip_logits") is True


def test_no_injection_for_shift_labels_only():
    # CP/SP collators pass shift_labels; the FLCE forward takes only labels, so stay unfused.
    captured = captured_inputs(make_trainer(), {"input_ids": LABELS, "shift_labels": LABELS})
    assert "skip_logits" not in captured


def test_no_injection_when_logits_needed():
    # prediction_loss_only=False → compute_metrics consumes logits downstream
    captured = captured_inputs(make_trainer(), {"input_ids": LABELS, "labels": LABELS}, prediction_loss_only=False)
    assert "skip_logits" not in captured


def test_no_injection_without_labels():
    captured = captured_inputs(make_trainer(), {"input_ids": LABELS})
    assert "skip_logits" not in captured


def test_no_injection_for_unpatched_forward():
    captured = captured_inputs(make_trainer(model=UnpatchedModel()), {"input_ids": LABELS, "labels": LABELS})
    assert "skip_logits" not in captured


def test_no_injection_without_liger():
    # use_liger_kernel=False → TRL computes entropy from outputs.logits
    captured = captured_inputs(make_trainer(use_liger_kernel=False), {"input_ids": LABELS, "labels": LABELS})
    assert "skip_logits" not in captured


def test_no_injection_in_cp_mode():
    # _compute_cp_metrics reads local-chunk logits in CP mode
    captured = captured_inputs(make_trainer(is_cp_mode=True), {"input_ids": LABELS, "labels": LABELS})
    assert "skip_logits" not in captured


def test_explicit_skip_logits_preserved():
    captured = captured_inputs(make_trainer(), {"input_ids": LABELS, "labels": LABELS, "skip_logits": False})
    assert captured["skip_logits"] is False


def test_signature_detection_cached():
    trainer = make_trainer()
    assert trainer._forward_accepts_skip_logits is True
    # Cached: the patch never changes after init, so a model swap must not re-probe.
    trainer.model = UnpatchedModel()
    assert trainer._forward_accepts_skip_logits is True


def test_toolkit_lce_forward_accepts_skip_logits():
    """The one fused forward every toolkit applier installs — the probe above finds nothing without it."""
    assert "skip_logits" in inspect.signature(build_lce_forward()).parameters


def test_toolkit_lce_forward_skip_logits_requires_labels():
    try:
        build_lce_forward()(SimpleNamespace(), skip_logits=True)
    except ValueError:
        return
    raise AssertionError("skip_logits=True without labels must raise ValueError")


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
