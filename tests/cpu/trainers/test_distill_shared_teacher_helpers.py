#!/usr/bin/env python
"""Self-distillation's reference gate, and the two primitives the distillation trainers share.

1. ``DistributedSelfDistillationTrainer`` declares EP and TP support and accepts an explicit
   ``reference_model``, but that reference is never parallelized: under EP/TP it stays a dense
   replica running the unpatched MoE path, so every reference log-prob — and the KL built from
   them — is silently biased. It must go through the same ``_validate_reference_model`` gate DPO
   uses, and it must fail BEFORE the model is moved to the device.
2. ``privileged_teacher_pass`` and ``shifted_token_cross_entropy`` each serve two trainers; the
   equivalence checks below pin them to the per-trainer formulas, and the last test states the
   difference the sharing deliberately keeps (per-sample vs global denominator).

Run: python tests/cpu/trainers/test_distill_shared_teacher_helpers.py
"""

from unittest import mock

import pytest
import torch
import torch.nn as nn
from accelerate import PartialState
from torch.nn.functional import cross_entropy
from trl import SFTTrainer

from src.data.spans import LABEL_IGNORE_INDEX
from src.distributed.parallelism_config import ParallelismConfig
from src.trainers.distillation.losses import (
    masked_token_mean,
    privileged_teacher_pass,
    shifted_token_cross_entropy,
)
from src.trainers.distillation.self_distillation import DistributedSelfDistillationTrainer
from src.trainers.sft import DistributedSFTTrainer

PartialState()  # the trainers' accelerate logger requires an initialized state


def _build(ep_size=1, tp_size=1, reference_model=None, reference_kl_coef=1.0):
    """Construct the trainer with only the HF/TRL machinery stubbed out; returns the setup calls."""
    setup_calls = []

    def _init_cfg(self, kwargs, **_):
        self.parallelism_config = ParallelismConfig(world_size=8, gpus_per_node=8, ep_size=ep_size, tp_size=tp_size)
        return kwargs

    with (
        mock.patch.object(DistributedSFTTrainer, "_init_distributed_config", _init_cfg),
        mock.patch.object(SFTTrainer, "__init__", return_value=None),
        mock.patch.object(DistributedSelfDistillationTrainer, "_setup_distributed_modes", return_value=None),
        mock.patch.object(DistributedSelfDistillationTrainer, "_resolve_stop_token_ids", return_value=None),
        mock.patch.object(
            DistributedSelfDistillationTrainer,
            "_setup_reference_model",
            lambda self: setup_calls.append(self._reference_model),
        ),
    ):
        DistributedSelfDistillationTrainer(reference_model=reference_model, reference_kl_coef=reference_kl_coef)
    return setup_calls


def test_explicit_reference_is_rejected_under_ep():
    with pytest.raises(ValueError, match="explicit ref_model is not supported under EP/TP"):
        _build(ep_size=8, reference_model=nn.Linear(2, 2))


def test_explicit_reference_is_rejected_under_tp():
    with pytest.raises(ValueError, match="explicit ref_model is not supported under EP/TP"):
        _build(tp_size=8, reference_model=nn.Linear(2, 2))


def test_the_gate_runs_before_the_reference_is_set_up():
    """Ordering is load-bearing: ``_setup_reference_model`` moves the whole replica onto the GPU."""
    with pytest.raises(ValueError, match="explicit ref_model is not supported under EP/TP"):
        _build(ep_size=8, reference_model=nn.Linear(2, 2), reference_kl_coef=0.5)
    assert _build(ep_size=1, reference_model=nn.Linear(2, 2)), "plain DP must still set the reference up"


def test_plain_data_parallel_reference_is_accepted():
    """Anti-vacuity: the reference is correct wherever nothing shards the policy differently."""
    assert len(_build(ep_size=1, tp_size=1, reference_model=nn.Linear(2, 2))) == 1


def test_no_reference_model_is_never_gated():
    assert _build(ep_size=8, reference_model=None) == []


def test_an_unused_reference_is_not_gated():
    """``reference_kl_coef == 0`` never reads the reference (compute_loss skips the term), so the
    gate must stay on the branch that actually consumes it."""
    assert _build(ep_size=8, reference_model=nn.Linear(2, 2), reference_kl_coef=0.0) == []


def _logits_and_labels(ignore_positions=((0, 2),)):
    torch.manual_seed(0)
    logits = torch.randn(2, 5, 7)
    labels = torch.randint(0, 7, (2, 5))
    for row, col in ignore_positions:
        labels[row, col] = LABEL_IGNORE_INDEX
    return logits, labels


def test_shifted_ce_matches_the_global_token_mean_it_replaced():
    """Teacher distillation's formula: one ``reduction='sum'`` over a clamped valid-token count."""
    logits, labels = _logits_and_labels()
    shift_logits, shift_labels = logits[:, :-1, :], labels[:, 1:]
    count = (shift_labels != LABEL_IGNORE_INDEX).sum().clamp(min=1)

    before = (
        cross_entropy(
            shift_logits.reshape(-1, shift_logits.size(-1)).float(),
            shift_labels.reshape(-1),
            ignore_index=LABEL_IGNORE_INDEX,
            reduction="sum",
        )
        / count
    )
    after = shifted_token_cross_entropy(shift_logits, shift_labels).sum() / count
    assert torch.allclose(before, after, rtol=0, atol=1e-5), (before.item(), after.item())


def test_shifted_ce_matches_the_per_sample_mean_it_replaced():
    """Self-distillation's formula: per-token CE reduced by ``masked_token_mean``."""
    logits, labels = _logits_and_labels()
    shift_logits, shift_labels = logits[:, :-1, :], labels[:, 1:]
    mask = shift_labels != LABEL_IGNORE_INDEX

    before = masked_token_mean(
        cross_entropy(
            shift_logits.reshape(-1, shift_logits.size(-1)).float(),
            shift_labels.reshape(-1),
            ignore_index=LABEL_IGNORE_INDEX,
            reduction="none",
        ).view(shift_labels.shape),
        mask,
    )
    after = masked_token_mean(shifted_token_cross_entropy(shift_logits, shift_labels), mask)
    assert torch.equal(before, after)


def test_shifted_ce_evaluates_in_fp32_and_zeroes_ignored_positions():
    """bf16 logits must not decide a log-sum-exp over the vocab, and an ignored target contributes
    nothing — the sum reduction above relies on that being exactly 0."""
    logits, labels = _logits_and_labels(ignore_positions=((0, 1), (1, 3)))
    token_ce = shifted_token_cross_entropy(logits[:, :-1, :].bfloat16(), labels[:, 1:])
    assert token_ce.dtype is torch.float32
    assert token_ce.shape == labels[:, 1:].shape
    assert token_ce[labels[:, 1:] == LABEL_IGNORE_INDEX].abs().max() == 0


def test_the_two_reductions_are_genuinely_different():
    """Why the two call sites keep their own denominators: on ragged rows a per-sample mean and a
    global token mean disagree, so collapsing them would have silently reweighted one trainer."""
    logits = torch.randn(2, 5, 7)
    labels = torch.randint(0, 7, (2, 5))
    labels[0, 1:] = LABEL_IGNORE_INDEX  # one short row, one full row
    shift_logits, shift_labels = logits[:, :-1, :], labels[:, 1:]
    token_ce = shifted_token_cross_entropy(shift_logits, shift_labels)
    mask = shift_labels != LABEL_IGNORE_INDEX

    per_sample = masked_token_mean(token_ce, mask)
    global_mean = token_ce.sum() / mask.sum().clamp(min=1)
    assert not torch.allclose(per_sample, global_mean)


class _Probe(nn.Module):
    """Records what the forward saw inside the teacher bracket."""

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(3, 3)
        self.seen = []

    def forward(self, x):
        self.seen.append((self.training, torch.is_grad_enabled()))
        return self.linear(x)


def test_privileged_teacher_pass_runs_frozen_and_in_eval():
    model = _Probe()
    model.train()
    with privileged_teacher_pass(model):
        out = model(torch.randn(1, 3))
    assert model.seen == [(False, False)], model.seen
    assert out.requires_grad is False
    assert model.training is True, "the training flag must be restored"


def test_privileged_teacher_pass_restores_training_after_a_raise():
    model = _Probe()
    model.train()
    with pytest.raises(RuntimeError), privileged_teacher_pass(model):
        raise RuntimeError("teacher forward blew up")
    assert model.training is True


def test_privileged_teacher_pass_leaves_an_eval_model_in_eval():
    model = _Probe()
    model.eval()
    with privileged_teacher_pass(model):
        pass
    assert model.training is False, "an eval-mode caller must not be flipped into train mode"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
