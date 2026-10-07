#!/usr/bin/env python
"""Self-distillation's construction gates, and the two primitives the distillation trainers share.

1. ``DistributedSelfDistillationTrainer`` sets a weighted reference up, and only a weighted one.
   Either term with nothing to compute it from (a weighted anchor without a reference, OPD on with a
   collator that builds no teacher branch) raises at construction rather than dropping out of the
   loss. The EP/TP report on the reference lives in tests/cpu/parallelism/test_reference_model_gate.py.
2. ``privileged_teacher_pass`` and ``shifted_token_cross_entropy`` each serve two trainers; the
   cross-entropy is pinned to an independent ``-log_softmax`` gather.

Run: python tests/cpu/trainers/test_distill_shared_teacher_helpers.py
"""

import types
from unittest import mock

import pytest
import torch
import torch.nn as nn
from accelerate import PartialState
from torch.nn.functional import log_softmax
from trl import SFTTrainer

from src.data.spans import LABEL_IGNORE_INDEX
from src.distributed.parallelism_config import ParallelismConfig
from src.trainers.distillation.losses import privileged_teacher_pass, shifted_token_cross_entropy
from src.trainers.distillation.self_distillation import DistributedSelfDistillationTrainer
from src.trainers.sft import DistributedSFTTrainer

PartialState()  # the trainers' accelerate logger requires an initialized state

_TEACHER_BRANCH_COLLATOR = types.SimpleNamespace(builds_teacher_branch=True)


def _build(
    setup_calls, ep_size=1, tp_size=1, reference_model=None, reference_kl_coef=1.0, sdpg_beta_base=0.0, collator=None
):
    """Construct the trainer with only the HF/TRL machinery stubbed out, recording every reference setup."""

    def _init_cfg(self, kwargs, **_):
        self.parallelism_config = ParallelismConfig(world_size=8, gpus_per_node=8, ep_size=ep_size, tp_size=tp_size)
        return kwargs

    def _sft_init(self, *args, **kwargs):
        self.data_collator = collator

    with (
        mock.patch.object(DistributedSFTTrainer, "_init_distributed_config", _init_cfg),
        mock.patch.object(SFTTrainer, "__init__", _sft_init),
        mock.patch.object(DistributedSelfDistillationTrainer, "_setup_distributed_modes", return_value=None),
        mock.patch.object(DistributedSelfDistillationTrainer, "_resolve_stop_token_ids", return_value=None),
        mock.patch.object(
            DistributedSelfDistillationTrainer,
            "_setup_reference_model",
            lambda self: setup_calls.append(self._reference_model),
        ),
    ):
        return DistributedSelfDistillationTrainer(
            reference_model=reference_model,
            reference_kl_coef=reference_kl_coef,
            reference_kl_loss="unnormalized_kl",
            confidence_weight_opd=True,
            opd_exclude_eos=True,
            sdpg_beta_base=sdpg_beta_base,
        )


@pytest.mark.parametrize(("ep_size", "tp_size"), [(1, 1), (8, 1), (1, 8)], ids=["dp", "ep", "tp"])
def test_a_weighted_reference_is_set_up_on_every_axis(ep_size, tp_size):
    setup_calls = []
    reference = nn.Linear(2, 2)
    _build(setup_calls, ep_size=ep_size, tp_size=tp_size, reference_model=reference)
    assert setup_calls == [reference]


@pytest.mark.parametrize("reference_model", [None, nn.Linear(2, 2)], ids=["no-reference", "unused-reference"])
def test_an_unweighted_anchor_is_never_gated(reference_model):
    """``reference_kl_coef == 0`` never reads the reference, so the gate stays on the branch that does."""
    setup_calls = []
    _build(setup_calls, ep_size=8, reference_model=reference_model, reference_kl_coef=0.0)
    assert setup_calls == []


def test_a_weighted_anchor_without_a_reference_raises():
    """``reference_kl_coef > 0`` with nothing to anchor to would drop ``L_ref`` from every step."""
    with pytest.raises(ValueError, match="no reference_model was passed"):
        _build([], reference_model=None, reference_kl_coef=0.5)


@pytest.mark.parametrize("collator", [None, types.SimpleNamespace(builds_teacher_branch=False)], ids=["plain", "off"])
def test_opd_on_needs_a_collator_that_builds_the_teacher_branch(collator):
    """Refused at construction, on every rank alike: a batch without ``teacher_*`` keys would skip OPD."""
    with pytest.raises(ValueError, match="needs the privileged teacher branch"):
        _build([], reference_kl_coef=0.0, sdpg_beta_base=1.0, collator=collator)


def test_opd_on_accepts_a_teacher_branch_collator_and_opd_off_needs_none():
    _build([], reference_kl_coef=0.0, sdpg_beta_base=1.0, collator=_TEACHER_BRANCH_COLLATOR)
    _build([], reference_kl_coef=0.0, sdpg_beta_base=0.0, collator=None)


class _Rows(nn.Module):
    def __init__(self, rows):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(1))
        self.config = types.SimpleNamespace(get_text_config=lambda: types.SimpleNamespace(vocab_size=rows))
        self.device = torch.device("cpu")


class _Tokenizer:
    def __init__(self, size):
        self.size = size

    def __len__(self):
        return self.size


def _reference_setup(policy_rows, reference_rows, tokenizer_len=10):
    trainer = object.__new__(DistributedSelfDistillationTrainer)
    trainer.model, trainer._reference_model = _Rows(policy_rows), _Rows(reference_rows)
    trainer.processing_class = _Tokenizer(tokenizer_len)
    trainer.reference_kl_coef = 0.1
    trainer._setup_reference_model()
    return trainer


def test_the_reference_shares_the_policy_vocab_check():
    """Same tokenizer: equal rows compare whole, padding past it is sliced away, too few rows raise."""
    assert _reference_setup(16, 16)._reference_vocab_width is None
    assert _reference_setup(16, 12)._reference_vocab_width == 10
    with pytest.raises(ValueError, match="fewer logit rows than the tokenizer"):
        _reference_setup(16, 8)


def test_shifted_ce_is_the_negative_gold_log_prob_and_zero_where_ignored():
    """Pinned to an independent ``-log_softmax`` gather, in fp32 from bf16 logits."""
    torch.manual_seed(0)
    logits = torch.randn(2, 4, 7).bfloat16()
    labels = torch.randint(0, 7, (2, 4))
    labels[0, 1] = labels[1, 3] = LABEL_IGNORE_INDEX
    token_ce = shifted_token_cross_entropy(logits, labels)
    assert token_ce.dtype is torch.float32
    expected = -log_softmax(logits.float(), dim=-1).gather(-1, labels.clamp_min(0).unsqueeze(-1)).squeeze(-1)
    expected[labels == LABEL_IGNORE_INDEX] = 0.0
    torch.testing.assert_close(token_ce, expected)


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
