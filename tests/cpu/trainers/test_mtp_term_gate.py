#!/usr/bin/env python
"""A head's term needs the model's own forward to see the labels: a trainer that scores its own logits, Context
Parallelism's labels=None and anything popping the labels would run the head and train nothing, so construction raises."""

from types import SimpleNamespace

import pytest
import torch.nn as nn

from halo.trainers import distributed_trainer
from halo.trainers.distillation.sdpg import DistributedSDPGTrainer
from halo.trainers.distillation.teacher_distillation import DistributedTeacherDistillationTrainer
from halo.trainers.distributed_trainer import DistributedTrainerMixin
from halo.trainers.grpo.environmental.trainer import DistributedAsyncEnvironmentalGRPOTrainer
from halo.trainers.grpo.offline.trainer import OfflineGRPOTrainer
from halo.trainers.grpo.online import DistributedOnlineGRPOTrainer
from halo.trainers.sft import DistributedSFTTrainer
from halo.trainers.trainer_rules import reject_unreachable_mtp_term
from tests.common.mtp_heads import tiny_model_with_head
from tests.common.tiny_models import TINY_MOE_FAMILIES, tiny_family_model


class _Config:
    def __init__(self, factor=None):
        if factor is not None:
            self.mtp_loss_scaling_factor = factor

    def get_text_config(self):
        return self


class _RemoteHeadModel(nn.Module):
    def __init__(self, factor=None):
        super().__init__()
        self.config = _Config(factor)


def _reject(model, **overrides) -> None:
    reject_unreachable_mtp_term(
        model,
        **{
            "trainer": "SomeTrainer",
            "labelled_forward": True,
            "is_cp_mode": False,
            "pops_labels": False,
            "rows_per_forward": 1,
            "eval_rows_per_forward": 1,
            **overrides,
        },
    )


def _live_head():
    return tiny_model_with_head("qwen3_5_moe_text")[0]


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"labelled_forward": False}, "no labels forward"),
        ({"is_cp_mode": True}, "labels=None"),
        ({"pops_labels": True}, "takes the labels"),
    ],
    ids=["unlabelled_trainer", "cp", "label_smoother"],
)
@pytest.mark.parametrize("build", [_live_head, lambda: _RemoteHeadModel(0.3)], ids=["native_head", "remote_head"])
def test_a_term_nothing_would_add_is_refused(build, overrides, match):
    with pytest.raises(ValueError, match=match):
        _reject(build(), **overrides)


def test_a_remote_head_past_one_row_per_forward_is_refused():
    """The remote head flattens rows before its shift: B>1 trains each row's last position on the next row's token."""
    with pytest.raises(ValueError, match="per_device_train_batch_size: 1"):
        _reject(_RemoteHeadModel(0.3), rows_per_forward=2)


def test_a_remote_head_past_one_row_in_an_evaluation_is_refused():
    """Its forward adds the flattened term with no training check, so ``eval_loss`` holds a row-crossing term."""
    with pytest.raises(ValueError, match="per_device_eval_batch_size: 1"):
        _reject(_RemoteHeadModel(0.3), eval_rows_per_forward=2)


def test_a_native_head_keeps_its_row_alignment_past_one_row():
    _reject(_live_head(), rows_per_forward=8, eval_rows_per_forward=8)


@pytest.mark.parametrize("name", ["qwen3_5_moe_text", "glm4_moe_lite"])
def test_a_head_the_freeze_patterns_froze_whole_is_refused(name):
    model, head = tiny_model_with_head(name)
    _reject(model)
    next(head.parameters()).requires_grad_(False)
    _reject(model)
    head.requires_grad_(False)
    with pytest.raises(ValueError, match="froze every parameter"):
        _reject(model)


@pytest.mark.parametrize("factor", [None, 0, 0.0], ids=["absent", "zero", "zero_float"])
def test_no_head_and_no_factor_passes_every_rule(factor):
    _reject(
        _RemoteHeadModel(factor),
        labelled_forward=False,
        is_cp_mode=True,
        pops_labels=True,
        rows_per_forward=8,
        eval_rows_per_forward=8,
    )


def test_a_model_with_no_head_is_untouched_by_the_gate():
    _reject(
        tiny_family_model(TINY_MOE_FAMILIES["qwen3_next"]),
        labelled_forward=False,
        is_cp_mode=True,
        pops_labels=True,
        rows_per_forward=8,
    )


class _StoppedBeforeTheWrap(DistributedTrainerMixin):
    """The spine's mode setup up to the gate, on what it reads there."""

    def __init__(self, model, *, is_cp_mode=False, label_smoother=None):
        self.model = model
        self.parallelism_config = SimpleNamespace(is_tp_mode=False, is_cp_mode=is_cp_mode)
        self.args = SimpleNamespace(
            optim="adamw_torch",
            per_device_train_batch_size=1,
            per_device_eval_batch_size=8,
            eval_strategy="no",
            eval_on_start=False,
        )
        self.label_smoother = label_smoother
        self.compute_loss_func = None
        self._bf16_optimizer = False

    def _restore_trl_overrides(self) -> None:
        pass

    def _disable_inert_router_logits(self) -> None:
        pass


def test_the_trainer_spine_runs_the_gate_at_mode_setup(monkeypatch):
    monkeypatch.setattr(distributed_trainer, "agree_rank_uniform_env", lambda: None)
    with pytest.raises(ValueError, match="no labels forward"):
        _StoppedBeforeTheWrap(_live_head())._configure_distributed_modes()


@pytest.mark.parametrize(
    ("eval_strategy", "eval_on_start", "rows"), [("no", False, 1), ("steps", False, 8), ("no", True, 8)]
)
def test_the_spine_hands_the_gate_the_eval_batch_only_where_the_run_evaluates(
    monkeypatch, eval_strategy, eval_on_start, rows
):
    """Off an evaluation the default eval batch of 8 forwards nothing, and must not refuse a run."""

    class _Stop(Exception):
        pass

    seen = {}

    def capture(_model, **kwargs):
        seen.update(kwargs)
        raise _Stop

    monkeypatch.setattr(distributed_trainer, "agree_rank_uniform_env", lambda: None)
    monkeypatch.setattr(distributed_trainer, "reject_unreachable_mtp_term", capture)
    host = _StoppedBeforeTheWrap(_live_head())
    host.args.eval_strategy, host.args.eval_on_start = eval_strategy, eval_on_start
    with pytest.raises(_Stop):
        host._configure_distributed_modes()
    assert seen["eval_rows_per_forward"] == rows


@pytest.mark.parametrize(
    "trainer",
    [DistributedOnlineGRPOTrainer, DistributedAsyncEnvironmentalGRPOTrainer, DistributedSDPGTrainer],
    ids=lambda trainer: trainer.__name__,
)
def test_a_trainer_that_syncs_a_generator_refuses_a_head_at_construction(trainer):
    """The sync walks ``named_parameters`` and every EP layer, so a live head would reach the server's main model. No
    entry script builds one for these, and this gate stands behind that."""
    assert trainer.SYNCS_TO_EXTERNAL_GENERATOR
    with pytest.raises(ValueError, match="no labels forward"):
        _reject(_live_head(), labelled_forward=trainer.ROUTER_AUX_LOSS_ADDED_BY == "model_forward")


def test_labelled_forward_reads_what_adds_the_router_aux_loss():
    assert DistributedSFTTrainer.ROUTER_AUX_LOSS_ADDED_BY == "model_forward"
    for trainer in (DistributedTrainerMixin, OfflineGRPOTrainer, DistributedTeacherDistillationTrainer):
        assert trainer.ROUTER_AUX_LOSS_ADDED_BY != "model_forward", trainer.__name__


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
