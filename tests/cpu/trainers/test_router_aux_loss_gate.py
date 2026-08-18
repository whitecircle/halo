#!/usr/bin/env python
"""Test the gate on MoE balancing that rides an aux loss a trainer's objective never adds.

``*ForCausalLM.forward`` folds ``router_aux_loss_coef * load_balancing_loss_func(...)`` into the
loss only where it builds that loss from ``labels``. Preference / log-prob / distillation / pooled-
head trainers forward without labels and assemble their own loss, so the aux term never enters the
graph: routing collapses over training while the run still pays for ``output_router_logits``.
``_validate_router_aux_loss_consumable`` raises on an EXPLICIT ``moe_balancing=aux_loss`` there and
warns when ``auto`` resolved to it — the non-PP counterpart of pipeline_parallel.split's rejection.

Run: python tests/cpu/trainers/test_router_aux_loss_gate.py
"""

import types

import pytest
import torch.nn as nn

import src.trainers.mixins.validation as validation_module
from src.models.moe_balancing import router_logits_forced_off
from src.trainers.distillation.self_distillation import DistributedSelfDistillationTrainer
from src.trainers.distillation.teacher_distillation import DistributedDistillationTrainer
from src.trainers.grpo.offline import OfflineGRPOTrainer
from src.trainers.mixins.base import DistributedTrainerMixin
from src.trainers.mixins.validation import ParallelismValidationMixin, active_router_aux_loss_coef
from src.trainers.preference.dpo import DistributedDPOTrainer
from src.trainers.preference.kto import DistributedKTOTrainer
from src.trainers.preference.smpo import SmoothMarginPOTrainer
from src.trainers.reward.bradley_terry import DistributedRewardTrainer
from src.trainers.sft import DistributedSFTTrainer


class _Cfg:
    def __init__(self, output_router_logits=True, router_aux_loss_coef=1e-3):
        self.output_router_logits = output_router_logits
        self.router_aux_loss_coef = router_aux_loss_coef

    def get_text_config(self):
        return self


class _Model(nn.Module):
    def __init__(self, config=None):
        super().__init__()
        self.config = config if config is not None else _Cfg()


class _Trainer(ParallelismValidationMixin):
    """Stand-in for a trainer whose loss cannot carry the aux term (the mixin default)."""

    _consumes_router_aux_loss = DistributedTrainerMixin._consumes_router_aux_loss
    # Borrowed, not reimplemented: a stub unwrap could pass while the real one is broken.
    _top_level_model = DistributedTrainerMixin._top_level_model

    def __init__(self, moe_balancing="auto", config=None):
        self.model = _Model(config)
        self._moe_balancing = moe_balancing


class _ConsumingTrainer(_Trainer):
    _consumes_router_aux_loss = True


def _validate(trainer):
    ParallelismValidationMixin._validate_router_aux_loss_consumable(trainer)


def _capture_warnings(trainer) -> list[str]:
    calls: list[str] = []
    original = validation_module.logger.warning
    validation_module.logger.warning = lambda msg, *a, **k: calls.append(str(msg))
    try:
        _validate(trainer)
    finally:
        validation_module.logger.warning = original
    return calls


def test_only_labels_forwarding_trainers_declare_consumption():
    # SFT hands the model ``labels`` (HF adds the aux term); everything else builds its own loss.
    assert DistributedSFTTrainer._consumes_router_aux_loss is True
    # KTO builds its own loss yet TRL adds the aux term back — consumption must be declared, not inferred.
    assert DistributedKTOTrainer._consumes_router_aux_loss is True
    for trainer_cls in (
        SmoothMarginPOTrainer,
        DistributedDPOTrainer,
        DistributedRewardTrainer,
        OfflineGRPOTrainer,
        DistributedDistillationTrainer,
        # Inherits DistributedSFTTrainer but forwards WITHOUT labels — must override back to False.
        DistributedSelfDistillationTrainer,
    ):
        assert trainer_cls._consumes_router_aux_loss is False, trainer_cls.__name__


def test_explicit_aux_loss_is_rejected():
    with pytest.raises(ValueError, match="balances nothing"):
        _validate(_Trainer(moe_balancing="aux_loss"))


def test_auto_resolved_aux_loss_warns():
    warnings = _capture_warnings(_Trainer(moe_balancing="auto"))
    assert any("INERT" in w for w in warnings), warnings


def test_consuming_trainer_passes():
    _validate(_ConsumingTrainer(moe_balancing="aux_loss"))


def test_router_logits_off_is_not_flagged():
    # The logits are already off, so nothing is being paid for.
    assert _capture_warnings(_Trainer(moe_balancing="aux_loss", config=_Cfg(output_router_logits=False))) == []


def test_zero_coefficient_is_not_flagged():
    # An aux-loss-free router (GLM-4 MoE Lite noaux_tc) has no term to drop.
    _validate(_Trainer(moe_balancing="aux_loss", config=_Cfg(router_aux_loss_coef=0.0)))


def test_dense_model_is_not_flagged():
    _validate(_Trainer(moe_balancing="aux_loss", config=types.SimpleNamespace()))


def test_auto_resolved_aux_loss_turns_the_router_logits_back_off():
    """Warning and then paying the tensor anyway is the worst of both: ``auto`` turns the flag on
    upstream of this gate, so the gate takes it back — stamped, or MoEMetricsCallback (built for
    the aux_loss mode) re-enables it at train begin."""
    cfg = _Cfg()
    warnings = _capture_warnings(_Trainer(moe_balancing="auto", config=cfg))
    assert cfg.output_router_logits is False
    assert router_logits_forced_off(cfg) is True
    assert any("INERT" in w for w in warnings), warnings


def test_router_logits_enabled_outside_auto_are_left_alone():
    """Anti-vacuity + scope: under an explicit mode the flag is the user's documented metrics opt-in,
    not the balancing strategy's doing, so it must survive the warning."""
    cfg = _Cfg()
    warnings = _capture_warnings(_Trainer(moe_balancing="none", config=cfg))
    assert cfg.output_router_logits is True
    assert router_logits_forced_off(cfg) is False
    assert any("INERT" in w for w in warnings), warnings


def test_consuming_trainer_keeps_router_logits_on():
    """The trainers that DO add the aux term must keep the flag the aux loss rides on."""
    cfg = _Cfg()
    _validate(_ConsumingTrainer(moe_balancing="auto", config=cfg))
    assert cfg.output_router_logits is True
    assert router_logits_forced_off(cfg) is False


def test_coef_read_from_the_text_config_of_a_multimodal_model():
    text_config = _Cfg()
    outer = types.SimpleNamespace(get_text_config=lambda: text_config)
    assert active_router_aux_loss_coef(_Model(outer)) == pytest.approx(1e-3)


def test_coef_zero_without_router_logits():
    assert active_router_aux_loss_coef(_Model(_Cfg(output_router_logits=False))) == 0.0


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
