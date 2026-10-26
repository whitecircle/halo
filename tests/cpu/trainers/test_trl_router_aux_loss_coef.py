#!/usr/bin/env python
"""TRL's router aux-loss coefficient is the balancing mode's, and TRL's config writes do not stick.

TRL 1.14 reads an unset ``router_aux_loss_coef`` off the model config and turns its aux loss on wherever
that is non-zero: a hub coefficient (gpt-oss ships 0.9) would then train the aux term under
``moe_balancing: none`` or a bias mode and add it to a policy-gradient loss. Its SFT trainer also writes
``output_router_logits`` and the coefficient onto the text config at construction, which would undo the
mode's own setting and reach the exported ``config.json``. The trainer mixin hands TRL the coefficient
the mode put in effect where TRL adds the term itself (DPO, KTO) and 0.0 everywhere else, refuses one
set on the trainer config, and restores the config fields after TRL's construction.

    python tests/cpu/trainers/test_trl_router_aux_loss_coef.py
"""

import types

import pytest
import torch.nn as nn
from transformers import CONFIG_MAPPING

from src.models.config_access import ROUTER_AUX_FIELDS, restore_config_fields, snapshot_config_fields
from src.trainers.grpo.environmental import DistributedAsyncEnvironmentalGRPOTrainer
from src.trainers.grpo.online import DistributedGRPOTrainer
from src.trainers.mixins.base import DistributedTrainerMixin
from src.trainers.preference.dpo import DistributedDPOTrainer
from src.trainers.preference.kto import DistributedKTOTrainer
from src.trainers.sft import DistributedSFTTrainer

HUB_COEF = 0.9


def _model(output_router_logits: bool) -> nn.Module:
    model = nn.Module()
    model.config = CONFIG_MAPPING["qwen3_moe"](
        router_aux_loss_coef=HUB_COEF, output_router_logits=output_router_logits
    )
    return model


def _handed(trainer_cls, output_router_logits: bool, coef=None) -> float:
    args = types.SimpleNamespace(router_aux_loss_coef=coef)
    host = object.__new__(trainer_cls)
    DistributedTrainerMixin._hand_trl_the_router_aux_loss_coef(host, _model(output_router_logits), args)
    return args.router_aux_loss_coef


@pytest.mark.parametrize("trainer_cls", [DistributedDPOTrainer, DistributedKTOTrainer])
def test_a_trainer_whose_trl_loss_adds_the_term_gets_the_mode_s_coefficient(trainer_cls):
    assert _handed(trainer_cls, output_router_logits=True) == HUB_COEF
    assert _handed(trainer_cls, output_router_logits=False) == 0.0, "a mode with the logits off trains no aux term"


@pytest.mark.parametrize(
    "trainer_cls", [DistributedSFTTrainer, DistributedGRPOTrainer, DistributedAsyncEnvironmentalGRPOTrainer]
)
def test_every_other_trl_trainer_gets_zero(trainer_cls):
    """SFT's model forward adds the term itself; a policy-gradient loss keeps it inert."""
    assert _handed(trainer_cls, output_router_logits=True) == 0.0


def test_a_coefficient_on_the_trainer_config_is_refused():
    with pytest.raises(ValueError, match="owned by moe_balancing"):
        _handed(DistributedSFTTrainer, output_router_logits=True, coef=0.01)


def test_trls_config_writes_are_undone():
    """What TRL 1.14's SFTTrainer does to the text config at construction, then the mixin's restore."""
    model = _model(output_router_logits=True)
    snapshot = snapshot_config_fields(model.config, ROUTER_AUX_FIELDS)
    text_config = model.config.get_text_config()
    text_config.output_router_logits, text_config.router_aux_loss_coef = False, 0.0
    restore_config_fields(snapshot)
    assert (text_config.output_router_logits, text_config.router_aux_loss_coef) == (True, HUB_COEF)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
