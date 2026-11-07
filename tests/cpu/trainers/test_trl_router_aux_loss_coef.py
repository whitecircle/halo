#!/usr/bin/env python
"""TRL's router aux-loss coefficient is the balancing mode's, and TRL's config writes do not stick.
TRL 1.14 trains the aux term wherever the config's coefficient is non-zero (gpt-oss ships 0.9), under any mode."""

import types
from unittest.mock import patch

import pytest
import torch.nn as nn
from transformers import CONFIG_MAPPING

import halo.trainers.distributed_trainer as spine
from halo.distributed.parallelism_config import ParallelismConfig
from halo.models.introspection.config_access import ROUTER_AUX_FIELDS, restore_config_fields, snapshot_config_fields
from halo.trainers.distributed_trainer import DistributedTrainerMixin
from halo.trainers.grpo.environmental.trainer import DistributedAsyncEnvironmentalGRPOTrainer
from halo.trainers.grpo.online import DistributedOnlineGRPOTrainer
from halo.trainers.preference.dpo import DistributedDPOTrainer
from halo.trainers.preference.kto import DistributedKTOTrainer
from halo.trainers.sft import DistributedSFTTrainer

HUB_COEF = 0.9


def _model(output_router_logits: bool) -> nn.Module:
    model = nn.Module()
    model.config = CONFIG_MAPPING["qwen3_moe"](
        router_aux_loss_coef=HUB_COEF, output_router_logits=output_router_logits
    )
    return model


def _handed(trainer_cls, output_router_logits: bool, coef=None) -> float:
    """The coefficient the trainer's own TrainingArguments step hands TRL; the filesystem steps are stubbed."""
    args = types.SimpleNamespace(router_aux_loss_coef=coef, output_dir="unused", use_liger_kernel=False)
    host = object.__new__(trainer_cls)
    host.parallelism_config = ParallelismConfig()
    with patch.object(spine, "verify_output_filesystem_sharing"), patch.object(spine, "align_save_on_each_node"):
        DistributedTrainerMixin._apply_training_args_overrides(host, args, _model(output_router_logits))
    return args.router_aux_loss_coef


@pytest.mark.parametrize("trainer_cls", [DistributedDPOTrainer, DistributedKTOTrainer])
def test_a_trainer_whose_trl_loss_adds_the_term_gets_the_mode_s_coefficient(trainer_cls):
    assert _handed(trainer_cls, output_router_logits=True) == HUB_COEF
    assert _handed(trainer_cls, output_router_logits=False) == 0.0, "a mode with the logits off trains no aux term"


@pytest.mark.parametrize(
    "trainer_cls", [DistributedSFTTrainer, DistributedOnlineGRPOTrainer, DistributedAsyncEnvironmentalGRPOTrainer]
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
