#!/usr/bin/env python
"""A trainer built with a positional ``model``/``args`` gets the setup a keyword-built one gets, and a
CP save finds the Ulysses wrapper beneath ``torch.compile``.

``_init_distributed_config`` runs before ``super().__init__`` and reads both arguments: the config to
surface ``fp32_grad_reduce``, resolve AdamWBF16, filter Liger and force ``save_on_each_node``, and the
model's config to force reentrant checkpointing on a MoE. Arguments passed positionally never reach
``kwargs``, so each ``(*args, **kwargs)`` trainer hands its positionals over and the mixin reads them
off the slot table of the TRL base they are forwarded to. The construction is cut short right after
that setup (TRL's own ctor needs a live model and, for GRPO, a server).

    python tests/cpu/trainers/test_positional_ctor_setup.py
"""

import types

import pytest
import torch
import torch.nn as nn
from accelerate import PartialState
from transformers import PretrainedConfig

from src.distributed.context_parallel.wrapper import UlyssesCPModelWrapper
from src.distributed.parallelism_config import ParallelismConfig
from src.trainers.distillation.sdpg import DistributedSDPGTrainer
from src.trainers.distillation.self_distillation import DistributedSelfDistillationTrainer
from src.trainers.grpo.environmental import DistributedAsyncEnvironmentalGRPOTrainer
from src.trainers.grpo.online import DistributedGRPOTrainer
from src.trainers.mixins.base import DistributedTrainerMixin
from src.trainers.preference.dpo import DistributedDPOTrainer
from src.trainers.preference.kto import DistributedKTOTrainer
from src.trainers.reward.bradley_terry import DistributedRewardTrainer
from src.trainers.sft import DistributedSFTTrainer

PartialState()  # the trainers' accelerate logger requires an initialized state

# (trainer, the positional slot of its config); the model is slot 0 on every one.
TRAINERS = [
    (DistributedSFTTrainer, 1),
    (DistributedSelfDistillationTrainer, 1),
    (DistributedRewardTrainer, 1),
    (DistributedDPOTrainer, 2),
    (DistributedKTOTrainer, 2),
    (DistributedGRPOTrainer, 2),
    (DistributedSDPGTrainer, 2),
    (DistributedAsyncEnvironmentalGRPOTrainer, 2),
]


class _SetupDone(Exception):
    """Carries control out of the trainer ctor once the distributed setup has run."""


@pytest.fixture(autouse=True)
def stop_after_distributed_setup(monkeypatch):
    real = DistributedTrainerMixin._init_distributed_config

    def _run_then_stop(self, kwargs, *args, **extra):
        real(self, kwargs, *args, **extra)
        raise _SetupDone

    monkeypatch.setattr(DistributedTrainerMixin, "_init_distributed_config", _run_then_stop)


def _training_args():
    return types.SimpleNamespace(
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs=None,
        use_liger_kernel=False,
        bf16=False,
        optim="adamw_torch",
        save_on_each_node=False,
    )


def _moe_model():
    """A MoE by its config, which is what the reentrant rule keys on when no EP/CP axis is set."""
    model = nn.Linear(2, 2)
    model.config = PretrainedConfig(num_experts=8)
    return model


def _construct(trainer_cls, *args, **kwargs):
    with pytest.raises(_SetupDone):
        trainer_cls(*args, parallelism_config=ParallelismConfig(), **kwargs)


def _assert_setup_applied(training_args):
    assert hasattr(training_args, "fp32_grad_reduce"), "the config never reached the mixin's setup"
    assert training_args.gradient_checkpointing_kwargs == {"use_reentrant": True}, (
        "the model's config never reached the reentrant-checkpointing rule, so a MoE would recompute "
        "its router non-reentrantly"
    )


@pytest.mark.parametrize(("trainer_cls", "args_slot"), TRAINERS, ids=lambda v: getattr(v, "__name__", str(v)))
def test_positional_model_and_config_get_the_setup(trainer_cls, args_slot):
    training_args = _training_args()
    positional = [_moe_model()] + [None] * (args_slot - 1) + [training_args]
    _construct(trainer_cls, *positional)
    _assert_setup_applied(training_args)


@pytest.mark.parametrize(("trainer_cls", "args_slot"), TRAINERS, ids=lambda v: getattr(v, "__name__", str(v)))
def test_keyword_model_and_config_get_the_same_setup(trainer_cls, args_slot):
    """The calling convention every script uses must be unaffected."""
    training_args = _training_args()
    _construct(trainer_cls, model=_moe_model(), args=training_args)
    _assert_setup_applied(training_args)


class _OwnSignatureSFTTrainer(DistributedSFTTrainer):
    """A user subclass with its own positional signature, forwarding positionally to the base."""

    def __init__(self, model, config, note=None, **kwargs):
        super().__init__(model, config, **kwargs)


def test_a_subclass_with_its_own_signature_is_read_through_the_base_it_forwards_to():
    """The slots are the forwarded base's, not the runtime class's: this subclass names its config
    ``config``, and a lookup of ``args`` in its own signature would find none."""
    training_args = _training_args()
    _construct(_OwnSignatureSFTTrainer, _moe_model(), training_args, "note")
    _assert_setup_applied(training_args)


class _CPWrapper(UlyssesCPModelWrapper):
    """The wrapper type without its attention patching, which needs a real attention stack."""

    def __init__(self):
        nn.Module.__init__(self)
        self.model = nn.Linear(2, 2)


def _cp_host(model):
    host = types.SimpleNamespace(model=model)
    host._top_level_model = types.MethodType(DistributedTrainerMixin._top_level_model, host)
    return host


def test_cp_wrapper_is_found_beneath_torch_compile():
    """``torch_compile`` leaves ``self.model`` an ``OptimizedModule``; the saver choice needs the wrapper."""
    wrapper = _CPWrapper()
    assert DistributedTrainerMixin._find_cp_wrapper(_cp_host(torch.compile(wrapper))) is wrapper


def test_cp_wrapper_is_found_uncompiled_and_absent_without_cp():
    wrapper = _CPWrapper()
    assert DistributedTrainerMixin._find_cp_wrapper(_cp_host(wrapper)) is wrapper
    assert DistributedTrainerMixin._find_cp_wrapper(_cp_host(torch.compile(nn.Linear(2, 2)))) is None


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
