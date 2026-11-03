#!/usr/bin/env python
"""Every trainer that computes its own loss must opt out of HF's loss-kwargs rescaling.
``**kwargs`` in a forward makes HF skip the ``/gradient_accumulation_steps`` division, so gradients scale with it."""

import inspect

import pytest
import torch
from transformers import TrainingArguments
from transformers.trainer import Trainer

from halo.trainers.distributed_trainer import DistributedTrainerMixin
from tests.common.rosters import distributed_trainer_classes

_FLAG = "LOSS_IS_OWN_MEAN"
_ASSIGNMENT = "model_accepts_loss_kwargs = False"
_OWN_MEAN = 8.0
_ACCUMULATION = 4


def _computes_its_own_loss(cls: type) -> bool:
    for name in ("compute_loss", "get_batch_loss_metrics"):
        function = getattr(cls, name, None)
        if function is not None and getattr(function, "__module__", "").startswith("halo.trainers"):
            return True
    return False


def _declares_the_attribute(cls: type) -> bool:
    return any(_FLAG in base.__dict__ for base in cls.__mro__ if base is not DistributedTrainerMixin)


def _assigns_the_flag(cls: type) -> bool:
    """A base between the trainer and HF's ``Trainer`` hard-assigns the flag.
    HF's ``Trainer`` and the mixin both carry the string, so a scan reaching them would certify every trainer."""
    for base in cls.__mro__:
        if base is Trainer:
            break
        if base is DistributedTrainerMixin:
            continue
        try:
            source = inspect.getsource(base)
        except (OSError, TypeError):  # C-implemented or dynamically built base
            continue
        if _ASSIGNMENT in source:
            return True
    return False


# Derived from the class hierarchy, so a new trainer joins by existing.
CUSTOM_LOSS_TRAINERS = [cls for cls in distributed_trainer_classes() if _computes_its_own_loss(cls)]


def test_the_derived_roster_covers_the_trainer_families():
    """An empty or tiny roster would make the contract test below assert nothing."""
    names = {cls.__name__ for cls in CUSTOM_LOSS_TRAINERS}
    assert len(names) >= 6, f"the roster collapsed to {sorted(names)} — the derivation stopped finding trainers"
    assert {"SMPOTrainer", "OfflineGRPOTrainer", "EmbeddingTrainer"} <= names, sorted(names)


@pytest.mark.parametrize("trainer", CUSTOM_LOSS_TRAINERS, ids=lambda cls: cls.__name__)
def test_every_custom_loss_trainer_opts_out_of_loss_kwargs_rescaling(trainer):
    assert _declares_the_attribute(trainer) or _assigns_the_flag(trainer), (
        f"{trainer.__name__} computes its own loss but neither declares {_FLAG} nor assigns "
        f"'{_ASSIGNMENT}' anywhere in its MRO: HF will keep whatever it inferred from the model's "
        f"forward signature and may drop the /gradient_accumulation_steps division."
    )


def test_the_gate_rejects_a_trainer_that_forgets_the_flag():
    class _Forgetful(DistributedTrainerMixin, Trainer):
        def compute_loss(self, *args, **kwargs):
            raise NotImplementedError

    assert not _declares_the_attribute(_Forgetful) and not _assigns_the_flag(_Forgetful), (
        "a trainer declaring neither spelling passes the contract test, which therefore certifies nothing"
    )


class _SpineStopped(Exception):
    """Carries control out of the spine once it is past the loss-kwargs opt-out."""


class _SpineHost:
    _deferred_liger_kernel = False
    model = None
    loss_is_scaled_for_ga = None  # HF's class default: infer it per step

    def __init__(self, loss_is_own_mean: bool):
        self.LOSS_IS_OWN_MEAN = loss_is_own_mean
        self.model_accepts_loss_kwargs = True  # what HF inferred from the forward signature

    def _snapshot_model_config_state(self, model_config):
        raise _SpineStopped


@pytest.mark.parametrize("loss_is_own_mean", [False, True])
def test_the_spine_applies_the_declared_attribute(loss_is_own_mean):
    """Declared-only: a trainer that says nothing keeps whatever HF inferred."""
    host = _SpineHost(loss_is_own_mean)

    with pytest.raises(_SpineStopped):
        DistributedTrainerMixin._restore_trl_overrides(host)

    assert host.model_accepts_loss_kwargs is not loss_is_own_mean, (
        f"_restore_trl_overrides left model_accepts_loss_kwargs={host.model_accepts_loss_kwargs!r} "
        f"for a trainer with {_FLAG}={loss_is_own_mean!r}"
    )
    assert host.loss_is_scaled_for_ga is (False if loss_is_own_mean else None)


class _OwnMeanLoss(Trainer):
    """Returns its own mean and ignores ``compute_loss_func``, as every ``LOSS_IS_OWN_MEAN`` trainer does."""

    LOSS_IS_OWN_MEAN = True
    _deferred_liger_kernel = False
    _pristine_special_token_ids = ["snapshotted"]

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        return model(inputs["x"]).sum() * 0.0 + _OWN_MEAN


def test_a_compute_loss_func_leaves_an_own_mean_loss_divided_by_accumulation(tmp_path):
    """HF drops the division whenever a ``compute_loss_func`` is set, on the premise that it scaled the loss."""
    trainer = _OwnMeanLoss(
        model=torch.nn.Linear(1, 1),
        args=TrainingArguments(output_dir=str(tmp_path), use_cpu=True, report_to="none"),
        compute_loss_func=lambda *args, **kwargs: pytest.fail("this trainer's loss never calls compute_loss_func"),
    )
    DistributedTrainerMixin._restore_trl_overrides(trainer)
    trainer.current_gradient_accumulation_steps = _ACCUMULATION

    loss = trainer.training_step(trainer.model, {"x": torch.ones(1, 1)})

    assert loss.item() == _OWN_MEAN / _ACCUMULATION


def test_the_default_leaves_hf_inference_alone():
    assert DistributedTrainerMixin.LOSS_IS_OWN_MEAN is False


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
