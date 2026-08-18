#!/usr/bin/env python
"""Every trainer that computes its own loss must opt out of HF's loss-kwargs rescaling.

HF infers ``model_accepts_loss_kwargs`` from the model's forward signature alone — any ``**kwargs``
makes it True — and then SKIPS the ``/gradient_accumulation_steps`` division, on the assumption that
the loss is a token sum the model already normalized by ``num_items_in_batch``. A trainer whose loss
is its own mean has to say otherwise, or its gradients scale with the accumulation count.

Two spellings satisfy that: the ``_loss_is_own_mean`` class attribute the mixin's
``_setup_distributed_modes`` applies, and a direct ``model_accepts_loss_kwargs = False`` assignment
in a base BELOW HF's ``Trainer`` (TRL's own GRPO/DPO/reward trainers do exactly that).

    python tests/cpu/trainers/test_loss_kwargs_contract.py
"""

import inspect

import pytest
from transformers.trainer import Trainer

from src.trainers.mixins.base import DistributedTrainerMixin
from tests.common.rosters import import_all_trainers

import_all_trainers()

_FLAG = "_loss_is_own_mean"
_ASSIGNMENT = "model_accepts_loss_kwargs = False"


def _subclasses(cls: type):
    for sub in cls.__subclasses__():
        yield sub
        yield from _subclasses(sub)


def _computes_its_own_loss(cls: type) -> bool:
    """The trainer's loss is written in this repo rather than inherited from HF/TRL."""
    for name in ("compute_loss", "get_batch_loss_metrics"):
        function = getattr(cls, name, None)
        if function is not None and getattr(function, "__module__", "").startswith("src.trainers"):
            return True
    return False


def _declares_the_attribute(cls: type) -> bool:
    """A subclass declares it; the mixin's own default (False) does not count."""
    return any(_FLAG in base.__dict__ for base in cls.__mro__ if base is not DistributedTrainerMixin)


def _assigns_the_flag(cls: type) -> bool:
    """A base between the trainer and HF's ``Trainer`` hard-assigns the flag.

    Scoped to those bases on purpose. HF's ``Trainer`` carries the string as part of its OWN
    inference and ``DistributedTrainerMixin`` as the conditional application of ``_FLAG``, so a scan
    that reached either would answer True for every trainer in the roster and certify nothing.
    """
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


# Derived from the class hierarchy, not listed: a new trainer joins by existing.
CUSTOM_LOSS_TRAINERS = sorted(
    (
        cls
        for cls in _subclasses(DistributedTrainerMixin)
        if cls.__module__.startswith("src.trainers") and _computes_its_own_loss(cls)
    ),
    key=lambda cls: cls.__name__,
)


def test_the_derived_roster_covers_the_trainer_families():
    """Anti-vacuity: an empty or tiny roster would make the contract test below assert nothing."""
    names = {cls.__name__ for cls in CUSTOM_LOSS_TRAINERS}
    assert len(names) >= 6, f"the roster collapsed to {sorted(names)} — the derivation stopped finding trainers"
    assert {"SmoothMarginPOTrainer", "OfflineGRPOTrainer", "EmbeddingTrainer"} <= names, sorted(names)


@pytest.mark.parametrize("trainer", CUSTOM_LOSS_TRAINERS, ids=lambda cls: cls.__name__)
def test_every_custom_loss_trainer_opts_out_of_loss_kwargs_rescaling(trainer):
    assert _declares_the_attribute(trainer) or _assigns_the_flag(trainer), (
        f"{trainer.__name__} computes its own loss but neither declares {_FLAG} nor assigns "
        f"'{_ASSIGNMENT}' anywhere in its MRO: HF will keep whatever it inferred from the model's "
        f"forward signature and may drop the /gradient_accumulation_steps division."
    )


def test_the_gate_rejects_a_trainer_that_forgets_the_flag():
    """Anti-tautology: the two spellings are the ONLY ways past the parametrization above, so a
    subclass that computes its own loss and says nothing must be rejected by both."""

    class _Forgetful(DistributedTrainerMixin, Trainer):
        def compute_loss(self, *args, **kwargs):
            raise NotImplementedError

    assert not _declares_the_attribute(_Forgetful) and not _assigns_the_flag(_Forgetful), (
        "a trainer declaring neither spelling passes the contract test, which therefore certifies nothing"
    )


class _SpineStopped(Exception):
    """Carries control out of the spine once it is past the loss-kwargs opt-out."""


class _SpineHost:
    """A bare host carrying only what the spine reads before the parallelism setup it cannot run."""

    _deferred_liger_kernel = False
    _loss_outside_model_forward = False
    model = None

    def __init__(self, loss_is_own_mean: bool):
        self._loss_is_own_mean = loss_is_own_mean
        self.model_accepts_loss_kwargs = True  # what HF inferred from the forward signature

    def _log_parallelism_config(self):
        raise _SpineStopped


@pytest.mark.parametrize("loss_is_own_mean", [False, True])
def test_the_spine_applies_the_declared_attribute(loss_is_own_mean):
    """The one place the attribute turns into HF's flag — and declared-only, so a trainer that says
    nothing keeps whatever HF inferred."""
    host = _SpineHost(loss_is_own_mean)

    with pytest.raises(_SpineStopped):
        DistributedTrainerMixin._setup_distributed_modes(host)

    assert host.model_accepts_loss_kwargs is not loss_is_own_mean, (
        f"_setup_distributed_modes left model_accepts_loss_kwargs={host.model_accepts_loss_kwargs!r} "
        f"for a trainer with {_FLAG}={loss_is_own_mean!r}"
    )


def test_the_default_leaves_hf_inference_alone():
    """Declared-only: the mixin's own default must not opt anything out on its behalf."""
    assert DistributedTrainerMixin._loss_is_own_mean is False


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
