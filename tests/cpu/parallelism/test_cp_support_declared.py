#!/usr/bin/env python
"""Context Parallelism is declare-to-enable per trainer, and the declaration is the whole gate.

Nothing inspects a trainer's loss for CP-safety: ``_validate_parallelism_modes`` reads only the
``_supports_cp`` class attribute (off on ``DistributedTrainerMixin``) before a ``cp_size > 1`` run
starts. A flag switched on for a trainer whose objective has not been verified under sequence
chunking trains silently wrong (a sequence log-prob pooled over one chunk, say), so the CP set is
pinned here, and each trainer's own resolved validator is driven with a CP config: every other
trainer must refuse it by name, the CP trainers must accept it.

Run: python tests/cpu/parallelism/test_cp_support_declared.py
"""

import pytest

from src.trainers.mixins.base import DistributedTrainerMixin
from tests.common.parallelism import make_parallelism_config
from tests.common.rosters import distributed_trainer_classes

# The trainers whose objective is verified under CP (CLAUDE.md, "Distributed trainers" CP column).
CP_TRAINERS = {"DistributedSFTTrainer", "SmoothMarginPOTrainer"}

# Launcher env that makes the validator refuse every parallel mode for an unrelated reason.
_ACCELERATE_VARS = ("ACCELERATE_MIXED_PRECISION", "ACCELERATE_USE_FSDP")

# Derived from the class hierarchy, not listed: a new trainer joins by existing, at the default off.
TRAINERS = distributed_trainer_classes()


def _with_cp_config(trainer_cls: type):
    """An uninitialised ``trainer_cls`` holding a CP config, so the validator the trainer itself
    resolves runs without the model, datasets and process groups construction would need."""
    trainer = object.__new__(trainer_cls)
    trainer.parallelism_config = make_parallelism_config(world_size=2, gpus_per_node=2, cp_size=2)
    assert trainer.parallelism_config.is_cp_mode, "premise: cp_size=2 must put the config in CP mode"
    return trainer


def test_the_derived_roster_covers_every_trainer():
    """Anti-vacuity: a roster that stopped finding trainers would make the checks below assert nothing."""
    names = {cls.__name__ for cls in TRAINERS}
    assert len(names) >= 13, f"the roster collapsed to {sorted(names)}"
    assert names >= CP_TRAINERS, f"CP trainers missing from the roster: {sorted(CP_TRAINERS - names)}"


def test_the_mixin_default_is_off():
    """Declare-to-enable: a trainer that never considered CP must not inherit it."""
    assert DistributedTrainerMixin._supports_cp is False


@pytest.mark.parametrize("trainer_cls", TRAINERS, ids=lambda cls: cls.__name__)
def test_cp_support_is_exactly_the_verified_set(trainer_cls):
    expected = trainer_cls.__name__ in CP_TRAINERS
    assert trainer_cls._supports_cp is expected, (
        f"{trainer_cls.__name__}._supports_cp is {trainer_cls._supports_cp!r}, expected {expected}: enabling CP "
        f"needs the objective verified under sequence chunking and this pin updated with it"
    )


@pytest.mark.parametrize("trainer_cls", TRAINERS, ids=lambda cls: cls.__name__)
def test_the_validator_enforces_the_declaration(trainer_cls, monkeypatch):
    for var in _ACCELERATE_VARS:
        monkeypatch.delenv(var, raising=False)
    trainer = _with_cp_config(trainer_cls)
    if trainer_cls.__name__ in CP_TRAINERS:
        trainer._validate_parallelism_modes()
    else:
        with pytest.raises(ValueError, match=rf"{trainer_cls.__name__} does not support Context Parallelism"):
            trainer._validate_parallelism_modes()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
