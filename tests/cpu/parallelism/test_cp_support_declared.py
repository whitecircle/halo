#!/usr/bin/env python
"""Context Parallelism is declare-to-enable per trainer, and the declaration is the whole gate.

Nothing inspects a trainer's loss for CP-safety: ``reject_undeclared_axes`` reads only the
``SUPPORTS_CP`` class attribute (off on ``DistributedTrainerMixin``) before a ``cp_size > 1`` run
starts. A flag switched on for a trainer whose objective has not been verified under sequence
chunking trains silently wrong (a sequence log-prob pooled over one chunk, say), so the CP set is
pinned here, and each trainer's own gate is driven with a CP size: every other trainer must refuse
it by name, the CP trainers must accept it.

Run: python tests/cpu/parallelism/test_cp_support_declared.py
"""

import pytest

from halo.trainers.distributed_trainer import DistributedTrainerMixin
from tests.common.rosters import distributed_trainer_classes

# The trainers whose objective is verified under CP (agent-docs/reference/trainer-architecture.md#trainer-compatibility).
CP_TRAINERS = {"DistributedSFTTrainer", "SmoothMarginPOTrainer", "OfflineGRPOTrainer"}

# Derived from the class hierarchy, not listed: a new trainer joins by existing, at the default off.
TRAINERS = distributed_trainer_classes()


def test_the_derived_roster_covers_every_trainer():
    """Anti-vacuity: a roster that stopped finding trainers would make the checks below assert nothing."""
    names = {cls.__name__ for cls in TRAINERS}
    assert len(names) >= 13, f"the roster collapsed to {sorted(names)}"
    assert names >= CP_TRAINERS, f"CP trainers missing from the roster: {sorted(CP_TRAINERS - names)}"


def test_the_mixin_default_is_off():
    """Declare-to-enable: a trainer that never considered CP must not inherit it."""
    assert DistributedTrainerMixin.SUPPORTS_CP is False


@pytest.mark.parametrize("trainer_cls", TRAINERS, ids=lambda cls: cls.__name__)
def test_cp_support_is_exactly_the_verified_set(trainer_cls):
    expected = trainer_cls.__name__ in CP_TRAINERS
    assert trainer_cls.SUPPORTS_CP is expected, (
        f"{trainer_cls.__name__}.SUPPORTS_CP is {trainer_cls.SUPPORTS_CP!r}, expected {expected}: enabling CP "
        f"needs the objective verified under sequence chunking and this pin updated with it"
    )


@pytest.mark.parametrize("trainer_cls", TRAINERS, ids=lambda cls: cls.__name__)
def test_the_gate_enforces_the_declaration(trainer_cls):
    if trainer_cls.__name__ in CP_TRAINERS:
        trainer_cls.reject_undeclared_axes(cp_size=2, pp_size=1)
    else:
        with pytest.raises(ValueError, match=rf"{trainer_cls.__name__} does not support Context Parallelism"):
            trainer_cls.reject_undeclared_axes(cp_size=2, pp_size=1)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
