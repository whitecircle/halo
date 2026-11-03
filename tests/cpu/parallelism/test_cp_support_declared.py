#!/usr/bin/env python
"""CP is declare-to-enable per trainer, and the declaration is the whole gate: nothing inspects a loss for CP-safety. A
flag on an unverified objective trains silently wrong, so the CP set is pinned."""

import pytest

from halo.trainers.distributed_trainer import DistributedTrainerMixin
from tests.common.rosters import distributed_trainer_classes

# Verified under CP (agent-docs/reference/trainer-architecture.md#trainer-compatibility).
CP_TRAINERS = {"DistributedSFTTrainer", "SMPOTrainer", "OfflineGRPOTrainer"}

# A new trainer joins by existing, at the default off.
TRAINERS = distributed_trainer_classes()


def test_the_derived_roster_covers_every_trainer():
    """Anti-vacuity: a roster finding no trainers would make the checks below assert nothing."""
    names = {cls.__name__ for cls in TRAINERS}
    assert len(names) >= 13, f"the roster collapsed to {sorted(names)}"
    assert names >= CP_TRAINERS, f"CP trainers missing from the roster: {sorted(CP_TRAINERS - names)}"


def test_the_mixin_default_is_off():
    """Declare-to-enable."""
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
