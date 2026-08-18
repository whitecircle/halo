#!/usr/bin/env python
"""
CPU contract test for the KTO trainer (TRL 1.6 integration).

Guards the wiring that GPU tests can't cheaply cover:
  - the distributed subclass really composes DistributedTrainerMixin with the
    correct TRL base, and declares the documented parallelism support flags;
  - the script-args dataclass constructs with sane defaults.

Run: python tests/cpu/trainers/test_kto_trainer.py
"""

import pytest


def test_kto_trainer_mro_and_flags():
    from trl import KTOTrainer

    from src.trainers.mixins.base import DistributedTrainerMixin
    from src.trainers.preference.kto import DistributedKTOTrainer

    assert issubclass(DistributedKTOTrainer, DistributedTrainerMixin)
    assert issubclass(DistributedKTOTrainer, KTOTrainer)
    # KTO mirrors DPO: EP/TP yes, CP no (full-sequence log-prob pooling + KL ref),
    # PP yes (apo_zero_unpaired only, precompute-only — gated in _validate_pp_mode).
    assert DistributedKTOTrainer._supports_ep is True
    assert DistributedKTOTrainer._supports_tp is True
    assert DistributedKTOTrainer._supports_cp is False
    assert DistributedKTOTrainer._supports_pp is True
    assert "kto" in DistributedKTOTrainer._tag_names


def test_kto_args_defaults():
    from src.args.kto_args import KTOScriptArguments

    args = KTOScriptArguments()
    assert args.completion_field == "completion"
    assert args.label_field == "label"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
