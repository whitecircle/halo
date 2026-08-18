#!/usr/bin/env python
"""``warn_if_flce_unreachable`` — the GRPO-family guard against a silently inert Liger FLCE.

GRPO objectives compute their loss outside the model's forward and never pass ``labels``, so an
applied ``fused_linear_cross_entropy`` (explicit, or a per-model default like Zaya's) never fires:
numerically neutral, but the logits plane the user enabled it to remove still materializes. The
warning at trainer construction is the only surfacing of that — these tests fail if it goes silent,
fires spuriously, or the GRPO trainers stop declaring the flag that routes to it.

    python tests/cpu/kernels/test_liger_flce_unreachable_warning.py
"""

import logging
import sys
from types import SimpleNamespace

import pytest
from accelerate import PartialState

from src.kernels.liger.orchestrator import warn_if_flce_unreachable

# The orchestrator logs through accelerate's rank-aware logger, which requires an initialized state.
PartialState()

_LOGGER_NAME = "src.kernels.liger.orchestrator"


def _config(applied):
    config = SimpleNamespace()
    if applied is not None:
        config._halo_liger_applied_config = applied
    return config


def test_warns_when_flce_applied(caplog):
    with caplog.at_level(logging.WARNING, logger=_LOGGER_NAME):
        warn_if_flce_unreachable(_config({"fused_linear_cross_entropy": True}), "OfflineGRPOTrainer")
    assert any("unreachable on OfflineGRPOTrainer" in r.message for r in caplog.records)
    assert any("use_chunked_grpo_logprobs" in r.message for r in caplog.records)


@pytest.mark.parametrize(
    "applied", [None, {}, {"fused_linear_cross_entropy": False}, {"cross_entropy": True}], ids=str
)
def test_silent_when_flce_not_applied(caplog, applied):
    # CE alone must NOT warn: it is the toolkit default and equally inert here — warning on every
    # GRPO run would train users to ignore the one that matters.
    with caplog.at_level(logging.WARNING, logger=_LOGGER_NAME):
        warn_if_flce_unreachable(_config(applied), "DistributedGRPOTrainer")
    assert not caplog.records


def test_grpo_family_declares_the_flag():
    # The warning only reaches users through _setup_distributed_modes' gate on this flag; a trainer
    # silently dropping it re-opens the inert-FLCE trap.
    from src.trainers.grpo.environmental import DistributedAsyncEnvironmentalGRPOTrainer
    from src.trainers.grpo.offline import OfflineGRPOTrainer
    from src.trainers.grpo.online import DistributedGRPOTrainer
    from src.trainers.mixins.base import DistributedTrainerMixin

    assert DistributedTrainerMixin._loss_outside_model_forward is False  # SFT-shaped trainers keep FLCE
    for trainer in (OfflineGRPOTrainer, DistributedGRPOTrainer, DistributedAsyncEnvironmentalGRPOTrainer):
        assert trainer._loss_outside_model_forward is True, trainer.__name__


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
