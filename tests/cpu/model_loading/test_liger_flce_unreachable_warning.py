#!/usr/bin/env python
"""``warn_if_flce_unreachable``: GRPO never passes ``labels``, so FLCE never fires while the logits plane it should
remove still materializes."""

import logging
from types import SimpleNamespace

import pytest

from halo.model_loading.liger.dispatch import warn_if_flce_unreachable
from halo.trainers.distributed_trainer import DistributedTrainerMixin
from halo.trainers.grpo.environmental.trainer import DistributedAsyncEnvironmentalGRPOTrainer
from halo.trainers.grpo.offline.trainer import OfflineGRPOTrainer
from halo.trainers.grpo.online import DistributedOnlineGRPOTrainer

_LOGGER_NAME = "halo.model_loading.liger.dispatch"


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
    # CE is the toolkit default and equally inert here; warning on every GRPO run trains users to ignore the one that
    # matters.
    with caplog.at_level(logging.WARNING, logger=_LOGGER_NAME):
        warn_if_flce_unreachable(_config(applied), "DistributedOnlineGRPOTrainer")
    assert not caplog.records


def test_grpo_family_declares_the_flag():
    # The warning reaches users only through _configure_distributed_modes' gate on this flag.
    assert DistributedTrainerMixin.POLICY_GRADIENT_LOSS is False  # SFT-shaped trainers keep FLCE.
    for trainer in (OfflineGRPOTrainer, DistributedOnlineGRPOTrainer, DistributedAsyncEnvironmentalGRPOTrainer):
        assert trainer.POLICY_GRADIENT_LOSS is True, trainer.__name__


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
