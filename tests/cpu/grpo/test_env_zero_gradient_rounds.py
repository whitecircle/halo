#!/usr/bin/env python
"""A run whose valid episodes all tie drops every group from the loss and logs loss 0 for as long as it runs, since the
empty-step halt reads validity only. A tool parser that leaves calls as text and an answerless environment both land
there, so a round with no token carrying a policy gradient warns and a run of them raises, until a round has trained: a
policy that nearly always solves, or nearly never does, ties for long stretches and still trains."""

import logging

import pytest
import torch

from halo.configs.grpo_environmental import AsyncRolloutConfig
from halo.trainers.grpo.environmental.step_batch import ZERO_GRADIENT_ROUND_KEY
from tests.common.env_grpo_batch import batch_host, build, episode, row


def _host(limit: int | None = 3, *, training: bool = True):
    host = batch_host([row([6]), row([7])], torch.full((2, 1), -1.0), training=training)
    host.async_config.zero_gradient_round_limit = limit
    return host


_TIED = (episode(1.0), episode(1.0))
_CONTRAST = (episode(1.0), episode(0.0))


def _logged(host) -> list[float]:
    return host._metrics["train"][ZERO_GRADIENT_ROUND_KEY]


def test_tied_rounds_warn_once_then_raise_at_the_limit(caplog):
    host = _host()
    with caplog.at_level(logging.WARNING):
        for _ in range(2):
            build(host, list(_TIED))
    assert _logged(host) == [1.0, 1.0]
    warned = [r.getMessage() for r in caplog.records if "No token carried a policy gradient" in r.getMessage()]
    assert len(warned) == 1 and "sampling/degenerate_group_frac=1.00" in warned[0]
    with pytest.raises(RuntimeError, match="zero_gradient_round_limit=3"):
        build(host, list(_TIED))


def test_a_round_with_a_contrast_resets_the_count():
    host = _host()
    for _ in range(2):
        build(host, list(_TIED))
    build(host, list(_CONTRAST))
    assert _logged(host) == [1.0, 1.0, 0.0]
    for _ in range(2):
        build(host, list(_TIED))
    assert _logged(host)[-1] == 1.0


def test_a_run_that_has_trained_is_never_stopped_by_a_streak(caplog):
    """DP 1 at 8 generations and a 99% solve rate ties 92% of its rounds: a streak of 50 is certain over a long run, and
    a raise would throw away everything since the last checkpoint."""
    host = _host()
    build(host, list(_CONTRAST))
    with caplog.at_level(logging.WARNING):
        for _ in range(8):
            build(host, list(_TIED))
    assert _logged(host) == [0.0] + [1.0] * 8
    warned = [r.getMessage() for r in caplog.records if "No token carried a policy gradient" in r.getMessage()]
    assert len(warned) == 1 and "in 3 train round(s)" in warned[0]


def test_a_null_limit_only_logs():
    host = _host(None)
    for _ in range(5):
        build(host, list(_TIED))
    assert _logged(host) == [1.0] * 5


def test_the_halt_names_the_knob_that_lifts_it():
    host = _host(1)
    with pytest.raises(RuntimeError, match="Raise it for a reward this sparse, or set it to null"):
        build(host, list(_TIED))


def test_an_eval_round_neither_counts_nor_reads_the_world():
    """Eval groups are not dropped, and the extra collective would have to run on every rank's eval too."""
    host = _host(1, training=False)
    build(host, list(_TIED))
    assert _logged(host) == [] and host._zero_gradient_rounds == 0


@pytest.mark.parametrize("limit", [0, -1])
def test_a_limit_that_could_never_hold_is_refused(limit):
    with pytest.raises(ValueError, match="zero_gradient_round_limit"):
        AsyncRolloutConfig(zero_gradient_round_limit=limit)


def test_the_ratio_masks_count_as_no_gradient():
    """IS bands that mask every token train nothing, whatever the advantages say."""
    host = _host()
    advantages = torch.tensor([1.0, -1.0])
    loss_mask = torch.ones(2, 1, dtype=torch.bool)
    host._track_zero_gradient_round(advantages, torch.zeros(2, 1), loss_mask, [])
    assert _logged(host) == [1.0]
    host._track_zero_gradient_round(advantages, torch.ones(2, 1), loss_mask, [])
    assert _logged(host) == [1.0, 0.0]


def test_a_resumed_streak_spends_the_budget_the_run_had_left():
    """Persisted with the checkpoint, so two rounds before the restart leave one round of the three after it."""
    first = _host()
    for _ in range(2):
        build(first, list(_TIED))
    resumed = _host()
    resumed._restore_halting_counters(first._halting_counters())
    with pytest.raises(RuntimeError, match="zero_gradient_round_limit=3"):
        build(resumed, list(_TIED))
    fresh = _host()
    build(fresh, list(_TIED))


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
