#!/usr/bin/env python
"""Batch-level metrics from rank-local counts (``WorldMetrics`` and ``gathered_fractions``).

TRL's ``GRPOTrainer.log`` averages each process's own ``_metrics`` list and only the main process
reports, so a fraction computed over one rank's rows is logged as the batch's: with the tied groups on
other ranks, ``sampling/degenerate_group_frac`` reads 0 while the drop fires. The accumulator records
``(numerator, denominator)`` per rank and folds the counts in one collective; the tests drive that
fold with peers standing in for the other ranks of a fake gather.

    python tests/cpu/grpo/test_world_metrics.py
"""

from collections import defaultdict

import pytest
import torch

from src.trainers.grpo.rollout.rollout_metrics import WorldMetrics, gathered_fractions


def _flush(world: WorldMetrics, *peers: WorldMetrics) -> dict[str, list[float]]:
    """Fold ``world`` with ``peers`` as the other ranks: the gather appends their pending entries."""
    target = defaultdict(list)
    world.flush(target, gather_fn=lambda own: [*own, *(peer._materialized() for peer in peers)])
    return target


def _peer(**fractions: tuple[float, float]) -> WorldMetrics:
    peer = WorldMetrics()
    for key, (numerator, denominator) in fractions.items():
        peer.fraction(key, numerator, denominator)
    return peer


def test_a_fraction_folds_the_counts_not_the_per_rank_means():
    """Rank 0: 0 of 24 trajectories dropped; rank 1: 8 of 8. The batch dropped 8 of 32 — the number a
    mean of per-rank fractions (0.5) and a main-process-only log (0.0) both miss."""
    world = WorldMetrics()
    world.fraction("sampling/degenerate_group_frac", torch.tensor(0), torch.tensor(24))
    logged = _flush(world, _peer(**{"sampling/degenerate_group_frac": (8, 8)}))
    assert logged["sampling/degenerate_group_frac"] == [pytest.approx(0.25)]


def test_a_key_only_a_peer_recorded_is_still_logged():
    """A site behind a data-dependent gate is reached by some ranks only; their counts still land."""
    logged = _flush(WorldMetrics(), _peer(**{"sampling/is_ratio_mean": (3.0, 2)}))
    assert logged["sampling/is_ratio_mean"] == [1.5]


def test_a_maximum_is_the_world_maximum():
    world, high, low = WorldMetrics(), WorldMetrics(), WorldMetrics()
    world.maximum("sampling/is_ratio_max", torch.tensor(1.2))
    high.maximum("sampling/is_ratio_max", 2.5)
    low.maximum("sampling/is_ratio_max", 0.4)
    assert _flush(world, high, low)["sampling/is_ratio_max"] == [2.5]


def test_a_zero_world_denominator_reads_zero_and_keys_come_out_sorted():
    world = WorldMetrics()
    world.fraction("b/second", 0, 0)
    world.fraction("a/first", 1, 4)
    logged = _flush(world, _peer(**{"b/second": (0, 0)}))
    assert list(logged) == ["a/first", "b/second"]
    assert logged["b/second"] == [0.0]


def test_a_flush_consumes_the_pending_entries():
    world = WorldMetrics()
    world.fraction("k", 1, 2)
    assert _flush(world)["k"] == [0.5]
    assert _flush(world) == {}, "nothing pending, nothing logged — a stale count must not repeat"


def test_recording_a_key_twice_in_one_step_is_refused():
    world = WorldMetrics()
    world.fraction("k", 1, 2)
    with pytest.raises(ValueError, match="already recorded"):
        world.fraction("k", 1, 2)


def test_a_key_recorded_as_a_fraction_here_and_a_maximum_elsewhere_is_refused():
    world, peer = WorldMetrics(), WorldMetrics()
    world.fraction("k", 1, 2)
    peer.maximum("k", 1.0)
    with pytest.raises(ValueError, match="different ranks"):
        _flush(world, peer)


def test_a_flush_lands_in_a_plain_dict_as_well_as_trl_defaultdict():
    world = WorldMetrics()
    world.fraction("k", 1, 4)
    target: dict[str, list[float]] = {}
    world.flush(target, gather_fn=list)
    assert target == {"k": [0.25]}


def test_gathered_fractions_fold_every_pair_in_one_gather():
    """The direct form for a site every rank reaches (the breaker, the online k3 clamp): one gather of
    all (numerator, denominator) pairs, then each world ratio — 4 of 24 + 6 of 8 → 10 / 32."""
    calls = []

    def gather(local):
        calls.append(local)
        return torch.cat([local, torch.tensor([6.0, 8.0, 0.0, 0.0], dtype=local.dtype)])

    fractions = gathered_fractions([(torch.tensor(4), 24), (torch.tensor(0), torch.tensor(0))], gather)
    assert fractions == [pytest.approx(10 / 32), 0.0]
    assert len(calls) == 1 and calls[0].dtype == torch.float64, "one collective, exact past 2^24 counts"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
