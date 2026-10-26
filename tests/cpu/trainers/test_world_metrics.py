#!/usr/bin/env python
"""``WorldMetrics``: world values from rank-local records, folded in one collective per flush.

TRL's ``log`` averages each process's own ``_metrics`` list and only the main process reports, so a fraction
computed over one rank's rows is logged as the batch's: with the tied groups on other ranks,
``sampling/degenerate_group_frac`` reads 0 while the drop fires. Each kind's fold is checked against a numpy
reference over the pooled population, with peers standing in for the other ranks of a fake gather; the group
form runs on real gloo ranks.

    python tests/cpu/trainers/test_world_metrics.py
"""

import datetime
import json
import math

import numpy as np
import pytest
import torch

from src.trainers.world_metrics import WorldMetrics
from tests.common.gloo import new_groups, run_gloo_ranks
from tests.common.world_metrics import fake_world

# Three ranks' samples of one population: (values, weights, mask), rank 1 holding a NaN at zero weight.
_RANKS = [
    ([0.5, -1.25, 3.0], [1.0, 2.0, 1.0], [True, True, False]),
    ([2.0, math.nan, -0.75, 4.5], [3.0, 0.0, 1.0, 2.0], [True, False, True, True]),
    ([1.5], [1.0], [True]),
]


def _flush(*ranks: WorldMetrics, **kwargs) -> list[dict[str, float]]:
    """Every rank's flush of a fake world made of ``ranks``."""
    with fake_world(*ranks):
        return [rank.flush(**kwargs) for rank in ranks]


def _pooled(column: int, *, masked: bool) -> np.ndarray:
    """One column of every rank's samples, concatenated; the masked samples alone when ``masked``."""
    return np.concatenate(
        [np.asarray(rank[column], dtype=np.float64)[np.asarray(rank[2]) if masked else slice(None)] for rank in _RANKS]
    )


def _recorded(record) -> list[WorldMetrics]:
    ranks = [WorldMetrics() for _ in _RANKS]
    for rank, (values, weights, mask) in zip(ranks, _RANKS, strict=True):
        record(rank, torch.tensor(values, dtype=torch.float64), torch.tensor(weights), torch.tensor(mask))
    return ranks


def test_a_fraction_folds_the_counts_not_the_per_rank_means():
    """Rank 0: 0 of 24 trajectories dropped; rank 1: 8 of 8. The batch dropped 8 of 32 — the number a
    mean of per-rank fractions (0.5) and a main-process-only log (0.0) both miss."""
    rank0, rank1 = WorldMetrics(), WorldMetrics()
    rank0.fraction("sampling/degenerate_group_frac", torch.tensor(0), torch.tensor(24))
    rank1.fraction("sampling/degenerate_group_frac", 8, 8)
    assert _flush(rank0, rank1)[0] == {"sampling/degenerate_group_frac": pytest.approx(0.25)}


def test_a_weighted_mean_matches_numpy_and_a_record_of_no_weight_adds_nothing_even_nan():
    def record(rank, values, weights, _mask):
        for value, weight in zip(values, weights, strict=True):
            rank.mean("m", value, weight)

    values, weights = _pooled(0, masked=False), _pooled(1, masked=False)
    kept = weights > 0
    expected = np.average(values[kept], weights=weights[kept])
    for logged in _flush(*_recorded(record)):
        assert logged["m"] == pytest.approx(expected, rel=1e-12)


def test_a_mean_of_python_numbers_and_one_of_no_world_weight():
    rank0, rank1 = WorldMetrics(), WorldMetrics()
    rank0.mean("m", 2.0, 3)
    rank0.mean("m", math.nan, 0)
    rank1.mean("m", 6.0)
    rank0.mean("padding_only", math.nan, 0)
    rank1.mean("padding_only", torch.tensor(math.nan), torch.tensor(0))
    logged = _flush(rank0, rank1)[1]
    assert logged == {"m": pytest.approx((2.0 * 3 + 6.0) / 4)}, "a key of no world weight is not reported"


def test_a_sum_adds_every_record_of_every_rank():
    def record(rank, values, _weights, mask):
        for value in values[mask]:
            rank.sum("s", value)

    assert _flush(*_recorded(record))[2]["s"] == pytest.approx(_pooled(0, masked=True).sum(), rel=1e-12)


def test_a_maximum_is_the_world_maximum_over_every_record():
    def record(rank, values, _weights, mask):
        for value in values[mask]:
            rank.maximum("x", value)

    assert _flush(*_recorded(record))[0]["x"] == _pooled(0, masked=True).max()


def test_moments_match_numpy_over_the_pooled_masked_values():
    """Two records on a rank and three ranks: the five keys are those of the whole masked population."""

    def record(rank, values, _weights, mask):
        even = torch.arange(len(values)) % 2 == 0
        rank.moments("positive/logps", values, mask & even)
        rank.moments("positive/logps", values, mask & ~even)

    pooled = _pooled(0, masked=True)
    expected = {
        "positive/logps_mean": pooled.mean(),
        "positive/logps_std": pooled.std(),
        "positive/logps_min": pooled.min(),
        "positive/logps_max": pooled.max(),
        "positive/logps_range": pooled.max() - pooled.min(),
    }
    for logged in _flush(*_recorded(record)):
        assert logged == pytest.approx(expected, rel=1e-12)


def test_moments_of_one_sample_have_no_spread_and_of_none_are_not_reported():
    one, none = WorldMetrics(), WorldMetrics()
    one.moments("k", torch.tensor([0.1, 7.0]), torch.tensor([True, False]))
    none.moments("k", torch.tensor([3.0]), torch.tensor([False]))
    none.moments("empty", torch.tensor([3.0]), torch.tensor([False]))
    logged = _flush(one, none)[0]
    assert logged["k_std"] == 0.0 and logged["k_mean"] == pytest.approx(0.1) and logged["k_range"] == 0.0
    assert not any(key.startswith("empty") for key in logged)


@pytest.mark.parametrize("q", [50, 90])
def test_a_percentile_is_the_nearest_rank_of_the_pooled_values(q):
    def record(rank, values, _weights, mask):
        for value in values[mask]:
            rank.percentile("p", value, q)

    assert _flush(*_recorded(record))[1]["p"] == np.percentile(_pooled(0, masked=True), q, method="inverted_cdf")


def test_the_effective_sample_frac_and_covariance_fold_the_masked_samples():
    def record(rank, values, weights, mask):
        rank.effective_sample_frac("ess", weights, mask)
        rank.covariance("cov", values, weights, mask)

    weights, values = _pooled(1, masked=True), _pooled(0, masked=True)
    ess = weights.sum() ** 2 / (len(weights) * (weights**2).sum())
    cov = np.cov(values, weights, bias=True)[0, 1]
    logged = _flush(*_recorded(record))[0]
    assert logged["ess"] == pytest.approx(ess, rel=1e-12)
    assert logged["cov"] == pytest.approx(cov, rel=1e-12)


def test_a_key_only_a_peer_recorded_is_still_logged():
    """A site behind a data-dependent gate is reached by some ranks only; their counts still land."""
    peer = WorldMetrics()
    peer.fraction("sampling/is_ratio_mean", 3.0, 2)
    assert _flush(WorldMetrics(), peer)[0] == {"sampling/is_ratio_mean": 1.5}


def test_a_zero_world_denominator_reads_zero_and_keys_come_out_sorted():
    rank0, rank1 = WorldMetrics(), WorldMetrics()
    rank0.fraction("b/second", 0, 0)
    rank0.fraction("a/first", 1, 4)
    rank1.fraction("b/second", 0, 0)
    logged = _flush(rank0, rank1)[0]
    assert list(logged) == ["a/first", "b/second"] and logged["b/second"] == 0.0


def test_a_flush_appends_to_the_target_and_consumes_the_records():
    world = WorldMetrics()
    world.fraction("k", 1, 4)
    world.fraction("k", 1, 4)
    target: dict[str, list[float]] = {"k": [0.75]}
    assert world.flush(target) == {"k": 0.25} and target == {"k": [0.75, 0.25]}
    assert not world and world.flush() == {}, "nothing pending, nothing logged — a stale count must not repeat"


def test_a_key_recorded_as_two_kinds_is_refused_on_a_rank_and_across_ranks():
    world, peer = WorldMetrics(), WorldMetrics()
    world.fraction("k", 1, 2)
    with pytest.raises(ValueError, match="recorded as fraction and as max"):
        world.maximum("k", 1.0)
    peer.maximum("k", 1.0)
    with pytest.raises(ValueError, match="different ranks"):
        _flush(world, peer)


def test_a_key_mismatch_raises_on_every_rank_naming_each_ranks_keys():
    ranks = [WorldMetrics() for _ in range(3)]
    for index, rank in enumerate(ranks):
        rank.mean("shared", 1.0)
        if index == 1:
            rank.mean("only_on_rank_1", 2.0, 0)
    with fake_world(*ranks):
        for rank in ranks:
            with pytest.raises(LookupError) as raised:
                rank.flush(key_mismatch_error=LookupError)
            assert raised.value.args == ({0: ["shared"], 1: ["only_on_rank_1", "shared"], 2: ["shared"]},)


def test_a_key_mismatch_check_lets_a_rank_with_no_records_abstain():
    ranks = [WorldMetrics(), WorldMetrics(), WorldMetrics()]
    ranks[1].mean("m", 3.0, 2)
    ranks[2].mean("m", 6.0, 1)
    for logged in _flush(*ranks, key_mismatch_error=LookupError):
        assert logged == {"m": pytest.approx(4.0)}


def _group_worker(rank: int, out: str) -> None:
    """Two groups of two: each folds its own ranks' records and never sees the other's."""
    group = new_groups(rank, [[0, 1], [2, 3]], datetime.timedelta(seconds=60))
    world = WorldMetrics()
    world.moments("v", torch.tensor([float(rank), 10.0 * rank]), torch.tensor([True, True]))
    with open(f"{out}.{rank}", "w") as fh:
        json.dump(world.flush(group=group), fh)


def test_a_group_flush_folds_that_groups_ranks_alone(tmp_path):
    out = str(tmp_path / "logs")
    run_gloo_ranks(_group_worker, 4, out, pg_timeout=datetime.timedelta(seconds=60))
    for rank in range(4):
        with open(f"{out}.{rank}") as fh:
            logged = json.load(fh)
        peers = [0.0, 1.0, 0.0, 10.0] if rank < 2 else [2.0, 3.0, 20.0, 30.0]
        assert logged["v_mean"] == pytest.approx(np.mean(peers)), f"rank {rank}"
        assert logged["v_max"] == max(peers), f"rank {rank}"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
