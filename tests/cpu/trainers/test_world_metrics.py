#!/usr/bin/env python
"""``WorldMetrics``: world values from rank-local records, folded in fixed-width all-reduces.

TRL's ``log`` averages each process's own ``_metrics`` list and only the main process reports, so a fraction
computed over one rank's rows is logged as the batch's: with the tied groups on other ranks,
``sampling/degenerate_group_frac`` reads 0 while the drop fires. Real gloo ranks flush every kind and each
fold is checked against a numpy reference over the pooled population. The non-percentile path must not reach an
object collective: ``all_gather_object`` costs ~3W device-to-host syncs and W unpickles per flush on NCCL.

    python tests/cpu/trainers/test_world_metrics.py
"""

import datetime
import json
import math
import pickle
from collections.abc import Callable, Sequence
from contextlib import ExitStack
from unittest.mock import patch

import numpy as np
import pytest
import torch
import torch.distributed as dist

from halo.trainers import world_metrics
from halo.trainers.world_metrics import WorldMetrics
from tests.common.gloo import new_groups, run_gloo_ranks
from tests.common.world_metrics import fake_world

# Three ranks' samples of one population: (values, weights, mask), rank 1 holding a NaN at zero weight.
_RANKS = [
    ([0.5, -1.25, 3.0], [1.0, 2.0, 1.0], [True, True, False]),
    ([2.0, math.nan, -0.75, 4.5], [3.0, 0.0, 1.0, 2.0], [True, False, True, True]),
    ([1.5], [1.0], [True]),
]
_ALL_RANKS = tuple(range(len(_RANKS)))
_PG_TIMEOUT = datetime.timedelta(seconds=60)
# The collectives a flush may issue, each counted per scenario.
_COLLECTIVES = ("all_reduce", "all_gather_object", "broadcast_object_list")
_OBJECT_COLLECTIVES = ("all_gather_object", "broadcast_object_list")


def _samples(rank: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    values, weights, mask = _RANKS[rank]
    return torch.tensor(values, dtype=torch.float64), torch.tensor(weights), torch.tensor(mask)


def _pooled(column: int, *, masked: bool, ranks: Sequence[int] = _ALL_RANKS) -> np.ndarray:
    """One column of ``ranks``' samples, concatenated; the masked samples alone when ``masked``."""
    return np.concatenate(
        [
            np.asarray(_RANKS[rank][column], dtype=np.float64)[np.asarray(_RANKS[rank][2]) if masked else slice(None)]
            for rank in ranks
        ]
    )


def _moments(key: str, values: np.ndarray) -> dict[str, float]:
    return {
        f"{key}_mean": values.mean(),
        f"{key}_std": values.std(),
        f"{key}_min": values.min(),
        f"{key}_max": values.max(),
        f"{key}_range": values.max() - values.min(),
    }


def _record_every_kind(world: WorldMetrics, rank: int) -> None:
    """Every kind but ``percentile`` from this rank's samples, plus the edge cases: a key of no world weight, moments
    only rank 0 has a sample of and moments nobody has one of, a zero world denominator, and a value an fp32 fold
    would round."""
    values, weights, mask = _samples(rank)
    world.fraction("frac", mask.sum(), mask.numel())
    for value, weight in zip(values, weights, strict=True):
        world.mean("mean", value, weight)
    for value in values[mask]:
        world.sum("sum", value)
        world.maximum("max", value)
    world.effective_sample_frac("ess", weights, mask)
    world.covariance("cov", values, weights, mask)
    even = torch.arange(len(values)) % 2 == 0
    world.moments("moments", values, mask & even)
    world.moments("moments", values, mask & ~even)
    world.moments("single", values, (torch.arange(len(values)) == 0) & (rank == 0))
    world.moments("empty", values, torch.zeros_like(mask))
    world.mean("padding_only", math.nan, 0)
    world.fraction("zero", 0, 0)
    world.mean("precise", 1.0 + 1e-10, 3)


def _every_kind_reference() -> dict[str, float]:
    """What :func:`_record_every_kind` must log, in the flush's key order, from numpy over the pooled samples."""
    values, weights = _pooled(0, masked=False), _pooled(1, masked=False)
    masked_values, masked_weights = _pooled(0, masked=True), _pooled(1, masked=True)
    kept = weights > 0
    return {
        "cov": np.cov(masked_values, masked_weights, bias=True)[0, 1],
        "ess": masked_weights.sum() ** 2 / (len(masked_weights) * (masked_weights**2).sum()),
        "frac": len(masked_values) / len(values),
        "max": masked_values.max(),
        "mean": np.average(values[kept], weights=weights[kept]),
        **_moments("moments", masked_values),
        "precise": 1.0 + 1e-10,
        **_moments("single", np.asarray(_RANKS[0][0][:1])),
        "sum": masked_values.sum(),
        "zero": 0.0,
    }


def _record_percentiles(world: WorldMetrics, rank: int) -> None:
    values, _, mask = _samples(rank)
    for value in values[mask]:
        world.percentile("p50", value, 50)
        world.percentile("p90", value, 90)
    world.fraction("frac", mask.sum(), mask.numel())


def _record_on_ranks_1_and_2(world: WorldMetrics, rank: int) -> None:
    """Rank 0 ran no batch: it records nothing and must learn the keys from rank 1."""
    if rank == 0:
        return
    values, _, mask = _samples(rank)
    world.mean("m", 3.0 * rank, 3 - rank)
    world.moments("v", values, mask)


def _record_a_gated_key(world: WorldMetrics, rank: int) -> None:
    """Only rank 1 reaches the data-dependent site, with a negative maximum a zero-filled peer would hide."""
    world.fraction("shared", rank, 2)
    if rank == 1:
        world.fraction("gated", 3, 4)
        world.maximum("gated_max", -5.0)
        world.percentile("gated_p50", 7.0, 50)


def _record_a_key_on_rank_1_alone(world: WorldMetrics, rank: int) -> None:
    world.mean("shared", 1.0)
    if rank == 1:
        world.mean("only_on_rank_1", 2.0, 0)


def _record_two_kinds_of_one_key(world: WorldMetrics, rank: int) -> None:
    if rank == 0:
        world.fraction("k", 1, 2)
    else:
        world.maximum("k", 1.0)


def _record_nothing(world: WorldMetrics, rank: int) -> None:
    pass


def _record_a_percentile_alone(world: WorldMetrics, rank: int) -> None:
    world.percentile("p", float(rank), 50)


# Per scenario: what each rank records, the flush's key_mismatch_error, and the object collectives it must not
# issue (each raises if called, so a transport that gathers objects there fails instead of passing).
_SCENARIOS: dict[str, tuple[Callable[[WorldMetrics, int], None], type[Exception] | None, tuple[str, ...]]] = {
    "every_kind": (_record_every_kind, None, _OBJECT_COLLECTIVES),
    "percentiles": (_record_percentiles, None, ("broadcast_object_list",)),
    "abstaining_rank": (_record_on_ranks_1_and_2, LookupError, ("all_gather_object",)),
    "gated_key": (_record_a_gated_key, None, ("broadcast_object_list",)),
    "key_mismatch": (_record_a_key_on_rank_1_alone, LookupError, ("broadcast_object_list",)),
    "kind_conflict": (_record_two_kinds_of_one_key, None, ("broadcast_object_list",)),
    "no_records": (_record_nothing, LookupError, _OBJECT_COLLECTIVES),
}


def _flush_outcome(world: WorldMetrics, key_mismatch_error: type[Exception] | None) -> dict:
    """The flush's logged metrics, or the error it raised as ``(type name, args)``."""
    try:
        return {"logged": world.flush(key_mismatch_error=key_mismatch_error)}
    except Exception as exc:  # the outcome under test: every rank must raise the same error together
        return {"raised": (type(exc).__name__, exc.args)}


def _scenario_worker(rank: int, out: str) -> None:
    """Every scenario in order, each rank recording its outcome (the logged metrics or the raised error) and how
    many times it called each collective."""
    outcomes = {}
    for name, (record, key_mismatch_error, forbidden) in _SCENARIOS.items():
        world = WorldMetrics()
        record(world, rank)
        with ExitStack() as stack:
            calls = {
                collective: stack.enter_context(
                    patch.object(dist, collective, side_effect=AssertionError(f"{collective} ran"))
                    if collective in forbidden
                    else patch.object(dist, collective, wraps=getattr(dist, collective))
                )
                for collective in _COLLECTIVES
            }
            outcome = _flush_outcome(world, key_mismatch_error)
        outcomes[name] = {**outcome, "calls": {collective: mock.call_count for collective, mock in calls.items()}}
    with open(f"{out}.{rank}", "wb") as fh:
        pickle.dump(outcomes, fh)


@pytest.fixture(scope="module")
def outcomes(tmp_path_factory) -> list[dict]:
    """Each rank's outcome of every scenario, from one spawned world of ``len(_RANKS)`` gloo ranks."""
    out = str(tmp_path_factory.mktemp("world_metrics") / "outcome")
    run_gloo_ranks(_scenario_worker, len(_RANKS), out, pg_timeout=_PG_TIMEOUT)
    loaded = []
    for rank in range(len(_RANKS)):
        with open(f"{out}.{rank}", "rb") as fh:
            loaded.append(pickle.load(fh))
    return loaded


def _logged(outcomes: list[dict], scenario: str) -> list[dict[str, float]]:
    """Every rank's logged metrics of ``scenario``, failing on a rank that raised instead."""
    logged = []
    for rank, outcome in enumerate(outcomes):
        assert "raised" not in outcome[scenario], f"rank {rank}: {outcome[scenario]}"
        logged.append(outcome[scenario]["logged"])
    return logged


def test_every_kind_folds_to_its_reference_in_fixed_width_all_reduces(outcomes):
    """The digest, the SUM and the MAX: three all-reduces at any world size, and no object collective."""
    reference = _every_kind_reference()
    for rank, logged in enumerate(_logged(outcomes, "every_kind")):
        assert list(logged) == list(reference), f"rank {rank}: the keys or their order"
        assert logged == pytest.approx(reference, rel=1e-12), f"rank {rank}"
        calls = outcomes[rank]["every_kind"]["calls"]
        assert calls == {"all_reduce": 3, "all_gather_object": 0, "broadcast_object_list": 0}, f"rank {rank}"


def test_a_percentile_is_the_nearest_rank_of_the_pooled_values(outcomes):
    """The one kind that ships raw values pools them in one object gather; the other keys still reduce."""
    pooled = _pooled(0, masked=True)
    reference = {
        "frac": len(pooled) / len(_pooled(0, masked=False)),
        "p50": np.percentile(pooled, 50, method="inverted_cdf"),
        "p90": np.percentile(pooled, 90, method="inverted_cdf"),
    }
    for rank, logged in enumerate(_logged(outcomes, "percentiles")):
        assert logged == pytest.approx(reference, rel=1e-12), f"rank {rank}"
        assert outcomes[rank]["percentiles"]["calls"]["all_gather_object"] == 1, f"rank {rank}"


def test_a_rank_with_no_records_learns_the_keys_and_weighs_nothing(outcomes):
    """Rank 0 abstains from the key check and learns the keys from rank 1 by broadcast, never by a gather."""
    reference = {"m": (3.0 * 2 + 6.0 * 1) / 3, **_moments("v", _pooled(0, masked=True, ranks=(1, 2)))}
    for rank, logged in enumerate(_logged(outcomes, "abstaining_rank")):
        assert logged == pytest.approx(reference, rel=1e-12), f"rank {rank}"
        assert outcomes[rank]["abstaining_rank"]["calls"]["broadcast_object_list"] == 1, f"rank {rank}"


def test_a_key_only_a_peer_recorded_is_still_logged(outcomes):
    """A site behind a data-dependent gate is reached by some ranks only; their counts still land."""
    reference = {"gated": 0.75, "gated_max": -5.0, "gated_p50": 7.0, "shared": (0 + 1 + 2) / 6}
    for rank, logged in enumerate(_logged(outcomes, "gated_key")):
        assert logged == pytest.approx(reference, rel=1e-12), f"rank {rank}"


def test_a_key_mismatch_raises_on_every_rank_naming_each_ranks_keys(outcomes):
    keys_by_rank = {0: ["shared"], 1: ["only_on_rank_1", "shared"], 2: ["shared"]}
    for rank, outcome in enumerate(outcomes):
        assert outcome["key_mismatch"].get("raised") == ("LookupError", (keys_by_rank,)), f"rank {rank}"


def test_a_key_recorded_as_two_kinds_across_ranks_raises_on_every_rank(outcomes):
    message = "k was recorded as ['fraction', 'max'] on different ranks"
    for rank, outcome in enumerate(outcomes):
        assert outcome["kind_conflict"].get("raised") == ("ValueError", (message,)), f"rank {rank}"


def test_a_flush_nobody_recorded_for_is_one_all_reduce_and_logs_nothing(outcomes):
    for rank, logged in enumerate(_logged(outcomes, "no_records")):
        assert logged == {}, f"rank {rank}"
        assert outcomes[rank]["no_records"]["calls"]["all_reduce"] == 1, f"rank {rank}"


@pytest.mark.parametrize("scenario", list(_SCENARIOS))
def test_the_fake_world_logs_or_raises_what_the_gloo_ranks_do(outcomes, scenario):
    """The trainers' metric tests flush through ``fake_world``: it must keep the flush's folds and its key checks,
    or a caller test pushing a key mismatch through it passes where the real ranks raise."""
    record, key_mismatch_error, _ = _SCENARIOS[scenario]
    ranks = [WorldMetrics() for _ in _RANKS]
    for rank, world in enumerate(ranks):
        record(world, rank)
    with fake_world(*ranks):
        faked = [_flush_outcome(world, key_mismatch_error) for world in ranks]
    for rank, fake in enumerate(faked):
        real = outcomes[rank][scenario]
        if "raised" in real:
            assert fake == {"raised": real["raised"]}, f"rank {rank}"
            continue
        assert "logged" in fake, f"rank {rank}: {fake}"
        assert list(fake["logged"]) == list(real["logged"]), f"rank {rank}: the keys or their order"
        assert fake["logged"] == pytest.approx(real["logged"], rel=1e-12), f"rank {rank}"


@pytest.mark.parametrize(
    ("record", "expected"),
    [(_record_nothing, {}), (_record_a_percentile_alone, {"p": 0.0})],
    ids=["nothing", "percentile"],
)
def test_a_flush_with_no_column_to_reduce_builds_no_tensor(record, expected):
    """A flush with nothing to all-reduce (an empty window, as SFT's last log flushes, or percentiles alone) touches
    no device: on NCCL an empty column tensor is still an allocation and a read-back."""
    world = WorldMetrics()
    record(world, 0)
    with patch.object(world_metrics, "collective_device", side_effect=AssertionError("a column tensor was built")):
        assert world.flush() == expected


def test_a_key_recorded_as_two_kinds_is_refused_on_a_rank():
    world = WorldMetrics()
    world.fraction("k", 1, 2)
    with pytest.raises(ValueError, match="recorded as fraction and as max"):
        world.maximum("k", 1.0)


def test_a_flush_appends_to_the_target_and_consumes_the_records():
    world = WorldMetrics()
    world.fraction("k", 1, 4)
    world.fraction("k", 1, 4)
    target: dict[str, list[float]] = {"k": [0.75]}
    assert world.flush(target) == {"k": 0.25} and target == {"k": [0.75, 0.25]}
    assert not world and world.flush() == {}, "nothing pending, nothing logged — a stale count must not repeat"


def _group_worker(rank: int, out: str) -> None:
    """Two groups of two: each folds its own ranks' records and never sees the other's. Rank 2 records nothing
    and learns the keys from rank 3, which is rank 1 of its group."""
    group = new_groups(rank, [[0, 1], [2, 3]], _PG_TIMEOUT)
    world = WorldMetrics()
    if rank != 2:
        world.moments("v", torch.tensor([float(rank), 10.0 * rank]), torch.tensor([True, True]))
        world.percentile("p", float(rank), 100)
    with open(f"{out}.{rank}", "w") as fh:
        json.dump(world.flush(group=group), fh)


def test_a_group_flush_folds_that_groups_ranks_alone(tmp_path):
    out = str(tmp_path / "logs")
    run_gloo_ranks(_group_worker, 4, out, pg_timeout=_PG_TIMEOUT)
    for rank in range(4):
        with open(f"{out}.{rank}") as fh:
            logged = json.load(fh)
        peers = [0.0, 1.0, 0.0, 10.0] if rank < 2 else [3.0, 30.0]
        assert logged["v_mean"] == pytest.approx(np.mean(peers)), f"rank {rank}"
        assert logged["v_min"] == min(peers) and logged["v_max"] == max(peers), f"rank {rank}"
        assert logged["p"] == (1.0 if rank < 2 else 3.0), f"rank {rank}"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
