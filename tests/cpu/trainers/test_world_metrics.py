#!/usr/bin/env python
"""``WorldMetrics``: world values from rank-local records, folded in fixed-width all-reduces.

TRL's ``log`` averages each process's own ``_metrics`` list and only the main process reports, so a fraction
computed over one rank's rows is logged as the batch's: with the tied groups on other ranks,
``sampling/degenerate_group_frac`` reads 0 while the drop fires. Real gloo ranks flush every kind twice and each
fold is checked against a numpy reference over the pooled population. Once the ranks agreed on their keys, the
non-percentile path must not reach an object collective: ``all_gather_object`` costs ~3W device-to-host syncs and
W unpickles per flush on NCCL.

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
from tests.common.world_metrics import fake_world, pooled_states, rank_states

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


# Per scenario: what each rank records, the flush's key_mismatch_error, and the object collectives its second
# flush, once the keys were agreed, must not issue (each raises if called, so a transport that gathers objects
# there fails instead of passing). No flush broadcasts an object.
_SCENARIOS: dict[str, tuple[Callable[[WorldMetrics, int], None], type[Exception] | None, tuple[str, ...]]] = {
    "every_kind": (_record_every_kind, None, _OBJECT_COLLECTIVES),
    "percentiles": (_record_percentiles, None, ("broadcast_object_list",)),
    "abstaining_rank": (_record_on_ranks_1_and_2, LookupError, _OBJECT_COLLECTIVES),
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


def _counted_flush(world: WorldMetrics, key_mismatch_error, forbidden: tuple[str, ...]) -> dict:
    """:func:`_flush_outcome` with how many times the flush called each collective, ``forbidden`` ones raising."""
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
    return {**outcome, "calls": {collective: mock.call_count for collective, mock in calls.items()}}


def _scenario_worker(rank: int, out: str) -> None:
    """Every scenario in order, recorded and flushed twice on one accumulator: each rank keeps both flushes'
    outcomes (the logged metrics or the raised error) and how many times each called every collective."""
    outcomes = {}
    for name, (record, key_mismatch_error, forbidden) in _SCENARIOS.items():
        world = WorldMetrics()
        record(world, rank)
        first = _counted_flush(world, key_mismatch_error, ("broadcast_object_list",))
        record(world, rank)
        outcomes[name] = {"first": first, "steady": _counted_flush(world, key_mismatch_error, forbidden)}
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
    """Every rank's logged metrics of ``scenario``, both flushes alike, failing on a rank that raised instead."""
    logged = []
    for rank, outcome in enumerate(outcomes):
        first, steady = outcome[scenario]["first"], outcome[scenario]["steady"]
        assert "raised" not in first and "raised" not in steady, f"rank {rank}: {outcome[scenario]}"
        assert list(first["logged"]) == list(steady["logged"]), f"rank {rank}: the flushes' keys or their order"
        assert first["logged"] == pytest.approx(steady["logged"], rel=1e-12), f"rank {rank}"
        logged.append(steady["logged"])
    return logged


def _calls(outcomes: list[dict], scenario: str, rank: int) -> tuple[dict, dict]:
    return outcomes[rank][scenario]["first"]["calls"], outcomes[rank][scenario]["steady"]["calls"]


def test_every_kind_folds_to_its_reference_in_fixed_width_all_reduces(outcomes):
    """The first flush agrees the keys in one object gather; every later one is one SUM and one MAX all-reduce at
    any world size, and no object collective."""
    reference = _every_kind_reference()
    for rank, logged in enumerate(_logged(outcomes, "every_kind")):
        assert list(logged) == list(reference), f"rank {rank}: the keys or their order"
        assert logged == pytest.approx(reference, rel=1e-12), f"rank {rank}"
        first, steady = _calls(outcomes, "every_kind", rank)
        assert first == {"all_reduce": 3, "all_gather_object": 1, "broadcast_object_list": 0}, f"rank {rank}"
        assert steady == {"all_reduce": 2, "all_gather_object": 0, "broadcast_object_list": 0}, f"rank {rank}"


def test_a_percentile_is_the_nearest_rank_of_the_pooled_values(outcomes):
    """The one kind that ships raw values pools them in one object gather on every flush; the other keys still
    reduce."""
    pooled = _pooled(0, masked=True)
    reference = {
        "frac": len(pooled) / len(_pooled(0, masked=False)),
        "p50": np.percentile(pooled, 50, method="inverted_cdf"),
        "p90": np.percentile(pooled, 90, method="inverted_cdf"),
    }
    for rank, logged in enumerate(_logged(outcomes, "percentiles")):
        assert logged == pytest.approx(reference, rel=1e-12), f"rank {rank}"
        assert _calls(outcomes, "percentiles", rank)[1]["all_gather_object"] == 1, f"rank {rank}"


def test_a_rank_with_no_records_learns_the_keys_and_weighs_nothing(outcomes):
    """Rank 0 abstains from the key check, learns the keys from the first flush's gather and then adds nothing to
    the fixed-width reduce."""
    reference = {"m": (3.0 * 2 + 6.0 * 1) / 3, **_moments("v", _pooled(0, masked=True, ranks=(1, 2)))}
    for rank, logged in enumerate(_logged(outcomes, "abstaining_rank")):
        assert logged == pytest.approx(reference, rel=1e-12), f"rank {rank}"
        assert _calls(outcomes, "abstaining_rank", rank)[1]["all_gather_object"] == 0, f"rank {rank}"


def test_a_key_only_a_peer_recorded_is_still_logged(outcomes):
    """A site behind a data-dependent gate is reached by some ranks only; their counts still land."""
    reference = {"gated": 0.75, "gated_max": -5.0, "gated_p50": 7.0, "shared": (0 + 1 + 2) / 6}
    for rank, logged in enumerate(_logged(outcomes, "gated_key")):
        assert logged == pytest.approx(reference, rel=1e-12), f"rank {rank}"


def test_a_key_mismatch_raises_on_every_rank_naming_each_ranks_keys(outcomes):
    keys_by_rank = {0: ["shared"], 1: ["only_on_rank_1", "shared"], 2: ["shared"]}
    for rank, outcome in enumerate(outcomes):
        for flush in ("first", "steady"):
            raised = outcome["key_mismatch"][flush].get("raised")
            assert raised == ("LookupError", (keys_by_rank,)), f"rank {rank}, {flush} flush"


def test_a_key_recorded_as_two_kinds_across_ranks_raises_on_every_rank(outcomes):
    message = "k was recorded as ['fraction', 'max'] on different ranks"
    for rank, outcome in enumerate(outcomes):
        for flush in ("first", "steady"):
            assert outcome["kind_conflict"][flush].get("raised") == ("ValueError", (message,)), f"rank {rank}"


def test_a_flush_nobody_recorded_for_is_one_all_reduce_and_logs_nothing(outcomes):
    for rank, logged in enumerate(_logged(outcomes, "no_records")):
        assert logged == {}, f"rank {rank}"
        for calls in _calls(outcomes, "no_records", rank):
            assert calls == {"all_reduce": 1, "all_gather_object": 0, "broadcast_object_list": 0}, f"rank {rank}"


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
        real = outcomes[rank][scenario]["first"]
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


def test_world_sums_keeps_the_order_of_mixed_tensors_and_numbers():
    """Numbers travel in one copy beside the tensors, so their places must survive the reassembly."""
    values = [torch.tensor(2), 3, torch.tensor(5.5, dtype=torch.float32), 7.25, torch.tensor(True)]
    assert world_metrics.world_sums(values) == [2.0, 3.0, 5.5, 7.25, 1.0]
    assert world_metrics.world_sums([1, 2]) == [1.0, 2.0]


def test_a_key_recorded_as_two_kinds_is_refused_on_a_rank():
    world = WorldMetrics()
    world.fraction("k", 1, 2)
    with pytest.raises(ValueError, match="recorded as fraction and as max"):
        world.maximum("k", 1.0)


@pytest.mark.parametrize("weight", [int, torch.tensor], ids=["number", "tensor"])
def test_a_tensor_mean_of_no_weight_adds_nothing_even_a_nan(weight):
    """A tensor value weighs by a number or a tensor; at a weight of 0 either way the record adds nothing, where a
    plain product would carry the NaN into the weighted sum."""
    world = WorldMetrics()
    world.mean("m", torch.tensor(math.nan), weight(0))
    world.mean("m", torch.tensor(2.0), weight(1))
    assert world.flush() == {"m": 2.0}


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


# The ranks reaching a data-dependent site in each of four windows on one accumulator: the key appears on rank 1,
# is reached nowhere, then appears on rank 0.
_GATED_WINDOWS = ((), (1,), (), (0,))


def _layout_worker(rank: int, out: str) -> None:
    world = WorldMetrics()
    windows = []
    for gated_ranks in _GATED_WINDOWS:
        world.fraction("shared", rank, 2)
        if rank in gated_ranks:
            world.maximum("gated_max", -5.0 - rank)
        windows.append(_counted_flush(world, None, ()))
    with open(f"{out}.{rank}", "wb") as fh:
        pickle.dump(windows, fh)


def test_a_new_key_is_gathered_once_and_then_rides_the_agreed_layout(tmp_path):
    """A key outside the agreed layout costs one object gather; the layout keeps it afterwards, so its absence
    reports nothing and its return on another rank is again a plain SUM and MAX all-reduce."""
    out = str(tmp_path / "windows")
    run_gloo_ranks(_layout_worker, 2, out, pg_timeout=_PG_TIMEOUT)
    expected = [
        ({"shared": 0.25}, {"all_reduce": 2, "all_gather_object": 1}),
        ({"gated_max": -6.0, "shared": 0.25}, {"all_reduce": 3, "all_gather_object": 1}),
        ({"shared": 0.25}, {"all_reduce": 2, "all_gather_object": 0}),
        ({"gated_max": -5.0, "shared": 0.25}, {"all_reduce": 2, "all_gather_object": 0}),
    ]
    for rank in range(2):
        with open(f"{out}.{rank}", "rb") as fh:
            windows = pickle.load(fh)
        for index, (window, (logged, calls)) in enumerate(zip(windows, expected, strict=True)):
            assert window["logged"] == logged, f"rank {rank}, window {index}"
            assert window["calls"] == {**calls, "broadcast_object_list": 0}, f"rank {rank}, window {index}"


def _dropped_key_worker(rank: int, out: str) -> None:
    """Both ranks record ``late`` in the first window, so the agreed layout holds it; in the second rank 1 drops it."""
    world = WorldMetrics()
    windows = []
    for window in range(2):
        world.mean("shared", 1.0)
        if window == 0 or rank == 0:
            world.mean("late", 2.0)
        windows.append(_flush_outcome(world, LookupError))
    with open(f"{out}.{rank}", "wb") as fh:
        pickle.dump(windows, fh)


def test_a_key_mismatch_inside_the_agreed_layout_raises_on_every_rank(tmp_path):
    """A rank dropping a key the layout holds still fits the layout: the fixed-width reduce must still tell the
    ranks' keys apart and raise, not log a key only some ranks recorded."""
    out = str(tmp_path / "windows")
    run_gloo_ranks(_dropped_key_worker, 2, out, pg_timeout=_PG_TIMEOUT)
    for rank in range(2):
        with open(f"{out}.{rank}", "rb") as fh:
            first, steady = pickle.load(fh)
        assert first == {"logged": {"late": 2.0, "shared": 1.0}}, f"rank {rank}"
        assert steady == {"raised": ("LookupError", ({0: ["late", "shared"], 1: ["shared"]},))}, f"rank {rank}"


def _alternating_groups_worker(rank: int, out: str) -> None:
    """One accumulator flushed over the world and over a subgroup in turn: each keeps its own agreed layout, so a
    rank of only one of them never reduces at the other's width."""
    group = new_groups(rank, [[0, 1], [2, 3]], _PG_TIMEOUT)
    world = WorldMetrics()
    windows = []
    for flush_group in (None, group, None, group):
        world.fraction("f", rank, 4)
        if flush_group is None:
            world.sum("world_only", 1)
        windows.append(_counted_flush(world, None, ()) if flush_group is None else _group_flush(world, group))
    with open(f"{out}.{rank}", "wb") as fh:
        pickle.dump(windows, fh)


def _group_flush(world: WorldMetrics, group) -> dict:
    with patch.object(dist, "all_gather_object", wraps=dist.all_gather_object) as gathers:
        logged = world.flush(group=group)
    return {"logged": logged, "calls": {"all_gather_object": gathers.call_count}}


def test_each_group_keeps_its_own_agreed_layout(tmp_path):
    out = str(tmp_path / "windows")
    run_gloo_ranks(_alternating_groups_worker, 4, out, pg_timeout=_PG_TIMEOUT)
    for rank in range(4):
        with open(f"{out}.{rank}", "rb") as fh:
            windows = pickle.load(fh)
        peers = (0, 1) if rank < 2 else (2, 3)
        for index, window in enumerate(windows):
            if index % 2 == 0:
                assert window["logged"] == {"f": 6 / 16, "world_only": 4.0}, f"rank {rank}, window {index}"
            else:
                assert window["logged"] == {"f": sum(peers) / 8}, f"rank {rank}, window {index}"
            gathers = window["calls"]["all_gather_object"]
            assert gathers == (1 if index < 2 else 0), f"rank {rank}, window {index}: the layout was not reused"


# Values for the exact-fold checks: magnitudes far apart, so a plain left-to-right float sum loses what Neumaier's
# compensation keeps, plus the non-finite and signed-zero cases.
_HARD_VALUES = (1e16, 1.0, -1e16, 3.141592653589793, 1e-8, -2.5e-9, 0.1, 0.2, 0.3, 1e308, -0.0, 0.0, 7.0)
_SPECIAL_VALUES = (math.nan, math.inf, -math.inf)


def _hard_columns(count: int, seed: int, specials: bool) -> list[list[float]]:
    rng = np.random.default_rng(seed)
    pool = _HARD_VALUES + (_SPECIAL_VALUES if specials else ())
    columns = []
    for _ in range(count):
        length = int(rng.integers(1, 12))
        columns.append([float(rng.choice(pool)) * float(rng.choice((1.0, -1.0, 0.5, 3.0))) for _ in range(length)])
    return columns


def _identity(value: float) -> tuple:
    """A float compared exactly: every NaN alike, the sign of a zero kept."""
    return ("nan",) if math.isnan(value) else (value, math.copysign(1.0, value))


def _left_fold(column: list[float]) -> float:
    total = 0.0
    for value in column:
        total += value
    return total


def test_the_device_fold_is_pythons_fold_bit_for_bit():
    """A column holding tensors folds on its device; the state must be the bits Python's ``sum``, ``max`` and
    ``min`` give the same column as floats, in record order, columns of every length padded together."""
    columns = _hard_columns(300, seed=0, specials=True)
    records = {}
    for index, column in enumerate(columns):
        kind = ("sum", "max", "moments")[index % 3]
        if kind == "moments":
            # The moments' min and max columns ride the same grid as the sums, negated into a max.
            records[f"k{index}"] = (
                kind,
                [(torch.tensor(v, dtype=torch.float64),) * 3 + (v, torch.tensor(v)) for v in column],
            )
        else:
            records[f"k{index}"] = (kind, [(torch.tensor(v, dtype=torch.float64),) for v in column])
    fold = world_metrics._RankFold(records)
    reference = rank_states(records)
    for key, (kind, state) in reference.items():
        folded = tuple(
            -float(value) if fold_kind is min else float(value)
            for fold_kind, value in zip(
                world_metrics._KINDS[kind][0],
                (fold._folded[(key, column)] for column in range(len(state))),
                strict=True,
            )
        )
        assert [_identity(v) for v in folded] == [_identity(v) for v in state], key


def test_the_exact_fold_data_tells_a_plain_sum_from_pythons():
    """The control for the check above: on these columns a plain left-to-right sum and ``torch.sum`` both differ
    from Python's compensated ``sum``, so a device fold that is either one fails it."""
    columns = _hard_columns(300, seed=0, specials=False)
    assert any(_left_fold(column) != sum(column) for column in columns)
    assert any(float(torch.tensor(column, dtype=torch.float64).sum()) != sum(column) for column in columns)


def _exact_worker(rank: int, out: str) -> None:
    """Two windows of mixed tensor and number records on one accumulator, each flush checked against the host
    fold of both ranks' records: at two ranks the backend's SUM and MAX are order-free, so they must agree exactly."""
    world = WorldMetrics()
    results = []
    for window in range(2):
        rng = np.random.default_rng(100 * window + rank)
        for column in _hard_columns(6, seed=100 * window + rank, specials=False):
            world.sum("s", torch.tensor(column[0], dtype=torch.float64))
            world.sum("s_numbers", column[-1])
            world.fraction("f", torch.tensor(column, dtype=torch.float32).sum(), len(column))
            world.mean("m", torch.tensor(float(rng.normal())), int(rng.integers(0, 3)))
            world.maximum("x", float(rng.normal()))
            values = torch.tensor(rng.normal(size=5) * 1e3)
            world.moments("v", values, torch.tensor(rng.integers(0, 2, size=5).astype(bool)))
            world.effective_sample_frac("e", values.abs(), torch.ones(5, dtype=torch.bool))
            world.covariance("c", values, values * 0.5 + 1.0, torch.ones(5, dtype=torch.bool))
        world.sum("nan", math.nan if rank == window else 1.0)
        local = rank_states(world._records)
        payloads = [None] * 2
        dist.all_gather_object(payloads, local)
        reference = {}
        for key, (kind, state) in pooled_states(payloads).items():
            reference.update(world_metrics._KINDS[kind][1](key, *state))
        logged = world.flush()
        results.append(
            (
                {key: _identity(value) for key, value in logged.items()},
                {key: _identity(value) for key, value in reference.items()},
            )
        )
    with open(f"{out}.{rank}", "wb") as fh:
        pickle.dump(results, fh)


def test_a_flush_equals_the_host_fold_of_every_ranks_records_exactly(tmp_path):
    out = str(tmp_path / "exact")
    run_gloo_ranks(_exact_worker, 2, out, pg_timeout=_PG_TIMEOUT)
    for rank in range(2):
        with open(f"{out}.{rank}", "rb") as fh:
            results = pickle.load(fh)
        for window, (logged, reference) in enumerate(results):
            assert logged == reference, f"rank {rank}, window {window}"
            assert logged["nan"] == ("nan",), "a NaN in a sum must reach the log"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
