#!/usr/bin/env python
"""A multi-worker coordinated map or filter pickles its callable once (``datasets`` does per worker task, about a
second each for a captured tokenizer) and writes what a single-process run writes. The cache name comes from the
bare callable, so ranks at different worker counts name one file."""

import functools
import json
import os

import pytest
from accelerate import PartialState
from datasets import Dataset, DatasetDict

import halo.data.pipeline.coordinated_map as coordination
from halo.data.pipeline.coordinated_map import coordinated_filter, coordinated_map
from halo.data.pipeline.map_cache_key import CACHE_KEY_ATTR, map_cache_file_name
from tests.common.gloo import run_gloo_ranks

_ROWS = Dataset.from_dict({"x": list(range(32))})

_WORKERS = 4

_RANKS = 2

# Each rank its own node on its own storage.
_PER_NODE_ENV = {"LOCAL_RANK": "0", "LOCAL_WORLD_SIZE": "1", "DIST_SHARED_FILESYSTEM": "0"}


class _CountedState:
    """Stands in for a tokenizer: counts the times this process pickles it."""

    pickles = 0

    def __init__(self, scale: int):
        self.scale = scale

    def __reduce__(self):
        _CountedState.pickles += 1
        return _CountedState, (self.scale,)


def _scaled(batch, indices, state):
    return {"y": [x * state.scale + i for x, i in zip(batch["x"], indices, strict=True)]}


def _divisible(row, state):
    return row["x"] % state.scale == 0


def _below(x, bound):
    return x < bound


def _closure_adding(offset: int):
    def add(row):
        return {"y": row["x"] + offset}

    return add


def _copied(row):
    return {"z": row["y"]}


async def _even(row):
    return row["x"] % 2 == 0


async def _halved(row):
    return {"y": row["x"] // 2}


@pytest.fixture(autouse=True)
def _isolated_cache(monkeypatch, tmp_path):
    monkeypatch.setenv("HF_DATASETS_CACHE", str(tmp_path))
    _CountedState.pickles = 0


@pytest.mark.parametrize("num_proc", [1, _WORKERS])
def test_a_map_pickles_its_state_once_whatever_the_worker_count(num_proc):
    mapped = coordinated_map(
        _ROWS,
        _scaled,
        desc="scale",
        num_proc=num_proc,
        batched=True,
        batch_size=5,
        with_indices=True,
        fn_kwargs={"state": _CountedState(3)},
    )
    assert mapped["y"] == [3 * x + x for x in _ROWS["x"]]
    assert _CountedState.pickles == 1


def test_every_split_of_a_dataset_dict_shares_the_one_pickle():
    splits = DatasetDict({"train": _ROWS, "test": _ROWS.select(range(8))})
    filtered = coordinated_filter(
        splits, _divisible, desc="divisible", num_proc=_WORKERS, fn_kwargs={"state": _CountedState(4)}
    )
    assert filtered["train"]["x"] == [x for x in _ROWS["x"] if x % 4 == 0]
    assert filtered["test"]["x"] == [0, 4]
    assert _CountedState.pickles == 1


def _rows_and_names(result: Dataset | DatasetDict) -> tuple:
    """The rows as Arrow tables (a filter's selection applied) and the stamped cache names, per split."""
    if isinstance(result, DatasetDict):
        return (
            {split: result[split].with_format("arrow")[:] for split in result},
            {split: getattr(result[split], CACHE_KEY_ATTR) for split in result},
        )
    return result.with_format("arrow")[:], getattr(result, CACHE_KEY_ATTR)


@pytest.mark.parametrize("as_splits", [False, True], ids=["dataset", "dataset-dict"])
@pytest.mark.parametrize(
    ("operation", "fn", "op_kwargs"),
    [
        (
            coordinated_map,
            _scaled,
            {"batched": True, "batch_size": 5, "with_indices": True, "fn_kwargs": {"state": _CountedState(3)}},
        ),
        (coordinated_map, _closure_adding(7), {}),
        (coordinated_filter, _divisible, {"fn_kwargs": {"state": _CountedState(4)}}),
        (coordinated_filter, functools.partial(_below, bound=20), {"input_columns": ["x"]}),
    ],
    ids=["batched-map-with-indices", "closure-map", "filter-with-fn-kwargs", "partial-filter-on-a-column"],
)
def test_workers_write_the_single_process_rows_under_its_cache_name(
    monkeypatch, tmp_path, as_splits, operation, fn, op_kwargs
):
    dataset = DatasetDict({"train": _ROWS, "test": _ROWS.select(range(8))}) if as_splits else _ROWS
    runs = []
    for num_proc in (1, _WORKERS):
        monkeypatch.setenv("HF_DATASETS_CACHE", str(tmp_path / f"workers-{num_proc}"))
        runs.append(_rows_and_names(operation(dataset, fn, desc="shape", num_proc=num_proc, **op_kwargs)))
    (single_rows, single_names), (pooled_rows, pooled_names) = runs
    assert pooled_rows == single_rows
    assert pooled_names == single_names


def test_the_cache_name_keys_the_bare_callable_not_its_wrapper():
    """Two closures differing only in a captured value write two cache files, each named from the bare callable."""
    stamps = []
    for offset in (1, 2):
        fn = _closure_adding(offset)
        mapped = coordinated_map(_ROWS, fn, desc="add", num_proc=_WORKERS)
        assert mapped["y"] == [x + offset for x in _ROWS["x"]]
        assert getattr(mapped, CACHE_KEY_ATTR) == map_cache_file_name("map", fn, _ROWS, "add", {})
        stamps.append(getattr(mapped, CACHE_KEY_ATTR))
    assert stamps[0] != stamps[1]


def test_the_cache_name_enters_the_fingerprint_datasets_takes():
    """A later uncoordinated map caches by fingerprint beside its input, so on equal fingerprints the second would
    read the first's file."""
    add = _closure_adding(7)
    mapped = [
        coordinated_map(_ROWS, add, desc="extras", num_proc=_WORKERS, cache_key_extras={"variant": variant})
        for variant in ("a", "b")
    ]
    assert mapped[0]._fingerprint != mapped[1]._fingerprint
    copied = [result.map(_copied) for result in mapped]
    assert copied[0].cache_files != copied[1].cache_files


def _scale_on_rank(rank: int, out_dir: str, per_node: bool) -> None:
    """Map on this rank, from its node's cache dir under per-node storage; record the rows, name and pickles."""
    PartialState()
    if per_node:
        os.environ["HF_DATASETS_CACHE"] = os.path.join(out_dir, f"node-{rank}")
    mapped = coordinated_map(
        _ROWS,
        _scaled,
        desc="ranks",
        num_proc=_WORKERS,
        batched=True,
        batch_size=5,
        with_indices=True,
        fn_kwargs={"state": _CountedState(3)},
    )
    record = {"y": list(mapped["y"]), "name": getattr(mapped, CACHE_KEY_ATTR), "pickles": _CountedState.pickles}
    with open(os.path.join(out_dir, f"rank-{rank}.json"), "w") as fh:
        json.dump(record, fh)


@pytest.mark.parametrize("per_node", [False, True], ids=["shared-filesystem", "per-node-storage"])
def test_every_rank_holds_the_single_process_rows_under_its_cache_name(tmp_path, per_node):
    """Every rank, writing or reading, pickles the state once and ends on the rows and cache name one process
    writes without workers."""
    reference = coordinated_map(
        _ROWS,
        _scaled,
        desc="ranks",
        num_proc=1,
        batched=True,
        batch_size=5,
        with_indices=True,
        fn_kwargs={"state": _CountedState(3)},
    )
    env = {"HF_DATASETS_CACHE": str(tmp_path / "shared"), **(_PER_NODE_ENV if per_node else {})}
    run_gloo_ranks(_scale_on_rank, _RANKS, str(tmp_path), per_node, env=env)
    for rank in range(_RANKS):
        record = json.loads((tmp_path / f"rank-{rank}.json").read_text())
        assert record == {"y": list(reference["y"]), "name": getattr(reference, CACHE_KEY_ATTR), "pickles": 1}


@pytest.mark.parametrize(
    "load_rank", [True, False], ids=["load-rank-across-workers", "follower-share-below-two-in-process"]
)
@pytest.mark.parametrize(
    ("operation", "fn", "column", "expected"),
    [
        (coordinated_map, _halved, "y", [x // 2 for x in _ROWS["x"]]),
        (coordinated_filter, _even, "x", [x for x in _ROWS["x"] if x % 2 == 0]),
    ],
    ids=["map", "filter"],
)
def test_a_coroutine_function_keeps_its_async_path(monkeypatch, load_rank, operation, fn, column, expected):
    """``datasets`` reads the coroutine flag off the callable it is handed; read off a wrapper, the map writes
    unawaited coroutines and the filter keeps every row."""
    monkeypatch.setattr(coordination, "is_fs_load_rank", lambda: load_rank)
    monkeypatch.setattr(coordination, "get_local_world_size", lambda: 2 * _WORKERS)
    assert operation(_ROWS, fn, desc="async", num_proc=_WORKERS)[column] == expected


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
