#!/usr/bin/env python
"""Coordinated dataset processing: writer election, cache reuse, and cross-rank agreement, on a cache dir every rank
shares.

Every check runs to the end of its test, which agrees the verdict before raising: an early exit past a collective would
strand the peers, turning a clean FAIL into a hang.
"""

import os
import shutil
import tempfile
from contextlib import contextmanager
from pathlib import Path

import datasets.config
import torch
import torch.distributed as dist
from datasets import Dataset, load_from_disk

from halo.data.pipeline.coordinated_map import coordinated_filter, coordinated_map, coordinated_map_and_filter
from halo.runtime.collectives import barrier
from halo.runtime.filesystem import is_fs_load_rank, is_input_shared_filesystem
from halo.runtime.ranks import is_global_main_process
from tests.common.distributed import reset_shared_filesystem_consensus
from tests.common.harness import gpu_test_main, record_check
from tests.common.rank_log import log, log_all

NUM_SAMPLES = 50

# A non-writer rank that loaded the writer's cache never calls the body, so its count must stay 0.
EXEC_COUNTS: dict[str, int] = {}


def create_test_dataset(num_samples: int = NUM_SAMPLES) -> Dataset:
    data = []
    for i in range(num_samples):
        length = (i % 5 + 1) * 10
        text = f"Sample {i}: " + " ".join([f"word{j}" for j in range(length)])
        value = i * 3 + 7
        data.append(
            {
                "text": text,
                "id": i,
                "value": value,
                "length": length,
            }
        )
    return Dataset.from_list(data)


def gather_objects(value):
    """COLLECTIVE: every rank must reach it."""
    gathered = [None] * dist.get_world_size()
    dist.all_gather_object(gathered, value)
    return gathered


def require_agreement(ok: bool, label: str) -> None:
    """COLLECTIVE. Raise unless every rank passed."""
    flag = torch.tensor([1 if ok else 0], device=f"cuda:{torch.cuda.current_device()}")
    dist.all_reduce(flag, op=dist.ReduceOp.MIN)
    if not flag.item():
        raise AssertionError(f"{label} failed on at least one rank; see the FAIL lines above")


def set_hf_datasets_cache(path: str) -> None:
    """``datasets.config`` latched the env at import, so an env write alone leaves ``ensure_cache_dir`` disagreeing."""
    os.environ["HF_DATASETS_CACHE"] = path
    datasets.config.HF_DATASETS_CACHE = Path(path)


@contextmanager
def shared_cache_dir(prefix: str):
    """Rank 0 creates it and broadcasts it: with a per-rank ``mkdtemp`` no rank can load another's cache file, so a
    broken election would look identical to a working one."""
    rank = dist.get_rank()
    path = tempfile.mkdtemp(prefix=prefix) if rank == 0 else None
    path = gather_objects(path)[0]
    previous = os.environ.get("HF_DATASETS_CACHE")
    previous_config = datasets.config.HF_DATASETS_CACHE
    set_hf_datasets_cache(path)
    try:
        yield path
    finally:
        barrier()  # Nobody may delete the shared dir while a peer still reads it.
        if previous is not None:
            set_hf_datasets_cache(previous)
        else:
            os.environ.pop("HF_DATASETS_CACHE", None)
            datasets.config.HF_DATASETS_CACHE = previous_config
        if rank == 0:
            shutil.rmtree(path, ignore_errors=True)


@contextmanager
def filesystem_mode(shared: bool):
    """Fakes one node per rank when non-shared: on a single-node launch local rank 0 is global rank 0, so the two
    election branches are otherwise indistinguishable. The production init's memoized flags are dropped on entry and
    exit."""
    saved = {k: os.environ.get(k) for k in ("DIST_SHARED_FILESYSTEM", "LOCAL_RANK", "LOCAL_WORLD_SIZE")}
    os.environ["DIST_SHARED_FILESYSTEM"] = "1" if shared else "0"
    if not shared:
        os.environ["LOCAL_RANK"] = "0"
        os.environ["LOCAL_WORLD_SIZE"] = "1"
    reset_shared_filesystem_consensus()
    try:
        assert is_input_shared_filesystem() == shared, "an agreed scope overrode the filesystem mode under test"
        yield
    finally:
        for key, value in saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        reset_shared_filesystem_consensus()


def on_disk_source(cache_dir: str, num_samples: int) -> Dataset:
    """``coordinated_map`` keys its cache on the source's identity, so an in-memory dataset per rank defeats cache
    sharing."""
    path = os.path.join(cache_dir, "source")
    # Global rank 0 writes it: under DIST_SHARED_FILESYSTEM=0 every rank is a local main, and an fs-aware election
    # would race on this shared scaffolding path.
    if is_global_main_process() and not os.path.exists(path):
        create_test_dataset(num_samples).save_to_disk(path)
    barrier()
    return load_from_disk(path)


def counting(key: str, fn):
    EXEC_COUNTS[key] = 0

    def wrapped(example):
        EXEC_COUNTS[key] += 1
        return fn(example)

    wrapped.__name__ = f"{key}_body"
    return wrapped


def check_election(key: str, expected_rows: int) -> bool:
    """COLLECTIVE, so never guard it behind an earlier failure. Only the fs-aware writer ranks ran the body, in
    full."""
    is_writer = is_fs_load_rank()
    executed = EXEC_COUNTS[key]
    ok = executed == (expected_rows if is_writer else 0)
    if not ok:
        log_all(
            f"    FAIL[{key}]: writer={is_writer} ran the body {executed}x, "
            f"expected {expected_rows if is_writer else 0}"
        )
    reports = gather_objects((dist.get_rank(), is_writer, executed))
    writers = [r for r in reports if r[1]]
    if not writers:
        log(f"    FAIL[{key}]: no rank was elected writer — the map ran nowhere: {reports}")
        ok = False
    if is_input_shared_filesystem() and len(writers) != 1:
        log(f"    FAIL[{key}]: shared FS must elect exactly ONE writer, got {writers}")
        ok = False
    log(f"    election[{key}] (rank, is_writer, rows_executed) = {reports}")
    return ok


def check_identical(label: str, value) -> bool:
    """COLLECTIVE. Every rank holds the same ``value``."""
    gathered = gather_objects(value)
    ok = all(item == gathered[0] for item in gathered)
    if not ok:
        log(f"    FAIL: ranks disagree on {label}: {gathered}")
    return ok


def _run_map_and_filter(shared: bool) -> None:
    with filesystem_mode(shared), shared_cache_dir(f"coord_{'shared' if shared else 'local'}_") as cache_dir:
        dataset = on_disk_source(cache_dir, 30)

        mapped = coordinated_map(
            dataset,
            counting("map", lambda ex: {"doubled": ex["value"] * 2}),
            desc="double_values",
            num_proc=1,
        )
        filtered = coordinated_filter(
            mapped,
            counting("filter", lambda ex: ex["length"] >= 30),
            desc="filter_long",
            num_proc=1,
        )

        ok = True
        if "doubled" not in mapped.column_names or len(mapped) != 30:
            log_all(f"    FAIL: map output shape wrong: cols={mapped.column_names} rows={len(mapped)}")
            ok = False
        elif [row["doubled"] for row in mapped] != [dataset[i]["value"] * 2 for i in range(30)]:
            log_all("    FAIL: map values wrong")
            ok = False

        expected_kept = sum(1 for i in range(30) if (i % 5 + 1) * 10 >= 30)
        if len(filtered) != expected_kept or any(row["length"] < 30 for row in filtered):
            log_all(f"    FAIL: filter kept {len(filtered)} rows (expected {expected_kept}) or kept a short row")
            ok = False

        # Collectives below run unconditionally: a rank bailing out here would strand its peers.
        ok = check_election("map", 30) and ok
        ok = check_election("filter", 30) and ok
        ok = check_identical("mapped doubled column", [row["doubled"] for row in mapped]) and ok
        ok = check_identical("filtered ids", [row["id"] for row in filtered]) and ok

    require_agreement(ok, f"map+filter ({'shared' if shared else 'non-shared'} filesystem)")


def test_shared_filesystem_mode() -> None:
    _run_map_and_filter(shared=True)


def test_non_shared_filesystem_mode() -> None:
    _run_map_and_filter(shared=False)


def _add_100(example):
    """Reused across both calls so the two share one cache key."""
    return {"transformed": example["value"] + 100}


def test_cache_file_is_the_shared_artifact() -> None:
    """A cache file appears in the shared dir and repeating the identical call runs the body nowhere: a file alone
    proves nothing about reuse."""
    with filesystem_mode(shared=True), shared_cache_dir("coord_cache_") as cache_dir:
        dataset = on_disk_source(cache_dir, 15)
        expected = [dataset[i]["value"] + 100 for i in range(15)]

        first = coordinated_map(dataset, counting("transform", _add_100), desc="cache_check", num_proc=1)
        ok = check_election("transform", 15)
        if [row["transformed"] for row in first] != expected:
            log_all("    FAIL: transform values wrong")
            ok = False

        arrow_files = sorted(f for f in os.listdir(cache_dir) if f.endswith(".arrow"))
        if not arrow_files:
            log(f"    FAIL: no .arrow cache file in {cache_dir}: {os.listdir(cache_dir)}")
            ok = False
        ok = check_identical("cache file listing", arrow_files) and ok

        # Same fn object, desc and source repeat the cache key, so nobody re-executes the body. counting() resets the
        # counter.
        repeat = coordinated_map(dataset, counting("transform", _add_100), desc="cache_check", num_proc=1)
        if EXEC_COUNTS["transform"] != 0:
            log_all(f"    FAIL: cached rerun re-executed the body {EXEC_COUNTS['transform']}x (cache not reused)")
            ok = False
        if [row["transformed"] for row in repeat] != expected:
            log_all("    FAIL: cached rerun values wrong")
            ok = False
        ok = check_identical("cached rerun output", [row["transformed"] for row in repeat]) and ok

    require_agreement(ok, "cache reuse")


def test_map_and_filter_combined() -> None:
    with filesystem_mode(shared=True), shared_cache_dir("coord_mapfilter_") as cache_dir:
        dataset = on_disk_source(cache_dir, 30)

        def process(example):
            if example["id"] % 2 == 0:
                return {"input_ids": None, "processed_text": None}
            return {"input_ids": [example["value"]], "processed_text": f"processed_{example['id']}"}

        result = coordinated_map_and_filter(
            dataset,
            counting("mapfilter", process),
            filter_field="input_ids",
            num_proc=1,
            remove_columns=dataset.column_names,
            desc="map_and_filter_test",
        )

        expected_count = sum(1 for i in range(30) if i % 2 != 0)
        ok = True
        if len(result) != expected_count or any(row["input_ids"] is None for row in result):
            log_all(f"    FAIL: expected {expected_count} rows with no None input_ids, got {len(result)}")
            ok = False
        ok = check_election("mapfilter", 30) and ok
        ok = check_identical("map+filter texts", [row["processed_text"] for row in result]) and ok

    require_agreement(ok, "map+filter combined")


def test_diverging_source_keys_a_distinct_cache() -> None:
    """The pre-sharded layout: the cache key folds in the source identity, or the writer's file would silently replace
    this rank's shard with rank 0's."""
    with filesystem_mode(shared=True), shared_cache_dir("coord_disjoint_") as cache_dir:
        rank = dist.get_rank()
        path = os.path.join(cache_dir, f"slice_{rank}")
        Dataset.from_list([{"id": rank * 100 + i, "value": rank * 100 + i} for i in range(8)]).save_to_disk(path)
        sliced = load_from_disk(path)

        mapped = coordinated_map(
            sliced,
            counting("disjoint", lambda ex: {"doubled": ex["value"] * 2}),
            desc="disjoint_slices",
            num_proc=1,
        )

        ok = True
        expected = [(rank * 100 + i) * 2 for i in range(8)]
        if [row["doubled"] for row in mapped] != expected:
            log_all(f"    FAIL: rank {rank} got {[row['doubled'] for row in mapped]}, expected its own {expected}")
            ok = False
        # No rank's slice can come from another's cache.
        if EXEC_COUNTS["disjoint"] != 8:
            log_all(f"    FAIL: rank {rank} executed the body {EXEC_COUNTS['disjoint']}x, expected 8 (own slice)")
            ok = False
        per_rank = gather_objects([row["doubled"] for row in mapped])
        if len({tuple(rows) for rows in per_rank}) != len(per_rank):
            log(f"    FAIL: two ranks share output despite disjoint sources — cache collision: {per_rank}")
            ok = False

    require_agreement(ok, "disjoint-source cache keys")


def run(ctx) -> dict:
    log(f"\n{'=' * 70}")
    log("  Coordinated Dataset Processing Tests")
    log(f"  World size: {ctx.world_size}, Rank: {ctx.rank}")
    log(f"  GPU: {torch.cuda.get_device_name(ctx.local_rank)}")
    log(f"{'=' * 70}\n")

    checks: dict[str, bool] = {}
    record_check(checks, "shared_filesystem_mode", test_shared_filesystem_mode)
    record_check(checks, "non_shared_filesystem_mode", test_non_shared_filesystem_mode)
    record_check(checks, "cache_file_is_the_shared_artifact", test_cache_file_is_the_shared_artifact)
    record_check(checks, "map_and_filter_combined", test_map_and_filter_combined)
    record_check(checks, "diverging_source_keys_a_distinct_cache", test_diverging_source_keys_a_distinct_cache)
    return {"checks": checks}


main = gpu_test_main(min_world_size=2, prefix="coordinated_processing", partial_state=False)(run)

if __name__ == "__main__":
    main()
