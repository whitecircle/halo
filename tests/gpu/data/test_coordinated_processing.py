#!/usr/bin/env python
"""Coordinated dataset processing: writer election, cache reuse, and cross-rank agreement.

``coordinated_map`` / ``coordinated_filter`` exist to make ONE rank per filesystem scope run each
map/filter and have the others load its cache file. Two properties are asserted, both of which need a
cache directory that every rank SHARES — with a per-rank temp dir the calls degenerate to "every rank
maps its own copy" and every assertion below still passes, testing nothing:

1. writer election — exactly the ``fs_aware_load_rank()`` ranks execute the map/filter body (the
   map fills a cache the peers then read, so it follows the INPUT flag: shared → global rank 0
   only; non-shared → each node's local rank 0);
2. cache reuse — the non-writer ranks get byte-identical results without running the body.

Every check runs to the end of its test and returns a bool that the runner agrees across ranks: an
early ``return`` past a collective would strand the peers, turning a clean FAIL into a hang.

Usage:
    torchrun --nproc_per_node=2 \
        tests/gpu/data/test_coordinated_processing.py
"""

import os
import shutil
import tempfile
import traceback
from contextlib import contextmanager
from pathlib import Path

import datasets.config
import torch
import torch.distributed as dist
from datasets import Dataset, load_from_disk

from src.data.pipeline.processing import (
    coordinated_filter,
    coordinated_map,
    process_dataset_with_map_and_filter,
)
from src.distributed.runtime import (
    barrier,
    fs_aware_load_rank,
    is_global_main_process,
    is_input_shared_filesystem,
)
from tests.common.harness import gpu_test_main
from tests.common.utils import log, log_all

# Configuration

NUM_SAMPLES = 50

# Counts how many rows each rank's map/filter body actually processed. A non-writer rank that loaded
# the writer's cache never calls the body, so this must stay 0 there.
EXEC_COUNTS: dict[str, int] = {}


# Utilities


def create_test_dataset(num_samples: int = NUM_SAMPLES) -> Dataset:
    """Create a deterministic synthetic dataset for testing."""
    data = []
    for i in range(num_samples):
        # Mix of short and long texts, with some that will be filtered
        length = (i % 5 + 1) * 10
        text = f"Sample {i}: " + " ".join([f"word{j}" for j in range(length)])
        value = i * 3 + 7  # Deterministic value for verification
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
    """All-gather ``value`` from every rank. Collective: every rank must reach it."""
    gathered = [None] * dist.get_world_size()
    dist.all_gather_object(gathered, value)
    return gathered


def agree(ok: bool) -> bool:
    """True only when EVERY rank passed. Collective, so it also keeps the ranks in lockstep."""
    flag = torch.tensor([1 if ok else 0], device=f"cuda:{torch.cuda.current_device()}")
    dist.all_reduce(flag, op=dist.ReduceOp.MIN)
    return bool(flag.item())


def set_hf_datasets_cache(path: str) -> None:
    """Point HF's dataset cache at ``path``. ``datasets.config`` latched the env at import, so the
    live config must be updated too — an env write alone leaves ``ensure_cache_dir`` disagreeing with
    what ``datasets`` actually uses."""
    os.environ["HF_DATASETS_CACHE"] = path
    datasets.config.HF_DATASETS_CACHE = Path(path)


@contextmanager
def shared_cache_dir(prefix: str):
    """A cache directory every rank sees at the SAME path — rank 0 creates it and broadcasts it.

    This is what makes the writer election observable at all: with a per-rank ``mkdtemp`` no rank can
    ever load another's cache file, so a broken election (or no coordination whatsoever) would look
    identical to a working one.
    """
    rank = dist.get_rank()
    path = tempfile.mkdtemp(prefix=prefix) if rank == 0 else None
    path = gather_objects(path)[0]
    previous = os.environ.get("HF_DATASETS_CACHE")
    previous_config = datasets.config.HF_DATASETS_CACHE
    set_hf_datasets_cache(path)
    try:
        yield path
    finally:
        barrier()  # nobody may delete the shared dir while a peer still reads it
        if previous is not None:
            set_hf_datasets_cache(previous)
        else:
            os.environ.pop("HF_DATASETS_CACHE", None)
            datasets.config.HF_DATASETS_CACHE = previous_config
        if rank == 0:
            shutil.rmtree(path, ignore_errors=True)


@contextmanager
def filesystem_mode(shared: bool):
    """Run the body under ``DIST_SHARED_FILESYSTEM``, faking one node per rank when non-shared.

    On a single-node launch local rank 0 IS global rank 0, so the two election branches are
    indistinguishable. Presenting every rank as its own node makes every rank a local main, which is
    exactly the per-node-local-storage contract: each writes and reads its own copy.
    """
    saved = {k: os.environ.get(k) for k in ("DIST_SHARED_FILESYSTEM", "LOCAL_RANK", "LOCAL_WORLD_SIZE")}
    os.environ["DIST_SHARED_FILESYSTEM"] = "1" if shared else "0"
    if not shared:
        os.environ["LOCAL_RANK"] = "0"
        os.environ["LOCAL_WORLD_SIZE"] = "1"
    try:
        yield
    finally:
        for key, value in saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def on_disk_source(cache_dir: str, num_samples: int) -> Dataset:
    """A byte-identical on-disk dataset for every rank.

    ``coordinated_map`` keys its cache file on the source dataset's identity, so an in-memory
    ``Dataset.from_list`` per rank would key differently per rank and defeat the cache sharing the
    election depends on.
    """
    path = os.path.join(cache_dir, "source")
    # Global rank 0 writes it unconditionally: under DIST_SHARED_FILESYSTEM=0 every rank is a local
    # main, and this scaffolding path is shared, so an fs-aware election would race here.
    if is_global_main_process() and not os.path.exists(path):
        create_test_dataset(num_samples).save_to_disk(path)
    barrier()
    return load_from_disk(path)


def counting(key: str, fn):
    """Wrap ``fn`` so every row it processes bumps ``EXEC_COUNTS[key]``."""
    EXEC_COUNTS[key] = 0

    def wrapped(example):
        EXEC_COUNTS[key] += 1
        return fn(example)

    wrapped.__name__ = f"{key}_body"
    return wrapped


def check_election(key: str, expected_rows: int) -> bool:
    """Assert only the fs-aware writer ranks ran the body, and that the writers ran it fully.

    Collective (it gathers), so every rank must reach it — never guard it behind an earlier failure.
    """
    is_writer = fs_aware_load_rank()
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
    """Assert every rank holds the same ``value``. Collective."""
    gathered = gather_objects(value)
    ok = all(item == gathered[0] for item in gathered)
    if not ok:
        log(f"    FAIL: ranks disagree on {label}: {gathered}")
    return ok


# Test Functions


def _run_map_and_filter(shared: bool) -> bool:
    """coordinated_map + coordinated_filter under one filesystem mode.

    Asserts the transform is correct, that exactly the elected writer ranks executed the bodies, and
    that every rank ends up with identical rows.
    """
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

        # Collectives below run unconditionally — a rank bailing out here would strand its peers.
        ok = check_election("map", 30) and ok
        ok = check_election("filter", 30) and ok
        ok = check_identical("mapped doubled column", [row["doubled"] for row in mapped]) and ok
        ok = check_identical("filtered ids", [row["id"] for row in filtered]) and ok

    return agree(ok)


def test_shared_filesystem_mode() -> bool:
    """DIST_SHARED_FILESYSTEM=1: global rank 0 alone maps; every other rank loads its cache."""
    return _run_map_and_filter(shared=True)


def test_non_shared_filesystem_mode() -> bool:
    """DIST_SHARED_FILESYSTEM=0: each node's local rank 0 maps its own node-local copy."""
    return _run_map_and_filter(shared=False)


def _add_100(example):
    """Reused across both calls of the cache-reuse test so the two share ONE cache key."""
    return {"transformed": example["value"] + 100}


def test_cache_file_is_the_shared_artifact() -> bool:
    """The writer's ``.arrow`` cache file is what every rank's result comes from.

    Both halves matter: a cache file appears in the shared dir, AND repeating the identical call
    executes the body nowhere. "A file exists somewhere" alone proves nothing about reuse.
    """
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

        # Byte-identical call (same fn object, same desc, same source) → the cache key repeats, so
        # NOBODY re-executes the body, writer included. counting() resets the counter.
        repeat = coordinated_map(dataset, counting("transform", _add_100), desc="cache_check", num_proc=1)
        if EXEC_COUNTS["transform"] != 0:
            log_all(f"    FAIL: cached rerun re-executed the body {EXEC_COUNTS['transform']}x (cache not reused)")
            ok = False
        if [row["transformed"] for row in repeat] != expected:
            log_all("    FAIL: cached rerun values wrong")
            ok = False
        ok = check_identical("cached rerun output", [row["transformed"] for row in repeat]) and ok

    return agree(ok)


def test_map_and_filter_combined() -> bool:
    """process_dataset_with_map_and_filter end-to-end: rejection sentinels dropped, ranks agree."""
    with filesystem_mode(shared=True), shared_cache_dir("coord_mapfilter_") as cache_dir:
        dataset = on_disk_source(cache_dir, 30)

        def process(example):
            """Emit a rejection sentinel (None input_ids) for even ids."""
            if example["id"] % 2 == 0:
                return {"input_ids": None, "processed_text": None}
            return {"input_ids": [example["value"]], "processed_text": f"processed_{example['id']}"}

        result = process_dataset_with_map_and_filter(
            dataset,
            counting("mapfilter", process),
            filter_field="input_ids",
            num_proc=1,
            remove_columns=dataset.column_names,
            desc="map_and_filter_test",
            log_stats=True,
        )

        expected_count = sum(1 for i in range(30) if i % 2 != 0)
        ok = True
        if len(result) != expected_count or any(row["input_ids"] is None for row in result):
            log_all(f"    FAIL: expected {expected_count} rows with no None input_ids, got {len(result)}")
            ok = False
        ok = check_election("mapfilter", 30) and ok
        ok = check_identical("map+filter texts", [row["processed_text"] for row in result]) and ok

    return agree(ok)


def test_diverging_source_keys_a_distinct_cache() -> bool:
    """Ranks holding DIFFERENT data must not collide on one cache file.

    This is the pre-sharded layout (each DP rank owns a disjoint slice): the cache key folds in the
    source dataset identity, so the writer's file must never be served to a rank whose source
    differs — that would silently replace this rank's shard with rank 0's.
    """
    with filesystem_mode(shared=True), shared_cache_dir("coord_disjoint_") as cache_dir:
        rank = dist.get_rank()
        path = os.path.join(cache_dir, f"slice_{rank}")
        # Disjoint per-rank slices, mirroring ShardedDatasetLoader's assignment.
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
        # Every rank must have run its own body: no rank's slice can come from another's cache.
        if EXEC_COUNTS["disjoint"] != 8:
            log_all(f"    FAIL: rank {rank} executed the body {EXEC_COUNTS['disjoint']}x, expected 8 (own slice)")
            ok = False
        per_rank = gather_objects([row["doubled"] for row in mapped])
        if len({tuple(rows) for rows in per_rank}) != len(per_rank):
            log(f"    FAIL: two ranks share output despite disjoint sources — cache collision: {per_rank}")
            ok = False

    return agree(ok)


# Test Runner

ALL_TESTS = [
    ("shared_filesystem_mode", test_shared_filesystem_mode),
    ("non_shared_filesystem_mode", test_non_shared_filesystem_mode),
    ("cache_file_is_the_shared_artifact", test_cache_file_is_the_shared_artifact),
    ("map_and_filter_combined", test_map_and_filter_combined),
    ("diverging_source_keys_a_distinct_cache", test_diverging_source_keys_a_distinct_cache),
]


def run(ctx) -> dict:
    """Run all coordinated processing tests."""
    log(f"\n{'=' * 70}")
    log("  Coordinated Dataset Processing Tests")
    log(f"  World size: {ctx.world_size}, Rank: {ctx.rank}")
    log(f"  GPU: {torch.cuda.get_device_name(ctx.local_rank)}")
    log(f"{'=' * 70}\n")

    results: dict[str, bool] = {}
    for test_name, test_fn in ALL_TESTS:
        log(f"\n  [{test_name}]")
        barrier()

        try:
            result = test_fn()
        except Exception as e:
            log_all(f"  [{test_name}] UNHANDLED EXCEPTION: {e}")
            traceback.print_exc()
            result = False

        # A rank that threw skipped this test's collectives; re-agree so the verdict (and the
        # rank alignment) is restored before the next test rather than deadlocking inside it.
        results[test_name] = agree(result)
        log(f"  [{test_name}] {'PASS' if results[test_name] else 'FAIL'}")

        barrier()

    return {"checks": results}


main = gpu_test_main(min_world_size=2, prefix="coordinated_processing", partial_state=False)(run)

if __name__ == "__main__":
    main()
