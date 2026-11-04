#!/usr/bin/env python
"""``fs_aware_main_first``: main-first coordination over the c10d store. A waiter parked in a collective through a
100B+ download dies at the NCCL watchdog on every peer, so it blocks on a store key."""

import datetime
import os
import time

import pytest
import torch.distributed as dist

from halo.env import DEFAULT_STORE_TIMEOUT_HOURS, store_timeout
from halo.runtime import consensus, filesystem
from halo.runtime.filesystem import fs_aware_main_first
from tests.common.distributed import FakeStore, reset_shared_filesystem_consensus
from tests.common.gloo import run_gloo_ranks

WORLD_SIZE = 2

# The body outlives the process-group timeout, far enough apart that a slow CI box cannot blur the two.
PG_TIMEOUT_SEC = 5
MAIN_BODY_SEC = 8

# Explicitly bounded: under the 4 h default a regression would surface only as a rank killed at the join deadline.
WAIT_TIMEOUT = datetime.timedelta(seconds=30)


def _worker(rank: int, tmp_dir: str) -> None:
    failures = []
    marker = os.path.join(tmp_dir, "downloaded.marker")
    with fs_aware_main_first("dl", timeout=WAIT_TIMEOUT):
        if rank == 0:
            time.sleep(0.5)  # Widen the race window a naive implementation would lose.
            with open(marker, "w") as fh:
                fh.write("done")
        elif not os.path.isfile(marker):
            failures.append("non-main rank entered before the main rank finished")

    # A short timeout, so a variant that leaves the waiter blocked fails in seconds.
    released_after = None
    t0 = time.time()
    try:
        with fs_aware_main_first("dl_fail", timeout=WAIT_TIMEOUT):
            if rank == 0:
                raise RuntimeError("download failed")
        released_after = time.time() - t0
    except RuntimeError as e:
        if rank == 0:
            released_after = time.time() - t0  # The main rank's own body raised, as intended.
        else:
            failures.append(f"waiter was not released by the main rank's failure: {e}")
    if rank != 0 and (released_after is None or released_after >= WAIT_TIMEOUT.total_seconds()):
        failures.append(f"waiter released only after {released_after}s (expected promptly)")

    # Stale done-keys must not leak into a fresh generation.
    marker2 = os.path.join(tmp_dir, "second.marker")
    with fs_aware_main_first("dl", timeout=WAIT_TIMEOUT):
        if rank == 0:
            time.sleep(0.2)
            with open(marker2, "w") as fh:
                fh.write("done")
        elif not os.path.isfile(marker2):
            failures.append("second use entered before the main rank finished (stale key?)")

    result = "PASS" if not failures else "FAIL: " + "; ".join(failures)
    with open(os.path.join(tmp_dir, f"result_{rank}.txt"), "w") as fh:
        fh.write(result)


def _watchdog_worker(rank: int, tmp_dir: str) -> None:
    marker = os.path.join(tmp_dir, "slow.marker")
    try:
        with fs_aware_main_first("slow_download", timeout=datetime.timedelta(seconds=60)):
            if rank == 0:
                time.sleep(MAIN_BODY_SEC)
                with open(marker, "w") as fh:
                    fh.write("done")
        ordered = rank == 0 or os.path.isfile(marker)
        result = "PASS" if ordered else "FAIL: waiter was released before the main rank finished"
    except Exception as e:  # The failure mode under test is a wait that raises.
        result = f"FAIL: {type(e).__name__}: {str(e).splitlines()[0][:160]}"
    with open(os.path.join(tmp_dir, f"watchdog_{rank}.txt"), "w") as fh:
        fh.write(result)


def _non_shared_fs_worker(rank: int, tmp_dir: str) -> None:
    with fs_aware_main_first("per_node_download", timeout=datetime.timedelta(seconds=30)):
        entered = time.time()
        time.sleep(1.0)  # Only serialized coordination would push the other rank past this.
    with open(os.path.join(tmp_dir, f"entered_{rank}.txt"), "w") as fh:
        fh.write(repr(entered))


def _split_fs_flag_worker(rank: int, tmp_dir: str) -> None:
    os.environ["DIST_SHARED_FILESYSTEM"] = "1" if rank == 0 else "0"
    try:
        filesystem.resolve_shared_filesystem_consensus()
        with open(os.path.join(tmp_dir, f"fs_{rank}.txt"), "w") as fh:
            fh.write(repr(filesystem.is_shared_filesystem()))
    finally:
        reset_shared_filesystem_consensus()


def _split_side_fs_flags_worker(rank: int, tmp_dir: str) -> None:
    os.environ.update(
        DIST_INPUT_SHARED_FILESYSTEM="1" if rank == 0 else "0",
        DIST_OUTPUT_SHARED_FILESYSTEM="0" if rank == 0 else "1",
    )
    os.environ.pop("DIST_SHARED_FILESYSTEM", None)
    try:
        filesystem.resolve_shared_filesystem_consensus()
        with open(os.path.join(tmp_dir, f"sides_{rank}.txt"), "w") as fh:
            fh.write(f"{filesystem.is_input_shared_filesystem()},{filesystem.is_output_shared_filesystem()}")
    finally:
        reset_shared_filesystem_consensus()


def test_store_main_first_orders_and_releases(tmp_path):
    run_gloo_ranks(_worker, WORLD_SIZE, str(tmp_path))
    for rank in range(WORLD_SIZE):
        with open(tmp_path / f"result_{rank}.txt") as fh:
            result = fh.read()
        assert result == "PASS", f"rank {rank}: {result}"


def test_main_body_may_outlive_the_process_group_timeout(tmp_path):
    """Under ``dist.barrier()`` gloo aborts the recv at PG_TIMEOUT_SEC, the NCCL-watchdog kill of a real download
    scaled down."""
    run_gloo_ranks(_watchdog_worker, WORLD_SIZE, str(tmp_path), pg_timeout=datetime.timedelta(seconds=PG_TIMEOUT_SEC))
    for rank in range(WORLD_SIZE):
        with open(tmp_path / f"watchdog_{rank}.txt") as fh:
            result = fh.read()
        assert result == "PASS", f"rank {rank}: {result}"


def test_non_shared_filesystem_scopes_coordination_per_node(tmp_path):
    """Serializing across the world turns an N-node launch into N sequential downloads."""
    run_gloo_ranks(
        _non_shared_fs_worker,
        WORLD_SIZE,
        str(tmp_path),
        env={"LOCAL_RANK": "0", "LOCAL_WORLD_SIZE": "1", "DIST_SHARED_FILESYSTEM": "0"},
    )
    entered = []
    for rank in range(WORLD_SIZE):
        with open(tmp_path / f"entered_{rank}.txt") as fh:
            entered.append(float(fh.read()))
    assert abs(entered[0] - entered[1]) < 0.5, f"nodes were serialized against each other: {entered}"


def test_shared_filesystem_flag_is_agreed_across_ranks(tmp_path):
    """A rank reading `0` while peers read `1` sits under another scope, so each waits out the full store timeout.
    Per-node env injection makes this realistic."""
    run_gloo_ranks(_split_fs_flag_worker, WORLD_SIZE, str(tmp_path))
    resolved = []
    for rank in range(WORLD_SIZE):
        with open(tmp_path / f"fs_{rank}.txt") as fh:
            resolved.append(fh.read())
    assert resolved[0] == resolved[1] == "True", f"ranks disagreed on the coordination scope: {resolved}"


def test_side_shared_filesystem_flags_are_agreed_across_ranks(tmp_path):
    """The side flags elect writers: split them and two nodes both believe they are the sole checkpoint writer, with
    no error."""
    run_gloo_ranks(_split_side_fs_flags_worker, WORLD_SIZE, str(tmp_path))
    resolved = []
    for rank in range(WORLD_SIZE):
        with open(tmp_path / f"sides_{rank}.txt") as fh:
            resolved.append(fh.read())
    assert resolved[0] == resolved[1], f"ranks disagreed on the input/output scope: {resolved}"
    # Rank 0's verdict wins.
    assert resolved[0] == "True,False", f"the consensus did not adopt rank 0's side flags: {resolved[0]}"


def test_no_distributed_is_passthrough():
    with fs_aware_main_first("noop"):
        pass  # Must not require an initialized process group.


@pytest.fixture()
def fake_world(monkeypatch):
    store = FakeStore()
    monkeypatch.setattr(dist, "is_available", lambda: True)
    monkeypatch.setattr(dist, "is_initialized", lambda: True)
    monkeypatch.setattr(consensus.c10d, "_get_default_store", lambda: store)
    monkeypatch.setattr(filesystem, "get_global_world_size", lambda: 2)
    monkeypatch.setenv("DIST_SHARED_FILESYSTEM", "1")
    reset_shared_filesystem_consensus()
    return store


def _enter_as(monkeypatch, rank: int, tag: str = "dl") -> None:
    monkeypatch.setattr(filesystem, "get_global_rank", lambda: rank)
    monkeypatch.setattr(filesystem, "is_global_main_process", lambda: rank == 0)
    with filesystem.fs_aware_main_first(tag):
        pass


def test_entry_count_divergence_is_detected_when_a_waiter_runs_ahead(fake_world, monkeypatch):
    """An extra entry leaves the rank off-by-one on a phase key peers never write, an unattributed multi-hour stall
    unless named."""
    _enter_as(monkeypatch, 0)
    _enter_as(monkeypatch, 1)  # Round 1 complete and collected.
    with pytest.raises(RuntimeError) as excinfo:
        _enter_as(monkeypatch, 1)  # Rank 1 enters a second time; rank 0 has not.
    message = str(excinfo.value)
    assert "main_first/dl" in message, f"the tag must be named: {message}"
    assert "entered this tag" in message, f"the entry-count divergence must be named: {message}"


def test_matched_entry_counts_stay_silent(fake_world, monkeypatch):
    """The main rank never waits for waiters, so repeated rounds of one tag are legitimate while counts match."""
    for _ in range(3):
        _enter_as(monkeypatch, 0)
        _enter_as(monkeypatch, 1)


def test_store_timeout_is_configurable(monkeypatch):
    monkeypatch.delenv("DIST_STORE_TIMEOUT_HOURS", raising=False)
    assert store_timeout() == datetime.timedelta(hours=DEFAULT_STORE_TIMEOUT_HOURS)
    # The default must stay long enough to wait out a 100B+ download.
    assert store_timeout() >= datetime.timedelta(hours=1)
    monkeypatch.setenv("DIST_STORE_TIMEOUT_HOURS", "9")
    assert store_timeout() == datetime.timedelta(hours=9)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
