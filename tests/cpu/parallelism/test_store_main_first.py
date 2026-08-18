#!/usr/bin/env python
"""``fs_aware_main_first`` — main-first coordination over the c10d store, not a NCCL barrier.

The main rank's body is unbounded single-rank work: a 100B+ ``snapshot_download``
(``_ensure_model_downloaded``), whole-corpus packing (``pack_dataset_coordinated``), an S3 dataset
fetch. Parking the waiters inside a collective for that duration kills the job at the NCCL watchdog
(``DIST_NCCL_TIMEOUT_MINUTES``) on EVERY peer, so the waiters must block on a store key with an
hours-scale wall-clock timeout instead.

Proven on real gloo groups: (1) ordering — the non-main rank enters the context only after the main
rank's body finished (observes its side effect); (2) release-on-failure — an exception in the main
rank's body still releases the waiters; (3) re-entrancy — a second use of one tag coordinates
independently of the first (per-call generation keys); (4) **no collective wait** — with a 2 s
process-group timeout and a 5 s main-rank body, every rank still completes, which a
``dist.barrier()`` rendezvous cannot do; (5) the timeout is the store knob
(``DIST_STORE_TIMEOUT_HOURS``), not a hardcoded 4 h.

    python tests/cpu/parallelism/test_store_main_first.py
"""

import contextlib
import datetime
import os
import time

import pytest
import torch.distributed as dist
import torch.multiprocessing as mp

from src.distributed import filesystem, runtime
from src.distributed.filesystem import fs_aware_main_first
from src.distributed.runtime import get_store_timeout
from src.env import DEFAULT_STORE_TIMEOUT_HOURS
from tests.common.distributed import FakeStore
from tests.common.ports import free_port

WORLD_SIZE = 2

# The main rank's body must be allowed to outlive the process-group timeout: that is the whole point
# of coordinating on the store. Kept far apart so a slow CI box cannot blur the two, and the PG timeout
# wide enough that spawn + torch-import skew between the ranks on a loaded host cannot trip the
# rendezvous before the body is even reached.
PG_TIMEOUT_SEC = 5
MAIN_BODY_SEC = 8

# Every store wait in this module is explicitly bounded: the 4 h production default would turn a
# regression into a CI stall rather than a failure (mp.start_processes(join=True) has no timeout).
WAIT_TIMEOUT = datetime.timedelta(seconds=30)


def _worker(rank: int, tmp_dir: str, port: int) -> None:
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port), RANK=str(rank), WORLD_SIZE=str(WORLD_SIZE))
    dist.init_process_group("gloo", rank=rank, world_size=WORLD_SIZE)
    failures = []
    marker = os.path.join(tmp_dir, "downloaded.marker")
    try:
        # (1) Ordering: main writes the marker inside the context; the waiter must see it.
        with fs_aware_main_first("dl", timeout=WAIT_TIMEOUT):
            if rank == 0:
                time.sleep(0.5)  # widen the race window a naive implementation would lose
                with open(marker, "w") as fh:
                    fh.write("done")
            elif not os.path.isfile(marker):
                failures.append("non-main rank entered before the main rank finished")

        # (2) Release-on-failure: the main rank raising must still release the waiters, PROMPTLY —
        # a short timeout, so a variant that leaves the waiter blocked fails the test in seconds
        # instead of stalling CI for the full store timeout.
        released_after = None
        t0 = time.time()
        try:
            with fs_aware_main_first("dl_fail", timeout=WAIT_TIMEOUT):
                if rank == 0:
                    raise RuntimeError("download failed")
            released_after = time.time() - t0
        except RuntimeError as e:
            if rank == 0:
                released_after = time.time() - t0  # the main rank's own body raised, as intended
            else:
                failures.append(f"waiter was not released by the main rank's failure: {e}")
        if rank != 0 and (released_after is None or released_after >= WAIT_TIMEOUT.total_seconds()):
            failures.append(f"waiter released only after {released_after}s (expected promptly)")

        # (3) Re-entrancy: a fresh generation coordinates again (stale done-keys must not leak).
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
    finally:
        dist.destroy_process_group()


def _watchdog_worker(rank: int, tmp_dir: str, port: int) -> None:
    """Main-rank body longer than the PG timeout: a collective-based wait dies, the store wait does not."""
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port), RANK=str(rank), WORLD_SIZE=str(WORLD_SIZE))
    dist.init_process_group(
        "gloo", rank=rank, world_size=WORLD_SIZE, timeout=datetime.timedelta(seconds=PG_TIMEOUT_SEC)
    )
    marker = os.path.join(tmp_dir, "slow.marker")
    try:
        with fs_aware_main_first("slow_download", timeout=datetime.timedelta(seconds=60)):
            if rank == 0:
                time.sleep(MAIN_BODY_SEC)
                with open(marker, "w") as fh:
                    fh.write("done")
        ordered = rank == 0 or os.path.isfile(marker)
        result = "PASS" if ordered else "FAIL: waiter was released before the main rank finished"
    except Exception as e:  # the failure mode under test is exactly "a wait that raises"
        result = f"FAIL: {type(e).__name__}: {str(e).splitlines()[0][:160]}"
    with open(os.path.join(tmp_dir, f"watchdog_{rank}.txt"), "w") as fh:
        fh.write(result)
    # Teardown of an already-aborted group must not mask the result written above.
    with contextlib.suppress(Exception):
        dist.destroy_process_group()


def _non_shared_fs_worker(rank: int, tmp_dir: str, port: int) -> None:
    """Two 1-rank "nodes" on a non-shared FS: each is its own main and must not wait on the other."""
    os.environ.update(
        MASTER_ADDR="127.0.0.1",
        MASTER_PORT=str(port),
        RANK=str(rank),
        WORLD_SIZE=str(WORLD_SIZE),
        LOCAL_RANK="0",
        LOCAL_WORLD_SIZE="1",
        DIST_SHARED_FILESYSTEM="0",
    )
    dist.init_process_group("gloo", rank=rank, world_size=WORLD_SIZE)
    try:
        with fs_aware_main_first("per_node_download", timeout=datetime.timedelta(seconds=30)):
            entered = time.time()
            time.sleep(1.0)  # only serialized coordination would push the other rank past this
        with open(os.path.join(tmp_dir, f"entered_{rank}.txt"), "w") as fh:
            fh.write(repr(entered))
    finally:
        dist.destroy_process_group()


def _split_fs_flag_worker(rank: int, tmp_dir: str, port: int) -> None:
    """Ranks disagree on DIST_SHARED_FILESYSTEM; after init they must still agree on the scope."""
    os.environ.update(
        MASTER_ADDR="127.0.0.1",
        MASTER_PORT=str(port),
        RANK=str(rank),
        WORLD_SIZE=str(WORLD_SIZE),
        LOCAL_RANK=str(rank),
        LOCAL_WORLD_SIZE=str(WORLD_SIZE),
        DIST_SHARED_FILESYSTEM="1" if rank == 0 else "0",
    )
    dist.init_process_group("gloo", rank=rank, world_size=WORLD_SIZE)
    try:
        runtime.resolve_shared_filesystem_consensus()
        with open(os.path.join(tmp_dir, f"fs_{rank}.txt"), "w") as fh:
            fh.write(repr(runtime.is_shared_filesystem()))
    finally:
        runtime.reset_shared_filesystem_consensus()
        dist.destroy_process_group()


def _split_side_fs_flags_worker(rank: int, tmp_dir: str, port: int) -> None:
    """Ranks disagree on the two SIDE flags, with the umbrella unset on both."""
    os.environ.update(
        MASTER_ADDR="127.0.0.1",
        MASTER_PORT=str(port),
        RANK=str(rank),
        WORLD_SIZE=str(WORLD_SIZE),
        LOCAL_RANK=str(rank),
        LOCAL_WORLD_SIZE=str(WORLD_SIZE),
        DIST_INPUT_SHARED_FILESYSTEM="1" if rank == 0 else "0",
        DIST_OUTPUT_SHARED_FILESYSTEM="0" if rank == 0 else "1",
    )
    os.environ.pop("DIST_SHARED_FILESYSTEM", None)
    dist.init_process_group("gloo", rank=rank, world_size=WORLD_SIZE)
    try:
        runtime.resolve_shared_filesystem_consensus()
        with open(os.path.join(tmp_dir, f"sides_{rank}.txt"), "w") as fh:
            fh.write(f"{runtime.is_input_shared_filesystem()},{runtime.is_output_shared_filesystem()}")
    finally:
        runtime.reset_shared_filesystem_consensus()
        dist.destroy_process_group()


def test_store_main_first_orders_and_releases(tmp_path):
    mp.start_processes(_worker, args=(str(tmp_path), free_port()), nprocs=WORLD_SIZE, join=True, start_method="spawn")
    for rank in range(WORLD_SIZE):
        with open(tmp_path / f"result_{rank}.txt") as fh:
            result = fh.read()
        assert result == "PASS", f"rank {rank}: {result}"


def test_main_body_may_outlive_the_process_group_timeout(tmp_path):
    """A main-rank body longer than the PG timeout must NOT fail the waiters.

    Fails when the waiters block in ``dist.barrier()``: gloo aborts the recv at PG_TIMEOUT_SEC and
    both ranks raise — the 30-min NCCL-watchdog kill of a real download, scaled down.
    """
    mp.start_processes(
        _watchdog_worker, args=(str(tmp_path), free_port()), nprocs=WORLD_SIZE, join=True, start_method="spawn"
    )
    for rank in range(WORLD_SIZE):
        with open(tmp_path / f"watchdog_{rank}.txt") as fh:
            result = fh.read()
        assert result == "PASS", f"rank {rank}: {result}"


def test_non_shared_filesystem_scopes_coordination_per_node(tmp_path):
    """On a non-shared FS each node coordinates only within itself — nodes run in parallel.

    Each node needs its own copy of the artifact, so serializing them across the whole world turns
    an N-node launch into N sequential downloads. Both 1-rank nodes must enter their body at once.
    """
    mp.start_processes(
        _non_shared_fs_worker, args=(str(tmp_path), free_port()), nprocs=WORLD_SIZE, join=True, start_method="spawn"
    )
    entered = []
    for rank in range(WORLD_SIZE):
        with open(tmp_path / f"entered_{rank}.txt") as fh:
            entered.append(float(fh.read()))
    assert abs(entered[0] - entered[1]) < 0.5, f"nodes were serialized against each other: {entered}"


def test_shared_filesystem_flag_is_agreed_across_ranks(tmp_path):
    """A split DIST_SHARED_FILESYSTEM must resolve to ONE value, rank 0's.

    The flag picks the coordination scope, so a rank reading `0` while its peers read `1` sits under
    `node{n}` while they sit under `shared` — they never see each other's keys and each waits out the
    full store timeout. Divergence is realistic (per-node env injection, a heterogeneous --env-file).
    """
    mp.start_processes(
        _split_fs_flag_worker, args=(str(tmp_path), free_port()), nprocs=WORLD_SIZE, join=True, start_method="spawn"
    )
    resolved = []
    for rank in range(WORLD_SIZE):
        with open(tmp_path / f"fs_{rank}.txt") as fh:
            resolved.append(fh.read())
    assert resolved[0] == resolved[1] == "True", f"ranks disagreed on the coordination scope: {resolved}"


def test_side_shared_filesystem_flags_are_agreed_across_ranks(tmp_path):
    """The consensus must agree the INPUT and OUTPUT flags, not just the umbrella.

    Agreeing only ``DIST_SHARED_FILESYSTEM`` leaves each rank on its own side flags, and the sides
    are what actually elect writers: ``fs_aware_load_rank`` reads input, ``fs_aware_save_rank``
    reads output. Split those and two nodes both believe they are the sole checkpoint writer while
    a third re-maps the corpus — with no error. The umbrella-only test above passes either way, so
    this is the case that pins the split.
    """
    mp.start_processes(
        _split_side_fs_flags_worker,
        args=(str(tmp_path), free_port()),
        nprocs=WORLD_SIZE,
        join=True,
        start_method="spawn",
    )
    resolved = []
    for rank in range(WORLD_SIZE):
        with open(tmp_path / f"sides_{rank}.txt") as fh:
            resolved.append(fh.read())
    assert resolved[0] == resolved[1], f"ranks disagreed on the input/output scope: {resolved}"
    # Rank 0's verdict wins, so the agreed pair must be rank 0's: input shared, output not.
    assert resolved[0] == "True,False", f"the consensus did not adopt rank 0's side flags: {resolved[0]}"


def test_no_distributed_is_passthrough():
    with fs_aware_main_first("noop"):
        pass  # must not require an initialized process group


@pytest.fixture()
def fake_world(monkeypatch):
    """Simulate one rank of a 2-rank shared-FS world against a shared in-memory store."""
    store = FakeStore()
    monkeypatch.setattr(dist, "is_available", lambda: True)
    monkeypatch.setattr(dist, "is_initialized", lambda: True)
    monkeypatch.setattr(runtime.c10d, "_get_default_store", lambda: store)
    monkeypatch.setattr(filesystem, "get_global_world_size", lambda: 2)
    monkeypatch.setenv("DIST_SHARED_FILESYSTEM", "1")
    monkeypatch.setattr(runtime, "_SHARED_FILESYSTEM_CONSENSUS", None)
    return store


def _enter_as(monkeypatch, rank: int, tag: str = "dl") -> None:
    monkeypatch.setattr(filesystem, "get_global_rank", lambda: rank)
    monkeypatch.setattr(filesystem, "is_global_main_process", lambda: rank == 0)
    with filesystem.fs_aware_main_first(tag):
        pass


def test_entry_count_divergence_is_detected_when_a_waiter_runs_ahead(fake_world, monkeypatch):
    """A rank that enters a tag more often than its peers must RAISE with the cause named.

    The phase number is each participant's private entry count, so an extra entry puts that rank
    permanently off-by-one: it then waits on a phase key its peers never write. Left undiagnosed
    that is an unattributed multi-hour stall, so the wait must name the tag, the knob, and the
    entry-count/scope divergence.
    """
    _enter_as(monkeypatch, 0)
    _enter_as(monkeypatch, 1)  # round 1 complete and collected
    with pytest.raises(RuntimeError) as excinfo:
        _enter_as(monkeypatch, 1)  # rank 1 enters a second time; rank 0 has not
    message = str(excinfo.value)
    assert "main_first/dl" in message, f"the tag must be named: {message}"
    assert "entered this tag" in message, f"the entry-count divergence must be named: {message}"


def test_matched_entry_counts_stay_silent(fake_world, monkeypatch):
    """The detector must not fire on the healthy pattern it has to coexist with: the main rank never
    waits for the waiters, so repeated rounds of one tag are legitimate as long as counts match."""
    for _ in range(3):
        _enter_as(monkeypatch, 0)
        _enter_as(monkeypatch, 1)


def test_store_timeout_is_configurable(monkeypatch):
    """The store wait is an env knob, and hours-scale — not the NCCL-watchdog scale, and not a
    hardcoded literal."""
    monkeypatch.delenv("DIST_STORE_TIMEOUT_HOURS", raising=False)
    assert get_store_timeout() == datetime.timedelta(hours=DEFAULT_STORE_TIMEOUT_HOURS)
    # Pinned against the value, not just against itself: the default must stay long enough to wait
    # out a 100B+ download, which is the entire reason this knob is separate from the NCCL timeout.
    assert get_store_timeout() >= datetime.timedelta(hours=1)
    monkeypatch.setenv("DIST_STORE_TIMEOUT_HOURS", "9")
    assert get_store_timeout() == datetime.timedelta(hours=9)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
