#!/usr/bin/env python
"""``store_reject_across_ranks`` — the failure-carrying join over the c10d store, not a collective.

The coordinated dataset ops join their two phases with this: the writer rank's map is unbounded
single-rank work (a fresh-cache tokenization of a large corpus), and a join spelled
``all_gather_object`` parks the peers INSIDE the process group for that whole duration — a map
longer than ``DIST_NCCL_TIMEOUT_MINUTES`` then kills the run with a watchdog abort blaming the
collective, not the map. The store transport keeps the ``reject_across_ranks`` contract (uniform raise, real cause,
caller's ``exc_type``) while bounding the wait by ``DIST_STORE_TIMEOUT_HOURS``.

Proven on real gloo groups: (1) a clean join passes on both ranks; (2) one rank's reason raises on
EVERY rank, naming the failing rank and cause; (3) **a rank may arrive later than the process-group
timeout** and the early rank still completes — the property the transport exists for, which
``all_gather_object`` cannot provide; (4) ``exc_type`` is honored. FakeStore paths pin the verdict
logic and the empty-reason sentinel single-process.

    python tests/cpu/parallelism/test_store_reject.py
"""

import contextlib
import datetime
import os
import time

import pytest
import torch.distributed as dist
import torch.multiprocessing as mp

from src.distributed import filesystem, runtime
from src.distributed.filesystem import store_reject_across_ranks
from tests.common.distributed import FakeStore
from tests.common.ports import free_port

WORLD_SIZE = 2

# The late rank must be allowed to outlive the process-group timeout: that is the whole point of
# joining on the store. Kept far apart so a slow CI box cannot blur the two, and the PG timeout
# itself wide enough that spawn + torch-import skew between the ranks on a loaded host cannot trip
# the rendezvous before the join is even reached.
PG_TIMEOUT_SEC = 5
LATE_JOIN_SEC = 8

# Every store wait is explicitly bounded: the 4 h production default would turn a regression into a
# CI stall rather than a failure (mp.start_processes(join=True) has no timeout).
WAIT_TIMEOUT = datetime.timedelta(seconds=30)


def _worker(rank: int, tmp_dir: str, port: int, failing_rank: int) -> None:
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port), RANK=str(rank), WORLD_SIZE=str(WORLD_SIZE))
    dist.init_process_group("gloo", rank=rank, world_size=WORLD_SIZE)
    try:
        reason = "ValueError: boom" if rank == failing_rank else None
        try:
            store_reject_across_ranks("t", reason, "unit-test join", timeout=WAIT_TIMEOUT)
            result = "NO RAISE"
        except RuntimeError as e:
            result = f"RuntimeError: {e}"
        with open(os.path.join(tmp_dir, f"result_{rank}.txt"), "w") as fh:
            fh.write(result)
    finally:
        dist.destroy_process_group()


def _late_rank_worker(rank: int, tmp_dir: str, port: int) -> None:
    """Rank 0 arrives at the join LATE_JOIN_SEC late, past the PG timeout; both must still pass."""
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port), RANK=str(rank), WORLD_SIZE=str(WORLD_SIZE))
    dist.init_process_group(
        "gloo", rank=rank, world_size=WORLD_SIZE, timeout=datetime.timedelta(seconds=PG_TIMEOUT_SEC)
    )
    try:
        if rank == 0:
            time.sleep(LATE_JOIN_SEC)
        store_reject_across_ranks("late", None, "late-join", timeout=datetime.timedelta(seconds=60))
        result = "PASS"
    except Exception as e:  # the failure mode under test is exactly "a join that raises"
        result = f"FAIL: {type(e).__name__}: {str(e).splitlines()[0][:160]}"
    with open(os.path.join(tmp_dir, f"late_{rank}.txt"), "w") as fh:
        fh.write(result)
    with contextlib.suppress(Exception):
        dist.destroy_process_group()


def test_clean_join_passes_on_both_ranks(tmp_path):
    mp.start_processes(
        _worker, args=(str(tmp_path), free_port(), -1), nprocs=WORLD_SIZE, join=True, start_method="spawn"
    )
    for rank in range(WORLD_SIZE):
        with open(tmp_path / f"result_{rank}.txt") as fh:
            assert fh.read() == "NO RAISE", f"rank {rank} raised on a clean join"


def test_one_ranks_reason_raises_on_every_rank_with_the_cause(tmp_path):
    mp.start_processes(
        _worker, args=(str(tmp_path), free_port(), 1), nprocs=WORLD_SIZE, join=True, start_method="spawn"
    )
    for rank in range(WORLD_SIZE):
        with open(tmp_path / f"result_{rank}.txt") as fh:
            result = fh.read()
        assert result != "NO RAISE", f"rank {rank} sailed past a failed join"
        assert "rank 1" in result and "boom" in result, f"rank {rank} was not told the real cause: {result}"


def test_late_rank_may_outlive_the_process_group_timeout(tmp_path):
    """A rank arriving later than the PG timeout must NOT fail the early rank.

    Fails when the join is a collective: gloo aborts the early rank's recv at PG_TIMEOUT_SEC — the
    30-min NCCL-watchdog kill of a real fresh-cache map, scaled down.
    """
    mp.start_processes(
        _late_rank_worker, args=(str(tmp_path), free_port()), nprocs=WORLD_SIZE, join=True, start_method="spawn"
    )
    for rank in range(WORLD_SIZE):
        with open(tmp_path / f"late_{rank}.txt") as fh:
            result = fh.read()
        assert result == "PASS", f"rank {rank}: {result}"


@pytest.fixture()
def fake_world(monkeypatch):
    """One rank of a 2-rank world against a shared in-memory store, peer keys pre-seedable."""
    store = FakeStore()
    monkeypatch.setattr(dist, "is_available", lambda: True)
    monkeypatch.setattr(dist, "is_initialized", lambda: True)
    monkeypatch.setattr(dist, "get_world_size", lambda: 2)
    monkeypatch.setattr(runtime.c10d, "_get_default_store", lambda: store)
    monkeypatch.setattr(filesystem, "get_global_rank", lambda: 0)
    monkeypatch.setattr(filesystem, "get_global_world_size", lambda: 2)
    return store


def _seed_peer_reason(store: FakeStore, tag: str, reason: str) -> None:
    # Rank 0's next entry of the tag is its first → phase 1; the peer's key lives under that prefix.
    store.kv[f"_halo_phase/reject/{tag}/world/p1/reason1"] = reason


def test_verdict_names_the_failing_rank_and_honors_exc_type(fake_world):
    _seed_peer_reason(fake_world, "vt", "OSError: disk full")
    with pytest.raises(ValueError, match=r"rank 1.*disk full"):
        store_reject_across_ranks("vt", None, "unit join", exc_type=ValueError, timeout=WAIT_TIMEOUT)


def test_empty_peer_reason_is_no_failure(fake_world):
    """The store carries strings, so None rides as "" — an empty reason must read as success, the
    same falsy contract reject_across_ranks has always had."""
    _seed_peer_reason(fake_world, "et", "")
    store_reject_across_ranks("et", None, "unit join", timeout=WAIT_TIMEOUT)


def test_own_reason_raises_even_when_peers_are_clean(fake_world):
    _seed_peer_reason(fake_world, "ot", "")
    with pytest.raises(RuntimeError, match=r"rank 0.*KeyError: 'prompt'"):
        store_reject_across_ranks("ot", "KeyError: 'prompt'", "unit join", timeout=WAIT_TIMEOUT)


def test_every_ranks_reason_is_read_in_one_round_trip(fake_world):
    """One multi-key read, not one request per rank.

    Reading the world's reason keys one at a time is O(world) round-trips on EVERY rank — O(world²)
    sequential requests through the single store server, ~262k for one join at world=512, and the
    coordinated dataset ops issue several joins per operation before the first training step.
    """
    _seed_peer_reason(fake_world, "mg", "")

    store_reject_across_ranks("mg", None, "unit join", timeout=WAIT_TIMEOUT)

    prefix = "_halo_phase/reject/mg/world/p1"
    assert fake_world.multi_get_keys == [[f"{prefix}/reason0", f"{prefix}/reason1"]]
    assert fake_world.get_keys == [], "a per-key read makes the join O(world) round-trips per rank"


def test_not_distributed_raises_locally_and_passes_clean():
    with pytest.raises(ValueError, match="local cause"):
        store_reject_across_ranks("nd", "local cause", "unit join", exc_type=ValueError)
    store_reject_across_ranks("nd", None, "unit join")


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
