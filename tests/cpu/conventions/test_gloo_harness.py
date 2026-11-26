#!/usr/bin/env python
"""``tests.common.gloo.run_gloo_ranks`` keeps the rendezvous store up until every rank has exited, spawns each rank
with the ``OMP_NUM_THREADS`` torchrun would, and kills a run once no rank works or at its ceiling. A store dying with an
early-exiting rank resets its peer's connection, a pool ``PartialState`` resized lets two forwards disagree on a loaded
xdist host, a wall-clock bound kills a healthy run that the host starves, and a rank waiting in gloo still uses CPU."""

import datetime
import fcntl
import json
import os
import subprocess
import sys
import threading
import time
from collections.abc import Callable
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
from accelerate import PartialState
from torch.distributed import distributed_c10d as c10d

from tests.common import gloo
from tests.common.gloo import run_gloo_ranks

WORLD_SIZE = 2
# Measured in this image: a rank waiting in gloo past init gains a CPU tick every 6 to 8 s, and a working rank, one of
# 48 on 2 CPUs, about four a second. The starved series works at half that.
_WAITING_SHARE = 0.01 / 6
_STARVED_SHARE = 0.02
# When the series' ranks stop working.
_WORK_END_S = 10


def _store_after_rank0_exits(rank: int, tmp_dir: str) -> None:
    """Rank 0 holds a lock only its process exit releases; rank 1 takes it, then sets and reads a key."""
    store = c10d._get_default_store()
    # A raw descriptor, never closed: the lock lasts exactly as long as the process holding it.
    lock = os.open(os.path.join(tmp_dir, "rank0.lock"), os.O_RDWR | os.O_CREAT)
    if rank == 0:
        fcntl.flock(lock, fcntl.LOCK_EX)
        store.set("rank0_locked", "1")
        return
    store.wait(["rank0_locked"])
    fcntl.flock(lock, fcntl.LOCK_EX)
    store.set("after_rank0_exit", "served")
    with open(os.path.join(tmp_dir, "rank1.txt"), "w") as fh:
        fh.write(store.get("after_rank0_exit").decode())


def test_the_store_outlives_the_first_rank_to_exit(tmp_path):
    run_gloo_ranks(_store_after_rank0_exits, WORLD_SIZE, str(tmp_path))
    assert (tmp_path / "rank1.txt").read_text() == "served"


def _thread_counts(rank: int, tmp_dir: str) -> None:
    """The intra-op thread count before and after accelerate's process state, and the OMP_NUM_THREADS the rank sees."""
    before = torch.get_num_threads()
    PartialState()
    record = {"before": before, "after": torch.get_num_threads(), "omp": os.environ.get("OMP_NUM_THREADS")}
    Path(tmp_dir, f"rank{rank}.json").write_text(json.dumps(record))


@pytest.mark.parametrize(("exported", "threads"), [(None, 1), ("3", 3)], ids=["torchrun_default", "caller_export"])
def test_each_rank_starts_with_torchrun_omp_threads_and_partial_state_keeps_them(
    tmp_path, monkeypatch, exported, threads
):
    if exported is None:
        monkeypatch.delenv("OMP_NUM_THREADS", raising=False)
    else:
        monkeypatch.setenv("OMP_NUM_THREADS", exported)
    run_gloo_ranks(_thread_counts, WORLD_SIZE, str(tmp_path))
    records = [json.loads((tmp_path / f"rank{rank}.json").read_text()) for rank in range(WORLD_SIZE)]
    assert records == [{"before": threads, "after": threads, "omp": str(threads)}] * WORLD_SIZE, records
    assert os.environ.get("OMP_NUM_THREADS") == exported, "the spawn's export leaked into the calling process"


def _rank(
    share: float, *, until: int | None = None, then: float | None = _WAITING_SHARE
) -> Callable[[int], float | None]:
    """The CPU seconds a rank has used ``t`` seconds after the spawn: ``share`` of a core, from ``until`` on ``then``
    (``None``: exited)."""

    def cpu_s(t: int) -> float | None:
        if until is None or t <= until:
            return share * t
        return None if then is None else share * until + then * (t - until)

    return cpu_s


def _first_hang(*ranks: Callable[[int], float | None], seconds: int) -> tuple[int, str] | None:
    """The first second the harness's poll would call the run hung, and why: one sample a second from the spawn."""
    hz = os.sysconf("SC_CLK_TCK")
    samples: list[tuple[float, list[int | None]]] = [(0.0, [0] * len(ranks))]
    for t in range(1, seconds + 1):
        samples.append((float(t), [None if (cpu := rank(t)) is None else int(cpu * hz) for rank in ranks]))
        reason = gloo._hang_reason(samples)
        if reason is not None:
            return t, reason
    return None


@pytest.mark.parametrize(
    "peer",
    [_rank(1.0, until=_WORK_END_S), _rank(1.0, until=_WORK_END_S, then=None)],
    ids=["peer_waits_too", "peer_exited"],
)
def test_a_run_whose_ranks_wait_in_gloo_is_hung_one_stall_bound_after_the_work_ends(peer):
    """gloo's event loop keeps a waiting rank ticking, so a rule counting any CPU at all never ends the run."""
    hang = _first_hang(_rank(1.0, until=_WORK_END_S), peer, seconds=600)
    assert hang is not None, "ranks waiting in gloo were never called hung"
    killed_at, reason = hang
    assert _WORK_END_S + gloo.GLOO_STALL_TIMEOUT_S <= killed_at <= _WORK_END_S + gloo.GLOO_STALL_TIMEOUT_S + 1
    assert reason.startswith(f"used under 1% of a core for {gloo.GLOO_STALL_TIMEOUT_S}s")


def test_ranks_found_exited_keep_the_work_they_did_before_exiting():
    """The poll can read every rank exited before the join reports them; their last minute of work still counts."""
    exits_at = int(gloo.GLOO_STALL_TIMEOUT_S) + _WORK_END_S
    assert _first_hang(*[_rank(1.0, until=exits_at, then=None)] * 2, seconds=exits_at + 1) is None


@pytest.mark.parametrize(
    "peer", [_rank(_STARVED_SHARE), _rank(_WAITING_SHARE)], ids=["peer_starved_too", "peer_waits_on_it"]
)
def test_a_starved_rank_still_working_runs_until_the_ceiling(peer):
    """A wall-clock bound short of the ceiling, or a working floor above what a starved host leaves a rank, ends the
    run first."""
    hang = _first_hang(_rank(_STARVED_SHARE), peer, seconds=int(gloo.GLOO_RUN_CEILING_S) + 1)
    assert hang is not None, "a run past the ceiling was never killed"
    killed_at, reason = hang
    assert killed_at == gloo.GLOO_RUN_CEILING_S
    assert reason.startswith(f"still running at the {gloo.GLOO_RUN_CEILING_S}s ceiling")


def _recv_from_a_peer_that_polls_without_end(rank: int, tmp_dir: str) -> None:
    """Rank 0 waits in a collective on rank 1, which polls for an event no one sets at about half a percent of a core:
    under the working floor, yet a CPU tick every two or three seconds."""
    (Path(tmp_dir) / f"pid_{rank}").write_text(str(os.getpid()))
    if rank == 0:
        dist.recv(torch.zeros(1), src=1)
        return
    never = threading.Event()
    while not never.wait(0.25):
        spun_from = time.thread_time()
        while time.thread_time() - spun_from < 7e-4:
            pass


def test_ranks_waiting_on_a_peer_that_polls_without_end_are_killed_at_the_stall_bound(monkeypatch, capfd, tmp_path):
    """The poller ticks more often than the bound, so a rule counting any CPU at all never sees the run stop and the
    collective times out instead. The kill prints where each rank stood."""
    monkeypatch.setattr(gloo, "GLOO_STALL_TIMEOUT_S", 4.0)
    with pytest.raises(TimeoutError, match=r"rank\(s\) \[0, 1\] of 2 used under 1% of a core for 4.0s"):
        run_gloo_ranks(
            _recv_from_a_peer_that_polls_without_end,
            WORLD_SIZE,
            str(tmp_path),
            pg_timeout=datetime.timedelta(seconds=30),
        )
    for rank in range(WORLD_SIZE):
        with pytest.raises(ProcessLookupError):
            os.kill(int((tmp_path / f"pid_{rank}").read_text()), 0)
    assert capfd.readouterr().err.count("in _recv_from_a_peer_that_polls_without_end") == WORLD_SIZE, (
        "one stack per rank"
    )


def test_the_tick_reader_counts_the_cpu_a_process_used_and_reads_none_once_it_is_reaped():
    hz = os.sysconf("SC_CLK_TCK")
    before, spun_from = gloo._cpu_ticks(os.getpid()), time.process_time()
    while time.process_time() - spun_from < 0.5:
        pass
    gained, used = gloo._cpu_ticks(os.getpid()) - before, time.process_time() - spun_from
    assert abs(gained - used * hz) <= 2, (gained, used)
    child = subprocess.Popen([sys.executable, "-c", "pass"])
    child.wait()
    assert gloo._cpu_ticks(child.pid) is None


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
