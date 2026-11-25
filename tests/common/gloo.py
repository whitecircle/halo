"""Spawned gloo ranks for CPU tests needing a real process group; a leaf, since every spawned rank re-imports it."""

import datetime
import faulthandler
import os
import signal
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

import torch.distributed as dist
import torch.multiprocessing as mp

from tests.common.ports import free_port

# A run in which no rank used _WORKING_CPU_SHARE of a core over this long is killed as hung. A wall-clock bound alone
# would kill healthy ranks on a starved host, where a run takes minutes longer. Twice the longest stretch a test stages
# with every rank idle, the 30 s hang of test_reentrant_tower_gradient.py.
GLOO_STALL_TIMEOUT_S = 60.0
# A rank counts as working only above this share of a core, so a healthy run must keep one rank above it, on its own
# CPU, in every GLOO_STALL_TIMEOUT_S: ranks that take turns or wait on child processes can fall under it on a starved
# host. A rank waiting in a collective, a store wait or a lock uses less, not none: each gloo group it holds runs an
# event loop that wakes about 100 times a second, about 0.14% of a core. A rank holding eight groups or more stays
# above the share while it waits, so its own collective timeout or GLOO_RUN_CEILING_S ends its hang.
_WORKING_CPU_SHARE = 0.01
# A run still going this long is killed whatever its CPU, so a rank that spins or polls without end cannot hold its
# xdist worker. Over twice the slowest healthy run measured: the reentrant FSDP2 fixture, 27 minutes on a starved host.
GLOO_RUN_CEILING_S = 3600.0
# How often the ranks' CPU time is read while they run.
_STALL_POLL_S = 1.0
# How long a hung rank gets to print its stacks before the next one is asked, or the kill.
_STACK_DUMP_S = 0.5
# Bounds the store's connect and untimed ops, among them the barrier that waits out the ranks' spawn skew on a loaded
# host.
STORE_TIMEOUT = datetime.timedelta(seconds=420)
# Set by a rank on reaching a step every rank must reach before any goes past it.
_STEP_KEY = "run_gloo_ranks/{step}/{rank}"
# What torchrun exports when a launch sets none. Without it accelerate sizes a gloo rank's pool past one thread, and on
# a loaded xdist host two identical forwards in one rank can disagree.
TORCHRUN_OMP_NUM_THREADS = "1"


def _cpu_ticks(pid: int) -> int | None:
    """User plus system CPU ``pid`` has used, in clock ticks; ``None`` once it is reaped."""
    try:
        stat = Path(f"/proc/{pid}/stat").read_text()
    except OSError:
        return None
    # utime and stime are fields 14 and 15; the command name before them may hold spaces, so split past its ")".
    utime, stime = stat.rsplit(")", 1)[1].split()[11:13]
    return int(utime) + int(stime)


def _ticks_by(samples: Sequence[tuple[float, Sequence[int | None]]], rank: int, by: float) -> int:
    """``rank``'s CPU ticks at its last reading no later than ``by``."""
    return next(ticks[rank] for when, ticks in reversed(samples) if when <= by and ticks[rank] is not None)


def _hang_reason(samples: Sequence[tuple[float, Sequence[int | None]]]) -> str | None:
    """Why the run counts as hung at its last sample, else ``None``: no rank used :data:`_WORKING_CPU_SHARE` of a core
    over the last :data:`GLOO_STALL_TIMEOUT_S`, or the run reached :data:`GLOO_RUN_CEILING_S`. A sample is a time and
    each rank's own CPU ticks, its children's never counted; the first is the spawn, at zero. An exited rank reads
    ``None`` and keeps its last count, so the work it did before exiting still counts."""
    at = samples[-1][0]
    if at - samples[0][0] >= GLOO_RUN_CEILING_S:
        return f"still running at the {GLOO_RUN_CEILING_S}s ceiling, far past the slowest healthy run"
    if at - samples[0][0] < GLOO_STALL_TIMEOUT_S:
        return None
    floor = _WORKING_CPU_SHARE * GLOO_STALL_TIMEOUT_S * os.sysconf("SC_CLK_TCK")
    window_start = at - GLOO_STALL_TIMEOUT_S
    if any(
        _ticks_by(samples, rank, at) - _ticks_by(samples, rank, window_start) >= floor
        for rank in range(len(samples[0][1]))
    ):
        return None
    return (
        f"used under {_WORKING_CPU_SHARE:.0%} of a core for {GLOO_STALL_TIMEOUT_S}s: waiting in a collective, a store "
        f"wait, a lock or a poll that no peer will end, or crashed and not yet exited"
    )


def _all_ranks_reach(store: dist.Store, step: str, rank: int, nprocs: int) -> None:
    """Return once every rank has reached ``step``: a barrier over the store, bounded by its timeout."""
    store.set(_STEP_KEY.format(step=step, rank=rank), "1")
    store.wait([_STEP_KEY.format(step=step, rank=peer) for peer in range(nprocs)])


def _gloo_rank(
    rank: int,
    worker: Callable[..., object],
    nprocs: int,
    port: int,
    pg_timeout: datetime.timedelta | None,
    env: dict[str, str],
    args: tuple,
) -> None:
    # A rank that crashes on a fatal signal, or that the harness is about to kill as hung, prints its threads' stacks.
    faulthandler.enable()
    faulthandler.register(signal.SIGUSR1, all_threads=True)
    os.environ.update(
        MASTER_ADDR="127.0.0.1",
        MASTER_PORT=str(port),
        RANK=str(rank),
        WORLD_SIZE=str(nprocs),
        LOCAL_RANK=str(rank),
        LOCAL_WORLD_SIZE=str(nprocs),
    )
    os.environ.update(env)
    # STORE_TIMEOUT in place of pg_timeout.
    store = dist.TCPStore("127.0.0.1", port, is_master=False, timeout=STORE_TIMEOUT)
    # Spawn and import skew is waited out here under STORE_TIMEOUT, not in init's rendezvous, which ``pg_timeout``
    # bounds.
    _all_ranks_reach(store, "joined", rank, nprocs)
    timeout_kwargs = {} if pg_timeout is None else {"timeout": pg_timeout}
    dist.init_process_group("gloo", rank=rank, world_size=nprocs, store=store, **timeout_kwargs)
    try:
        # gloo's init returns on one rank while a peer is still connecting, and a peer whose pair closes mid-connect
        # fails its init. So no worker starts before every rank is through init.
        _all_ranks_reach(store, "initialized", rank, nprocs)
        worker(rank, *args)
    finally:
        dist.destroy_process_group()


def new_groups(rank: int, rank_lists: list[list[int]], timeout: datetime.timedelta) -> dist.ProcessGroup | None:
    """Create a group for every list in ``rank_lists`` and return the one holding ``rank``. COLLECTIVE, same lists in
    the same order. ``dist.new_group`` without ``timeout`` gets gloo's 30-minute default, so a stuck collective would
    raise nothing before the harness kills the run."""
    mine = None
    for ranks in rank_lists:
        group = dist.new_group(ranks, timeout=timeout)
        if rank in ranks:
            mine = group
    return mine


def run_gloo_ranks(
    worker: Callable[..., object],
    nprocs: int,
    *args,
    pg_timeout: datetime.timedelta | None = None,
    env: Mapping[str, str] | None = None,
) -> None:
    """Run ``worker(rank, *args)`` in ``nprocs`` spawned processes sharing one gloo group, with torchrun's env plus
    ``env``. ``worker`` must be module-level, since spawn pickles it by reference. A rank that raises or dies fails
    this call once peers stop. ``OMP_NUM_THREADS`` in ``env`` lands too late for OpenMP. The ranks are killed with
    ``TimeoutError`` once none has used :data:`_WORKING_CPU_SHARE` of a core over :data:`GLOO_STALL_TIMEOUT_S`, or
    once the run reaches :data:`GLOO_RUN_CEILING_S`."""
    # Hosted here so the store outlives every rank: a rank may exit while a peer still has store traffic to send.
    store = dist.TCPStore("127.0.0.1", free_port(), is_master=True, timeout=STORE_TIMEOUT, wait_for_workers=False)
    # By key rather than patch.dict, which rewrites the whole environment under the store's live thread.
    exports_omp = nprocs > 1 and "OMP_NUM_THREADS" not in os.environ
    if exports_omp:
        os.environ["OMP_NUM_THREADS"] = TORCHRUN_OMP_NUM_THREADS
    try:
        context = mp.start_processes(
            _gloo_rank,
            args=(worker, nprocs, store.port, pg_timeout, dict(env or {}), args),
            nprocs=nprocs,
            join=False,
            start_method="spawn",
        )
    finally:
        if exports_omp:
            del os.environ["OMP_NUM_THREADS"]
    samples: list[tuple[float, list[int | None]]] = [(time.monotonic(), [0] * nprocs)]
    while not context.join(timeout=_STALL_POLL_S):
        # An exited rank is not read: once joined, its pid may name another process.
        ticks = [None if process.exitcode is not None else _cpu_ticks(process.pid) for process in context.processes]
        samples.append((time.monotonic(), ticks))
        reason = _hang_reason(samples)
        if reason is not None:
            stuck = [rank for rank, process in enumerate(context.processes) if process.is_alive()]
            for process in context.processes:
                if process.is_alive():
                    # One rank at a time: stacks written together interleave on the stderr the ranks share.
                    os.kill(process.pid, signal.SIGUSR1)
                    time.sleep(_STACK_DUMP_S)
            for process in context.processes:
                if process.is_alive():
                    process.kill()
                process.join()
            raise TimeoutError(f"{worker.__qualname__}: rank(s) {stuck} of {nprocs} {reason}; killed")
