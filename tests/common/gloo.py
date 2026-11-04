"""Spawned gloo ranks for CPU tests needing a real process group; a leaf, since every spawned rank re-imports it."""

import datetime
import os
import time
from collections.abc import Callable, Mapping

import torch.distributed as dist
import torch.multiprocessing as mp

from tests.common.ports import free_port

# End-to-end bound on a spawned run. The margin absorbs an 8-way xdist host starving the rendezvous, so only a rank
# stuck in a collective or a store wait reaches it.
GLOO_JOIN_TIMEOUT_S = 420.0
# Bounds the store's connect and untimed ops. Held to the join bound: a short collective bound would fail the connect
# on a loaded host.
STORE_TIMEOUT = datetime.timedelta(seconds=GLOO_JOIN_TIMEOUT_S)
# Set by a rank on reaching a step every rank must reach before any goes past it.
_STEP_KEY = "run_gloo_ranks/{step}/{rank}"
# What torchrun exports when a launch sets none. Without it accelerate sizes a gloo rank's pool past one thread, and on
# a loaded xdist host two identical forwards in one rank can disagree.
TORCHRUN_OMP_NUM_THREADS = "1"


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
    the same order. ``dist.new_group`` without ``timeout`` gets gloo's 30-minute default, so a stuck rank would wait
    out the spawn bound."""
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
    this call once peers stop. ``OMP_NUM_THREADS`` in ``env`` lands too late for OpenMP. Ranks past
    :data:`GLOO_JOIN_TIMEOUT_S` are killed with ``TimeoutError``."""
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
    deadline = time.monotonic() + GLOO_JOIN_TIMEOUT_S
    while not context.join(timeout=max(0.0, deadline - time.monotonic())):
        if time.monotonic() >= deadline:
            stuck = [rank for rank, process in enumerate(context.processes) if process.is_alive()]
            for process in context.processes:
                if process.is_alive():
                    process.kill()
                process.join()
            raise TimeoutError(
                f"{worker.__qualname__}: rank(s) {stuck} of {nprocs} still running {GLOO_JOIN_TIMEOUT_S}s after "
                f"the spawn — stuck in a collective or a store wait; killed"
            )
