"""Spawned gloo ranks for CPU tests that need a real multi-rank process group.

A leaf on purpose: every spawned rank re-imports this module, so it pulls in nothing past torch.
"""

import datetime
import os
import time
from collections.abc import Callable, Mapping

import torch.distributed as dist
import torch.multiprocessing as mp

from tests.common.ports import free_port

# End-to-end bound on a spawned gloo run: spawn, imports and the worker's own bounded waits. The margin
# absorbs an 8-way xdist host starving the rendezvous, so only a rank stuck in a collective or a store
# wait reaches it.
GLOO_JOIN_TIMEOUT_S = 420.0
# The rendezvous store's own timeout, which bounds its connect and any store op not given one. Held to
# the join bound rather than ``pg_timeout``: a short collective bound would fail the connect on a
# loaded host before the test reaches what it measures.
STORE_TIMEOUT = datetime.timedelta(seconds=GLOO_JOIN_TIMEOUT_S)
# The store key a rank sets on reaching a step every rank must reach before any goes past it.
_STEP_KEY = "run_gloo_ranks/{step}/{rank}"


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
    """One spawned rank of :func:`run_gloo_ranks`: the launcher env, then the group around ``worker``."""
    os.environ.update(
        MASTER_ADDR="127.0.0.1",
        MASTER_PORT=str(port),
        RANK=str(rank),
        WORLD_SIZE=str(nprocs),
        LOCAL_RANK=str(rank),
        LOCAL_WORLD_SIZE=str(nprocs),
    )
    os.environ.update(env)
    # A client of the store :func:`run_gloo_ranks` hosts (what ``env://`` builds under torchrun's agent
    # store), with STORE_TIMEOUT in place of pg_timeout.
    store = dist.TCPStore("127.0.0.1", port, is_master=False, timeout=STORE_TIMEOUT)
    # The spawn and import skew between ranks is waited out here, under STORE_TIMEOUT, rather than in
    # init's rendezvous, which ``pg_timeout`` bounds.
    _all_ranks_reach(store, "joined", rank, nprocs)
    timeout_kwargs = {} if pg_timeout is None else {"timeout": pg_timeout}
    dist.init_process_group("gloo", rank=rank, world_size=nprocs, store=store, **timeout_kwargs)
    try:
        # gloo's init returns on one rank while a peer is still connecting to it, and a peer whose pair is
        # closed mid-connect fails its init. A worker without a collective lets a rank reach the teardown
        # below at once, so no worker starts before every rank is through init.
        _all_ranks_reach(store, "initialized", rank, nprocs)
        worker(rank, *args)
    finally:
        dist.destroy_process_group()


def new_groups(rank: int, rank_lists: list[list[int]], timeout: datetime.timedelta) -> dist.ProcessGroup | None:
    """Create a group for every list in ``rank_lists`` and return the one holding ``rank``. COLLECTIVE.

    Every rank must call it with the same lists in the same order. ``timeout`` bounds the subgroups'
    collectives as ``run_gloo_ranks``' ``pg_timeout`` bounds the default group's: ``dist.new_group``
    without one gets gloo's 30-minute default, so a rank stuck in a subgroup would wait out the spawn
    bound instead of failing.
    """
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
    """Run ``worker(rank, *args)`` in ``nprocs`` spawned processes that share one gloo group.

    Every rank gets the env ``torchrun`` gives a single-node job (``MASTER_ADDR``, ``MASTER_PORT`` from
    :func:`~tests.common.ports.free_port`, ``RANK``, ``WORLD_SIZE``, ``LOCAL_RANK``,
    ``LOCAL_WORLD_SIZE``) with ``env`` applied on top, identically on every rank; a value that has to
    differ per rank is the worker's to set. The rendezvous store at ``MASTER_PORT`` is served from this
    process until every rank has exited. The group is initialized before ``worker``, which starts on no
    rank until every rank has initialized it, and destroyed after it on every exit path. ``pg_timeout``
    bounds each collective (``None`` keeps torch's default); set it where the test reads what a stuck
    collective raises. The rendezvous store keeps :data:`STORE_TIMEOUT` either way.

    ``worker`` must be a module-level function, since spawn pickles it by reference. A rank that
    raises or dies fails this call once its peers are stopped (``ProcessRaisedException`` /
    ``ProcessExitedException``), so a worker may assert directly; a test whose subject is a rank that
    raises records each rank's outcome in a file instead. Ranks still running
    :data:`GLOO_JOIN_TIMEOUT_S` after the spawn are killed and ``TimeoutError`` names them.
    """
    # Hosted here, as torchrun's agent hosts it, so the store outlives every rank: a rank may exit while a
    # peer still has store traffic to send (a main-first waiter leaving its phase), and a store hosted in
    # that rank would go down under the peer.
    store = dist.TCPStore("127.0.0.1", free_port(), is_master=True, timeout=STORE_TIMEOUT, wait_for_workers=False)
    context = mp.start_processes(
        _gloo_rank,
        args=(worker, nprocs, store.port, pg_timeout, dict(env or {}), args),
        nprocs=nprocs,
        join=False,
        start_method="spawn",
    )
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
