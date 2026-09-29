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
    # The store ``env://`` rendezvous builds, with STORE_TIMEOUT in place of pg_timeout.
    store = dist.TCPStore("127.0.0.1", port, nprocs, is_master=rank == 0, timeout=STORE_TIMEOUT, multi_tenant=True)
    timeout_kwargs = {} if pg_timeout is None else {"timeout": pg_timeout}
    dist.init_process_group("gloo", rank=rank, world_size=nprocs, store=store, **timeout_kwargs)
    try:
        worker(rank, *args)
    finally:
        dist.destroy_process_group()


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
    differ per rank is the worker's to set. The group is initialized before ``worker`` and destroyed
    after it on every exit path. ``pg_timeout`` bounds each collective (``None`` keeps torch's
    default); set it where the test reads what a stuck collective raises. The rendezvous store keeps
    :data:`STORE_TIMEOUT` either way.

    ``worker`` must be a module-level function, since spawn pickles it by reference. A rank that
    raises or dies fails this call once its peers are stopped (``ProcessRaisedException`` /
    ``ProcessExitedException``), so a worker may assert directly; a test whose subject is a rank that
    raises records each rank's outcome in a file instead. Ranks still running
    :data:`GLOO_JOIN_TIMEOUT_S` after the spawn are killed and ``TimeoutError`` names them.
    """
    context = mp.start_processes(
        _gloo_rank,
        args=(worker, nprocs, free_port(), pg_timeout, dict(env or {}), args),
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
