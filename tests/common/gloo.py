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
    timeout_kwargs = {} if pg_timeout is None else {"timeout": pg_timeout}
    dist.init_process_group("gloo", rank=rank, world_size=nprocs, **timeout_kwargs)
    try:
        worker(rank, *args)
    finally:
        dist.destroy_process_group()


def run_gloo_ranks(
    worker: Callable[..., object],
    nprocs: int,
    *args,
    timeout: float = GLOO_JOIN_TIMEOUT_S,
    pg_timeout: datetime.timedelta | None = None,
    env: Mapping[str, str] | None = None,
) -> None:
    """Run ``worker(rank, *args)`` in ``nprocs`` spawned processes that share one gloo group.

    Every rank gets the env ``torchrun`` gives a single-node job (``MASTER_ADDR``, ``MASTER_PORT`` from
    :func:`~tests.common.ports.free_port`, ``RANK``, ``WORLD_SIZE``, ``LOCAL_RANK``,
    ``LOCAL_WORLD_SIZE``) with ``env`` applied on top, identically on every rank; a value that has to
    differ per rank is the worker's to set. The group is initialized before ``worker`` and destroyed
    after it on every exit path. ``pg_timeout`` bounds each collective (``None`` keeps torch's
    default); set it where the test reads what a stuck collective raises.

    ``worker`` must be a module-level function, since spawn pickles it by reference. A rank that
    raises or dies fails this call once its peers are stopped (``ProcessRaisedException`` /
    ``ProcessExitedException``), so a worker may assert directly; a test whose subject is a rank that
    raises records each rank's outcome in a file instead. Ranks still running ``timeout`` seconds
    after the spawn are killed and ``TimeoutError`` names them.
    """
    context = mp.start_processes(
        _gloo_rank,
        args=(worker, nprocs, free_port(), pg_timeout, dict(env or {}), args),
        nprocs=nprocs,
        join=False,
        start_method="spawn",
    )
    deadline = time.monotonic() + timeout
    while not context.join(timeout=max(0.0, deadline - time.monotonic())):
        if time.monotonic() >= deadline:
            stuck = [rank for rank, process in enumerate(context.processes) if process.is_alive()]
            for process in context.processes:
                if process.is_alive():
                    process.kill()
                process.join()
            raise TimeoutError(
                f"{worker.__qualname__}: rank(s) {stuck} of {nprocs} still running {timeout}s after the spawn "
                f"— stuck in a collective or a store wait; killed"
            )
