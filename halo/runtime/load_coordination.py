"""Weight-load coordination: the per-node throttle, its world join and the global load span."""

import logging
import time
from collections.abc import Callable
from contextlib import contextmanager

import torch
import torch.distributed as dist

from halo.env import store_timeout
from halo.runtime.collectives import collective_device
from halo.runtime.consensus import StorePhase, store_join_recorded_failure
from halo.runtime.ranks import (
    get_local_rank,
    get_local_world_size,
    get_node_rank,
    has_process_group,
    is_global_main_process,
    is_multi_rank_run,
)

logger = logging.getLogger(__name__)

# Ranks per node loading at once when ``max_concurrent_loading`` is unset; an explicit value is not capped.
MAX_CONCURRENT_LOADING_CAP = 4


def resolve_load_concurrency(max_concurrent: int | None, local_world_size: int) -> int:
    """An explicit value passes through (``0`` = no throttle); ``None`` is half the node's width up to the cap (4 on an
    8-GPU node, 2 on a 4-GPU tray, where a flat 4 would admit every rank)."""
    if max_concurrent is not None:
        return max_concurrent
    return min(MAX_CONCURRENT_LOADING_CAP, max(1, local_world_size // 2))


@contextmanager
def sequential_load_within_node(max_concurrent: int | None = 1):
    """Admit a node's local ranks in batches of ``max_concurrent`` (0 = all, ``None`` per
    :func:`resolve_load_concurrency`) to spare host RAM. Waits on the c10d store; a long load outlasts NCCL."""
    if not has_process_group():
        yield
        return

    local_rank = get_local_rank()
    local_world_size = get_local_world_size()
    max_concurrent = resolve_load_concurrency(max_concurrent, local_world_size)

    if local_world_size <= 1 or max_concurrent == 0 or max_concurrent >= local_world_size:
        yield
        return

    phase = StorePhase("seq_load/model", f"node{get_node_rank()}", local_rank, local_world_size, store_timeout())

    batch = local_rank // max_concurrent
    try:
        # In the try: a timed-out wait must still mark this rank done, or later batches wait out the store timeout.
        phase.wait([f"rank{r}_done" for r in range(batch * max_concurrent)])
        yield
    finally:
        phase.set(f"rank{local_rank}_done")
        phase.finish([f"rank{r}_done" for r in range(local_world_size)])


@contextmanager
def joined_node_load(what: str, max_concurrent: int | None):
    """Per-node throttled load whose exit joins the world over the store, where loads outlasting the NCCL watchdog meet
    and a rank-local failure raises on every rank. COLLECTIVE-equivalent: every rank enters once per load."""
    failure: BaseException | None = None
    try:
        with sequential_load_within_node(max_concurrent=max_concurrent):
            yield
    except BaseException as exc:  # a KeyboardInterrupt must reach the join too, or the peers wait it out
        failure = exc
    store_join_recorded_failure("node_load", failure, what)


def _log_global_load_duration_seconds(*, tag: str, method: str, t_start_wall: float, t_end_wall: float) -> float:
    """COLLECTIVE: log and return the span ``max(t_end) - min(t_start)`` across ranks (needs NTP-synced hosts)."""
    local_sec = t_end_wall - t_start_wall
    if not is_multi_rank_run():
        logger.info(
            f"[{tag}] model load timing: method={method} global_load_duration_sec={local_sec:.3f} (single process)"
        )
        return local_sec

    device = collective_device()

    st = torch.tensor([t_start_wall], dtype=torch.float64, device=device)
    en = torch.tensor([t_end_wall], dtype=torch.float64, device=device)
    dist.all_reduce(st, op=dist.ReduceOp.MIN)
    dist.all_reduce(en, op=dist.ReduceOp.MAX)
    global_sec = max(0.0, en.item() - st.item())

    if is_global_main_process():
        logger.info(
            f"[{tag}] model load timing: method={method} global_load_duration_sec={global_sec:.3f} "
            f"(world_size={dist.get_world_size()} wall span min_start→max_end across ranks)"
        )
    return global_sec


def timed_global_load[T](tag: str, method: str, load_fn: Callable[[], T]) -> T:
    """COLLECTIVE: run ``load_fn`` and log its global wall span. The log is not in a ``finally``, since a collective
    entered while one rank unwinds turns that rank's traceback into a timeout on every other rank."""
    t_start_wall = time.time()
    loaded = load_fn()
    _log_global_load_duration_seconds(tag=tag, method=method, t_start_wall=t_start_wall, t_end_wall=time.time())
    return loaded
