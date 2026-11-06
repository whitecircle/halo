"""Rank 0's half of SGLang's weight-update group: a plain c10d group over a ``PrefixStore(group_name, ...)``-wrapped
TCPStore, mirroring ``sglang.srt.utils.common.init_custom_process_group``, asserted by ``Dockerfile.sglang``."""

import logging
import threading
from contextlib import contextmanager
from datetime import timedelta

import torch
import torch.distributed as dist
from torch.distributed import distributed_c10d as c10d

from halo.runtime.ranks import has_process_group
from halo.weight_sync.transport.rendezvous import RendezvousListener

logger = logging.getLogger(__name__)


@contextmanager
def _store_bootstrapped_communicator():
    """Clear the bound device so the new group bootstraps through its store. An eagerly initialized default group makes
    c10d build the communicator with ``ncclCommSplit`` off the trainer's own, which holds only trainer ranks, and the
    trainer would broadcast into its own world, silently."""
    default_pg = c10d._get_default_group() if has_process_group() else None
    bound = getattr(default_pg, "bound_device_id", None) if default_pg is not None else None
    if bound is not None:
        default_pg.bound_device_id = None
    try:
        yield
    finally:
        if bound is not None:
            default_pg.bound_device_id = bound


def build_weight_update_group(
    master_address: str,
    listener: RendezvousListener,
    world_size: int,
    device: torch.device,
    *,
    group_name: str,
    timeout_s: float,
) -> tuple[dist.ProcessGroup, dist.TCPStore]:
    """Join ``group_name`` as rank 0 on a store hosted on ``listener``; the caller keeps the store alive, as dropping
    it frees the port. Built directly, as ``torch.distributed.rendezvous`` under torchrun returns the elastic agent's.
    Passing ``device_id`` would init the communicator eagerly and deadlock against SGLang's lazy join."""
    if device.type == "cuda":
        torch.cuda.set_device(device)
    store = listener.start_store(master_address, world_size, timeout_s=timeout_s, wait_for_workers=False)
    # SGLang prefixes its store with the group name; an unprefixed one here reads different keys and hangs.
    prefixed = dist.PrefixStore(group_name, store)
    try:
        with _store_bootstrapped_communicator():
            group, _ = c10d._new_process_group_helper(
                world_size,
                0,
                [],
                c10d.Backend("nccl"),
                prefixed,
                group_name=group_name,
                timeout=timedelta(seconds=timeout_s),
            )
    except Exception:
        # So the failed attempt does not keep the port from a retry.
        listener.free_port()
        raise
    # Not a subset of the trainer's world; without SGLang's identity rank map `broadcast(src=0)` cannot find the root.
    c10d._world.pg_group_ranks[group] = {i: i for i in range(world_size)}
    logger.info(
        f"Weight-update group formed: rank 0 of {world_size} at {master_address}:{listener.port} (group {group_name!r})"
    )
    return group, store


def release_weight_update_group(group: dist.ProcessGroup | None, *, destroy: bool) -> None:
    """Never raises. ``destroy=False`` drops only torch's rank-map entry, as at interpreter exit the engine never
    enters destroy and the NCCL destroy would block."""
    if group is None:
        return
    if destroy:
        try:
            dist.destroy_process_group(group)
        except Exception as e:
            logger.warning(f"Error destroying the weight-update group: {e}")
    c10d._world.pg_group_ranks.pop(group, None)


def abort_weight_update_group(group: dist.ProcessGroup | None, *, timeout_s: float) -> bool:
    """Abort ``group``, ending kernels spinning on a silent peer; no destroy may follow. Never raises, and returns after
    ``timeout_s`` even when the abort does not: torch's abort waits on a communicator another thread may be tearing
    down, which leaves the group parked on its own thread. Returns False for a group left parked, which the caller must
    not release."""
    if group is None:
        return True

    def abort() -> None:
        try:
            c10d._abort_process_group(group)
        except Exception as e:
            logger.warning(f"Aborting the SGLang weight-update group failed: {e}")

    aborting = threading.Thread(target=abort, name="weight-update abort", daemon=True)
    aborting.start()
    aborting.join(timeout_s)
    if aborting.is_alive():
        logger.error(
            f"Aborting the SGLang weight-update group did not return within {timeout_s:.0f}s; it is left parked."
        )
    return not aborting.is_alive()
