"""Rank 0's half of SGLang's weight-update group: a plain c10d group over a ``PrefixStore(group_name, ...)``-wrapped
TCPStore, mirroring ``sglang.srt.utils.common.init_custom_process_group`` without importing sglang.
``Dockerfile.sglang`` asserts at build that both halves of that convention still hold."""

import logging
from contextlib import contextmanager
from datetime import timedelta

import torch
import torch.distributed as dist
from torch.distributed import distributed_c10d as c10d

from src.weight_sync.transport.rendezvous import RendezvousListener

logger = logging.getLogger(__name__)


@contextmanager
def _store_bootstrapped_communicator():
    """Force the new group to bootstrap through its store instead of splitting the default one.

    An eagerly initialized default group (``init_process_group(device_id=...)``) makes c10d build the
    communicator with ``ncclCommSplit`` off the trainer's own, which holds only trainer ranks: the trainer
    would broadcast into its own world, silently. Clearing the bound device selects the store handshake.
    """
    default_pg = c10d._get_default_group() if dist.is_available() and dist.is_initialized() else None
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
    """Host the rendezvous store on ``listener`` and join group ``group_name`` as rank 0 of ``world_size``.

    Returns the group and the store, which the caller keeps alive: dropping it frees the port. ``timeout_s``
    is the client's formation deadline, so neither half gives up first. The store is built directly, since
    ``torch.distributed.rendezvous`` under torchrun returns the elastic agent's. ``device`` is made current
    but not passed as ``device_id``, which would init the communicator eagerly and deadlock against SGLang's
    lazy join.
    """
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
        # Dropped so the listener dies with the failed attempt and a retry can rebind the port.
        store = None
        raise
    # The group is no subset of the trainer's world, so it declares SGLang's identity rank map, without which
    # `broadcast(..., src=0)` cannot translate the root rank.
    c10d._world.pg_group_ranks[group] = {i: i for i in range(world_size)}
    logger.info(
        f"Weight-update group formed: rank 0 of {world_size} at {master_address}:{listener.port} (group {group_name!r})"
    )
    return group, store


def release_weight_update_group(group: dist.ProcessGroup | None, *, destroy: bool) -> None:
    """Release a group from :func:`build_weight_update_group`; never raises. ``destroy=False`` drops only torch's
    rank-map entry: at interpreter exit the engine never enters destroy, so the NCCL destroy would block."""
    if group is None:
        return
    if destroy:
        try:
            dist.destroy_process_group(group)
        except Exception as e:  # teardown is best-effort; never mask the original failure
            logger.warning(f"Error destroying the weight-update group: {e}")
    c10d._world.pg_group_ranks.pop(group, None)


def abort_weight_update_group(group: dist.ProcessGroup | None) -> None:
    """Abort ``group``, ending kernels spinning on a silent peer; no destroy may follow. Never raises."""
    if group is None:
        return
    try:
        c10d._abort_process_group(group)
    except Exception as e:
        logger.warning(f"Aborting the SGLang weight-update group failed: {e}")
