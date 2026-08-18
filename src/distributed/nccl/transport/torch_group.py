"""Torch-native NCCL group for SGLang weight sync: the trainer joins as rank 0, the engine as 1..N.

SGLang builds its weight-update group with ``init_custom_process_group`` — a plain c10d group over a
TCPStore, wrapped in ``PrefixStore(group_name, ...)`` — and then broadcasts real typed tensors with
``torch.distributed.broadcast(..., src=0)``. That is a different bootstrap from the vLLM path in
``stateless_group.py``, which exchanges a pickled ``ncclUniqueId`` through raw store keys and
drives pynccl directly. The two cannot interoperate: c10d derives its own store keys for the
handshake, so the trainer has to speak c10d here rather than reuse the vendored pynccl communicator.

This mirrors SGLang's side (``sglang.srt.utils.common.init_custom_process_group``) without importing
sglang — the training image never installs it. ``Dockerfile.sglang`` asserts at build time that the
two load-bearing halves of that convention still hold.
"""

import logging
from contextlib import contextmanager
from datetime import timedelta

import torch
import torch.distributed as dist
from torch.distributed import distributed_c10d as c10d

logger = logging.getLogger(__name__)

# SGLang's own default; it is the PrefixStore prefix, so both sides must agree on it.
DEFAULT_WEIGHT_UPDATE_GROUP_NAME = "weight_update_group"


@contextmanager
def _store_bootstrapped_communicator():
    """Force the new group to bootstrap through its store instead of splitting the default one.

    When the default process group was initialized eagerly (``init_process_group(device_id=...)``,
    which is what the trainer's launcher does), ``_new_process_group_helper`` takes
    ``split_from = _get_split_source(default_pg)`` and builds the communicator with
    ``ncclCommSplit`` off the trainer's OWN communicator. A split can only ever contain ranks of its
    parent, so the engine — a separate process that joined through the store — is not in it: the
    trainer ends up broadcasting into its own world while the engine waits out the collective, with
    no error on either side and no second communicator in ``NCCL_DEBUG``. Clearing the bound device
    for the duration selects the store handshake, the one bootstrap both sides can reach.
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


def create_weight_update_group(
    master_address: str,
    master_port: int,
    world_size: int,
    device: torch.device,
    group_name: str = DEFAULT_WEIGHT_UPDATE_GROUP_NAME,
    *,
    timeout_s: float,
) -> tuple[dist.ProcessGroup, dist.TCPStore]:
    """Host the rendezvous store and join the weight-update group as rank 0.

    ``world_size`` counts the trainer: ``1 + tp_size * dp_size``. Returns the group and the store,
    which the caller must keep alive — dropping the store closes the listener and the engine can no
    longer re-join. Pair with :func:`destroy_weight_update_group`.

    ``timeout_s`` has no default: this half of the handshake blocks against a concurrent HTTP request
    to the engine and the client owns that deadline (``_GROUP_FORMATION_TIMEOUT_S``); a second
    spelling here would be free to drift into one half giving up first.

    The store is constructed directly rather than through ``torch.distributed.rendezvous``, which
    under ``torchrun`` hands back the elastic agent's store — rank 0 would join as a client and both
    sides wait out the timeout. ``wait_for_workers=False`` because the engine only dials in after the
    caller's ``/init_weights_update_group`` request, issued concurrently with this call.

    ``device`` is made current but deliberately NOT passed as ``device_id``: NCCL infers the device
    from the calling thread at the first collective, and an unpinned trainer can infer a GPU the
    tensors do not live on, while ``device_id`` initializes the communicator eagerly and deadlocks
    against SGLang's lazy join (its peer calls ``ncclCommInitRank`` only at its first broadcast).
    """
    if device.type == "cuda":
        torch.cuda.set_device(device)
    store = dist.TCPStore(
        host_name=master_address,
        port=master_port,
        world_size=world_size,
        is_master=True,
        timeout=timedelta(seconds=timeout_s),
        wait_for_workers=False,
    )
    # SGLang wraps the rendezvous store in PrefixStore(group_name) before handing it to c10d, so an
    # unprefixed store here would leave the two sides reading different keys and hanging.
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
        # Drop this frame's reference so the listener dies with the failed attempt; otherwise it
        # outlives the raise and the port cannot be rebound by a retry.
        store = None
        raise
    # c10d resolves a global rank to a group rank through this map. The group is not a subset of the
    # trainer's own world (its peers are engine processes), so declare the identity mapping SGLang
    # also declares — without it `broadcast(..., src=0)` cannot translate the root rank.
    c10d._world.pg_group_ranks[group] = {i: i for i in range(world_size)}
    logger.info(
        f"Weight-update group formed: rank 0 of {world_size} at {master_address}:{master_port} (group {group_name!r})"
    )
    return group, store


def destroy_weight_update_group(group: dist.ProcessGroup | None) -> None:
    """Tear down a group from :func:`create_weight_update_group`, tolerating an already-dead peer.

    Runs on cleanup paths where the engine may already be gone, so a failure here must never mask
    the error that triggered the teardown.
    """
    if group is None:
        return
    try:
        dist.destroy_process_group(group)
    except Exception as e:  # teardown is best-effort; never mask the original failure
        logger.warning(f"Error destroying the weight-update group: {e}")
    c10d._world.pg_group_ranks.pop(group, None)


def drop_weight_update_group_bookkeeping(group: dist.ProcessGroup | None) -> None:
    """Forget a group WITHOUT the NCCL destroy — the interpreter-exit teardown.

    At exit the group's peers are engine processes that never enter destroy, so with an interrupted
    sync in flight ``dist.destroy_process_group`` blocks rather than raising and wedges the exiting
    process; process teardown reclaims the communicator anyway. Only the rank map entry is dropped
    (the same no-collectives-at-teardown rule as the EP dispatcher's ``__del__``).
    """
    if group is not None:
        c10d._world.pg_group_ranks.pop(group, None)
