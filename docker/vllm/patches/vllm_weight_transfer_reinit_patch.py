"""Destroy vLLM's weight-transfer NCCL communicator on re-init instead of leaking it.

``NCCLWeightTransferEngine.init_transfer_engine`` builds a fresh ``PyNcclCommunicator`` (a new
``ncclCommInitRank``) on every ``/init_weight_transfer_engine`` and only drops the reference to the
previous one. ``PyNcclCommunicator`` has no ``__del__``, so every trainer that reconnects without a
``/shutdown_weight_transfer_engine`` first (any trainer that crashed or was killed) leaves a live
communicator, with its CUDA streams and NCCL buffers, on each engine-core worker. After a few such
cycles ``ncclCommInitRank`` fails with an unhandled CUDA error while ``/health`` still answers 200.
Upstream's ``shutdown()`` destroys the communicator, so only the re-init is patched. Only the dense
engine is patched: it is the backend this image serves with, and the sparse one speaks a wire protocol
the toolkit's client does not.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

# Stamped on the replacement so ``Dockerfile.vllm`` can assert the patch took.
PATCH_MARKER = "_halo_weight_transfer_reinit"

_APPLIED = False


def _destroy_group(engine) -> None:
    """Abort and drop the communicator ``engine`` still holds, if any."""
    group = engine.model_update_group
    if group is None:
        return
    # Dropped before the abort so a destroy that raises leaves no dead communicator reachable.
    engine.model_update_group = None
    try:
        group.destroy()
    except Exception:  # noqa: BLE001 — cleanup must not fail the init around it
        logger.exception("failed to destroy the previous weight-transfer communicator")


def apply() -> None:
    """Idempotently patch the NCCL weight-transfer engine's re-init."""
    global _APPLIED
    if _APPLIED:
        return

    from vllm.distributed.weight_transfer import nccl_engine  # noqa: PLC0415 — vLLM-only lazy import

    engine_cls = nccl_engine.NCCLWeightTransferEngine
    original_init = engine_cls.init_transfer_engine

    def init_transfer_engine(self, init_info):
        _destroy_group(self)
        return original_init(self, init_info)

    setattr(init_transfer_engine, PATCH_MARKER, True)
    engine_cls.init_transfer_engine = init_transfer_engine
    _APPLIED = True
    logger.info("vllm_weight_transfer_reinit_patch applied to %s.init_transfer_engine", engine_cls.__name__)
