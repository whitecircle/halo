"""Destroy vLLM's weight-transfer NCCL communicator on re-init instead of leaking it.

``NCCLWeightTransferEngine.init_transfer_engine`` builds a fresh ``PyNcclCommunicator`` (a new
``ncclCommInitRank``) on every ``/init_weight_transfer_engine`` and only drops the reference to the
previous one, and ``shutdown()`` does the same; ``PyNcclCommunicator`` has no ``__del__``, so every
trainer that connects strands a live communicator — with the CUDA streams and NCCL buffers behind it
— on each engine-core worker. After a handful of connect/close cycles ``ncclCommInitRank`` fails with
an unhandled CUDA error while ``/health`` still answers 200, which is what a trainer restarted
against a long-lived server runs into. Only the dense engine is patched: it is the backend this
image serves with, and the sparse one speaks a wire protocol the toolkit's client does not.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

# Stamped on the replacements so ``Dockerfile.vllm`` can assert the patch took.
PATCH_MARKER = "_halo_weight_transfer_reinit"

_APPLIED = False


def _destroy_group(engine) -> None:
    """Abort and drop the communicator ``engine`` still holds, if any."""
    group = engine.model_update_group
    if group is None:
        return
    # Dropped before the abort so a destroy that raises cannot leave a dead communicator reachable.
    engine.model_update_group = None
    try:
        group.destroy()
    except Exception:  # noqa: BLE001 — cleanup must never fail the init or shutdown around it
        logger.exception("failed to destroy the previous weight-transfer communicator")


def apply() -> None:
    """Idempotently patch the NCCL weight-transfer engine's communicator lifecycle."""
    global _APPLIED
    if _APPLIED:
        return

    from vllm.distributed.weight_transfer import nccl_engine  # noqa: PLC0415 — vLLM-only lazy import

    engine_cls = nccl_engine.NCCLWeightTransferEngine
    original_init = engine_cls.init_transfer_engine

    def init_transfer_engine(self, init_info):
        _destroy_group(self)
        return original_init(self, init_info)

    def shutdown(self):
        _destroy_group(self)

    for replacement in (init_transfer_engine, shutdown):
        setattr(replacement, PATCH_MARKER, True)
    engine_cls.init_transfer_engine = init_transfer_engine
    engine_cls.shutdown = shutdown
    _APPLIED = True
    logger.info("vllm_weight_transfer_reinit_patch applied to %s", engine_cls.__name__)
