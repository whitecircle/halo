"""Rank 0's half of vLLM's weight-transfer bootstrap, over a rendezvous store that leaves the global torch.distributed
state alone. Vendored from vLLM v0.18.0, Apache-2.0."""

import logging
import pickle

import torch
from torch.distributed import TCPStore

from halo.env import env_str
from halo.weight_sync.transport.bounded_waits import bounded_stream_sync
from halo.weight_sync.transport.nccl_library import (
    NCCLLibrary,
    buffer_type,
    cudaStream_t,
    nccl_data_type,
    ncclComm_t,
    ncclUniqueId,
)
from halo.weight_sync.transport.rendezvous import RendezvousListener

logger = logging.getLogger(__name__)

# Bounds the one-key id handshake alone, not DIST_STORE_TIMEOUT_HOURS' whole-job waits.
_STORE_TIMEOUT_S = 300

# The key of rank 0's first StatelessProcessGroup broadcast, where vLLM's workers read the id.
_UNIQUE_ID_KEY = "broadcast_from/0/0"

# vLLM's own truthy spellings for its VLLM_DISABLE_PYNCCL, narrower than the toolkit's env_flag.
_VLLM_TRUE_VALUES = ("1", "true")

# The warm-up all-reduce proving the new communicator talks; it also covers the peer's own NCCL init.
_WARMUP_SYNC_TIMEOUT_S = 120.0


def vllm_pynccl_disabled() -> bool:
    return env_str("VLLM_DISABLE_PYNCCL", "").strip().lower() in _VLLM_TRUE_VALUES


class UniqueIdStore:
    """Rank 0's rendezvous store, carrying only the NCCL unique id."""

    def __init__(self, store: TCPStore, world_size: int):
        self._store: TCPStore | None = store
        self.world_size = world_size

    @classmethod
    def create(cls, *, host: str, listener: RendezvousListener, world_size: int) -> "UniqueIdStore":
        """Host the store on ``listener`` and block until every engine worker joined it."""
        return cls(listener.start_store(host, world_size, timeout_s=_STORE_TIMEOUT_S, use_libuv=False), world_size)

    def publish(self, unique_id: ncclUniqueId) -> None:
        if self._store is None:
            raise RuntimeError("The unique-id store is closed — form a new group to publish an id.")
        self._store.set(_UNIQUE_ID_KEY, pickle.dumps(unique_id))

    def close(self) -> None:
        """Release the listener so the group port can be rebound, even while a thread parked in ``ncclCommInitRank``
        still holds this object; idempotent."""
        self._store = None


class PyNcclCommunicator:
    def __init__(self, store: UniqueIdStore, device: torch.device):
        world_size = store.world_size
        self.device = device
        # Set by abort(), so a null-handle communicator fails by name rather than "invalid argument".
        self.aborted = False

        if world_size < 2:
            raise RuntimeError(
                f"The vLLM engine reported no workers (group world size {world_size}), so a weight sync would reach "
                f"nothing and the engine would keep serving the initial policy for the whole run."
            )
        if vllm_pynccl_disabled():
            raise RuntimeError(
                "VLLM_DISABLE_PYNCCL is set in the trainer process, which would silently disable the "
                "NCCL weight-sync broadcast — the vLLM generator would keep serving the initial policy "
                "for the whole run. It belongs to the vLLM server only; do not pass the server's "
                "--env-file to the trainer."
            )
        try:
            self.nccl = NCCLLibrary()
        except Exception as e:
            # Disabling instead would block the server for the full timeout on a broadcast that never comes.
            raise ImportError(f"libnccl.so.2 unavailable — NCCL weight sync cannot run: {e}") from e

        unique_id = self.nccl.ncclGetUniqueId()
        logger.info("NCCL version: %s", self.nccl.ncclGetVersion())
        store.publish(unique_id)

        with torch.cuda.device(device):
            self.comm: ncclComm_t = self.nccl.ncclCommInitRank(world_size, unique_id, 0)
            stream = torch.cuda.current_stream()
            data = torch.zeros(1, device=device)
            out = torch.empty_like(data)
            self.nccl.ncclAllReduce(
                buffer_type(data.data_ptr()),
                buffer_type(out.data_ptr()),
                1,
                nccl_data_type(data.dtype),
                0,
                self.comm,
                cudaStream_t(stream.cuda_stream),
            )
            bounded_stream_sync(stream, timeout_s=_WARMUP_SYNC_TIMEOUT_S, what="warm-up all-reduce", abort=self.abort)
            del data, out

    def abort(self) -> None:
        """Kill in-flight NCCL kernels; unusable afterwards."""
        self.aborted = True
        if not self.comm:
            return
        try:
            self.nccl.ncclCommAbort(self.comm)
        except Exception as e:  # best-effort on an already-failing path
            logger.warning(f"ncclCommAbort failed: {e}")
        self.comm = ncclComm_t()

    def broadcast(self, tensor: torch.Tensor) -> None:
        if self.aborted:
            raise RuntimeError(
                "NCCL weight-sync communicator was aborted after a failed collective and cannot be "
                "reused — rebuild the client (reconnect) before broadcasting again."
            )
        if tensor.device != self.device:
            raise ValueError(
                f"Weight-sync broadcast of a tensor on {tensor.device}, but the communicator was formed on "
                f"{self.device}."
            )
        stream = torch.cuda.current_stream()
        buffer = buffer_type(tensor.data_ptr())
        self.nccl.ncclBroadcast(
            buffer,
            buffer,
            tensor.numel(),
            nccl_data_type(tensor.dtype),
            0,
            self.comm,
            cudaStream_t(stream.cuda_stream),
        )
