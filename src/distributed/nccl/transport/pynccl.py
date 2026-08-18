"""Lightweight NCCL communicator for weight broadcast; exchanges the unique ID over a StatelessProcessGroup. Vendored from vLLM v0.18.0, Apache-2.0."""

import logging
import time

import torch

from src.distributed.nccl._nccl_wrapper import (
    NCCLLibrary,
    buffer_type,
    cudaStream_t,
    ncclComm_t,
    ncclDataTypeEnum,
    ncclUniqueId,
)
from src.distributed.nccl.transport.stateless_group import StatelessProcessGroup
from src.env import env_positive_float, env_str

logger = logging.getLogger(__name__)


# Parsed once at import so a malformed value can't raise mid-RL-step.
_SYNC_TIMEOUT_OVERRIDE = env_positive_float("HALO_NCCL_SYNC_TIMEOUT_SECONDS", None)

# vLLM's own truthy spellings for VLLM_DISABLE_PYNCCL, and only those. It is the SERVER's variable:
# reading it with the toolkit's env_flag would additionally accept "yes"/"on", so a value the server
# ignores would refuse the trainer here. A third-party var must be read exactly as its owner reads it.
_VLLM_TRUE_VALUES = ("1", "true")

# Deadline for the one-element warm-up all-reduce that proves the freshly built communicator can
# actually talk; generous because it also covers the peer's own NCCL init.
_WARMUP_SYNC_TIMEOUT_S = 120.0
# Poll cadence of the bounded sync loop — a busy-wait tight enough to be responsive, loose enough
# to leave the GIL alone while the transfer runs.
_SYNC_POLL_INTERVAL_S = 0.05


def vllm_pynccl_disabled() -> bool:
    """Whether the vLLM server's ``VLLM_DISABLE_PYNCCL`` is set, parsed the way vLLM parses it."""
    return env_str("VLLM_DISABLE_PYNCCL", "").strip().lower() in _VLLM_TRUE_VALUES


def bounded_stream_sync(stream: torch.cuda.Stream, timeout_s: float, what: str) -> None:
    """Drain ``stream`` with a deadline: a collective whose peer never arrives spins forever, and these comms have no torch watchdog."""
    if _SYNC_TIMEOUT_OVERRIDE is not None:
        timeout_s = _SYNC_TIMEOUT_OVERRIDE
    event = torch.cuda.Event()
    event.record(stream)
    deadline = time.monotonic() + timeout_s
    while not event.query():
        if time.monotonic() > deadline:
            raise RuntimeError(
                f"NCCL weight-sync {what} did not complete within {timeout_s:.0f}s — the peer "
                f"never joined the collective. Causes: dead/wedged vLLM server; a group address "
                f"the peer cannot reach (VLLM_GROUP_HOST — same-host setups need loopback, "
                f"multi-homed nodes the NIC the vLLM host can route to); or, on hosts WITHOUT "
                f"InfiniBand, the training image's OFI/Gin NCCL plugin wedging the transport — "
                f"launch the trainer with NCCL_NET=Socket NCCL_IB_DISABLE=1."
            )
        time.sleep(_SYNC_POLL_INTERVAL_S)


class PyNcclCommunicator:
    """NCCL communicator for weight broadcast; exposes broadcast() and abort()."""

    def __init__(self, group: StatelessProcessGroup, device: int | str | torch.device):
        self.rank = group.rank
        self.world_size = group.world_size
        self.group = group

        if isinstance(device, int):
            device = torch.device(f"cuda:{device}")
        elif isinstance(device, str):
            device = torch.device(device)
        # Assigned before the disabled short-circuit: callers read ``.device`` to stage weights, and a
        # missing attribute would surface as an AttributeError with the server already paused.
        self.device = device
        # Set by abort(); a NULL-handle communicator must fail by name, not as "NCCL error: invalid argument".
        self.aborted = False

        # VLLM_DISABLE_PYNCCL is a vLLM SERVER knob. Reaching the trainer (a server-aimed --env-file is
        # the usual way) it would disable the broadcast outright: the policy never reaches the generator
        # and RL keeps training against a stale one, with nothing raised. Same failure the ImportError
        # below refuses to allow, so refuse it here too.
        if vllm_pynccl_disabled() and self.world_size > 1:
            raise RuntimeError(
                "VLLM_DISABLE_PYNCCL is set in the trainer process, which would silently disable the "
                "NCCL weight-sync broadcast — the vLLM generator would keep serving the initial policy "
                "for the whole run. It belongs to the vLLM server only; do not pass the server's "
                "--env-file to the trainer."
            )
        if self.world_size == 1:
            # Nobody to broadcast to; a no-op here is the correct behaviour, not a hidden failure.
            self.disabled = True
            return

        try:
            self.nccl = NCCLLibrary()
        except Exception as e:
            # Fail loud: a silently disabled communicator broadcasts nothing while the server blocks for the full timeout.
            raise ImportError(f"libnccl.so.2 unavailable — NCCL weight sync cannot run: {e}") from e

        self.disabled = False

        if self.rank == 0:
            self.unique_id = self.nccl.ncclGetUniqueId()
            logger.info("NCCL version: %s", self.nccl.ncclGetVersion())
        else:
            self.unique_id = ncclUniqueId()

        self.unique_id = group.broadcast_obj(self.unique_id, src=0)

        with torch.cuda.device(device):
            self.comm: ncclComm_t = self.nccl.ncclCommInitRank(
                self.world_size,
                self.unique_id,
                self.rank,
            )
            stream = torch.cuda.current_stream()
            data = torch.zeros(1, device=device)
            out = torch.empty_like(data)
            self.nccl.ncclAllReduce(
                buffer_type(data.data_ptr()),
                buffer_type(out.data_ptr()),
                1,
                ncclDataTypeEnum.from_torch(data.dtype),
                0,
                self.comm,
                cudaStream_t(stream.cuda_stream),
            )
            try:
                bounded_stream_sync(stream, timeout_s=_WARMUP_SYNC_TIMEOUT_S, what="warm-up all-reduce")
            except RuntimeError:
                self.abort()  # kill the spinning kernel or every later device sync hangs
                raise
            del data, out

    def abort(self) -> None:
        """Terminate the communicator, killing in-flight NCCL kernels; unusable afterwards."""
        self.aborted = True
        if self.disabled or not getattr(self, "comm", None):
            return
        try:
            self.nccl.ncclCommAbort(self.comm)
        except Exception as e:  # abort is best-effort cleanup on an already-failing path
            logger.warning(f"ncclCommAbort failed: {e}")
        self.comm = ncclComm_t()

    def broadcast(self, tensor: torch.Tensor, src: int):
        if self.disabled:
            return
        if self.aborted:
            raise RuntimeError(
                "NCCL weight-sync communicator was aborted after a failed collective and cannot be "
                "reused — rebuild the client (reconnect) before broadcasting again."
            )
        assert tensor.device == self.device
        stream = torch.cuda.current_stream()
        sendbuff = buffer_type(tensor.data_ptr()) if src == self.rank else buffer_type()
        self.nccl.ncclBroadcast(
            sendbuff,
            buffer_type(tensor.data_ptr()),
            tensor.numel(),
            ncclDataTypeEnum.from_torch(tensor.dtype),
            src,
            self.comm,
            cudaStream_t(stream.cuda_stream),
        )
