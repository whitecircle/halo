"""Deadline-bounded host waits on the CUDA work a weight-sync broadcast queued, which no torch watchdog covers."""

import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager

import torch

from halo.env import env_positive_float

# Covers the update RPC that declares a chunk and the drain of its broadcasts. Read once at import so a malformed
# value cannot raise mid-RL-step.
CHUNK_SYNC_TIMEOUT_S = env_positive_float("HALO_NCCL_SYNC_TIMEOUT_SECONDS", None) or 600.0


@contextmanager
def _aborting_on_failure(abort: Callable[[], None]) -> Iterator[None]:
    try:
        yield
    except RuntimeError:
        abort()
        raise


def bounded_event_sync(event: torch.cuda.Event, timeout_s: float, what: str, *, abort: Callable[[], None]) -> None:
    """Wait for a recorded ``event``; when ``timeout_s`` passes or the wait fails, ``abort`` the transport (a
    kernel spinning on a dead peer hangs every later device sync) and raise."""
    deadline = time.monotonic() + timeout_s
    with _aborting_on_failure(abort):
        while not event.query():
            if time.monotonic() > deadline:
                raise RuntimeError(
                    f"NCCL weight-sync {what} did not complete within {timeout_s:.0f}s — the peer "
                    f"never joined the collective. Causes: a dead or wedged rollout server; a group "
                    f"address the peer cannot reach (VLLM_GROUP_HOST / SGLANG_GROUP_HOST — same-host "
                    f"setups need loopback, multi-homed nodes the NIC the server can route to); the two "
                    f"containers on different NCCL transports (NCCL_NET / NCCL_NET_PLUGIN must match) or "
                    f"on different aws-ofi-nccl builds; or cuMem disabled on one side only "
                    f"(NCCL_CUMEM_ENABLE). scripts/profiling/weight_sync_transport.py reports the "
                    f"transport the group actually formed on."
                )
            # 1 ms, as the poll is on the per-chunk critical path and still yields the GIL.
            time.sleep(0.001)


def bounded_stream_sync(stream: torch.cuda.Stream, timeout_s: float, what: str, *, abort: Callable[[], None]) -> None:
    """Drain everything queued on ``stream`` so far, under :func:`bounded_event_sync`'s deadline and abort."""
    event = torch.cuda.Event()
    with _aborting_on_failure(abort):
        event.record(stream)
    bounded_event_sync(event, timeout_s, what, abort=abort)
