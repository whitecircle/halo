"""Deadline-bounded host waits on the CUDA work a weight-sync broadcast queued: these comms have no torch
watchdog, so a peer that never joins would otherwise park the trainer."""

import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager

import torch

from src.env import env_positive_float

# Deadline for draining one chunk's broadcasts (a vLLM packed buffer, a SGLang chunk's sends), sized for a
# full chunk over a slow link; HALO_NCCL_SYNC_TIMEOUT_SECONDS replaces it, while the warm-up and the
# failure-path cleanup keep their own shorter deadlines. Read once at import, so a malformed value cannot
# raise mid-RL-step.
BROADCAST_DRAIN_TIMEOUT_S = env_positive_float("HALO_NCCL_SYNC_TIMEOUT_SECONDS", None) or 600.0

# Poll cadence of the bounded sync loop. It sits on the critical path once per drained buffer or
# chunk, a few milliseconds of transfer each, so the granularity is a direct per-chunk tax; 1 ms
# still yields the GIL between polls.
_SYNC_POLL_INTERVAL_S = 0.001


@contextmanager
def _aborting_on_failure(abort: Callable[[], None]) -> Iterator[None]:
    """Run ``abort`` before a failed wait's error propagates."""
    try:
        yield
    except RuntimeError:
        abort()
        raise


def bounded_event_sync(event: torch.cuda.Event, timeout_s: float, what: str, *, abort: Callable[[], None]) -> None:
    """Wait for a recorded ``event``; when ``timeout_s`` passes or the wait itself fails, call ``abort`` and raise.

    ``abort`` ends the transport the event's work runs on: a kernel left spinning on a peer that never
    answers hangs every later device synchronization until the abort kills it, and a transport whose wait
    failed (a sticky CUDA error) is then refused by name rather than reused or destroyed as if healthy.
    """
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
            time.sleep(_SYNC_POLL_INTERVAL_S)


def bounded_stream_sync(stream: torch.cuda.Stream, timeout_s: float, what: str, *, abort: Callable[[], None]) -> None:
    """Drain everything queued on ``stream`` so far, under :func:`bounded_event_sync`'s deadline and abort."""
    event = torch.cuda.Event()
    with _aborting_on_failure(abort):
        event.record(stream)
    bounded_event_sync(event, timeout_s, what, abort=abort)
