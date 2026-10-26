#!/usr/bin/env python
"""``bounded_event_sync`` is what stands between a wedged peer and a parked trainer.

Both clients drain their broadcasts through it, and it sits on the critical path once per chunk:
its deadline must fire and abort the transport (the group has no watchdog the trainer can rely on, and
a kernel left spinning hangs every later device sync), a wait that fails outright must abort it too,
its poll cadence is a per-chunk tax, and the env knob sets the drain deadline once, at import. The
vLLM communicator's warm-up and the packed producer's drain must each abort their own communicator.

    python tests/cpu/weight_sync/test_bounded_waits.py
"""

import contextlib
import time
from types import SimpleNamespace

import pytest
import torch

from halo.weight_sync.transport import bounded_waits, packed_broadcast, pynccl
from halo.weight_sync.transport.nccl_library import ncclComm_t, ncclUniqueId
from halo.weight_sync.transport.packed_broadcast import PACKED_NUM_BUFFERS, packed_broadcast_producer
from halo.weight_sync.transport.pynccl import PyNcclCommunicator
from tests.common.repo import probe_findings

_DRAIN_MARKER = "DRAIN_DEADLINE:"
_COMM_HANDLE = 0x5EED


class _Event:
    def __init__(self, ready_after_s: float):
        self._ready_at = time.monotonic() + ready_after_s

    def query(self) -> bool:
        return time.monotonic() >= self._ready_at


class _PendingEvent:
    """An event recorded behind a collective whose peer never answers."""

    def record(self, stream=None) -> None:
        pass

    def query(self) -> bool:
        return False


class _FaultedEvent:
    """An event whose ``failing_call`` surfaces a sticky CUDA error left by other work on the device."""

    def __init__(self, failing_call: str):
        self._failing_call = failing_call

    def record(self, stream=None) -> None:
        self._raise_if("record")

    def query(self) -> bool:
        self._raise_if("query")
        return True

    def _raise_if(self, call: str) -> None:
        if call == self._failing_call:
            raise torch.AcceleratorError("CUDA error: an illegal memory access was encountered")


class _Stream:
    device = torch.device("cpu")

    def wait_stream(self, other) -> None:
        pass


class _Communicator:
    def __init__(self):
        self.aborted = False

    def abort(self) -> None:
        self.aborted = True


class _WarmUpNeverCompletesLibrary:
    """libnccl whose communicator forms but whose warm-up all-reduce never completes; records each abort."""

    def __init__(self):
        self.aborted: list[int] = []

    def ncclGetUniqueId(self) -> ncclUniqueId:
        return ncclUniqueId()

    def ncclGetVersion(self) -> str:
        return "2.0.0"

    def ncclCommInitRank(self, world_size, unique_id, rank) -> ncclComm_t:
        return ncclComm_t(_COMM_HANDLE)

    def ncclAllReduce(self, *args) -> None:
        pass

    def ncclCommAbort(self, comm: ncclComm_t) -> None:
        self.aborted.append(comm.value)


def _never_called():
    raise AssertionError("abort ran for a wait that completed")


def test_a_never_ready_event_aborts_and_raises_at_the_deadline_instead_of_parking():
    aborted: list[bool] = []
    started = time.monotonic()
    with pytest.raises(RuntimeError, match="did not complete"):
        bounded_waits.bounded_event_sync(
            _Event(float("inf")), timeout_s=0.05, what="test broadcast", abort=lambda: aborted.append(True)
        )
    assert time.monotonic() - started < 1.0, "the deadline did not bound the wait"
    assert aborted == [True], "the deadline raised without aborting the transport the kernel spins on"


@pytest.mark.parametrize("failing_call", ["record", "query"])
def test_a_wait_that_fails_outright_aborts_once_before_its_error_propagates(monkeypatch, failing_call):
    """A sticky CUDA error leaves the transport as unusable as a deadline does: unaborted, the vLLM
    communicator would not be refused by name and the SGLang group would be destroyed as if healthy."""
    aborted: list[bool] = []
    monkeypatch.setattr(torch.cuda, "Event", lambda: _FaultedEvent(failing_call))
    with pytest.raises(torch.AcceleratorError, match="illegal memory access"):
        bounded_waits.bounded_stream_sync(
            _Stream(), timeout_s=60.0, what="test broadcast", abort=lambda: aborted.append(True)
        )
    assert aborted == [True], f"a failed {failing_call} aborted the transport {len(aborted)} times, not once"


def test_the_poll_cadence_is_kept_at_a_millisecond():
    """A 1 GiB chunk crosses EFA in ~11 ms and the wait runs once per chunk, so a coarse poll
    (50 ms) is a multiple of the transfer itself on every chunk."""
    started = time.monotonic()
    bounded_waits.bounded_event_sync(_Event(0.002), timeout_s=1.0, what="test broadcast", abort=_never_called)
    assert time.monotonic() - started < 0.04, "an event ready after 2 ms took a coarse poll interval to notice"


@pytest.mark.parametrize(("value", "expected"), [(None, 600.0), ("2.5", 2.5), ("0", 600.0), ("soon", 600.0)])
def test_the_drain_deadline_is_read_once_from_the_env(monkeypatch, value, expected):
    """``HALO_NCCL_SYNC_TIMEOUT_SECONDS`` sets the drain deadline at import; a malformed or non-positive
    value falls back rather than raising mid-RL-step. Read in a fresh interpreter, since the module
    latches it."""
    if value is None:
        monkeypatch.delenv("HALO_NCCL_SYNC_TIMEOUT_SECONDS", raising=False)
    else:
        monkeypatch.setenv("HALO_NCCL_SYNC_TIMEOUT_SECONDS", value)
    script = (
        "from halo.weight_sync.transport import bounded_waits; "
        f"print({_DRAIN_MARKER!r} + repr(bounded_waits.BROADCAST_DRAIN_TIMEOUT_S))"
    )
    [deadline] = probe_findings(script, _DRAIN_MARKER)
    assert float(deadline) == expected


def test_a_warm_up_that_passes_its_deadline_aborts_the_communicator(monkeypatch):
    """The warm-up all-reduce is the new communicator's first collective; a peer that never joins it
    leaves the kernel spinning, and every later device sync hangs behind it unless the abort ends it."""
    library = _WarmUpNeverCompletesLibrary()
    monkeypatch.setattr(pynccl, "NCCLLibrary", lambda: library)
    monkeypatch.setattr(pynccl, "vllm_pynccl_disabled", lambda: False)
    monkeypatch.setattr(pynccl, "_WARMUP_SYNC_TIMEOUT_S", 0.0)
    monkeypatch.setattr(torch.cuda, "device", lambda device: contextlib.nullcontext())
    monkeypatch.setattr(torch.cuda, "current_stream", lambda device=None: SimpleNamespace(cuda_stream=0))
    monkeypatch.setattr(torch.cuda, "Event", _PendingEvent)
    store = SimpleNamespace(world_size=2, publish=lambda unique_id: None)

    with pytest.raises(RuntimeError, match="warm-up all-reduce did not complete"):
        PyNcclCommunicator(store, device=torch.device("cpu"))
    assert library.aborted == [_COMM_HANDLE], "the warm-up deadline did not abort the communicator it formed"


def test_a_packed_drain_that_passes_its_deadline_aborts_the_communicator(monkeypatch):
    """The producer waits out a pack stream before reusing its buffer; a peer that stopped receiving
    leaves the broadcast queued there spinning, and only the communicator's abort ends it."""
    communicator = _Communicator()
    monkeypatch.setattr(packed_broadcast, "BROADCAST_DRAIN_TIMEOUT_S", 0.0)
    monkeypatch.setattr(torch.cuda, "Event", _PendingEvent)
    monkeypatch.setattr(torch.cuda, "current_stream", lambda device=None: _Stream())
    real_empty = torch.empty
    # The producer's placeholder buffers name the CUDA device, which a CPU run does not have.
    monkeypatch.setattr(torch, "empty", lambda *args, device=None, **kwargs: real_empty(*args, **kwargs))

    with pytest.raises(RuntimeError, match="packed weight broadcast did not complete"):
        packed_broadcast_producer(
            iter([("w", torch.ones(4))]), communicator, streams=[_Stream() for _ in range(PACKED_NUM_BUFFERS)]
        )
    assert communicator.aborted, "the drain deadline did not abort the communicator its broadcasts run on"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
