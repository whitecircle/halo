"""A weight-sync chunk staged once into the buffers a broadcast reads, for every communicator that sends it. Memory
held for staging is one chunk (the largest tensor, when it rides alone), however many servers receive it."""

import threading
import weakref
from abc import ABC, abstractmethod
from collections.abc import Sequence
from contextlib import AbstractContextManager, nullcontext

import torch

from halo.weight_sync.chunking import WEIGHT_SYNC_CHUNK_BYTES
from halo.weight_sync.transport.packed_broadcast import allocate_send_buffer, hold_for_stream, pack_tensors


def _on(stream: torch.cuda.Stream | None) -> AbstractContextManager:
    return torch.cuda.stream(stream) if stream is not None else nullcontext()


def _record_ready(stream: torch.cuda.Stream | None) -> torch.cuda.Event | None:
    if stream is None:
        return None
    ready = torch.cuda.Event()
    ready.record(stream)
    return ready


class StagedChunk:
    """The tensors one chunk puts on the wire, in order, read-only to every sender. A sender queues its reads after
    :attr:`ready`, holds :attr:`wire` for its stream and reports the event its last read retires on."""

    def __init__(self, sources: Sequence[torch.Tensor], wire: list[torch.Tensor], ready: torch.cuda.Event | None):
        self.wire = wire
        self.ready = ready
        # Weak, so a staged chunk neither pins its snapshots nor matches a later sync's tensor at a reused address.
        self._sources = [weakref.ref(tensor) for tensor in sources]
        self._readers: list[torch.cuda.Event] = []
        self._lock = threading.Lock()

    @property
    def readers(self) -> list[torch.cuda.Event]:
        """The events recorded after each sender's reads of :attr:`wire`."""
        with self._lock:
            return list(self._readers)

    def add_reader(self, event: torch.cuda.Event) -> None:
        with self._lock:
            self._readers.append(event)

    def stages(self, tensors: Sequence[torch.Tensor]) -> bool:
        """Whether these are the very tensors this was staged from."""
        return len(tensors) == len(self._sources) and all(
            ref() is tensor for ref, tensor in zip(self._sources, tensors, strict=True)
        )


class ChunkStager(ABC):
    """Stages each chunk once and hands the same :class:`StagedChunk` to every caller of it, so concurrent senders
    share one buffer. A new chunk replaces the last, and :meth:`release` ends a sync."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._staged: StagedChunk | None = None

    @property
    @abstractmethod
    def staged_bytes(self) -> int:
        """Device bytes this stager holds for staging."""

    @abstractmethod
    def _stage(self, tensors: Sequence[torch.Tensor], readers: list[torch.cuda.Event]) -> StagedChunk:
        """Stage ``tensors`` on the current stream. ``readers`` are the events of the senders of the chunk this one
        replaces, which a reused buffer must wait on."""

    @abstractmethod
    def _drop_buffers(self) -> None:
        """Free what is held across chunks."""

    def stage(self, chunk: Sequence[tuple[str, torch.Tensor]]) -> StagedChunk:
        """The staging of ``chunk``, made by whichever caller comes first. A snapshot must be complete."""
        tensors = [tensor for _, tensor in chunk]
        with self._lock:
            if self._staged is not None and self._staged.stages(tensors):
                return self._staged
            readers = [] if self._staged is None else self._staged.readers
            self._staged = None
            self._staged = self._stage(tensors, readers)
            return self._staged

    def release(self) -> None:
        """Drop the last chunk and every buffer; the senders must have queued their reads (the allocator keeps a block
        another stream still reads)."""
        with self._lock:
            self._staged = None
            self._drop_buffers()


class PackedChunkStager(ChunkStager):
    """A chunk packed into byte buffers of the packed wire format, one fresh allocation per chunk."""

    @property
    def staged_bytes(self) -> int:
        return 0 if self._staged is None else sum(buffer.nbytes for buffer in self._staged.wire)

    def _stage(self, tensors: Sequence[torch.Tensor], readers: list[torch.cuda.Event]) -> StagedChunk:
        stream = torch.cuda.current_stream(tensors[0].device) if tensors[0].is_cuda else None
        if stream is not None:
            for tensor in tensors:
                hold_for_stream(tensor, stream)
        return StagedChunk(tensors, pack_tensors(tensors), _record_ready(stream))

    def _drop_buffers(self) -> None:
        """Nothing is held between chunks beyond the last one."""


class ArenaChunkStager(ChunkStager):
    """A chunk copied into one arena that every chunk of a sync reuses, as a block freed mid-chunk carries a pending
    NCCL event the next upload would wait on inside the allocator. The copies run on a stream of their own that waits
    for the previous chunk's senders."""

    def __init__(self) -> None:
        super().__init__()
        self._arena: torch.Tensor | None = None
        self._stream: torch.cuda.Stream | None = None

    @property
    def staged_bytes(self) -> int:
        return 0 if self._arena is None else self._arena.nbytes

    def _stage(self, tensors: Sequence[torch.Tensor], readers: list[torch.cuda.Event]) -> StagedChunk:
        device = tensors[0].device
        offsets: list[int] = []
        total = 0
        for tensor in tensors:
            offsets.append(total)
            # 256-byte aligned, a multiple of every dtype's size, so a uint8 slice views as the tensor.
            total += (tensor.nbytes + 255) // 256 * 256
        caller = None
        if device.type == "cuda":
            caller = torch.cuda.current_stream(device)
            if self._stream is None:
                self._stream = torch.cuda.Stream(device=device)
            # The arena's block may last have been used on the caller's stream (``allocate_send_buffer``), and a
            # reused arena is still being read by the previous chunk's senders.
            self._stream.wait_stream(caller)
            for reader in readers:
                self._stream.wait_event(reader)
        if self._arena is None or self._arena.numel() < total:
            size = max(total, WEIGHT_SYNC_CHUNK_BYTES)
            if caller is None:
                self._arena = torch.empty(size, dtype=torch.uint8, device=device)
            else:
                self._arena = allocate_send_buffer(size, device, pool_stream=caller, send_stream=self._stream)
        views = []
        with _on(self._stream if caller is not None else None):
            for tensor, offset in zip(tensors, offsets, strict=True):
                view = self._arena[offset : offset + tensor.nbytes].view(tensor.dtype).view(tensor.shape)
                view.copy_(tensor, non_blocking=True)
                if caller is not None:
                    hold_for_stream(tensor, self._stream)
                views.append(view)
        return StagedChunk(tensors, views, _record_ready(self._stream if caller is not None else None))

    def _drop_buffers(self) -> None:
        self._arena = None
