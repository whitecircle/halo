"""One weight update streamed to rollout servers, shared by a single server's client and by a group of servers: the chunk
buffer and the open, send and close of an update, over the engine steps a subclass supplies."""

import logging
from abc import ABC, abstractmethod
from collections.abc import Iterable

import torch

from halo.weight_sync.chunking import declared_co_load_groups, split_co_loaded

logger = logging.getLogger(__name__)


class UpdateStream(ABC):
    """Snapshots buffered by reference, cut into chunks and sent into an update that opens with the first chunk and
    closes with the tail. A fact declared by annotation alone has no default, so a subclass that leaves one out fails
    by name where it is read."""

    BACKEND_NAME: str
    # Per-module parameter suffixes the engine's loader takes only together, so they ride one chunk.
    CO_LOADED_PARAM_GROUPS: tuple[tuple[str, ...], ...] = ()

    def __init__(self) -> None:
        # Set once an interrupted sync left the engine unservable; every later sync is refused with it.
        self._unusable_reason: str | None = None
        self._reset_buffer_state()

    @abstractmethod
    def _start_engine_update(self) -> None:
        """Quiesce the engine(s) and open their update; a failure leaves nothing paused and the sync replayable."""

    @abstractmethod
    def _broadcast_chunk(self, chunk: list[tuple[str, torch.Tensor]], final: bool) -> None:
        """Declare and broadcast one chunk, returning once the engine(s) have it; ``final`` marks the last."""

    @abstractmethod
    def _finish_engine_update(self, tail: list[tuple[str, torch.Tensor]]) -> None:
        """Send ``tail`` (possibly empty) as the final chunk and end the update's session."""

    @abstractmethod
    def _mark_unusable(self) -> None:
        """Refuse every later sync and leave the engine(s) paused, as they hold neither policy. Idempotent."""

    @abstractmethod
    def resume_engine(self) -> None:
        """Lift the pause after a sync; never raises on a server failure, which must not mask the real error."""

    @abstractmethod
    def require_resumed(self) -> None:
        """Raise when the pause stayed after a good push, as every rollout would then queue."""

    @abstractmethod
    def abort_weight_update(self) -> None:
        """Drop the buffer and close an update the sync cannot finish; never raises. Once a chunk went out the
        engine(s) hold neither policy and stay paused."""

    @abstractmethod
    def close_communicator(self, *, at_exit: bool = False) -> None:
        """Close an interrupted update, lift the pause and release the transport; ``at_exit`` is the exit hook's call."""

    @property
    def buffered_bytes(self) -> int:
        """Bytes buffered for the next chunk, members held for a co-load partner included."""
        return self._buffered_bytes

    @property
    def can_replay_sync(self) -> bool:
        """False once a chunk is on the wire: the trainer keeps no copy, so a replay would send only the tail."""
        return self._chunks_sent == 0

    def scope_co_load_groups(self, module_names: Iterable[str]) -> None:
        """Narrow the co-load groups to the members the pushed model's modules declare."""
        self._co_load_groups = declared_co_load_groups(self.CO_LOADED_PARAM_GROUPS, module_names)

    def buffer_param(self, name: str, snapshot: torch.Tensor) -> None:
        """Buffer a snapshot by reference for the next chunk; a pool shares it across streams, read-only."""
        self._param_buffer.append((name, snapshot))
        self._buffered_bytes += snapshot.nbytes
        if snapshot.is_cuda:
            # Re-recorded per param on the copy's own stream, so the newest record covers every copy before it.
            if self._snapshot_event is None:
                self._snapshot_event = torch.cuda.Event()
            self._snapshot_event.record(torch.cuda.current_stream(snapshot.device))

    def drain_param_buffer(self) -> list[tuple[str, torch.Tensor]]:
        buffered = self._param_buffer
        self._param_buffer = []
        self._buffered_bytes = 0
        return buffered

    def complete_snapshots(self) -> None:
        """Wait on the newest copy's event; a device synchronize would also wait on every server's broadcasts."""
        if self._snapshot_event is not None:
            self._snapshot_event.synchronize()

    def flush_chunk(self) -> None:
        """Send what is buffered as one chunk into the open update; an incomplete co-load group stays buffered."""
        chunk, held = split_co_loaded(self._param_buffer, self._co_load_groups)
        if not chunk:
            return
        # Completed before the quiesce, so a sticky CUDA error here leaves the engine serving.
        self.complete_snapshots()
        self._open_update()
        self._transmit_chunk(chunk)
        # Dropped only once on the wire, as a replayable failed chunk is re-sent by the reconnect path.
        self._param_buffer = held
        self._buffered_bytes = sum(snapshot.nbytes for _, snapshot in held)

    def finish_weight_update(self) -> None:
        """Send the tail chunk, close the update and resume the engine(s); no-op if nothing was buffered."""
        if not self._param_buffer and not self._update_open:
            return
        # A group still incomplete at the tail never completes, and the engine drops a lone half silently.
        _, held = split_co_loaded(self._param_buffer, self._co_load_groups)
        if held:
            raise RuntimeError(
                f"the weight sync ended with co-loaded parameters missing their partners: "
                f"{[name for name, _ in held]}. The {self.BACKEND_NAME} loader discards a half that "
                f"arrives on its own, so this update cannot be closed as complete."
            )
        self._open_update()
        self._close_update(self._param_buffer)
        self.drain_param_buffer()  # only once the tail landed
        # A later failure before the next open concerns an engine this sync finished with.
        self._chunks_sent = 0

    def _reset_buffer_state(self) -> None:
        self._param_buffer: list[tuple[str, torch.Tensor]] = []
        self._buffered_bytes = 0
        self._co_load_groups = self.CO_LOADED_PARAM_GROUPS
        self._update_open = False
        # Non-zero means the engine is partly rewritten.
        self._chunks_sent = 0
        self._snapshot_event: torch.cuda.Event | None = None

    def _open_update(self) -> None:
        if self._update_open:
            return
        if self._unusable_reason is not None:
            raise RuntimeError(self._unusable_reason)
        # Before the engine is touched, so a failure to open leaves the sync replayable.
        self._chunks_sent = 0
        self._start_engine_update()
        self._update_open = True

    def _transmit_chunk(self, chunk: list[tuple[str, torch.Tensor]], final: bool = False) -> None:
        """Put one chunk on the wire, counting it and disowning the engine when it fails mid-update."""
        if not chunk:
            return
        try:
            self._broadcast_chunk(chunk, final)
        except BaseException:
            # Marked before the close's finally would resume the engine.
            if self._chunks_sent:
                self._mark_unusable()
            raise
        self._chunks_sent += 1

    def _close_update(self, tail: list[tuple[str, torch.Tensor]]) -> None:
        """Send the final chunk, close the update and resume, whatever the tail does. ``_chunks_sent`` stays, so a
        failed close still reports the sync unreplayable."""
        snapshot_error: Exception | None = None
        try:
            self.complete_snapshots()
        except Exception as e:
            # Only the close can end the open update. The tail goes, as its copies may not have landed.
            logger.error(f"Weight-sync snapshots did not complete: {e}")
            snapshot_error, tail = e, []
            if self._chunks_sent:
                # Dropping the tail after earlier chunks landed leaves a mixed model, as a failed chunk does.
                self._mark_unusable()
        try:
            self._finish_engine_update(tail)
        except BaseException:
            # A close that failed after chunks landed leaves the reload unfinished: the pause stays, as for a failed
            # chunk, and is marked before the finally would lift it.
            if self._chunks_sent:
                self._mark_unusable()
            raise
        finally:
            self._update_open = False
            self.resume_engine()
        if snapshot_error is not None:
            raise snapshot_error
        self.require_resumed()
