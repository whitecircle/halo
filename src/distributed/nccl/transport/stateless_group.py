"""TCPStore-based NCCL unique-ID exchange that does not pollute global torch.distributed state. Vendored from vLLM v0.18.0, Apache-2.0."""

import dataclasses
import pickle
import socket
import threading
import time
from collections import deque
from collections.abc import Iterator
from contextlib import contextmanager, nullcontext
from datetime import timedelta
from typing import Any

import torch
from torch.distributed import TCPStore

# How long a broadcast entry may sit in the store for slow receivers before the sender reaps it.
_DATA_EXPIRATION_SECONDS = 3600

# TCPStore rendezvous deadline. The group publishes one NCCL unique id and is torn down, so this
# bounds the id handshake alone, not DIST_STORE_TIMEOUT_HOURS' whole-job coordination waits.
_STORE_TIMEOUT_SECONDS = 300


class RendezvousListener:
    """An IPv4 socket bound and listening on ``bind_address:port`` for a ``TCPStore`` master to take over.

    Opened before the peers are told where to dial, and held from the bind on: a port probed and
    released first can be taken in between, and a peer retrying against a port nobody holds can be
    handed that port as its own source (a TCP self-connect), after which the bind fails with the peer
    already waiting. ``port`` 0 lets the kernel assign one; :attr:`port` is the port actually bound.

    Handed over as ``master_listen_fd`` (:meth:`handover`), the fd is the store's only listener; a
    master that opens its own listens on every interface whatever host it is given. Until then the
    socket is this object's, and :meth:`close` (or leaving its ``with`` block) frees the port.
    """

    def __init__(self, bind_address: str, port: int):
        listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            listener.bind((bind_address, port))
            listener.listen()
        except BaseException:
            listener.close()
            raise
        self.bind_address = bind_address
        self.port: int = listener.getsockname()[1]
        self.handed_over = False
        self._socket: socket.socket | None = listener
        # The SGLang client hands the listener over on its group-formation thread while its failure
        # path may close it from the caller's.
        self._lock = threading.Lock()

    @contextmanager
    def handover(self) -> Iterator[int]:
        """Yield the fd to the one ``TCPStore(master_listen_fd=...)`` call that takes it over.

        The store owns the fd from that call on, closing it in its destructor and on a failed start,
        so the socket is detached on exit rather than closed: closing it under a live store aborts the
        store's daemon thread, and after a failed start it would close whatever reused the fd number.
        The store must be given :attr:`port`, which it checks the fd against.
        """
        with self._lock:
            listener, self._socket = self._socket, None
            if listener is None:
                state = "handed to a store" if self.handed_over else "closed"
                raise RuntimeError(f"The rendezvous listener on {self.bind_address}:{self.port} was already {state}.")
            self.handed_over = True
        try:
            yield listener.fileno()
        finally:
            listener.detach()

    def close(self) -> None:
        """Close the socket and free the port, unless a store took it over; idempotent."""
        with self._lock:
            listener, self._socket = self._socket, None
        if listener is not None:
            listener.close()

    def __enter__(self) -> "RendezvousListener":
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()


@dataclasses.dataclass
class StatelessProcessGroup:
    """NCCL unique-ID publication via TCPStore (only broadcast_obj is kept)."""

    rank: int
    world_size: int
    store: torch._C._distributed_c10d.Store | None

    data_expiration_seconds: int = _DATA_EXPIRATION_SECONDS
    broadcast_send_counter: int = 0
    entries: deque[tuple[str, float]] = dataclasses.field(default_factory=deque)

    def __post_init__(self):
        assert self.rank < self.world_size

    def broadcast_obj(self, obj: Any, src: int) -> Any:
        """Publish ``obj`` under ``src``'s next broadcast key for the engine ranks to read.

        Send-only: the trainer holds rank 0 of this group and every exchange it drives has ``src=0``,
        so a receiving rank here means the group was built wrong.
        """
        if self.store is None:
            raise RuntimeError("StatelessProcessGroup is closed — build a new group to exchange objects.")
        if self.rank != src:
            raise RuntimeError(
                f"StatelessProcessGroup.broadcast_obj is send-only, but rank {self.rank} was asked to "
                f"receive from rank {src}. The trainer must own this group as rank 0."
            )
        while self.entries and time.time() - self.entries[0][1] > self.data_expiration_seconds:
            self.store.delete_key(self.entries.popleft()[0])
        key = f"broadcast_from/{src}/{self.broadcast_send_counter}"
        self.store.set(key, pickle.dumps(obj))
        self.broadcast_send_counter += 1
        self.entries.append((key, time.time()))
        return obj

    @staticmethod
    def create(
        host: str,
        port: int,
        rank: int,
        world_size: int,
        listener: RendezvousListener | None = None,
    ) -> "StatelessProcessGroup":
        """Create a StatelessProcessGroup without polluting global torch.distributed state.

        Rank 0 hosts the store on ``listener``, already listening on ``port`` (its
        :attr:`RendezvousListener.port`): the master blocks until every rank has joined, so the other
        ranks are told where to dial before this call, and the listener is what they reach meanwhile.
        """
        launch_server = rank == 0
        if launch_server and listener is None:
            raise ValueError("Rank 0 hosts the store on a RendezvousListener opened before its peers dial it.")
        if launch_server and listener.port != port:
            # The store checks the fd against ``port``, and a mismatch fails its start after taking the fd.
            raise ValueError(f"Rank 0 was given port {port}, but its rendezvous listener holds {listener.port}.")
        handover = listener.handover() if launch_server else nullcontext()
        with handover as listen_fd:
            store = TCPStore(
                host_name=host,
                port=port,
                world_size=world_size,
                is_master=launch_server,
                timeout=timedelta(seconds=_STORE_TIMEOUT_SECONDS),
                use_libuv=False,
                master_listen_fd=listen_fd,
            )

        return StatelessProcessGroup(rank=rank, world_size=world_size, store=store)

    def close(self) -> None:
        """Release rank 0's listener so ``port`` can be rebound; idempotent, and a no-op off rank 0.

        The store owns the listening fd (:meth:`RendezvousListener.handover`), so dropping the last store
        reference runs its destructor, which stops the daemon and closes the fd. Dropping the
        reference here rather than at garbage-collection time makes the release deterministic even
        while another holder of this group (a thread parked in ``ncclCommInitRank``, a traceback) is
        still alive.
        """
        self.store = None
