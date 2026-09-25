"""TCPStore-based NCCL unique-ID exchange that does not pollute global torch.distributed state. Vendored from vLLM v0.18.0, Apache-2.0."""

import dataclasses
import pickle
import socket
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


@contextmanager
def rendezvous_listener(bind_address: str, port: int) -> Iterator[int]:
    """Yield the fd of an IPv4 socket listening on ``bind_address:port``, for a ``TCPStore`` master.

    Handed over as ``master_listen_fd``, the fd is the store's only listener; a master that opens its
    own listens on every interface whatever host it is given. The store owns the fd from that call
    on, closing it in its destructor and on a failed start, so the socket is detached on exit rather
    than closed: closing it under a live store aborts the store's daemon thread, and after a failed
    start it would close whatever reused the fd number.
    """
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listener.bind((bind_address, port))
        listener.listen()
    except BaseException:
        listener.close()
        raise
    try:
        yield listener.fileno()
    finally:
        listener.detach()


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
        bind_address: str | None = None,
    ) -> "StatelessProcessGroup":
        """Create a StatelessProcessGroup without polluting global torch.distributed state.

        ``bind_address`` (default ``host``) is the address rank 0's listener binds: ``host`` resolved,
        or every interface where the caller opted in for an advertised address that is not local
        (NAT, a port mapping).
        """
        launch_server = rank == 0
        listener = rendezvous_listener(bind_address or host, port) if launch_server else nullcontext()
        with listener as listen_fd:
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

        The store owns the listening fd (:func:`rendezvous_listener`), so dropping the last store
        reference runs its destructor, which stops the daemon and closes the fd. Dropping the
        reference here rather than at garbage-collection time makes the release deterministic even
        while another holder of this group (a thread parked in ``ncclCommInitRank``, a traceback) is
        still alive.
        """
        self.store = None
