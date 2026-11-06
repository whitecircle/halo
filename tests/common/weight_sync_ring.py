"""Real SGLang and vLLM clients over fake engine control planes with one shared stand-in for the group's ring, which
blocks as an NCCL broadcast does until the trainer's half is aborted."""

import threading
from typing import Any

import torch

from halo.weight_sync.clients.base import WeightSyncClientBase
from halo.weight_sync.clients.sglang import SGLangWeightSyncClient
from halo.weight_sync.clients.vllm import VLLMWeightSyncClient
from halo.weight_sync.transport.pynccl import UniqueIdStore
from tests.common.fake_engines import FakeEngineServer, FakeSGLangServer, FakeVLLMServer


class FakeRing:
    """What the leader's broadcast crosses. A send completes once ``peers_ready`` is set and blocks until then, or until
    :meth:`abort`, as a broadcast does with a peer that never posted its receive."""

    def __init__(self, *, ready: bool = True):
        self.chunks: list[list[str]] = []
        self.peers_ready = threading.Event()
        if ready:
            self.peers_ready.set()
        self.aborted = threading.Event()
        self.released = 0

    def form_group(self) -> None:
        """A re-formed group is a new communicator, which the previous one's abort does not reach."""
        self.aborted.clear()

    def send(self, chunk: list[tuple[str, torch.Tensor]]) -> None:
        self.chunks.append([name for name, _ in chunk])
        while not (self.peers_ready.is_set() or self.aborted.is_set()):
            self.aborted.wait(0.01)
        if self.aborted.is_set():
            raise RuntimeError("the ring was aborted mid-broadcast")

    def abort(self) -> None:
        self.aborted.set()


class _FakeComm:
    def __init__(self, ring: FakeRing):
        self.aborted = False
        self.device = torch.device("cpu")
        self._ring = ring

    def abort(self) -> None:
        self.aborted = True
        self._ring.abort()


class RingSGLangClient(SGLangWeightSyncClient):
    """The control plane is real; the transport is :attr:`ring`."""

    ring: FakeRing

    def prepare_local_formation(self, master_address, listener, world_size):
        self.ring.form_group()
        return lambda: object()

    def require_transport(self) -> None:
        if self._transport is None:
            raise RuntimeError("Call init_communicator() first")

    def send_chunk(self, chunk) -> None:
        self.ring.send(chunk)

    def abort_transport(self) -> None:
        self.ring.abort()

    def _release_transport(self, *, at_exit: bool) -> None:
        self.ring.released += 1
        if not self.group_releases_remote_half:
            self.destroy_remote_group()


class RingVLLMClient(VLLMWeightSyncClient):
    """The control plane and the rendezvous store are real; the communicator is :attr:`ring`."""

    ring: FakeRing

    def prepare_local_formation(self, master_address, listener, world_size):
        self._unique_id_store = UniqueIdStore.create(host=master_address, listener=listener, world_size=world_size)
        self.ring.form_group()
        return lambda: _FakeComm(self.ring)

    def prepare_chunk_send(self, chunk) -> None:
        self.require_transport()

    def send_chunk(self, chunk) -> None:
        self.ring.send(chunk)

    def abort_transport(self) -> None:
        if self._transport is not None:
            self._transport.abort()

    def _release_transport(self, *, at_exit: bool) -> None:
        self.ring.released += 1
        if self._unique_id_store is not None:
            self._unique_id_store.close()
            self._unique_id_store = None


def ring_client_class(engine: str, ring: FakeRing) -> type[WeightSyncClientBase]:
    """The ring client of ``engine`` (``"sglang"`` or ``"vllm"``), bound to ``ring``."""
    base = {"sglang": RingSGLangClient, "vllm": RingVLLMClient}[engine]
    return type(f"Ring{base.__name__}", (base,), {"ring": ring})


def fake_servers(engine: str, worlds: list[int]) -> list[FakeEngineServer]:
    """One fake server per entry of ``worlds``, each reporting that many engine ranks."""
    servers: list[Any] = []
    for world in worlds:
        server = FakeSGLangServer() if engine == "sglang" else FakeVLLMServer()
        if engine == "sglang":
            server.tp_size = world
        else:
            server.world_size = world
            server.instance_id = "launch-0"
        servers.append(server)
    return servers
