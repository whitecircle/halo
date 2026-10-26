"""Weight-sync wire golden: what rank 0's bootstrap of a vLLM weight-transfer group hands its rendezvous
store and writes there, and the class its NCCL unique id pickles as. The vLLM image unpickles that id as
its own class through a shim at the same module path (``Dockerfile.vllm``), so the path, qualname, fields,
pickle bytes and store key are a contract with a separately built image: a commit that moves the class
regenerates this golden and moves the shim with it.
"""

from __future__ import annotations

import ctypes
import json
import os
import socket
from contextlib import nullcontext
from datetime import timedelta
from pathlib import Path
from unittest.mock import patch

import torch

from src.distributed.nccl._nccl_wrapper import ncclUniqueId
from src.distributed.nccl.transport import pynccl, stateless_group
from src.distributed.nccl.transport.pynccl import PyNcclCommunicator
from src.distributed.nccl.transport.stateless_group import RendezvousListener, StatelessProcessGroup
from tests.common.goldens import regen_main

GOLDEN_PATH = Path(__file__).with_suffix(".json")
# 128 bytes that are their own offsets, so the golden shows the id carried verbatim and in order.
PINNED_ID_BYTES = bytes(range(128))
# A vLLM server of one worker: the trainer's group has two ranks.
WORLD_SIZE = 2
GROUP_HOST = "127.0.0.1"


class _EngineNeverJoins(Exception):
    """Ends the bootstrap where a real ``ncclCommInitRank`` blocks until the engine read the id and joined."""


class _RecordingStore:
    """The ``TCPStore`` rank 0 builds: records its arguments and every write, and owns the listener fd
    it is handed, as the real store does."""

    def __init__(self, *args, **kwargs):
        self.args, self.kwargs = args, kwargs
        self.writes: list[tuple[str, str, bytes]] = []
        self.listen_port: int | None = None
        fd = kwargs.get("master_listen_fd")
        if fd is not None:
            with socket.socket(fileno=os.dup(fd)) as listening:
                self.listen_port = listening.getsockname()[1]
            os.close(fd)

    def set(self, key: str, value: bytes) -> None:
        self.writes.append(("set", key, value))

    def delete_key(self, key: str) -> bool:
        self.writes.append(("delete_key", key, b""))
        return True


class _PinnedIdLibrary:
    """libnccl as rank 0's bootstrap calls it: hands out the pinned id and records the communicator init."""

    def __init__(self):
        self.inits: list[dict] = []

    def ncclGetUniqueId(self) -> ncclUniqueId:
        return ncclUniqueId(internal=(ctypes.c_byte * len(PINNED_ID_BYTES)).from_buffer_copy(PINNED_ID_BYTES))

    def ncclGetVersion(self) -> str:
        return "2.0.0"

    def ncclCommInitRank(self, world_size: int, unique_id: ncclUniqueId, rank: int):
        self.inits.append({"world_size": world_size, "unique_id": bytes(unique_id).hex(), "rank": rank})
        raise _EngineNeverJoins


def _ctype_spelling(ctype) -> str:
    if issubclass(ctype, ctypes.Array):
        return f"{ctype._type_.__name__}[{ctype._length_}]"
    return ctype.__name__


def _store_arguments(store: _RecordingStore, listener: RendezvousListener) -> dict:
    """The store's keyword arguments, with the listener's port and fd written as what they are."""
    kwargs = dict(store.kwargs)
    if kwargs.get("port") == listener.port:
        kwargs["port"] = "<listener port>"
    if "master_listen_fd" in kwargs and store.listen_port == listener.port:
        kwargs["master_listen_fd"] = "<listener fd>"
    if isinstance(kwargs.get("timeout"), timedelta):
        kwargs["timeout"] = f"{kwargs['timeout'].total_seconds():g}s"
    return {"args": [repr(arg) for arg in store.args], "kwargs": kwargs}


def bootstrap_record() -> dict:
    """Rank 0's bootstrap of a two-rank group on a recording store and a faked libnccl: the store's
    arguments, its writes up to the communicator init, and the init's arguments; plus the unique id
    class's module, qualname and fields."""
    library = _PinnedIdLibrary()
    listener = RendezvousListener(GROUP_HOST, 0)
    with (
        listener,
        # The server's knob, which the communicator refuses in the trainer: a host export must not end
        # the record early.
        patch.object(pynccl, "vllm_pynccl_disabled", lambda: False),
        patch.object(stateless_group, "TCPStore", _RecordingStore),
        patch.object(pynccl, "NCCLLibrary", lambda: library),
        patch.object(torch.cuda, "device", lambda device: nullcontext()),
    ):
        group = StatelessProcessGroup.create(
            host=GROUP_HOST, port=listener.port, rank=0, world_size=WORLD_SIZE, listener=listener
        )
        store = group.store
        try:
            PyNcclCommunicator(group, device=torch.device("cuda", 0))
        except _EngineNeverJoins:
            pass
        else:
            raise AssertionError("the communicator was built without initializing NCCL")
    return {
        "unique_id_class": {
            "module": ncclUniqueId.__module__,
            "qualname": ncclUniqueId.__qualname__,
            "fields": [[name, _ctype_spelling(ctype)] for name, ctype in ncclUniqueId._fields_],
        },
        "store": _store_arguments(store, listener),
        "store_writes": [[operation, key, value.hex()] for operation, key, value in store.writes],
        "comm_init": library.inits,
    }


def render_golden() -> str:
    return json.dumps(bootstrap_record(), indent=2) + "\n"


if __name__ == "__main__":
    regen_main(GOLDEN_PATH, render_golden)
