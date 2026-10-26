#!/usr/bin/env python
"""The weight-sync bootstraps' wire contracts. vLLM's is pinned against the weight-sync wire golden: the
rendezvous store rank 0 builds, the one write that publishes its NCCL unique id, the id the communicator
then initializes with, and the class the vLLM image's shim must answer for. SGLang's rendezvous store is
pinned directly: the options its group forms on.

Run: python tests/cpu/weight_sync/test_wire_pins.py
"""

import json
import os
import pickle
import re
import socket
from datetime import timedelta

import pytest
import torch

from halo.weight_sync.transport import rendezvous
from halo.weight_sync.transport.c10d_group import build_weight_update_group
from halo.weight_sync.transport.nccl_library import ncclUniqueId
from halo.weight_sync.transport.rendezvous import RendezvousListener
from tests.common.goldens.weight_sync_wire import GOLDEN_PATH, PINNED_ID_BYTES, WORLD_SIZE, bootstrap_record
from tests.common.repo import REPO_ROOT

GOLDEN = json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))
VLLM_DOCKERFILE = REPO_ROOT / "Dockerfile.vllm"
VLLM_VERIFY_SCRIPT = REPO_ROOT / "docker/vllm/verify_image.py"
SHIM_ROOT = "/opt/nccl_compat"
SGLANG_GROUP_TIMEOUT_S = 7.0


class _StoreBuilt(Exception):
    """Ends SGLang's group formation once its store is constructed, before c10d needs a peer."""


@pytest.fixture(scope="module")
def record():
    return bootstrap_record()


def test_the_bootstrap_matches_the_wire_golden(record):
    assert record == GOLDEN


def test_rank_0_publishes_the_id_once_under_the_first_broadcast_key(record):
    """The engine's workers read ``broadcast_from/0/0`` and unpickle it; a second write or another key
    leaves them waiting on a key nothing writes."""
    [[operation, key, value]] = record["store_writes"]
    assert (operation, key) == ("set", "broadcast_from/0/0")
    published = pickle.loads(bytes.fromhex(value))
    assert type(published) is ncclUniqueId and bytes(published) == PINNED_ID_BYTES
    assert record["comm_init"] == [{"world_size": WORLD_SIZE, "unique_id": PINNED_ID_BYTES.hex(), "rank": 0}]


def test_the_store_blocks_until_the_engine_joined(record):
    """A master that does not wait for its workers would hand the id to an engine that has not dialed in."""
    kwargs = record["store"]["kwargs"]
    assert kwargs["is_master"] is True and kwargs["world_size"] == WORLD_SIZE
    assert "wait_for_workers" not in kwargs, "the master no longer waits for the engine's workers"
    assert kwargs["master_listen_fd"] == "<listener fd>", "the store does not listen on the advertised listener"


def test_the_vllm_image_shims_the_module_the_id_pickles_from():
    """The image writes a shim module re-exporting vLLM's own ``ncclUniqueId`` at the trainer's module path,
    creates every package above it, and its build check asserts that the two classes are one."""
    dockerfile = VLLM_DOCKERFILE.read_text(encoding="utf-8")
    module = GOLDEN["unique_id_class"]["module"]
    assert module == ncclUniqueId.__module__
    path = module.replace(".", "/")
    assert f"> {SHIM_ROOT}/{path}.py" in dockerfile, f"Dockerfile.vllm writes no shim at {path}.py"
    packages = module.split(".")[:-1]
    for depth in range(1, len(packages) + 1):
        init = f"{SHIM_ROOT}/{'/'.join(packages[:depth])}/__init__.py"
        assert init in dockerfile, f"Dockerfile.vllm creates no {init}, so the shim cannot import"
    assert str(VLLM_VERIFY_SCRIPT.relative_to(REPO_ROOT)) in dockerfile, "Dockerfile.vllm runs no image build check"
    check = VLLM_VERIFY_SCRIPT.read_text(encoding="utf-8")
    asserted = re.findall(r"^from (\S+) import ncclUniqueId as shim_unique_id$", check, re.M)
    assert asserted == [module], f"the image's build assert checks {asserted}, not {module}"
    assert "assert shim_unique_id is ncclUniqueId" in check, "the build check no longer compares the two classes"


def test_the_sglang_store_does_not_wait_for_the_engine(monkeypatch):
    """SGLang's engine dials in only once it has the join request, which the client sends while the store
    is being built; the master must return without waiting for its workers, serve on the listener the
    engine was told to dial, and share the group's formation deadline."""
    built: list[tuple[dict, int]] = []

    def recording_store(**kwargs):
        listen_fd = kwargs["master_listen_fd"]
        with socket.socket(fileno=os.dup(listen_fd)) as listening:
            built.append((kwargs, listening.getsockname()[1]))
        os.close(listen_fd)  # the store owns the fd it is handed
        raise _StoreBuilt

    monkeypatch.setattr(rendezvous, "TCPStore", recording_store)
    with RendezvousListener("127.0.0.1", 0) as listener, pytest.raises(_StoreBuilt):
        build_weight_update_group(
            "127.0.0.1",
            listener,
            WORLD_SIZE,
            torch.device("cpu"),
            group_name="weight_update_group_test",
            timeout_s=SGLANG_GROUP_TIMEOUT_S,
        )

    [(kwargs, listen_port)] = built
    assert listen_port == listener.port, "the store does not serve on the listener the engine was told to dial"
    assert kwargs == {
        "host_name": "127.0.0.1",
        "port": listener.port,
        "world_size": WORLD_SIZE,
        "is_master": True,
        "timeout": timedelta(seconds=SGLANG_GROUP_TIMEOUT_S),
        "wait_for_workers": False,
        "master_listen_fd": kwargs["master_listen_fd"],
    }


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
