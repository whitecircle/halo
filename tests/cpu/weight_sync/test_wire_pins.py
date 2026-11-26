#!/usr/bin/env python
"""The weight-sync bootstraps' wire contracts. vLLM's is pinned against a golden: the store rank 0 builds, the write
publishing its NCCL unique id, and the class the image's shim answers for. SGLang's store is pinned directly."""

import json
import os
import pickle
import re
import socket
from datetime import timedelta

import pytest
import torch
from torch.distributed import distributed_c10d as c10d

from halo.weight_sync.clients import sglang as sglang_module
from halo.weight_sync.clients.sglang import SGLangWeightSyncClient
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
SGLANG_STORE_TIMEOUT_S = 7.0
SGLANG_GROUP_TIMEOUT_S = 700.0


class _StoreBuilt(Exception):
    """Ends SGLang's group formation once its store is constructed, before c10d needs a peer."""


@pytest.fixture(scope="module")
def record():
    return bootstrap_record()


def test_the_bootstrap_matches_the_wire_golden(record):
    assert record == GOLDEN


def test_rank_0_publishes_the_id_once_under_the_first_broadcast_key(record):
    """The engine's workers read ``broadcast_from/0/0`` and unpickle it; another key leaves them waiting forever."""
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
    """The shim re-exports vLLM's ``ncclUniqueId`` at the trainer's module path; the build check asserts one class."""
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
    """SGLang's engine dials in only once it has the join request, sent while the store is being built."""
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
            store_timeout_s=SGLANG_STORE_TIMEOUT_S,
            group_timeout_s=SGLANG_GROUP_TIMEOUT_S,
        )

    [(kwargs, listen_port)] = built
    assert listen_port == listener.port, "the store does not serve on the listener the engine was told to dial"
    assert kwargs == {
        "host_name": "127.0.0.1",
        "port": listener.port,
        "world_size": WORLD_SIZE,
        "is_master": True,
        "timeout": timedelta(seconds=SGLANG_STORE_TIMEOUT_S),
        "wait_for_workers": False,
        "master_listen_fd": kwargs["master_listen_fd"],
    }


def test_the_sglang_group_outlives_the_formation_wait(monkeypatch):
    """torch applies a group's timeout to every later work item, so the formation wait must not be the group's."""
    groups: list[timedelta] = []

    def recording_helper(*args, timeout, **kwargs):
        groups.append(timeout)
        raise _StoreBuilt

    monkeypatch.setattr(c10d, "_new_process_group_helper", recording_helper)
    with RendezvousListener("127.0.0.1", 0) as listener, pytest.raises(_StoreBuilt):
        build_weight_update_group(
            "127.0.0.1",
            listener,
            WORLD_SIZE,
            torch.device("cpu"),
            group_name="weight_update_group_test",
            store_timeout_s=SGLANG_STORE_TIMEOUT_S,
            group_timeout_s=SGLANG_GROUP_TIMEOUT_S,
        )
    assert groups == [timedelta(seconds=SGLANG_GROUP_TIMEOUT_S)]


def test_the_sglang_client_gives_its_group_the_chunk_deadline_and_its_formation_the_formation_deadline(monkeypatch):
    built: list[dict] = []
    monkeypatch.setattr(sglang_module, "build_weight_update_group", lambda **kwargs: built.append(kwargs))
    monkeypatch.setattr(sglang_module, "GROUP_FORMATION_TIMEOUT_S", 11.0)
    monkeypatch.setattr(sglang_module, "CHUNK_SYNC_TIMEOUT_S", 777.0)
    client = SGLangWeightSyncClient.__new__(SGLangWeightSyncClient)
    client.sync_device, client.group_name = torch.device("cpu"), "g"
    client.prepare_local_formation("127.0.0.1", None, 2)()
    [kwargs] = built
    assert (kwargs["store_timeout_s"], kwargs["group_timeout_s"]) == (11.0, 777.0)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
