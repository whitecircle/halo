#!/usr/bin/env python
"""The vLLM weight-transfer bootstrap's wire contract, pinned against the weight-sync wire golden: the
rendezvous store rank 0 builds, the one write that publishes its NCCL unique id, the id the communicator
then initializes with, and the class the vLLM image's shim must answer for.

Run: python tests/cpu/weight_sync/test_wire_pins.py
"""

import json
import pickle
import re

import pytest

from src.distributed.nccl._nccl_wrapper import ncclUniqueId
from tests.common.goldens.weight_sync_wire import GOLDEN_PATH, PINNED_ID_BYTES, WORLD_SIZE, bootstrap_record
from tests.common.repo import REPO_ROOT

GOLDEN = json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))
VLLM_DOCKERFILE = REPO_ROOT / "Dockerfile.vllm"
SHIM_ROOT = "/opt/nccl_compat"


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
    creates every package above it, and asserts at build that the two classes are one."""
    dockerfile = VLLM_DOCKERFILE.read_text(encoding="utf-8")
    module = GOLDEN["unique_id_class"]["module"]
    assert module == ncclUniqueId.__module__
    path = module.replace(".", "/")
    assert f"> {SHIM_ROOT}/{path}.py" in dockerfile, f"Dockerfile.vllm writes no shim at {path}.py"
    packages = module.split(".")[:-1]
    for depth in range(1, len(packages) + 1):
        init = f"{SHIM_ROOT}/{'/'.join(packages[:depth])}/__init__.py"
        assert init in dockerfile, f"Dockerfile.vllm creates no {init}, so the shim cannot import"
    asserted = re.findall(r"from (\S+) import ncclUniqueId; from vllm\.\S+ import ncclUniqueId as v", dockerfile)
    assert asserted == [module], f"the image's build assert checks {asserted}, not {module}"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
