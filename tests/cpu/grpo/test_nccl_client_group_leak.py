"""Both weight-sync clients must release what they already built when the SERVER half of group
formation fails.

``_wait_for_calls`` raises the moment either half of the handshake errors, so the local half's
product — SGLang's c10d group + rendezvous TCPStore, vLLM's ncclComm — can already exist while the
client attribute that owns it is still ``None``. Leaking it is not a spare object: c10d keeps the
process-global registration under the derived group name and the store keeps its listener on the
group port, so ``reconnect_client`` rebuilds the same name and torch refuses it ("The specified
group name has already been created") — the server becomes permanently unreachable for weight sync
without restarting the trainer. The vLLM twin leaks an ncclComm and the device memory NCCL pinned
for it for the life of the process.

The first two tests drive the failure with the local half completing FIRST (an event, not a sleep),
which is exactly the window the release has to cover. The third covers the other end of that window:
the local half never starting at all.

    python tests/cpu/grpo/test_nccl_client_group_leak.py
"""

import threading
from unittest.mock import patch

import pytest

import src.distributed.nccl.clients.sglang as sglang_module
import src.distributed.nccl.clients.vllm as vllm_module
from src.distributed.nccl.clients.sglang import SGLangWeightSyncClient
from src.distributed.nccl.clients.vllm import VLLMWeightSyncClient

_SERVER_REFUSED = "server refused the group-init request"


class _FakeResponse:
    def __init__(self, payload: dict):
        self._payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


def _failing_server_call(built: threading.Event):
    """Stand-in for the engine's group-init POST: fails only once the local half has succeeded."""

    def _post_once(path, **kwargs):
        assert built.wait(timeout=30), "the local half never completed — test would not exercise the leak window"
        raise RuntimeError(_SERVER_REFUSED)

    return _post_once


def test_sglang_releases_a_group_it_formed_when_the_server_half_fails(monkeypatch):
    with patch.object(SGLangWeightSyncClient, "check_server"):
        client = SGLangWeightSyncClient(base_url="http://localhost:30000")

    group, store = object(), object()
    built = threading.Event()
    destroyed: list[object] = []

    def fake_create(**kwargs):
        built.set()
        return group, store

    monkeypatch.setattr(client, "fetch_engine_world_size", lambda: 1)
    monkeypatch.setattr(client, "_destroy_remote_group", lambda: None)
    monkeypatch.setattr(client, "_post_once", _failing_server_call(built))
    monkeypatch.setattr(sglang_module, "create_weight_update_group", fake_create)
    monkeypatch.setattr(sglang_module, "destroy_weight_update_group", destroyed.append)

    with pytest.raises(RuntimeError, match=_SERVER_REFUSED):
        client.init_communicator(device="cpu")

    assert destroyed == [group], (
        f"the formed c10d group was not destroyed (destroy called with {destroyed}) — its process-global "
        f"registration under the derived group name survives and every later reconnect to this server "
        f"is refused with 'group name has already been created'"
    )
    assert client._group is None and client._store is None


def test_sglang_releases_the_group_when_the_local_half_cannot_even_start(monkeypatch):
    """``_AsyncCall.__init__`` starts a thread, and a thread-starved process (Ray actors + rollout
    threads) raises there. The handler must still release and report THAT error — reading the
    unbound call handle instead would raise UnboundLocalError, replacing the diagnosis and skipping
    the release that keeps the group name reusable."""
    with patch.object(SGLangWeightSyncClient, "check_server"):
        client = SGLangWeightSyncClient(base_url="http://localhost:30000")

    destroyed: list[object] = []
    real_async_call = sglang_module._AsyncCall

    def no_threads(*, name, fn):
        if name == "weight-update group formation":
            raise RuntimeError("can't start new thread")
        return real_async_call(name=name, fn=fn)

    monkeypatch.setattr(client, "fetch_engine_world_size", lambda: 1)
    monkeypatch.setattr(client, "_destroy_remote_group", lambda: None)
    monkeypatch.setattr(client, "_post_once", lambda path, **kwargs: _FakeResponse({}))
    monkeypatch.setattr(sglang_module, "_AsyncCall", no_threads)
    monkeypatch.setattr(sglang_module, "destroy_weight_update_group", destroyed.append)

    with pytest.raises(RuntimeError, match="can't start new thread"):
        client.init_communicator(device="cpu")

    assert destroyed == [None], "the failure handler skipped _release_group, so the group name stays claimed"
    assert client._group is None and client._store is None


def test_vllm_aborts_a_communicator_it_built_when_the_server_half_fails(monkeypatch):
    with patch.object(VLLMWeightSyncClient, "check_server"):
        client = VLLMWeightSyncClient(base_url="http://localhost:8000")

    built = threading.Event()
    aborted: list[str] = []
    closed: list[str] = []

    class _FakeCommunicator:
        def __init__(self, group, device):
            built.set()

        def abort(self):
            aborted.append("abort")

    class _FakeGroup:
        @staticmethod
        def create(**kwargs):
            return _FakeGroup()

        def close(self):
            closed.append("close")

    monkeypatch.setattr(client, "probe_generation", lambda *a, **kw: None)
    monkeypatch.setattr(client, "_post_once", _failing_server_call(built))
    monkeypatch.setattr(vllm_module.requests, "get", lambda url, timeout=None: _FakeResponse({"world_size": 1}))
    monkeypatch.setattr(vllm_module, "StatelessProcessGroup", _FakeGroup)
    monkeypatch.setattr(vllm_module, "PyNcclCommunicator", _FakeCommunicator)

    with pytest.raises(RuntimeError, match=_SERVER_REFUSED):
        client.init_communicator(device="cpu")

    assert aborted == ["abort"], (
        "the built NCCL communicator was never aborted — its ncclComm and the device memory NCCL "
        "pinned for it leak for the life of the trainer process"
    )
    assert closed == ["close"], "the rendezvous group must still be closed so its port can be rebound"
    assert client.communicator is None


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
