"""Both weight-sync clients must release what they already built when the SERVER half of group
formation fails.

``wait_for_calls`` raises the moment either half of the handshake errors, so the local half's
product — SGLang's c10d group + rendezvous TCPStore, vLLM's ncclComm — can already exist while the
client attribute that owns it is still ``None``. Leaking it is not a spare object: c10d keeps the
process-global registration under the derived group name and the store keeps its listener on the
group port, so a reconnect rebuilds the same name and torch refuses it ("The specified
group name has already been created") — the server becomes permanently unreachable for weight sync
without restarting the trainer. The vLLM twin leaks an ncclComm and the device memory NCCL pinned
for it for the life of the process.

The first two tests drive the failure with the local half completing FIRST (an event, not a sleep),
which is exactly the window the release has to cover. The third covers the other end of that window:
the local half never starting at all.

    python tests/cpu/weight_sync/test_group_leak.py
"""

import socket
import threading

import pytest

import halo.weight_sync.clients.base as base_module
import halo.weight_sync.clients.sglang as sglang_module
import halo.weight_sync.clients.vllm as vllm_module
from halo.weight_sync.clients.sglang import SGLangWeightSyncClient
from halo.weight_sync.clients.vllm import VLLMWeightSyncClient
from tests.common.ports import free_port
from tests.common.weight_sync import offline_client

_SERVER_REFUSED = "server refused the group-init request"


def _recording_destroy(destroyed: list):
    """A ``release_weight_update_group`` that records each group it was asked to destroy."""

    def release(group, *, destroy):
        if destroy:
            destroyed.append(group)

    return release


def _failing_server_call(built: threading.Event):
    """Stand-in for the engine's group-init POST: it fails only once the local half has succeeded."""

    def request(method, path, **kwargs):
        assert built.wait(timeout=30), "the local half never completed — test would not exercise the leak window"
        raise RuntimeError(_SERVER_REFUSED)

    return request


def test_sglang_releases_a_group_it_formed_when_the_server_half_fails(monkeypatch):
    client = offline_client(SGLangWeightSyncClient, "http://localhost:30000")

    group, store = object(), object()
    built = threading.Event()
    destroyed: list[object] = []

    def fake_create(**kwargs):
        built.set()
        return group, store

    monkeypatch.setattr(client, "_fetch_engine_world_size", lambda: 1)
    monkeypatch.setattr(client, "_destroy_remote_group", lambda: None)
    monkeypatch.setattr(client, "request", _failing_server_call(built))
    monkeypatch.setattr(sglang_module, "build_weight_update_group", fake_create)
    monkeypatch.setattr(sglang_module, "release_weight_update_group", _recording_destroy(destroyed))

    with pytest.raises(RuntimeError, match=_SERVER_REFUSED):
        client.init_communicator(device="cpu")

    assert destroyed == [group], (
        f"the formed c10d group was not destroyed (destroy called with {destroyed}) — its process-global "
        f"registration under the derived group name survives and every later reconnect to this server "
        f"is refused with 'group name has already been created'"
    )
    assert client._transport is None


def test_sglang_releases_the_group_when_the_local_half_cannot_even_start(monkeypatch):
    """``AsyncCall.__init__`` starts a thread, and a thread-starved process (Ray actors + rollout
    threads) raises there. The handler must still release and report THAT error — reading the
    unbound call handle instead would raise UnboundLocalError, replacing the diagnosis and skipping
    the release that keeps the group name reusable. The listener no store took is closed with it."""
    port = free_port()
    client = offline_client(SGLangWeightSyncClient, "http://localhost:30000", group_port=port)

    destroyed: list[object] = []
    real_async_call = base_module.AsyncCall

    def no_threads(fn, name="server call"):
        if name == "weight-update group formation":
            raise RuntimeError("can't start new thread")
        return real_async_call(fn, name=name)

    monkeypatch.setattr(client, "_fetch_engine_world_size", lambda: 1)
    monkeypatch.setattr(client, "_destroy_remote_group", lambda: None)
    monkeypatch.setattr(client, "request", lambda method, path, **kwargs: None)
    monkeypatch.setattr(base_module, "AsyncCall", no_threads)
    monkeypatch.setattr(sglang_module, "release_weight_update_group", _recording_destroy(destroyed))

    with pytest.raises(RuntimeError, match="can't start new thread"):
        client.init_communicator(device="cpu")

    assert destroyed == [None], "the failure handler skipped _release_transport, so the group name stays claimed"
    assert client._transport is None
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as rebind:
        rebind.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        rebind.bind(("127.0.0.1", port))  # EADDRINUSE while the listener no store took is still open


def test_vllm_aborts_a_communicator_it_built_when_the_server_half_fails(monkeypatch):
    client = offline_client(VLLMWeightSyncClient, "http://localhost:8000")

    built = threading.Event()
    aborted: list[str] = []
    closed: list[str] = []

    class _FakeCommunicator:
        def __init__(self, store, device):
            built.set()

        def abort(self):
            aborted.append("abort")

    class _FakeStore:
        @staticmethod
        def create(**kwargs):
            return _FakeStore()

        def close(self):
            closed.append("close")

    monkeypatch.setattr(client, "_fetch_engine_world_size", lambda: 1)
    monkeypatch.setattr(client, "request", _failing_server_call(built))
    monkeypatch.setattr(vllm_module, "UniqueIdStore", _FakeStore)
    monkeypatch.setattr(vllm_module, "PyNcclCommunicator", _FakeCommunicator)

    with pytest.raises(RuntimeError, match=_SERVER_REFUSED):
        client.init_communicator(device="cpu")

    assert aborted == ["abort"], (
        "the built NCCL communicator was never aborted — its ncclComm and the device memory NCCL "
        "pinned for it leak for the life of the trainer process"
    )
    assert closed == ["close"], "the unique-id store must still be closed so its port can be rebound"
    assert client._transport is None


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
