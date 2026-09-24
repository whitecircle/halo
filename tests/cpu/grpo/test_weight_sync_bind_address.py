"""The vLLM weight-sync rendezvous store must listen only on the address the engine is told to dial.

The store is unauthenticated, and the engine unpickles the NCCL bootstrap it reads from it while the
group forms. Bound on every interface, any host that reaches the group port in that window can plant
a pickle and run code in the rollout server. The listener therefore binds the advertised address; a
wide bind is the explicit ``HALO_WEIGHT_SYNC_BIND_ALL`` opt-in, and an advertised address this host
cannot bind alone is refused before the engine is asked to join, never silently widened.

The tests drive the real client and the real ``StatelessProcessGroup`` listener (a group of one, so
no engine peer is needed) and read the bound address off the listening socket.

    python tests/cpu/grpo/test_weight_sync_bind_address.py
"""

import logging
import socket
from unittest.mock import patch

import pytest

import src.distributed.nccl.clients.vllm as vllm_module
from src.distributed.nccl.clients.base import _get_ip, _is_loopback
from src.distributed.nccl.clients.sglang import SGLangWeightSyncClient
from src.distributed.nccl.clients.vllm import VLLMWeightSyncClient
from tests.common.ports import free_port

BIND_ALL_ENV = "HALO_WEIGHT_SYNC_BIND_ALL"
# RFC 5737 documentation range: never an address of the test host.
NON_LOCAL_ADDRESS = "203.0.113.7"


class _Response:
    def __init__(self, payload: dict):
        self._payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


class _GroupHoldingCommunicator:
    """Stand-in for ``PyNcclCommunicator``, which keeps the group (and so its listener) it formed from."""

    def __init__(self, group, device):
        self.group = group

    def abort(self):
        pass


@pytest.fixture
def engine_requests() -> list[dict]:
    """Bodies of the group-join requests the client sent the engine, in order."""
    return []


@pytest.fixture
def vllm_client(monkeypatch, engine_requests):
    """A real client against a local server address, its HTTP half stubbed.

    ``/get_world_size`` answers 0, so the group is the trainer alone: ``create()`` binds the listener
    and returns without waiting for an engine peer.
    """
    monkeypatch.delenv(BIND_ALL_ENV, raising=False)
    monkeypatch.delenv("VLLM_GROUP_HOST", raising=False)
    with patch.object(VLLMWeightSyncClient, "check_server"):
        client = VLLMWeightSyncClient(base_url="http://127.0.0.1:8000")
    monkeypatch.setattr(client, "probe_generation", lambda: None)
    monkeypatch.setattr(
        client, "_post_once", lambda path, **kwargs: engine_requests.append(kwargs["json"]) or _Response({})
    )
    monkeypatch.setattr(vllm_module.requests, "get", lambda url, timeout=None: _Response({"world_size": 0}))
    monkeypatch.setattr(vllm_module, "PyNcclCommunicator", _GroupHoldingCommunicator)
    yield client
    client.close_communicator()


@pytest.mark.parametrize("advertise_nic", [False, True], ids=["loopback", "nic"])
def test_the_listener_binds_the_advertised_address(vllm_client, engine_requests, advertise_nic):
    """A same-host group advertises loopback and takes no connection from the network; a group
    advertising the trainer's NIC (a server on another node) listens on that NIC alone."""
    expected = "127.0.0.1"
    if advertise_nic:
        expected = _get_ip()
        if _is_loopback(expected):
            pytest.skip("no default route: this host has no NIC address to advertise")
        vllm_client.group_host = expected

    vllm_client.init_communicator(device="cpu")

    advertised = engine_requests[0]["init_info"]
    bound = vllm_client._process_group.socket.getsockname()
    assert advertised["master_address"] == expected
    assert bound == (expected, advertised["master_port"]), (
        f"the rendezvous listener bound {bound} while the engine was told to dial "
        f"{advertised['master_address']}:{advertised['master_port']}"
    )


def test_the_opt_in_binds_every_interface(vllm_client, engine_requests, monkeypatch, caplog):
    """``HALO_WEIGHT_SYNC_BIND_ALL`` is the one way to a wide bind, for a NAT or port-mapped trainer, and
    it is announced at WARNING: an operator must see that the unauthenticated store is on the network."""
    monkeypatch.setenv(BIND_ALL_ENV, "1")

    with caplog.at_level(logging.WARNING):
        vllm_client.init_communicator(device="cpu")

    port = engine_requests[0]["init_info"]["master_port"]
    assert vllm_client._process_group.socket.getsockname() == ("0.0.0.0", port)
    warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert any(BIND_ALL_ENV in m and "every interface" in m for m in warnings), (
        f"the wide bind was not announced at WARNING: {warnings}"
    )


@pytest.mark.parametrize("advertised", [NON_LOCAL_ADDRESS, "0.0.0.0"])
def test_an_unbindable_advertised_address_is_refused_before_the_engine_joins(
    vllm_client, engine_requests, monkeypatch, advertised
):
    """A NAT address (not local) or the wildcard (every interface) would need the wide bind, so
    without the opt-in it raises naming the knob, before any listener or engine request exists."""
    formed: list[dict] = []
    monkeypatch.setattr(
        vllm_module.StatelessProcessGroup, "create", staticmethod(lambda **kwargs: formed.append(kwargs))
    )
    vllm_client.group_host = advertised

    with pytest.raises(RuntimeError, match=BIND_ALL_ENV):
        vllm_client.init_communicator(device="cpu")

    assert not formed, f"a rendezvous listener was created for {advertised}: {formed}"
    assert not engine_requests, "the engine was asked to join a group whose listener was refused"


def test_a_name_resolving_to_loopback_is_refused_for_a_remote_server(vllm_client, engine_requests, monkeypatch):
    """A trainer hostname that ``/etc/hosts`` maps to ``127.0.1.1`` would put the listener on loopback,
    where an engine on another host can never connect; it raises instead of timing out the group."""
    formed: list[dict] = []
    monkeypatch.setattr(
        vllm_module.StatelessProcessGroup, "create", staticmethod(lambda **kwargs: formed.append(kwargs))
    )
    resolve = socket.gethostbyname
    monkeypatch.setattr(socket, "gethostbyname", lambda name: "127.0.1.1" if name == "trainer" else resolve(name))
    vllm_client.host = NON_LOCAL_ADDRESS  # the server sits on another host
    vllm_client.group_host = "trainer"

    with pytest.raises(RuntimeError, match=BIND_ALL_ENV):
        vllm_client.init_communicator(device="cpu")

    assert not formed, f"a rendezvous listener was created on loopback for a remote server: {formed}"
    assert not engine_requests, "the engine was asked to dial a listener it cannot reach"


def test_sglang_resolves_a_non_local_address_to_its_every_interface_listener(monkeypatch):
    """SGLang's store is torch's ``TCPStore``, whose master listens on every interface whatever address
    it is given, so a NAT-advertised address stays accepted there without the opt-in."""
    monkeypatch.delenv(BIND_ALL_ENV, raising=False)
    port = free_port()
    with patch.object(SGLangWeightSyncClient, "check_server"):
        client = SGLangWeightSyncClient(
            base_url="http://127.0.0.1:30000", group_port=port, group_host=NON_LOCAL_ADDRESS
        )

    assert client._resolve_group_address() == (NON_LOCAL_ADDRESS, port, "0.0.0.0")


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
