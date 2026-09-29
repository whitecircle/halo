"""The weight-sync rendezvous store must listen only on the address the engine is told to dial, and
already be listening there when the engine is told.

The store is unauthenticated, and the engine reads the NCCL bootstrap from it while the group forms
(vLLM unpickles it). Bound on every interface, any host that reaches the group port in that window
can plant a bootstrap of its own. The listener therefore binds the advertised address on both
engines, and the engine is sent that resolved address rather than a name it might resolve
differently; a wide bind is the explicit ``HALO_WEIGHT_SYNC_BIND_ALL`` opt-in, and an advertised
address this host cannot bind alone is refused before the engine is asked to join, never silently
widened.

The listener is bound before the engine's join request goes out. Told first, the engine dials a port
nobody holds: another socket can take a probed-and-released port in between, and the engine's own
retried connect can be handed that port as its source (a TCP self-connect), so the trainer's bind
fails with the engine already waiting on a group that never forms.

The tests drive the real clients and their real rendezvous stores (a group of one, so no engine
peer is needed) and read the bound address off the kernel's listener table, as ``ss -ltn`` does.
The engine's join request runs inline, at the moment the client issues it, so what it records is
what a real engine would find when it dials in.

    python tests/cpu/grpo/test_weight_sync_bind_address.py
"""

import logging
import socket
import sys
from pathlib import Path
from typing import NamedTuple
from unittest.mock import patch

import pytest
from torch.distributed import distributed_c10d as c10d

import src.distributed.nccl.clients.sglang as sglang_module
import src.distributed.nccl.clients.vllm as vllm_module
from src.distributed.nccl.addresses import is_loopback
from src.distributed.nccl.clients.base import _get_ip
from src.distributed.nccl.clients.sglang import SGLangWeightSyncClient
from src.distributed.nccl.clients.vllm import VLLMWeightSyncClient
from src.distributed.nccl.transport.stateless_group import RendezvousListener
from tests.common.ports import free_port

BIND_ALL_ENV = "HALO_WEIGHT_SYNC_BIND_ALL"
# RFC 5737 documentation range: never an address of the test host.
NON_LOCAL_ADDRESS = "203.0.113.7"
_TCP_LISTEN = "0A"


def _listening_addresses(port: int) -> list[str]:
    """Every address the kernel holds a TCP listener on at ``port``, IPv4 and IPv6 alike."""
    addresses = []
    for table, family in (("/proc/net/tcp", socket.AF_INET), ("/proc/net/tcp6", socket.AF_INET6)):
        for row in Path(table).read_text().splitlines()[1:]:
            fields = row.split()
            address_hex, port_hex = fields[1].split(":")
            if int(port_hex, 16) != port or fields[3] != _TCP_LISTEN:
                continue
            raw = bytes.fromhex(address_hex)
            # The kernel prints the address as 32-bit words in host (little-endian) byte order.
            packed = b"".join(raw[i : i + 4][::-1] for i in range(0, len(raw), 4))
            addresses.append(socket.inet_ntop(family, packed))
    return addresses


class _JoinRequest(NamedTuple):
    """A group-join request the engine received, and who listened on its port when it arrived."""

    address: str
    port: int
    listening: list[str]


class _InlineCall:
    """``_AsyncCall`` run on the issuing thread, so the call happens at the moment it is issued."""

    def __init__(self, fn, name: str = "server call"):
        self.name, self.result, self.error = name, None, None
        try:
            self.result = fn()
        except Exception as e:
            self.error = e

    def is_alive(self) -> bool:
        return False

    def join(self, grace: float) -> "_InlineCall":
        return self

    def wait(self, timeout: float = 0.0):
        if self.error:
            raise self.error


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


def _group_without_nccl(*args, **kwargs):
    """Stand-in for the c10d NCCL group, which needs a GPU; the rendezvous store is built before it."""
    return object(), None


def _recording_join(engine_requests: list[_JoinRequest], server_calls: list[str]):
    def post_once(path, **kwargs):
        server_calls.append(path)
        body = kwargs["json"]
        group = body.get("init_info", body)  # vLLM nests the group fields, SGLang does not
        port = group["master_port"]
        engine_requests.append(_JoinRequest(group["master_address"], port, _listening_addresses(port)))
        return _Response({})

    return post_once


@pytest.fixture
def engine_requests() -> list[_JoinRequest]:
    """Each group-join request the client sent the engine, in order."""
    return []


@pytest.fixture
def server_calls() -> list[str]:
    """Route of every request the client sent the engine while forming the group, in order."""
    return []


@pytest.fixture
def opened_listeners(monkeypatch) -> list[RendezvousListener]:
    """Every rendezvous listener either client opened, in order."""
    opened: list[RendezvousListener] = []
    real_init = RendezvousListener.__init__

    def recording(self, bind_address, port):
        real_init(self, bind_address, port)
        opened.append(self)

    monkeypatch.setattr(RendezvousListener, "__init__", recording)
    return opened


@pytest.fixture(params=["vllm", "sglang"])
def client(request, monkeypatch, engine_requests, server_calls, opened_listeners):
    """A real client of each engine against a local server address, its HTTP half stubbed.

    The engine reports no ranks, so the group is the trainer alone: the rendezvous store takes the
    listener and returns without waiting for an engine peer.
    """
    monkeypatch.delenv(BIND_ALL_ENV, raising=False)
    if request.param == "vllm":
        monkeypatch.delenv("VLLM_GROUP_HOST", raising=False)
        with patch.object(VLLMWeightSyncClient, "check_server"):
            client = VLLMWeightSyncClient(base_url="http://127.0.0.1:8000")
        monkeypatch.setattr(client, "probe_generation", lambda: None)
        monkeypatch.setattr(vllm_module.requests, "get", lambda url, timeout=None: _Response({"world_size": 0}))
        monkeypatch.setattr(vllm_module, "PyNcclCommunicator", _GroupHoldingCommunicator)
        monkeypatch.setattr(vllm_module, "_AsyncCall", _InlineCall)
    else:
        monkeypatch.delenv("SGLANG_GROUP_HOST", raising=False)
        with patch.object(SGLangWeightSyncClient, "check_server"):
            client = SGLangWeightSyncClient(base_url="http://127.0.0.1:30000")
        monkeypatch.setattr(client, "fetch_engine_world_size", lambda: 0)
        monkeypatch.setattr(
            client, "_destroy_remote_group", lambda: server_calls.append("/destroy_weights_update_group")
        )
        monkeypatch.setattr(c10d, "_new_process_group_helper", _group_without_nccl)
        monkeypatch.setattr(sglang_module, "_AsyncCall", _InlineCall)
    monkeypatch.setattr(client, "_post_once", _recording_join(engine_requests, server_calls))
    yield client
    client.close_communicator()


@pytest.mark.parametrize("advertise_nic", [False, True], ids=["loopback", "nic"])
def test_the_listener_binds_the_advertised_address(client, engine_requests, advertise_nic):
    """A same-host group advertises loopback and takes no connection from the network; a group
    advertising the trainer's NIC (a server on another node) listens on that NIC alone."""
    expected = "127.0.0.1"
    if advertise_nic:
        expected = _get_ip()
        if is_loopback(expected):
            pytest.skip("no default route: this host has no NIC address to advertise")
        client.group_host = expected

    client.init_communicator(device="cpu")

    [(advertised, port, _)] = engine_requests
    assert advertised == expected
    assert _listening_addresses(port) == [expected], (
        f"the rendezvous store listens on {_listening_addresses(port)} while the engine was told to "
        f"dial {advertised}:{port}"
    )


def test_a_name_is_advertised_as_the_address_the_listener_binds(client, engine_requests):
    """The engine resolves a name on its own host, where it can map elsewhere; it is sent the address
    the listener took, so it dials the listener rather than wherever the name points there."""
    client.group_host = "localhost"

    client.init_communicator(device="cpu")

    [(advertised, port, _)] = engine_requests
    assert advertised == "127.0.0.1", f"the engine was sent {advertised!r}, not the address the listener binds"
    assert _listening_addresses(port) == ["127.0.0.1"]


def test_the_opt_in_binds_every_interface(client, engine_requests, monkeypatch, caplog):
    """``HALO_WEIGHT_SYNC_BIND_ALL`` is the one way to a wide bind, for a NAT or port-mapped trainer, and
    it is announced at WARNING: an operator must see that the unauthenticated store is on the network."""
    monkeypatch.setenv(BIND_ALL_ENV, "1")

    with caplog.at_level(logging.WARNING):
        client.init_communicator(device="cpu")

    [(_, port, _)] = engine_requests
    assert _listening_addresses(port) == ["0.0.0.0"]
    warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert any(BIND_ALL_ENV in m and "every interface" in m for m in warnings), (
        f"the wide bind was not announced at WARNING: {warnings}"
    )


@pytest.mark.parametrize("advertised", [NON_LOCAL_ADDRESS, "0.0.0.0"])
def test_an_unbindable_advertised_address_is_refused_before_the_engine_joins(
    client, engine_requests, opened_listeners, advertised
):
    """A NAT address (not local) or the wildcard (every interface) would need the wide bind, so
    without the opt-in it raises naming the knob, before any listener or engine request exists."""
    client.group_host = advertised

    with pytest.raises(RuntimeError, match=BIND_ALL_ENV):
        client.init_communicator(device="cpu")

    assert not opened_listeners, f"a rendezvous listener was opened for {advertised}: {opened_listeners}"
    assert not engine_requests, "the engine was asked to join a group whose listener was refused"


def test_a_name_resolving_to_loopback_is_refused_for_a_remote_server(
    client, engine_requests, opened_listeners, monkeypatch
):
    """A trainer hostname that ``/etc/hosts`` maps to ``127.0.1.1`` would put the listener on loopback,
    where an engine on another host can never connect; it raises instead of timing out the group."""
    resolve = socket.gethostbyname
    monkeypatch.setattr(socket, "gethostbyname", lambda name: "127.0.1.1" if name == "trainer" else resolve(name))
    client.host = NON_LOCAL_ADDRESS  # the server sits on another host
    client.group_host = "trainer"

    with pytest.raises(RuntimeError, match=BIND_ALL_ENV):
        client.init_communicator(device="cpu")

    assert not opened_listeners, f"a listener was opened on loopback for a remote server: {opened_listeners}"
    assert not engine_requests, "the engine was asked to dial a listener it cannot reach"


@pytest.mark.parametrize("bind_all", [False, True], ids=["narrow", "bind_all"])
def test_an_ipv6_group_address_is_refused_as_unsupported(
    client, engine_requests, opened_listeners, monkeypatch, bind_all
):
    """The listener is IPv4-only, the wide one included, so an IPv6 group address is refused as
    unsupported in either mode, without pointing at an opt-in that could not serve it."""
    if bind_all:
        monkeypatch.setenv(BIND_ALL_ENV, "1")
    client.group_host = "::1"

    with pytest.raises(RuntimeError, match="IPv6 is unsupported") as refused:
        client.init_communicator(device="cpu")

    assert BIND_ALL_ENV not in str(refused.value), f"the refusal suggests the IPv4-only opt-in: {refused.value}"
    assert not opened_listeners, f"a rendezvous listener was opened for an IPv6 address: {opened_listeners}"
    assert not engine_requests, "the engine was asked to dial an IPv6 address no listener takes"


def test_a_configured_port_is_listening_when_the_engine_is_told_to_dial_it(client, engine_requests):
    """Told before the listener exists, the engine dials a port nobody holds, where its retried connect
    can take the port as its own source and fail the trainer's bind with the engine already waiting."""
    client.group_port = free_port()

    client.init_communicator(device="cpu")

    [request] = engine_requests
    assert request.port == client.group_port
    assert request.listening == ["127.0.0.1"], (
        f"the engine was told to dial 127.0.0.1:{request.port} while listeners there were {request.listening}"
    )


def test_an_unset_port_is_the_one_the_kernel_gave_the_live_listener(client, engine_requests, opened_listeners):
    """``group_port`` 0 binds port 0 and advertises what the kernel assigned, the port the listener
    already holds; a probed-and-released port could be taken before anything binds it again."""
    client.init_communicator(device="cpu")

    [listener] = opened_listeners
    [request] = engine_requests
    assert (request.port, request.listening) == (listener.port, ["127.0.0.1"]), (
        f"the engine was told port {request.port} (listeners {request.listening}) while the live listener "
        f"holds {listener.port}"
    )
    assert listener.handed_over, "the store did not take over the listener the engine was told about"
    assert _listening_addresses(listener.port) == ["127.0.0.1"], "the store does not serve the advertised port"


def test_a_failure_before_the_store_takes_the_listener_closes_it(
    client, engine_requests, opened_listeners, monkeypatch
):
    """A listener the store never took is closed, not detached: the port is free again, and the engine
    was never asked to join."""
    port = client.group_port = free_port()
    held_at_failure: list[list[str]] = []

    def no_thread(fn, name: str = "server call"):
        held_at_failure.append(_listening_addresses(port))
        raise RuntimeError("can't start new thread")

    monkeypatch.setattr(sys.modules[type(client).__module__], "_AsyncCall", no_thread)

    with pytest.raises(RuntimeError, match="can't start new thread"):
        client.init_communicator(device="cpu")

    assert held_at_failure == [["127.0.0.1"]], f"the port was not held when the join was issued: {held_at_failure}"
    [listener] = opened_listeners
    assert not listener.handed_over
    assert not engine_requests, "the engine was asked to join a group whose listener was closed"
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as rebind:
        rebind.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        rebind.bind(("127.0.0.1", port))  # EADDRINUSE while a detached listener still holds the port


def test_a_configured_port_already_held_raises_before_the_engine_is_asked_anything(client, server_calls):
    """The bind comes first, so a port another socket holds fails before any request reaches the engine,
    and the error names the port rather than a topology knob."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as holder:
        holder.bind(("127.0.0.1", 0))
        holder.listen()
        port = client.group_port = holder.getsockname()[1]

        with pytest.raises(RuntimeError, match="Could not bind the weight-transfer group port") as refused:
            client.init_communicator(device="cpu")

    assert f"127.0.0.1:{port}" in str(refused.value)
    assert "was not asked to join" in str(refused.value)
    assert not server_calls, f"the engine was sent {server_calls} for a group whose port could not be bound"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
