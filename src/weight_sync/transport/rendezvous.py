"""Where a weight-update group's rendezvous store listens: the address the engine is told to dial, and a
socket already bound there before it is told."""

import errno
import logging
import socket
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import timedelta

from torch.distributed import TCPStore

from src.env import env_flag, env_str
from src.inference.endpoints import is_loopback

logger = logging.getLogger(__name__)

_ALL_INTERFACES = "0.0.0.0"
_BIND_ALL_ENV = "HALO_WEIGHT_SYNC_BIND_ALL"
# UDP "connect" target used only to make the kernel pick this host's outbound interface, whose local
# address is then its routable IP. No packet is sent, so the address only has to be off-link and
# non-loopback; a missing route falls back to loopback in :func:`_outbound_ip`.
_OUTBOUND_ROUTE_PROBE = ("8.8.8.8", 80)


class RendezvousListener:
    """An IPv4 socket bound and listening on ``bind_address:port`` for a ``TCPStore`` master to take over.

    Opened before the peers are told where to dial, and held from the bind on: a port probed and
    released first can be taken in between, and a peer retrying against a port nobody holds can be
    handed that port as its own source (a TCP self-connect), after which the bind fails with the peer
    already waiting. ``port`` 0 lets the kernel assign one; :attr:`port` is the port actually bound.

    Handed to the store :meth:`start_store` builds, the fd is the store's only listener; a master that
    opens its own listens on every interface whatever host it is given. Until then the socket is this
    object's, and :meth:`close` (or leaving its ``with`` block) frees the port.
    """

    def __init__(self, bind_address: str, port: int):
        listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            listener.bind((bind_address, port))
            listener.listen()
        except BaseException:
            listener.close()
            raise
        self.bind_address = bind_address
        self.port: int = listener.getsockname()[1]
        self.handed_over = False
        self._socket: socket.socket | None = listener
        # The SGLang client hands the listener over on its group-formation thread while its failure
        # path may close it from the caller's.
        self._lock = threading.Lock()

    def start_store(self, host: str, world_size: int, *, timeout_s: float, **options) -> TCPStore:
        """The master ``TCPStore`` of a group of ``world_size`` ranks, serving on this listener.

        ``options`` are the store's own keyword options (``use_libuv``, ``wait_for_workers``), set by each
        group's bootstrap to what its engine side expects.
        """
        with self._handover() as listen_fd:
            return TCPStore(
                host_name=host,
                port=self.port,
                world_size=world_size,
                is_master=True,
                timeout=timedelta(seconds=timeout_s),
                **options,
                master_listen_fd=listen_fd,
            )

    @contextmanager
    def _handover(self) -> Iterator[int]:
        """Yield the fd to the one ``TCPStore(master_listen_fd=...)`` call that takes it over.

        The store owns the fd from that call on, closing it in its destructor and on a failed start,
        so the socket is detached on exit rather than closed: closing it under a live store aborts the
        store's daemon thread, and after a failed start it would close whatever reused the fd number.
        The store must be given :attr:`port`, which it checks the fd against.
        """
        with self._lock:
            listener, self._socket = self._socket, None
            if listener is None:
                state = "handed to a store" if self.handed_over else "closed"
                raise RuntimeError(f"The rendezvous listener on {self.bind_address}:{self.port} was already {state}.")
            self.handed_over = True
        try:
            yield listener.fileno()
        finally:
            listener.detach()

    def close(self) -> None:
        """Close the socket and free the port, unless a store took it over; idempotent."""
        with self._lock:
            listener, self._socket = self._socket, None
        if listener is not None:
            listener.close()

    def __enter__(self) -> "RendezvousListener":
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()


def _outbound_ip() -> str:
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        try:
            s.connect(_OUTBOUND_ROUTE_PROBE)
            return s.getsockname()[0]
        except OSError:
            return "127.0.0.1"


def _is_local_address(host: str) -> bool:
    """True when ``host`` is a bindable address of this machine (covers same-host-by-routable-IP)."""
    if is_loopback(host):
        return True
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind((host, 0))
        return True
    except OSError:
        return False


def _master_address(
    *, engine: str, server_url: str, server_host: str, advertised_host: str | None, host_env: str
) -> str:
    """The address the engine's workers dial back to, as configured or detected.

    Resolution order: ``advertised_host``, the engine's group-host env var ``host_env``, loopback when
    the server is on this machine, then the default-route NIC. Raises when a remote server would be
    told to dial a loopback address, which forms no group and times out.
    """
    master_address = advertised_host or (env_str(host_env) if host_env else None)
    if not master_address:
        # Same-host groups stay on loopback: firewalls drop hairpin traffic on the external NIC.
        master_address = "127.0.0.1" if _is_local_address(server_host) else _outbound_ip()

    if is_loopback(master_address) and not _is_local_address(server_host):
        raise RuntimeError(
            f"{engine} server {server_url} is on a remote host ({server_host}) but the "
            f"NCCL weight-sync group address resolved to loopback ({master_address}). The server's "
            f"workers would dial back to themselves and the group would never form. Set "
            f"{host_env or 'the group host'} (or the per-server group_host) to the "
            f"trainer's routable IP on the subnet the server can reach."
        )
    return master_address


def _bind_address(master_address: str, *, engine: str, server_host: str, host_env: str) -> str:
    """The local address the rendezvous listener binds: the advertised ``master_address``, resolved.

    The store is unauthenticated and the engine reads the group's bootstrap from it, so it listens
    only where the engine is told to dial. A name resolves once, here, to the one IPv4 address the
    listener and the engine both use; the listener is IPv4-only, so an address with no IPv4 form
    (an IPv6 literal, a name without an A record) raises whatever the bind mode.
    ``HALO_WEIGHT_SYNC_BIND_ALL`` widens the listener to every interface for a trainer reached
    through NAT or a port mapping, whose advertised address is not its own. Without it, an address
    that is not local, the wildcard, or a name resolving to loopback while the server is remote
    raises rather than widening.
    """
    host_knob = f"{host_env or 'the group host'} (or the per-server group_host)"
    try:
        bind_address = socket.gethostbyname(master_address)
    except OSError:
        raise RuntimeError(
            f"{engine} weight-sync group address {master_address} has no IPv4 address on "
            f"this host. The rendezvous store listens on IPv4 only; IPv6 is unsupported. Set "
            f"{host_knob} to this host's IPv4 address on an interface the server reaches."
        ) from None
    if env_flag(_BIND_ALL_ENV):
        logger.warning(
            f"{_BIND_ALL_ENV} is set: the {engine} weight-sync rendezvous store "
            f"listens on every interface, not only on {master_address}. The store is unauthenticated and "
            f"the engine reads the group's bootstrap from it while the group forms (vLLM unpickles it), "
            f"so any host that reaches the group port then can run code in the rollout server. Keep the "
            f"port reachable only by trusted hosts."
        )
        return _ALL_INTERFACES
    if (
        bind_address == _ALL_INTERFACES
        or not _is_local_address(bind_address)
        or (is_loopback(bind_address) and not _is_local_address(server_host))
    ):
        raise RuntimeError(
            f"{engine} weight-sync group address {master_address} (resolves to "
            f"{bind_address}) is not a single address of this host that the server at {server_host} can "
            f"reach. The group's rendezvous store is unauthenticated, so it listens only on the address "
            f"it advertises. Set {host_knob} to this host's IP on an interface the server reaches. For "
            f"a trainer reached through NAT or a port mapping, set {_BIND_ALL_ENV}=1 to "
            f"listen on every interface, only on a network where no untrusted host can reach the group port."
        )
    return bind_address


def open_group_rendezvous(
    *, engine: str, server_url: str, server_host: str, advertised_host: str | None, host_env: str, port: int
) -> tuple[str, RendezvousListener]:
    """``(master address, listener)``: the group's rendezvous, listening before the engine hears of it.

    The listener binds the resolved master address (:func:`_master_address`, :func:`_bind_address`)
    and ``port``, or a port the kernel assigns when it is 0, and the engine is told the port it holds
    (:class:`RendezvousListener`). Outside ``HALO_WEIGHT_SYNC_BIND_ALL`` the engine is sent the bind
    address rather than the name it was resolved from, since a name the server resolves differently
    would be dialed where nothing listens. A configured port another socket holds raises here, before
    the engine is asked anything, rather than leaving it waiting on a group that cannot form. The
    caller hands the listener to the store, or closes it.
    """
    master_address = _master_address(
        engine=engine,
        server_url=server_url,
        server_host=server_host,
        advertised_host=advertised_host,
        host_env=host_env,
    )
    bind_address = _bind_address(master_address, engine=engine, server_host=server_host, host_env=host_env)
    if bind_address != _ALL_INTERFACES:
        master_address = bind_address
    try:
        listener = RendezvousListener(bind_address, max(port, 0))
    except OSError as e:
        if port > 0 and e.errno == errno.EADDRINUSE:
            raise RuntimeError(
                f"Could not bind the weight-transfer group port {bind_address}:{port} ({e}); the "
                f"{engine} server {server_url} was not asked to join. A connected client holds "
                f"its group port until close_communicator(), so every server — and every trainer process "
                f"sharing this host — needs its own group_port / vllm_group_port; a port inside "
                f"net.ipv4.ip_local_port_range can also be taken as an outbound connection's source port "
                f"unless reserved (net.ipv4.ip_local_reserved_ports)."
            ) from e
        raise
    return master_address, listener
