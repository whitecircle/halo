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

from halo.env import env_flag, env_str
from halo.inference.endpoints import is_loopback

logger = logging.getLogger(__name__)

_ALL_INTERFACES = "0.0.0.0"
_BIND_ALL_ENV = "HALO_WEIGHT_SYNC_BIND_ALL"
# Any port does, as a UDP connect only asks the kernel for a route and sends no packet.
_ROUTE_PROBE_PORT = 9


class RendezvousListener:
    """An IPv4 socket listening on ``bind_address:port`` (0 = kernel-assigned), for a ``TCPStore`` master to take over.
    Held from the bind on, as a port probed and released first can be taken in between, or handed to a retrying peer
    as its own source port."""

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
        # The handover may run on a formation thread while a failure path closes from the caller's.
        self._lock = threading.Lock()

    def start_store(self, host: str, world_size: int, *, timeout_s: float, **options) -> TCPStore:
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
        """Yield the fd to the one ``TCPStore(master_listen_fd=...)`` call that owns it from then on. The socket is
        detached, as closing it would abort a live store's daemon or a reused fd number."""
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
        """Free the port unless a store took it over; idempotent."""
        with self._lock:
            listener, self._socket = self._socket, None
        if listener is not None:
            listener.close()

    def __enter__(self) -> "RendezvousListener":
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()


def _route_source_ip(host: str) -> str:
    """This host's IPv4 address on the route to ``host``; loopback when no route reaches it or it does not resolve."""
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        try:
            s.connect((host, _ROUTE_PROBE_PORT))
            return s.getsockname()[0]
        except OSError:
            return "127.0.0.1"


def _is_local_address(host: str) -> bool:
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
    """The address the engine's workers dial back to: ``advertised_host``, else ``host_env``, else loopback for a
    server on this machine, else this host's address on the route to the server."""
    master_address = advertised_host or env_str(host_env)
    if not master_address:
        # Same-host groups stay on loopback, as firewalls drop hairpin traffic on the external NIC.
        master_address = "127.0.0.1" if _is_local_address(server_host) else _route_source_ip(server_host)

    if is_loopback(master_address) and not _is_local_address(server_host):
        raise RuntimeError(
            f"{engine} server {server_url} is on a remote host ({server_host}) but the "
            f"NCCL weight-sync group address resolved to loopback ({master_address}): the address set is "
            f"loopback, or no route from this host reaches the server. The server's workers would dial "
            f"back to themselves and the group would never form. Set "
            f"{host_env} (or the per-server group_host) to the "
            f"trainer's routable IP on the subnet the server can reach."
        )
    return master_address


def _bind_address(master_address: str, *, engine: str, server_host: str, host_env: str) -> str:
    """``master_address`` resolved once, since the unauthenticated store listens only where the engine dials.
    ``HALO_WEIGHT_SYNC_BIND_ALL`` widens it to every interface for a trainer behind NAT."""
    host_knob = f"{host_env} (or the per-server group_host)"
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
    """``(master address, listener)``, listening before the engine hears of it. The engine is sent the bound address,
    as the server could resolve the name elsewhere. The caller hands the listener to the store, or closes it."""
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
                f"sharing this host — needs its own group_port (rollout_group_port; vllm_group_port under Online "
                f"GRPO); a port inside "
                f"net.ipv4.ip_local_port_range can also be taken as an outbound connection's source port "
                f"unless reserved (net.ipv4.ip_local_reserved_ports)."
            ) from e
        raise
    return master_address, listener
