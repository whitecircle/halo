"""Weight-sync clients for several rollout servers, replicas of one policy, kept in step: they share one snapshot
per parameter and one chunk decision, and either each form a group of their own or all form one."""

import concurrent.futures
import logging
import threading
from collections.abc import Callable, Iterable
from functools import partial
from typing import Any, get_args

import torch

from halo.weight_sync.chunking import WEIGHT_SYNC_CHUNK_BYTES, resolve_sync_device, snapshot_to_sync_device
from halo.weight_sync.clients.base import WeightSyncClientBase
from halo.weight_sync.group import WeightSyncGroup
from halo.weight_sync.group_layout import SyncTopology, resolve_group_host, resolve_node_label

logger = logging.getLogger(__name__)


class WeightSyncPool:
    """One weight-sync client per rollout server, each with its own NCCL group and trainer-side group port, or with
    ``single_group`` one group over every server, which sends each chunk once."""

    def __init__(
        self,
        server_configs: list[dict[str, Any]],
        *,
        connection_timeout: float,
        client_cls: type[WeightSyncClientBase],
        base_group_port: int,
        topology: SyncTopology = "per_server",
    ):
        """``server_configs``: one ``{"url", "group_port", "group_host", "node"}`` dict per server, ``url`` required. A
        server without its own ``group_port`` takes ``base_group_port + index``, or a kernel-assigned port at base 0.
        Under ``single_group`` the one group binds ``base_group_port`` and ``group_port`` is refused."""
        self.server_configs = server_configs
        self.connection_timeout = connection_timeout
        self.base_group_port = base_group_port
        self._clients: list[WeightSyncClientBase] = []
        self._initialized = False
        self._device: torch.device | None = None
        self._co_load_module_names: tuple[str, ...] | None = None
        # Only the pool knows when every server is done with a snapshot.
        self._buffered_bytes = 0
        # NCCL groups cannot form concurrently, so reconnects from parallel flush threads take turns.
        self._reconnect_lock = threading.Lock()
        self._client_cls = client_cls
        if topology not in get_args(SyncTopology):
            raise ValueError(f"Unknown rollout_sync_topology {topology!r}; supported: {list(get_args(SyncTopology))}")
        self._group: WeightSyncGroup | None = None
        self._group_host = resolve_group_host(server_configs) if topology == "single_group" else None
        self._topology = topology

        # Kernel-assigned ports never collide.
        effective_ports = [self._group_port(i) for i in range(len(server_configs))]
        fixed_ports = [port for port in effective_ports if port]
        if topology == "per_server" and len(set(fixed_ports)) != len(fixed_ports):
            raise ValueError(
                f"Duplicate weight-sync group_port across servers: {effective_ports}. Each server "
                f"needs a UNIQUE group_port — it is bound on the trainer host (one listener "
                f"per server), so distinct {client_cls.BACKEND_NAME} hosts do NOT make a shared "
                f"port safe. URLs: {[c['url'] for c in server_configs]}"
            )

        logger.info(f"WeightSyncPool created for {len(server_configs)} servers: {[c['url'] for c in server_configs]}")

    def init_communicator(self, device: torch.device | str | int) -> None:
        """Form one NCCL group per server, in turn, on the trainer's GPU ``device``; a failure closes the clients
        already connected and raises with the server's error."""
        if self._initialized:
            logger.warning("WeightSyncPool already initialized, skipping")
            return

        device = resolve_sync_device(device)
        if self._topology == "single_group":
            self._init_group(device)
            return

        for i, config in enumerate(self.server_configs):
            url = config["url"]
            port = self._group_port(i)
            logger.info(
                f"Initializing {self._client_cls.BACKEND_NAME} weight-sync client "
                f"{i + 1}/{len(self.server_configs)}: {url} (port {port}, device {device})"
            )
            try:
                client = self._connect_client(i, device)
            except Exception as e:
                # Else the already-initialized clients hold their TCPStore listeners until atexit.
                self.close_communicator()
                raise RuntimeError(
                    f"{self._client_cls.BACKEND_NAME} weight-sync client {i + 1}/{len(self.server_configs)} "
                    f"for {url} (group_port {port}) failed to initialize: {type(e).__name__}: {e}"
                ) from e
            self._clients.append(client)
            logger.info(f"  Connected to {url}")

        self._device = device
        self._initialized = True
        logger.info(f"WeightSyncPool initialized: {len(self._clients)} clients connected")

    def scope_co_load_groups(self, module_names: Iterable[str]) -> None:
        """Narrow every client's co-load groups to the pushed model, rebuilt clients included."""
        self._co_load_module_names = tuple(module_names)
        for client in self._buffer_holders:
            client.scope_co_load_groups(self._co_load_module_names)

    def update_named_param(self, name: str, weights: torch.Tensor) -> None:
        """Buffer one shared snapshot in every client, sending the pending chunk to every server first when it would
        overflow, as a snapshot may go only once every server sent it."""
        if not self._initialized:
            raise RuntimeError("WeightSyncPool not initialized. Call init_communicator() first.")
        if not self._clients:
            return
        snapshot = snapshot_to_sync_device(
            name,
            weights,
            device=self._device,
            buffered_bytes=self._buffered_bytes,
            budget=WEIGHT_SYNC_CHUNK_BYTES,
            flush=self._flush_chunk_to_every_server,
        )
        for client in self._buffer_holders:
            client.buffer_param(name, snapshot)
        self._buffered_bytes += snapshot.nbytes

    def finish_weight_update(self) -> None:
        """Send the tail chunk to every server and close their updates; no-op if nothing was buffered. The fan-out
        shares the forwarding rank's GPU and NICs, so it costs the sum of the pushes, not the slowest one."""
        if not self._initialized or not self._clients:
            return
        self._run_on_every_client(lambda index: self._clients[index].finish_weight_update(), "final flush")
        self._buffered_bytes = 0

    def abort_weight_update(self) -> None:
        """Close the open update on every server after a failed sync; never raises."""
        for client in self._clients:
            client.abort_weight_update()
        self._buffered_bytes = 0

    def close_communicator(self) -> None:
        for client in self._clients:
            try:
                client.close_communicator()
            except Exception as e:
                logger.warning(f"Error closing client: {e}")

        self._clients = []
        self._group = None
        self._initialized = False
        logger.info("WeightSyncPool closed all connections")

    @property
    def group(self) -> WeightSyncGroup | None:
        """The one group over every server, ``None`` under ``per_server`` or before formation."""
        return self._group

    @property
    def _buffer_holders(self) -> list[WeightSyncClientBase]:
        """The clients that hold the shared snapshots: every server's own, or the group's leader."""
        return [self._group.leader] if self._group is not None else self._clients

    def _init_group(self, device: torch.device) -> None:
        """Build every server's client and form the one group; a failure closes it and raises with the cause."""
        clients = [
            self._client_cls(
                base_url=config["url"],
                connection_timeout=self.connection_timeout,
                group_host=self._group_host,
            )
            for config in self.server_configs
        ]
        group = WeightSyncGroup(
            clients,
            nodes=[resolve_node_label(config) for config in self.server_configs],
            group_port=self.base_group_port,
            group_host=self._group_host,
        )
        try:
            group.init_communicator(device)
        except Exception as e:
            group.close_communicator()
            raise RuntimeError(
                f"{self._client_cls.BACKEND_NAME} weight-sync group over {len(clients)} servers failed to form "
                f"(group port {self.base_group_port}): {type(e).__name__}: {e}"
            ) from e
        self._group = group
        self._clients = [group]
        if self._co_load_module_names is not None:
            group.leader.scope_co_load_groups(self._co_load_module_names)
        self._device = device
        self._initialized = True
        logger.info(f"WeightSyncPool initialized: one group over {len(clients)} servers")

    def _group_port(self, index: int) -> int:
        config = self.server_configs[index]
        if "group_port" in config:
            return config["group_port"]
        return self.base_group_port + index if self.base_group_port else 0

    def _connect_client(self, index: int, device: torch.device) -> WeightSyncClientBase:
        config = self.server_configs[index]
        client = self._client_cls(
            base_url=config["url"],
            group_port=self._group_port(index),
            connection_timeout=self.connection_timeout,
            group_host=config.get("group_host"),
        )
        client.init_communicator(device=device)
        if self._co_load_module_names is not None:
            client.scope_co_load_groups(self._co_load_module_names)
        return client

    def _flush_chunk_to_every_server(self) -> None:
        self._run_on_every_client(lambda index: self._clients[index].flush_chunk(), "chunk flush")
        # Every client holds the same snapshots, so what one kept back for a co-load partner they all did.
        self._buffered_bytes = max((client.buffered_bytes for client in self._buffer_holders), default=0)

    def _run_on_every_client(self, operation: Callable[[int], None], what: str) -> None:
        """Run ``operation(index)`` on every client concurrently, raising one aggregated failure. Indexed, so a
        retry reaches the client a reconnect swapped in. A group recovers itself, and its failure is final."""
        if self._group is not None:
            try:
                operation(0)
            except Exception:
                self._group.leader.drain_param_buffer()
                raise
            return
        run = partial(self._with_recovery, operation, what)
        if len(self._clients) == 1:
            errors = [run(0)]
        else:
            with concurrent.futures.ThreadPoolExecutor(max_workers=len(self._clients)) as executor:
                errors = list(executor.map(run, range(len(self._clients))))
        errors = [e for e in errors if e is not None]
        if errors:
            raise RuntimeError(
                f"Weight sync ({what}) failed for {len(errors)}/{len(self._clients)} "
                f"{self._client_cls.BACKEND_NAME} server(s) — "
                f"continuing would train on stale-policy rollouts. Failures: {'; '.join(errors)}"
            )

    def _with_recovery(self, operation: Callable[[int], None], what: str, index: int) -> str | None:
        """Run one client's step, reconnecting and retrying once where nothing of the sync landed yet; ``None`` on
        success, else the error for the aggregated raise."""
        client = self._clients[index]
        url = self.server_configs[index]["url"]
        try:
            operation(index)
            return None
        except Exception as e:
            first_error = e
        if not client.can_replay_sync:
            client.drain_param_buffer()
            return (
                f"{url}: {first_error} (not retried: this sync was already streaming, so the engine "
                f"holds part of the new weights and the trainer no longer has the rest to re-send — "
                f"restart that server before serving again)"
            )
        logger.error(f"Weight-sync {what} failed for {url}: {first_error}; attempting one reconnect")
        try:
            with self._reconnect_lock:
                self._reconnect(index)
            operation(index)
            logger.warning(f"Weight sync to {url} recovered after reconnect")
            return None
        except Exception as retry_error:
            # The pool entry holds the snapshots: the original after a failed reconnect, else the replacement.
            self._clients[index].drain_param_buffer()
            return f"{url}: {first_error} (reconnect retry failed: {retry_error})"

    def _reconnect(self, index: int) -> None:
        """Replace server ``index``'s client, carrying its buffer over; the replacement serves base weights until the
        re-flush. The old client closes first, lifting its failed sync's pause and freeing the group port."""
        url = self.server_configs[index]["url"]
        old = self._clients[index]
        logger.warning(
            f"Reconnecting weight-sync client for {url} on group_port {self._group_port(index)}; "
            f"full re-sync required before rollouts"
        )
        buffered = old.drain_param_buffer()
        try:
            old.close_communicator()
        except Exception as e:  # the old client is already dead; never mask the rebuild
            logger.warning(f"Error closing stale client for {url}: {e}")
        client = self._connect_client(index, self._device)
        for name, snapshot in buffered:
            client.buffer_param(name, snapshot)
        self._clients[index] = client
