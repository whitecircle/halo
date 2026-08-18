"""Pool of vendored NCCL weight-sync clients, one per rollout server, plus the context-window
preflight that reads each server's model card before the trainer is built.

No vllm/sglang package dependency; generation requests go over HTTP separately (see ray_actors.py).
Constraints: weight sync runs from the main process only; the trainer must use a different GPU than
the rollout server (NCCL requires distinct devices); one server URL per weight-sync group.
"""

import concurrent.futures
import logging
import threading
from collections.abc import Callable
from functools import partial
from typing import Any

import torch

from src.distributed.nccl.clients.base import (
    WEIGHT_SYNC_CHUNK_BYTES,
    BaseWeightSyncClient,
    PinnedHostBufferPool,
    payload_bytes,
    starts_new_chunk,
    validate_syncable_param,
)
from src.distributed.nccl.clients.vllm import VLLMWeightSyncClient as VLLMClient
from src.distributed.nccl.registry import resolve_weight_sync_client
from src.distributed.runtime import broadcast_from_rank0, is_global_main_process

logger = logging.getLogger(__name__)

# First trainer-side NCCL group port; server N without an explicit ``group_port`` takes this + N.
DEFAULT_WEIGHT_SYNC_GROUP_PORT = 51216


def _fetch_served_max_model_len(client_cls: type[BaseWeightSyncClient], url: str) -> int | None:
    """The served model's context window, or None when it could not be read (server down, or a model
    card without ``max_model_len``); a preflight probe should not be what fails the run."""
    client = None
    try:
        client = client_cls(base_url=url)
        return client.served_max_model_len()
    except Exception as e:
        logger.warning(f"Could not read max_model_len from {url}: {e}")
        return None
    finally:
        if client is not None:
            client.session.close()


def verify_context_window(
    client_cls: type[BaseWeightSyncClient],
    urls: list[str],
    single_turn_tokens: int,
    full_trajectory_tokens: int | None,
) -> None:
    """Verify each rollout server's context window fits the configured generation budget.

    ``single_turn_tokens`` (max_prompt_length + per-call generation cap) is a hard requirement, so this
    raises. ``full_trajectory_tokens`` (worst-case multi-turn budget) is advisory, since a rollout
    growing past the context OOMs the trainer's forward before the fail-on-overflow check, so it warns.
    """
    for url in urls:
        mml = _fetch_served_max_model_len(client_cls, url)
        if mml is None:
            continue
        logger.info(f"Rollout server {url}: max_model_len={mml}")
        if single_turn_tokens > mml:
            raise ValueError(
                f"Rollout server {url} context window ({mml}) is smaller than one rollout turn "
                f"({single_turn_tokens} = max_prompt_length + per-turn generation). Lower "
                f"rollout_max_tokens/max_completion_length or serve a longer-context model."
            )
        if full_trajectory_tokens and full_trajectory_tokens > mml:
            logger.warning(
                f"Rollout server {url} context window ({mml}) < worst-case trajectory budget "
                f"({full_trajectory_tokens} = max_prompt_length + max_turns × rollout_max_tokens). A long "
                f"multi-turn rollout that grows past {mml} tokens can OOM the training forward before the "
                f"fail-on-overflow check — lower max_turns or rollout_max_tokens so the worst case fits."
            )


def verify_context_window_synced(
    urls: list[str], single_turn_tokens: int, full_trajectory_tokens: int | None = None, *, backend: str
) -> None:
    """Collective-safe :func:`verify_context_window`; call on every rank.

    The HTTP probe runs on rank 0 only (one probe, no log spam); the hard-requirement verdict is
    broadcast so all ranks raise together instead of hanging on the next barrier. The backend lookup
    runs on every rank: it reads no server, and a rank-0-only raise there would leave the peers
    blocked in the broadcast below.
    """
    client_cls = resolve_weight_sync_client(backend)
    error: str | None = None
    if is_global_main_process():
        try:
            verify_context_window(client_cls, urls, single_turn_tokens, full_trajectory_tokens)
        except ValueError as e:
            error = str(e)
    error = broadcast_from_rank0(error)
    if error is not None:
        raise ValueError(error)


class InferenceClientManager:
    """Manage one weight-sync client per rollout server (each with its own NCCL group and unique
    group_port), keeping multiple servers in weight-sync with the trainer."""

    def __init__(
        self,
        server_configs: list[dict[str, Any]],
        connection_timeout: float = 120.0,
        client_cls: type[BaseWeightSyncClient] = VLLMClient,
        base_group_port: int = DEFAULT_WEIGHT_SYNC_GROUP_PORT,
    ):
        """``server_configs`` is a list of ``{"url", "group_port", "group_host"}`` dicts, one client
        per server (only ``url`` is required).

        ``base_group_port`` is the first trainer-side NCCL group port, handed to server N that
        declares no ``group_port`` of its own as ``base_group_port + N``. It is the run's
        ``vllm_group_port``, so the setting means the same thing whichever client shape is built.

        ``client_cls`` selects the engine (``resolve_weight_sync_client``); every server in one
        manager speaks the same one, since they are replicas of a single served policy.
        """
        self.server_configs = server_configs
        self.connection_timeout = connection_timeout
        self.base_group_port = base_group_port
        self._clients = []
        self._initialized = False
        self._device = None
        # Page-locking is amortized across params and across syncs: a fresh pinned allocation per
        # param stalls the trainer for seconds at 20B+. Bounded to one chunk (see the pool).
        self._host_buffer_pool = PinnedHostBufferPool(WEIGHT_SYNC_CHUNK_BYTES)
        # Bytes buffered since the last chunk went out. The manager makes the chunk decision because
        # only it can tell when every server is done with the shared snapshots.
        self._buffered_bytes = 0
        # NCCL groups cannot form concurrently: serializes reconnects across parallel flush threads.
        self._reconnect_lock = threading.Lock()
        # Single construction seam, so init and reconnect cannot drift onto different engines.
        self._client_factory = client_cls

        # group_port is bound on the trainer host, so two servers sharing one collide.
        effective_ports = [self._group_port(i) for i in range(len(server_configs))]
        if len(set(effective_ports)) != len(effective_ports):
            raise ValueError(
                f"Duplicate weight-sync group_port across servers: {effective_ports}. Each server "
                f"needs a UNIQUE group_port — it is bound on the trainer host (one listener "
                f"per server), so distinct {client_cls.BACKEND_NAME} hosts do NOT make a shared "
                f"port safe. URLs: {[c['url'] for c in server_configs]}"
            )

        logger.info(
            f"InferenceClientManager created for {len(server_configs)} servers: {[c['url'] for c in server_configs]}"
        )

    def _group_port(self, index: int) -> int:
        """The trainer-side NCCL group port for one server: its configured value, else the base + index."""
        return self.server_configs[index].get("group_port", self.base_group_port + index)

    def init_communicators(self, device: torch.device | str | int):
        """Create a separate NCCL process group per server (sequentially — groups can't init
        concurrently). ``device`` is the trainer's GPU and must differ from every server's GPU
        (raises RuntimeError otherwise, cleaning up already-initialized clients)."""
        if self._initialized:
            logger.warning("InferenceClientManager already initialized, skipping")
            return

        if isinstance(device, torch.device) and device.type == "cuda" and device.index is None:
            device = torch.device("cuda", torch.cuda.current_device())

        for i, config in enumerate(self.server_configs):
            url = config["url"]
            port = self._group_port(i)
            # Optional per-server routable trainer NIC for the NCCL group (multi-homed nodes).
            group_host = config.get("group_host")

            logger.info(
                f"Initializing {self._client_factory.BACKEND_NAME} weight-sync client "
                f"{i + 1}/{len(self.server_configs)}: {url} (port {port})"
            )

            client = self._client_factory(
                base_url=url,
                group_port=port,
                connection_timeout=self.connection_timeout,
                group_host=group_host,
            )

            try:
                client.init_communicator(device=device)
                self._clients.append(client)
                logger.info(f"  Connected to {url}")
            except Exception as e:
                # Else the already-initialized clients hold their TCPStore listeners until atexit.
                self.close_communicators()
                raise RuntimeError(
                    f"Failed to initialize client for {url}: {e}\n"
                    f"Ensure the {self._client_factory.BACKEND_NAME} server is on a different GPU "
                    f"than the trainer (device={device})"
                ) from e

        self._device = device  # reconnect_client re-forms NCCL groups on the same trainer device
        self._initialized = True
        logger.info(f"InferenceClientManager initialized: {len(self._clients)} clients connected")

    def update_model_params(self, model: torch.nn.Module):
        """Sync weights to the rollout servers one at a time, so at least N-1 stay available for
        generation while a server takes its broadcast."""
        if not self._initialized:
            raise RuntimeError("InferenceClientManager not initialized. Call init_communicators() first.")

        for i, client in enumerate(self._clients):
            url = self.server_configs[i]["url"]
            logger.debug(f"Rolling sync: updating server {i + 1}/{len(self._clients)} ({url})")
            try:
                client.update_model_params(model)
            except Exception as e:
                # The raise alone does not name which of the N servers refused the weights.
                logger.error(f"Rolling sync failed for server {i + 1} ({url}): {e}")
                raise

        logger.debug(f"Synced weights to {len(self._clients)} {self._client_factory.BACKEND_NAME} servers")

    def update_named_param(self, name: str, weights: torch.Tensor):
        """Send one pre-gathered named parameter to all servers, for trainers that must gather EP/TP
        weights first (``update_model_params`` iterates ``model.named_parameters()`` instead).

        One read-only host snapshot per param is shared by reference across every client's buffer, so
        pinned host RAM stays ~1× chunk rather than N_servers×, and a client's flush/clear drops only
        its own references.

        The chunk decision is the manager's, not each client's: a shared snapshot may only be recycled
        once every server has sent the chunk holding it. The buffer is therefore drained here, on every
        server concurrently, and the pool released after.
        """
        if not self._initialized:
            raise RuntimeError("InferenceClientManager not initialized. Call init_communicators() first.")
        if not self._clients:
            return
        # On the source, before the snapshot and before any chunk quiesces a server (this path never
        # reaches the client's own update_named_param, which is where the single-server check lives).
        validate_syncable_param(name, weights)
        # Flushed before the budget is exceeded, on the same rule the clients buffer by, and before
        # the snapshot, so the pool can hand this param a buffer the flush just recycled.
        if starts_new_chunk(self._buffered_bytes, payload_bytes(weights), WEIGHT_SYNC_CHUNK_BYTES):
            self._flush_chunk_to_every_server()
        host = self._host_buffer_pool.snapshot(weights)
        for client in self._clients:
            client.buffer_host_param(name, host)
        self._buffered_bytes += payload_bytes(host)

    def abort_weight_update(self):
        """Close the open update on every server after a failed sync; never raises (see the client)."""
        for client in self._clients:
            client.abort_weight_update()
        self._host_buffer_pool.release()
        self._buffered_bytes = 0

    def _flush_chunk_to_every_server(self):
        """Send the buffered chunk to every server concurrently, then recycle its host buffers.

        The mid-gather half of the streamed sync: the update stays open on each server (the tail and
        the close come from ``reset_prefix_cache``), so this is the point where a chunk stops being
        replayable. A server that fails from here on is reported rather than reconnected.
        """
        self._run_on_every_client(lambda index: self._clients[index].flush_chunk(), "chunk flush")
        # Only now: every server has broadcast the chunk holding these buffers.
        self._host_buffer_pool.release()
        self._buffered_bytes = 0

    def reset_prefix_cache(self):
        """Send the tail chunk to all rollout servers and close their updates (no-op if nothing was buffered).

        Servers flush concurrently (each client has its own NCCL communicator and per-call streams), so
        the trainer stall is ~max(per-server flush) instead of the sum. The async D2H snapshot copies
        complete before any producer thread reads host memory. Returns only once every flush is done,
        which is what makes the pooled host buffers safe to recycle.

        Raises RuntimeError if a server still fails after one reconnect and re-flush attempt; the
        alternative would leave that server serving stale-policy rollouts.
        """
        if not self._initialized or not self._clients:
            return
        self._run_on_every_client(lambda index: self._clients[index].reset_prefix_cache(), "final flush")
        self._host_buffer_pool.release()
        self._buffered_bytes = 0

    def _run_on_every_client(self, operation: Callable[[int], None], what: str) -> None:
        """Run ``operation(index)`` on every client concurrently, raising one aggregated failure.

        Indexed rather than closed over a client object: the recovery path below swaps a client in
        the pool, and the retry has to reach the replacement.
        """
        run = partial(self._with_recovery, operation, what)
        if len(self._clients) == 1:  # no thread-pool overhead for the single-server case
            errors = [run(0)]
        else:
            with concurrent.futures.ThreadPoolExecutor(max_workers=len(self._clients)) as executor:
                errors = list(executor.map(run, range(len(self._clients))))
        errors = [e for e in errors if e is not None]
        if errors:
            raise RuntimeError(
                f"Weight sync ({what}) failed for {len(errors)}/{len(self._clients)} "
                f"{self._client_factory.BACKEND_NAME} server(s) — "
                f"continuing would train on stale-policy rollouts. Failures: {'; '.join(errors)}"
            )

    def _with_recovery(self, operation: Callable[[int], None], what: str, index: int) -> str | None:
        """Run one client's step; on failure make one reconnect and retry attempt where that can work.

        Returns ``None`` on success (including a recovered step) or the error string for the
        aggregated raise. The failed client's buffer is retained through the reconnect attempt
        (``reconnect_client`` moves the buffered references onto the fresh client) and dropped only
        when the retry also fails; clearing first would lose the params the retry needs.

        A client that has already streamed a chunk cannot be recovered: the trainer keeps no copy of
        what went out, so a fresh engine would receive only the remainder and serve a model that is
        part old and part new. That case is reported with the reason.
        """
        client = self._clients[index]
        url = self.server_configs[index]["url"]
        try:
            operation(index)
            return None
        except Exception as e:  # aggregated and re-raised by the caller
            first_error = e
        if not client.can_replay_sync:
            client.drain_param_buffer()  # dropped rather than re-broadcast: this sync cannot be replayed
            return (
                f"{url}: {first_error} (not retried: this sync was already streaming, so the engine "
                f"holds part of the new weights and the trainer no longer has the rest to re-send — "
                f"restart that server before serving again)"
            )
        logger.error(f"Weight-sync {what} failed for {url}: {first_error}; attempting one reconnect")
        try:
            with self._reconnect_lock:  # NCCL groups cannot form concurrently
                new_client = self.reconnect_client(index)
            del new_client  # the pool entry is what `operation` reaches
            operation(index)
            logger.warning(f"Weight sync to {url} recovered after reconnect")
            return None
        except Exception as retry_error:  # aggregated and re-raised by the caller
            # The pool entry is the only client that can still hold this sync's snapshot: a failed
            # reconnect leaves the original there, and a failed retry leaves the new client, whose
            # predecessor was already drained by ``reconnect_client``.
            self._clients[index].drain_param_buffer()
            return f"{url}: {first_error} (reconnect retry failed: {retry_error})"

    def reconnect_client(self, index: int):
        """Rebuild the weight-sync client for one server after an engine container restart.

        The old NCCL communicator cannot re-form, so this builds a fresh client on the server's
        configured ``group_port``: retire the old client (draining its pending buffer and lifting any
        pause it left on the server, which also releases the port listener), wait for the server, form
        the NCCL group on the trainer device, move the drained buffer onto the new client, and swap it
        into the pool. Returns the new client; raises if the server stays unreachable.

        The replacement starts on base-checkpoint weights, so it is only usable inside
        :meth:`_with_recovery`, whose retained buffer is this sync's full param snapshot and is
        re-flushed immediately. Any other caller must give the new client a full push of its own.
        """
        if not self._initialized or self._device is None:
            raise RuntimeError("InferenceClientManager.reconnect_client requires init_communicators() first.")
        if not 0 <= index < len(self._clients):
            raise IndexError(f"Client index {index} out of range (have {len(self._clients)} clients)")
        config = self.server_configs[index]
        old = self._clients[index]
        port = self._group_port(index)
        logger.warning(
            f"Reconnecting weight-sync client for {config['url']} on group_port {port}; "
            f"full re-sync required before rollouts"
        )
        # Retire the old client first: it holds the /resume for the pause its failed sync left behind.
        buffered = old.drain_param_buffer()
        try:
            old.close_communicator()
        except Exception as e:  # the old client is already dead; never mask the rebuild
            logger.warning(f"Error closing stale client for {config['url']}: {e}")
        client = self._client_factory(
            base_url=config["url"],
            group_port=port,
            connection_timeout=self.connection_timeout,
            group_host=config.get("group_host"),
        )
        client.init_communicator(device=self._device)
        for name, host in buffered:
            client.buffer_host_param(name, host)
        self._clients[index] = client
        return client

    def close_communicators(self):
        """Close every engine client connection and clean up."""
        for client in self._clients:
            try:
                client.close_communicator()
            except Exception as e:
                logger.warning(f"Error closing client: {e}")

        self._clients = []
        self._initialized = False
        logger.info("InferenceClientManager closed all connections")

    @property
    def num_servers(self) -> int:
        """Number of rollout servers configured (or connected if initialized)."""
        if self._initialized:
            return len(self._clients)
        return len(self.server_configs)
