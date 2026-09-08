"""NCCL weight sync for SGLang servers. Trainer is rank 0, engine workers are 1..N.

SGLang's contract differs from vLLM's in three ways that shape this client:

* Bootstrap: the engine joins a plain c10d group (``/init_weights_update_group``), so the trainer
  speaks c10d too (``transport/torch_group.py``) rather than the vendored pynccl path.
* Wire format: the engine allocates one empty tensor per declared name and receives each with
  ``torch.distributed.broadcast(src=0)``, so weights travel as typed tensors in declaration order
  rather than the packed uint8 buffers the vLLM protocol uses.
* Quiescence: the engine's post-update cache flush asserts the scheduler is fully idle, and a failed
  assert takes down the server rather than returning an error, so every sync is bracketed by
  ``/pause_generation`` and ``/continue_generation``.

The group is ordinary NCCL, same as vLLM's: on one host it takes CUDA IPC between the two
containers, across nodes the fabric. The one engine-side requirement is cuMem parity —
``NCCL_CUMEM_ENABLE=1`` in the server container, which SGLang otherwise sets to 0 process-wide while
the trainer's NCCL has it on, and the mismatch fails the first cross-container import
(``docker-compose.sglang.yml`` sets it).

The training image never imports sglang (its transformers pin conflicts); everything here is HTTP
plus torch. ``Dockerfile.sglang`` asserts the server side of this contract at image build.
"""

import atexit
import logging
from collections.abc import Iterator
from urllib.parse import urlparse

import torch
import torch.distributed as dist
from torch.distributed import distributed_c10d as c10d

from src.distributed.nccl.clients.base import (
    _CLEANUP_TIMEOUT_S,
    _GROUP_FORMATION_TIMEOUT_S,
    _HTTP_PROBE_TIMEOUT_S,
    _SERVER_ERROR_GRACE_S,
    _WEIGHT_UPDATE_TIMEOUT_S,
    BaseWeightSyncClient,
    _AsyncCall,
    _wait_for_calls,
)
from src.distributed.nccl.transport.pynccl import bounded_stream_sync
from src.distributed.nccl.transport.torch_group import (
    DEFAULT_WEIGHT_UPDATE_GROUP_NAME,
    create_weight_update_group,
    destroy_weight_update_group,
    drop_weight_update_group_bookkeeping,
)

logger = logging.getLogger(__name__)

# Engine control-plane routes used from more than one call site (the call and its async-call label,
# or both the sync and the teardown path), so a route rename cannot leave a stale label behind.
_EP_INIT_GROUP = "/init_weights_update_group"
_EP_UPDATE_FROM_DIST = "/update_weights_from_distributed"
_EP_CONTINUE = "/continue_generation"
# Byte alignment of each tensor inside the staging arena: a multiple of every dtype's element size,
# so a uint8 slice can be viewed as the tensor's dtype.
_ARENA_ALIGNMENT = 256


class SGLangWeightSyncClient(BaseWeightSyncClient):
    """NCCL weight sync client for SGLang servers."""

    BACKEND_KEY = "sglang"
    BACKEND_NAME = "SGLang"
    GROUP_HOST_ENV = "SGLANG_GROUP_HOST"
    # SGLang loads MoE experts as transformers stores them: one fused pair per layer.
    EXPERT_LAYOUT = BaseWeightSyncClient.FUSED_EXPERT_LAYOUT
    RESUME_ENDPOINT = _EP_CONTINUE
    # An empty body is rejected: the endpoint takes a request dataclass, so it needs JSON.
    RESUME_PAYLOAD: dict | None = {}

    def __init__(
        self,
        base_url: str,
        group_port: int = 0,
        connection_timeout: float = 0.0,
        group_host: str | None = None,
        group_name: str | None = None,
    ):
        self._group: dist.ProcessGroup | None = None
        self._store = None
        # Set once a drain deadline aborted the group; a destroy on an aborted group raises, and a
        # broadcast on one hangs.
        self._aborted = False
        # Device staging arena the chunk's uploads are carved from (see ``_stage_on_device``).
        self._device_arena: torch.Tensor | None = None
        super().__init__(
            base_url=base_url,
            group_port=group_port,
            connection_timeout=connection_timeout,
            group_host=group_host,
        )
        # Unique per server by default: c10d registers group names process-globally on the trainer, so
        # a second client re-using one fixed name fails with "group name has already been created".
        # Keyed on the server endpoint rather than group_port, where 0 means "auto-pick at group
        # formation" and would collide every auto-port client on one name.
        self.group_name = group_name or f"{DEFAULT_WEIGHT_UPDATE_GROUP_NAME}_{urlparse(self.base_url).netloc}"

    def fetch_engine_world_size(self) -> int:
        """Number of engine ranks that will join the group.

        SGLang assigns each worker ``rank_offset + tp_rank``. Under ``--enable-dp-attention``
        ``tp_rank`` already enumerates every GPU (one TP group spans the server), so the group is sized
        by ``tp_size`` alone; multiplying by ``dp_size`` oversizes the c10d group and hangs formation.
        Plain ``--dp-size`` replicas each restart ``tp_rank`` at 0, so their ranks collide in the
        update group and no sizing can address them, which is rejected above. The layout is read from
        the server rather than from config so the two cannot disagree after a serve-flag change.
        """
        resp = self.session.get(f"{self.base_url}/server_info", timeout=_HTTP_PROBE_TIMEOUT_S)
        resp.raise_for_status()
        info = resp.json()
        try:
            tp_size = int(info["tp_size"])
            dp_size = int(info.get("dp_size", 1) or 1)
        except (KeyError, TypeError, ValueError) as e:
            raise RuntimeError(
                f"Could not read the parallel layout from {self.base_url}/server_info "
                f"(need 'tp_size', optionally 'dp_size'); got keys {sorted(info)[:25]}. "
                f"The weight-sync group cannot be sized without it."
            ) from e
        if dp_size > 1 and not bool(info.get("enable_dp_attention", False)):
            raise RuntimeError(
                f"SGLang weight sync cannot address a plain --dp-size {dp_size} server: each "
                f"replica restarts tp_rank at 0, so workers collide on rank_offset + tp_rank in "
                f"the weight-update group. Serve with --enable-dp-attention or dp-size 1."
            )
        return tp_size

    def init_communicator(self, device: torch.device | str | int = 0):
        """Form the weight-update group: trainer rank 0, engine ranks 1..N."""
        engine_ws = self.fetch_engine_world_size()
        world_size = engine_ws + 1
        master_address, master_port = self._resolve_group_address()
        self._resolve_sync_device(device)

        logger.info(f"SGLang NCCL init: engine_ws={engine_ws}, master={master_address}:{master_port}")

        # The engine registers the group by name as soon as it is asked to join and rejects any later
        # join under that name, so a trainer that died between the request and a working group leaves
        # the server unable to weight-sync until restarted. Clearing a stale registration first is a
        # no-op on the first attempt.
        self._destroy_remote_group()

        server_call = _AsyncCall(
            name=_EP_INIT_GROUP,
            fn=lambda: self._post_once(
                _EP_INIT_GROUP,
                timeout=_GROUP_FORMATION_TIMEOUT_S,
                json={
                    "master_address": master_address,
                    "master_port": master_port,
                    "rank_offset": 1,
                    "world_size": world_size,
                    "group_name": self.group_name,
                    "backend": "nccl",
                },
            ),
        )
        group_call: _AsyncCall | None = None
        try:
            # Both halves block until the other arrives (the engine's request returns once the group
            # formed, group creation returns once the engine joined), so they run concurrently and
            # whichever fails first wins.
            group_call = _AsyncCall(
                name="weight-update group formation",
                fn=lambda: create_weight_update_group(
                    master_address=master_address,
                    master_port=master_port,
                    world_size=world_size,
                    device=self.sync_device,
                    group_name=self.group_name,
                    timeout_s=_GROUP_FORMATION_TIMEOUT_S,
                ),
            )
            _wait_for_calls([server_call, group_call], timeout=_GROUP_FORMATION_TIMEOUT_S)
            self._group, self._store = group_call.result
        except Exception as e:
            # `_wait_for_calls` raises as soon as either half fails, so a group this rank already
            # formed lives only on the call handle. Adopt it before releasing, or c10d keeps the
            # registration under `self.group_name` and the store keeps its listener on `master_port`,
            # and every later reconnect is rejected with "group name has already been created".
            formed = group_call.join(_SERVER_ERROR_GRACE_S).result if group_call is not None else None
            if formed is not None:
                self._group, self._store = formed
            self._release_group()
            if e is server_call.error:
                raise  # the server named its own failure; a topology hint would mislead
            raise RuntimeError(
                f"SGLang weight-update group formation failed (advertised master "
                f"{master_address}:{master_port}; server {self.base_url}). If the SGLang node cannot "
                f"reach that address, set {self.GROUP_HOST_ENV} to an interface routable from it. "
                f"A port already held by another client on this host fails the same way — each server "
                f"needs its own group_port. Original error: {type(e).__name__}: {e}"
            ) from e

        # The atexit invocation skips the local NCCL destroy: the group's peers are engine
        # processes that never enter destroy, so with an interrupted sync in flight
        # ``dist.destroy_process_group`` blocks rather than raising and wedges interpreter exit.
        # Explicit close_communicator() calls (the reconnect path) keep the full destroy.
        atexit.register(self.close_communicator, _local_destroy=False)
        logger.info("SGLang NCCL weight transfer initialized")

    def begin_weight_update(self):
        """Quiesce the engine. SGLang has no separate reload phase; the chunk requests are the update.

        Unwinds its own quiesce on failure: nothing has been broadcast yet, so the server is left
        serving rather than paused.
        """
        if self._group is None:
            raise RuntimeError("Call init_communicator() first")
        if self._aborted:
            raise RuntimeError(
                "SGLang weight-sync group was aborted after a broadcast deadline and cannot be reused "
                "— rebuild the client (reconnect) before syncing again."
            )
        # The group was formed against this device; a collective issued while a different device is
        # current fails inside NCCL with a bare "invalid argument", after the engine has already
        # allocated its receive buffers and considers itself mid-update.
        if self.sync_device is not None and self.sync_device.type == "cuda":
            torch.cuda.set_device(self.sync_device)
        self._paused = True
        try:
            # The engine's post-update flush asserts a fully idle scheduler and SIGQUITs the server
            # when that fails, so draining first is required for correctness.
            self._post("/pause_generation", json={"mode": "abort"})
        except Exception:
            self._lift_pause(_CLEANUP_TIMEOUT_S, context="after a failed pause")
            raise

    def _broadcast_chunk(self, chunk: list[tuple[str, torch.Tensor]], final: bool = False):
        """Broadcast one chunk; the cache invalidation rides the final one.

        The caller's byte budget is the engine's allocation bound: it allocates ``torch.empty`` for
        every name in a request before receiving any of them.
        """
        self._send_chunk(chunk, flush_cache=final)

    def end_weight_update(self, tail: list[tuple[str, torch.Tensor]]):
        """Send the final chunk with the cache flush, then resume; the resume runs on every path."""
        try:
            tail = list(tail)
            if tail:
                self.send_weights(tail, final=True)
            else:
                # The cache invalidation rides the last chunk, so with nothing left to send it is
                # requested directly; otherwise the new weights serve against a stale prefix cache.
                self._post("/flush_cache")
        finally:
            # Always lift the pause; a server left paused wedges every later rollout.
            self._lift_pause(_CLEANUP_TIMEOUT_S, context="after sync")

    def _send_chunk(self, chunk: list[tuple[str, torch.Tensor]], flush_cache: bool):
        """Declare one chunk over HTTP, then broadcast its tensors in the declared order.

        The engine allocates every tensor in the chunk, issues an async broadcast per name and then
        waits on all of them, so the trainer must issue matching broadcasts in exactly this order.
        """
        names = [name for name, _ in chunk]
        dtypes = [str(param.dtype).split(".")[-1] for _, param in chunk]
        shapes = [list(param.shape) for _, param in chunk]

        # A repeated name desynchronizes the stream: the engine keys its receive buffers by name and
        # reads the duplicate once while this loop broadcasts it twice. Every later tensor then lands
        # one send out of step and NCCL reports a truncated message naming the wrong param, so the
        # duplicate is rejected here while the offender is still identifiable.
        if len(set(names)) != len(names):
            repeated = sorted({name for name in names if names.count(name) > 1})
            raise ValueError(
                f"Duplicate parameter name(s) in one weight-sync chunk: {repeated}. The engine "
                f"receives each name once, so the broadcast order would desynchronize."
            )

        server_call = _AsyncCall(
            name=_EP_UPDATE_FROM_DIST,
            fn=lambda: self._post_once(
                _EP_UPDATE_FROM_DIST,
                timeout=_WEIGHT_UPDATE_TIMEOUT_S,
                json={
                    "names": names,
                    "dtypes": dtypes,
                    "shapes": shapes,
                    "group_name": self.group_name,
                    # Only the final chunk invalidates the cache: an intermediate flush would run once
                    # per chunk for no benefit, and each one asserts an idle scheduler.
                    "flush_cache": flush_cache,
                },
            ),
        )
        try:
            for tensor in self._stage_on_device([param for _, param in chunk]):
                # Synchronous only at stream level: ProcessGroupNCCL's wait() orders the current
                # stream behind the collective and returns. An async_op work per tensor instead
                # hands hundreds of outstanding works to the watchdog, whose per-work event polling
                # costs the push a fifth of its rate.
                dist.broadcast(tensor, src=0, group=self._group)
            self._drain_broadcasts()
            # The drain proves the local sends completed; delivery is confirmed by the engine's own
            # reply, which returns once every declared tensor landed.
            server_call.wait(timeout=_WEIGHT_UPDATE_TIMEOUT_S)
        except Exception as e:
            self._raise_if_server_failed(
                e,
                server_call,
                _EP_UPDATE_FROM_DIST,
                hint=(
                    " The engine reports a partially updated model after such a failure — restart "
                    "the server before serving again."
                ),
            )
            raise

    def _stage_on_device(self, params: list[torch.Tensor]) -> Iterator[torch.Tensor]:
        """Upload a chunk into one persistent device arena, yielding each tensor's view as it is issued.

        One arena rather than one allocation per tensor: a block freed mid-chunk returns to the
        caching allocator carrying a pending event on the NCCL stream, and the next upload that reuses
        it waits on that event inside the allocator — a host block inside the very collective the
        drain deadline bounds — while fresh allocations per chunk churn the allocator on the critical
        path. The arena is reused across chunks and syncs (each chunk drains before the next is
        staged), sized to the chunk budget and grown only for a tensor above it, the same footprint
        as the vLLM producer's packed buffers.

        A generator so the caller broadcasts each view right after its upload is issued: a collective
        enqueued then only waits for that one copy, and the NIC drains tensor N while tensor N+1
        crosses PCIe. Issuing every upload first would make the first send wait for the last copy,
        serializing the two transfers per chunk.
        """
        offsets: list[int] = []
        total = 0
        for param in params:
            offsets.append(total)
            total += -(-param.numel() * param.element_size() // _ARENA_ALIGNMENT) * _ARENA_ALIGNMENT
        if self._device_arena is None or self._device_arena.numel() < total:
            self._device_arena = None  # release before growing, so both never coexist on the device
            self._device_arena = torch.empty(total, dtype=torch.uint8, device=self.sync_device)
        for param, offset in zip(params, offsets, strict=True):
            nbytes = param.numel() * param.element_size()
            view = self._device_arena[offset : offset + nbytes].view(param.dtype).view(param.shape)
            view.copy_(param, non_blocking=True)
            yield view

    def _drain_broadcasts(self) -> None:
        """Wait for the chunk's broadcasts with a deadline, aborting the group when it passes.

        Every broadcast left the current stream ordered behind it, so an event recorded now covers
        the whole chunk; polling that event keeps the host off the CUDA driver while a peer that never
        joins would otherwise park the trainer indefinitely (the group carries no watchdog the trainer
        can rely on). The abort ends the spinning kernel, without which every later device
        synchronization hangs too.
        """
        try:
            bounded_stream_sync(
                torch.cuda.current_stream(self.sync_device),
                timeout_s=_WEIGHT_UPDATE_TIMEOUT_S,
                what=f"{self.BACKEND_NAME} weight broadcast",
            )
        except RuntimeError:
            self._abort_group()
            raise

    def _abort_group(self) -> None:
        """Abort the weight-update group; torch drops its bookkeeping, so no destroy may follow."""
        self._aborted = True
        if self._group is None:
            return
        try:
            c10d._abort_process_group(self._group)
        except Exception as e:  # best-effort cleanup on an already-failing path
            logger.warning(f"Aborting the SGLang weight-update group failed: {e}")

    def _destroy_remote_group(self):
        """Drop the engine's registration of this client's group name, so the name can be joined again.

        Best-effort: this runs on setup and teardown paths where the server may be unreachable or may
        never have registered anything, and a failure here must not mask the original error.
        """
        try:
            self._post_if_supported(
                "/destroy_weights_update_group",
                timeout=_CLEANUP_TIMEOUT_S,
                json={"group_name": self.group_name},
            )
        except Exception as e:  # advisory cleanup; never mask the caller's failure
            logger.debug(f"Could not clear weight-update group {self.group_name!r} on {self.base_url}: {e}")

    def _release_group(self, *, local_destroy: bool = True):
        # The engine is asked to drop its half while the local destroy runs: under NCCL's cuMem
        # transports each side's finalize waits for the other, so telling the engine only after the
        # local destroy returned parks that destroy forever. The engine-side release also runs when
        # the local group is skipped or was aborted: a local teardown alone leaves the server
        # rejecting every future join under the name.
        remote = _AsyncCall(name="/destroy_weights_update_group", fn=self._destroy_remote_group)
        if local_destroy and not self._aborted:
            destroy_weight_update_group(self._group)
        else:
            drop_weight_update_group_bookkeeping(self._group)
        remote.join(_CLEANUP_TIMEOUT_S)
        self._group = None
        # Dropping the last store reference stops its daemon and frees the rendezvous port.
        self._store = None

    def close_communicator(self, *, _local_destroy: bool = True):
        # Close an update an interrupted sync left open and lift any pause with it, so generation is
        # not wedged after the trainer exits. An abort that left the engine unservable keeps its pause.
        self.abort_weight_update()
        self._lift_pause_at_close()
        self._release_group(local_destroy=_local_destroy)
        self._device_arena = None
        self._finalize_close()
