"""NCCL weight sync for SGLang servers. Trainer is rank 0, engine workers are 1..N.

SGLang's contract differs from vLLM's in three ways that shape this client:

* **Bootstrap** — the engine joins a plain c10d group (``/init_weights_update_group``), so the
  trainer speaks c10d too (``transport/torch_group.py``) rather than the vendored pynccl path.
* **Wire format** — the engine allocates one empty tensor per declared name and receives each with
  ``torch.distributed.broadcast(src=0)``. Weights therefore travel as typed tensors in declaration
  order, not as the packed uint8 buffers the vLLM protocol uses.
* **Quiescence** — the engine's post-update cache flush asserts the scheduler is fully idle, and a
  failed assert takes down the whole server rather than returning an error. Every sync is bracketed
  by ``/pause_generation`` and ``/continue_generation``.

The training image never imports sglang (its transformers pin conflicts); everything here is HTTP
plus torch. ``Dockerfile.sglang`` asserts the server side of this contract at image build.
"""

import atexit
import logging
from urllib.parse import urlparse

import torch
import torch.distributed as dist

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
from src.distributed.nccl.transport.torch_group import (
    DEFAULT_WEIGHT_UPDATE_GROUP_NAME,
    create_weight_update_group,
    destroy_weight_update_group,
    drop_weight_update_group_bookkeeping,
)

logger = logging.getLogger(__name__)

# Engine control-plane routes reached from more than one call site (the call itself and its
# async-call label, or both the sync and the teardown path). Spelled once so a route rename cannot
# leave a stale label pointing at an endpoint that no longer exists.
_EP_INIT_GROUP = "/init_weights_update_group"
_EP_UPDATE_FROM_DIST = "/update_weights_from_distributed"
_EP_CONTINUE = "/continue_generation"


class SGLangWeightSyncClient(BaseWeightSyncClient):
    """NCCL weight sync client for SGLang servers."""

    BACKEND_KEY = "sglang"
    BACKEND_NAME = "SGLang"
    GROUP_HOST_ENV = "SGLANG_GROUP_HOST"
    # SGLang loads MoE experts as transformers stores them: one fused pair per layer.
    EXPERT_LAYOUT = BaseWeightSyncClient.FUSED_EXPERT_LAYOUT
    # The cross-container sync needs NCCL's CUDA-IPC transports disabled; DeepEP needs them enabled
    # for symmetric memory. Both are process-global and NCCL caches them on first read, so one
    # process cannot serve both — see agent-docs/infrastructure/rollout-servers.md.
    SUPPORTS_EXPERT_PARALLEL = False
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
        super().__init__(
            base_url=base_url,
            group_port=group_port,
            connection_timeout=connection_timeout,
            group_host=group_host,
        )
        # Unique per server by default: c10d registers group names process-globally on the trainer,
        # so a second server's client re-using one fixed name fails with "group name has already
        # been created". Keyed on the server endpoint (unique per server by construction), not on
        # group_port — 0 there means "auto-pick at group formation" and would collide every
        # auto-port client on one name. The server side joins under this name via the init request.
        self.group_name = group_name or f"{DEFAULT_WEIGHT_UPDATE_GROUP_NAME}_{urlparse(self.base_url).netloc}"

    def fetch_engine_world_size(self) -> int:
        """Number of engine ranks that will join the group.

        SGLang assigns each worker ``rank_offset + tp_rank``. Under ``--enable-dp-attention``
        ``tp_rank`` already enumerates every GPU (one TP group spans the server), so the group is
        sized by ``tp_size`` alone — multiplying by ``dp_size`` oversizes the c10d group and hangs
        formation. Plain ``--dp-size`` replicas each restart ``tp_rank`` at 0, so their ranks
        collide in the update group — no sizing can address them; refused. Reading the layout from
        the server rather than from config keeps the two from disagreeing after a serve-flag change.
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

        # The engine registers the group by NAME the moment it is asked to join, and then refuses any
        # later join under that name. A trainer that died between the request and a working group —
        # or simply lost the race to bind its port — therefore leaves the server permanently unable to
        # weight-sync until it is restarted. Clear a stale registration first; it is a no-op the first
        # time round, and the difference between a recoverable and an unrecoverable server after any
        # crash.
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
            # Both halves block until the other arrives — the engine's request only returns once the
            # group formed, and group creation only returns once the engine joined — so they run
            # concurrently and whichever fails first wins.
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
            # `_wait_for_calls` raises the moment EITHER half fails, so a group this rank already
            # formed is still unowned — it lives only on the call handle. Adopt it before releasing,
            # or c10d keeps the registration under `self.group_name` and the store keeps its
            # listener on `master_port`, and every later reconnect to this server is refused with
            # "The specified group name has already been created".
            formed = group_call.join(_SERVER_ERROR_GRACE_S).result if group_call is not None else None
            if formed is not None:
                self._group, self._store = formed
            self._release_group()
            if e is server_call.error:
                raise  # the server named its own failure; a topology hint would only mislead
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
        """Quiesce the engine. SGLang has no separate reload phase — the chunk requests are the update.

        Unwinds its own quiesce on failure: nothing has been broadcast yet, so the server must be
        left serving rather than paused.
        """
        if self._group is None:
            raise RuntimeError("Call init_communicator() first")
        # The group was formed against this device; a collective issued while a different device is
        # current fails inside NCCL with a bare "invalid argument", after the engine has already
        # allocated its receive buffers and considers itself mid-update.
        if self.sync_device is not None and self.sync_device.type == "cuda":
            torch.cuda.set_device(self.sync_device)
        self._paused = True
        try:
            # The engine's post-update flush asserts a fully idle scheduler and SIGQUITs the server
            # when that fails, so draining first is a correctness requirement, not an optimization.
            self._post("/pause_generation", json={"mode": "abort"})
        except Exception:
            self._lift_pause(_CLEANUP_TIMEOUT_S, context="after a failed pause")
            raise

    def _broadcast_chunk(self, chunk: list[tuple[str, torch.Tensor]], final: bool = False):
        """Broadcast one chunk; the cache invalidation rides the final one.

        The caller's byte budget is the engine's allocation bound: it allocates ``torch.empty`` for
        EVERY name in a request before receiving any of them.
        """
        self._send_chunk(chunk, flush_cache=final)

    def end_weight_update(self, tail: list[tuple[str, torch.Tensor]]):
        """Send the final chunk with the cache flush, then resume — resuming on every path."""
        try:
            tail = list(tail)
            if tail:
                self.send_weights(tail, final=True)
            else:
                # The cache invalidation rides the last chunk; with nothing left to send it has to be
                # asked for directly, or the engine serves the new weights against a stale prefix cache.
                self._post("/flush_cache")
        finally:
            # Always lift the pause — a server left paused wedges every later rollout.
            self._lift_pause(_CLEANUP_TIMEOUT_S, context="after sync")

    def _send_chunk(self, chunk: list[tuple[str, torch.Tensor]], flush_cache: bool):
        """Declare one chunk over HTTP, then broadcast its tensors in the declared order.

        The engine allocates every tensor in the chunk, issues an async broadcast per name and then
        waits on all of them, so the trainer must issue matching broadcasts in exactly this order.
        """
        names = [name for name, _ in chunk]
        dtypes = [str(param.dtype).split(".")[-1] for _, param in chunk]
        shapes = [list(param.shape) for _, param in chunk]

        # A repeated name desynchronizes the stream: the engine keys its receive buffers by name, so
        # it allocates and reads the duplicate ONCE while this loop broadcasts it twice. Every later
        # tensor then lands one send out of step and NCCL reports a truncated message
        # ("received N bytes instead of M") naming a param that is not the faulty one — so refuse
        # here, where the offender is still identifiable, rather than mid-broadcast.
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
                    # Only the final chunk invalidates the cache: an intermediate flush would run
                    # once per chunk for no benefit, and each one asserts an idle scheduler.
                    "flush_cache": flush_cache,
                },
            ),
        )
        try:
            for _, param in chunk:
                # Buffered params live in pinned host memory; upload one at a time so only a single
                # tensor transits the GPU, mirroring the vLLM producer's staging. The upload is bound
                # to a local first: passing the temporary inline would drop the last Python reference
                # at statement end, leaving the live NCCL read relying purely on allocator stream
                # bookkeeping.
                staged = param.to(self.sync_device, non_blocking=True)
                # Blocking form, but that is no proof of delivery: ProcessGroupNCCL's wait() only
                # orders the local stream either way. What rules out a sync that never reached the
                # engine — and a served policy left silently stale — is the server's own reply,
                # joined at server_call.wait() below. Synchronous here keeps the sends in the
                # declared order with no handles to track.
                dist.broadcast(staged, src=0, group=self._group)
                del staged
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

    def _destroy_remote_group(self):
        """Drop the engine's registration of this client's group name, so the name can be joined again.

        Best-effort on purpose: this runs on setup and teardown paths where the server may be
        unreachable or may never have registered anything, and a failure here must not mask the error
        that led here.
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
        if local_destroy:
            destroy_weight_update_group(self._group)
        else:
            drop_weight_update_group_bookkeeping(self._group)
        self._group = None
        # Dropping the last store reference stops its daemon and frees the rendezvous port.
        self._store = None
        # Release the name on the engine too — a local teardown alone would leave the server refusing
        # every future join under it.
        self._destroy_remote_group()

    def close_communicator(self, *, _local_destroy: bool = True):
        # Close an update an interrupted sync left open, and lift any pause with it, so generation is
        # not wedged after the trainer exits — unless the abort left the engine unservable, which
        # keeps its pause on purpose.
        self.abort_weight_update()
        self._lift_pause_at_close()
        self._release_group(local_destroy=_local_destroy)
        self._finalize_close()
