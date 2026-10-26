"""NCCL weight sync for SGLang servers. Trainer is rank 0, engine workers are 1..N.

SGLang's contract differs from vLLM's in three ways that shape this client:

* Bootstrap: the engine joins a plain c10d group (``/init_weights_update_group``), so the trainer
  speaks c10d too (``src/weight_sync/transport/c10d_group.py``) rather than the vendored pynccl path.
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
from collections import deque
from collections.abc import Iterator
from urllib.parse import urlparse

import torch
import torch.distributed as dist

from src.distributed.expert_parallel.layers.step3p7 import EPStep3p7MoELayer
from src.weight_sync.chunking import WEIGHT_SYNC_CHUNK_BYTES, describe_chunk
from src.weight_sync.clients.base import (
    _SERVER_ERROR_GRACE_S,
    CLEANUP_TIMEOUT_S,
    GROUP_FORMATION_TIMEOUT_S,
    HTTP_PROBE_TIMEOUT_S,
    POST_TIMEOUT_S,
    RESCORE_TIMEOUT_S,
    WEIGHT_UPDATE_TIMEOUT_S,
    AsyncCall,
    WeightSyncClientBase,
    _wait_for_calls,
    unregistered_bailing_model_types,
)
from src.weight_sync.transport.bounded_waits import BROADCAST_DRAIN_TIMEOUT_S, bounded_event_sync, bounded_stream_sync
from src.weight_sync.transport.c10d_group import (
    abort_weight_update_group,
    build_weight_update_group,
    release_weight_update_group,
)
from src.weight_sync.transport.send_buffers import allocate_send_buffer, hold_for_stream

logger = logging.getLogger(__name__)

# SGLang's own default group name, the prefix of each client's per-server one.
_GROUP_NAME_PREFIX = "weight_update_group"
# Byte alignment of each tensor inside the staging arena: a multiple of every dtype's element size,
# so a uint8 slice can be viewed as the tensor's dtype.
_ARENA_ALIGNMENT = 256
# Chunks whose sends the host has not settled yet: a chunk's sends are waited for this many chunks
# later, so a chunk's declaration never waits on the previous chunk's tail (see ``_send_chunk``).
_INFLIGHT_CHUNKS = 2


def _aligned_bytes(nbytes: int) -> int:
    """``nbytes`` rounded up to the arena alignment."""
    return (nbytes + _ARENA_ALIGNMENT - 1) // _ARENA_ALIGNMENT * _ARENA_ALIGNMENT


class SGLangWeightSyncClient(WeightSyncClientBase):
    """NCCL weight sync client for SGLang servers."""

    BACKEND_KEY = "sglang"
    BACKEND_NAME = "SGLang"
    ENGINE_RELEASE = "SGLang 0.5.17"
    GROUP_HOST_ENV = "SGLANG_GROUP_HOST"
    UNSERVABLE_MODEL_TYPES = {
        "mistral4": f"{ENGINE_RELEASE} registers no Mistral4 model class (agent-docs/models/mistral4.md#serving)",
        **unregistered_bailing_model_types(ENGINE_RELEASE),
        "zaya": (
            f"{ENGINE_RELEASE}'s zaya loader reads the pre-transformers-5.14 per-expert checkpoint "
            "(zaya_block.experts.local_experts.N.linear_fc1), not the native fused layout the trainer holds"
        ),
        "laguna": (
            f"{ENGINE_RELEASE}'s laguna loader asserts every routed-expert tensor of every sparse layer in each "
            "load_weights call, so it refuses the chunked online update"
        ),
        # Every Step-3 spelling the trainer knows resolves to the same SGLang model file.
        **dict.fromkeys(
            EPStep3p7MoELayer.HF_MODEL_TYPES,
            f"{ENGINE_RELEASE}'s step3p5 loader asserts full parameter coverage in each load_weights call, so it "
            "refuses the chunked online update",
        ),
        "deepseek_v4": (
            f"{ENGINE_RELEASE}'s deepseek_v4 loader maps per-expert w1/w3/w2 names where the gather emits the "
            "fused pair, and no end-to-end sync has been validated for the family"
        ),
    }
    # SGLang's MLA loaders concatenate ``q_a_proj`` and ``kv_a_proj_with_mqa`` into one fused parameter
    # from a cache local to each ``load_weights`` call, so a half arriving without the other in the
    # same request is discarded without error. The fusion exists only where the model has both
    # projections (``q_lora_rank`` set), which the per-model scoping reads off the module tree.
    CO_LOADED_PARAM_GROUPS = (("self_attn.q_a_proj.weight", "self_attn.kv_a_proj_with_mqa.weight"),)
    RESUME_ENDPOINT = "/continue_generation"
    # An empty body is rejected: the endpoint takes a request dataclass, so it needs JSON.
    RESUME_REQUEST = {"json": {}}

    def __init__(self, *args, group_name: str | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        # Unique per server by default: c10d registers group names process-globally on the trainer, so
        # a second client re-using one fixed name fails with "group name has already been created".
        # Keyed on the server endpoint rather than group_port, where 0 means "auto-pick at group
        # formation" and would collide every auto-port client on one name.
        self.group_name = group_name or f"{_GROUP_NAME_PREFIX}_{urlparse(self.base_url).netloc}"

    def _reset_transport_state(self) -> None:
        self._group: dist.ProcessGroup | None = None
        self._store = None
        # Set once a drain deadline aborted the group; a destroy on an aborted group raises, and a
        # broadcast on one hangs.
        self._aborted = False
        # Device staging arena the chunks' uploads are carved from, and the events recorded after
        # the latest chunks' broadcasts (see ``_stage_on_device`` and ``_send_chunk``).
        self._arena: torch.Tensor | None = None
        self._inflight: deque[torch.cuda.Event] = deque()
        # Every upload and broadcast runs on this stream, never on the caller's: a synchronous
        # ``dist.broadcast`` orders the issuing stream behind the collective, and on the default
        # stream that would put the next chunk's staged copies (and the trainer's own work) behind
        # this chunk's sends, serializing the tail of one chunk with the next.
        self._send_stream: torch.cuda.Stream | None = None

    def score_completion_logprobs(
        self, prompt_ids: list[int], completion_ids: list[int], timeout: float = RESCORE_TIMEOUT_S
    ) -> list[float]:
        """SGLang's prefill log-probs come from its native ``/generate`` route: ``return_logprob`` with
        ``logprob_start_len`` returns one ``[logprob, token_id, text]`` entry per input token from
        that position on, and the first entry of the window is its anchor with no log-prob. The
        window therefore opens on the last prompt token, whose entry is dropped, and the completion's
        entries follow it. The ids are checked position by position — an off-by-one in the engine's
        contract must surface as a miss, never as a shifted reference.
        """
        if not prompt_ids:
            raise ValueError(f"{self.BACKEND_NAME} re-score needs at least one prompt token to anchor the window")
        body = {
            "input_ids": list(prompt_ids) + list(completion_ids),
            "sampling_params": {"max_new_tokens": 0},  # prefill-only request
            "return_logprob": True,
            "logprob_start_len": len(prompt_ids) - 1,
        }
        resp = self.request("POST", "/generate", retry=True, timeout=timeout, json=body)
        entries = resp.json()["meta_info"]["input_token_logprobs"]
        if len(entries) != len(completion_ids) + 1:
            raise ValueError(
                f"{self.BACKEND_NAME} returned {len(entries)} input log-probs for a {len(completion_ids)}-token "
                f"completion and its anchor"
            )
        anchor, entries = entries[0], entries[1:]
        if int(anchor[1]) != int(prompt_ids[-1]):
            raise ValueError(
                f"{self.BACKEND_NAME} input log-probs open on token {anchor[1]}, not the last prompt token {prompt_ids[-1]}"
            )
        values = []
        for position, (tok, entry) in enumerate(zip(completion_ids, entries, strict=True)):
            if int(entry[1]) != int(tok):
                raise ValueError(
                    f"{self.BACKEND_NAME} input log-probs are misaligned at completion position {position}: "
                    f"token {entry[1]} where {tok} was sent"
                )
            values.append(float(entry[0]))
        return values

    def fetch_engine_world_size(self) -> int:
        """Number of engine ranks that will join the group.

        SGLang assigns each worker ``rank_offset + tp_rank``. Under ``--enable-dp-attention``
        ``tp_rank`` already enumerates every GPU (one TP group spans the server), so the group is sized
        by ``tp_size`` alone; multiplying by ``dp_size`` oversizes the c10d group and hangs formation.
        Plain ``--dp-size`` replicas each restart ``tp_rank`` at 0, so their ranks collide in the
        update group and no sizing can address them, which is rejected below. The layout is read from
        the server rather than from config so the two cannot disagree after a serve-flag change.
        """
        info = self.request("GET", "/server_info", retry=True, timeout=HTTP_PROBE_TIMEOUT_S).json()
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
        self._resolve_sync_device(device)
        master_address, listener = self._open_group_rendezvous()
        master_port = listener.port

        logger.info(
            f"SGLang NCCL init: engine_ws={engine_ws}, master={master_address}:{master_port}, "
            f"listening on {listener.bind_address}"
        )

        # Closed on exit unless the store took the listener over.
        with listener:
            # The engine registers the group by name as soon as it is asked to join and rejects any
            # later join under that name, so a trainer that died between the request and a working
            # group leaves the server unable to weight-sync until restarted. Clearing a stale
            # registration first is a no-op on the first attempt.
            self._destroy_remote_group()

            server_call = self._start_collective_rpc(
                "/init_weights_update_group",
                timeout=GROUP_FORMATION_TIMEOUT_S,
                json={
                    "master_address": master_address,
                    "master_port": master_port,
                    "rank_offset": 1,
                    "world_size": world_size,
                    "group_name": self.group_name,
                    "backend": "nccl",
                },
            )
            group_call: AsyncCall | None = None
            try:
                # Both halves block until the other arrives (the engine's request returns once the
                # group formed, group creation returns once the engine joined), so they run
                # concurrently and whichever fails first wins.
                group_call = AsyncCall(
                    name="weight-update group formation",
                    fn=lambda: build_weight_update_group(
                        master_address=master_address,
                        listener=listener,
                        world_size=world_size,
                        device=self.sync_device,
                        group_name=self.group_name,
                        timeout_s=GROUP_FORMATION_TIMEOUT_S,
                    ),
                )
                _wait_for_calls([server_call, group_call], timeout=GROUP_FORMATION_TIMEOUT_S)
                self._group, self._store = group_call.result
            except Exception as e:
                # `_wait_for_calls` raises as soon as either half fails, so a group this rank already
                # formed lives only on the call handle. Adopt it before releasing, or c10d keeps the
                # registration under `self.group_name` and the store keeps its listener on
                # `master_port`, and every later reconnect is rejected with "group name has already
                # been created".
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
                    f"Original error: {type(e).__name__}: {e}"
                ) from e

        # Interpreter exit skips the local NCCL destroy (release_weight_update_group says why); explicit
        # close_communicator() calls, the reconnect path, keep it.
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
            self.request("POST", "/pause_generation", retry=True, timeout=POST_TIMEOUT_S, json={"mode": "abort"})
        except Exception:
            self._lift_pause(CLEANUP_TIMEOUT_S, context="after a failed pause")
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
                self.request("POST", "/flush_cache", retry=True, timeout=POST_TIMEOUT_S)
            self._release_arena()
        finally:
            # Always lift the pause; a server left paused wedges every later rollout.
            self._lift_pause(CLEANUP_TIMEOUT_S, context="after sync")

    def _send_chunk(self, chunk: list[tuple[str, torch.Tensor]], flush_cache: bool):
        """Declare one chunk over HTTP, then broadcast its tensors in the declared order.

        The engine allocates every tensor in the chunk, issues an async broadcast per name and then
        waits on all of them, so the trainer must issue matching broadcasts in exactly this order.
        """
        names, dtypes, shapes = describe_chunk(chunk)
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

        server_call = self._start_collective_rpc(
            "/update_weights_from_distributed",
            timeout=WEIGHT_UPDATE_TIMEOUT_S,
            json={
                "names": names,
                "dtypes": dtypes,
                "shapes": shapes,
                "group_name": self.group_name,
                # Only the final chunk invalidates the cache: an intermediate flush would run once
                # per chunk for no benefit, and each one asserts an idle scheduler.
                "flush_cache": flush_cache,
            },
        )
        # The engine replies once it has queued the receives and the load (its wait is a stream
        # wait), so this chunk's sends may still be pending. The host does not wait for them here,
        # where it would serialize with the next chunk's declaration: the sends are settled two
        # chunks later (``_stage_on_device``) and, on any failure, right here under a short deadline
        # that aborts the group.
        if self._send_stream is None:
            self._send_stream = torch.cuda.Stream(device=self.sync_device)
        caller_stream = torch.cuda.current_stream(self.sync_device)
        # The arena's block may last have been used on the caller's stream (``allocate_send_buffer``).
        self._send_stream.wait_stream(caller_stream)
        try:
            with torch.cuda.stream(self._send_stream):
                for tensor in self._stage_on_device([param for _, param in chunk], caller_stream):
                    # Synchronous only at stream level: ProcessGroupNCCL's wait() orders the send
                    # stream behind the collective and returns. An async_op work per tensor would
                    # hand hundreds of outstanding works to the watchdog's per-work event polling.
                    dist.broadcast(tensor, src=0, group=self._group)
                event = torch.cuda.Event()
                event.record(self._send_stream)
            self._inflight.append(event)
            server_call.wait(timeout=WEIGHT_UPDATE_TIMEOUT_S)
        except Exception as e:
            # A kernel whose peer answered with an error, or never answered, spins on the stream and
            # hangs every later device synchronization; the abort ends it. The whole stream is
            # drained: a chunk that failed mid-upload recorded no event of its own. A drain that
            # passes its deadline is the usual shape of the engine's own rejection, so the server's
            # error is still read before the drain's is raised.
            drain_error: RuntimeError | None = None
            try:
                self._drain_sends(timeout_s=CLEANUP_TIMEOUT_S)
            except RuntimeError as deadline:
                drain_error = deadline
            self._raise_if_server_failed(
                e,
                server_call,
                hint=(
                    " The engine reports a partially updated model after such a failure — restart "
                    "the server before serving again."
                ),
            )
            if drain_error is not None:
                raise drain_error from e
            raise

    def _stage_on_device(self, params: list[torch.Tensor], pool_stream: torch.cuda.Stream) -> Iterator[torch.Tensor]:
        """Upload a chunk into the device arena, yielding each tensor's view as it is issued.

        One arena rather than one allocation per tensor: a block freed mid-chunk returns to the
        caching allocator carrying a pending event on the NCCL stream, and the next upload that reuses
        it waits on that event inside the allocator — a host block inside the very collective the
        settle deadline bounds — while fresh allocations per chunk churn the allocator on the critical
        path. Reusing it across chunks is safe by stream order alone: uploads and sends share
        ``_send_stream``, so a chunk's copies queue behind the previous chunk's sends on the device
        with no host wait. The host settles a chunk's sends two chunks later, which bounds a wedged
        peer without serializing each chunk's tail with the next declaration. The arena is sized to
        the chunk budget, grown for a tensor above it, and lives for one sync (``_release_arena``).

        The arena is drawn from ``pool_stream``'s pool (the caller's, which the send stream already
        waits on), not the send stream's, which this client owns (:func:`allocate_send_buffer`), and
        each source is held for the send stream that copies it (:func:`hold_for_stream`).

        A generator so each view is handed over right after its copy is issued, in the order the
        engine receives them.
        """
        self._settle_inflight(keep=_INFLIGHT_CHUNKS - 1)
        offsets: list[int] = []
        total = 0
        for param in params:
            offsets.append(total)
            total += _aligned_bytes(param.nbytes)
        if self._arena is None or self._arena.numel() < total:
            # Regrown only once the previous chunk's sends retired: the old arena is still their input,
            # and dropping it under a pending collective leaves the allocator to wait on that
            # collective, outside every deadline here, if the larger block does not fit beside it.
            self._release_arena()
            self._arena = allocate_send_buffer(
                max(total, WEIGHT_SYNC_CHUNK_BYTES),
                self.sync_device,
                pool_stream=pool_stream,
                send_stream=self._send_stream,
            )
        arena = self._arena
        for param, offset in zip(params, offsets, strict=True):
            view = arena[offset : offset + param.nbytes].view(param.dtype).view(param.shape)
            view.copy_(param, non_blocking=True)
            hold_for_stream(param, self._send_stream)
            yield view

    def _release_arena(self) -> None:
        """Settle the outstanding sends and drop the arena, at the end of a sync.

        Held across syncs it would pin the sync GPU at the largest chunk's size (the largest tensor,
        once a fused expert pair exceeds the budget) for the whole run; dropped, that memory is the
        allocator's between syncs and the next sync's first chunk takes it back from the pool. One
        settle per sync rather than per chunk keeps the deferred settle's overlap.
        """
        self._settle_inflight(keep=0)
        self._arena = None

    def _settle_inflight(self, *, keep: int) -> None:
        """Wait for the oldest chunks' sends until ``keep`` chunks remain in flight, under the drain deadline.

        Polling each chunk's event keeps the host off the CUDA driver while a peer that never joins would
        otherwise park the trainer indefinitely; the group carries no watchdog the trainer can rely on.
        """
        while len(self._inflight) > keep:
            bounded_event_sync(
                self._inflight.popleft(),
                timeout_s=BROADCAST_DRAIN_TIMEOUT_S,
                what=f"{self.BACKEND_NAME} weight broadcast",
                abort=self._abort_group,
            )

    def _drain_sends(self, *, timeout_s: float) -> None:
        """Settle everything issued on the send stream so far, on the failure path.

        The failing chunk recorded no event of its own, and the previous chunk's sends may be
        spinning on the same peer, so the stream is drained as a whole rather than by chunk.
        """
        self._inflight.clear()
        bounded_stream_sync(
            self._send_stream,
            timeout_s=timeout_s,
            what=f"{self.BACKEND_NAME} weight broadcast",
            abort=self._abort_group,
        )

    def _abort_group(self) -> None:
        """Abort the group; ``_aborted`` keeps close from destroying a group torch already dropped."""
        self._aborted = True
        abort_weight_update_group(self._group)

    def _destroy_remote_group(self):
        """Drop the engine's registration of this client's group name, so the name can be joined again.

        Best-effort: this runs on setup and teardown paths where the server may be unreachable or may
        never have registered anything, and a failure here must not mask the original error.
        """
        try:
            resp = self.request(
                "POST",
                "/destroy_weights_update_group",
                retry=False,
                timeout=CLEANUP_TIMEOUT_S,
                check=False,
                json={"group_name": self.group_name},
            )
            if resp.status_code != 404:  # a server without the route has no registration to drop
                resp.raise_for_status()
        except Exception as e:  # advisory cleanup; never mask the caller's failure
            logger.debug(f"Could not clear weight-update group {self.group_name!r} on {self.base_url}: {e}")

    def _release_group(self, *, local_destroy: bool = True):
        # The engine is asked to drop its half while the local destroy runs: under NCCL's cuMem
        # transports each side's finalize waits for the other, so telling the engine only after the
        # local destroy returned parks that destroy forever. The engine-side release also runs when
        # the local group is skipped or was aborted: a local teardown alone leaves the server
        # rejecting every future join under the name.
        remote = AsyncCall(self._destroy_remote_group)
        # An aborted group is already gone from torch, and a destroy on it raises.
        release_weight_update_group(self._group, destroy=local_destroy and not self._aborted)
        remote.join(CLEANUP_TIMEOUT_S)
        self._group = None
        # Dropping the last store reference stops its daemon and frees the rendezvous port.
        self._store = None

    def close_communicator(self, *, _local_destroy: bool = True):
        # Close an update an interrupted sync left open and lift any pause with it, so generation is
        # not wedged after the trainer exits. An abort that left the engine unservable keeps its pause.
        self.abort_weight_update()
        self._lift_pause_at_close()
        # A client that never formed its group (or already released it) has nothing to release, and
        # the engine-side POST would only delay the close.
        if self._group is not None or self._store is not None:
            self._release_group(local_destroy=_local_destroy)
        self._arena = None
        self._inflight.clear()
        self._send_stream = None
        self._finalize_close()
