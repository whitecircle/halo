"""Weight sync to an SGLang server: the engine joins a plain c10d group and receives typed tensors in declared
order, one ``torch.distributed.broadcast`` each. Its server container needs ``NCCL_CUMEM_ENABLE=1``, which
SGLang otherwise turns off process-wide, failing the first cross-container import. The sglang package is
never imported; ``Dockerfile.sglang`` asserts the server side of this contract at build.
"""

import logging
from collections import deque
from collections.abc import Callable, Iterator
from functools import partial
from urllib.parse import urlparse

import torch
import torch.distributed as dist

from src.distributed.expert_parallel.layers.step3p7 import EPStep3p7MoELayer
from src.weight_sync.chunking import WEIGHT_SYNC_CHUNK_BYTES, describe_chunk
from src.weight_sync.clients.base import (
    CLEANUP_TIMEOUT_S,
    GROUP_FORMATION_TIMEOUT_S,
    HTTP_PROBE_TIMEOUT_S,
    POST_TIMEOUT_S,
    RESCORE_TIMEOUT_S,
    WEIGHT_UPDATE_TIMEOUT_S,
    AsyncCall,
    WeightSyncClientBase,
    unregistered_bailing_model_types,
)
from src.weight_sync.transport.bounded_waits import BROADCAST_DRAIN_TIMEOUT_S, bounded_event_sync, bounded_stream_sync
from src.weight_sync.transport.c10d_group import (
    abort_weight_update_group,
    build_weight_update_group,
    release_weight_update_group,
)
from src.weight_sync.transport.rendezvous import RendezvousListener
from src.weight_sync.transport.send_buffers import allocate_send_buffer, hold_for_stream

logger = logging.getLogger(__name__)

# Chunks whose sends the host has not settled yet, so a declaration never waits on the previous chunk's tail.
_INFLIGHT_CHUNKS = 2


class SGLangWeightSyncClient(WeightSyncClientBase):
    """Weight-sync client for an SGLang server."""

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
    # The MLA loaders fuse this pair from a cache local to one ``load_weights`` call and silently drop a half
    # that arrives alone; the fusion exists only where the model has both (``q_lora_rank`` set).
    CO_LOADED_PARAM_GROUPS = (("self_attn.q_a_proj.weight", "self_attn.kv_a_proj_with_mqa.weight"),)
    PAUSE_ENDPOINT = "/pause_generation"
    # The post-update flush asserts an idle scheduler and SIGQUITs the server otherwise, so requests are aborted.
    PAUSE_REQUEST = {"json": {"mode": "abort"}}
    RESUME_ENDPOINT = "/continue_generation"
    # The endpoint takes a request dataclass and rejects an empty body.
    RESUME_REQUEST = {"json": {}}
    _INIT_LOG = "SGLang NCCL init: engine_ws={engine_ws}, master={master}:{port}, listening on {bind}"
    _READY_LOG = "SGLang NCCL weight transfer initialized"
    _FORMATION_STEP = "weight-update group formation"
    _FORMATION_FAILURE = (
        "SGLang weight-update group formation failed (advertised master {master}:{port}; server {url}). If the "
        "SGLang node cannot reach that address, set {env} to an interface routable from it."
    )

    def __init__(self, *args, group_name: str | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        # One per server under SGLang's default prefix: c10d registers group names process-globally, and a
        # group_port of 0 (auto-pick) would collide every auto-port client on one name.
        self.group_name = group_name or f"weight_update_group_{urlparse(self.base_url).netloc}"

    def _fetch_engine_world_size(self) -> int:
        """``tp_size`` from ``/server_info``: each worker joins as ``rank_offset + tp_rank``, and under
        ``--enable-dp-attention`` ``tp_rank`` already spans every GPU. Plain ``--dp-size`` replicas each restart
        ``tp_rank`` at 0, so no sizing can address them."""
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

    def _request_engine_join(self, master_address: str, port: int, world_size: int) -> AsyncCall:
        # The engine registers the name on the join request and refuses a second join under it, so a stale
        # registration a dead trainer left is cleared first.
        self._destroy_remote_group()
        return self._start_collective_rpc(
            "/init_weights_update_group",
            timeout=GROUP_FORMATION_TIMEOUT_S,
            json={
                "master_address": master_address,
                "master_port": port,
                "rank_offset": 1,
                "world_size": world_size,
                "group_name": self.group_name,
                "backend": "nccl",
            },
        )

    def _prepare_local_formation(
        self, master_address: str, listener: RendezvousListener, world_size: int
    ) -> Callable[[], tuple[dist.ProcessGroup, dist.TCPStore]]:
        return partial(
            build_weight_update_group,
            master_address=master_address,
            listener=listener,
            world_size=world_size,
            device=self.sync_device,
            group_name=self.group_name,
            timeout_s=GROUP_FORMATION_TIMEOUT_S,
        )

    def _release_transport(self, *, at_exit: bool) -> None:
        # The engine drops its half while the local destroy runs, since under cuMem each side's finalize waits
        # for the other; it runs even with no local group, or the server refuses every later join.
        remote = AsyncCall(self._destroy_remote_group)
        group = None if self._transport is None else self._transport[0]
        # Interpreter exit skips the local NCCL destroy, and an aborted group is already gone from torch.
        release_weight_update_group(group, destroy=not at_exit and not self._aborted)
        remote.join(CLEANUP_TIMEOUT_S)
        self._arena = None
        self._inflight.clear()
        self._send_stream = None

    def _reset_transport_state(self) -> None:
        # Set once a drain deadline aborted the group, on which a destroy raises and a broadcast hangs.
        self._aborted = False
        # The device arena each chunk's uploads are carved from.
        self._arena: torch.Tensor | None = None
        # Events recorded after the latest chunks' broadcasts, settled _INFLIGHT_CHUNKS later.
        self._inflight: deque[torch.cuda.Event] = deque()
        # Uploads and broadcasts run here, never on the caller's stream, which a synchronous broadcast would
        # order behind the collective with the next chunk's copies and the trainer's own work.
        self._send_stream: torch.cuda.Stream | None = None

    def _start_engine_update(self) -> None:
        """Quiesce the engine, unwinding on failure; the chunk requests are the update."""
        if self._transport is None:
            raise RuntimeError("Call init_communicator() first")
        if self._aborted:
            raise RuntimeError(
                "SGLang weight-sync group was aborted after a broadcast deadline and cannot be reused "
                "— rebuild the client (reconnect) before syncing again."
            )
        # A collective issued with another device current fails with a bare "invalid argument" mid-update.
        if self.sync_device is not None and self.sync_device.type == "cuda":
            torch.cuda.set_device(self.sync_device)
        try:
            self._quiesce()
        except Exception:
            self._lift_pause(CLEANUP_TIMEOUT_S, context="after a failed pause")
            raise

    def _broadcast_chunk(self, chunk: list[tuple[str, torch.Tensor]], final: bool) -> None:
        """Declare one chunk, then broadcast its tensors in declared order; the cache flush rides the final one.
        The engine allocates every declared tensor before receiving any, so the chunk budget bounds it."""
        names, dtypes, shapes = describe_chunk(chunk)
        # The engine keys its receive buffers by name, so a repeated name puts every later send out of step.
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
                # Each flush asserts an idle scheduler, so only the final chunk asks for one.
                "flush_cache": final,
            },
        )
        # The engine replies once it queued the receives, so the sends may still be pending; they are settled
        # two chunks later, or here under a short deadline on failure.
        if self._send_stream is None:
            self._send_stream = torch.cuda.Stream(device=self.sync_device)
        caller_stream = torch.cuda.current_stream(self.sync_device)
        # The arena's block may last have been used on the caller's stream (``allocate_send_buffer``).
        self._send_stream.wait_stream(caller_stream)
        try:
            with torch.cuda.stream(self._send_stream):
                for tensor in self._upload_to_arena([param for _, param in chunk], caller_stream):
                    # Synchronous at stream level only; an async_op work per tensor would flood the watchdog.
                    dist.broadcast(tensor, src=0, group=self._transport[0])
                event = torch.cuda.Event()
                event.record(self._send_stream)
            self._inflight.append(event)
            server_call.wait(timeout=WEIGHT_UPDATE_TIMEOUT_S)
        except Exception as e:
            # A kernel whose peer failed spins and hangs every later device sync; the drain's deadline aborts
            # it. The whole stream is drained, as the failing chunk recorded no event of its own. A passed
            # deadline is the usual shape of the engine's own rejection, so that is read first.
            self._inflight.clear()
            drain_error: RuntimeError | None = None
            try:
                bounded_stream_sync(
                    self._send_stream,
                    timeout_s=CLEANUP_TIMEOUT_S,
                    what=f"{self.BACKEND_NAME} weight broadcast",
                    abort=self._abort_group,
                )
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

    def _finish_engine_update(self, tail: list[tuple[str, torch.Tensor]]) -> None:
        """Send the final chunk with the cache flush, then settle the sends and drop the arena."""
        if tail:
            self._transmit_chunk(tail, final=True)
        else:
            # With no chunk to ride on, the flush is requested directly, or the new weights serve a stale cache.
            self.request("POST", "/flush_cache", retry=True, timeout=POST_TIMEOUT_S)
        self._release_arena()

    def score_completion_logprobs(
        self, prompt_ids: list[int], completion_ids: list[int], timeout: float = RESCORE_TIMEOUT_S
    ) -> list[float]:
        """One prefill through ``/generate``: the window opens on the last prompt token (its entry, which carries no
        log-prob, is dropped) and every id is checked, so an off-by-one is a miss, never a shifted reference."""
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

    def _destroy_remote_group(self):
        """Drop the engine's registration of this client's group name; best-effort, never masks a failure."""
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

    def _upload_to_arena(self, params: list[torch.Tensor], pool_stream: torch.cuda.Stream) -> Iterator[torch.Tensor]:
        """Upload a chunk into the device arena, yielding each tensor's view as its copy is issued.

        One arena, not one allocation per tensor: a block freed mid-chunk carries a pending NCCL event the
        next upload would wait on inside the allocator. Reuse across chunks is safe by stream order, as uploads
        and sends share ``_send_stream``. The arena is sized to the chunk budget (grown for a larger tensor),
        drawn from ``pool_stream``'s pool and lives for one sync.
        """
        self._settle_inflight(keep=_INFLIGHT_CHUNKS - 1)
        offsets: list[int] = []
        total = 0
        for param in params:
            offsets.append(total)
            # 256-byte aligned, a multiple of every dtype's size, so a uint8 slice views as the tensor.
            total += (param.nbytes + 255) // 256 * 256
        if self._arena is None or self._arena.numel() < total:
            # Regrown only once the previous sends retired: they still read the old arena.
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

    def _settle_inflight(self, *, keep: int) -> None:
        """Wait for the oldest chunks' sends until ``keep`` remain in flight, under the drain deadline."""
        while len(self._inflight) > keep:
            bounded_event_sync(
                self._inflight.popleft(),
                timeout_s=BROADCAST_DRAIN_TIMEOUT_S,
                what=f"{self.BACKEND_NAME} weight broadcast",
                abort=self._abort_group,
            )

    def _abort_group(self) -> None:
        """Abort the group; ``_aborted`` keeps close from destroying a group torch already dropped."""
        self._aborted = True
        abort_weight_update_group(self._transport[0])

    def _release_arena(self) -> None:
        """Settle the outstanding sends and drop the arena, which held across syncs would pin the sync GPU."""
        self._settle_inflight(keep=0)
        self._arena = None
