"""Weight sync to an SGLang server: a plain c10d group, one typed broadcast per tensor in declared order. The server
needs ``NCCL_CUMEM_ENABLE=1``; SGLang otherwise turns cuMem off and the first cross-container import fails."""

import logging
from collections import deque
from collections.abc import Callable, Iterator
from functools import partial
from typing import NoReturn
from urllib.parse import urlparse

import requests
import torch
import torch.distributed as dist

from halo.models.moe.family_specs import STEP3P7
from halo.weight_sync.chunking import WEIGHT_SYNC_CHUNK_BYTES, describe_chunk
from halo.weight_sync.clients.base import (
    CLEANUP_TIMEOUT_S,
    GROUP_FORMATION_TIMEOUT_S,
    HTTP_PROBE_TIMEOUT_S,
    POST_TIMEOUT_S,
    RESCORE_TIMEOUT_S,
    SESSION_CLOSE_TIMEOUT_S,
    GroupMembershipLost,
    ServerFlags,
    WeightSyncClientBase,
    unservable_bailing_model_types,
)
from halo.weight_sync.clients.server_calls import AsyncCall
from halo.weight_sync.transport.bounded_waits import CHUNK_SYNC_TIMEOUT_S, bounded_event_sync, bounded_stream_sync
from halo.weight_sync.transport.c10d_group import (
    abort_weight_update_group,
    build_weight_update_group,
    release_weight_update_group,
)
from halo.weight_sync.transport.packed_broadcast import allocate_send_buffer, hold_for_stream
from halo.weight_sync.transport.rendezvous import RendezvousListener

logger = logging.getLogger(__name__)

# Chunks whose sends the host has not settled yet, so a declaration never waits on the previous chunk's tail.
_INFLIGHT_CHUNKS = 2
# Past the 20 s the server's /health gives a scheduler step before answering 503 itself, so a busy engine answers.
_ENGINE_ANSWER_TIMEOUT_S = 30.0


class SGLangWeightSyncClient(WeightSyncClientBase):
    BACKEND_KEY = "sglang"
    BACKEND_NAME = "SGLang"
    # /health generates through the scheduler, so a pause a dead trainer left would read as an unreachable server.
    LIVENESS_ROUTE = "/v1/models"
    ENGINE_RELEASE = "SGLang 0.5.22"
    GROUP_HOST_ENV = "SGLANG_GROUP_HOST"
    UNSERVABLE_MODEL_TYPES = {
        "mistral4": f"{ENGINE_RELEASE} registers no Mistral4 model class (agent-docs/models/mistral4.md#serving)",
        **unservable_bailing_model_types(ENGINE_RELEASE),
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
            STEP3P7.model_types,
            f"{ENGINE_RELEASE}'s step3p5 loader asserts full parameter coverage in each load_weights call, so it "
            "refuses the chunked online update",
        ),
        "deepseek_v4": (
            f"{ENGINE_RELEASE}'s deepseek_v4 loader maps per-expert w1/w3/w2 names where the gather emits the "
            "fused pair, and no end-to-end sync has been validated for the family"
        ),
    }
    # The MLA loaders fuse this pair from a cache local to one ``load_weights`` call and silently drop a lone half.
    CO_LOADED_PARAM_GROUPS = (("self_attn.q_a_proj.weight", "self_attn.kv_a_proj_with_mqa.weight"),)
    PAUSE_ENDPOINT = "/pause_generation"
    # The post-update flush needs an idle scheduler, so in-flight requests are aborted.
    PAUSE_REQUEST = {"json": {"mode": "abort"}}
    PAUSE_ABORTS_IN_FLIGHT = True
    RESUME_ENDPOINT = "/continue_generation"
    # The endpoint takes a request dataclass and rejects an empty body.
    RESUME_REQUEST = {"json": {}}
    # The engine refuses an update outside this session; its close re-runs each layer's post-load processing.
    SESSION_OPEN_ENDPOINT = "/begin_weight_update"
    SESSION_CLOSE_ENDPOINT = "/end_weight_update"
    SESSION_REQUEST = {"json": {}}
    IMAGE_REMEDY = "Serve from the toolkit's SGLang image (make build-sglang), or its docker-compose.sglang.yml."
    MOE_BACKEND_FLAG = "--moe-runner-backend triton (SGLANG_MOE_RUNNER_BACKEND)"
    REASONING_PARSER_FLAG = "--reasoning-parser (SGLANG_REASONING_PARSER)"
    TOOL_PARSER_FLAG = "--tool-call-parser (SGLANG_TOOL_PARSER, the family's name where auto cannot read the template)"
    CHAT_TEMPLATE_FLAG = "--chat-template (SGLANG_CHAT_TEMPLATE)"
    _INIT_LOG = "SGLang NCCL init: engine_ws={engine_ws}, master={master}:{port}, listening on {bind}"
    _READY_LOG = "SGLang NCCL weight transfer initialized"
    _FORMATION_STEP = "weight-update group formation"
    _FORMATION_FAILURE = (
        "SGLang weight-update group formation failed (advertised master {master}:{port}; server {url}). If the "
        "SGLang node cannot reach that address, set {env} to an interface routable from it."
    )

    def __init__(self, *args, group_name: str | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        # c10d registers group names process-globally, and keying on group_port 0 would collide auto-port clients.
        self.group_name = group_name or f"weight_update_group_{urlparse(self.base_url).netloc}"

    def read_server_flags(self) -> ServerFlags:
        return self._flags_in(self._server_info())

    @staticmethod
    def _flags_in(info: dict) -> ServerFlags:
        return ServerFlags(
            toolkit_image=bool(info.get("halo_patches")),
            moe_backend=info.get("moe_runner_backend"),
            # A server with no parser reports null; a reply without the key says nothing.
            reasoning_parser=(info["reasoning_parser"] or "") if "reasoning_parser" in info else None,
            tool_call_parser=(info["tool_call_parser"] or "") if "tool_call_parser" in info else None,
            captures_routing=info.get("enable_return_routed_experts"),
            torch_compile=info.get("enable_torch_compile"),
        )

    def _server_info(self) -> dict:
        return self.request("GET", "/server_info", retry=True, timeout=HTTP_PROBE_TIMEOUT_S).json()

    def fetch_engine_world_size(self) -> int:
        """Workers join as ``rank_offset + tp_rank``; under ``--attn-dp-size`` ``tp_rank`` spans every GPU."""
        info = self._server_info()
        self._require_toolkit_image(self._flags_in(info))
        self.probe_generation()
        try:
            tp_size = int(info["tp_size"])
            dp_size = int(info.get("dp_size", 1) or 1)
        except (KeyError, TypeError, ValueError) as e:
            raise RuntimeError(
                f"Could not read the parallel layout from {self.base_url}/server_info "
                f"(need 'tp_size', optionally 'dp_size'); got keys {sorted(info)[:25]}. "
                f"The weight-sync group cannot be sized without it."
            ) from e
        if dp_size > 1:
            raise RuntimeError(
                f"SGLang weight sync cannot address a plain --dp-size {dp_size} server: each "
                f"replica restarts tp_rank at 0, so workers collide on rank_offset + tp_rank in "
                f"the weight-update group. Serve with --attn-dp-size or --dp-size 1."
            )
        return tp_size

    def request_engine_join(
        self, master_address: str, port: int, world_size: int, rank_offset: int = 1, group_name: str | None = None
    ) -> AsyncCall:
        if group_name is not None:
            self.group_name = group_name
        # The engine refuses a second join under a registered name, so a dead trainer's entry is cleared first, and a
        # second begin under its open session. Unchecked, as with no session open the close answers 400.
        self.destroy_remote_group()
        self.request(
            "POST",
            self.SESSION_CLOSE_ENDPOINT,
            retry=False,
            timeout=SESSION_CLOSE_TIMEOUT_S,
            check=False,
            **self.SESSION_REQUEST,
        )
        return self._start_collective_rpc(
            "/init_weights_update_group",
            timeout=GROUP_FORMATION_TIMEOUT_S,
            json={
                "master_address": master_address,
                "master_port": port,
                "rank_offset": rank_offset,
                "world_size": world_size,
                "group_name": self.group_name,
                "backend": "nccl",
            },
        )

    def prepare_local_formation(
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
        group = None if self._transport is None else self._transport[0]
        # Interpreter exit skips the local NCCL destroy, and an aborted group is already gone from torch.
        destroy = not at_exit and not self._aborted
        # Ahead of the remote destroy: /health generates through the scheduler that destroy occupies, so asked during
        # it a healthy engine reads as dead, and aborting here strands its half of the destroy.
        answers = not (destroy and group is not None) or self._engine_answers()
        # Runs during the local destroy, since under cuMem each side's finalize waits for the other, and even with
        # no local group, or the server refuses every later join.
        remote = AsyncCall(self.destroy_remote_group)
        if not answers:
            logger.warning(f"SGLang server {self.base_url} does not answer, so its weight-update group is aborted.")
            self.abort_transport()
            destroy = False
        # An engine that never enters its half parks the destroy, with the failure that brought us here unraised.
        local = AsyncCall(partial(release_weight_update_group, group, destroy=destroy), name="weight-update destroy")
        if local.join(CLEANUP_TIMEOUT_S).is_alive():
            logger.error(f"Destroying the weight-update group of {self.base_url} is still blocked; aborting it.")
            self.abort_transport()
        remote.join(CLEANUP_TIMEOUT_S)
        self._arena = None
        self._inflight.clear()
        self._send_stream = None

    def _reset_transport_state(self) -> None:
        # Set once a drain deadline aborted the group, on which a destroy raises and a broadcast hangs.
        self._aborted = False
        self._arena: torch.Tensor | None = None
        # Events recorded after the latest chunks' broadcasts, settled _INFLIGHT_CHUNKS later.
        self._inflight: deque[torch.cuda.Event] = deque()
        # On the caller's stream a synchronous broadcast would order the next copies and the trainer's work behind it.
        self._send_stream: torch.cuda.Stream | None = None

    def require_transport(self) -> None:
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

    def prepare_chunk_send(self, chunk: list[tuple[str, torch.Tensor]]) -> None:
        names = [name for name, _ in chunk]
        if len(set(names)) != len(names):
            repeated = sorted({name for name in names if names.count(name) > 1})
            raise ValueError(
                f"Duplicate parameter name(s) in one weight-sync chunk: {repeated}. The engine "
                f"receives each name once, so the broadcast order would desynchronize."
            )

    def declare_chunk(self, chunk: list[tuple[str, torch.Tensor]], final: bool) -> AsyncCall:
        """The engine allocates every declared tensor before receiving any, so the chunk budget bounds it."""
        names, dtypes, shapes = describe_chunk(chunk)
        return self._start_collective_rpc(
            "/update_weights_from_distributed",
            timeout=CHUNK_SYNC_TIMEOUT_S,
            json={
                "names": names,
                "dtypes": dtypes,
                "shapes": shapes,
                "group_name": self.group_name,
                # Each flush needs an idle scheduler, so only the final chunk asks for one.
                "flush_cache": final,
            },
        )

    def send_chunk(self, chunk: list[tuple[str, torch.Tensor]]) -> None:
        """The engine replies on queueing its receives; the sends settle two chunks later, or on failure."""
        if self._send_stream is None:
            self._send_stream = torch.cuda.Stream(device=self.sync_device)
        caller_stream = torch.cuda.current_stream(self.sync_device)
        # The arena's block may last have been used on the caller's stream (``allocate_send_buffer``).
        self._send_stream.wait_stream(caller_stream)
        with torch.cuda.stream(self._send_stream):
            for tensor in self._upload_to_arena([param for _, param in chunk], caller_stream):
                # Synchronous at stream level only; an async_op work per tensor would flood the watchdog.
                dist.broadcast(tensor, src=0, group=self._transport[0])
            event = torch.cuda.Event()
            event.record(self._send_stream)
        self._inflight.append(event)

    def _fail_chunk(self, error: Exception, declared: AsyncCall) -> NoReturn:
        # A kernel whose peer failed spins and hangs every later device sync, so the drain's deadline aborts it.
        # A passed deadline is usually the engine's own rejection, so that is read first.
        self._inflight.clear()
        drain_error: RuntimeError | None = None
        if self._send_stream is not None:
            try:
                bounded_stream_sync(
                    self._send_stream,
                    timeout_s=CLEANUP_TIMEOUT_S,
                    what=f"{self.BACKEND_NAME} weight broadcast",
                    abort=self.abort_transport,
                )
            except RuntimeError as deadline:
                drain_error = deadline
        self._raise_if_server_failed(
            error,
            declared,
            hint=(
                " The engine reports a partially updated model after such a failure — restart "
                "the server before serving again."
            ),
        )
        if drain_error is not None:
            raise drain_error from error
        raise error

    def _send_tail(self, tail: list[tuple[str, torch.Tensor]]) -> None:
        if tail:
            self._transmit_chunk(tail, final=True)
        else:
            self.flush_engine_cache()
        self.release_send_buffers()

    def flush_engine_cache(self) -> None:
        """With no final chunk to ride on, the flush is requested directly, or the new weights serve a stale cache."""
        self.request("POST", "/flush_cache", retry=True, timeout=POST_TIMEOUT_S)

    def verify_group_membership(self) -> None:
        """An empty declare is a no-op on an engine that holds the group and an error on one that does not, which a
        real declare would turn into peers parked in a receive nobody can free."""
        try:
            self._start_collective_rpc(
                "/update_weights_from_distributed",
                timeout=POST_TIMEOUT_S,
                json={"names": [], "dtypes": [], "shapes": [], "group_name": self.group_name, "flush_cache": False},
            ).wait(timeout=POST_TIMEOUT_S)
        except requests.exceptions.HTTPError as e:
            raise GroupMembershipLost(
                f"SGLang server {self.base_url} holds no weight-update group {self.group_name!r} ({e}); it restarted "
                f"since the group formed."
            ) from e

    def reject_group_membership(self) -> None:
        """``pp_size`` is read by no rank formula, so a pipeline-parallel engine's stages would all take
        ``rank_offset + tp_rank`` and collide."""
        pp_size = int(self._server_info().get("pp_size", 1) or 1)
        if pp_size > 1:
            raise RuntimeError(
                f"SGLang server {self.base_url} runs --pp-size {pp_size}: its workers rank as rank_offset + tp_rank, "
                f"so the stages collide in a group shared with other servers. Serve with --pp-size 1, or set "
                f"rollout_sync_topology: per_server."
            )

    def score_completion_logprobs(
        self, prompt_ids: list[int], completion_ids: list[int], timeout: float = RESCORE_TIMEOUT_S
    ) -> list[float]:
        """One prefill through ``/generate``. The window opens on the last prompt token and every id is checked, so
        an off-by-one is a miss, never a shifted reference."""
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

    def _still_starting(self) -> bool:
        """``/v1/models`` answers ahead of the warm-up generate, whose first forward can outlast the generation probe on
        a cold kernel cache. ``/health`` answers 503 at once until that generate ends and then generates itself, so a
        read timeout means the server started and something holds it, which the probe names."""
        try:
            response = self.request("GET", "/health", retry=False, timeout=HTTP_PROBE_TIMEOUT_S, check=False)
        except requests.exceptions.Timeout:
            return False
        return response.status_code == 503

    def _engine_answers(self) -> bool:
        try:
            response = self.request("GET", "/health", retry=False, timeout=_ENGINE_ANSWER_TIMEOUT_S, check=False)
        except requests.exceptions.RequestException:
            return False
        # A 503 is a scheduler step past the server's own 20 s window: alive, and the destroy's join bounds a wedged one.
        return response.status_code in (200, 503)

    def destroy_remote_group(self):
        """Best-effort; never masks the caller's failure."""
        try:
            self.request(
                "POST",
                "/destroy_weights_update_group",
                retry=False,
                timeout=CLEANUP_TIMEOUT_S,
                json={"group_name": self.group_name},
            )
        except Exception as e:
            logger.debug(f"Could not clear weight-update group {self.group_name!r} on {self.base_url}: {e}")

    def _upload_to_arena(self, params: list[torch.Tensor], pool_stream: torch.cuda.Stream) -> Iterator[torch.Tensor]:
        """Yield each tensor's arena view as its copy is issued. One arena, since a block freed mid-chunk carries a
        pending NCCL event the next upload would wait on inside the allocator. Reuse across chunks is safe by stream
        order, as uploads and sends share ``_send_stream``."""
        self._settle_inflight(keep=_INFLIGHT_CHUNKS - 1)
        offsets: list[int] = []
        total = 0
        for param in params:
            offsets.append(total)
            # 256-byte aligned, a multiple of every dtype's size, so a uint8 slice views as the tensor.
            total += (param.nbytes + 255) // 256 * 256
        if self._arena is None or self._arena.numel() < total:
            # Only once the previous sends retired, as they still read the old arena.
            self.release_send_buffers()
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
        while len(self._inflight) > keep:
            bounded_event_sync(
                self._inflight.popleft(),
                timeout_s=CHUNK_SYNC_TIMEOUT_S,
                what=f"{self.BACKEND_NAME} weight broadcast",
                abort=self.abort_transport,
            )

    def abort_transport(self) -> None:
        self._aborted = True
        abort_weight_update_group(self._transport[0])

    def release_send_buffers(self) -> None:
        """Held across syncs, the arena would pin the sync GPU."""
        self._settle_inflight(keep=0)
        self._arena = None
