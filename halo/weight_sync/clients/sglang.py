"""Weight sync to an SGLang server: a plain c10d group, one typed broadcast per tensor in declared order. The server
needs ``NCCL_CUMEM_ENABLE=1``; SGLang otherwise turns cuMem off and the first cross-container import fails."""

import logging
import re
from collections import deque
from collections.abc import Callable
from functools import partial
from typing import NoReturn
from urllib.parse import urlparse

import requests
import torch
import torch.distributed as dist

from halo.models.moe.family_specs import STEP3P7
from halo.weight_sync.chunking import describe_chunk
from halo.weight_sync.clients.base import (
    CLEANUP_TIMEOUT_S,
    GROUP_FORMATION_TIMEOUT_S,
    HTTP_PROBE_TIMEOUT_S,
    RESCORE_TIMEOUT_S,
    ServedBase,
    ServerFlags,
    WeightSyncClientBase,
    pick_launch_facts,
    unservable_bailing_model_types,
)
from halo.weight_sync.clients.server_calls import AsyncCall, announced_wait, wait_announced
from halo.weight_sync.staging import ArenaChunkStager, StagedChunk
from halo.weight_sync.transport.bounded_waits import CHUNK_SYNC_TIMEOUT_S, bounded_event_sync, bounded_stream_sync
from halo.weight_sync.transport.c10d_group import (
    abort_weight_update_group,
    build_weight_update_group,
    release_weight_update_group,
)
from halo.weight_sync.transport.packed_broadcast import hold_for_stream
from halo.weight_sync.transport.rendezvous import RendezvousListener

logger = logging.getLogger(__name__)

# Chunks whose sends the host has not settled yet, so a declaration never waits on the previous chunk's tail.
_INFLIGHT_CHUNKS = 2
# Past the 20 s the server's /health gives a scheduler step before answering 503 itself, so a busy engine answers.
_ENGINE_ANSWER_TIMEOUT_S = 30.0
# A load or unload waits for the scheduler between batches and, for a load, for the adapter's host-to-device copy.
_ADAPTER_UPDATE_TIMEOUT_S = 120.0


class SGLangWeightSyncClient(WeightSyncClientBase):
    BACKEND_KEY = "sglang"
    BACKEND_NAME = "SGLang"
    # /health generates through the scheduler, so a pause a dead trainer left would read as an unreachable server.
    LIVENESS_ROUTE = "/v1/models"
    ENGINE_RELEASE = "SGLang 0.5.22"
    GROUP_HOST_ENV = "SGLANG_GROUP_HOST"
    PROBES_GROUP_MEMBERSHIP = True
    CHUNK_STAGER = ArenaChunkStager
    # With no final chunk to ride on, the flush is requested directly, or the new weights serve a stale cache.
    CACHE_FLUSH_ENDPOINT = "/flush_cache"
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
    ADAPTER_SYNC_REFUSAL = None
    # The adapted module names the load validates against: the attention and dense-MLP projections. Qwen3.5's
    # linear-attention ``in_proj_ba`` is not adaptable, and the stacked ``in_proj_qkvz`` has no counterpart in the
    # trainer's split names.
    ADAPTER_SERVABLE_MODULES = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
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
    LAUNCH_FACTS = {
        name: (name,)
        for name in (
            "tp_size",
            "dp_size",
            "attn_dp_size",
            "pp_size",
            "ep_size",
            "max_running_requests",
            "chunked_prefill_size",
            "max_prefill_tokens",
            "attention_backend",
            "prefill_attention_backend",
            "decode_attention_backend",
            "disable_cuda_graph",
            "cuda_graph_config",
            "speculative_algorithm",
            "enable_lora",
            "max_lora_rank",
            "lora_target_modules",
            "max_loaded_loras",
            "max_loras_per_batch",
        )
    }
    _INIT_LOG = "SGLang NCCL init: engine_ws={engine_ws}, master={master}:{port}, listening on {bind}"
    _READY_LOG = "SGLang NCCL weight transfer initialized"
    FORMATION_STEP = "weight-update group formation"
    _FORMATION_FAILURE = (
        "SGLang weight-update group formation failed (advertised master {master}:{port}; server {url}). If the "
        "SGLang node cannot reach that address, set {env} to an interface routable from it."
    )

    def __init__(self, *args, group_name: str | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        # c10d registers group names process-globally, and keying on group_port 0 would collide auto-port clients.
        self._own_group_name = f"weight_update_group_{urlparse(self.base_url).netloc}"
        self.group_name = group_name or self._own_group_name
        self._joined_names: set[str] = set()
        # Names the adapter a score request reads through, when the run serves its policy as one.
        self.adapter_source: Callable[[], str | None] | None = None
        self._engine_ranks: int | None = None

    def read_server_flags(self) -> ServerFlags:
        return self._flags_in(self._server_info())

    @classmethod
    def _flags_in(cls, info: dict) -> ServerFlags:
        return ServerFlags(
            toolkit_image=bool(info.get("halo_patches")),
            moe_backend=info.get("moe_runner_backend"),
            # A server with no parser reports null; a reply without the key says nothing.
            reasoning_parser=(info["reasoning_parser"] or "") if "reasoning_parser" in info else None,
            tool_call_parser=(info["tool_call_parser"] or "") if "tool_call_parser" in info else None,
            captures_routing=info.get("enable_return_routed_experts"),
            torch_compile=info.get("enable_torch_compile"),
            dtype=info.get("dtype"),
            quantization=(info["quantization"] or "") if "quantization" in info else None,
            pipeline_stages=int(info.get("pp_size") or 1),
            extras=pick_launch_facts(info, cls.LAUNCH_FACTS),
        )

    @property
    def layout_read_bound_s(self) -> float:
        """Each attempt of the retried ``/server_info`` read waits ``HTTP_PROBE_TIMEOUT_S``, and the backoff between
        attempts doubles from the policy's factor."""
        retry = self._session.get_adapter(self.base_url).max_retries
        attempts = retry.total + 1
        return attempts * HTTP_PROBE_TIMEOUT_S + sum(retry.backoff_factor * 2**i for i in range(attempts - 1))

    def _server_info(self) -> dict:
        return self.request("GET", "/server_info", retry=True, timeout=HTTP_PROBE_TIMEOUT_S).json()

    def _world_size_in(self, info: dict) -> int:
        """Workers join as ``rank_offset + tp_rank``; under ``--attn-dp-size`` ``tp_rank`` spans every GPU."""
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

    def read_engine_layout(self, *, require: bool) -> tuple[int, int]:
        info = self._server_info()
        flags = self._flags_in(info)
        if require:
            self._require_toolkit_image(flags)
            self.probe_generation()
        return self._world_size_in(info), flags.pipeline_stages

    def request_engine_join(
        self, master_address: str, port: int, world_size: int, rank_offset: int, group_name: str | None
    ) -> AsyncCall:
        if group_name is not None:
            self.group_name = group_name
        # The engine refuses a second join under a registered name, so a dead trainer's entry is cleared first, and a
        # second begin under its open session. Unchecked, as with no session open the close answers 400. The
        # per-server name is cleared too on the first join (a restart may switch topology), but later only if this
        # client formed it: on a server still parked in an earlier destroy, asking for a group that was never
        # formed here is answered only when that destroy ends.
        names = [self.group_name]
        if not self._joined_names or self._own_group_name in self._joined_names:
            names.insert(0, self._own_group_name)
        for name in dict.fromkeys(names):
            self._destroy_group_named(name)
        self._joined_names.add(self.group_name)
        try:
            with announced_wait(self.base_url, "its session close before it joins a group", CLEANUP_TIMEOUT_S):
                self.request(
                    "POST",
                    self.SESSION_CLOSE_ENDPOINT,
                    retry=False,
                    timeout=CLEANUP_TIMEOUT_S,
                    check=False,
                    **self.SESSION_REQUEST,
                )
        except requests.exceptions.Timeout as e:
            raise RuntimeError(
                f"SGLang server {self.base_url} did not answer a session close within {CLEANUP_TIMEOUT_S}s, so it "
                f"cannot join a weight-update group. Its scheduler is likely parked in the destroy of an earlier "
                f"group, which only a restart frees."
            ) from e
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
        self._rendezvous = listener
        return partial(
            build_weight_update_group,
            master_address=master_address,
            listener=listener,
            world_size=world_size,
            device=self.sync_device,
            group_name=self.group_name,
            store_timeout_s=GROUP_FORMATION_TIMEOUT_S,
            group_timeout_s=CHUNK_SYNC_TIMEOUT_S,
        )

    def _release_transport(self, *, at_exit: bool, peer_gone: bool = False) -> None:
        # A group whose abort is still running on its thread is left to it: releasing would race that thread.
        group = None if self._transport is None or self._abort_parked else self._transport[0]
        if peer_gone:
            # No destroy, health probe or remote release: each waits out a bound on a peer that will never answer.
            if group is not None:
                # An abort that dials the dead peer is bounded; the group is dropped either way.
                aborting = AsyncCall(self.abort_transport, name="weight-update abort")
                if aborting.join(CLEANUP_TIMEOUT_S).is_alive():
                    logger.error(f"Aborting the weight-update group of {self.base_url} is still blocked; dropping it.")
                release_weight_update_group(group, destroy=False)
            self._free_transport_leftovers()
            return
        # Interpreter exit skips the local NCCL destroy, and an aborted group is already gone from torch.
        destroy = not at_exit and not self._aborted
        # Ahead of the remote destroy: /health generates through the scheduler that destroy occupies, so asked during
        # it a healthy engine reads as dead, and aborting here strands its half of the destroy.
        answers = not (destroy and group is not None) or self._engine_answers()
        # Runs during the local destroy, since under cuMem each side's finalize waits for the other, and even with
        # no local group, or the server refuses every later join. A group that runs it for every server does so itself.
        remote = None if self.group_releases_remote_half else AsyncCall(self.destroy_remote_group)
        if not answers:
            logger.warning(f"SGLang server {self.base_url} does not answer, so its weight-update group is aborted.")
            self.abort_transport()
            destroy = False
        # An engine that never enters its half parks the destroy, with the failure that brought us here unraised.
        local = AsyncCall(partial(release_weight_update_group, group, destroy=destroy), name="weight-update destroy")
        wait_announced(
            local, self.base_url, "the trainer's half of its weight-update group to be released", CLEANUP_TIMEOUT_S
        )
        if local.is_alive():
            self.abort_transport()
        if remote is not None:  # the destroy request announces its own expiry
            remote.join(CLEANUP_TIMEOUT_S)
        self._free_transport_leftovers()

    def _free_transport_leftovers(self) -> None:
        # A destroy or abort left running holds the group, and through it the store, so the port is stopped directly.
        if self._rendezvous is not None:
            self._rendezvous.free_port()
            self._rendezvous = None
        self.release_chunk_staging()
        self._inflight.clear()
        self._send_stream = None

    def _reset_transport_state(self) -> None:
        # Set once a drain deadline aborted the group, on which a destroy raises and a broadcast hangs.
        self._aborted = False
        self._abort_parked = False
        self._rendezvous: RendezvousListener | None = None
        # Events recorded after the latest chunks' broadcasts, settled _INFLIGHT_CHUNKS later.
        self._inflight: deque[torch.cuda.Event] = deque()
        # On the caller's stream a synchronous broadcast would order the next copies and the trainer's work behind it.
        self._send_stream: torch.cuda.Stream | None = None

    def verify_transport(self) -> None:
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

    def send_staged(self, staged: StagedChunk) -> None:
        """The engine replies on queueing its receives; the sends settle two chunks later, or on failure."""
        self._settle_inflight(keep=_INFLIGHT_CHUNKS - 1)
        if self._send_stream is None:
            self._send_stream = torch.cuda.Stream(device=self.sync_device)
        with torch.cuda.stream(self._send_stream):
            if staged.ready is not None:
                self._send_stream.wait_event(staged.ready)
            for tensor in staged.wire:
                hold_for_stream(tensor, self._send_stream)
                # Synchronous at stream level only; an async_op work per tensor would flood the watchdog.
                dist.broadcast(tensor, src=0, group=self._transport[0])
            event = torch.cuda.Event()
            event.record(self._send_stream)
        staged.add_reader(event)
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

    @classmethod
    def adapter_request_model(cls, base_model: str, adapter: str) -> str:
        """The ``model`` a request names to be served through ``adapter``: SGLang reads everything after the first colon."""
        return f"{base_model}:{adapter}"

    def load_adapter(self, name: str, config: dict, payload: str) -> None:
        """Register ``payload`` (:func:`~halo.weight_sync.adapter_payload.serialize_adapter_tensors`) as the adapter
        ``name``. The route takes one copy per tensor-parallel rank, each rank deserializing its own."""
        if self._engine_ranks is None:
            self._engine_ranks = self.read_engine_layout(require=False)[0]
        body = {"lora_name": name, "config_dict": config, "serialized_named_tensors": [payload] * self._engine_ranks}
        response = self.request(
            "POST",
            "/load_lora_adapter_from_tensors",
            retry=False,
            timeout=_ADAPTER_UPDATE_TIMEOUT_S,
            check=False,
            json=body,
        )
        self._require_adapter_update(response, f"load adapter {name!r}")

    def unload_adapter(self, name: str) -> None:
        """Drop the adapter ``name``; the engine first waits for the requests still running under it."""
        response = self.request(
            "POST",
            "/unload_lora_adapter",
            retry=False,
            timeout=_ADAPTER_UPDATE_TIMEOUT_S,
            check=False,
            json={"lora_name": name},
        )
        self._require_adapter_update(response, f"unload adapter {name!r}")

    @classmethod
    def is_adapter_missing(cls, error: str, adapter: str) -> bool:
        """Matches the engine's "Got LoRA adapter that has never been loaded: <name>" 400, whose body is raw JSON."""
        return re.search(rf"never been loaded: {re.escape(adapter)}(?:\\n|\n|$)", error) is not None

    def served_base(self) -> ServedBase:
        info = self._server_info()
        return ServedBase(
            model_id=info.get("served_model_name"),
            path=(info.get("model_path") or "").rstrip("/") or None,
            revision=info.get("revision") or None,
        )

    def loaded_adapters(self) -> list[str]:
        """The adapters registered on the engine: the model cards that name a parent, after the base's own."""
        return [str(card["id"]) for card in self._served_model_cards() if card.get("parent")]

    def _require_adapter_update(self, response: requests.Response, what: str) -> None:
        try:
            reply = response.json()
        except ValueError:
            reply = {}
        if response.ok and reply.get("success"):
            return
        raise RuntimeError(
            f"SGLang {self.base_url} could not {what}: HTTP {response.status_code}, "
            f"{reply.get('error_message') or response.text[:300]}"
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
        adapter = self.adapter_source() if self.adapter_source is not None else None
        if adapter is not None:
            body["lora_path"] = adapter
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
            with announced_wait(self.base_url, "/health before the group's destroy", _ENGINE_ANSWER_TIMEOUT_S):
                response = self.request("GET", "/health", retry=False, timeout=_ENGINE_ANSWER_TIMEOUT_S, check=False)
        except requests.exceptions.RequestException:
            return False
        # A 503 is a scheduler step past the server's own 20 s window: alive, and the destroy's join bounds a wedged one.
        return response.status_code in (200, 503)

    def destroy_remote_group(self) -> None:
        self._destroy_group_named(self.group_name)

    def _destroy_group_named(self, group_name: str) -> None:
        """Best-effort; never masks the caller's failure."""
        try:
            with announced_wait(
                self.base_url, f"weight-update group {group_name!r} to be destroyed", CLEANUP_TIMEOUT_S
            ):
                self.request(
                    "POST",
                    "/destroy_weights_update_group",
                    retry=False,
                    timeout=CLEANUP_TIMEOUT_S,
                    json={"group_name": group_name},
                )
        except Exception as e:
            logger.debug(f"Could not clear weight-update group {group_name!r} on {self.base_url}: {e}")

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
        aborting = AsyncCall(partial(abort_weight_update_group, self._transport[0]), name="weight-update abort")
        wait_announced(
            aborting, self.base_url, "the trainer's half of its weight-update group to abort", CLEANUP_TIMEOUT_S
        )
        if aborting.is_alive():
            self._abort_parked = True

    def release_send_buffers(self) -> None:
        """Held across syncs, the arena would pin the sync GPU."""
        self._settle_inflight(keep=0)
        self.release_chunk_staging()
