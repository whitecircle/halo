"""Weight sync to a vLLM server: the trainer joins as rank 0 over the vendored pynccl bootstrap and streams
packed broadcasts into a layerwise reload. The vllm package is never imported."""

import logging
import re
from collections.abc import Callable
from functools import partial
from typing import NoReturn

import requests
import torch

from halo.weight_sync.chunking import describe_chunk
from halo.weight_sync.clients.base import (
    CLEANUP_TIMEOUT_S,
    GENERATION_PROBE_TIMEOUT_S,
    GROUP_FORMATION_TIMEOUT_S,
    HTTP_PROBE_TIMEOUT_S,
    RESCORE_TIMEOUT_S,
    ServerFlags,
    WeightSyncClientBase,
    pick_launch_facts,
    unservable_bailing_model_types,
)
from halo.weight_sync.clients.server_calls import AsyncCall, announced_wait
from halo.weight_sync.staging import PackedChunkStager, StagedChunk
from halo.weight_sync.transport.bounded_waits import CHUNK_SYNC_TIMEOUT_S
from halo.weight_sync.transport.packed_broadcast import PACKED_BUFFER_BYTES, PACKED_NUM_BUFFERS, broadcast_packed
from halo.weight_sync.transport.pynccl import PyNcclCommunicator, UniqueIdStore
from halo.weight_sync.transport.rendezvous import RendezvousListener

logger = logging.getLogger(__name__)

# The oldest wire this client speaks; the packed geometry rides /init_weight_transfer_engine, never an update.
_MIN_SERVER_VERSION = (0, 32, 0)
# The route's first call collects the host's environment, once.
_SERVER_INFO_TIMEOUT_S = 60.0
# docker/vllm/patches/vllm_image_identity_patch.py declares it; a test pins the two together.
_IDENTITY_KEY = "VLLM_HALO_PATCHES"
# Listed on /server_info as the server reads it; this client never reads its own environment for it.
_STRICT_TOOL_CALLING_KEY = "VLLM_ENFORCE_STRICT_TOOL_CALLING"
_TOKEN_ID_PREFIX = "token_id:"


class VLLMWeightSyncClient(WeightSyncClientBase):
    BACKEND_KEY = "vllm"
    BACKEND_NAME = "vLLM"
    ENGINE_RELEASE = "vLLM 0.32.0"
    GROUP_HOST_ENV = "VLLM_GROUP_HOST"
    REPORTS_INSTANCE_ID = True
    CHUNK_STAGER = PackedChunkStager
    PAUSE_ENDPOINT = "/pause"
    # mode=keep resumes in-flight generations under the new weights (the staleness the IS ratio corrects);
    # vLLM's default abort returns each as a fragment with an ordinary stop reason.
    PAUSE_REQUEST = {"params": {"mode": "keep"}}
    RESUME_ENDPOINT = "/resume"
    SESSION_OPEN_ENDPOINT = "/start_weight_update"
    SESSION_CLOSE_ENDPOINT = "/finish_weight_update"
    # Layerwise reload: a failed chunk also ends the engine's session, so no close can restore the model after it.
    SESSION_OPEN_UNLOADS_MODEL = True
    IMAGE_REMEDY = (
        "Serve from the toolkit's vLLM image (make build-vllm), and keep its PYTHONPATH, which loads the patches."
    )
    MOE_BACKEND_FLAG = "--moe-backend triton (VLLM_MOE_BACKEND)"
    REASONING_PARSER_FLAG = "--reasoning-parser (VLLM_REASONING_PARSER)"
    TOOL_PARSER_FLAG = "--enable-auto-tool-choice --tool-call-parser (VLLM_TOOL_PARSER)"
    CHAT_TEMPLATE_FLAG = "--chat-template (VLLM_CHAT_TEMPLATE)"
    ADAPTER_SYNC_REFUSAL = (
        f"{ENGINE_RELEASE} loads an adapter only from a directory both containers mount, and its decode with an "
        f"adapter served ran over 5% slower than the merged weights on the model measured"
    )
    UNSERVABLE_MODEL_TYPES = {
        "zaya": f"{ENGINE_RELEASE} ships no Zaya implementation (agent-docs/models/zaya.md)",
        "mistral4": (
            f"{ENGINE_RELEASE} registers no Mistral4ForCausalLM and maps no mistral4 model_type, so the "
            "composite loader has no class for the text tower a toolkit export writes "
            "(agent-docs/models/mistral4.md#serving)"
        ),
        "deepseek_v4": (
            f"{ENGINE_RELEASE}'s deepseek_v4 loader maps per-expert w1/w2/w3 names where the gather emits the fused pair"
        ),
        **unservable_bailing_model_types(ENGINE_RELEASE),
    }
    # The Laguna loader registers the bias under an internal name and skips the exported hub key.
    DROPPED_BALANCING_BIAS_MODEL_TYPES = frozenset({"laguna"})
    UNENFORCEABLE_THINKING_BUDGET_MODEL_TYPES = {
        "mistral4": (
            f"{ENGINE_RELEASE} serves it with its mistral_common tokenizer, which encodes the [THINK] and [/THINK] "
            "markers as plain text while the model writes them as control tokens"
        ),
    }
    INTERRUPTED_UPDATE_STATE = (
        "a model that is part old weights and part new, with any layer that straddled the last chunk "
        "boundary materialized from UNINITIALIZED storage by the layerwise reload"
    )
    LAUNCH_FACTS = {
        "tensor_parallel_size": ("parallel_config", "tensor_parallel_size"),
        "data_parallel_size": ("parallel_config", "data_parallel_size"),
        "pipeline_parallel_size": ("parallel_config", "pipeline_parallel_size"),
        "expert_parallel": ("parallel_config", "enable_expert_parallel"),
        "max_num_seqs": ("scheduler_config", "max_num_seqs"),
        "max_num_batched_tokens": ("scheduler_config", "max_num_batched_tokens"),
        "attention_backend": ("attention_config", "backend"),
        "compilation_mode": ("compilation_config", "mode"),
        "cudagraph_mode": ("compilation_config", "cudagraph_mode"),
    }
    _INIT_LOG = "NCCL init: inference_ws={engine_ws}, master={master}:{port}, listening on {bind}"
    _READY_LOG = "NCCL weight transfer initialized"
    FORMATION_STEP = "NCCL group formation"
    _FORMATION_FAILURE = (
        "Weight-transfer group formation failed (advertised master {master}:{port}; vLLM server {url}). If the "
        "vLLM node cannot reach that address, set {env} to an interface routable from the vLLM server (and "
        "NCCL_SOCKET_IFNAME for the data plane on multi-homed hosts)."
    )

    def read_server_flags(self) -> ServerFlags:
        info = self.request(
            "GET", "/server_info", retry=False, timeout=_SERVER_INFO_TIMEOUT_S, params={"config_format": "json"}
        ).json()
        env, config = info.get("vllm_env", {}), info.get("vllm_config")
        config = config if isinstance(config, dict) else {}
        model = config.get("model_config", {})
        return ServerFlags(
            toolkit_image=bool(env.get(_IDENTITY_KEY)),
            moe_backend=config.get("kernel_config", {}).get("moe_backend"),
            reasoning_parser=config.get("structured_outputs_config", {}).get("reasoning_parser"),
            strict_tool_calling=env.get(_STRICT_TOOL_CALLING_KEY),
            captures_routing=config.get("aux_output_config", {}).get("enable_return_routed_experts"),
            dtype=model.get("dtype"),
            quantization=(model["quantization"] or "") if "quantization" in model else None,
            instance_id=config.get("instance_id"),
            pipeline_stages=config.get("parallel_config", {}).get("pipeline_parallel_size"),
            extras=pick_launch_facts(config, self.LAUNCH_FACTS),
        )

    def reports_sampled_token_ids(self) -> bool | None:
        """``--return-tokens-as-token-ids`` shows on no reply field but the logprobs themselves."""
        reply = self.request(
            "POST",
            "/v1/chat/completions",
            retry=True,
            timeout=GENERATION_PROBE_TIMEOUT_S,
            json={
                "model": self.served_model_id(),
                "messages": [{"role": "user", "content": "ping"}],
                "max_tokens": 1,
                "logprobs": True,
            },
        ).json()
        return reply["choices"][0]["logprobs"]["content"][0]["token"].startswith(_TOKEN_ID_PREFIX)

    def accepts_tool_requests(self) -> bool | None:
        """Without a parser the renderer answers a tool-bearing request with a 400 naming it; ``tools`` needs no
        ``tool_choice``, which defaults to ``auto`` as on every rollout. Any other 400 is another fault's."""
        reply = self.request(
            "POST",
            "/v1/chat/completions",
            retry=True,
            timeout=GENERATION_PROBE_TIMEOUT_S,
            check=False,
            json={
                "model": self.served_model_id(),
                "messages": [{"role": "user", "content": "ping"}],
                "max_tokens": 1,
                "tools": [
                    {
                        "type": "function",
                        "function": {
                            "name": "probe",
                            "description": "Probe.",
                            "parameters": {"type": "object", "properties": {}},
                        },
                    }
                ],
            },
        )
        return not (reply.status_code == 400 and "--tool-call-parser" in reply.text)

    def read_engine_layout(self, *, require: bool) -> tuple[int, int]:
        # An engine off this wire, off this image, paused or generation-wedged is reported before the group forms.
        if require:
            self._check_server_version()
        try:
            flags = self.read_server_flags()
        except (requests.exceptions.RequestException, ValueError) as e:
            # As in the preflights, a report that cannot be read refuses nothing; a group needs the launch id, though.
            logger.warning(f"Could not read {self.base_url}/server_info, so its image goes unchecked: {e}")
            flags = None
        else:
            if require:
                self._require_toolkit_image(flags)
                self._group_instance = flags.instance_id
        if require:
            self.probe_generation()
        reply = self.request("GET", "/get_world_size", retry=False, timeout=HTTP_PROBE_TIMEOUT_S).json()
        return reply["world_size"], (flags.pipeline_stages if flags else None) or 1

    def request_engine_join(
        self, master_address: str, port: int, world_size: int, rank_offset: int, group_name: str | None
    ) -> AsyncCall:
        """The engine keeps one group, so ``group_name`` has nothing to name."""
        # Told before the store is built, as the master blocks until every worker joined.
        return self._start_collective_rpc(
            "/init_weight_transfer_engine",
            timeout=GROUP_FORMATION_TIMEOUT_S,
            json={
                "init_info": {
                    "master_address": master_address,
                    "master_port": port,
                    "rank_offset": rank_offset,
                    "world_size": world_size,
                    # Read once per connection; unset, the engine receives tensor by tensor.
                    "packed": True,
                    "packed_buffer_size_bytes": PACKED_BUFFER_BYTES,
                    "packed_num_buffers": PACKED_NUM_BUFFERS,
                }
            },
        )

    def prepare_local_formation(
        self, master_address: str, listener: RendezvousListener, world_size: int
    ) -> Callable[[], PyNcclCommunicator]:
        self._unique_id_store = UniqueIdStore.create(host=master_address, listener=listener, world_size=world_size)
        # ncclCommInitRank blocks with no deadline, so it runs beside the engine's join, not ahead of it.
        return partial(PyNcclCommunicator, self._unique_id_store, device=self._sync_device)

    def _release_transport(self, *, at_exit: bool, peer_gone: bool = False) -> None:
        if self._transport is not None:
            # With no __del__, a dropped communicator's ncclComm and pinned memory outlive every retired client.
            self._transport.abort()
        self._packed_streams = None
        if self._unique_id_store is not None:
            # Frees group_port; a comm thread parked in ncclCommInitRank still holds the store.
            self._unique_id_store.close()
            self._unique_id_store = None

    def _reset_transport_state(self) -> None:
        self._unique_id_store: UniqueIdStore | None = None
        # Created once on the communicator's device; flushes are serialized.
        self._packed_streams: list[torch.cuda.Stream] | None = None

    def verify_transport(self) -> None:
        self._require_communicator()

    def destroy_remote_group(self) -> None:
        try:
            with announced_wait(self.base_url, "its weight-transfer engine to shut down", CLEANUP_TIMEOUT_S):
                self.request("POST", "/shutdown_weight_transfer_engine", retry=False, timeout=CLEANUP_TIMEOUT_S)
        except Exception as e:
            logger.debug(f"Could not shut down the weight-transfer engine of {self.base_url}: {e}")

    def abort_transport(self) -> None:
        if self._transport is not None:
            self._transport.abort()

    def prepare_chunk_send(self, chunk: list[tuple[str, torch.Tensor]]) -> None:
        """Everything that can refuse the chunk runs first, as the RPC holds the engine in its collective."""
        communicator = self._require_communicator()
        device = communicator.device
        if self._packed_streams is None and torch.device(device).type == "cuda":
            with torch.cuda.device(device):
                self._packed_streams = [torch.cuda.Stream() for _ in range(PACKED_NUM_BUFFERS)]
        # A copy made on a pack stream would stay in that stream's pool, so the producer copies nothing.
        misplaced = [name for name, param in chunk if param.device != device]
        if misplaced:
            raise RuntimeError(
                f"Weight-sync snapshots {misplaced} are not on the sync device {device}: every snapshot is "
                f"taken there once init_communicator ran."
            )

    def declare_chunk(self, chunk: list[tuple[str, torch.Tensor]], final: bool) -> AsyncCall:
        """``final`` is unused, as this engine closes with its own ``/finish_weight_update``. vLLM replies once it has
        received the chunk, so the RPC is bounded by the chunk's own deadline."""
        names, dtype_names, shapes = describe_chunk(chunk)
        return self._start_collective_rpc(
            "/update_weights",
            timeout=CHUNK_SYNC_TIMEOUT_S,
            json={"update_info": {"names": names, "dtype_names": dtype_names, "shapes": shapes}},
        )

    def send_staged(self, staged: StagedChunk) -> None:
        broadcast_packed(staged.wire, group=self._transport, streams=self._packed_streams, ready=staged.ready)
        logger.debug(f"Streamed {len(staged.wire)} packed buffers to vLLM")

    def _fail_chunk(self, error: Exception, declared: AsyncCall) -> NoReturn:
        self._raise_if_server_failed(error, declared)
        raise error

    def score_completion_logprobs(
        self, prompt_ids: list[int], completion_ids: list[int], timeout: float = RESCORE_TIMEOUT_S
    ) -> list[float]:
        """One prefill through the completions ``prompt_logprobs`` echo, the raw (pre-sampler) distribution."""
        body = {
            "model": self.served_model_id(),
            "prompt": list(prompt_ids) + list(completion_ids),
            "max_tokens": 1,
            "prompt_logprobs": 0,
            "temperature": 0.0,
        }
        resp = self.request("POST", "/v1/completions", retry=True, timeout=timeout, json=body)
        entries = resp.json()["choices"][0]["prompt_logprobs"]
        expected = len(prompt_ids) + len(completion_ids)
        if entries is None or len(entries) != expected:
            raise ValueError(
                f"{self.BACKEND_NAME} returned {None if entries is None else len(entries)} prompt log-probs "
                f"for a {expected}-token sequence"
            )
        # Keyed by the token id (as a string); the actual token is always present at prompt_logprobs=0.
        return [
            float(entry[str(tok)]["logprob"])
            for tok, entry in zip(completion_ids, entries[len(prompt_ids) :], strict=True)
        ]

    def _check_server_version(self) -> None:
        body = self.request("GET", "/version", retry=False, timeout=HTTP_PROBE_TIMEOUT_S).json()
        reported = body.get("version") if isinstance(body, dict) else None
        match = re.match(r"(\d+)\.(\d+)\.(\d+)", str(reported)) if reported is not None else None
        if match is None or tuple(int(part) for part in match.groups()) < _MIN_SERVER_VERSION:
            minimum = ".".join(map(str, _MIN_SERVER_VERSION))
            raise RuntimeError(
                f"vLLM server {self.base_url} runs vLLM {reported!r}; this client speaks the weight-transfer "
                f"protocol of vLLM >= {minimum}, which carries the packed geometry on "
                f"/init_weight_transfer_engine. Serve from the toolkit's vLLM image (make build-vllm)."
            )

    def _read_paused(self) -> bool:
        try:
            resp = self.request("GET", "/is_paused", retry=False, timeout=HTTP_PROBE_TIMEOUT_S, check=False)
            return resp.status_code == 200 and bool(resp.json()["is_paused"])
        except (requests.exceptions.RequestException, ValueError, KeyError) as e:
            logger.debug(f"Could not read pause state from {self.base_url}: {e}")
            return False

    def _require_communicator(self) -> PyNcclCommunicator:
        if self._transport is None:
            raise RuntimeError("Call init_communicator() first")
        if self._transport.aborted:
            raise RuntimeError(
                f"The weight-sync NCCL communicator for {self.base_url} was aborted by a failed "
                f"collective and cannot be reused — reconnect the client before syncing again."
            )
        return self._transport
