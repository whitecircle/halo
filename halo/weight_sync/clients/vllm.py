"""Weight sync to a vLLM server: the trainer joins as rank 0 over the vendored pynccl bootstrap and streams
packed broadcasts into a layerwise reload. The vllm package is never imported."""

import logging
import re
from collections.abc import Callable
from functools import partial

import requests
import torch

from halo.weight_sync.chunking import describe_chunk, resolve_sync_device
from halo.weight_sync.clients.base import (
    CLEANUP_TIMEOUT_S,
    GROUP_FORMATION_TIMEOUT_S,
    HTTP_PROBE_TIMEOUT_S,
    RESCORE_TIMEOUT_S,
    AsyncCall,
    WeightSyncClientBase,
    unservable_bailing_model_types,
)
from halo.weight_sync.transport.bounded_waits import CHUNK_SYNC_TIMEOUT_S
from halo.weight_sync.transport.packed_broadcast import (
    PACKED_BUFFER_BYTES,
    PACKED_NUM_BUFFERS,
    packed_broadcast_producer,
)
from halo.weight_sync.transport.pynccl import PyNcclCommunicator, UniqueIdStore
from halo.weight_sync.transport.rendezvous import RendezvousListener

logger = logging.getLogger(__name__)

# A merely busy scheduler still answers a 1-token probe in time; a wedged one is reported, not waited out.
_GENERATION_PROBE_TIMEOUT_S = 60.0
# The oldest engine whose weight-transfer wire this client speaks: the packed geometry rides
# /init_weight_transfer_engine, and an update that carries it is refused.
_MIN_SERVER_VERSION = (0, 32, 0)


class VLLMWeightSyncClient(WeightSyncClientBase):
    """Weight-sync client for a vLLM server."""

    BACKEND_KEY = "vllm"
    BACKEND_NAME = "vLLM"
    ENGINE_RELEASE = "vLLM 0.32.0"
    GROUP_HOST_ENV = "VLLM_GROUP_HOST"
    PAUSE_ENDPOINT = "/pause"
    # mode=keep resumes in-flight generations under the new weights (the staleness the IS ratio corrects);
    # vLLM's default abort returns each as a fragment with an ordinary stop reason.
    PAUSE_REQUEST = {"params": {"mode": "keep"}}
    RESUME_ENDPOINT = "/resume"
    # The layerwise reload phase.
    SESSION_OPEN_ENDPOINT = "/start_weight_update"
    SESSION_CLOSE_ENDPOINT = "/finish_weight_update"
    UNSERVABLE_MODEL_TYPES = {
        "zaya": f"{ENGINE_RELEASE} ships no Zaya implementation (agent-docs/models/zaya.md)",
        "mistral4": (
            f"{ENGINE_RELEASE} registers no Mistral4ForCausalLM and maps no mistral4 model_type, so the "
            "composite loader has no class for the text tower a toolkit export writes "
            "(agent-docs/models/mistral4.md#serving)"
        ),
        "deepseek_v4": (
            f"{ENGINE_RELEASE}'s DeepSeek-V4 loader targets the fp8/fp4-packed release layout, which no gather can emit"
        ),
        **unservable_bailing_model_types(ENGINE_RELEASE),
    }
    # The Laguna loader registers the bias under an internal name and skips the exported hub key.
    DROPPED_BALANCING_BIAS_MODEL_TYPES = frozenset({"laguna"})
    INTERRUPTED_UPDATE_STATE = (
        "a model that is part old weights and part new, with any layer that straddled the last chunk "
        "boundary materialized from UNINITIALIZED storage by the layerwise reload"
    )
    _INIT_LOG = "NCCL init: inference_ws={engine_ws}, master={master}:{port}, listening on {bind}"
    _READY_LOG = "NCCL weight transfer initialized"
    _FORMATION_STEP = "NCCL group formation"
    _FORMATION_FAILURE = (
        "Weight-transfer group formation failed (advertised master {master}:{port}; vLLM server {url}). If the "
        "vLLM node cannot reach that address, set {env} to an interface routable from the vLLM server (and "
        "NCCL_SOCKET_IFNAME for the data plane on multi-homed hosts)."
    )

    def _fetch_engine_world_size(self) -> int:
        # An engine off this wire, paused or generation-wedged is reported before the group forms.
        self._check_server_version()
        self._probe_generation()
        return self.request("GET", "/get_world_size", retry=False, timeout=HTTP_PROBE_TIMEOUT_S).json()["world_size"]

    def _request_engine_join(self, master_address: str, port: int, world_size: int) -> AsyncCall:
        # Told before the store is built: the master blocks until every worker joined.
        return self._start_collective_rpc(
            "/init_weight_transfer_engine",
            timeout=GROUP_FORMATION_TIMEOUT_S,
            json={
                "init_info": {
                    "master_address": master_address,
                    "master_port": port,
                    "rank_offset": 1,
                    "world_size": world_size,
                    # Read once per connection: the engine derives every chunk boundary from it, and
                    # unset it receives tensor by tensor.
                    "packed": True,
                    "packed_buffer_size_bytes": PACKED_BUFFER_BYTES,
                    "packed_num_buffers": PACKED_NUM_BUFFERS,
                }
            },
        )

    def _prepare_local_formation(
        self, master_address: str, listener: RendezvousListener, world_size: int
    ) -> Callable[[], PyNcclCommunicator]:
        self._unique_id_store = UniqueIdStore.create(host=master_address, listener=listener, world_size=world_size)
        # ncclCommInitRank blocks with no deadline, so it runs beside the engine's join, not ahead of it.
        return partial(PyNcclCommunicator, self._unique_id_store, device=self._sync_device)

    def _release_transport(self, *, at_exit: bool) -> None:
        if self._transport is not None:
            # Aborted, not dropped: without a __del__ its ncclComm and pinned memory outlive every retired client.
            self._transport.abort()
        self._packed_streams = None
        if self._unique_id_store is not None:
            # Frees group_port; a comm thread parked in ncclCommInitRank still holds the store.
            self._unique_id_store.close()
            self._unique_id_store = None

    def _reset_transport_state(self) -> None:
        # Holds the listener the store took over, until the release frees it.
        self._unique_id_store: UniqueIdStore | None = None
        # The producer's pack streams, created once on the communicator's device; flushes are serialized.
        self._packed_streams: list[torch.cuda.Stream] | None = None

    def _start_engine_update(self) -> None:
        """Pause the engine and open its layerwise reload phase, unwinding both on failure."""
        self._require_communicator()
        try:
            self._quiesce()
            self._open_session()
        except Exception:
            self._close_session()
            self._lift_pause(CLEANUP_TIMEOUT_S, context="after sync")
            raise

    def _broadcast_chunk(self, chunk: list[tuple[str, torch.Tensor]], final: bool) -> None:
        """Declare one chunk over ``/update_weights``, then broadcast it packed into the open phase. ``final`` is
        unused: this engine closes with its own ``/finish_weight_update``."""
        names, dtype_names, shapes = describe_chunk(chunk)

        server_call: AsyncCall | None = None
        try:
            # vLLM replies once it has received the chunk, so the RPC is bounded by the chunk's own deadline.
            server_call = self._start_collective_rpc(
                "/update_weights",
                timeout=CHUNK_SYNC_TIMEOUT_S,
                json={"update_info": {"names": names, "dtype_names": dtype_names, "shapes": shapes}},
            )

            communicator = self._require_communicator()
            device = communicator.device
            if self._packed_streams is None and torch.device(device).type == "cuda":
                with torch.cuda.device(device):
                    self._packed_streams = [torch.cuda.Stream() for _ in range(PACKED_NUM_BUFFERS)]
            # The producer copies nothing: a copy made on a pack stream would stay in that stream's pool.
            misplaced = [name for name, param in chunk if param.device != device]
            if misplaced:
                raise RuntimeError(
                    f"Weight-sync snapshots {misplaced} are not on the sync device {device}: every snapshot is "
                    f"taken there once init_communicator ran."
                )
            packed_broadcast_producer(iterator=iter(chunk), group=communicator, streams=self._packed_streams)

            server_call.wait(timeout=CHUNK_SYNC_TIMEOUT_S)
        except Exception as e:
            self._raise_if_server_failed(e, server_call)
            raise
        logger.debug(f"Streamed {len(chunk)} params to vLLM")

    def _finish_engine_update(self, tail: list[tuple[str, torch.Tensor]]) -> None:
        """Send the final chunk and close the layerwise reload; the close runs on every path."""
        try:
            self._transmit_chunk(tail, final=True)
            self._end_session()
        finally:
            self._close_session()

    def score_completion_logprobs(
        self, prompt_ids: list[int], completion_ids: list[int], timeout: float = RESCORE_TIMEOUT_S
    ) -> list[float]:
        """One prefill through the completions ``prompt_logprobs`` echo: the raw (pre-sampler) distribution."""
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

    def init_communicator(self, device: torch.device | str | int = 0) -> None:
        # Recorded before the liveness probe that opens the engine's world-size read.
        self._sync_device = resolve_sync_device(device)
        super().init_communicator(device)

    def _check_server_version(self) -> None:
        """Refuse an engine older than the weight-transfer wire this client speaks (``GET /version``)."""
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

    def _probe_generation(self):
        """Fail fast when ``/health`` is up but generation is wedged: a single-shot 1-token completion that only a
        hang fails. A paused engine queues it instead, so the pause is asked first."""
        try:
            resp = self.request("GET", "/is_paused", retry=False, timeout=HTTP_PROBE_TIMEOUT_S, check=False)
            paused = resp.status_code == 200 and bool(resp.json()["is_paused"])
        except (requests.exceptions.RequestException, ValueError, KeyError) as e:
            logger.debug(f"Could not read pause state from {self.base_url}: {e}")
            paused = False
        if paused:
            raise RuntimeError(
                f"vLLM server {self.base_url} is PAUSED for a weight update (GET /is_paused) — generation "
                f"requests would queue instead of running. A trainer died mid weight sync without lifting "
                f"its /pause. Lift it (POST {self.base_url}{self.RESUME_ENDPOINT}) before reconnecting — unless that "
                f"sync had already streamed part of the model, in which case the server holds a partly "
                f"written model and has to be RESTARTED instead (its log names the last update)."
            )
        try:
            model_id = self._served_model_cards()[0]["id"]
        except Exception as e:
            logger.warning(f"Generation probe: could not resolve served model ({e}); skipping probe")
            return
        try:
            self.request(
                "POST",
                "/v1/completions",
                retry=False,
                timeout=_GENERATION_PROBE_TIMEOUT_S,
                check=False,
                json={"model": model_id, "prompt": "ping", "max_tokens": 1, "temperature": 0.0},
            )
        except requests.exceptions.Timeout as e:
            raise RuntimeError(
                f"vLLM server {self.base_url} answers /health but GENERATION is wedged (no response "
                f"in {_GENERATION_PROBE_TIMEOUT_S:.0f}s to a 1-token probe). A previous trainer most "
                f"likely died mid-run while attached to the weight-transfer engine (SIGKILL skips cleanup), leaving the "
                f"server's scheduler stuck. RESTART the vLLM server container to recover."
            ) from e
        except requests.exceptions.RequestException as e:
            # Responsive but rejected the probe (e.g. raw /v1/completions disabled): not wedged.
            logger.warning(f"Generation probe: server responded with an error (not wedged): {e}")

    def _require_communicator(self) -> PyNcclCommunicator:
        """The live communicator; raises when there is none or it was aborted."""
        if self._transport is None:
            raise RuntimeError("Call init_communicator() first")
        if self._transport.aborted:
            raise RuntimeError(
                f"The weight-sync NCCL communicator for {self.base_url} was aborted by a failed "
                f"collective and cannot be reused — reconnect the client before syncing again."
            )
        return self._transport
