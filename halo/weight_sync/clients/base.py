"""The engine-agnostic half of a weight-sync client: HTTP control plane, group lifecycle and streamed update.
The trainer joins as rank 0 and streams the gathered parameters in byte-bounded chunks while the engine is quiesced."""

import atexit
import logging
import socket
import time
from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping
from types import MappingProxyType
from typing import Any, NamedTuple, NoReturn
from urllib.parse import urlparse

import requests
import torch
import urllib3
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from halo.env import env_positive_float
from halo.weight_sync.chunking import WEIGHT_SYNC_CHUNK_BYTES, resolve_sync_device, snapshot_to_sync_device
from halo.weight_sync.clients.server_calls import AsyncCall, announced_wait, wait_announced, wait_for_calls
from halo.weight_sync.staging import ChunkStager, StagedChunk
from halo.weight_sync.transport.bounded_waits import CHUNK_SYNC_TIMEOUT_S
from halo.weight_sync.transport.rendezvous import RendezvousListener, open_group_rendezvous
from halo.weight_sync.update_stream import UpdateStream

logger = logging.getLogger(__name__)

# Per attempt; the calls that block on the engine have their own deadlines.
POST_TIMEOUT_S = 60
HTTP_PROBE_TIMEOUT_S = 10
# A merely busy scheduler still answers a 1-token probe in time; a wedged one is reported, not waited out.
GENERATION_PROBE_TIMEOUT_S = 60.0
# One group's formation: the engine's join and the trainer's rendezvous. Read once at import, like the chunk deadline.
GROUP_FORMATION_TIMEOUT_S = env_positive_float("HALO_WEIGHT_SYNC_FORMATION_TIMEOUT_SECONDS", 120.0)
# The trainer is rank 0.
FIRST_ENGINE_RANK = 1
# The close waits for the engine to finish applying the last chunk.
SESSION_CLOSE_TIMEOUT_S = 600
# Cleanup runs on a failing path, where a long block only delays the real raise.
CLEANUP_TIMEOUT_S = 30
# One prefill of a training row, up to 250k tokens on a server that is also generating.
RESCORE_TIMEOUT_S = 600.0
_RESUME_RETRY_BACKOFF_S = (1.0, 2.0, 4.0)
# Lets an in-flight server call record its error once the local side failed.
SERVER_ERROR_GRACE_S = 5.0
_DEFAULT_RETRY = Retry(
    total=5,
    connect=5,
    read=5,
    status=3,
    status_forcelist=[500, 502, 503],
    backoff_factor=2,
    allowed_methods=["POST", "GET"],
)


class GroupMembershipLost(RuntimeError):
    """The engine no longer holds the weight-sync group this client formed (a restart), so the group must re-form."""


class ServerFlags(NamedTuple):
    """What a server reports on ``/server_info``; ``None`` where the reply carries no such field, so a renamed one reads
    as unread, not as wrong."""

    toolkit_image: bool
    moe_backend: str | None = None
    # Empty when the server runs without one.
    reasoning_parser: str | None = None
    tool_call_parser: str | None = None
    strict_tool_calling: bool | None = None
    captures_routing: bool | None = None
    torch_compile: bool | None = None
    dtype: str | None = None
    # Empty when the server runs unquantized.
    quantization: str | None = None
    instance_id: str | None = None
    # Pipeline stages of the engine; a group ranks workers by tensor-parallel rank alone, so stages would collide.
    pipeline_stages: int | None = None
    # The engine's own launch facts (parallel sizes, batch caps, attention and graph modes), logged and never compared.
    extras: Mapping[str, object] = MappingProxyType({})


class ServedBase(NamedTuple):
    """The frozen base an engine serves, as its launch named it; ``None`` where unreported."""

    model_id: str | None
    path: str | None
    revision: str | None


def pick_launch_facts(source: Mapping[str, Any], facts: Mapping[str, tuple[str, ...]]) -> dict[str, object]:
    """``name`` → the value at each key path in ``source``, leaving out a path the reply does not carry."""
    found: dict[str, object] = {}
    for name, path in facts.items():
        value: object = source
        for key in path:
            if not isinstance(value, Mapping) or key not in value:
                break
            value = value[key]
        else:
            found[name] = value
    return found


def unservable_bailing_model_types(release: str) -> dict[str, str]:
    """``model_type`` → loader fact for the Bailing spellings neither pinned engine takes an online update for."""
    return {
        "bailing_hybrid": (
            f"{release} registers Ling 3.0's BailingMoeV3ForCausalLM, but no online update into it has been "
            "validated end to end"
        ),
        "bailing_moe_linear": (
            f"Ring's checkpoints declare BailingMoeLinearV2ForCausalLM where {release} registers BailingMoeV2_5ForCausalLM"
        ),
    }


class WeightSyncClientBase(UpdateStream, ABC):
    """One engine's weight-sync client."""

    # The ``rollout_backend`` value the registry keys this client by.
    BACKEND_KEY: str
    LIVENESS_ROUTE: str = "/health"
    # The pinned server release every roster fact below describes.
    ENGINE_RELEASE: str
    # Env var naming the trainer NIC this engine dials back to.
    GROUP_HOST_ENV: str
    PAUSE_ENDPOINT: str
    PAUSE_REQUEST: Mapping[str, object] = {}
    RESUME_ENDPOINT: str
    RESUME_REQUEST: Mapping[str, object] = {}
    # True when the pause drops in-flight requests, so every turn a sync lands on is re-issued (``max_retries``).
    PAUSE_ABORTS_IN_FLIGHT: bool
    # The session the engine brackets an update with, inside the quiesce.
    SESSION_OPEN_ENDPOINT: str
    SESSION_CLOSE_ENDPOINT: str
    # The open takes the weights away until a close restores them, so a session no close can end leaves no model.
    SESSION_OPEN_UNLOADS_MODEL: bool = False
    SESSION_REQUEST: Mapping[str, object] = {}
    # ``model_type`` → why the pinned release cannot take an online update of the family (the gate quotes it).
    UNSERVABLE_MODEL_TYPES: Mapping[str, str] = {}
    # Families whose exported router balancing bias the engine's loader drops at serve time.
    DROPPED_BALANCING_BIAS_MODEL_TYPES: frozenset[str] = frozenset()
    # ``model_type`` → why the engine accepts a thinking budget for the family that it can never enforce.
    UNENFORCEABLE_THINKING_BUDGET_MODEL_TYPES: Mapping[str, str] = {}
    SUPPORTS_ENGINE_RESCORE = True
    # Why this engine cannot serve the policy as an adapter beside the frozen base, ``None`` where it can.
    ADAPTER_SYNC_REFUSAL: str | None
    # The adapted module names the engine validates and serves.
    ADAPTER_SERVABLE_MODULES: tuple[str, ...] = ()
    # The engine reports a launch id on /server_info, which tells a restart from a live group.
    REPORTS_INSTANCE_ID = False
    # No launch id, so a restart shows only as an engine that holds no group: an empty declare tells.
    PROBES_GROUP_MEMBERSHIP = False
    # ``ServerFlags.extras`` name → key path on the reply of ``/server_info`` (vLLM: under ``vllm_config``).
    LAUNCH_FACTS: Mapping[str, tuple[str, ...]]
    # How this engine's chunks are staged for its broadcast.
    CHUNK_STAGER: type[ChunkStager]
    # Route that flushes the engine's caches of the old weights where its final chunk carries no flag for it.
    CACHE_FLUSH_ENDPOINT: str | None = None
    # What an update interrupted mid-stream leaves the engine holding, quoted in the refusal.
    INTERRUPTED_UPDATE_STATE = "a model that is part old weights and part new"
    # The launch flags a preflight refusal names as the remedy.
    IMAGE_REMEDY: str
    MOE_BACKEND_FLAG: str
    REASONING_PARSER_FLAG: str
    TOOL_PARSER_FLAG: str
    CHAT_TEMPLATE_FLAG: str
    # Format fields: init log ``engine_ws, master, port, bind``; failure hint ``master, port, url, env``.
    _INIT_LOG: str
    _READY_LOG: str
    FORMATION_STEP: str
    _FORMATION_FAILURE: str

    def __init__(
        self,
        base_url: str,
        group_port: int = 0,
        connection_timeout: float = 0.0,
        group_host: str | None = None,
        *,
        retry_policy: Retry = _DEFAULT_RETRY,
    ):
        super().__init__()
        self._session = requests.Session()
        adapter = HTTPAdapter(max_retries=retry_policy)
        self._session.mount("http://", adapter)
        self._session.mount("https://", adapter)

        parsed = urlparse(base_url)
        if not parsed.hostname:
            raise ValueError(
                f"{self.BACKEND_NAME} base_url {base_url!r} has no host — it must include a scheme, "
                f"e.g. 'http://localhost:8000'."
            )
        self.host = socket.gethostbyname(parsed.hostname)
        self.base_url = f"{parsed.scheme or 'http'}://{parsed.netloc}{parsed.path}"

        self.group_port = group_port
        # NIC advertised for the NCCL group; NCCL_SOCKET_IFNAME picks the data plane.
        self.group_host = group_host
        # A pool flushes on worker threads, where the current device is thread-local.
        self._sync_device: torch.device | None = None
        self._served_model_id: str | None = None
        # The close or the exit hook must lift a quiesced engine.
        self._paused = False
        # An update session opened server-side (its reply possibly lost), which only the session close ends.
        self._update_session_open = False
        # The engine answered the open.
        self._update_session_confirmed = False
        self._transport: Any = None
        # The launch this client's group formed with, ``None`` when unread.
        self._group_instance: str | None = None
        # Set by a group that releases the engine's half of its group itself, beside the trainer's.
        self.group_releases_remote_half = False
        self._stager = self.CHUNK_STAGER()
        self._owns_stager = True
        self._reset_transport_state()
        self.check_server(connection_timeout)

    @abstractmethod
    def read_engine_layout(self, *, require: bool) -> tuple[int, int]:
        """``(engine ranks that will join the group, pipeline stages)`` from one read of the server; refuses a layout no
        rank can address. ``require`` also refuses an engine, image or state no update could reach."""

    def require_engine_world_size(self) -> int:
        return self.read_engine_layout(require=True)[0]

    @abstractmethod
    def request_engine_join(
        self, master_address: str, port: int, world_size: int, rank_offset: int, group_name: str | None
    ) -> AsyncCall:
        """Start the engine's request to join the group as ranks ``rank_offset`` onward, on a background thread.
        ``group_name`` names a group of several servers on an engine that keeps many (SGLang adopts it)."""

    @abstractmethod
    def destroy_remote_group(self) -> None:
        """Best-effort release of the engine's half of its weight-sync group; never raises."""

    @abstractmethod
    def abort_transport(self) -> None:
        """Abort the trainer's half of the group so kernels spinning on a silent peer end; unusable afterwards."""

    @abstractmethod
    def prepare_local_formation(
        self, master_address: str, listener: RendezvousListener, world_size: int
    ) -> Callable[[], Any]:
        """Run the caller-thread part of formation; return its blocking step, whose result becomes ``_transport``."""

    @abstractmethod
    def _release_transport(self, *, at_exit: bool, peer_gone: bool = False) -> None:
        """Release ``_transport`` (``None`` after a formation that built nothing) and the engine's side. ``peer_gone``
        marks a restarted server: nothing may wait for, probe or call the old peer."""

    @abstractmethod
    def _reset_transport_state(self) -> None:
        """Set the engine transport's fields to those of a client that formed nothing."""

    @abstractmethod
    def verify_transport(self) -> None:
        """Raise on a transport no update can go over, before the engine is touched."""

    @abstractmethod
    def prepare_chunk_send(self, chunk: list[tuple[str, torch.Tensor]]) -> None:
        """Raise on a chunk the engine or the transport would refuse, before any RPC holds the engine in its
        collective."""

    @abstractmethod
    def declare_chunk(self, chunk: list[tuple[str, torch.Tensor]], final: bool) -> AsyncCall:
        """Start the engine's receive of ``chunk``, on a background thread; ``final`` marks the last."""

    @abstractmethod
    def send_staged(self, staged: StagedChunk) -> None:
        """Queue a staged chunk on the group's transport, as rank 0; it only reads ``staged``."""

    @abstractmethod
    def _fail_chunk(self, error: Exception, declared: AsyncCall) -> NoReturn:
        """Raise ``error``, or the engine's own failure on ``declared`` when it recorded one, after the transport was
        left in a state a later close can release."""

    @abstractmethod
    def release_send_buffers(self) -> None:
        """Release the staging the transport holds between syncs, which would pin the sync GPU."""

    @property
    def layout_read_bound_s(self) -> float:
        """The longest a read of the engine's layout can take."""
        return HTTP_PROBE_TIMEOUT_S

    def send_chunk(self, chunk: list[tuple[str, torch.Tensor]]) -> None:
        """Stage ``chunk``, unless a client sharing the stager already did, and queue it."""
        self.send_staged(self._stager.stage(chunk))

    def share_stager(self, stager: ChunkStager) -> None:
        """Stage through ``stager``, which several clients of one pool share and whose owner releases."""
        self._stager = stager
        self._owns_stager = False

    def release_chunk_staging(self) -> None:
        """Release the staging this client owns; a shared stager is its pool's to release."""
        if self._owns_stager:
            self._stager.release()

    @property
    def has_transport(self) -> bool:
        return self._transport is not None

    @property
    def sync_device(self) -> torch.device | None:
        """The GPU the transport was formed on, ``None`` before ``init_communicator``."""
        return self._sync_device

    @sync_device.setter
    def sync_device(self, device: torch.device | str | int) -> None:
        self._sync_device = resolve_sync_device(device)

    def request(
        self, method: str, path: str, *, retry: bool, timeout: float, check: bool = True, **kwargs
    ) -> requests.Response:
        """Send ``method`` to ``path``, raising on an error status unless ``check`` is off. ``retry`` goes over the
        retrying session; a blocking engine RPC must not, since a retry queues a second collective call."""
        url = f"{self.base_url}{path}"
        try:
            if retry:
                resp = self._session.request(method, url, timeout=timeout, **kwargs)
            else:
                resp = requests.request(method, url, timeout=timeout, **kwargs)
        except requests.exceptions.ConnectionError as e:
            # A body that stalls after the headers surfaces as a ConnectionError wrapping urllib3's read timeout,
            # which no bounded wait would name as an expiry.
            if e.args and isinstance(e.args[0], urllib3.exceptions.ReadTimeoutError):
                raise requests.exceptions.ReadTimeout(e.args[0], request=e.request) from e
            raise
        if check:
            try:
                resp.raise_for_status()
            except requests.exceptions.HTTPError as e:
                # The route's own reason, which the bare status drops: "Bad Request" names no stale session.
                raise requests.exceptions.HTTPError(f"{e}: {resp.text[:500]}", response=resp) from e
        return resp

    def check_server(self, total_timeout: float = 0.0):
        """Poll the liveness route until it answers 200 or ``total_timeout`` passes."""
        start = time.time()
        last_error: Exception | None = None
        last_result = "no response"
        while True:
            try:
                # A blackholed route would otherwise block past the outer deadline.
                status = self.request(
                    "GET", self.LIVENESS_ROUTE, retry=False, timeout=HTTP_PROBE_TIMEOUT_S, check=False
                ).status_code
                last_result = f"HTTP {status}"
                if status == 200:
                    if not self._still_starting():
                        logger.info(f"{self.BACKEND_NAME} server is up")
                        return
                    last_result += ", and its start has not finished"
            except requests.exceptions.RequestException as exc:
                last_error, last_result = exc, f"{type(exc).__name__}: {exc}"
            # A route answering 503 forever is as dead as an unreachable one.
            if time.time() - start >= total_timeout:
                raise ConnectionError(
                    f"{self.BACKEND_NAME} server unreachable at {self.base_url} after {total_timeout}s "
                    f"(last {self.LIVENESS_ROUTE} result: {last_result}). A server still loading needs a longer wait: "
                    f"rollout_connection_timeout, or vllm_server_timeout on online GRPO."
                ) from last_error
            time.sleep(2.0)

    def _still_starting(self) -> bool:
        """Whether a server answering the liveness route has yet to finish starting, for a route that answers early."""
        return False

    @classmethod
    def adapter_request_model(cls, base_model: str, adapter: str) -> str:
        """The ``model`` a request names to be served through ``adapter``."""
        raise NotImplementedError(f"{cls.BACKEND_NAME} serves no adapter: {cls.ADAPTER_SYNC_REFUSAL}")

    def load_adapter(self, name: str, config: dict, payload: str) -> None:
        """Register the serialized adapter ``payload`` as ``name``; raises where the engine refuses."""
        raise NotImplementedError(f"{self.BACKEND_NAME} serves no adapter: {self.ADAPTER_SYNC_REFUSAL}")

    def unload_adapter(self, name: str) -> None:
        """Drop the adapter ``name`` once the requests running under it finished."""
        raise NotImplementedError(f"{self.BACKEND_NAME} serves no adapter: {self.ADAPTER_SYNC_REFUSAL}")

    def loaded_adapters(self) -> list[str]:
        """The adapters registered on the engine."""
        raise NotImplementedError(f"{self.BACKEND_NAME} serves no adapter: {self.ADAPTER_SYNC_REFUSAL}")

    def served_model_id(self) -> str:
        if self._served_model_id is None:
            cards = self._served_model_cards()
            if not cards:
                raise RuntimeError(f"{self.base_url}/v1/models lists no model")
            self._served_model_id = str(cards[0]["id"])
        return self._served_model_id

    def served_base(self) -> ServedBase:
        """The base this engine launched with, which a restart serves again."""
        raise NotImplementedError(f"{self.BACKEND_NAME} serves no adapter: {self.ADAPTER_SYNC_REFUSAL}")

    @classmethod
    def is_adapter_missing(cls, error: str, adapter: str) -> bool:
        """Whether an episode's ``error`` is the engine saying it holds no adapter ``adapter``."""
        return False

    @abstractmethod
    def read_server_flags(self) -> ServerFlags:
        """The flags this server reports; raises on a transport or decode error."""

    def reports_sampled_token_ids(self) -> bool | None:
        """Whether a turn carries the ids it sampled, ``None`` where the request itself asks for them."""
        return None

    def _read_paused(self) -> bool:
        """Whether the engine reports a pause; one with no such route reads as not paused and queues the probe."""
        return False

    def probe_generation(self) -> None:
        """Fail fast when the server is up but generation is held, a trainer's SIGKILL mid-sync having skipped the exit
        hook's resume. A paused engine would queue every request, so the pause is asked first where it can be."""
        restart = (
            "unless that sync had already streamed part of the model, in which case the server holds a partly written "
            "model and has to be RESTARTED instead (its log names the last update)"
        )
        if self._read_paused():
            raise RuntimeError(
                f"{self.BACKEND_NAME} server {self.base_url} is PAUSED for a weight update: generation requests would "
                f"queue instead of running. A trainer died mid weight sync without lifting it. Lift it (POST "
                f"{self.base_url}{self.RESUME_ENDPOINT}) before reconnecting, {restart}."
            )
        try:
            model_id = self.served_model_id()
        except Exception as e:
            logger.warning(f"Generation probe: could not resolve served model ({e}); skipping probe")
            return
        try:
            self.request(
                "POST",
                "/v1/completions",
                retry=False,
                timeout=GENERATION_PROBE_TIMEOUT_S,
                check=False,
                json={"model": model_id, "prompt": "ping", "max_tokens": 1, "temperature": 0.0},
            )
        except requests.exceptions.Timeout as e:
            raise RuntimeError(
                f"{self.BACKEND_NAME} server {self.base_url} is up but GENERATION is paused or wedged (no "
                f"response in {GENERATION_PROBE_TIMEOUT_S:.0f}s to a 1-token probe). A previous trainer most likely "
                f"died mid-run while attached to the weight-sync routes (SIGKILL skips cleanup). Lift a pause (POST "
                f"{self.base_url}{self.RESUME_ENDPOINT}) before reconnecting, {restart}."
            ) from e
        except requests.exceptions.RequestException as e:
            # For example raw /v1/completions disabled.
            logger.warning(f"Generation probe: server responded with an error (not wedged): {e}")

    def accepts_tool_requests(self) -> bool | None:
        """Whether a request carrying tools is served, ``None`` where the server reports its parser instead."""
        return None

    def _require_toolkit_image(self, flags: ServerFlags) -> None:
        """Refuse a server off the toolkit's image: its routes answer, and its update lands wrong behind a ``200``."""
        if not flags.toolkit_image:
            raise RuntimeError(
                f"{self.BACKEND_NAME} server {self.base_url} does not report the toolkit's image on /server_info. "
                f"Upstream {self.ENGINE_RELEASE} takes the same routes without the patches a weight sync needs, so "
                f"updates would land wrong with no error. {self.IMAGE_REMEDY}"
            )

    @abstractmethod
    def score_completion_logprobs(
        self, prompt_ids: list[int], completion_ids: list[int], timeout: float = RESCORE_TIMEOUT_S
    ) -> list[float]:
        """Per-token log-probs of ``completion_ids`` under the engine's CURRENT weights, from one prefill. Raises on a
        transport, shape or key error."""

    def served_max_model_len(self) -> int | None:
        for card in self._served_model_cards():
            if card.get("max_model_len"):
                return int(card["max_model_len"])
        return None

    def init_communicator(self, device: torch.device | str | int = 0) -> None:
        """Form the weight-transfer group: the trainer is rank 0, the engine's workers 1..N."""
        world_size = self.require_engine_world_size() + 1
        self._sync_device = resolve_sync_device(device)
        master_address, listener = open_group_rendezvous(
            engine=self.BACKEND_NAME,
            server_url=self.base_url,
            server_host=self.host,
            advertised_host=self.group_host,
            host_env=self.GROUP_HOST_ENV,
            port=self.group_port,
        )
        master_port = listener.port
        logger.info(
            self._INIT_LOG.format(
                engine_ws=world_size - 1, master=master_address, port=master_port, bind=listener.bind_address
            )
        )
        # Closed on exit unless the store took the listener over.
        with listener:
            server_call = self.request_engine_join(master_address, master_port, world_size, FIRST_ENGINE_RANK, None)
            local_call: AsyncCall | None = None
            try:
                # Both halves block until the other arrives, so they run concurrently and the first failure wins.
                local_call = AsyncCall(
                    self.prepare_local_formation(master_address, listener, world_size), name=self.FORMATION_STEP
                )
                wait_for_calls([server_call, local_call], timeout=GROUP_FORMATION_TIMEOUT_S)
                self.adopt_transport(local_call)
            except Exception as e:
                self.discard_transport(local_call)
                if e is server_call.error:
                    raise  # the server named its own failure; a topology hint would mislead
                # A wrong-but-routable master surfaces as a bare store or handshake timeout, so name the knob.
                raise RuntimeError(self.formation_failure(master_address, master_port, self.base_url, e)) from e
        atexit.register(self.close_communicator, at_exit=True)
        logger.info(self._READY_LOG)

    def formation_failure(self, master_address: str, port: int, urls: str, error: Exception) -> str:
        """What a formation that failed with no server's own error names: the advertised address and its knob."""
        failure = self._FORMATION_FAILURE.format(master=master_address, port=port, url=urls, env=self.GROUP_HOST_ENV)
        return f"{failure} Original error: {type(error).__name__}: {error}"

    def adopt_transport(self, formed: AsyncCall) -> None:
        """Take the transport a finished local formation built."""
        self._transport = formed.result

    def discard_transport(self, formed: AsyncCall | None) -> None:
        """Release what a failed formation left, ``formed`` being its local half if it started. A transport already
        formed lives only on the call handle; unreleased, later reconnects are refused."""
        if formed is not None:
            if formed.is_alive():
                wait_announced(formed, self.base_url, "its local formation step to end", SERVER_ERROR_GRACE_S)
            self._transport = formed.result
        self._release_transport(at_exit=False)
        self._transport = None
        self._group_instance = None
        self._reset_transport_state()

    def update_named_param(self, name: str, weights: torch.Tensor):
        """Buffer a snapshot of one gathered param, sending the pending chunk first when it would overflow."""
        snapshot = snapshot_to_sync_device(
            name,
            weights,
            device=self.sync_device,
            buffered_bytes=self._buffered_bytes,
            budget=WEIGHT_SYNC_CHUNK_BYTES,
            flush=self.flush_chunk,
        )
        self.buffer_param(name, snapshot)

    def abort_weight_update(self):
        """Close an update the sync opened but cannot finish and drop the buffer; never raises. Once a chunk went
        out there is no state to return to, so the engine stays paused (:meth:`_mark_unusable`)."""
        self.drain_param_buffer()
        self.release_chunk_staging()
        if not self._update_open:
            return
        if self.can_replay_sync:
            try:
                self._close_update([])
            except Exception as e:  # the caller's own failure is the one worth reporting
                logger.warning(f"Failed to close the interrupted weight update on {self.base_url}: {e}")
            return
        # Left paused with the update open; the flag is cleared so a second abort does not report again.
        self._update_open = False
        self._mark_unusable()

    def close_communicator(self, *, at_exit: bool = False) -> None:
        """Close an interrupted update, lift the pause and release the transport; ``at_exit`` is the exit hook's
        call. An engine an abort left unservable keeps its pause."""
        self.abort_weight_update()
        if self._paused:
            self._lift_pause(HTTP_PROBE_TIMEOUT_S, context="at close_communicator")
        if self._transport is not None:
            self._release_transport(at_exit=at_exit)
            self._transport = None
        self._reset_buffer_state()
        # Left registered, the hook pins every retired client and its resume POST until interpreter exit.
        atexit.unregister(self.close_communicator)

    def close_communicator_of_restarted_server(self) -> None:
        """Drop the communicator of a server known to have restarted, with no request to the server and no handshake
        with the old peer, which is gone. The new server holds no pause, update or group of this client's."""
        self._paused = False
        if self._transport is not None:
            self._release_transport(at_exit=False, peer_gone=True)
            self._transport = None
        self._reset_buffer_state()
        atexit.unregister(self.close_communicator)

    def close_http_session(self) -> None:
        self._session.close()

    def _served_model_cards(self) -> list[dict]:
        return list(self.request("GET", "/v1/models", retry=True, timeout=HTTP_PROBE_TIMEOUT_S).json().get("data", []))

    def _start_collective_rpc(self, path: str, *, timeout: float, json: dict) -> AsyncCall:
        """A single-shot POST on a background thread, for a route answered once the engine's collective returned."""
        return AsyncCall(name=path, fn=lambda: self.request("POST", path, retry=False, timeout=timeout, json=json))

    def _start_engine_update(self) -> None:
        # Before the pause and the session, which unload a restarted engine's model: its first chunk would find no group.
        self.verify_engine_identity()
        self.verify_transport()
        self.open_engine_update()

    def open_engine_update(self) -> None:
        """Quiesce the engine and open its update, unwinding both if either fails: nothing is sent yet."""
        try:
            self._quiesce()
            self.open_update_session()
        except Exception:
            self.cancel_engine_update("after a failed pause")
            raise

    def cancel_engine_update(self, context: str) -> None:
        """Close the update session and resume an engine whose sync did not reach it; never raises."""
        self.abandon_update_session()
        self._lift_pause(CLEANUP_TIMEOUT_S, context=context)

    def end_engine_update(self, *, flushed: bool) -> None:
        """End the session once its update landed. ``flushed`` says the final chunk carried the engine's cache flush;
        raises if the close failed, as the update is then not complete."""
        try:
            if not flushed and self.CACHE_FLUSH_ENDPOINT is not None:
                self.request("POST", self.CACHE_FLUSH_ENDPOINT, retry=True, timeout=POST_TIMEOUT_S)
            self._end_update_session()
        finally:
            self.abandon_update_session()

    def adopt_engine_identity(self) -> None:
        """Take the engine's current launch as the one its group forms with. Raises where an engine that reports launch
        ids reports none, as a restart could then not be told from a live group."""
        if not self.REPORTS_INSTANCE_ID:
            return
        instance = self.read_server_flags().instance_id
        if instance is None:
            raise RuntimeError(
                f"{self.BACKEND_NAME} server {self.base_url} reports no launch id on /server_info, so a restart of "
                f"it could not be told from a live weight-sync group. Serve from the toolkit's image: {self.IMAGE_REMEDY}"
            )
        self._group_instance = instance

    def verify_engine_identity(self) -> None:
        """Raise :class:`GroupMembershipLost` when the engine is not the launch this client's group formed with; an
        unread launch is not checked."""
        if self._group_instance is None:
            return
        instance = self.read_server_flags().instance_id
        if instance != self._group_instance:
            raise GroupMembershipLost(
                f"{self.BACKEND_NAME} server {self.base_url} restarted after this client's group formed (instance "
                f"{self._group_instance}, now {instance}), so that group is gone."
            )

    def verify_group_membership(self) -> None:
        """Raise :class:`GroupMembershipLost` when the engine, in its open session, holds no such group. An empty
        declare is a no-op on an engine that holds the group and an error on one that does not, which a real declare
        would turn into peers parked in a receive nobody can free."""
        if not self.PROBES_GROUP_MEMBERSHIP:
            return
        try:
            self.declare_chunk([], False).wait(timeout=POST_TIMEOUT_S)
        except requests.exceptions.HTTPError as e:
            raise GroupMembershipLost(
                f"{self.BACKEND_NAME} server {self.base_url} holds no weight-update group ({e}); it restarted since "
                f"the group formed."
            ) from e

    def _quiesce(self) -> None:
        """Pause the engine for an update; flagged first, since an engine pauses before its reply can be lost."""
        self._paused = True
        self.request("POST", self.PAUSE_ENDPOINT, retry=True, timeout=POST_TIMEOUT_S, **self.PAUSE_REQUEST)

    def open_update_session(self) -> None:
        """Open the engine's update session; flagged first, as the pause is. Single-shot, as a session retry would
        issue the collective RPC twice."""
        self._update_session_open = True
        self.request("POST", self.SESSION_OPEN_ENDPOINT, retry=False, timeout=POST_TIMEOUT_S, **self.SESSION_REQUEST)
        self._update_session_confirmed = True

    def _end_update_session(self) -> None:
        """Close the session once the update landed; raises, as the update is then not complete."""
        self.request(
            "POST", self.SESSION_CLOSE_ENDPOINT, retry=False, timeout=SESSION_CLOSE_TIMEOUT_S, **self.SESSION_REQUEST
        )
        self._update_session_open = False
        self._update_session_confirmed = False

    def abandon_update_session(self) -> None:
        """Close a session left open; never raises, as the caller is already failing."""
        if not self._update_session_open:
            return
        try:
            with announced_wait(self.base_url, "the interrupted update session to close", CLEANUP_TIMEOUT_S):
                self.request(
                    "POST", self.SESSION_CLOSE_ENDPOINT, retry=False, timeout=CLEANUP_TIMEOUT_S, **self.SESSION_REQUEST
                )
        except Exception as e:
            logger.warning(f"Failed to close the interrupted weight-update session on {self.base_url}: {e}")
            if self._update_session_confirmed and self.SESSION_OPEN_UNLOADS_MODEL:
                self._mark_unusable()
        self._update_session_open = False
        self._update_session_confirmed = False

    def _lift_pause(self, timeout: float, context: str) -> None:
        """Resume a quiesced engine unless it was left partly written. Never raises on a server failure, which must
        not mask the real error, and is sent once, as the session's backoff would block ~80s at interpreter exit."""
        if self._unusable_reason is not None:
            logger.error(f"Leaving {self.base_url} paused {context}: {self._unusable_reason}")
            return
        # Outside the try, so a client without the route fails by name, not as a dead server.
        endpoint = self.RESUME_ENDPOINT
        try:
            self.request("POST", endpoint, retry=False, timeout=timeout, **self.RESUME_REQUEST)
            self._paused = False
        except Exception as e:  # a paused server wedges later rollouts, but so does raising
            logger.warning(f"Failed to resume paused {self.BACKEND_NAME} server {context}: {e}")

    def _broadcast_chunk(self, chunk: list[tuple[str, torch.Tensor]], final: bool) -> None:
        """Declare and broadcast one chunk, returning once the engine has it; ``final`` marks the last."""
        self.prepare_chunk_send(chunk)
        declared = self.declare_chunk(chunk, final)
        try:
            self.send_chunk(chunk)
            declared.wait(timeout=CHUNK_SYNC_TIMEOUT_S)
        except Exception as e:
            self._fail_chunk(e, declared)

    def _raise_if_server_failed(self, error: Exception, server_call: AsyncCall, hint: str = "") -> None:
        """Raise ``error`` naming the server's own failure, which sits on an unjoined thread, when it recorded one."""
        wait_announced(server_call, self.base_url, f"its {server_call.name} call to report", SERVER_ERROR_GRACE_S)
        server_error = server_call.error
        if server_error is not None and server_error is not error:
            raise RuntimeError(
                f"{self.BACKEND_NAME} weight sync to {self.base_url} failed "
                f"({type(error).__name__}: {error}); the server's {server_call.name} call failed with: "
                f"{type(server_error).__name__}: {server_error}.{hint}"
            ) from server_error

    def mark_unusable_after(self, chunks_sent: int) -> None:
        """:meth:`_mark_unusable` for a group's sync, whose chunk count this client never saw."""
        self._chunks_sent = chunks_sent
        self._mark_unusable()

    def _mark_unusable(self) -> None:
        """Refuse every later sync and leave the engine paused: it holds neither policy, and the trainer keeps no
        copy of what landed to repair it. Idempotent."""
        if self._unusable_reason is not None:
            return
        state = self.INTERRUPTED_UPDATE_STATE if self._chunks_sent else "no model, as the update's open unloaded it"
        self._unusable_reason = (
            f"{self.BACKEND_NAME} server {self.base_url} was left mid-update by an interrupted weight "
            f"sync, after {self._chunks_sent} chunk(s) had already been broadcast: it holds "
            f"{state}. It is left PAUSED instead of resumed so it cannot serve "
            f"that model, and this client refuses further syncs. RESTART the {self.BACKEND_NAME} server "
            f"before using it again — the trainer cannot repair it."
        )
        logger.error(self._unusable_reason)

    def resume_engine(self) -> None:
        self._lift_pause(CLEANUP_TIMEOUT_S, context="after sync")

    def require_resumed(self) -> None:
        """A good push still counts for nothing if the pause stays: every rollout would wait out its request timeouts and
        the run would die on a timeout that never names the pause. After the close only, as a raise from its finally
        would mask the failure it unwinds."""
        for backoff in _RESUME_RETRY_BACKOFF_S:
            if not self._paused:
                return
            time.sleep(backoff)
            self.resume_engine()
        if self._paused:
            raise RuntimeError(
                f"{self.BACKEND_NAME} server {self.base_url} stayed paused after a good weight push: POST "
                f"{self.RESUME_ENDPOINT} failed {1 + len(_RESUME_RETRY_BACKOFF_S)} times, so every rollout would queue. "
                f"Check the server, lift the pause (POST {self.base_url}{self.RESUME_ENDPOINT}), or restart it."
            )

    def _finish_engine_update(self, tail: list[tuple[str, torch.Tensor]]) -> None:
        """Send ``tail`` and close the update; the resume follows."""
        try:
            self._transmit_chunk(tail, final=True)
            self.release_send_buffers()
            self.end_engine_update(flushed=bool(tail))
        finally:
            self.abandon_update_session()
