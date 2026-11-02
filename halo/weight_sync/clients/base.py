"""The engine-agnostic half of a weight-sync client: the HTTP control plane, the group lifecycle and the
streamed update. The trainer joins the engine's group as rank 0 and, while the engine is quiesced, streams
the gathered parameters in byte-bounded chunks; the engine hooks are each client's."""

import atexit
import logging
import socket
import time
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable, Mapping
from typing import Any
from urllib.parse import urlparse

import requests
import torch
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from halo.weight_sync.chunking import (
    WEIGHT_SYNC_CHUNK_BYTES,
    declared_co_load_groups,
    resolve_sync_device,
    snapshot_to_sync_device,
    split_co_loaded,
)
from halo.weight_sync.clients.server_calls import AsyncCall, wait_for_calls
from halo.weight_sync.transport.rendezvous import RendezvousListener, open_group_rendezvous

logger = logging.getLogger(__name__)

# Per-attempt deadline of a short control-plane POST; the calls that block on the engine have their own.
POST_TIMEOUT_S = 60
HTTP_PROBE_TIMEOUT_S = 10
GROUP_FORMATION_TIMEOUT_S = 120
# Closing the update session waits for the engine to finish applying the last chunk.
SESSION_CLOSE_TIMEOUT_S = 600
# Cleanup POSTs run on an already-failing path, where a long block only delays the real raise.
CLEANUP_TIMEOUT_S = 30
# A re-score is one prefill of a training row: a 250k-token row on a server that is also generating.
RESCORE_TIMEOUT_S = 600.0
# Grace for a still-in-flight server call to record its error once the local side has already failed.
_SERVER_ERROR_GRACE_S = 5.0
# What a client's session retries by default: a refused connection, a dropped read or a 5xx, with backoff.
_DEFAULT_RETRY = Retry(
    total=5,
    connect=5,
    read=5,
    status=3,
    status_forcelist=[500, 502, 503],
    backoff_factor=2,
    allowed_methods=["POST", "GET"],
)


def unservable_bailing_model_types(release: str) -> dict[str, str]:
    """``model_type`` → loader fact for the Bailing spellings neither pinned engine takes an online update for,
    quoting ``release``."""
    return {
        "bailing_hybrid": (
            f"{release} registers Ling 3.0's BailingMoeV3ForCausalLM, but no online update into it has been "
            "validated end to end"
        ),
        "bailing_moe_linear": (
            f"Ring's checkpoints declare BailingMoeLinearV2ForCausalLM where {release} registers BailingMoeV2_5ForCausalLM"
        ),
    }


class WeightSyncClientBase(ABC):
    """One engine's weight-sync client. A fact declared by annotation alone has no default, so a client that
    leaves one out fails by name where it is read."""

    # The ``rollout_backend`` config value the registry keys this client by.
    BACKEND_KEY: str
    # The engine's name in operator-facing errors and logs.
    BACKEND_NAME: str
    # The pinned server release every roster fact below describes.
    ENGINE_RELEASE: str
    # Env var naming the trainer NIC this engine dials back to.
    GROUP_HOST_ENV: str
    # Routes that quiesce the engine for an update and lift it, with the request keywords each takes.
    PAUSE_ENDPOINT: str
    PAUSE_REQUEST: Mapping[str, object] = {}
    RESUME_ENDPOINT: str
    RESUME_REQUEST: Mapping[str, object] = {}
    # Whether that pause drops the requests in flight, so every turn a sync lands on is re-issued (``max_retries``).
    PAUSE_ABORTS_IN_FLIGHT: bool
    # The session the engine brackets an update with inside the quiesce, and the request keywords both routes take.
    SESSION_OPEN_ENDPOINT: str
    SESSION_CLOSE_ENDPOINT: str
    SESSION_REQUEST: Mapping[str, object] = {}
    # ``model_type`` → why the pinned release cannot take an online update of the family (the gate quotes it).
    UNSERVABLE_MODEL_TYPES: Mapping[str, str] = {}
    # Per-module parameter suffixes the engine's loader takes only together, so they ride one chunk.
    CO_LOADED_PARAM_GROUPS: tuple[tuple[str, ...], ...] = ()
    # Families whose exported router balancing bias the engine's loader drops at serve time.
    DROPPED_BALANCING_BIAS_MODEL_TYPES: frozenset[str] = frozenset()
    # ``model_type`` → why the engine accepts a thinking budget for the family that it can never enforce.
    UNENFORCEABLE_THINKING_BUDGET_MODEL_TYPES: Mapping[str, str] = {}
    # Whether :meth:`score_completion_logprobs` can re-score under the engine's current weights.
    SUPPORTS_ENGINE_RESCORE = True
    # What an update interrupted mid-stream leaves the engine holding, quoted in the refusal that follows.
    INTERRUPTED_UPDATE_STATE = "a model that is part old weights and part new"
    # Group-formation texts: the init log (``engine_ws``, ``master``, ``port``, ``bind``), the ready log, the
    # local formation step's name and the failure hint (``master``, ``port``, ``url``, ``env``).
    _INIT_LOG: str
    _READY_LOG: str
    _FORMATION_STEP: str
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
        # Carries the requests sent with ``retry``.
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
        # Recorded rather than inferred: a pool flushes on worker threads, where the current device is
        # thread-local.
        self._sync_device: torch.device | None = None
        self._served_model_id: str | None = None
        # Set once an interrupted sync left the engine unservable; every later sync is refused with it.
        self._unusable_reason: str | None = None
        # The engine may be quiesced, so the close or the exit hook must lift it.
        self._paused = False
        # An update session opened server-side (its reply possibly lost), which only the session close ends.
        self._session_open = False
        # What the group formation built, released by the close.
        self._transport: Any = None
        self._reset_buffer_state()
        self._reset_transport_state()
        self.check_server(connection_timeout)

    @abstractmethod
    def _fetch_engine_world_size(self) -> int:
        """Number of engine ranks that will join the group, read from the server."""

    @abstractmethod
    def _request_engine_join(self, master_address: str, port: int, world_size: int) -> AsyncCall:
        """Start the engine's request to join the group as ranks 1..N, on a background thread."""

    @abstractmethod
    def _prepare_local_formation(
        self, master_address: str, listener: RendezvousListener, world_size: int
    ) -> Callable[[], Any]:
        """Run the caller-thread part of the trainer's formation; return its blocking step, whose result becomes
        ``_transport``."""

    @abstractmethod
    def _release_transport(self, *, at_exit: bool) -> None:
        """Release ``_transport`` (``None`` after a formation that built nothing) and the engine's side."""

    @abstractmethod
    def _reset_transport_state(self) -> None:
        """Set the engine transport's fields to those of a client that formed nothing."""

    @abstractmethod
    def _start_engine_update(self) -> None:
        """Quiesce the engine and open its update, unwinding both if either fails: nothing is sent yet."""

    @abstractmethod
    def _broadcast_chunk(self, chunk: list[tuple[str, torch.Tensor]], final: bool) -> None:
        """Declare and broadcast one chunk, returning once the engine has it; ``final`` marks the last."""

    @abstractmethod
    def _finish_engine_update(self, tail: list[tuple[str, torch.Tensor]]) -> None:
        """Send ``tail`` (possibly empty) as the final chunk and close the update; the resume follows."""

    @property
    def sync_device(self) -> torch.device | None:
        """The GPU the transport was formed on, ``None`` before ``init_communicator``."""
        return self._sync_device

    @property
    def buffered_bytes(self) -> int:
        """Bytes buffered for the next chunk, members held for a co-load partner included."""
        return self._buffered_bytes

    @property
    def can_replay_sync(self) -> bool:
        """Whether re-sending this client's buffer restores a known state: not once a chunk is on the wire, since
        the trainer keeps no copy of what went out and a fresh engine would get only the tail."""
        return self._chunks_sent == 0

    def request(
        self, method: str, path: str, *, retry: bool, timeout: float, check: bool = True, **kwargs
    ) -> requests.Response:
        """Send ``method`` to ``path``, raising on an error status unless ``check`` is off. ``retry`` goes over the
        retrying session; a blocking engine RPC must not, since a retry queues a second collective call."""
        url = f"{self.base_url}{path}"
        if retry:
            resp = self._session.request(method, url, timeout=timeout, **kwargs)
        else:
            resp = requests.request(method, url, timeout=timeout, **kwargs)
        if check:
            resp.raise_for_status()
        return resp

    def check_server(self, total_timeout: float = 0.0):
        """Poll ``/health`` until it answers 200 or ``total_timeout`` passes."""
        start = time.time()
        last_error: Exception | None = None
        last_result = "no response"
        while True:
            try:
                # Bounded GET: a blackholed route would otherwise block past the outer deadline.
                status = self.request(
                    "GET", "/health", retry=False, timeout=HTTP_PROBE_TIMEOUT_S, check=False
                ).status_code
                if status == 200:
                    logger.info(f"{self.BACKEND_NAME} server is up")
                    return
                last_result = f"HTTP {status}"
            except requests.exceptions.RequestException as exc:
                last_error, last_result = exc, f"{type(exc).__name__}: {exc}"
            # Every outcome is under the deadline: a route answering 503 forever is as dead as an unreachable one.
            if time.time() - start >= total_timeout:
                raise ConnectionError(
                    f"{self.BACKEND_NAME} server unreachable at {self.base_url} after {total_timeout}s "
                    f"(last /health result: {last_result})"
                ) from last_error
            time.sleep(2.0)

    def served_model_id(self) -> str:
        """The id the engine serves the policy under, read once."""
        if self._served_model_id is None:
            self._served_model_id = str(self._served_model_cards()[0]["id"])
        return self._served_model_id

    @abstractmethod
    def score_completion_logprobs(
        self, prompt_ids: list[int], completion_ids: list[int], timeout: float = RESCORE_TIMEOUT_S
    ) -> list[float]:
        """Per-token log-probs of ``completion_ids`` after ``prompt_ids`` under the engine's CURRENT weights, from one
        prefill. Raises on a transport, shape or key error."""

    def served_max_model_len(self) -> int | None:
        """Context window the served model declares, or ``None`` when no card carries one."""
        for card in self._served_model_cards():
            if card.get("max_model_len"):
                return int(card["max_model_len"])
        return None

    def init_communicator(self, device: torch.device | str | int = 0) -> None:
        """Form the weight-transfer group: the trainer is rank 0, the engine's workers 1..N."""
        world_size = self._fetch_engine_world_size() + 1
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
            server_call = self._request_engine_join(master_address, master_port, world_size)
            local_call: AsyncCall | None = None
            try:
                # Both halves block until the other arrives, so they run concurrently and the first failure wins.
                local_call = AsyncCall(
                    self._prepare_local_formation(master_address, listener, world_size), name=self._FORMATION_STEP
                )
                wait_for_calls([server_call, local_call], timeout=GROUP_FORMATION_TIMEOUT_S)
                self._transport = local_call.result
            except Exception as e:
                # A transport this rank already formed lives only on the call handle; unreleased, it leaks its
                # group name, ncclComm or port and every later reconnect to this server is refused.
                if local_call is not None:
                    self._transport = local_call.join(_SERVER_ERROR_GRACE_S).result
                self._release_transport(at_exit=False)
                self._transport = None
                self._reset_transport_state()
                if e is server_call.error:
                    raise  # the server named its own failure; a topology hint would mislead
                # A wrong-but-routable master surfaces as a bare store or handshake timeout, so name the knob.
                failure = self._FORMATION_FAILURE.format(
                    master=master_address, port=master_port, url=self.base_url, env=self.GROUP_HOST_ENV
                )
                raise RuntimeError(f"{failure} Original error: {type(e).__name__}: {e}") from e
        atexit.register(self.close_communicator, at_exit=True)
        logger.info(self._READY_LOG)

    def scope_co_load_groups(self, module_names: Iterable[str]) -> None:
        """Narrow the co-load groups to the members the pushed model's modules declare."""
        self._co_load_groups = declared_co_load_groups(self.CO_LOADED_PARAM_GROUPS, module_names)

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

    def buffer_param(self, name: str, snapshot: torch.Tensor):
        """Buffer a snapshot by reference for the next chunk; a pool shares it across clients, read-only."""
        self._param_buffer.append((name, snapshot))
        self._buffered_bytes += snapshot.nbytes
        if snapshot.is_cuda:
            # Re-recorded per param on the copy's own stream, so the newest record covers every copy before it.
            if self._snapshot_event is None:
                self._snapshot_event = torch.cuda.Event()
            self._snapshot_event.record(torch.cuda.current_stream(snapshot.device))

    def flush_chunk(self):
        """Send what is buffered as one chunk into the open update; an incomplete co-load group stays buffered."""
        chunk, held = split_co_loaded(self._param_buffer, self._co_load_groups)
        if not chunk:
            return
        # Completed before the quiesce, so a sticky CUDA error here leaves the engine serving.
        self._complete_snapshots()
        self._open_update()
        self._transmit_chunk(chunk)
        # Dropped only once on the wire: a replayable failed chunk is re-sent by the reconnect path.
        self._param_buffer = held
        self._buffered_bytes = sum(snapshot.nbytes for _, snapshot in held)

    def finish_weight_update(self) -> None:
        """Send the tail chunk, close the update and resume the engine; no-op if nothing was buffered."""
        if not self._param_buffer and not self._update_open:
            return
        # A group still incomplete at the tail never completes, and the engine drops a lone half silently.
        _, held = split_co_loaded(self._param_buffer, self._co_load_groups)
        if held:
            raise RuntimeError(
                f"the weight sync ended with co-loaded parameters missing their partners: "
                f"{[name for name, _ in held]}. The {self.BACKEND_NAME} loader discards a half that "
                f"arrives on its own, so this update cannot be closed as complete."
            )
        self._open_update()
        self._close_update(self._param_buffer)
        self.drain_param_buffer()  # only once the tail landed — see flush_chunk

    def abort_weight_update(self):
        """Close an update the sync opened but cannot finish and drop the buffer; never raises.

        With nothing on the wire the engine still holds its weights, so the update is closed and the engine
        resumed; once a chunk went out there is no such state to return to (:meth:`_mark_unusable`).
        """
        self.drain_param_buffer()
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

    def drain_param_buffer(self) -> list[tuple[str, torch.Tensor]]:
        """Hand off and clear the buffered params (a reconnect moves them onto the replacement)."""
        buffered = self._param_buffer
        self._param_buffer = []
        self._buffered_bytes = 0
        return buffered

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

    def close_session(self) -> None:
        """Release the HTTP session, all a client that formed no group holds."""
        self._session.close()

    def _reset_buffer_state(self) -> None:
        """Initialize (or drop) the streamed-chunk state."""
        self._param_buffer: list[tuple[str, torch.Tensor]] = []
        self._buffered_bytes = 0
        self._co_load_groups = self.CO_LOADED_PARAM_GROUPS
        # A quiesced update this client opened mid-gather and has not closed yet.
        self._update_open = False
        # Chunks on the wire in the open update; non-zero means the engine is partly rewritten.
        self._chunks_sent = 0
        # Recorded after the newest snapshot copy; the flush waits on it, not on the whole device.
        self._snapshot_event: torch.cuda.Event | None = None

    def _served_model_cards(self) -> list[dict]:
        """The engine's ``/v1/models`` cards."""
        return list(self.request("GET", "/v1/models", retry=True, timeout=HTTP_PROBE_TIMEOUT_S).json().get("data", []))

    def _start_collective_rpc(self, path: str, *, timeout: float, json: dict) -> AsyncCall:
        """A single-shot POST of ``path`` on a background thread, for a route answered once the engine's side of
        a collective returned."""
        return AsyncCall(name=path, fn=lambda: self.request("POST", path, retry=False, timeout=timeout, json=json))

    def _complete_snapshots(self) -> None:
        """Complete the async snapshot copies before a producer's own streams read them: a wait on the newest
        copy's event, not a device synchronize, which would also wait on every server's broadcasts."""
        if self._snapshot_event is not None:
            self._snapshot_event.synchronize()

    def _open_update(self):
        """Open the quiesced update once per sync."""
        if self._update_open:
            return
        if self._unusable_reason is not None:
            raise RuntimeError(self._unusable_reason)
        # Before the engine is touched: a failure to open leaves the sync replayable.
        self._chunks_sent = 0
        self._start_engine_update()
        self._update_open = True

    def _quiesce(self) -> None:
        """Pause the engine for an update; flagged first, since an engine pauses before its reply can be lost."""
        self._paused = True
        self.request("POST", self.PAUSE_ENDPOINT, retry=True, timeout=POST_TIMEOUT_S, **self.PAUSE_REQUEST)

    def _open_session(self) -> None:
        """Open the engine's update session; flagged first, as the pause is. Single-shot: the route is a collective
        RPC, which a session retry would issue twice."""
        self._session_open = True
        self.request("POST", self.SESSION_OPEN_ENDPOINT, retry=False, timeout=POST_TIMEOUT_S, **self.SESSION_REQUEST)

    def _end_session(self) -> None:
        """Close the session once the update landed; raises, as the update is then not complete."""
        self.request(
            "POST", self.SESSION_CLOSE_ENDPOINT, retry=False, timeout=SESSION_CLOSE_TIMEOUT_S, **self.SESSION_REQUEST
        )
        self._session_open = False

    def _close_session(self) -> None:
        """Close a session left open; never raises, as the caller is already failing."""
        if not self._session_open:
            return
        try:
            self.request(
                "POST", self.SESSION_CLOSE_ENDPOINT, retry=False, timeout=CLEANUP_TIMEOUT_S, **self.SESSION_REQUEST
            )
        except Exception as e:  # advisory on a failing path
            logger.warning(f"Failed to close the interrupted weight-update session on {self.base_url}: {e}")
        self._session_open = False

    def _lift_pause(self, timeout: float, context: str) -> None:
        """Resume a quiesced engine, unless it was left partly written (:meth:`_mark_unusable`).

        Never raises on a server failure, which must not mask the real error, and is sent once: the session's
        backoff would block ~80s at interpreter exit.
        """
        if self._unusable_reason is not None:
            logger.error(f"Leaving {self.base_url} paused {context}: {self._unusable_reason}")
            return
        # Read outside the try: a client without the route must fail by name, not as a dead server.
        endpoint = self.RESUME_ENDPOINT
        try:
            self.request("POST", endpoint, retry=False, timeout=timeout, **self.RESUME_REQUEST)
            self._paused = False
        except Exception as e:  # advisory: a paused server wedges later rollouts, but so does raising
            logger.warning(f"Failed to resume paused {self.BACKEND_NAME} server {context}: {e}")

    def _transmit_chunk(self, chunk: list[tuple[str, torch.Tensor]], final: bool = False) -> None:
        """Put one chunk on the wire, counting it and disowning the engine when it fails mid-update."""
        if not chunk:
            return
        try:
            self._broadcast_chunk(chunk, final)
        except BaseException:
            # An earlier chunk landed: marked before the close's finally would resume the engine.
            if self._chunks_sent:
                self._mark_unusable()
            raise
        self._chunks_sent += 1

    def _raise_if_server_failed(self, error: Exception, server_call: AsyncCall | None, hint: str = "") -> None:
        """Raise ``error`` naming the server's own failure, which sits on an unjoined thread, when it recorded one."""
        server_error = server_call.join(_SERVER_ERROR_GRACE_S).error if server_call is not None else None
        if server_error is not None and server_error is not error:
            raise RuntimeError(
                f"{self.BACKEND_NAME} weight sync to {self.base_url} failed "
                f"({type(error).__name__}: {error}); the server's {server_call.name} call failed with: "
                f"{type(server_error).__name__}: {server_error}.{hint}"
            ) from server_error

    def _mark_unusable(self) -> None:
        """Refuse every later sync and leave the engine paused (idempotent).

        A sync interrupted after part of the model went out leaves the engine holding neither policy, and the
        trainer keeps no copy of what landed, so it cannot repair it.
        """
        if self._unusable_reason is not None:
            return
        self._unusable_reason = (
            f"{self.BACKEND_NAME} server {self.base_url} was left mid-update by an interrupted weight "
            f"sync, after {self._chunks_sent} chunk(s) had already been broadcast: it holds "
            f"{self.INTERRUPTED_UPDATE_STATE}. It is left PAUSED instead of resumed so it cannot serve "
            f"that model, and this client refuses further syncs. RESTART the {self.BACKEND_NAME} server "
            f"before using it again — the trainer cannot repair it."
        )
        logger.error(self._unusable_reason)

    def _close_update(self, tail: list[tuple[str, torch.Tensor]]):
        """Send the final chunk, close the update and resume, whatever the tail does. ``_chunks_sent`` stays, so a
        failed close still reports the sync unreplayable."""
        snapshot_error: Exception | None = None
        try:
            self._complete_snapshots()
        except Exception as e:
            # Inside the close: only it can end the open update. The tail goes, as its copies may not have landed.
            logger.error(f"Weight-sync snapshots did not complete for {self.base_url}: {e}")
            snapshot_error, tail = e, []
            if self._chunks_sent:
                # Dropping the tail after earlier chunks landed leaves a mixed model, as a failed chunk does.
                self._mark_unusable()
        try:
            self._finish_engine_update(tail)
        finally:
            self._update_open = False
            # A server left paused wedges every later rollout.
            self._lift_pause(CLEANUP_TIMEOUT_S, context="after sync")
        if snapshot_error is not None:
            raise snapshot_error
