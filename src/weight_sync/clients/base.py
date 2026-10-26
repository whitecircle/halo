"""The engine-agnostic half of the weight-sync clients: the HTTP control plane and the streamed update.

The trainer joins the engine's process group as rank 0 and, while the engine is quiesced, streams the
gathered parameters to it in byte-bounded chunks (``chunking``). Only the HTTP verbs and the broadcast
transport differ per engine; those are in ``VLLMWeightSyncClient`` / ``SGLangWeightSyncClient``.
"""

import atexit
import logging
import socket
import threading
import time
from collections.abc import Iterable, Mapping
from typing import NamedTuple
from urllib.parse import urlparse

import requests
import torch
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from src.weight_sync.chunking import (
    WEIGHT_SYNC_CHUNK_BYTES,
    declared_co_load_groups,
    resolve_sync_device,
    snapshot_to_sync_device,
    split_co_loaded,
)
from src.weight_sync.transport.rendezvous import RendezvousListener, open_group_rendezvous

logger = logging.getLogger(__name__)

# Bailing spellings whose checkpoints declare a model class neither pinned engine registers; each
# client quotes them for its own release through :func:`unregistered_bailing_model_types`.
_UNREGISTERED_BAILING_CLASSES = (
    ("bailing_hybrid", "{release} registers no model class for Ling 3.0's BailingMoeV3ForCausalLM"),
    (
        "bailing_moe_linear",
        "Ring's checkpoints declare BailingMoeLinearV2ForCausalLM where {release} registers BailingMoeV2_5ForCausalLM",
    ),
)

# Per-attempt deadline of a short control-plane POST; the calls that block on the engine have their own.
POST_TIMEOUT_S = 60
HTTP_PROBE_TIMEOUT_S = 10
GROUP_FORMATION_TIMEOUT_S = 120
WEIGHT_UPDATE_TIMEOUT_S = 600
# Cleanup POSTs run on an already-failing path, where a long block only delays the real raise.
CLEANUP_TIMEOUT_S = 30
# A re-score is one prefill of a training row: a 250k-token row on a server that is also generating.
RESCORE_TIMEOUT_S = 600.0
_LOGPROB_PROBE_TIMEOUT_S = 60.0
# Grace for a still-in-flight server call to record its error once the local side has already failed.
_SERVER_ERROR_GRACE_S = 5.0
_ASYNC_POLL_INTERVAL_S = 0.1
# Cadence of the startup /health poll, which runs against a server that may still be loading weights.
_HEALTH_RETRY_INTERVAL_S = 2.0
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

# Sampler-logprob semantics probe: 1-token completions of one prompt under three sampler settings.
_LOGPROB_PROBE_PROMPT = "The capital of France is"
_LOGPROB_PROBE_TEMPERATURE = 2.0
_LOGPROB_PROBE_TOP_P = 0.5
# Temperature-processed logprobs halve the top-1/top-2 gap at temperature 2; raw ones leave it whole.
_TEMPERATURE_APPLIED_GAP_RATIO_MAX = 0.75
# A nucleus renormalization lifts the top token's logprob by -log(nucleus mass) at top-p 0.5.
_NUCLEUS_RENORM_MIN_SHIFT_NATS = 1e-3


def unregistered_bailing_model_types(release: str) -> dict[str, str]:
    """``model_type`` → loader fact for the Bailing spellings the pinned engine ``release`` has no class for."""
    return {model_type: fact.format(release=release) for model_type, fact in _UNREGISTERED_BAILING_CLASSES}


class AsyncCall:
    """Run a function in a background thread, join later."""

    def __init__(self, fn, name: str = "server call"):
        self.error: Exception | None = None
        self.result = None
        self.name = name
        self._thread = threading.Thread(target=self._run, args=(fn,), daemon=True)
        self._thread.start()

    def _run(self, fn):
        try:
            self.result = fn()
        except Exception as e:
            self.error = e

    def is_alive(self) -> bool:
        return self._thread.is_alive()

    def wait(self, timeout: float = WEIGHT_UPDATE_TIMEOUT_S):
        self._thread.join(timeout=timeout)
        if self._thread.is_alive():  # join() returns without signalling a timeout
            raise TimeoutError(f"{self.name} still blocked after {timeout:.0f}s — server wedged?")
        if self.error:
            raise self.error

    def join(self, grace: float) -> "AsyncCall":
        """Wait up to ``grace`` seconds, then read ``.error`` / ``.result`` off the returned handle.

        Both readers run where the local side has already failed and ``_wait_for_calls`` raised as
        soon as any call did. An in-flight call may still record the server's own rejection, or hand
        back a resource (a formed group, a built communicator) that only this handle holds.
        """
        self._thread.join(timeout=grace)
        return self


def _wait_for_calls(calls: list[AsyncCall], timeout: float) -> None:
    """Block until every call returned, raising as soon as one of them fails.

    Polled rather than joined in sequence: the calls that set up the weight-transfer group complete
    only together (the engine's group-init request replies once the group formed, and the trainer's
    group creation returns once the engine joined), so a serial join would park forever if the other
    side had already failed.
    """
    deadline = time.monotonic() + timeout
    while True:
        for call in calls:
            if call.error is not None:
                raise call.error
        if not any(call.is_alive() for call in calls):
            return
        if time.monotonic() > deadline:
            blocked = ", ".join(call.name for call in calls if call.is_alive())
            raise TimeoutError(f"Weight-transfer setup still blocked after {timeout:.0f}s: {blocked}")
        time.sleep(_ASYNC_POLL_INTERVAL_S)


class SamplerLogprobSemantics(NamedTuple):
    """What the engine's reported per-token logprobs already account for; ``None`` = the probe could not tell."""

    temperature_applied: bool | None
    nucleus_renormalized: bool | None


class WeightSyncClientBase:
    """HTTP control plane and streamed-update state shared by every engine.

    Subclasses implement the engine-specific half (``_reset_transport_state``, ``init_communicator``,
    ``begin_weight_update``, ``_broadcast_chunk``, ``end_weight_update``, ``close_communicator``) and
    declare the engine facts below; those declared by annotation alone have no default, so a client that
    leaves one out fails by name where it is read. :meth:`score_completion_logprobs` is the OpenAI completions
    ``prompt_logprobs`` echo (vLLM); an engine with another prefill-log-prob route overrides it (SGLang:
    ``/generate`` with ``logprob_start_len``).
    """

    # The ``rollout_backend`` config value; the registry is built from these.
    BACKEND_KEY: str
    # The engine's name in operator-facing errors and logs.
    BACKEND_NAME: str
    # The pinned server release every roster fact below describes.
    ENGINE_RELEASE: str
    # Env var naming the trainer NIC the engine dials back to; per engine, as the servers can sit on different hosts.
    GROUP_HOST_ENV: str
    # Route that lifts the weight-sync quiesce, and the request keywords beside it.
    RESUME_ENDPOINT: str
    RESUME_REQUEST: Mapping[str, object] = {}
    # ``model_type`` → why this engine cannot serve the family (no model class registered, or a loader
    # that cannot take an online update of the layout the trainer holds). Quoted by the construction
    # gate; per engine, since the two pinned servers register different rosters.
    UNSERVABLE_MODEL_TYPES: Mapping[str, str] = {}
    # Parameter-name suffixes the engine's loader consumes only as a set, per module: each entry lists
    # the suffixes, and parameters sharing the prefix before them must ride one chunk. Empty where the
    # engine loads every tensor on its own. Scoped per model by :meth:`scope_co_load_groups`.
    CO_LOADED_PARAM_GROUPS: tuple[tuple[str, ...], ...] = ()
    # Whether :meth:`score_completion_logprobs` can re-score a sequence under the engine's current weights
    # (the ``isr_engine_reference`` trust region), read by the trainer's construction gate.
    SUPPORTS_ENGINE_RESCORE = True
    # What an update interrupted mid-stream leaves the engine holding, quoted in the refusal that
    # follows one. Declared per client because it follows from the engine's own reload model.
    INTERRUPTED_UPDATE_STATE = "a model that is part old weights and part new"

    def __init__(
        self,
        base_url: str,
        group_port: int = 0,
        connection_timeout: float = 0.0,
        group_host: str | None = None,
        *,
        retry_policy: Retry = _DEFAULT_RETRY,
    ):
        # Carries the requests sent with ``retry``; ``retry_policy`` is what it retries.
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
        # NIC advertised for the NCCL group. Distinct from NCCL_SOCKET_IFNAME (data plane).
        self.group_host = group_host
        # The GPU the transport was formed on, set by ``init_communicator``. Recorded rather than
        # inferred: the flush runs on a worker thread when several servers sync at once, and torch's
        # current device and current stream are both thread-local.
        self._sync_device: torch.device | None = None
        # Cached by ``served_model_id``; None until the first completions-route call.
        self._served_model_id: str | None = None
        # Set once an interrupted sync left the engine unservable; every later sync is refused with it.
        self._unusable_reason: str | None = None
        # Server quiesced for a weight sync; close_communicator/atexit lift it if the trainer dies.
        self._paused = False
        self._reset_buffer_state()
        self._reset_transport_state()
        self.check_server(connection_timeout)

    def request(
        self, method: str, path: str, *, retry: bool, timeout: float, check: bool = True, **kwargs
    ) -> requests.Response:
        """Send ``method`` to ``path`` on the server, raising on an error status unless ``check`` is off.

        ``retry`` sends it over the session, which retries it under the client's policy; without it the
        request goes out once. A blocking engine RPC must go out once: a retry queues a second collective
        call and wedges the engine.
        """
        url = f"{self.base_url}{path}"
        if retry:
            resp = self._session.request(method, url, timeout=timeout, **kwargs)
        else:
            resp = requests.request(method, url, timeout=timeout, **kwargs)
        if check:
            resp.raise_for_status()
        return resp

    def _start_collective_rpc(self, path: str, *, timeout: float, json: dict) -> AsyncCall:
        """Start a single-shot POST of ``path`` on a background thread, for a route the engine answers only
        once its side of a collective returned. The handle is labelled with the route."""
        return AsyncCall(name=path, fn=lambda: self.request("POST", path, retry=False, timeout=timeout, json=json))

    def check_server(self, total_timeout: float = 0.0):
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
            # The deadline governs every outcome: a route answering 503/401 forever is as dead as an
            # unreachable one, and checking it only on the exception path would spin without bound.
            if time.time() - start >= total_timeout:
                raise ConnectionError(
                    f"{self.BACKEND_NAME} server unreachable at {self.base_url} after {total_timeout}s "
                    f"(last /health result: {last_result})"
                ) from last_error
            time.sleep(_HEALTH_RETRY_INTERVAL_S)

    def _served_model_cards(self) -> list[dict]:
        """The engine's ``/v1/models`` model cards. Raises when the server cannot be read."""
        return list(self.request("GET", "/v1/models", retry=True, timeout=HTTP_PROBE_TIMEOUT_S).json().get("data", []))

    def served_model_id(self) -> str:
        """The id the engine serves the policy under, read once and cached for the completions route."""
        if self._served_model_id is None:
            self._served_model_id = str(self._served_model_cards()[0]["id"])
        return self._served_model_id

    def score_completion_logprobs(
        self, prompt_ids: list[int], completion_ids: list[int], timeout: float = RESCORE_TIMEOUT_S
    ) -> list[float]:
        """Per-token log-probs of ``completion_ids`` after ``prompt_ids`` under the engine's CURRENT
        weights: one prefill through the completions ``prompt_logprobs`` echo, the raw (pre-sampler)
        distribution. Raises on a transport, shape or key error — the caller decides what a failed
        row means.
        """
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

    def served_max_model_len(self) -> int | None:
        """Context window the served model declares, or ``None`` when no card carries one."""
        for card in self._served_model_cards():
            if card.get("max_model_len"):
                return int(card["max_model_len"])
        return None

    def _probe_top_logprobs(
        self, model_id: str, temperature: float, top_p: float, filters: Mapping[str, int | float]
    ) -> list[float] | None:
        """The top logprob values (descending) of one 1-token completion, or ``None`` when unreadable."""
        body = {
            "model": model_id,
            "prompt": _LOGPROB_PROBE_PROMPT,
            "max_tokens": 1,
            "temperature": temperature,
            "top_p": top_p,
            **filters,
            "logprobs": 2,
            "seed": 0,
        }
        try:
            resp = self.request("POST", "/v1/completions", retry=True, timeout=_LOGPROB_PROBE_TIMEOUT_S, json=body)
            top = resp.json()["choices"][0]["logprobs"]["top_logprobs"][0]
            values = sorted((float(v) for v in top.values() if v is not None), reverse=True)
        except (requests.exceptions.RequestException, KeyError, IndexError, TypeError, ValueError) as e:
            logger.warning(
                f"Sampler-logprob probe against {self.base_url} failed (T={temperature}, top_p={top_p}): {e}"
            )
            return None
        return values

    def probe_sampler_logprob_semantics(
        self, top_k: int, min_p: float, repetition_penalty: float
    ) -> SamplerLogprobSemantics:
        """Whether the engine's per-token logprobs are the SAMPLING distribution's — after temperature,
        and after a top-p renormalization — or the raw pre-processor values.

        The prefill logits of one prompt are the same under every sampler setting, so the reported
        values alone tell: temperature 2 halves the top-1/top-2 gap only when it is applied before the
        logprobs are taken, and top-p 0.5 lifts the top token's logprob only when the nucleus is
        renormalized into them. vLLM reports raw values unless served with ``--logprobs-mode
        processed_logprobs`` (temperature and nucleus both applied); SGLang reports post-temperature,
        pre-nucleus values unless ``SGLANG_RETURN_ORIGINAL_LOGPROB`` is set.

        ``top_k``, ``min_p`` and ``repetition_penalty`` ride on all three requests, and the caller passes
        their off values: each request moves one knob from the identity sampler, a filter set to a run's
        value could cut the second token the gap is read from, and one left out takes the model's
        generation_config default on vLLM's completions route.
        """
        try:
            model_id = self.served_model_id()
        except Exception as e:
            logger.warning(f"Sampler-logprob probe: could not resolve the model served at {self.base_url}: {e}")
            return SamplerLogprobSemantics(None, None)
        filters = {"top_k": top_k, "min_p": min_p, "repetition_penalty": repetition_penalty}
        base = self._probe_top_logprobs(model_id, 1.0, 1.0, filters)
        hot = self._probe_top_logprobs(model_id, _LOGPROB_PROBE_TEMPERATURE, 1.0, filters)
        nucleus = self._probe_top_logprobs(model_id, 1.0, _LOGPROB_PROBE_TOP_P, filters)
        if base is None or len(base) < 2:
            return SamplerLogprobSemantics(None, None)
        temperature_applied = None
        base_gap = base[0] - base[1]
        if hot is not None and len(hot) >= 2 and base_gap > 0:
            temperature_applied = (hot[0] - hot[1]) / base_gap < _TEMPERATURE_APPLIED_GAP_RATIO_MAX
        nucleus_renormalized = None if nucleus is None else nucleus[0] - base[0] > _NUCLEUS_RENORM_MIN_SHIFT_NATS
        return SamplerLogprobSemantics(temperature_applied, nucleus_renormalized)

    def _open_group_rendezvous(self) -> tuple[str, RendezvousListener]:
        """:func:`open_group_rendezvous` for this client's server, group host and group port."""
        return open_group_rendezvous(
            engine=self.BACKEND_NAME,
            server_url=self.base_url,
            server_host=self.host,
            advertised_host=self.group_host,
            host_env=self.GROUP_HOST_ENV,
            port=self.group_port,
        )

    def _lift_pause(self, timeout: float, context: str) -> None:
        """Resume a quiesced engine, clearing ``_paused`` only once the server confirmed.

        An engine an interrupted sync left partly written is not resumed (:meth:`_mark_unusable`):
        that pause is deliberate, and every resume path, the close's own ``finally`` included, would
        otherwise put the partly-written model back on the wire.

        Does not raise on a server-side failure: both callers run on a path that is already failing or
        tearing down, where a dead server must not mask the real error. Sent once: the session retries
        POSTs with backoff and would block ~80s at interpreter exit.
        """
        if self._unusable_reason is not None:
            logger.error(f"Leaving {self.base_url} paused {context}: {self._unusable_reason}")
            return
        # Resolved outside the try: the handler below covers a dead server, not a subclass that
        # quiesces without declaring its route, which must fail by name.
        endpoint = self.RESUME_ENDPOINT
        try:
            self.request(
                "POST",
                endpoint,
                retry=False,
                timeout=timeout,
                **self.RESUME_REQUEST,
            )
            self._paused = False
        except Exception as e:  # advisory: a paused server wedges later rollouts, but so does raising
            logger.warning(f"Failed to resume paused {self.BACKEND_NAME} server {context}: {e}")

    def _raise_if_server_failed(self, error: Exception, server_call: AsyncCall | None, hint: str = "") -> None:
        """Re-raise ``error`` naming the server's own failure, when it recorded one.

        The local half of a sync fails downstream of whatever the server rejected, and that rejection
        sits on a daemon thread nobody joins. Returns without raising when the server recorded no
        error, so the caller re-raises the original.
        """
        server_error = server_call.join(_SERVER_ERROR_GRACE_S).error if server_call is not None else None
        if server_error is not None and server_error is not error:
            raise RuntimeError(
                f"{self.BACKEND_NAME} weight sync to {self.base_url} failed "
                f"({type(error).__name__}: {error}); the server's {server_call.name} call failed with: "
                f"{type(server_error).__name__}: {server_error}.{hint}"
            ) from server_error

    def _finalize_close(self) -> None:
        """Shared tail of ``close_communicator``: drop the pending snapshots and the exit hook.

        ``init_communicator`` makes the atexit registration; left in place it pins every retired
        client, and its resume POST, until interpreter exit.
        """
        self._reset_buffer_state()
        atexit.unregister(self.close_communicator)

    def _reset_buffer_state(self) -> None:
        """Initialize (or drop) the streamed-chunk state: the pending chunk, its size, and the update
        it belongs to. Kept in one place so a bare or reconnected client cannot start with half of it."""
        self._param_buffer: list[tuple[str, torch.Tensor]] = []
        self._buffered_bytes = 0
        self._co_load_groups = self.CO_LOADED_PARAM_GROUPS
        # A quiesced update this client opened mid-gather and has not closed yet.
        self._update_open = False
        # Chunks already on the wire in the open update. Non-zero means the engine is partly
        # rewritten, so no replay of this sync can put it back in a known state.
        self._chunks_sent = 0
        # Recorded after the newest staged copy; the flush waits on it, not on the whole device.
        self._staged_event: torch.cuda.Event | None = None

    @property
    def sync_device(self) -> torch.device | None:
        """The GPU the transport was formed on, or ``None`` before ``init_communicator`` ran."""
        return self._sync_device

    def _resolve_sync_device(self, device: torch.device | str | int) -> torch.device:
        """Record :func:`resolve_sync_device` of ``device`` as this client's :attr:`sync_device`."""
        self._sync_device = resolve_sync_device(device)
        return self._sync_device

    def _reset_transport_state(self) -> None:
        """Initialize the engine transport's fields to those of a client that has formed nothing yet."""
        raise NotImplementedError

    def init_communicator(self, device: torch.device | str | int = 0):
        raise NotImplementedError

    def begin_weight_update(self):
        """Quiesce the engine and open its weight-update phase.

        Must unwind its own quiesce if it raises: the caller has sent nothing yet, so the server is
        left serving rather than paused behind a half-opened phase.
        """
        raise NotImplementedError

    def _broadcast_chunk(self, chunk: list[tuple[str, torch.Tensor]], final: bool):
        """Declare and broadcast one non-empty chunk into the open phase, returning once the engine
        has it. ``final`` marks the last chunk, for an engine whose close rides on it."""
        raise NotImplementedError

    def send_weights(self, named_params: list[tuple[str, torch.Tensor]], final: bool = False) -> None:
        """Put one chunk on the wire, counting it and disowning the engine when it fails mid-update.

        Every chunk goes through here, mid-gather flushes and tail alike, so the count behind
        ``can_replay_sync`` and the decision that keeps a partly written engine off the wire are made
        once rather than per engine.
        """
        chunk = list(named_params)
        if not chunk:
            return
        try:
            self._broadcast_chunk(chunk, final)
        except BaseException:
            # An earlier chunk landed, so the engine holds neither policy whole. Marked here, before
            # the close's finally would resume it (see :meth:`_mark_unusable`).
            if self._chunks_sent:
                self._mark_unusable()
            raise
        self._chunks_sent += 1

    def end_weight_update(self, tail: list[tuple[str, torch.Tensor]]):
        """Send ``tail`` (possibly empty) as the final chunk, then close the phase and resume.

        Closes and resumes even when the tail fails: an engine left paused behind an open phase wedges
        every later rollout.
        """
        raise NotImplementedError

    def close_communicator(self):
        raise NotImplementedError

    def close_session(self) -> None:
        """Release the HTTP session: all a client that never formed a communicator holds."""
        self._session.close()

    def scope_co_load_groups(self, module_names: Iterable[str]) -> None:
        """Bind the co-load groups to the model whose modules ``module_names`` are, for every sync."""
        self._co_load_groups = declared_co_load_groups(self.CO_LOADED_PARAM_GROUPS, module_names)

    @property
    def buffered_bytes(self) -> int:
        """Bytes staged for the next chunk, the members held for a co-load partner included."""
        return self._buffered_bytes

    def buffer_param(self, name: str, snapshot: torch.Tensor):
        """Buffer a pre-made snapshot (from :func:`snapshot_to_sync_device`) by reference, for the next chunk.

        The snapshot may be shared across clients (a multi-server fan-out stages one copy per param),
        so it must not be mutated. This only buffers; the multi-server caller decides when the chunk
        goes out, which is what keeps the shared snapshots alive long enough for all of them.
        """
        self._param_buffer.append((name, snapshot))
        self._buffered_bytes += snapshot.nbytes
        if snapshot.is_cuda:
            # Re-recorded per param on the copy's own stream (this thread's current one on that
            # device), so the newest record covers every staged copy issued before it.
            if self._staged_event is None:
                self._staged_event = torch.cuda.Event()
            self._staged_event.record(torch.cuda.current_stream(snapshot.device))

    def update_named_param(self, name: str, weights: torch.Tensor):
        """Buffer one gathered param, streaming a chunk to the engine whenever the next would overflow.

        The snapshot lands on the sync GPU (``chunking`` says why not pinned host memory) and the chunk
        goes out mid-gather; the caller's rank-uniform failure guard covers a failure here.
        """
        # Checked per param, not once before the sync: the update opens with the first chunk, so a later
        # refusal lands mid-stream and the abort path handles the engine's state. Before init_communicator
        # there is no sync device and the copy stays on the source's.
        snapshot = snapshot_to_sync_device(
            name,
            weights,
            device=self.sync_device,
            buffered_bytes=self._buffered_bytes,
            budget=WEIGHT_SYNC_CHUNK_BYTES,
            flush=self.flush_chunk,
        )
        self.buffer_param(name, snapshot)

    def flush_chunk(self):
        """Send what is buffered as one chunk, leaving the update open for the chunks that follow.

        The members of a co-load group the buffer holds only part of stay buffered for the next
        chunk (:func:`split_co_loaded`); a buffer that is nothing but such members sends nothing.
        """
        chunk, held = split_co_loaded(self._param_buffer, self._co_load_groups)
        if not chunk:
            return
        # Completed before the quiesce, so a sticky CUDA error here leaves the engine serving rather
        # than paused behind a reload phase.
        self._complete_snapshots()
        self._open_update()
        self.send_weights(chunk)
        # Dropped only once on the wire: a chunk that failed while the sync is still replayable is
        # re-sent by the reconnect path, and dropping it first would lose those params.
        self._param_buffer = held
        self._buffered_bytes = sum(snapshot.nbytes for _, snapshot in held)

    def reset_prefix_cache(self):
        """Send the tail chunk, close the update and resume the engine; no-op if nothing was buffered.

        Name kept for TRL API compat (the engine's KV cache is invalidated as part of the update).
        """
        if not self._param_buffer and not self._update_open:
            return
        # A group still incomplete at the tail never completes: the engine would take the half and
        # drop it without error, so the update is refused rather than closed as landed.
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

    def _open_update(self):
        """Open the quiesced update once per sync; later chunks join the one already open."""
        if self._update_open:
            return
        if self._unusable_reason is not None:
            raise RuntimeError(self._unusable_reason)
        # Before the engine is touched: a failure to even open leaves nothing of this sync on the
        # wire, which is what makes it replayable.
        self._chunks_sent = 0
        self.begin_weight_update()
        self._update_open = True

    def _close_update(self, tail: list[tuple[str, torch.Tensor]]):
        """Send the final chunk and close the update, whatever it does.

        ``_chunks_sent`` is deliberately not reset: a close that failed after earlier chunks landed
        must still report this sync as unreplayable.
        """
        snapshot_error: Exception | None = None
        try:
            self._complete_snapshots()
        except Exception as e:
            # Inside the close rather than before it: the engine is already quiesced behind an open
            # reload only this close can end. The tail is dropped with it, since those copies may not
            # have landed.
            logger.error(f"Weight-sync snapshots did not complete for {self.base_url}: {e}")
            snapshot_error, tail = e, []
            if self._chunks_sent:
                # Dropping the tail after earlier chunks landed leaves the same mixed model a failed
                # chunk does, and the close below would resume the engine onto it.
                self._mark_unusable()
        try:
            self.end_weight_update(tail)
        finally:
            self._update_open = False
        if snapshot_error is not None:
            raise snapshot_error

    def _complete_snapshots(self) -> None:
        """Complete the async snapshot copies before a producer's own streams read them.

        A host wait on the event recorded after the newest copy, not a whole-device synchronize: the
        device also carries the previous chunk's broadcasts and, in a multi-server fan-out, every
        other server's, and waiting for those serialized each chunk's tail with the next one. The
        event is bound to the copies' own stream on the sync device, so it is correct from a worker
        thread whose current device and stream are its own (both are thread-local).
        """
        if self._staged_event is not None:
            self._staged_event.synchronize()

    def abort_weight_update(self):
        """Close an update the sync opened but cannot finish, and drop what is still buffered.

        Does not raise; it runs where the caller is already failing.

        With nothing on the wire the engine still holds the weights it was serving, so the phase is
        closed and the engine resumed; left quiesced behind an open reload it would reject every later
        sync and queue rollouts behind a paused scheduler.

        Once a chunk has gone out there is no such state to return to (see :meth:`_mark_unusable`).
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
        # Left paused with the phase open; only the local flag is cleared, so a second abort
        # (close_communicator, atexit) does not repeat the report.
        self._update_open = False
        self._mark_unusable()

    def _mark_unusable(self) -> None:
        """Reject every later sync on this client and leave its engine paused rather than serving.

        Idempotent: the failed send and the abort that follows both reach here, and the report is
        wanted once.

        A sync interrupted after part of the model went out leaves the engine holding neither the old
        policy nor the new one, and vLLM's layerwise reload materializes a layer whose tensors
        straddled a chunk boundary from uninitialized storage until the rest arrives, which never
        happens. The trainer keeps no copy of what already landed, so it cannot repair the engine.
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

    def _lift_pause_at_close(self) -> None:
        """Resume a server this client left paused at teardown (a deliberate pause survives it)."""
        if self._paused:
            self._lift_pause(HTTP_PROBE_TIMEOUT_S, context="at close_communicator")

    @property
    def can_replay_sync(self) -> bool:
        """Whether re-sending this client's buffer would restore a known state on its server.

        False once a chunk is on the wire: the trainer streams and keeps no copy of what already went
        out, so a reconnect plus re-flush would leave the fresh engine holding only the tail. A
        mid-stream failure therefore reaches the caller instead of being retried.
        """
        return self._chunks_sent == 0

    def drain_param_buffer(self) -> list[tuple[str, torch.Tensor]]:
        """Hand off the buffered params, clearing this client's buffer. The reconnect path uses this to
        move a failed client's pending snapshot references onto its replacement, so the references the
        retry flush needs survive."""
        buffered = self._param_buffer
        self._param_buffer = []
        self._buffered_bytes = 0
        return buffered
