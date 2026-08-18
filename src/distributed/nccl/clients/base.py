"""Engine-agnostic half of the weight-sync clients: HTTP plumbing, group addressing, host buffering.

Both supported rollout engines take the same shape — the trainer joins the engine's process group as
rank 0 and, while the engine is quiesced, streams the gathered parameters to it in byte-bounded
chunks. Both wire protocols are chunk-oriented (one quiesce, N declared payloads), so the host never
holds more than one chunk: buffering the whole model would cost ~1× model of pinned host RAM on the
forwarding rank — 800 GB at 400B, which no host has. Only the HTTP verbs and the broadcast transport
differ, and those live in the subclasses (``VLLMWeightSyncClient``, ``SGLangWeightSyncClient``).

The gather layer (``src/trainers/grpo/rollout/weight_sync.py``) drives whichever client it is
given through ``update_named_param`` + ``reset_prefix_cache``, so anything added here is available to
both engines without touching that layer.
"""

import atexit
import logging
import socket
import threading
import time
from urllib.parse import urlparse

import requests
import torch
from requests.adapters import HTTPAdapter
from torch import nn
from torch.distributed.tensor import DTensor
from urllib3.util.retry import Retry

from src.distributed.nccl.transport.packed_tensor import DEFAULT_PACKED_BUFFER_SIZE_BYTES
from src.env import env_str

logger = logging.getLogger(__name__)

# Host-side budget for one streamed chunk. The buffer is drained into the engine as soon as it is
# reached, so the forwarding rank's weight-sync host footprint is this plus the single largest
# tensor (one above the budget becomes its own chunk — both wire protocols describe whole tensors),
# instead of one full model. Matched to the packed transport's own staging buffer, which is what a
# chunk is re-packed into anyway.
WEIGHT_SYNC_CHUNK_BYTES = DEFAULT_PACKED_BUFFER_SIZE_BYTES

_LOOPBACK_HOSTS = {"127.0.0.1", "0.0.0.0", "::1", "localhost"}
# UDP "connect" target used only to make the kernel pick this host's outbound interface, whose local
# address is then its routable IP. No packet is ever sent, so the address just has to be off-link and
# non-loopback and is never contacted — an air-gapped node resolves it too (a missing route falls back
# to loopback in :func:`_get_ip`).
_OUTBOUND_ROUTE_PROBE = ("8.8.8.8", 80)

# Default per-attempt deadline for a control-plane POST. Applies to the quiesce/capability calls
# (`/pause`, `/pause_generation`, `/start_weight_update`, `/tokenize`) — the long ones name their own
# constant below.
_POST_TIMEOUT_S = 60

_HTTP_PROBE_TIMEOUT_S = 10
_GROUP_FORMATION_TIMEOUT_S = 120
_WEIGHT_UPDATE_TIMEOUT_S = 600
# Cleanup POSTs run on an already-failing path, where a long block only delays the real raise.
_CLEANUP_TIMEOUT_S = 30
_ASYNC_POLL_INTERVAL_S = 0.1
# Grace for a still-in-flight server call to record its error once the local side has already failed.
_SERVER_ERROR_GRACE_S = 5.0
# Cadence of the startup /health poll, which runs against a server that may still be loading weights.
_HEALTH_RETRY_INTERVAL_S = 2.0


def payload_bytes(tensor: torch.Tensor) -> int:
    """Wire size of one tensor — the unit every chunk budget below is counted in."""
    return tensor.numel() * tensor.element_size()


def starts_new_chunk(buffered_bytes: int, item_bytes: int, budget: int) -> bool:
    """Whether the next tensor has to open a NEW chunk: the budget is honored BEFORE it is exceeded.

    Cutting after it is reached instead would put the tensor that crossed the budget in the chunk
    that goes out — a budget's worth of extra pinned host memory on the trainer, and the same
    overshoot in the receive buffers an engine allocates for a declared chunk. A tensor larger than
    the budget still gets a chunk of its own: both wire protocols describe whole tensors.
    """
    return buffered_bytes > 0 and buffered_bytes + item_bytes > budget


def chunk_by_bytes(named_params: list[tuple[str, torch.Tensor]], budget: int) -> list[list[tuple[str, torch.Tensor]]]:
    """Split a whole payload into chunks, on the same boundary rule the streamed path buffers by.

    One rule for both entry points: a payload handed over in a single call is cut exactly where
    streaming it would have cut it, instead of being re-split by a second budget downstream.
    """
    chunks: list[list[tuple[str, torch.Tensor]]] = []
    current: list[tuple[str, torch.Tensor]] = []
    current_bytes = 0
    for name, param in named_params:
        item_bytes = payload_bytes(param)
        if starts_new_chunk(current_bytes, item_bytes, budget):
            chunks.append(current)
            current, current_bytes = [], 0
        current.append((name, param))
        current_bytes += item_bytes
    if current:
        chunks.append(current)
    return chunks


def _get_open_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def _get_ip() -> str:
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        try:
            s.connect(_OUTBOUND_ROUTE_PROBE)
            return s.getsockname()[0]
        except OSError:
            return "127.0.0.1"


def _is_loopback(host: str) -> bool:
    return host in _LOOPBACK_HOSTS or host.startswith("127.")


def _is_local_address(host: str) -> bool:
    """True when ``host`` is an address of THIS machine (bindable), catching the same-host-by-routable-IP case."""
    if _is_loopback(host):
        return True
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind((host, 0))
        return True
    except OSError:
        return False


def _host_snapshot(src: torch.Tensor, into: torch.Tensor | None = None) -> torch.Tensor:
    """Copy ``src`` (already detached and contiguous) to the host, reusing ``into`` when given.

    One home for the two decisions both snapshot paths must agree on: the buffer is page-locked
    exactly when the source is CUDA, and the D2H copy is async on the caller's stream — the flush
    synchronizes before reading.
    """
    host = torch.empty_like(src, device="cpu", pin_memory=src.is_cuda) if into is None else into
    host.copy_(src, non_blocking=src.is_cuda)
    return host


class _AsyncCall:
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

    def wait(self, timeout: float = _WEIGHT_UPDATE_TIMEOUT_S):
        self._thread.join(timeout=timeout)
        if self._thread.is_alive():  # join() returns silently on timeout — fail loud instead
            raise TimeoutError(f"{self.name} still blocked after {timeout:.0f}s — server wedged?")
        if self.error:
            raise self.error

    def join(self, grace: float) -> "_AsyncCall":
        """Wait up to ``grace`` seconds, then read ``.error`` / ``.result`` off the returned handle.

        Both readers run where the local side has already failed and ``_wait_for_calls`` raised the
        moment ANY call did: an in-flight call may still be about to record the server's own
        rejection (the diagnosis), or to hand back a resource — a formed group, a built communicator
        — that only this handle holds and that leaks unless the caller reclaims it.
        """
        self._thread.join(timeout=grace)
        return self


def _wait_for_calls(calls: list[_AsyncCall], timeout: float) -> None:
    """Block until every call returned, raising as soon as ANY of them fails.

    Polled rather than joined in sequence: the calls that set up the weight-transfer group only
    complete together (the engine's group-init request replies only once the group formed, and the
    trainer's own group creation returns only once the engine joined), so a serial join on either
    would park forever when the other has already failed.
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


def validate_syncable_param(name: str, param: torch.Tensor) -> None:
    """Reject params neither broadcast protocol can carry, while the server is still serving.

    A DTensor reports its GLOBAL ``numel()``/``shape`` against a LOCAL ``data_ptr()``, so a sharded
    param would be broadcast past the end of its shard. A 0-dim tensor has no last dimension to
    re-view as bytes, so it raises inside the vLLM producer — with the server already paused and
    mid-reload — and gives the SGLang consumer a shape it cannot allocate against.
    """
    if isinstance(param, DTensor):
        raise TypeError(
            f"Parameter {name!r} is a DTensor: the weight-sync client broadcasts raw local storage but "
            f"describes it with the global shape, so a sharded param would be read past its shard. "
            f"Materialize it first (materialize_dtensor / full_tensor)."
        )
    if param.ndim == 0:
        raise ValueError(
            f"Parameter {name!r} is 0-dimensional: the broadcast protocols re-view every tensor as bytes, "
            f"which needs a last dimension, and the consumer rebuilds it from the declared shape. "
            f"Give the parameter an explicit shape (e.g. (1,)) in the model."
        )


class PinnedHostBufferPool:
    """Recycled pinned host buffers for weight-sync snapshots, keyed by ``(shape, dtype)``.

    Page-locking fresh memory is slow, so a new pinned buffer per param per sync stalls the trainer
    for seconds at 20B+ scale — buffers are therefore pinned once and reused across syncs. They are
    reused WITHIN a sync too: :meth:`release` returns the chunk's buffers to the free list once that
    chunk is on the wire, so the pool holds one chunk's worth rather than one buffer per parameter
    (which is the whole model, permanently pinned).

    The free list is BOUNDED by that same budget; a buffer past it is dropped. Retention is per
    ``(shape, dtype)`` and one sync presents many shapes (per-expert tensors, dense layers, the
    embedding pair), so an unbounded list keeps the largest chunk seen *for every shape* — several
    budgets of non-swappable host RAM, plus a dedicated buffer per oversize tensor for the life of
    the process. A stream of like-shaped params still reuses everything.

    Reuse safety: a chunk's broadcast completes synchronously before its buffers are released, so a
    released buffer is provably not in flight. A snapshot is immutable only until the release that
    follows its chunk — consumers must not retain references past it.
    """

    def __init__(self, budget: int):
        self._budget = budget
        self._free: dict[tuple[tuple[int, ...], torch.dtype], list[torch.Tensor]] = {}
        self._free_bytes = 0
        self._checked_out: list[torch.Tensor] = []

    def snapshot(self, weights: torch.Tensor) -> torch.Tensor:
        """Copy ``weights`` into a pooled host buffer (pinned on first use for this shape/dtype)."""
        src = weights.detach().contiguous()
        available = self._free.get((tuple(src.shape), src.dtype))
        reused = available.pop() if available else None
        if reused is not None:
            self._free_bytes -= payload_bytes(reused)
        host = _host_snapshot(src, reused)
        self._checked_out.append(host)
        return host

    def release(self) -> None:
        """Return the checked-out buffers to the free list, dropping what the budget cannot hold.

        Call only once the chunk holding them has been fully broadcast.
        """
        for host in self._checked_out:
            host_bytes = payload_bytes(host)
            if self._free_bytes + host_bytes > self._budget:
                continue  # dropped rather than pinned for the process lifetime — see the class docstring
            self._free.setdefault((tuple(host.shape), host.dtype), []).append(host)
            self._free_bytes += host_bytes
        self._checked_out = []

    @property
    def retained_bytes(self) -> int:
        """Pinned host memory this pool is holding for reuse — bounded by its budget."""
        return self._free_bytes


class BaseWeightSyncClient:
    """HTTP session, group addressing and host-side parameter buffering shared by every engine.

    Subclasses implement the engine-specific half — ``init_communicator``, ``begin_weight_update``,
    ``_broadcast_chunk``, ``end_weight_update``, ``close_communicator`` — and declare:

    * ``BACKEND_KEY`` — the ``rollout_backend`` config value; the package builds its client registry
      from these, so a new engine needs no parallel table;
    * ``BACKEND_NAME`` — used verbatim in operator-facing errors and logs;
    * ``GROUP_HOST_ENV`` — the env var naming the trainer NIC the engine dials back to, per engine
      because the two servers can sit on different hosts;
    * ``RESUME_ENDPOINT`` / ``RESUME_PAYLOAD`` — the route (and body, where the engine's handler takes
      a request object) that lifts the sync quiesce, used by :meth:`_lift_pause` on both the
      post-sync and the teardown path;
    * ``EXPERT_LAYOUT`` (default ``"unfused"``) — which expert layout this engine's loader accepts;
    * ``SUPPORTS_EXPERT_PARALLEL`` (default ``True``) — whether this engine's weight sync survives
      alongside DeepEP in one trainer process, read by the gather layer.
    """

    BACKEND_KEY = ""
    BACKEND_NAME = "inference server"
    GROUP_HOST_ENV = ""
    # Route that lifts the weight-sync quiesce. Annotated, not defaulted: a client that quiesces
    # without declaring its own route must fail by name, not POST to the server root. RESUME_PAYLOAD
    # stays None for an engine whose handler takes no body; one parsing a request dataclass rejects
    # an empty body, so it declares {}.
    RESUME_ENDPOINT: str
    RESUME_PAYLOAD: dict | None = None
    # Which expert layout this engine's loader accepts. vLLM takes the per-expert tensors, SGLang the
    # fused pair transformers itself stores, and the gather has to produce whichever the receiver
    # wants — sending the other one is rejected on arrival, mid-update. Both spellings are named so
    # a client declares one of them and the gather can recognize BOTH: a bare `!=` would route an
    # unrecognized third layout into whichever branch it happens to miss.
    UNFUSED_EXPERT_LAYOUT = "unfused"
    FUSED_EXPERT_LAYOUT = "fused"
    EXPERT_LAYOUT = UNFUSED_EXPERT_LAYOUT
    # Whether this engine's weight sync survives alongside DeepEP in one trainer process. Declared
    # per client because the obstruction is the engine's transport, not anything about the model.
    SUPPORTS_EXPERT_PARALLEL = True
    # What an update interrupted mid-stream leaves the engine holding, quoted in the refusal that
    # follows one. Declared per client because it follows from the engine's own reload model.
    INTERRUPTED_UPDATE_STATE = "a model that is part old weights and part new"

    # The GPU this client's transport was formed on, set by ``init_communicator``. Named rather than
    # inferred: the flush runs on a worker thread when several servers sync at once, and torch's
    # current device and current stream are both thread-local.
    _sync_device: torch.device | None = None
    # Set once an interrupted sync left the engine unservable; every later sync is refused with it.
    _unusable_reason: str | None = None
    # Update-phase state, defaulted here as well as set in ``__init__``: every method that reads it
    # decides whether the engine is quiesced or partly rewritten, and must answer on ANY instance
    # rather than raise AttributeError. The pending chunk itself is NOT defaulted here — a mutable
    # class attribute would be shared by every client — so ``_reset_buffer_state`` stays its home.
    _paused = False
    _update_open = False
    _chunks_sent = 0
    _buffered_bytes = 0

    def __init__(
        self,
        base_url: str,
        group_port: int = 0,
        connection_timeout: float = 0.0,
        group_host: str | None = None,
    ):
        self.session = requests.Session()
        adapter = HTTPAdapter(
            max_retries=Retry(
                total=5,
                connect=5,
                read=5,
                status=3,
                status_forcelist=[500, 502, 503],
                backoff_factor=2,
                allowed_methods=["POST", "GET"],
            )
        )
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)

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
        self._reset_buffer_state()
        # Server quiesced for a weight sync; if the trainer dies before lifting it, close_communicator/atexit do.
        self._paused = False
        self.check_server(connection_timeout)

    def _post(self, path: str, timeout: float = _POST_TIMEOUT_S, **kwargs):
        resp = self.session.post(f"{self.base_url}{path}", timeout=timeout, **kwargs)
        resp.raise_for_status()
        return resp

    def _post_once(self, path: str, timeout: float = _POST_TIMEOUT_S, **kwargs):
        """Single-shot POST for non-idempotent blocking endpoints: a retry would queue a second collective RPC and wedge the engine."""
        resp = requests.post(f"{self.base_url}{path}", timeout=timeout, **kwargs)
        resp.raise_for_status()
        return resp

    def _post_if_supported(self, path: str, timeout: float = _POST_TIMEOUT_S, **kwargs) -> bool:
        """POST to an optional endpoint; treat 404 as "this server lacks it". Returns whether it exists."""
        resp = requests.post(f"{self.base_url}{path}", timeout=timeout, **kwargs)
        if resp.status_code == 404:
            return False
        resp.raise_for_status()
        return True

    def check_server(self, total_timeout: float = 0.0):
        start = time.time()
        last_error: Exception | None = None
        last_result = "no response"
        while True:
            try:
                # Bounded GET: a blackholed route would otherwise block past the outer deadline.
                status = requests.get(f"{self.base_url}/health", timeout=_HTTP_PROBE_TIMEOUT_S).status_code
                if status == 200:
                    logger.info(f"{self.BACKEND_NAME} server is up")
                    return
                last_result = f"HTTP {status}"
            except requests.exceptions.RequestException as exc:
                last_error, last_result = exc, f"{type(exc).__name__}: {exc}"
            # The deadline governs EVERY outcome: a route answering 503/401 forever is as dead as an
            # unreachable one, and only checking it on the exception path spins without bound.
            if time.time() - start >= total_timeout:
                raise ConnectionError(
                    f"{self.BACKEND_NAME} server unreachable at {self.base_url} after {total_timeout}s "
                    f"(last /health result: {last_result})"
                ) from last_error
            time.sleep(_HEALTH_RETRY_INTERVAL_S)

    def served_model_cards(self, timeout: float = _HTTP_PROBE_TIMEOUT_S) -> list[dict]:
        """The engine's ``/v1/models`` model cards. Raises when the server cannot be read."""
        resp = self.session.get(f"{self.base_url}/v1/models", timeout=timeout)
        resp.raise_for_status()
        return list(resp.json().get("data", []))

    def served_max_model_len(self) -> int | None:
        """Context window the served model declares, or ``None`` when no card carries one."""
        for card in self.served_model_cards():
            if card.get("max_model_len"):
                return int(card["max_model_len"])
        return None

    def _resolve_group_address(self) -> tuple[str, int]:
        """The address the engine's workers dial back to for the weight-transfer group.

        Resolution: explicit ``group_host`` → the backend's group-host env var → loopback when the
        server is on this machine → the default-route NIC. Raises when a remote server would be told
        to dial a loopback address, which forms no group and only times out.
        """
        master_port = self.group_port if self.group_port > 0 else _get_open_port()
        master_address = self.group_host or (env_str(self.GROUP_HOST_ENV) if self.GROUP_HOST_ENV else None)
        if not master_address:
            # Same-host groups MUST stay on loopback: firewalls drop hairpin traffic on the external NIC, wedging the first collective.
            master_address = "127.0.0.1" if _is_local_address(self.host) else _get_ip()

        if _is_loopback(master_address) and not _is_local_address(self.host):
            raise RuntimeError(
                f"{self.BACKEND_NAME} server {self.base_url} is on a remote host ({self.host}) but the "
                f"NCCL weight-sync group address resolved to loopback ({master_address}). The server's "
                f"workers would dial back to themselves and the group would never form. Set "
                f"{self.GROUP_HOST_ENV or 'the group host'} (or the per-server group_host) to the "
                f"trainer's routable IP on the subnet the server can reach."
            )
        return master_address, master_port

    def _lift_pause(self, timeout: float, context: str) -> None:
        """Resume a quiesced engine, clearing ``_paused`` only once the server confirmed.

        An engine an interrupted sync left partly written is NOT resumed (:meth:`_mark_unusable`):
        that pause is deliberate, and every resume path — the close's own ``finally`` included —
        would otherwise put exactly the model the mark exists to withhold back on the wire.

        Never raises *for a server-side failure*: both callers run on a path that is already failing
        or already tearing down, where a dead server must not mask the real error. ``_post_once``,
        not ``_post`` — the shared session retries POSTs with backoff, so against a dead server this
        would block ~80s at interpreter exit.
        """
        if self._unusable_reason is not None:
            logger.error(f"Leaving {self.base_url} paused {context}: {self._unusable_reason}")
            return
        # Resolved outside the try: the handler below is for a dead server, not for a subclass that
        # quiesces without declaring its route — that must fail by name, as RESUME_ENDPOINT promises.
        endpoint = self.RESUME_ENDPOINT
        try:
            self._post_once(
                endpoint,
                timeout=timeout,
                **({} if self.RESUME_PAYLOAD is None else {"json": self.RESUME_PAYLOAD}),
            )
            self._paused = False
        except Exception as e:  # advisory: a server left paused wedges later rollouts, but so does raising here
            logger.warning(f"Failed to resume paused {self.BACKEND_NAME} server {context}: {e}")

    def _raise_if_server_failed(
        self, error: Exception, server_call: "_AsyncCall | None", endpoint: str, hint: str = ""
    ) -> None:
        """Re-raise ``error`` naming the server's own failure, when it recorded one.

        The local half of a sync only ever fails downstream of whatever the server rejected, and that
        rejection lives on a daemon thread nobody joins — surface it or lose the diagnosis. Returns
        without raising when the server is blameless, so the caller re-raises the original.
        """
        server_error = server_call.join(_SERVER_ERROR_GRACE_S).error if server_call is not None else None
        if server_error is not None and server_error is not error:
            raise RuntimeError(
                f"{self.BACKEND_NAME} weight sync to {self.base_url} failed "
                f"({type(error).__name__}: {error}); the server's {endpoint} call failed with: "
                f"{type(server_error).__name__}: {server_error}.{hint}"
            ) from server_error

    def _finalize_close(self) -> None:
        """Shared tail of ``close_communicator``: drop the pending snapshots and the exit hook.

        The atexit registration is made by ``init_communicator``; left registered it pins every
        retired client — and its resume POST — until interpreter exit.
        """
        self._reset_buffer_state()
        atexit.unregister(self.close_communicator)

    def _reset_buffer_state(self) -> None:
        """Initialize (or drop) the streamed-chunk state: the pending chunk, its size, and the update
        it belongs to. One home, so a bare/reconnected client cannot start with half of it."""
        self._param_buffer: list[tuple[str, torch.Tensor]] = []
        self._buffered_bytes = 0
        # A quiesced update this client opened mid-gather and has not closed yet.
        self._update_open = False
        # Chunks already on the wire in the open update. Non-zero means the engine is partly
        # rewritten, so no replay of this sync can put it back in a known state.
        self._chunks_sent = 0

    @property
    def sync_device(self) -> torch.device | None:
        """The GPU the transport was formed on, or ``None`` before ``init_communicator`` ran."""
        return self._sync_device

    def _resolve_sync_device(self, device: torch.device | str | int) -> torch.device:
        """The explicit ``cuda:N`` form of ``device``, recorded as this client's :attr:`sync_device`.

        Explicit index: ``torch.device("cuda")`` compares unequal to ``cuda:0`` and carries no index
        for the D2H completion sync to name.
        """
        resolved = device if isinstance(device, torch.device) else torch.device(device)
        if resolved.type == "cuda" and resolved.index is None:
            resolved = torch.device("cuda", torch.cuda.current_device())
        self._sync_device = resolved
        return resolved

    def init_communicator(self, device: torch.device | str | int = 0):
        raise NotImplementedError

    def begin_weight_update(self):
        """Quiesce the engine and open its weight-update phase.

        Must unwind its own quiesce if it raises: the caller has sent nothing yet, so the server has
        to be left serving rather than paused behind a half-opened phase.
        """
        raise NotImplementedError

    def _broadcast_chunk(self, chunk: list[tuple[str, torch.Tensor]], final: bool):
        """Declare and broadcast ONE non-empty chunk into the open phase, returning once the engine
        has it. ``final`` marks the last chunk of the update, for an engine whose close rides it."""
        raise NotImplementedError

    def send_weights(self, named_params: list[tuple[str, torch.Tensor]], final: bool = False) -> None:
        """Put one chunk on the wire, counting it and disowning the engine when it fails mid-update.

        The single seam every chunk goes through — the mid-gather flushes and the tail alike — so
        the count that decides ``can_replay_sync`` and the verdict that keeps a partly written
        engine off the wire are taken in one place rather than per engine.
        """
        chunk = list(named_params)
        if not chunk:
            return
        try:
            self._broadcast_chunk(chunk, final)
        except BaseException:
            # An earlier chunk landed, so the engine now holds neither policy whole. Marked HERE,
            # before the close's own finally resumes it — see :meth:`_mark_unusable`.
            if self._chunks_sent:
                self._mark_unusable()
            raise
        self._chunks_sent += 1

    def end_weight_update(self, tail: list[tuple[str, torch.Tensor]]):
        """Send ``tail`` (possibly empty) as the final chunk, then close the phase and resume.

        Closes and resumes even when the tail fails — an engine left paused behind an open phase
        wedges every later rollout.
        """
        raise NotImplementedError

    def close_communicator(self):
        raise NotImplementedError

    def sync_model_weights(self, named_params: list[tuple[str, torch.Tensor]]):
        """Push a whole payload inside one quiesce: open the phase, stream it in chunks, close.

        The single-call form of the streamed path below, for callers that already hold the entire
        payload (``update_model_params``, the reconnect replay). Same phases and the same byte
        budget — declaring a whole model in one request has the engine allocate receive buffers for
        all of it before the first tensor arrives.
        """
        named_params = list(named_params)
        if not named_params:
            return
        # Before the quiesce: this caller holds the whole payload, so a param neither protocol can
        # carry fails while the server is still serving rather than once it is paused.
        for name, param in named_params:
            validate_syncable_param(name, param)
        chunks = chunk_by_bytes(named_params, WEIGHT_SYNC_CHUNK_BYTES)
        self._open_update()
        try:
            for chunk in chunks[:-1]:
                self.send_weights(chunk)
        except BaseException:
            # Only the close ends the phase, and a chunk that failed mid-payload never reaches it.
            self.abort_weight_update()
            raise
        self._close_update(chunks[-1])

    def update_model_params(self, model: nn.Module):
        """Sync every model param in one quiesced update."""
        self.sync_model_weights([(n, p.data) for n, p in model.named_parameters()])

    @staticmethod
    def snapshot_to_host(weights: torch.Tensor) -> torch.Tensor:
        """One host snapshot of ``weights`` (pinned when the source is CUDA), copy-not-alias.

        The snapshot is read-only after creation, so one can be shared across several clients'
        buffers (see ``buffer_host_param``). The D2H copy is async on the caller's stream;
        ``reset_prefix_cache`` synchronizes before reading.
        """
        return _host_snapshot(weights.detach().contiguous())

    def buffer_host_param(self, name: str, host: torch.Tensor):
        """Buffer a pre-made host snapshot (from ``snapshot_to_host``) by reference, for the next chunk.

        The snapshot may be shared across clients (a multi-server fan-out buffers ONE pinned copy
        per param), so it must never be mutated, and the sharer must not recycle it until every
        client has sent the chunk holding it. Buffering only — the multi-server caller decides when
        the chunk goes out, because that decision is what keeps the shared snapshots alive long
        enough for all of them.
        """
        if host.device.type != "cpu":
            raise ValueError(
                f"buffer_host_param expects a CPU host snapshot for {name!r}, got {host.device} — "
                f"buffering on-device tensors holds a full model copy until the flush. "
                f"Use snapshot_to_host()."
            )
        self._param_buffer.append((name, host))
        self._buffered_bytes += payload_bytes(host)

    def update_named_param(self, name: str, weights: torch.Tensor):
        """Buffer one gathered param, streaming a chunk to the engine whenever the next would overflow.

        Buffered on CPU (pinned when the source is CUDA): buffering on GPU would hold a full model
        copy on the forwarding rank. The host copy is also a snapshot, so a later in-place mutation
        (e.g. PEFT unmerge) cannot revert a buffered weight before it is sent. The chunk goes out
        mid-gather, which is what bounds the host footprint — the caller's rank-uniform failure
        guard is what keeps a failure here from stranding its peers in the next gather.
        """
        # On the source, before the snapshot: a param neither protocol can carry is refused here. It
        # cannot be refused before the quiesce — this path opens the update as soon as the first
        # chunk fills, so a later refusal lands mid-stream and the abort path owns the engine's state.
        validate_syncable_param(name, weights)
        host = self.snapshot_to_host(weights)
        if starts_new_chunk(self._buffered_bytes, payload_bytes(host), WEIGHT_SYNC_CHUNK_BYTES):
            self.flush_chunk()
        self.buffer_host_param(name, host)

    def flush_chunk(self):
        """Send what is buffered as one chunk, leaving the update open for the chunks that follow."""
        if not self._param_buffer:
            return
        # Completed BEFORE the quiesce: a sticky CUDA error here then leaves the engine serving
        # rather than paused behind a reload phase it never asked for.
        self._complete_host_snapshots()
        self._open_update()
        self.send_weights(self._param_buffer)
        # Dropped only once it is on the wire: a chunk that failed while the sync is still replayable
        # is re-sent by the reconnect path, and draining first would silently lose those params.
        self.drain_param_buffer()

    def reset_prefix_cache(self):
        """Send the tail chunk, close the update and resume the engine; no-op if nothing was buffered.

        Name kept for TRL API compat (the engine's KV cache is invalidated as part of the update).
        """
        if not self._param_buffer and not self._update_open:
            return
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

        ``_chunks_sent`` survives on purpose: a close that failed after earlier chunks landed must
        still report this sync as unreplayable.
        """
        snapshot_error: Exception | None = None
        try:
            self._complete_host_snapshots()
        except Exception as e:
            # Inside the close, not before it: the engine is already quiesced behind an open reload
            # that only this close can end, and a sticky CUDA error here is exactly the state a
            # failing sync is in. The tail is dropped with it — copies that may not have landed.
            logger.error(f"Weight-sync host snapshots did not complete for {self.base_url}: {e}")
            snapshot_error, tail = e, []
            if self._chunks_sent:
                # Dropping the tail after earlier chunks landed leaves the same mixed model a failed
                # chunk does, and the close below would otherwise resume the engine onto it.
                self._mark_unusable()
        try:
            self.end_weight_update(tail)
        finally:
            self._update_open = False
        if snapshot_error is not None:
            raise snapshot_error

    def _complete_host_snapshots(self) -> None:
        """Complete the async D2H snapshot copies before a producer reads that host memory.

        Bound to :attr:`sync_device`, and a whole-device synchronize: the flush runs on a worker
        thread once more than one server is synced at a time, and torch's current device and current
        stream are both thread-local — a fresh worker would complete device 0's copies while this
        rank's are still in flight, and the engine would be broadcast whatever the buffers hold.
        """
        device = self.sync_device
        if device is not None and device.type == "cuda":
            torch.cuda.synchronize(device)

    def abort_weight_update(self):
        """Close an update the sync opened but cannot finish, and drop what is still buffered.

        Never raises — it runs where the caller is already failing.

        With nothing on the wire yet the engine still holds the weights it was serving, so the phase
        is closed and the engine resumed: left quiesced behind an open reload it would refuse every
        later sync and queue every rollout behind a paused scheduler.

        Once a chunk HAS gone out there is no such state to return to — see :meth:`_mark_unusable`.
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
        # Deliberately left paused, with the phase open: only the local flag is cleared, so a second
        # abort (close_communicator, atexit) does not repeat the report.
        self._update_open = False
        self._mark_unusable()

    def _mark_unusable(self) -> None:
        """Refuse every later sync on this client, and leave its engine paused rather than serving.

        Idempotent: the send that failed and the abort that follows it both reach here, and the
        operator needs the report once.

        A sync interrupted after part of the model went out leaves the engine holding neither the
        old policy nor the new one, and vLLM's layerwise reload materializes a layer whose tensors
        straddled a chunk boundary from UNINITIALIZED storage until the rest arrives — which now
        never happens. Resuming that engine serves it; nothing on the trainer side can repair it,
        because the trainer keeps no copy of what already landed.
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
            self._lift_pause(_HTTP_PROBE_TIMEOUT_S, context="at close_communicator")

    @property
    def can_replay_sync(self) -> bool:
        """Whether re-sending this client's buffer would restore a known state on its server.

        False once a chunk is on the wire: the trainer streams and keeps no copy of what already
        went out, so a reconnect + re-flush would leave the fresh engine holding the tail alone.
        A mid-stream failure has to reach the caller instead of being retried into a half-updated
        engine that answers /health.
        """
        return self._chunks_sent == 0

    def drain_param_buffer(self) -> list[tuple[str, torch.Tensor]]:
        """Hand off the buffered params, clearing this client's buffer. The reconnect path moves a
        failed client's pending snapshot references onto its replacement with this — clearing first
        would drop the very references the retry flush needs."""
        buffered = self._param_buffer
        self._param_buffer = []
        self._buffered_bytes = 0
        return buffered
