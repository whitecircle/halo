"""VLLMWeightSyncClient — NCCL weight sync + HTTP generation for vLLM. Trainer is rank 0, workers rank 1+; weights sent via packed NCCL broadcast. No vllm package needed."""

import atexit
import errno
import logging

import requests
import torch

from src.distributed.nccl.clients.base import (
    _CLEANUP_TIMEOUT_S,
    _GROUP_FORMATION_TIMEOUT_S,
    _HTTP_PROBE_TIMEOUT_S,
    _SERVER_ERROR_GRACE_S,
    _WEIGHT_UPDATE_TIMEOUT_S,
    BaseWeightSyncClient,
    _AsyncCall,
    _wait_for_calls,
)
from src.distributed.nccl.transport.packed_tensor import (
    DEFAULT_PACKED_BUFFER_SIZE_BYTES,
    DEFAULT_PACKED_NUM_BUFFERS,
    packed_broadcast_producer,
)
from src.distributed.nccl.transport.pynccl import PyNcclCommunicator
from src.distributed.nccl.transport.stateless_group import StatelessProcessGroup
from src.env import env_positive_float, resolve_nccl_timeout_minutes, watchdog_bounded_seconds

logger = logging.getLogger(__name__)

# Server control-plane routes reached from more than one call site (the call itself, its async-call
# label, the error text that names it). Spelled once so a route rename cannot leave a stale label
# pointing at an endpoint that no longer exists.
_EP_INIT_ENGINE = "/init_weight_transfer_engine"
_EP_UPDATE_WEIGHTS = "/update_weights"
_EP_FINISH_UPDATE = "/finish_weight_update"
_EP_RESUME = "/resume"

# Deadline for the 1-token liveness probe. Long enough that a merely busy scheduler still answers,
# short enough that a wedged one is reported instead of parking the trainer for the sync deadline.
_GENERATION_PROBE_TIMEOUT_S = 60.0


def _build_sampling_params(
    n: int = 1,
    temperature: float = 1.0,
    top_p: float = 1.0,
    top_k: int = 0,
    min_p: float = 0.0,
    max_tokens: int = 16,
    repetition_penalty: float = 1.0,
    logprobs: int = 1,
    extra: dict | None = None,
) -> dict:
    params = {"n": n, "temperature": temperature, "top_p": top_p, "max_tokens": max_tokens, "logprobs": logprobs}
    if repetition_penalty != 1.0:
        params["repetition_penalty"] = repetition_penalty
    if top_k > 0:
        params["top_k"] = top_k
    if min_p > 0.0:
        params["min_p"] = min_p
    if extra:
        params.update(extra)
    return params


class VLLMWeightSyncClient(BaseWeightSyncClient):
    """NCCL weight sync client for vLLM servers."""

    BACKEND_KEY = "vllm"
    BACKEND_NAME = "vLLM"
    GROUP_HOST_ENV = "VLLM_GROUP_HOST"
    RESUME_ENDPOINT = _EP_RESUME
    # The layerwise reload processes a layer once all of its tensors arrived; one whose tensors
    # straddled the interrupted chunk boundary is materialized from uninitialized storage while it
    # waits for the rest, and that storage is live once the phase closes.
    INTERRUPTED_UPDATE_STATE = (
        "a model that is part old weights and part new, with any layer that straddled the last chunk "
        "boundary materialized from UNINITIALIZED storage by the layerwise reload"
    )

    def __init__(
        self,
        base_url: str,
        group_port: int = 0,
        connection_timeout: float = 0.0,
        group_host: str | None = None,
    ):
        self.communicator: PyNcclCommunicator | None = None
        # Kept only so close_communicator can release its listener: the group carries the NCCL id at setup and nothing after.
        self._process_group: StatelessProcessGroup | None = None
        # Persistent producer streams: per-call streams strand every sync's pack allocations in
        # unreachable per-stream allocator pools (see packed_broadcast_producer). Lazily created on
        # the communicator's device; a client's flushes are serialized, so reuse is race-free.
        self._packed_streams: list[torch.cuda.Stream] | None = None
        # A layerwise reload opened server-side (possibly by a request whose reply was lost), which
        # only /finish_weight_update can end — every later sync fails against it otherwise.
        self._phase_started = False
        # Single-shot: a retrying-session read-timeout would re-queue a full generation server-side.
        # Derived, not a literal: the call blocks every peer at the next collective, so it has to stay
        # under the watchdog that would abort them — including when the watchdog is lowered.
        self._generation_timeout = env_positive_float(
            "HALO_VLLM_GENERATION_TIMEOUT_SECONDS", watchdog_bounded_seconds()
        )
        watchdog = resolve_nccl_timeout_minutes() * 60
        if self._generation_timeout >= watchdog:
            logger.warning(
                f"HALO_VLLM_GENERATION_TIMEOUT_SECONDS={self._generation_timeout:.0f}s is at or above the "
                f"NCCL collective watchdog ({watchdog:.0f}s): a slow generation holds every peer at the "
                f"next collective until the watchdog aborts the run instead. Lower it, or raise "
                f"DIST_NCCL_TIMEOUT_MINUTES."
            )
        super().__init__(
            base_url=base_url,
            group_port=group_port,
            connection_timeout=connection_timeout,
            group_host=group_host,
        )

    def server_reports_paused(self) -> bool | None:
        """Server-side pause state via ``GET /is_paused``, or ``None`` when the server does not report one."""
        try:
            resp = requests.get(f"{self.base_url}/is_paused", timeout=_HTTP_PROBE_TIMEOUT_S)
            if resp.status_code != 200:
                return None
            return bool(resp.json()["is_paused"])
        except (requests.exceptions.RequestException, ValueError, KeyError) as e:
            logger.debug(f"Could not read pause state from {self.base_url}: {e}")
            return None

    def probe_generation(self):
        """Bounded 1-token probe to fail fast when /health is up but generation is wedged.

        Only a HANG means wedged: the completion below is a plain request, so a timeout is the
        verdict rather than something the session would retry away.
        """
        # A paused engine QUEUES new requests instead of rejecting them, so the probe below would only
        # time out and blame a wedged scheduler. Ask the server before guessing.
        if self.server_reports_paused():
            raise RuntimeError(
                f"vLLM server {self.base_url} is PAUSED for a weight update (GET /is_paused) — generation "
                f"requests would queue instead of running. A trainer died mid weight sync without lifting "
                f"its /pause. Lift it (POST {self.base_url}{_EP_RESUME}) before reconnecting — unless that "
                f"sync had already streamed part of the model, in which case the server holds a partly "
                f"written model and has to be RESTARTED instead (its log names the last update)."
            )
        try:
            model_id = self.served_model_cards()[0]["id"]
        except Exception as e:
            logger.warning(f"probe_generation: could not resolve served model ({e}); skipping probe")
            return
        try:
            requests.post(
                f"{self.base_url}/v1/completions",
                json={"model": model_id, "prompt": "ping", "max_tokens": 1, "temperature": 0.0},
                timeout=_GENERATION_PROBE_TIMEOUT_S,
            )
        except requests.exceptions.Timeout as e:
            raise RuntimeError(
                f"vLLM server {self.base_url} answers /health but GENERATION is wedged (no response "
                f"in {_GENERATION_PROBE_TIMEOUT_S:.0f}s to a 1-token probe). A previous trainer most "
                f"likely died mid-run while attached to the weight-transfer engine (SIGKILL skips cleanup), leaving the "
                f"server's scheduler stuck. RESTART the vLLM server container to recover."
            ) from e
        except requests.exceptions.RequestException as e:
            # Responsive but rejected the probe (e.g. raw /v1/completions disabled) — not wedged.
            logger.warning(f"probe_generation: server responded with an error (not wedged): {e}")

    def init_communicator(self, device: torch.device | str | int = 0):
        """Init NCCL group. Trainer=rank 0, vLLM workers=rank 1+."""
        device = self._resolve_sync_device(device)
        # Detect a paused or generation-wedged server before forming the group or running rollouts that would only hang.
        self.probe_generation()
        resp = requests.get(f"{self.base_url}/get_world_size", timeout=_HTTP_PROBE_TIMEOUT_S)
        resp.raise_for_status()
        inference_ws = resp.json()["world_size"]
        world_size = inference_ws + 1
        master_address, master_port = self._resolve_group_address()

        logger.info(f"NCCL init: inference_ws={inference_ws}, master={master_address}:{master_port}")

        server_call = _AsyncCall(
            name=_EP_INIT_ENGINE,
            fn=lambda: self._post_once(
                _EP_INIT_ENGINE,
                timeout=_GROUP_FORMATION_TIMEOUT_S,
                json={
                    "init_info": {
                        "master_address": master_address,
                        "master_port": master_port,
                        "rank_offset": 1,
                        "world_size": world_size,
                    }
                },
            ),
        )

        pg: StatelessProcessGroup | None = None
        comm_call: _AsyncCall | None = None
        try:
            # Bind the listener to all interfaces so a multi-homed trainer accepts the connection regardless of NIC.
            pg = StatelessProcessGroup.create(
                host=master_address,
                port=master_port,
                rank=0,
                world_size=world_size,
                bind_host="0.0.0.0",
            )
            # ncclCommInitRank is unconditionally blocking (the wrapper binds no non-blocking init and
            # NCCL has no comm-init deadline), so a server that never joins would park the trainer for
            # good with its HTTP error unread. Both sides run concurrently; whichever fails first wins.
            comm_call = _AsyncCall(
                name="NCCL group formation",
                fn=lambda: PyNcclCommunicator(pg, device=device),
            )
            _wait_for_calls([server_call, comm_call], timeout=_GROUP_FORMATION_TIMEOUT_S)
            self.communicator = comm_call.result
        except Exception as e:
            # `_wait_for_calls` raises the moment EITHER half fails, so a communicator this rank
            # already built is still unowned — it lives only on the call handle, and dropping the
            # reference leaks its ncclComm (and the device memory NCCL pinned for it) for the life
            # of the process. Abort it before releasing the group.
            built = comm_call.join(_SERVER_ERROR_GRACE_S).result if comm_call is not None else None
            if built is not None:
                built.abort()
            if pg is not None:
                # The comm thread stays parked in ncclCommInitRank holding this group, so dropping
                # references would never free the port; close() releases the listener from the group itself.
                pg.close()
            if e is server_call.error:
                raise  # the server named its own failure; a network-topology hint would only mislead
            if isinstance(e, OSError) and e.errno == errno.EADDRINUSE:
                # Not a topology problem: the hint below would send the caller chasing VLLM_GROUP_HOST
                # instead of the port another live group already holds.
                raise RuntimeError(
                    f"Could not bind the weight-transfer group port {master_port} on this host ({e}). "
                    f"A connected client holds its group port until close_communicator(), so every vLLM "
                    f"server — and every trainer process sharing this host — needs its own "
                    f"vllm_group_port / group_port."
                ) from e
            # A wrong-but-routable master surfaces as a bare TCPStore/handshake timeout — name the knob.
            raise RuntimeError(
                f"Weight-transfer group formation failed (advertised master "
                f"{master_address}:{master_port}; vLLM server {self.base_url}). If the vLLM node "
                f"cannot reach that address, set VLLM_GROUP_HOST to an interface routable from the "
                f"vLLM server (and NCCL_SOCKET_IFNAME for the data plane on multi-homed hosts). "
                f"Original error: {type(e).__name__}: {e}"
            ) from e

        self._process_group = pg
        atexit.register(self.close_communicator)
        logger.info("NCCL weight transfer initialized")

    def _require_communicator(self) -> PyNcclCommunicator:
        """The live communicator, or a loud refusal — an aborted one broadcasts into nothing."""
        if self.communicator is None:
            raise RuntimeError("Call init_communicator() first")
        if self.communicator.aborted:
            raise RuntimeError(
                f"The weight-sync NCCL communicator for {self.base_url} was aborted by a failed "
                f"collective and cannot be reused — reconnect the client before syncing again."
            )
        return self.communicator

    def begin_weight_update(self):
        """Pause the engine and open its layerwise reload phase.

        Unwinds itself on failure: nothing has been broadcast yet, so a server that ends up paused
        behind a half-opened phase would only wedge the next rollout round.
        """
        self._require_communicator()
        # ``_paused`` means "the server MAY be paused": vLLM pauses the engine BEFORE writing the reply,
        # so a lost reply strands it paused unless the flag already covers the in-flight request.
        self._paused = True
        try:
            self._post("/pause")
            # Set BEFORE the POST, same reason as the pause flag: a lost reply still leaves the
            # layerwise reload open server-side, and only the close can end it.
            self._phase_started = True
            # Single-shot: /start_weight_update is a collective RPC, and a session retry would issue a second one.
            self._post_once("/start_weight_update")
        except Exception:
            self._close_phase_and_resume()
            raise

    def _broadcast_chunk(self, named_params: list[tuple[str, torch.Tensor]], final: bool = False):
        """Declare one chunk over HTTP, then broadcast it packed into the open phase.

        ``/update_weights`` takes one chunk of an open session (``start_weight_update`` … N chunks …
        ``finish_weight_update``), so the trainer streams the model through this rather than
        declaring it all at once — which would mean holding the whole model in pinned host memory.
        ``final`` is unused: this engine's close is its own ``/finish_weight_update`` call.
        """
        del final
        # The payload was validated where it entered the client (before any quiesce); here it is only
        # described, in the order the consumer will unpack it.
        names, dtype_names, shapes = [], [], []
        for name, param in named_params:
            names.append(name)
            dtype_names.append(str(param.dtype).split(".")[-1])
            shapes.append(list(param.shape))

        server_call: _AsyncCall | None = None
        try:
            server_call = _AsyncCall(
                name=_EP_UPDATE_WEIGHTS,
                fn=lambda: self._post_once(
                    _EP_UPDATE_WEIGHTS,
                    timeout=_WEIGHT_UPDATE_TIMEOUT_S,
                    json={
                        "update_info": {
                            "names": names,
                            "dtype_names": dtype_names,
                            "shapes": shapes,
                            "packed": True,
                            # On the wire rather than implied: the consumer derives its chunk boundaries
                            # from these, and a silent mismatch desynchronizes every broadcast count.
                            "packed_buffer_size_bytes": DEFAULT_PACKED_BUFFER_SIZE_BYTES,
                            "packed_num_buffers": DEFAULT_PACKED_NUM_BUFFERS,
                        }
                    },
                ),
            )

            # post_iter_func re-uploads CPU-buffered tensors pack by pack, so at most one pack transits the GPU.
            communicator = self._require_communicator()
            device = communicator.device
            if self._packed_streams is None and torch.device(device).type == "cuda":
                with torch.cuda.device(device):
                    self._packed_streams = [torch.cuda.Stream() for _ in range(DEFAULT_PACKED_NUM_BUFFERS)]
            packed_broadcast_producer(
                iterator=iter(named_params),
                group=communicator,
                src=0,
                post_iter_func=lambda item: item[1].to(device, non_blocking=True),
                streams=self._packed_streams,
            )

            server_call.wait(timeout=_WEIGHT_UPDATE_TIMEOUT_S)
        except Exception as e:
            self._raise_if_server_failed(e, server_call, _EP_UPDATE_WEIGHTS)
            raise
        logger.debug(f"Streamed {len(named_params)} params to vLLM")

    def end_weight_update(self, tail: list[tuple[str, torch.Tensor]]):
        """Send the final chunk, close the layerwise reload and resume — closing on every path."""
        try:
            self.send_weights(tail, final=True)
            self._post_once(_EP_FINISH_UPDATE, timeout=_WEIGHT_UPDATE_TIMEOUT_S)
            self._phase_started = False
        finally:
            self._close_phase_and_resume()

    def _close_phase_and_resume(self):
        """Close a started-but-unfinished phase and lift the pause. Never raises — it runs where the
        caller is already failing, and a phase left open wedges every later sync."""
        if self._phase_started:
            try:
                self._post_once(_EP_FINISH_UPDATE, timeout=_CLEANUP_TIMEOUT_S)
            except Exception as e:
                logger.warning(f"Failed to close interrupted weight update: {e}")
            self._phase_started = False
        # Always lift the pause — a server left paused wedges every later rollout.
        self._lift_pause(_CLEANUP_TIMEOUT_S, context="after sync")

    def close_communicator(self):
        # Close an update an interrupted sync left open, and lift any pause with it, so generation is
        # not wedged after the trainer exits — unless the abort left the engine unservable, which
        # keeps its pause on purpose.
        self.abort_weight_update()
        self._lift_pause_at_close()
        if self.communicator is not None:
            # ncclCommAbort rather than a dropped reference: PyNcclCommunicator has no __del__, so
            # the communicator — and the device memory NCCL pinned for it — would outlive every
            # client this process retires, one leak per reconnect and per served endpoint.
            self.communicator.abort()
        self.communicator = None
        self._packed_streams = None
        if self._process_group is not None:
            # Frees group_port for the next client on this host; the group only ever carried the NCCL id.
            self._process_group.close()
            self._process_group = None
        self._finalize_close()

    def _tokenize_text(self, text: str) -> list[int]:
        return self._post("/tokenize", json={"prompt": text}).json()["tokens"]

    def _completion_ids(self, choice: dict) -> list[int]:
        """The sampled token IDs the server returned for one choice.

        Never reconstructed from the decoded text: a BPE re-merge does not have to reproduce the
        sampled stream, and ids that disagree with the logprobs desync the GRPO importance ratio
        silently. A missing field is a server that ignored ``return_token_ids``, so it raises.
        """
        token_ids = choice.get("token_ids")
        if token_ids is None:
            raise RuntimeError(
                f"vLLM server {self.base_url} returned a completion without `token_ids` even though "
                f"the request set `return_token_ids`. Recovering the ids by re-tokenizing the text "
                f"would silently desync the GRPO importance-sampling ratio, so the rollout is "
                f"refused — serve this model on vLLM >= 0.10.2, which honors the flag."
            )
        return list(token_ids)

    def generate(
        self,
        prompts: list[str] | list[list[int]],
        images: list | None = None,
        n: int = 1,
        repetition_penalty: float = 1.0,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int = 0,
        min_p: float = 0.0,
        max_tokens: int = 16,
        logprobs: int | None = 0,
        structured_outputs_regex: str | None = None,
        generation_kwargs: dict | None = None,
        **kwargs,
    ) -> dict[str, list]:
        """Generate completions via native vLLM ``/v1/completions``.

        Matches TRL's ``VLLMClient.generate`` contract, returning ``{prompt_ids, completion_ids,
        logprobs, logprob_token_ids}``. Only the sampled token's logprob is available here (num_logprobs=1).
        """
        if images is not None and any(image is not None for image in images):
            raise NotImplementedError(
                "This client's /v1/completions path is text-only — VLM online GRPO with images "
                "is unsupported (generations would silently be image-blind)."
            )
        # /v1/completions needs logprobs >= 1 to populate token_logprobs; GRPO's default is 0, so floor at 1.
        sampling = _build_sampling_params(
            n=n,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            min_p=min_p,
            max_tokens=max_tokens,
            repetition_penalty=repetition_penalty,
            logprobs=max(int(logprobs or 0), 1),
            extra=generation_kwargs,
        )
        sampling["return_token_ids"] = True
        if structured_outputs_regex:
            sampling["guided_regex"] = structured_outputs_regex

        prompt_ids: list[list[int]] = []
        all_completion_ids: list[list[int]] = []
        all_logprobs: list[list[list[float]]] = []
        all_logprob_token_ids: list[list[list[int]]] = []

        for prompt in prompts:
            is_token_ids = isinstance(prompt, (list, tuple))
            prompt_ids.append(list(prompt) if is_token_ids else self._tokenize_text(prompt))

        if not prompts:
            return {"prompt_ids": [], "completion_ids": [], "logprobs": [], "logprob_token_ids": []}

        # ONE batched request: continuous batching schedules all prompts together (wall-clock ~max, not sum).
        data = self._post_once(
            "/v1/completions", timeout=self._generation_timeout, json={"prompt": list(prompts), **sampling}
        ).json()
        for choice in sorted(data["choices"], key=lambda c: c.get("index", 0)):
            token_logprobs = (choice.get("logprobs") or {}).get("token_logprobs") or []
            completion_ids = self._completion_ids(choice)
            all_completion_ids.append(completion_ids)
            all_logprobs.append([[v] for v in token_logprobs])
            all_logprob_token_ids.append([[tid] for tid in completion_ids[: len(token_logprobs)]])

        return {
            "prompt_ids": prompt_ids,
            "completion_ids": all_completion_ids,
            "logprobs": all_logprobs,
            "logprob_token_ids": all_logprob_token_ids,
        }
