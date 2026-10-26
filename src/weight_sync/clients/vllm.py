"""NCCL weight sync for vLLM servers.

The trainer joins as rank 0 and the vLLM workers as rank 1+; weights go over a packed NCCL broadcast.
The vllm package is never imported here.
"""

import atexit
import logging

import requests
import torch

from src.weight_sync.chunking import describe_chunk
from src.weight_sync.clients.base import (
    _SERVER_ERROR_GRACE_S,
    CLEANUP_TIMEOUT_S,
    GROUP_FORMATION_TIMEOUT_S,
    HTTP_PROBE_TIMEOUT_S,
    POST_TIMEOUT_S,
    WEIGHT_UPDATE_TIMEOUT_S,
    AsyncCall,
    WeightSyncClientBase,
    _wait_for_calls,
    unregistered_bailing_model_types,
)
from src.weight_sync.transport.packed_broadcast import (
    PACKED_BUFFER_BYTES,
    PACKED_NUM_BUFFERS,
    packed_broadcast_producer,
)
from src.weight_sync.transport.pynccl import PyNcclCommunicator, UniqueIdStore

logger = logging.getLogger(__name__)

# Deadline for the 1-token liveness probe. Long enough that a merely busy scheduler still answers,
# short enough that a wedged one is reported instead of parking the trainer for the sync deadline.
_GENERATION_PROBE_TIMEOUT_S = 60.0


class VLLMWeightSyncClient(WeightSyncClientBase):
    """NCCL weight sync client for vLLM servers."""

    BACKEND_KEY = "vllm"
    BACKEND_NAME = "vLLM"
    ENGINE_RELEASE = "vLLM 0.26.0"
    GROUP_HOST_ENV = "VLLM_GROUP_HOST"
    RESUME_ENDPOINT = "/resume"
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
        **unregistered_bailing_model_types(ENGINE_RELEASE),
    }
    # The layerwise reload processes a layer once all its tensors arrived; one whose tensors straddled
    # the interrupted chunk boundary is materialized from uninitialized storage while it waits for the
    # rest, and that storage is live once the phase closes.
    INTERRUPTED_UPDATE_STATE = (
        "a model that is part old weights and part new, with any layer that straddled the last chunk "
        "boundary materialized from UNINITIALIZED storage by the layerwise reload"
    )

    def _reset_transport_state(self) -> None:
        self.communicator: PyNcclCommunicator | None = None
        # Kept so close_communicator can release the listener the store took over.
        self._unique_id_store: UniqueIdStore | None = None
        # The producer's pack streams, created lazily on the communicator's device; a client's flushes
        # are serialized, so reuse is race-free.
        self._packed_streams: list[torch.cuda.Stream] | None = None
        # A layerwise reload opened server-side (possibly by a request whose reply was lost), which
        # only /finish_weight_update can end — every later sync fails against it otherwise.
        self._phase_started = False

    def _server_reports_paused(self) -> bool | None:
        """Server-side pause state via ``GET /is_paused``, or ``None`` when the server does not report one."""
        try:
            resp = self.request("GET", "/is_paused", retry=False, timeout=HTTP_PROBE_TIMEOUT_S, check=False)
            if resp.status_code != 200:
                return None
            return bool(resp.json()["is_paused"])
        except (requests.exceptions.RequestException, ValueError, KeyError) as e:
            logger.debug(f"Could not read pause state from {self.base_url}: {e}")
            return None

    def _probe_generation(self):
        """Bounded 1-token probe to fail fast when /health is up but generation is wedged.

        Only a hang counts as wedged: the completion below goes out once, so a timeout is conclusive
        rather than something the session would retry away.
        """
        # A paused engine queues new requests instead of rejecting them, so the probe below would only
        # time out and report a wedged scheduler. Ask the server first.
        if self._server_reports_paused():
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

    def init_communicator(self, device: torch.device | str | int = 0):
        """Init NCCL group. Trainer=rank 0, vLLM workers=rank 1+."""
        device = self._resolve_sync_device(device)
        # Detect a paused or generation-wedged server before forming the group or running rollouts.
        self._probe_generation()
        resp = self.request("GET", "/get_world_size", retry=False, timeout=HTTP_PROBE_TIMEOUT_S)
        inference_ws = resp.json()["world_size"]
        world_size = inference_ws + 1
        master_address, listener = self._open_group_rendezvous()
        master_port = listener.port

        logger.info(
            f"NCCL init: inference_ws={inference_ws}, master={master_address}:{master_port}, "
            f"listening on {listener.bind_address}"
        )

        # Closed on exit unless the store took the listener over.
        with listener:
            # Told before the store is built: the master blocks until every worker joined.
            server_call = self._start_collective_rpc(
                "/init_weight_transfer_engine",
                timeout=GROUP_FORMATION_TIMEOUT_S,
                json={
                    "init_info": {
                        "master_address": master_address,
                        "master_port": master_port,
                        "rank_offset": 1,
                        "world_size": world_size,
                    }
                },
            )

            store: UniqueIdStore | None = None
            comm_call: AsyncCall | None = None
            try:
                store = UniqueIdStore.create(host=master_address, listener=listener, world_size=world_size)
                # ncclCommInitRank is unconditionally blocking (the wrapper binds no non-blocking init
                # and NCCL has no comm-init deadline), so a server that never joins would park the
                # trainer with its HTTP error unread. Both sides run concurrently; whichever fails
                # first wins.
                comm_call = AsyncCall(
                    name="NCCL group formation",
                    fn=lambda: PyNcclCommunicator(store, device=device),
                )
                _wait_for_calls([server_call, comm_call], timeout=GROUP_FORMATION_TIMEOUT_S)
                self.communicator = comm_call.result
            except Exception as e:
                # `_wait_for_calls` raises as soon as either half fails, so a communicator this rank
                # has already built lives only on the call handle; dropping that reference would leak
                # its ncclComm and the device memory NCCL pinned for it. Abort it before releasing
                # the store.
                built = comm_call.join(_SERVER_ERROR_GRACE_S).result if comm_call is not None else None
                if built is not None:
                    built.abort()
                if store is not None:
                    # The comm thread stays parked in ncclCommInitRank holding this store, so dropping
                    # references would not free the port; close() releases the listener from it.
                    store.close()
                if e is server_call.error:
                    raise  # the server named its own failure; a network-topology hint would mislead
                # A wrong-but-routable master surfaces as a bare TCPStore/handshake timeout, so name the knob.
                raise RuntimeError(
                    f"Weight-transfer group formation failed (advertised master "
                    f"{master_address}:{master_port}; vLLM server {self.base_url}). If the vLLM node "
                    f"cannot reach that address, set VLLM_GROUP_HOST to an interface routable from the "
                    f"vLLM server (and NCCL_SOCKET_IFNAME for the data plane on multi-homed hosts). "
                    f"Original error: {type(e).__name__}: {e}"
                ) from e

        self._unique_id_store = store
        atexit.register(self.close_communicator)
        logger.info("NCCL weight transfer initialized")

    def _require_communicator(self) -> PyNcclCommunicator:
        """The live communicator, raising when there is none; an aborted one broadcasts into nothing."""
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

        Unwinds itself on failure: nothing has been broadcast yet, so a server left paused behind a
        half-opened phase would wedge the next rollout round.
        """
        self._require_communicator()
        # ``_paused`` means the server may be paused: vLLM pauses the engine before writing the reply,
        # so a lost reply leaves it paused unless the flag already covers the in-flight request.
        self._paused = True
        try:
            # mode=keep freezes in-flight generations and resumes them under the new weights (the
            # one-step staleness the sampling-logprob IS ratio corrects). vLLM's default is abort,
            # which returns every in-flight request as a fragment with an ordinary stop reason.
            self.request("POST", "/pause", retry=True, timeout=POST_TIMEOUT_S, params={"mode": "keep"})
            # Set before the POST for the same reason as the pause flag: a lost reply still leaves the
            # layerwise reload open server-side, and only the close can end it.
            self._phase_started = True
            # Single-shot: /start_weight_update is a collective RPC and a session retry would issue two.
            self.request("POST", "/start_weight_update", retry=False, timeout=POST_TIMEOUT_S)
        except Exception:
            self._close_phase_and_resume()
            raise

    def _broadcast_chunk(self, named_params: list[tuple[str, torch.Tensor]], final: bool = False):
        """Declare one chunk over HTTP, then broadcast it packed into the open phase.

        ``/update_weights`` takes one chunk of an open session (``start_weight_update``, N chunks,
        ``finish_weight_update``), so the trainer streams the model rather than declaring it all at
        once, which would stage the whole model on the forwarding rank. ``final`` is unused here: this
        engine's close is its own ``/finish_weight_update`` call.
        """
        del final
        # Each param was validated where it entered the sync (``update_named_param``); here the chunk is
        # only described, in the order the consumer will unpack it.
        names, dtype_names, shapes = describe_chunk(named_params)

        server_call: AsyncCall | None = None
        try:
            server_call = self._start_collective_rpc(
                "/update_weights",
                timeout=WEIGHT_UPDATE_TIMEOUT_S,
                json={
                    "update_info": {
                        "names": names,
                        "dtype_names": dtype_names,
                        "shapes": shapes,
                        "packed": True,
                        # Sent explicitly: the consumer derives its chunk boundaries from these,
                        # and a mismatch desynchronizes every broadcast count.
                        "packed_buffer_size_bytes": PACKED_BUFFER_BYTES,
                        "packed_num_buffers": PACKED_NUM_BUFFERS,
                    }
                },
            )

            communicator = self._require_communicator()
            device = communicator.device
            if self._packed_streams is None and torch.device(device).type == "cuda":
                with torch.cuda.device(device):
                    self._packed_streams = [torch.cuda.Stream() for _ in range(PACKED_NUM_BUFFERS)]
            # A no-op once init_communicator set the sync device, where every snapshot is staged; one
            # buffered before it is landed here, on the caller's stream, rather than by the producer on a
            # pack stream, whose pool would keep the copy once this client closes.
            landed = [(name, param.to(device, non_blocking=True)) for name, param in named_params]
            packed_broadcast_producer(iterator=iter(landed), group=communicator, streams=self._packed_streams)

            server_call.wait(timeout=WEIGHT_UPDATE_TIMEOUT_S)
        except Exception as e:
            self._raise_if_server_failed(e, server_call)
            raise
        logger.debug(f"Streamed {len(named_params)} params to vLLM")

    def end_weight_update(self, tail: list[tuple[str, torch.Tensor]]):
        """Send the final chunk, close the layerwise reload and resume; the close runs on every path."""
        try:
            self.send_weights(tail, final=True)
            self.request("POST", "/finish_weight_update", retry=False, timeout=WEIGHT_UPDATE_TIMEOUT_S)
            self._phase_started = False
        finally:
            self._close_phase_and_resume()

    def _close_phase_and_resume(self):
        """Close a started-but-unfinished phase and lift the pause. Does not raise: it runs where the
        caller is already failing, and a phase left open wedges every later sync."""
        if self._phase_started:
            try:
                self.request("POST", "/finish_weight_update", retry=False, timeout=CLEANUP_TIMEOUT_S)
            except Exception as e:
                logger.warning(f"Failed to close interrupted weight update: {e}")
            self._phase_started = False
        # Always lift the pause; a server left paused wedges every later rollout.
        self._lift_pause(CLEANUP_TIMEOUT_S, context="after sync")

    def close_communicator(self):
        # Close an update an interrupted sync left open and lift any pause with it, so generation is
        # not wedged after the trainer exits. An abort that left the engine unservable keeps its pause.
        self.abort_weight_update()
        self._lift_pause_at_close()
        if self.communicator is not None:
            # ncclCommAbort rather than a dropped reference: PyNcclCommunicator has no __del__, so the
            # communicator and the device memory NCCL pinned for it would outlive every retired client,
            # one leak per reconnect and per served endpoint.
            self.communicator.abort()
        self.communicator = None
        self._packed_streams = None
        if self._unique_id_store is not None:
            # Frees group_port for the next client on this host; the store only carried the NCCL id.
            self._unique_id_store.close()
            self._unique_id_store = None
        self._finalize_close()
