#!/usr/bin/env python
"""The vLLM weight-sync client's HTTP lifecycle must never strand the server, and never swallow the
server's own diagnosis.

Every defect pinned here leaves a *running* RL job broken with no usable error:

- ``/pause`` takes effect server-side BEFORE vLLM writes its reply, so a lost reply must still count
  as "may be paused" — otherwise nothing ever issues ``/resume`` and every later rollout queues
  forever behind a paused engine that still answers ``/health`` with 200.
- ``/start_weight_update`` opens a layerwise reload (params moved to meta); a lost reply must still
  be closed by ``/finish_weight_update`` or the next sync hits a half-reloaded model.
- ``/update_weights`` runs on a background thread. When the local producer fails, that thread holds
  the server's rejection (e.g. an unsupported dtype) — dropping it turns a precise 500 into
  "broadcast did not complete within 600s".
- ``check_server`` must honour its deadline for a server answering 503/401/404, not only for an
  unreachable one, else startup spins silently forever.
- A paused server must be *reported* as paused: the probe's ``/v1/completions`` request is QUEUED by
  a paused engine, so it can only time out and misreport a wedged scheduler.
- The reconnect path must retire the old client (which owns the ``/resume``) before probing its
  replacement, or recovery from a paused server is impossible.
- The ``/v1/models`` read must go over the retrying session: a still-warming server answers 503, and
  a bare request would fail the startup context-window preflight on it.

Tests drive the REAL client against a fake vLLM HTTP server that records its own state, so they fail
on the ordering/bookkeeping bugs rather than on mocked call counts.

Run: ``python tests/cpu/grpo/test_weight_sync_protocol.py`` (or ``pytest -m cpu``).
"""

from __future__ import annotations

import contextlib
import json
import socket
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest
import requests
import torch
import torch.distributed as dist
from torch.distributed.tensor import Shard, distribute_tensor, init_device_mesh

from src.distributed.nccl.clients import vllm as wsc
from src.distributed.nccl.clients.vllm import VLLMWeightSyncClient
from src.distributed.nccl.transport.packed_tensor import DEFAULT_PACKED_BUFFER_SIZE_BYTES, DEFAULT_PACKED_NUM_BUFFERS
from src.distributed.nccl.transport.pynccl import PyNcclCommunicator
from src.distributed.nccl.transport.stateless_group import StatelessProcessGroup
from src.trainers.grpo.rollout.weight_sync_clients import InferenceClientManager, verify_context_window
from tests.common.ports import free_port

# Well above the client deadlines each test monkeypatches down, so a block is observed, not waited out.
JOIN_TIMEOUT_S = 30.0


class FakeVLLMServer:
    """Fake vLLM weight-transfer endpoints that track real engine state (paused / update open).

    ``status`` overrides a path's HTTP status while the *state change* still happens first, exactly as
    vLLM's handlers do (pause the engine, then write the reply). ``refuse`` models the other failure
    shape — the request never reached the engine, so no state changed.
    """

    def __init__(self):
        self.paused = False
        self.update_open = False
        # /get_world_size: 0 makes the trainer group a world of one, so create() returns without a peer
        # while rank 0 still binds the listener the port tests exercise.
        self.world_size = 1
        # Declared on the model card only when set, as a server without one reports no context window.
        self.max_model_len: int | None = None
        self.requests: list[str] = []
        self.bodies: dict[str, dict] = {}
        self.status: dict[str, int] = {}
        self.refuse: set[str] = set()
        self._lock = threading.Lock()
        server = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.0"

            def log_message(self, *args):
                pass

            def _reply(self, code: int, body: dict | None = None):
                payload = json.dumps(body or {}).encode()
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

            def do_GET(self):
                path = self.path.split("?")[0]
                with server._lock:
                    server.requests.append(f"GET {path}")
                code = server.status.get(path, 200)
                if path == "/health":
                    self._reply(code)
                elif path == "/is_paused":
                    self._reply(code, {"is_paused": server.paused})
                elif path == "/v1/models":
                    card = {"id": "fake-model"}
                    if server.max_model_len is not None:
                        card["max_model_len"] = server.max_model_len
                    self._reply(code, {"data": [card]})
                elif path == "/get_world_size":
                    self._reply(code, {"world_size": server.world_size})
                else:
                    self._reply(404)

            def do_POST(self):
                length = int(self.headers.get("Content-Length") or 0)
                raw = self.rfile.read(length)
                path = self.path.split("?")[0]
                with server._lock:
                    server.requests.append(f"POST {path}")
                    if raw:
                        server.bodies[path] = json.loads(raw)
                    if path in server.refuse:
                        self._reply(503)
                        return
                    # The engine acts first; only the reply may fail.
                    if path == "/pause":
                        server.paused = True
                    elif path == "/resume":
                        server.paused = False
                    elif path == "/start_weight_update":
                        server.update_open = True
                    elif path == "/finish_weight_update":
                        server.update_open = False
                self._reply(server.status.get(path, 200))

        self._httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self._httpd.server_address[1]}"
        self._thread = threading.Thread(target=self._httpd.serve_forever, daemon=True)
        self._thread.start()

    def count(self, entry: str) -> int:
        with self._lock:
            return self.requests.count(entry)

    def close(self):
        self._httpd.shutdown()
        self._httpd.server_close()


class FakeCommunicator:
    """Stand-in for ``PyNcclCommunicator`` (the real one needs a GPU and a live peer)."""

    def __init__(self, aborted: bool = False):
        self.device = torch.device("cpu")
        self.aborted = aborted

    def abort(self):
        self.aborted = True

    def broadcast(self, tensor, src=0):
        pass


class _GroupHoldingCommunicator(FakeCommunicator):
    """``PyNcclCommunicator`` keeps the process group it formed from (``self.group``) — the reason a
    live communicator keeps its ``group_port`` bound."""

    def __init__(self, group, device):
        super().__init__()
        self.group = group


@pytest.fixture
def server():
    fake = FakeVLLMServer()
    yield fake
    fake.close()


@pytest.fixture
def client(server, monkeypatch):
    """A real client attached to the fake server, with the cleanup/HTTP deadlines shortened."""
    monkeypatch.setattr(wsc, "_CLEANUP_TIMEOUT_S", 5)
    monkeypatch.setattr(wsc, "_HTTP_PROBE_TIMEOUT_S", 5)
    monkeypatch.setattr(wsc, "_WEIGHT_UPDATE_TIMEOUT_S", 10)
    handle = VLLMWeightSyncClient(base_url=server.url, connection_timeout=5.0)
    handle.communicator = FakeCommunicator()
    return handle


def _no_op_producer(**kwargs):
    for _ in kwargs["iterator"]:
        pass


def _bind_error(port: int) -> str | None:
    """``None`` if a listener can be bound on ``port`` right now, else the OS error."""
    with contextlib.closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as probe:
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            probe.bind(("0.0.0.0", port))
            probe.listen()
        except OSError as e:
            return str(e)
    return None


def _bindable(port: int) -> bool:
    return _bind_error(port) is None


def _run_bounded(fn) -> list:
    """Run ``fn`` on a thread and return ``[outcome]``, or ``[]`` if it never returned.

    An unbounded spin is one of the defects under test, so it must be observed as one instead of
    hanging the suite (``pytest-timeout`` is not installed in the image).
    """
    outcome: list = []

    def runner():
        try:
            fn()
            outcome.append(None)
        except BaseException as e:
            outcome.append(e)

    thread = threading.Thread(target=runner, daemon=True)
    thread.start()
    thread.join(timeout=JOIN_TIMEOUT_S)
    return outcome


def test_lost_pause_reply_still_resumes_the_server(server, client, monkeypatch):
    """A ``/pause`` whose reply fails must still be resumed: the engine is already paused."""
    monkeypatch.setattr(wsc, "packed_broadcast_producer", _no_op_producer)
    server.status["/pause"] = 500

    with pytest.raises(Exception, match="/pause"):
        client.sync_model_weights([("w", torch.zeros(4, dtype=torch.bfloat16))])

    assert client._paused is True or server.count("POST /resume") > 0, (
        "the pause flag was only set AFTER the POST, so a lost reply leaves no record of the pause"
    )
    assert server.count("POST /resume") > 0, "no /resume was issued for a pause that took effect server-side"
    assert server.paused is False, "server left PAUSED — every later rollout queues behind it forever"


def test_pause_left_by_a_failed_sync_is_lifted_on_close(server, client, monkeypatch):
    """The atexit/close path must lift a pause whose reply never arrived."""
    monkeypatch.setattr(wsc, "packed_broadcast_producer", _no_op_producer)
    server.status["/pause"] = 500
    server.refuse.add("/resume")  # the in-sync resume never lands either; only close() can recover

    with pytest.raises(Exception, match="/pause"):
        client.sync_model_weights([("w", torch.zeros(4, dtype=torch.bfloat16))])
    assert server.paused is True  # precondition for what close_communicator must repair

    server.refuse.clear()
    client.close_communicator()
    assert server.paused is False, "close_communicator did not resume a server its own sync left paused"


def test_failed_start_weight_update_is_closed(server, client, monkeypatch):
    """A ``/start_weight_update`` that may have taken effect must be closed by the finally."""
    monkeypatch.setattr(wsc, "packed_broadcast_producer", _no_op_producer)
    server.status["/start_weight_update"] = 500

    with pytest.raises(Exception, match="start_weight_update"):
        client.sync_model_weights([("w", torch.zeros(4, dtype=torch.bfloat16))])

    assert server.count("POST /finish_weight_update") > 0, (
        "a started-but-failed weight update was never closed — the server stays mid-layerwise-reload"
    )
    assert server.update_open is False
    assert server.paused is False


def test_start_weight_update_read_timeout_is_closed(server, client, monkeypatch):
    """Same defect via the real-world trigger: the reply is lost, not refused."""
    monkeypatch.setattr(wsc, "packed_broadcast_producer", _no_op_producer)
    real_post = requests.post

    def timing_out_post(url, *args, **kwargs):
        if url.endswith("/start_weight_update"):
            real_post(url, *args, **kwargs)  # the server DOES open the update
            raise requests.exceptions.ReadTimeout("simulated lost reply")
        return real_post(url, *args, **kwargs)

    monkeypatch.setattr(requests, "post", timing_out_post)

    with pytest.raises(requests.exceptions.ReadTimeout):
        client.sync_model_weights([("w", torch.zeros(4, dtype=torch.bfloat16))])

    assert server.update_open is False, "a lost /start_weight_update reply left the update open"
    assert server.paused is False


def test_producer_failure_surfaces_the_server_rejection(server, client, monkeypatch):
    """The server's ``/update_weights`` error is the root cause; the producer's is the symptom."""

    def failing_producer(**kwargs):
        raise RuntimeError("NCCL weight-sync packed weight broadcast did not complete within 600s")

    monkeypatch.setattr(wsc, "packed_broadcast_producer", failing_producer)
    server.status["/update_weights"] = 500

    with pytest.raises(Exception) as excinfo:
        client.sync_model_weights([("w", torch.zeros(4, dtype=torch.bfloat16))])

    reported = f"{excinfo.value}\n{excinfo.value.__cause__}"
    assert "/update_weights" in reported and "500" in reported, (
        f"the server's rejection died with its daemon thread; caller only saw: {excinfo.value}"
    )
    assert server.paused is False


def test_update_weights_declares_the_packed_layout(server, client, monkeypatch):
    """The chunk-boundary parameters must be on the wire, not implied by matching defaults."""
    monkeypatch.setattr(wsc, "packed_broadcast_producer", _no_op_producer)

    client.sync_model_weights([("w", torch.zeros(4, dtype=torch.bfloat16))])

    update_info = server.bodies["/update_weights"]["update_info"]
    assert update_info["packed_buffer_size_bytes"] == DEFAULT_PACKED_BUFFER_SIZE_BYTES
    assert update_info["packed_num_buffers"] == DEFAULT_PACKED_NUM_BUFFERS
    assert update_info["names"] == ["w"] and update_info["shapes"] == [[4]]


def test_a_mid_stream_abort_leaves_the_engine_paused_and_refuses_the_next_sync(server, client, monkeypatch):
    """A sync interrupted AFTER a chunk went out must not resume the engine as if it were healthy.

    It then holds neither the old policy nor the new one, and the layerwise reload materializes a
    layer whose tensors straddled the boundary from uninitialized storage while it waits for the
    rest — which never arrives, because the trainer keeps no copy of what already landed. Resuming
    serves exactly that, behind a 200 ``/health``.
    """
    monkeypatch.setattr(wsc, "packed_broadcast_producer", _no_op_producer)
    monkeypatch.setattr("src.distributed.nccl.clients.base.WEIGHT_SYNC_CHUNK_BYTES", 512)

    for name in ("a", "b"):  # 1 KiB each: the second crosses the budget, so the first chunk goes out
        client.update_named_param(name, torch.zeros(512, dtype=torch.bfloat16))
    assert client._chunks_sent == 1 and server.update_open, "nothing was streamed — the test is vacuous"

    server.status["/update_weights"] = 500
    with pytest.raises(requests.exceptions.HTTPError, match="500"):
        client.update_named_param("c", torch.zeros(512, dtype=torch.bfloat16))
    client.abort_weight_update()

    assert server.count("POST /resume") == 0, "an engine holding a half-written model was resumed"
    assert server.paused is True and server.update_open is True, (
        "the interrupted update was closed and the engine released to serve a model that is part "
        "old, part new, and part uninitialized"
    )
    with pytest.raises(RuntimeError, match="RESTART"):
        client.sync_model_weights([("d", torch.zeros(4, dtype=torch.bfloat16))])
    client.close_communicator()
    assert server.paused is True, "close_communicator resumed a server that has to be restarted"


def test_a_failed_tail_chunk_after_a_streamed_one_is_not_resumed(server, client, monkeypatch):
    """The tail is a chunk like any other, and its close resumes the engine in a ``finally``.

    Failing it once an earlier chunk landed leaves the same part-old-part-new model as failing
    mid-stream, so the verdict has to be taken where the chunk fails — before that ``finally`` puts
    the engine back on the wire.
    """
    monkeypatch.setattr(wsc, "packed_broadcast_producer", _no_op_producer)
    monkeypatch.setattr("src.distributed.nccl.clients.base.WEIGHT_SYNC_CHUNK_BYTES", 512)

    for name in ("a", "b"):
        client.update_named_param(name, torch.zeros(512, dtype=torch.bfloat16))
    assert client._chunks_sent == 1, "nothing was streamed before the tail — the test is vacuous"

    server.status["/update_weights"] = 500
    with pytest.raises(requests.exceptions.HTTPError, match="500"):
        client.reset_prefix_cache()

    assert server.count("POST /resume") == 0, "the engine was resumed holding a half-written model"
    assert server.paused is True
    assert server.update_open is False, "the reload phase was left open by the failed tail"
    with pytest.raises(RuntimeError, match="RESTART"):
        client.sync_model_weights([("d", torch.zeros(4, dtype=torch.bfloat16))])


def test_an_abort_before_any_chunk_still_closes_and_resumes_the_server(server, client, monkeypatch):
    """Anti-over-rejection: with nothing on the wire the engine still holds the weights it was
    serving, so the phase must be closed and the pause lifted — left quiesced it refuses every later
    sync and queues every rollout."""
    monkeypatch.setattr(wsc, "packed_broadcast_producer", _no_op_producer)
    monkeypatch.setattr("src.distributed.nccl.clients.base.WEIGHT_SYNC_CHUNK_BYTES", 512)
    server.status["/update_weights"] = 500

    client.update_named_param("a", torch.zeros(512, dtype=torch.bfloat16))
    with pytest.raises(requests.exceptions.HTTPError, match="500"):
        client.update_named_param("b", torch.zeros(512, dtype=torch.bfloat16))
    assert client._chunks_sent == 0 and client.can_replay_sync, "the first chunk landed — wrong scenario"

    client.abort_weight_update()

    assert server.update_open is False, "the reload phase was left open behind a failed first chunk"
    assert server.paused is False, "the engine stayed paused although nothing had been streamed"
    server.status.clear()
    client.sync_model_weights([("w", torch.zeros(4, dtype=torch.bfloat16))])  # the client is still usable


def test_a_failed_snapshot_completion_still_closes_the_update(server, client, monkeypatch):
    """The D2H completion runs INSIDE the close: a sticky CUDA error there is exactly the state a
    failing sync is in, and skipping ``/finish_weight_update`` leaves the engine mid-reload while
    ``/health`` answers 200 — every later ``/start_weight_update`` fails and the probe cannot see it."""
    monkeypatch.setattr(wsc, "packed_broadcast_producer", _no_op_producer)

    def sticky_cuda_error(self):
        raise RuntimeError("CUDA error: an illegal memory access was encountered")

    monkeypatch.setattr(VLLMWeightSyncClient, "_complete_host_snapshots", sticky_cuda_error)

    client.update_named_param("w", torch.zeros(4, dtype=torch.bfloat16))
    with pytest.raises(RuntimeError, match="illegal memory access"):
        client.reset_prefix_cache()

    assert server.update_open is False, "the layerwise reload was left open by the failed D2H sync"
    assert server.paused is False, "the engine was left quiesced by the failed D2H sync"
    assert client._update_open is False


def test_a_dropped_tail_after_a_streamed_chunk_is_not_resumed(server, client, monkeypatch):
    """The third way a sync ends mid-model: the D2H sync fails, so the tail is dropped rather than
    sent. The engine then holds the earlier chunks and its old weights for the rest — the same mixed
    model, so the close must still refuse to resume it."""
    monkeypatch.setattr(wsc, "packed_broadcast_producer", _no_op_producer)
    monkeypatch.setattr("src.distributed.nccl.clients.base.WEIGHT_SYNC_CHUNK_BYTES", 512)

    for name in ("a", "b"):
        client.update_named_param(name, torch.zeros(512, dtype=torch.bfloat16))
    assert client._chunks_sent == 1, "nothing was streamed before the tail — the test is vacuous"

    def sticky_cuda_error(self):
        raise RuntimeError("CUDA error: an illegal memory access was encountered")

    monkeypatch.setattr(VLLMWeightSyncClient, "_complete_host_snapshots", sticky_cuda_error)
    with pytest.raises(RuntimeError, match="illegal memory access"):
        client.reset_prefix_cache()

    assert server.update_open is False, "the reload phase was left open"
    assert server.count("POST /resume") == 0 and server.paused is True, (
        "the engine was resumed holding the streamed chunks and nothing of the dropped tail"
    )


def test_the_d2h_completion_names_the_forwarding_ranks_device(client, monkeypatch):
    """The flush runs on a worker thread once more than one server is synced, and torch's current
    device AND current stream are thread-local: a device-implicit sync there completes device 0's
    copies while this rank's are still in flight, and the engine is broadcast whatever the pinned
    buffers happen to hold."""
    synchronized: list = []
    monkeypatch.setattr(torch.cuda, "synchronize", lambda device=None: synchronized.append(device))

    client._sync_device = torch.device("cuda", 3)
    client._complete_host_snapshots()
    assert synchronized == [torch.device("cuda", 3)], f"synchronized {synchronized} instead of the client's device"

    client._sync_device = None  # before init_communicator there is nothing to complete
    client._complete_host_snapshots()
    assert synchronized == [torch.device("cuda", 3)]


def test_check_server_honours_the_deadline_on_a_non_200_health(server, monkeypatch):
    """A route answering 503 forever is as dead as an unreachable one."""
    monkeypatch.setattr(wsc, "_HTTP_PROBE_TIMEOUT_S", 5)
    server.status["/health"] = 503

    outcome = _run_bounded(lambda: VLLMWeightSyncClient(base_url=server.url, connection_timeout=1.0))

    assert outcome, "check_server never returned — the deadline is only consulted on the exception path"
    assert isinstance(outcome[0], ConnectionError), f"expected ConnectionError, got {outcome[0]!r}"
    assert "503" in str(outcome[0]), f"the last /health status is not reported: {outcome[0]}"


def test_probe_generation_reports_a_paused_server(server, client):
    """A paused engine queues the probe request, so guessing from a timeout misreports the cause."""
    requests.post(f"{server.url}/pause")
    assert server.paused is True

    outcome = _run_bounded(client.probe_generation)

    assert outcome, "probe_generation blocked instead of reading /is_paused"
    assert isinstance(outcome[0], RuntimeError), f"a paused server was not reported: {outcome[0]!r}"
    assert "PAUSED" in str(outcome[0])
    assert "wedged" not in str(outcome[0]), "a paused server must not be reported as a wedged scheduler"


def test_probe_generation_passes_on_a_serving_server(server, client):
    """Anti-vacuity: the pause check must not reject a healthy server."""
    assert _run_bounded(client.probe_generation) == [None]


def _record_session_gets(client, monkeypatch) -> list[str]:
    """URLs the client's retrying session GETs, recorded without disabling the adapter."""
    seen: list[str] = []
    real_get = client.session.get

    def spy(url, *args, **kwargs):
        seen.append(url)
        return real_get(url, *args, **kwargs)

    monkeypatch.setattr(client.session, "get", spy)
    return seen


def test_the_model_card_read_goes_over_the_retrying_session(server, client, monkeypatch):
    """A bare request would skip the session's 503 retry, failing the preflight on a warming server."""
    server.max_model_len = 4096
    session_gets = _record_session_gets(client, monkeypatch)

    assert client.served_max_model_len() == 4096
    assert [url for url in session_gets if url.endswith("/v1/models")], (
        "the model-card read bypassed the client's retrying session"
    )


def test_a_card_without_a_context_window_reports_none(server, client):
    """The preflight skips rather than raising on a server whose card omits ``max_model_len``."""
    assert client.served_max_model_len() is None


def test_probe_generation_resolves_the_served_model_over_the_session(server, client, monkeypatch):
    session_gets = _record_session_gets(client, monkeypatch)

    client.probe_generation()

    assert [url for url in session_gets if url.endswith("/v1/models")], (
        "probe_generation resolved the served model outside the client's session"
    )


def test_context_window_preflight_raises_on_a_server_that_cannot_hold_one_turn(server):
    """The startup gate builds its own client per URL, so a too-small window fails before training."""
    server.max_model_len = 1024

    with pytest.raises(ValueError, match="context window"):
        verify_context_window(VLLMWeightSyncClient, [server.url], 4096, None)

    verify_context_window(VLLMWeightSyncClient, [server.url], 512, None)


def test_context_window_preflight_skips_an_unreachable_server(server):
    """An unreadable probe must never be what fails the run — it warns and moves on."""
    server.close()

    verify_context_window(VLLMWeightSyncClient, [server.url], 10**9, None)


def test_group_port_is_released_when_group_formation_fails(server, monkeypatch):
    """A failed group formation must give its ``group_port`` back, or every retry EADDRINUSEs.

    The NCCL side is unconditionally blocking, so when the server's half fails the comm thread stays
    parked in ``ncclCommInitRank`` holding the process group — dropping references cannot free the
    listener, only closing it out of the group can. Without that, one wedged connect attempt burns the
    configured port for the whole process: the reconnect path and every later client on that port die
    with ``Address already in use`` instead of retrying.
    """
    monkeypatch.setattr(wsc, "_HTTP_PROBE_TIMEOUT_S", 5)
    monkeypatch.setattr(wsc, "_GROUP_FORMATION_TIMEOUT_S", 20)
    server.world_size = 0
    server.status["/init_weight_transfer_engine"] = 500
    port = free_port()
    handle = VLLMWeightSyncClient(base_url=server.url, group_port=port, connection_timeout=5.0)

    released = threading.Event()
    # Parked like the real blocking init, holding the group like PyNcclCommunicator.group does.
    monkeypatch.setattr(wsc, "PyNcclCommunicator", lambda pg, device: (pg, released.wait()))
    try:
        outcome = _run_bounded(lambda: handle.init_communicator(device="cpu"))
        assert outcome, "init_communicator parked in the blocking group init"
        assert isinstance(outcome[0], Exception), f"a 500 from the server did not raise: {outcome[0]!r}"
        assert _bindable(port), f"group_port {port} still held after a failed formation: {_bind_error(port)}"
    finally:
        released.set()


def test_group_port_is_released_on_close(server, monkeypatch):
    """``close_communicator`` must free the port so the next client can bind it.

    A trainer that syncs to several servers in sequence, or a test process that builds one trainer
    after another, reuses the configured ``vllm_group_port``; a listener surviving the close makes
    every run after the first fail before it starts.
    """
    monkeypatch.setattr(wsc, "_HTTP_PROBE_TIMEOUT_S", 5)
    monkeypatch.setattr(wsc, "_GROUP_FORMATION_TIMEOUT_S", 20)
    monkeypatch.setattr(wsc, "PyNcclCommunicator", _GroupHoldingCommunicator)
    server.world_size = 0
    port = free_port()

    handle = VLLMWeightSyncClient(base_url=server.url, group_port=port, connection_timeout=5.0)
    handle.init_communicator(device="cpu")
    assert not _bindable(port), "the group listener was never bound — the test proves nothing"

    # Pinned like a parked ncclCommInitRank thread would: the release must not need a GC of the comm.
    pinned = handle.communicator
    handle.close_communicator()

    assert _bindable(port), f"group_port {port} still held after close_communicator: {_bind_error(port)}"
    assert pinned.group.store is None, "the closed group still holds its TCPStore"


def test_a_closed_group_cannot_broadcast():
    """Closing releases the store, so a later exchange must say so instead of raising AttributeError."""
    group = StatelessProcessGroup.create(host="127.0.0.1", port=free_port(), rank=0, world_size=1)
    group.close()
    group.close()  # idempotent: close runs from cleanup paths that may already have run
    with pytest.raises(RuntimeError, match="closed"):
        group.broadcast_obj("x", src=0)


def test_init_communicator_surfaces_a_server_side_init_failure(server, monkeypatch):
    """A 500 from ``/init_weight_transfer_engine`` must win over the blocking ``ncclCommInitRank``."""
    monkeypatch.setattr(wsc, "_HTTP_PROBE_TIMEOUT_S", 5)
    monkeypatch.setattr(wsc, "_GROUP_FORMATION_TIMEOUT_S", 20)
    handle = VLLMWeightSyncClient(base_url=server.url, connection_timeout=5.0)
    server.status["/init_weight_transfer_engine"] = 500

    released = threading.Event()
    closed: list[bool] = []
    stub_group = type("StubGroup", (), {"close": lambda self: closed.append(True)})()
    monkeypatch.setattr(wsc.StatelessProcessGroup, "create", staticmethod(lambda **kwargs: stub_group))
    # ncclCommInitRank blocks until the peer joins; a server that failed never will.
    monkeypatch.setattr(wsc, "PyNcclCommunicator", lambda pg, device: released.wait())

    outcome = _run_bounded(lambda: handle.init_communicator(device="cpu"))
    released.set()

    assert outcome, "init_communicator parked in the blocking group init with the HTTP error unread"
    error = str(outcome[0])
    assert "500" in error, f"the server's own failure was not surfaced: {error}"
    assert "VLLM_GROUP_HOST" not in error, f"a server-side 500 was misreported as a topology error: {error}"
    assert closed, "the group formed for a failed init was left holding its listener"


def test_zero_dim_param_is_rejected_before_the_server_is_paused(server, client, monkeypatch):
    """A param the packed protocol cannot carry must fail while the server is still serving."""
    monkeypatch.setattr(wsc, "packed_broadcast_producer", _no_op_producer)

    with pytest.raises(ValueError, match="0-dimensional"):
        client.sync_model_weights([("scale", torch.zeros((), dtype=torch.bfloat16))])

    assert server.count("POST /pause") == 0, "the server was paused for a payload that can never be sent"
    assert server.paused is False


def test_dtensor_param_is_rejected(server, client, tmp_path, monkeypatch):
    """A DTensor's global ``numel()`` against a local ``data_ptr()`` would broadcast out of bounds."""
    monkeypatch.setattr(wsc, "packed_broadcast_producer", _no_op_producer)
    dist.init_process_group("gloo", rank=0, world_size=1, init_method=f"file://{tmp_path / 'pg'}")
    try:
        mesh = init_device_mesh("cpu", (1,))
        sharded = distribute_tensor(torch.zeros(8, 4, dtype=torch.bfloat16), mesh, [Shard(0)])
        with pytest.raises(TypeError, match="DTensor"):
            client.sync_model_weights([("w", sharded)])
    finally:
        dist.destroy_process_group()

    assert server.count("POST /pause") == 0


def test_close_aborts_the_communicator(server, client):
    """``close_communicator`` must abort the NCCL communicator, not just drop the reference.

    ``PyNcclCommunicator`` has no ``__del__``, so a dropped-but-live communicator strands its device
    memory for the process lifetime — one leak per reconnect and per served endpoint.
    """
    communicator = FakeCommunicator()
    client.communicator = communicator

    client.close_communicator()

    assert communicator.aborted, "close_communicator dropped the communicator without aborting it"
    assert client.communicator is None


def test_aborted_communicator_is_refused_by_name(server, client):
    """An aborted communicator holds a NULL handle; reusing it must not read as an argument bug."""
    client.communicator = FakeCommunicator(aborted=True)

    with pytest.raises(RuntimeError, match="aborted"):
        client.sync_model_weights([("w", torch.zeros(4, dtype=torch.bfloat16))])

    assert server.count("POST /pause") == 0, "the server was paused for a sync that cannot broadcast"


def test_aborted_communicator_refuses_to_broadcast():
    """The communicator itself must fail by name, for callers that hold it directly."""
    comm = PyNcclCommunicator.__new__(PyNcclCommunicator)
    comm.disabled = False
    comm.aborted = True
    comm.device = torch.device("cpu")

    with pytest.raises(RuntimeError, match="aborted"):
        comm.broadcast(torch.zeros(1), src=0)


def test_disabled_communicator_still_exposes_its_device():
    """``sync_model_weights`` reads ``.device`` to stage weights, after the server is already paused."""
    group = StatelessProcessGroup.create(host="127.0.0.1", port=free_port(), rank=0, world_size=1)
    try:
        comm = PyNcclCommunicator(group, device="cpu")  # world_size 1 takes the disabled path
    finally:
        group.close()

    assert comm.disabled is True
    assert comm.device == torch.device("cpu"), "a disabled communicator must still carry its device"


def test_reconnect_retires_the_old_client_before_probing_the_replacement():
    """The old client owns the ``/resume`` for the pause its failed sync left; a replacement probed
    first would only time out against the still-paused server."""
    order: list[str] = []

    class RecordingClient:
        def __init__(self, **kwargs):
            self.base_url = kwargs["base_url"]
            self.group_port = kwargs.get("group_port")
            self._param_buffer: list = []
            order.append("construct new client")

        def init_communicator(self, device):
            order.append("probe + init new client")

        def close_communicator(self):
            order.append("close old client")

        def drain_param_buffer(self):
            order.append("drain old buffer")
            buffered, self._param_buffer = self._param_buffer, []
            return buffered

        def buffer_host_param(self, name, host):
            self._param_buffer.append((name, host))

    manager = InferenceClientManager(server_configs=[{"url": "http://server0:8000", "group_port": 51216}])
    old = RecordingClient(base_url="http://server0:8000")
    old.buffer_host_param("w", torch.zeros(4))
    order.clear()
    manager._clients = [old]
    manager._initialized = True
    manager._device = torch.device("cpu")
    manager._client_factory = RecordingClient

    new_client = manager.reconnect_client(0)

    assert order.index("close old client") < order.index("probe + init new client"), (
        f"the replacement was probed before the old client lifted its pause: {order}"
    )
    assert [name for name, _ in new_client._param_buffer] == ["w"], "the pending buffer was not carried over"
    assert manager._clients[0] is new_client, "the replacement was not swapped into the client pool"
    # The engine node dials back to the configured port, so the replacement must rebind it, not drift.
    assert new_client.group_port == 51216, f"reconnect moved off the configured group_port: {new_client.group_port}"


def test_unconfigured_group_ports_are_seeded_from_the_run_knob():
    """``vllm_group_port`` must mean the same thing whichever client shape is built.

    An entry without its own ``group_port`` takes ``base_group_port + index``; binding the module
    default instead would listen on 51216/51217 for a run that configured 52000, while the engines
    dial the port they were launched with — every sync then times out with nothing to point at.
    """
    manager = InferenceClientManager(
        server_configs=[
            {"url": "http://s0:8000"},
            {"url": "http://s1:8000"},
            {"url": "http://s2:8000", "group_port": 9000},
        ],
        base_group_port=52000,
    )
    assert [manager._group_port(i) for i in range(3)] == [52000, 52001, 9000]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
