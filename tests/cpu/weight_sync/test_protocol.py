#!/usr/bin/env python
"""The vLLM weight-sync client's HTTP lifecycle must never strand the server or swallow the server's own diagnosis.
Each defect pinned here leaves a running RL job broken with no usable error. Tests use a recording fake server."""

from __future__ import annotations

import contextlib
import logging
import socket
import threading
from collections import deque

import pytest
import requests
import torch
import torch.distributed as dist
from torch.distributed.tensor import Shard, distribute_tensor, init_device_mesh

from halo.trainers.grpo.rollout.engine_preflight import (
    _verify_context_window_on_rank0,
    _verify_sampler_logprob_reference_on_rank0,
    probe_sampler_logprob_semantics,
    verify_context_window,
)
from halo.weight_sync.clients import base as client_base
from halo.weight_sync.clients import vllm as wsc
from halo.weight_sync.clients.sglang import SGLangWeightSyncClient
from halo.weight_sync.clients.vllm import VLLMWeightSyncClient
from halo.weight_sync.transport import pynccl
from halo.weight_sync.transport.nccl_library import ncclUniqueId
from halo.weight_sync.transport.packed_broadcast import PACKED_BUFFER_BYTES, PACKED_NUM_BUFFERS
from halo.weight_sync.transport.pynccl import PyNcclCommunicator, UniqueIdStore
from halo.weight_sync.transport.rendezvous import RendezvousListener
from tests.common.fake_engines import FakeSGLangServer, FakeVLLMServer, fake_server_class
from tests.common.ports import free_port
from tests.common.weight_sync import offline_client, push

# Well above the client deadlines each test monkeypatches down, so a block is observed, not waited out.
JOIN_TIMEOUT_S = 30.0


class FakeCommunicator:
    def __init__(self, aborted: bool = False):
        self.device = torch.device("cpu")
        self.aborted = aborted

    def abort(self):
        self.aborted = True

    def broadcast(self, tensor):
        pass


class _StoreHoldingCommunicator(FakeCommunicator):
    """Holds the unique-id store it formed from, as the frame of a thread parked in ``ncclCommInitRank`` does."""

    def __init__(self, store, device):
        super().__init__()
        self.store = store


class _BlockingInitLibrary:
    """``ncclCommInitRank`` parks, as the real one does until the engine joins, and calls ``on_parked`` first."""

    def __init__(self, on_parked):
        self.parked = threading.Event()
        self.released = threading.Event()
        self._on_parked = on_parked

    def ncclGetUniqueId(self) -> ncclUniqueId:
        return ncclUniqueId()

    def ncclGetVersion(self) -> str:
        return "2.0.0"

    def ncclCommInitRank(self, world_size, unique_id, rank):
        self.parked.set()
        self._on_parked()
        self.released.wait()
        raise RuntimeError("NCCL error: the engine never joined")


@pytest.fixture
def server():
    fake = FakeVLLMServer()
    yield fake
    fake.close()


@pytest.fixture
def client(server, monkeypatch):
    monkeypatch.setattr(client_base, "CLEANUP_TIMEOUT_S", 5)
    monkeypatch.setattr(wsc, "HTTP_PROBE_TIMEOUT_S", 5)
    monkeypatch.setattr(wsc, "CHUNK_SYNC_TIMEOUT_S", 10)
    handle = VLLMWeightSyncClient(base_url=server.url, connection_timeout=5.0)
    handle._transport = FakeCommunicator()
    return handle


def _no_op_producer(buffers, **kwargs):
    """Consumes the packed buffers without a peer."""


def _bind_error(port: int) -> str | None:
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
    """Returns ``[outcome]``, or ``[]`` if ``fn`` never returned: an unbounded spin is a defect under test, and
    ``pytest-timeout`` is not in the image."""
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


def test_pause_keeps_in_flight_generations(server, client, monkeypatch):
    """``mode=abort``, the default, kills the prefetched rollout round and hands each back as a fragment with an
    ordinary stop reason. ``keep`` freezes those requests to resume under the new weights."""
    monkeypatch.setattr(wsc, "broadcast_packed", _no_op_producer)

    push(client, [("w", torch.zeros(4, dtype=torch.bfloat16))])

    assert server.count("POST /pause") == 1
    assert "mode=keep" in server.queries["/pause"], f"pause query was {server.queries['/pause']!r}"


def test_a_chunk_waits_on_its_update_rpc_for_the_configured_chunk_deadline(server, client, monkeypatch):
    """vLLM replies only once it received the chunk, so a fixed RPC timeout below ``HALO_NCCL_SYNC_TIMEOUT_SECONDS``
    would fail a chunk the knob allows."""
    monkeypatch.setattr(wsc, "broadcast_packed", _no_op_producer)
    monkeypatch.setattr(wsc, "CHUNK_SYNC_TIMEOUT_S", 1234.0)
    monkeypatch.setattr(client_base, "CHUNK_SYNC_TIMEOUT_S", 1234.0)
    deadlines = []
    real_start = client._start_collective_rpc

    def start(path, *, timeout, json):
        call = real_start(path, timeout=timeout, json=json)
        if path == "/update_weights":
            deadlines.append(("rpc", timeout))
            real_wait = call.wait
            call.wait = lambda timeout: deadlines.append(("wait", timeout)) or real_wait(timeout)
        return call

    monkeypatch.setattr(client, "_start_collective_rpc", start)
    push(client, [("w", torch.zeros(4, dtype=torch.bfloat16))])

    assert deadlines == [("rpc", 1234.0), ("wait", 1234.0)], deadlines


def test_lost_pause_reply_still_resumes_the_server(server, client, monkeypatch):
    monkeypatch.setattr(wsc, "broadcast_packed", _no_op_producer)
    server.status["/pause"] = 500

    with pytest.raises(Exception, match="/pause"):
        push(client, [("w", torch.zeros(4, dtype=torch.bfloat16))])

    assert client._paused is True or server.count("POST /resume") > 0, (
        "the pause flag was only set AFTER the POST, so a lost reply leaves no record of the pause"
    )
    assert server.count("POST /resume") > 0, "no /resume was issued for a pause that took effect server-side"
    assert server.paused is False, "server left PAUSED — every later rollout queues behind it forever"


def test_pause_left_by_a_failed_sync_is_lifted_on_close(server, client, monkeypatch):
    monkeypatch.setattr(wsc, "broadcast_packed", _no_op_producer)
    server.status["/pause"] = 500
    server.refuse.add("/resume")  # the in-sync resume never lands either; only close() can recover

    with pytest.raises(Exception, match="/pause"):
        push(client, [("w", torch.zeros(4, dtype=torch.bfloat16))])
    assert server.paused is True  # precondition for what close_communicator must repair

    server.refuse.clear()
    client.close_communicator()
    assert server.paused is False, "close_communicator did not resume a server its own sync left paused"


def test_failed_start_weight_update_is_closed(server, client, monkeypatch):
    monkeypatch.setattr(wsc, "broadcast_packed", _no_op_producer)
    server.status["/start_weight_update"] = 500

    with pytest.raises(Exception, match="start_weight_update"):
        push(client, [("w", torch.zeros(4, dtype=torch.bfloat16))])

    assert server.count("POST /finish_weight_update") > 0, (
        "a started-but-failed weight update was never closed — the server stays mid-layerwise-reload"
    )
    assert server.update_open is False
    assert server.paused is False


def test_start_weight_update_read_timeout_is_closed(server, client, monkeypatch):
    monkeypatch.setattr(wsc, "broadcast_packed", _no_op_producer)
    real_request = requests.request

    def timing_out_request(method, url, *args, **kwargs):
        if url.endswith("/start_weight_update"):
            real_request(method, url, *args, **kwargs)  # the server does open the update
            raise requests.exceptions.ReadTimeout("simulated lost reply")
        return real_request(method, url, *args, **kwargs)

    monkeypatch.setattr(requests, "request", timing_out_request)

    with pytest.raises(requests.exceptions.ReadTimeout):
        push(client, [("w", torch.zeros(4, dtype=torch.bfloat16))])

    assert server.update_open is False, "a lost /start_weight_update reply left the update open"
    assert server.paused is False


def test_producer_failure_surfaces_the_server_rejection(server, client, monkeypatch):
    """The server's ``/update_weights`` error is the root cause; the producer's is the symptom."""

    def failing_producer(buffers, **kwargs):
        raise RuntimeError("NCCL weight-sync packed weight broadcast did not complete within 600s")

    monkeypatch.setattr(wsc, "broadcast_packed", failing_producer)
    server.status["/update_weights"] = 500

    with pytest.raises(Exception) as excinfo:
        push(client, [("w", torch.zeros(4, dtype=torch.bfloat16))])

    reported = f"{excinfo.value}\n{excinfo.value.__cause__}"
    assert "/update_weights" in reported and "500" in reported, (
        f"the server's rejection died with its daemon thread; caller only saw: {excinfo.value}"
    )
    assert server.paused is False


def test_an_update_declares_only_names_dtypes_and_shapes(server, client, monkeypatch):
    """The engine refuses an update carrying any other field; the packed geometry rides the init."""
    monkeypatch.setattr(wsc, "broadcast_packed", _no_op_producer)

    push(client, [("w", torch.zeros(4, dtype=torch.bfloat16))])

    update_info = server.bodies["/update_weights"]["update_info"]
    assert update_info == {"names": ["w"], "dtype_names": ["bfloat16"], "shapes": [[4]]}


def test_a_mid_stream_abort_leaves_the_engine_paused_and_refuses_the_next_sync(server, client, monkeypatch):
    """The engine holds neither policy, and the layerwise reload materializes a layer straddling the boundary from
    uninitialized storage, waiting for a rest that never arrives. Resuming serves that behind a 200 ``/health``."""
    monkeypatch.setattr(wsc, "broadcast_packed", _no_op_producer)
    monkeypatch.setattr("halo.weight_sync.clients.base.WEIGHT_SYNC_CHUNK_BYTES", 512)

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
        push(client, [("d", torch.zeros(4, dtype=torch.bfloat16))])
    client.close_communicator()
    assert server.paused is True, "close_communicator resumed a server that has to be restarted"


def test_a_close_that_fails_after_chunks_landed_leaves_the_engine_paused(server, client, monkeypatch):
    """A reload left unfinished holds the same mixed model as a failed chunk, and a resume serves it."""
    monkeypatch.setattr(wsc, "broadcast_packed", _no_op_producer)
    server.status["/finish_weight_update"] = 500
    with pytest.raises(requests.exceptions.HTTPError, match="500"):
        push(client, [("a", torch.zeros(4, dtype=torch.bfloat16))])

    assert server.count("POST /resume") == 0 and server.paused is True
    with pytest.raises(RuntimeError, match="RESTART"):
        push(client, [("b", torch.zeros(4, dtype=torch.bfloat16))])


def _sglang_with_a_group() -> tuple[FakeSGLangServer, SGLangWeightSyncClient]:
    server = FakeSGLangServer()
    client = SGLangWeightSyncClient(base_url=server.url)
    client._transport = (object(), None)
    return server, client


def test_a_flush_that_fails_at_the_close_leaves_sglang_paused(monkeypatch):
    """With no tail chunk to carry the flush it goes by its own route, through the same close."""
    server, client = _sglang_with_a_group()
    try:
        monkeypatch.setattr(client, "_start_engine_update", client._quiesce)
        server.status["/flush_cache"] = 500
        client._open_update()
        client._chunks_sent = 1
        monkeypatch.setattr(client, "request", _single_shot_request(client.request))
        with pytest.raises(requests.exceptions.HTTPError, match="500"):
            client._close_update([])
        assert server.count("POST /continue_generation") == 0 and server.paused is True
        assert client._unusable_reason is not None
    finally:
        server.close()


def _single_shot_request(request):
    """The session's status retries would back off ~80 s on a scripted 500."""

    def single_shot(method, path, **kwargs):
        kwargs["retry"] = False
        return request(method, path, **kwargs)

    return single_shot


def test_a_lost_resume_is_retried_after_a_good_push(server, client, monkeypatch):
    """One failed POST left the engine paused with the push counted as done: the next rollouts queued to their timeout."""
    monkeypatch.setattr(wsc, "broadcast_packed", _no_op_producer)
    monkeypatch.setattr(client_base, "_RESUME_RETRY_BACKOFF_S", (0.0, 0.0, 0.0))
    server.replies["/resume"] = deque([500, 500])

    push(client, [("a", torch.zeros(4, dtype=torch.bfloat16))])

    assert server.paused is False and server.count("POST /resume") == 3


def test_a_resume_that_never_lands_raises_naming_the_server(server, client, monkeypatch):
    monkeypatch.setattr(wsc, "broadcast_packed", _no_op_producer)
    monkeypatch.setattr(client_base, "_RESUME_RETRY_BACKOFF_S", (0.0, 0.0, 0.0))
    server.status["/resume"] = 500

    with pytest.raises(RuntimeError, match="stayed paused after a good weight push") as refused:
        client.update_named_param("a", torch.zeros(4, dtype=torch.bfloat16))
        client.finish_weight_update()

    assert server.url in str(refused.value) and server.count("POST /resume") == 4
    assert client._unusable_reason is None, "a good push holds the new policy: lifting the pause by hand serves it"


def test_a_failed_tail_chunk_after_a_streamed_one_is_not_resumed(server, client, monkeypatch):
    """The close resumes the engine in a ``finally``, so the verdict has to be taken where the chunk fails."""
    monkeypatch.setattr(wsc, "broadcast_packed", _no_op_producer)
    monkeypatch.setattr("halo.weight_sync.clients.base.WEIGHT_SYNC_CHUNK_BYTES", 512)

    for name in ("a", "b"):
        client.update_named_param(name, torch.zeros(512, dtype=torch.bfloat16))
    assert client._chunks_sent == 1, "nothing was streamed before the tail — the test is vacuous"

    server.status["/update_weights"] = 500
    with pytest.raises(requests.exceptions.HTTPError, match="500"):
        client.finish_weight_update()

    assert server.count("POST /resume") == 0, "the engine was resumed holding a half-written model"
    assert server.paused is True
    assert server.update_open is False, "the reload phase was left open by the failed tail"
    with pytest.raises(RuntimeError, match="RESTART"):
        push(client, [("d", torch.zeros(4, dtype=torch.bfloat16))])


def test_an_abort_before_any_chunk_still_closes_and_resumes_the_server(server, client, monkeypatch):
    """With nothing on the wire the engine still holds its weights; left quiesced it refuses every later sync and
    queues every rollout."""
    monkeypatch.setattr(wsc, "broadcast_packed", _no_op_producer)
    monkeypatch.setattr("halo.weight_sync.clients.base.WEIGHT_SYNC_CHUNK_BYTES", 512)
    server.status["/update_weights"] = 500

    client.update_named_param("a", torch.zeros(512, dtype=torch.bfloat16))
    with pytest.raises(requests.exceptions.HTTPError, match="500"):
        client.update_named_param("b", torch.zeros(512, dtype=torch.bfloat16))
    assert client._chunks_sent == 0 and client.can_replay_sync, "the first chunk landed — wrong scenario"

    client.abort_weight_update()

    assert server.update_open is False, "the reload phase was left open behind a failed first chunk"
    assert server.paused is False, "the engine stayed paused although nothing had been streamed"
    server.status.clear()
    push(client, [("w", torch.zeros(4, dtype=torch.bfloat16))])  # the client is still usable


def test_a_failed_snapshot_completion_still_closes_the_update(server, client, monkeypatch):
    """The completion runs inside the close. Skipping ``/finish_weight_update`` leaves the engine mid-reload while
    ``/health`` answers 200, so every later ``/start_weight_update`` fails and the probe cannot see it."""
    monkeypatch.setattr(wsc, "broadcast_packed", _no_op_producer)

    def sticky_cuda_error(self):
        raise RuntimeError("CUDA error: an illegal memory access was encountered")

    monkeypatch.setattr(VLLMWeightSyncClient, "complete_snapshots", sticky_cuda_error)

    client.update_named_param("w", torch.zeros(4, dtype=torch.bfloat16))
    with pytest.raises(RuntimeError, match="illegal memory access"):
        client.finish_weight_update()

    assert server.update_open is False, "the layerwise reload was left open by the failed snapshot completion"
    assert server.paused is False, "the engine was left quiesced by the failed snapshot completion"
    assert client._update_open is False


def test_a_dropped_tail_after_a_streamed_chunk_is_not_resumed(server, client, monkeypatch):
    """The engine holds the earlier chunks and its old weights for the rest, the same mixed model, so the close must
    not resume it."""
    monkeypatch.setattr(wsc, "broadcast_packed", _no_op_producer)
    monkeypatch.setattr("halo.weight_sync.clients.base.WEIGHT_SYNC_CHUNK_BYTES", 512)

    for name in ("a", "b"):
        client.update_named_param(name, torch.zeros(512, dtype=torch.bfloat16))
    assert client._chunks_sent == 1, "nothing was streamed before the tail — the test is vacuous"

    def sticky_cuda_error(self):
        raise RuntimeError("CUDA error: an illegal memory access was encountered")

    monkeypatch.setattr(VLLMWeightSyncClient, "complete_snapshots", sticky_cuda_error)
    with pytest.raises(RuntimeError, match="illegal memory access"):
        client.finish_weight_update()

    assert server.update_open is False, "the reload phase was left open"
    assert server.count("POST /resume") == 0 and server.paused is True, (
        "the engine was resumed holding the streamed chunks and nothing of the dropped tail"
    )


def test_the_snapshot_completion_waits_on_the_staged_copies_not_the_device(client, monkeypatch):
    """A whole-device synchronize also waits for the previous chunk's broadcasts and, in a multi-server fan-out, every
    other server's, which serializes each chunk's tail with the next."""
    synchronized: list = []
    monkeypatch.setattr(torch.cuda, "synchronize", lambda device=None: synchronized.append(device))
    client._sync_device = torch.device("cuda", 3)  # a guarded whole-device sync would fire on it

    class _Event:
        def __init__(self):
            self.waited = 0

        def synchronize(self):
            self.waited += 1

    event = _Event()
    client._snapshot_event = event
    client.complete_snapshots()
    assert event.waited == 1, "the flush did not wait for the staged copies"
    assert synchronized == [], "the flush synchronized the whole device instead of the staged copies"

    client._snapshot_event = None
    client.complete_snapshots()
    assert synchronized == [] and event.waited == 1


def test_check_server_honours_the_deadline_on_a_non_200_health(server, monkeypatch):
    monkeypatch.setattr(wsc, "HTTP_PROBE_TIMEOUT_S", 5)
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
    assert _run_bounded(client.probe_generation) == [None]


def _record_session_gets(client, monkeypatch) -> list[str]:
    seen: list[str] = []
    real_request = client._session.request

    def spy(method, url, *args, **kwargs):
        if method == "GET":
            seen.append(url)
        return real_request(method, url, *args, **kwargs)

    monkeypatch.setattr(client._session, "request", spy)
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
    assert client.served_max_model_len() is None


def test_probe_generation_resolves_the_served_model_over_the_session(server, client, monkeypatch):
    session_gets = _record_session_gets(client, monkeypatch)

    client.probe_generation()

    assert [url for url in session_gets if url.endswith("/v1/models")], (
        "probe_generation resolved the served model outside the client's session"
    )


def test_sglang_replicas_are_refused_while_attention_dp_sizes_the_group_off_its_tp_size():
    """Replicas restart ``tp_rank`` at 0 and collide in the update group. Attention DP, which the pinned SGLang resolves
    to ``dp_size`` 1 from either spelling, keeps ``tp_rank`` over every GPU."""
    server = FakeSGLangServer()
    server.tp_size = 4
    try:
        client = SGLangWeightSyncClient(base_url=server.url)
        assert client.require_engine_world_size() == 4
        server.dp_size = 2
        with pytest.raises(RuntimeError, match="plain --dp-size 2 server") as refused:
            client.require_engine_world_size()
    finally:
        server.close()
    assert "--attn-dp-size" in str(refused.value) and "--enable-dp-attention" not in str(refused.value)


def test_a_paused_sglang_server_is_refused_with_the_route_that_lifts_it(monkeypatch):
    """SGLang has no pause read and a paused engine queues requests, so each startup probe waited out 60 s, passed
    unread, and the first push died on a bare Bad Request."""
    monkeypatch.setattr(client_base, "GENERATION_PROBE_TIMEOUT_S", 0.5)
    server = FakeSGLangServer()
    server.hold.add("/v1/completions")
    try:
        client = SGLangWeightSyncClient(base_url=server.url)
        with pytest.raises(RuntimeError, match="paused or wedged") as refused:
            client.require_engine_world_size()
    finally:
        server.close()
    assert "/continue_generation" in str(refused.value) and "RESTARTED" in str(refused.value)


def test_a_paused_server_is_refused_by_the_preflight_on_every_engine(monkeypatch):
    monkeypatch.setattr(client_base, "GENERATION_PROBE_TIMEOUT_S", 0.5)
    for client_cls in (VLLMWeightSyncClient, SGLangWeightSyncClient):
        server = fake_server_class(client_cls)()
        server.paused = True
        server.hold.add("/v1/completions")
        try:
            with pytest.raises(ValueError, match="PAUSED|paused or wedged"):
                verify_context_window([server.url], 16, backend=client_cls.BACKEND_KEY, connection_timeout=5.0)
        finally:
            server.close()


def test_a_join_clears_a_dead_trainers_open_session_on_sglang(monkeypatch):
    """A second begin answers 400 until the session closes, and the close answers 400 with none open, which must not
    fail the join."""
    server = FakeSGLangServer()
    try:
        client = SGLangWeightSyncClient(base_url=server.url)
        server.session_open = True
        call = client.request_engine_join("127.0.0.1", 1, 2, 1, None)
        call.join(5.0)
        assert server.session_open is False
        call = client.request_engine_join("127.0.0.1", 1, 2, 1, None)
        call.join(5.0)
        assert server.count("POST /end_weight_update") == 2
    finally:
        server.close()


def test_a_control_routes_refusal_carries_its_reason():
    """``raise_for_status`` keeps the status alone, and 'Bad Request' names no stale session."""
    server = FakeSGLangServer()
    try:
        client = SGLangWeightSyncClient(base_url=server.url)
        with pytest.raises(requests.exceptions.HTTPError, match="must alternate"):
            client.request("POST", "/end_weight_update", retry=False, timeout=5, json={})
    finally:
        server.close()


def test_context_window_preflight_raises_on_a_server_that_cannot_hold_one_turn(server):
    server.max_model_len = 1024

    with pytest.raises(ValueError, match="context window"):
        _verify_context_window_on_rank0(VLLMWeightSyncClient, [server.url], 4096, None)

    _verify_context_window_on_rank0(VLLMWeightSyncClient, [server.url], 512, None)


# The filters' off values the preflight hands the probe, so each probe request moves one knob only.
_FILTERS_OFF = {"top_k": -1, "min_p": 0.0, "repetition_penalty": 1.0}


def _preflight(
    url: str,
    temperature: float,
    top_p: float,
    sequence_ratio_active: bool,
    *,
    top_k=-1,
    min_p=0.0,
    repetition_penalty=1.0,
):
    _verify_sampler_logprob_reference_on_rank0(
        VLLMWeightSyncClient, [url], temperature, top_p, top_k, min_p, repetition_penalty, sequence_ratio_active
    )


def test_sampler_logprob_probe_reads_each_engine_default(server, client):
    """vLLM's raw default carries neither temperature nor nucleus; its processed mode carries both;
    SGLang's default carries the temperature only."""
    assert probe_sampler_logprob_semantics(client, **_FILTERS_OFF) == (False, False)
    server.logprobs_mode = "processed_logprobs"
    assert probe_sampler_logprob_semantics(client, **_FILTERS_OFF) == (True, True)
    server.logprobs_mode = "sglang_default"
    assert probe_sampler_logprob_semantics(client, **_FILTERS_OFF) == (True, False)


def test_sampler_logprob_preflight_refuses_a_raw_reference_at_any_temperature_but_one(server):
    """Raw (pre-temperature) logprobs make every IS weight π^T/π^1, so they are refused at any other temperature."""
    with pytest.raises(ValueError, match="processed_logprobs"):
        _preflight(server.url, 1.1, 1.0, True)
    with pytest.raises(ValueError, match="RAW"):
        _preflight(server.url, 0.7, 1.0, False)
    # At temperature 1 with no band a raw reference is the sampling distribution's: nothing to probe.
    probes_before = server.count("POST /v1/completions")
    _preflight(server.url, 1.0, 0.95, False, top_k=20, min_p=0.05)
    assert server.count("POST /v1/completions") == probes_before, "temperature 1 without a band must not spend a probe"

    server.logprobs_mode = "processed_logprobs"
    _preflight(server.url, 1.1, 1.0, True)


def test_sampler_logprob_preflight_refuses_a_nucleus_reference_under_the_geometric_band(server):
    """A nucleus-renormalized reference with top-p < 1 shifts every uncertain position by its nucleus mass, which the
    geometric band reads as drift."""
    server.logprobs_mode = "processed_logprobs"
    with pytest.raises(ValueError, match="rollout_top_p: 1.0"):
        _preflight(server.url, 1.0, 0.95, True)
    _preflight(server.url, 1.0, 0.95, False)
    server.logprobs_mode = "sglang_default"
    _preflight(server.url, 1.1, 0.95, True)


@pytest.mark.parametrize("cut", [{"top_k": 20}, {"min_p": 0.05}], ids=["top_k", "min_p"])
def test_sampler_logprob_preflight_refuses_a_top_k_or_min_p_cut_like_a_nucleus(server, cut):
    """vLLM's processed logprobs are taken after top-k and min-p as after top-p. The probe runs with every filter off,
    so the cut it measures is its own top-p 0.5, not the run's."""
    server.logprobs_mode = "processed_logprobs"
    name = next(iter(cut))
    with pytest.raises(ValueError, match=f"{name}={cut[name]}"):
        _preflight(server.url, 1.0, 1.0, True, **cut)
    probes = [body for path, body in server.posted if path == "/v1/completions"]
    assert len(probes) == 3
    assert all({key: body[key] for key in _FILTERS_OFF} == _FILTERS_OFF for body in probes)
    _preflight(server.url, 1.0, 1.0, False, **cut)


def test_trl_s_top_k_off_value_is_no_cut(server):
    """TRL spells an off top-k 0, the rollout config -1."""
    server.logprobs_mode = "processed_logprobs"
    for off in (0, -1):
        _preflight(server.url, 1.0, 1.0, True, top_k=off)
    assert server.count("POST /v1/completions") == 0


def test_a_repetition_penalty_is_refused_under_a_sequence_summed_ratio_without_a_probe(server):
    """Both engines' sampling logprobs carry the penalty and the trainer's never do, so no server verdict clears it."""
    server.logprobs_mode = "sglang_default"
    with pytest.raises(ValueError, match="repetition_penalty=1.1"):
        _preflight(server.url, 1.0, 1.0, True, repetition_penalty=1.1)
    assert server.count("POST /v1/completions") == 0
    _preflight(server.url, 1.0, 1.0, False, repetition_penalty=1.1)


def test_sampler_logprob_preflight_warns_on_an_unreachable_server(server, caplog):
    """An unreadable probe must never be what fails the run."""
    server.close()
    with caplog.at_level(logging.WARNING):
        _preflight(server.url, 1.1, 1.0, True)
    assert any("could not verify" in r.getMessage() for r in caplog.records)


def test_context_window_preflight_skips_an_unreachable_server(server):
    server.close()

    _verify_context_window_on_rank0(VLLMWeightSyncClient, [server.url], 10**9, None)


@pytest.mark.parametrize("client_cls", [VLLMWeightSyncClient, SGLangWeightSyncClient], ids=["vllm", "sglang"])
def test_the_preflight_probe_reads_each_engines_server(client_cls, caplog):
    """The probe degrades any failure to a warning, so a client that cannot be built from a URL would pass every
    preflight without reading a server."""
    server = fake_server_class(client_cls)()
    try:
        server.max_model_len = 1024
        with caplog.at_level(logging.WARNING):
            _verify_context_window_on_rank0(client_cls, [server.url], 512, None)
            with pytest.raises(ValueError, match=r"context window \(1024\)"):
                _verify_context_window_on_rank0(client_cls, [server.url], 4096, None)
    finally:
        server.close()
    assert not [record.getMessage() for record in caplog.records if "Could not" in record.getMessage()]


def test_group_port_is_released_when_group_formation_fails(server, monkeypatch):
    """``ncclCommInitRank`` blocks, so the formation thread stays parked with its frame holding the unique-id store.
    Closing the store frees the listener only while the communicator keeps no reference to its ``TCPStore``; otherwise
    one wedged connect burns the port for the whole process."""
    monkeypatch.setattr(wsc, "HTTP_PROBE_TIMEOUT_S", 5)
    monkeypatch.setattr(client_base, "GROUP_FORMATION_TIMEOUT_S", 20)
    # The server's 500 is held until the communicator parked, so the failure lands with the init in flight.
    server.status["/init_weight_transfer_engine"] = 500
    server.hold.add("/init_weight_transfer_engine")
    library = _BlockingInitLibrary(on_parked=server.release)
    monkeypatch.setattr(pynccl, "NCCLLibrary", lambda: library)
    monkeypatch.setattr(pynccl, "vllm_pynccl_disabled", lambda: False)
    monkeypatch.setattr(torch.cuda, "device", lambda device: contextlib.nullcontext())
    port = free_port()
    handle = VLLMWeightSyncClient(base_url=server.url, group_port=port, connection_timeout=5.0)
    try:
        outcome = _run_bounded(lambda: handle.init_communicator(device="cpu"))
        assert library.parked.is_set(), "the communicator never reached ncclCommInitRank — nothing was parked"
        assert outcome, "init_communicator parked in the blocking group init"
        assert isinstance(outcome[0], Exception), f"a 500 from the server did not raise: {outcome[0]!r}"
        assert _bindable(port), f"group_port {port} still held after a failed formation: {_bind_error(port)}"
    finally:
        library.released.set()


def test_group_port_is_released_on_close(server, monkeypatch):
    """A listener surviving the close makes every later run on the configured group port fail before it starts."""
    monkeypatch.setattr(wsc, "HTTP_PROBE_TIMEOUT_S", 5)
    monkeypatch.setattr(client_base, "GROUP_FORMATION_TIMEOUT_S", 20)
    monkeypatch.setattr(wsc, "PyNcclCommunicator", _StoreHoldingCommunicator)
    server.world_size = 0
    port = free_port()

    handle = VLLMWeightSyncClient(base_url=server.url, group_port=port, connection_timeout=5.0)
    handle.init_communicator(device="cpu")
    assert not _bindable(port), "the group listener was never bound — the test proves nothing"

    # Pinned as a parked ncclCommInitRank thread would, so the release must not need a GC of the comm.
    pinned = handle._transport
    handle.close_communicator()

    assert _bindable(port), f"group_port {port} still held after close_communicator: {_bind_error(port)}"
    assert pinned.aborted, "close_communicator released the port without aborting the communicator"


def test_a_closed_store_cannot_publish():
    """A later publish must say so instead of raising AttributeError."""
    listener = RendezvousListener("127.0.0.1", 0)
    store = UniqueIdStore.create(host="127.0.0.1", listener=listener, world_size=1)
    store.close()
    store.close()  # idempotent: close runs from cleanup paths that may already have run
    with pytest.raises(RuntimeError, match="closed"):
        store.publish(ncclUniqueId())


def test_init_communicator_surfaces_a_server_side_init_failure(server, monkeypatch):
    """A 500 from ``/init_weight_transfer_engine`` must win over the blocking ``ncclCommInitRank``."""
    monkeypatch.setattr(wsc, "HTTP_PROBE_TIMEOUT_S", 5)
    monkeypatch.setattr(client_base, "GROUP_FORMATION_TIMEOUT_S", 20)
    handle = VLLMWeightSyncClient(base_url=server.url, connection_timeout=5.0)
    server.status["/init_weight_transfer_engine"] = 500
    server.world_size = 0  # the stub forms no store, so no engine worker may dial the listener

    released = threading.Event()
    closed: list[bool] = []
    stub_store = type("StubStore", (), {"close": lambda self: closed.append(True)})()
    monkeypatch.setattr(wsc.UniqueIdStore, "create", staticmethod(lambda **kwargs: stub_store))
    # ncclCommInitRank blocks until the peer joins; a server that failed never will.
    monkeypatch.setattr(wsc, "PyNcclCommunicator", lambda store, device: released.wait())

    outcome = _run_bounded(lambda: handle.init_communicator(device="cpu"))
    released.set()

    assert outcome, "init_communicator parked in the blocking group init with the HTTP error unread"
    error = str(outcome[0])
    assert "500" in error, f"the server's own failure was not surfaced: {error}"
    assert "VLLM_GROUP_HOST" not in error, f"a server-side 500 was misreported as a topology error: {error}"
    assert closed, "the store formed for a failed init was left holding its listener"


def test_init_communicator_refuses_an_engine_older_than_its_wire(server, monkeypatch):
    """An engine before the init-carried packed geometry would refuse the first update with a bare 500 mid-sync."""
    monkeypatch.setattr(wsc, "HTTP_PROBE_TIMEOUT_S", 5)
    server.version = "0.26.0"
    handle = VLLMWeightSyncClient(base_url=server.url, connection_timeout=5.0)

    with pytest.raises(RuntimeError, match=r"runs vLLM '0\.26\.0'"):
        handle.init_communicator(device="cpu")
    assert "POST /init_weight_transfer_engine" not in server.requests


def test_init_communicator_refuses_a_server_that_reports_no_version(server, monkeypatch):
    monkeypatch.setattr(wsc, "HTTP_PROBE_TIMEOUT_S", 5)
    server.version = None
    handle = VLLMWeightSyncClient(base_url=server.url, connection_timeout=5.0)

    with pytest.raises(RuntimeError, match="runs vLLM None"):
        handle.init_communicator(device="cpu")
    assert "POST /init_weight_transfer_engine" not in server.requests


def test_the_packed_geometry_rides_the_init_request(server, monkeypatch):
    """An update carrying the geometry is refused, and an init without it leaves the engine receiving tensor by tensor
    while the trainer sends packed."""
    monkeypatch.setattr(wsc, "HTTP_PROBE_TIMEOUT_S", 5)
    monkeypatch.setattr(client_base, "GROUP_FORMATION_TIMEOUT_S", 20)
    monkeypatch.setattr(wsc, "PyNcclCommunicator", _StoreHoldingCommunicator)
    server.world_size = 0
    handle = VLLMWeightSyncClient(base_url=server.url, connection_timeout=5.0)
    handle.init_communicator(device="cpu")
    try:
        init = server.bodies["/init_weight_transfer_engine"]["init_info"]
        assert init["packed"] is True
        assert (init["packed_buffer_size_bytes"], init["packed_num_buffers"]) == (
            PACKED_BUFFER_BYTES,
            PACKED_NUM_BUFFERS,
        )
    finally:
        handle.close_communicator()


def test_zero_dim_param_is_rejected_before_the_server_is_paused(server, client, monkeypatch):
    monkeypatch.setattr(wsc, "broadcast_packed", _no_op_producer)

    with pytest.raises(ValueError, match="0-dimensional"):
        push(client, [("scale", torch.zeros((), dtype=torch.bfloat16))])

    assert server.count("POST /pause") == 0, "the server was paused for a payload that can never be sent"
    assert server.paused is False


def test_dtensor_param_is_rejected(server, client, tmp_path, monkeypatch):
    """A DTensor's global ``numel()`` against a local ``data_ptr()`` would broadcast out of bounds."""
    monkeypatch.setattr(wsc, "broadcast_packed", _no_op_producer)
    dist.init_process_group("gloo", rank=0, world_size=1, init_method=f"file://{tmp_path / 'pg'}")
    try:
        mesh = init_device_mesh("cpu", (1,))
        sharded = distribute_tensor(torch.zeros(8, 4, dtype=torch.bfloat16), mesh, [Shard(0)])
        with pytest.raises(TypeError, match="DTensor"):
            push(client, [("w", sharded)])
    finally:
        dist.destroy_process_group()

    assert server.count("POST /pause") == 0


def test_close_aborts_the_communicator(server, client):
    """``PyNcclCommunicator`` has no ``__del__``, so a dropped-but-live communicator strands its device memory, one
    leak per reconnect and per served endpoint."""
    communicator = FakeCommunicator()
    client._transport = communicator

    client.close_communicator()

    assert communicator.aborted, "close_communicator dropped the communicator without aborting it"
    assert client._transport is None


def test_aborted_communicator_is_refused_by_name(server, client):
    """An aborted communicator holds a NULL handle; reusing it must not read as an argument bug."""
    client._transport = FakeCommunicator(aborted=True)

    with pytest.raises(RuntimeError, match="aborted"):
        push(client, [("w", torch.zeros(4, dtype=torch.bfloat16))])

    assert server.count("POST /pause") == 0, "the server was paused for a sync that cannot broadcast"


def test_a_communicator_aborted_mid_update_keeps_the_engine_out_of_the_next_broadcast(server, client, monkeypatch):
    """The RPC enters the engine's collective, where it would wait on a broadcast that never comes."""
    monkeypatch.setattr(wsc, "broadcast_packed", _no_op_producer)
    client.update_named_param("a", torch.zeros(4, dtype=torch.bfloat16))
    client.flush_chunk()
    assert server.count("POST /update_weights") == 1
    client._transport.aborted = True
    client.update_named_param("b", torch.zeros(4, dtype=torch.bfloat16))

    with pytest.raises(RuntimeError, match="aborted"):
        client.flush_chunk()

    assert server.count("POST /update_weights") == 1


def test_a_misplaced_snapshot_keeps_the_engine_out_of_the_broadcast(server, client):
    client._transport.device = torch.device("meta")
    client.update_named_param("w", torch.zeros(4, dtype=torch.bfloat16))

    with pytest.raises(RuntimeError, match="not on the sync device"):
        client.flush_chunk()

    assert server.count("POST /update_weights") == 0


def test_aborted_communicator_refuses_to_broadcast():
    comm = PyNcclCommunicator.__new__(PyNcclCommunicator)
    comm.aborted = True
    comm.device = torch.device("cpu")

    with pytest.raises(RuntimeError, match="aborted"):
        comm.broadcast(torch.zeros(1))


def test_a_tensor_off_the_communicators_device_is_refused():
    """NCCL reads the raw pointer on the communicator's device; a tensor elsewhere must fail by name."""
    comm = PyNcclCommunicator.__new__(PyNcclCommunicator)
    comm.aborted = False
    comm.device = torch.device("cuda", 0)

    with pytest.raises(ValueError, match="formed on cuda:0"):
        comm.broadcast(torch.zeros(1))


def test_an_engine_with_no_workers_is_refused():
    """The trainer alone in the group broadcasts to nobody, and the engine keeps serving the initial policy."""
    listener = RendezvousListener("127.0.0.1", 0)
    store = UniqueIdStore.create(host="127.0.0.1", listener=listener, world_size=1)
    try:
        with pytest.raises(RuntimeError, match="reported no workers"):
            PyNcclCommunicator(store, device=torch.device("cpu"))
    finally:
        store.close()


def test_score_completion_logprobs_returns_the_completion_slice_by_token_id():
    """The prompt_logprobs echo is keyed by token id (as a string) and aligned to the whole sequence."""
    server = FakeVLLMServer()
    try:
        client = offline_client(VLLMWeightSyncClient, server.url)
        values = client.score_completion_logprobs([11, 12, 13], [14, 15])
        assert values == [pytest.approx(-((14 % 7) + 1) / 10), pytest.approx(-((15 % 7) + 1) / 10)]
        assert server.bodies["/v1/completions"]["prompt"] == [11, 12, 13, 14, 15]
        assert server.bodies["/v1/completions"]["prompt_logprobs"] == 0
        assert client.served_model_id() == "fake-model"
    finally:
        server.close()


def test_a_server_listing_no_model_fails_by_name(client, monkeypatch):
    monkeypatch.setattr(client, "_served_model_cards", list)
    with pytest.raises(RuntimeError, match="lists no model"):
        client.served_model_id()


def test_sglang_score_completion_logprobs_reads_the_generate_prefill_slice():
    """SGLang's re-score goes through /generate with logprob_start_len on the last prompt token, whose entry anchors
    the window with no log-prob. The completion's entries follow as [logprob, token_id, text]."""
    server = FakeSGLangServer()
    try:
        client = offline_client(SGLangWeightSyncClient, server.url)
        values = client.score_completion_logprobs([11, 12, 13], [14, 15])
        assert values == [pytest.approx(-((14 % 7) + 1) / 10), pytest.approx(-((15 % 7) + 1) / 10)]
        body = server.bodies["/generate"]
        assert body["input_ids"] == [11, 12, 13, 14, 15] and body["logprob_start_len"] == 2
        assert body["return_logprob"] is True
    finally:
        server.close()


def test_sglang_score_completion_logprobs_refuses_a_window_that_does_not_open_on_the_prompt():
    """A window whose anchor is not the last prompt token is a shifted reference, never silently read."""
    server = FakeSGLangServer()
    try:
        client = offline_client(SGLangWeightSyncClient, server.url)
        with pytest.raises(ValueError, match="at least one prompt token"):
            client.score_completion_logprobs([], [14, 15])
    finally:
        server.close()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
