"""Weight-sync request-transcript golden: every HTTP request each engine's client sends through its public
lifecycle, step by step, in scenarios that succeed and that fail at each step, plus what each step puts on
the NCCL wire, does to the local group, and leaves registered at exit. A request is recorded as it leaves
the client (route, query, body, timeout, retrying session or single shot, retry policy) and counted where
the fake engine receives it, so a retry shows as a second receipt.
"""

from __future__ import annotations

import atexit
import dataclasses
import inspect
import json
import logging
import os
import threading
from collections import deque
from collections.abc import Callable, Iterator
from contextlib import ExitStack, contextmanager, nullcontext
from pathlib import Path
from typing import Literal
from unittest.mock import patch
from urllib.parse import urlencode, urlsplit

import requests
import torch
from urllib3.util.retry import Retry

from src.weight_sync.clients import base as client_base
from src.weight_sync.clients import sglang as sglang_client
from src.weight_sync.clients import vllm as vllm_client
from src.weight_sync.clients.base import WeightSyncClientBase
from src.weight_sync.clients.sglang import SGLangWeightSyncClient
from src.weight_sync.clients.vllm import VLLMWeightSyncClient
from src.weight_sync.transport import rendezvous
from tests.common.fake_engines import CALL_ID_HEADER, FakeEngineServer, fake_server_class
from tests.common.goldens import regen_main

GOLDEN_PATH = Path(__file__).with_suffix(".json")
# Small enough that the dense and MoE payloads stream in several chunks.
CHUNK_BYTES = 4096
DEVICE = "cpu"
# A request's retry policy is spelled as the fields it sets away from these.
_RETRY_DEFAULTS = Retry()
_RETRY_FIELDS = tuple(inspect.signature(Retry).parameters)

# (name, dtype, shape) per tensor, pushed in order. The MoE payload puts a chunk boundary between the
# halves of the MLA pair SGLang's loader fuses, and fused 3D experts beside an fp32 router bias.
PAYLOADS = {
    "single": (("model.norm.weight", torch.bfloat16, (64,)),),
    "dense": (
        ("model.embed_tokens.weight", torch.bfloat16, (64, 16)),
        ("model.layers.0.input_layernorm.weight", torch.bfloat16, (16,)),
        ("model.layers.0.self_attn.q_proj.weight", torch.bfloat16, (16, 16)),
        ("model.layers.0.self_attn.k_proj.weight", torch.bfloat16, (8, 16)),
        ("model.layers.0.self_attn.v_proj.weight", torch.bfloat16, (8, 16)),
        ("model.layers.0.self_attn.o_proj.weight", torch.bfloat16, (16, 16)),
        ("model.layers.0.post_attention_layernorm.weight", torch.bfloat16, (16,)),
        ("model.layers.0.mlp.gate_proj.weight", torch.bfloat16, (32, 16)),
        ("model.layers.0.mlp.up_proj.weight", torch.bfloat16, (32, 16)),
        ("model.layers.0.mlp.down_proj.weight", torch.bfloat16, (16, 32)),
        ("model.norm.weight", torch.bfloat16, (16,)),
        ("lm_head.weight", torch.bfloat16, (64, 16)),
    ),
    "moe": (
        ("model.embed_tokens.weight", torch.bfloat16, (64, 16)),
        ("model.layers.0.self_attn.q_a_proj.weight", torch.bfloat16, (8, 16)),
        ("model.layers.0.mlp.experts.gate_up_proj", torch.bfloat16, (4, 32, 16)),
        ("model.layers.0.self_attn.kv_a_proj_with_mqa.weight", torch.bfloat16, (12, 16)),
        ("model.layers.0.mlp.experts.down_proj", torch.bfloat16, (4, 16, 16)),
        ("model.layers.0.mlp.gate.weight", torch.bfloat16, (4, 16)),
        ("model.layers.0.mlp.gate.e_score_correction_bias", torch.float32, (4,)),
        ("model.norm.weight", torch.bfloat16, (16,)),
        ("lm_head.weight", torch.bfloat16, (64, 16)),
    ),
}


# How the trainer-side group formation goes (:class:`_Transport`).
Formation = Literal["ok", "fail", "block"]


class _Stream:
    def __init__(self, device=None):
        pass

    def wait_stream(self, other) -> None:
        pass


class _Event:
    def record(self, stream=None) -> None:
        pass

    def query(self) -> bool:
        return True

    def synchronize(self) -> None:
        pass


class _Communicator:
    """The vLLM client's NCCL communicator: carries its device and reports its abort."""

    def __init__(self, transport: _Transport, device):
        self.device = torch.device(device)
        self.aborted = False
        self._transport = transport

    def abort(self) -> None:
        self.aborted = True
        self._transport.events.append("abort communicator")


class _Transport:
    """The NCCL half of one client: forms its group as ``formation`` says and records what reaches it.

    ``ok`` forms it. ``fail`` raises once ``engine_asked_to_join`` returns, as a formation failing after
    the engine dialed in, so the record does not race that request. ``block`` waits for :meth:`release`,
    as an init whose peer never joins, then raises.
    """

    def __init__(self, formation: Formation, engine_asked_to_join: Callable[[], None]):
        self.formation = formation
        self.events: list[str] = []
        self._engine_asked_to_join = engine_asked_to_join
        self._released = threading.Event()

    def _form(self, what: str) -> None:
        if self.formation == "fail":
            self._engine_asked_to_join()
            raise RuntimeError("NCCL error: unhandled system error")
        if self.formation == "block":
            self._released.wait()
            raise RuntimeError("the engine never joined the group")
        self.events.append(f"form {what}")

    def send(self, tensor: torch.Tensor) -> None:
        self.events.append(f"send {str(tensor.dtype).removeprefix('torch.')}{list(tensor.shape)}")

    def form_communicator(self, store, device) -> _Communicator:
        self._form("communicator")
        return _Communicator(self, device)

    def packed_producer(self, *, iterator, **kwargs) -> None:
        for _, tensor in iterator:
            self.send(tensor)

    def form_group(self, *args, **kwargs) -> tuple[object, object]:
        self._form("group")
        return object(), object()

    def broadcast(self, tensor: torch.Tensor, src: int, group) -> None:
        self.send(tensor)

    def release_group(self, group, *, destroy: bool) -> None:
        if group is not None:
            self.events.append("destroy group" if destroy else "forget group")

    def release(self) -> None:
        self._released.set()


def _vllm_transport(transport: _Transport) -> list:
    return [
        patch.object(vllm_client, "PyNcclCommunicator", transport.form_communicator),
        patch.object(vllm_client, "packed_broadcast_producer", transport.packed_producer),
    ]


def _sglang_transport(transport: _Transport) -> list:
    return [
        patch.object(sglang_client, "build_weight_update_group", transport.form_group),
        patch.object(sglang_client, "release_weight_update_group", transport.release_group),
        patch.object(torch.distributed, "broadcast", transport.broadcast),
        # The arena is a CPU tensor here, and the real call takes CUDA tensors only.
        patch.object(torch.Tensor, "record_stream", lambda tensor, stream: None),
        patch.object(torch.cuda, "Stream", _Stream),
        patch.object(torch.cuda, "current_stream", lambda device=None: _Stream()),
        patch.object(torch.cuda, "stream", lambda stream: nullcontext()),
        patch.object(torch.cuda, "Event", _Event),
    ]


@dataclasses.dataclass(frozen=True)
class Engine:
    """One engine's client, the patches that put its transport on the CPU, and the routes a scenario
    scripts failures on."""

    client_class: type[WeightSyncClientBase]
    transport_patches: Callable[[_Transport], list]
    init_route: str
    declaration_route: str
    finish_route: str


ENGINES = {
    "vllm": Engine(
        VLLMWeightSyncClient,
        _vllm_transport,
        init_route="/init_weight_transfer_engine",
        declaration_route="/update_weights",
        finish_route="/finish_weight_update",
    ),
    "sglang": Engine(
        SGLangWeightSyncClient,
        _sglang_transport,
        init_route="/init_weights_update_group",
        declaration_route="/update_weights_from_distributed",
        finish_route="/end_weight_update",
    ),
}


def _no_failures(engine: Engine, server: FakeEngineServer) -> None:
    pass


def _fail_init(engine: Engine, server: FakeEngineServer) -> None:
    server.status[engine.init_route] = 500


def _hold_init(engine: Engine, server: FakeEngineServer) -> None:
    server.hold.add(engine.init_route)


def _fail_finish_once(engine: Engine, server: FakeEngineServer) -> None:
    server.replies[engine.finish_route] = deque([500])


def _fail_finish(engine: Engine, server: FakeEngineServer) -> None:
    server.status[engine.finish_route] = 500


def _fail_first_declaration(engine: Engine, server: FakeEngineServer) -> None:
    server.replies[engine.declaration_route] = deque([500])


def _fail_second_declaration(engine: Engine, server: FakeEngineServer) -> None:
    server.replies[engine.declaration_route] = deque([200, 500])


@dataclasses.dataclass(frozen=True)
class Scenario:
    """``steps``, each a :class:`Driver` method, run in order whatever each one raises, pushing ``payload``.

    ``script`` sets the server's failures, ``formation`` how the trainer-side group forms
    (:class:`_Transport`), and ``formation_thread_fails`` fails the start of that formation's thread.
    """

    key: str
    payload: str
    steps: tuple[str, ...]
    script: Callable[[Engine, FakeEngineServer], None] = _no_failures
    formation: Formation = "ok"
    formation_thread_fails: bool = False


_SYNC = ("construct", "init_communicator", "update_named_param", "finish_weight_update")
_FAILED_INIT = ("construct", "init_communicator", "close_communicator")
_ABORTED = ("construct", "init_communicator", "update_named_param", "abort_weight_update", "close_communicator")

SCENARIOS = (
    Scenario("dense sync", "dense", (*_SYNC, "close_communicator")),
    Scenario("moe sync", "moe", (*_SYNC, "close_communicator")),
    Scenario(
        "formation fails on the server before the trainer formed",
        "single",
        _FAILED_INIT,
        script=_fail_init,
        formation="block",
    ),
    Scenario("formation fails on the server after the trainer formed", "single", _FAILED_INIT, script=_fail_init),
    Scenario("formation fails on the trainer", "single", _FAILED_INIT, script=_hold_init, formation="fail"),
    Scenario("formation thread fails to start", "single", _FAILED_INIT, formation_thread_fails=True),
    Scenario("finish fails once", "single", (*_SYNC, "close_communicator"), script=_fail_finish_once),
    Scenario("finish fails every time", "single", (*_SYNC, "close_communicator"), script=_fail_finish),
    Scenario("abort before any chunk landed", "dense", _ABORTED, script=_fail_first_declaration),
    Scenario("mid-stream abort", "dense", _ABORTED, script=_fail_second_declaration),
    Scenario(
        "unusable client at exit",
        "dense",
        ("construct", "init_communicator", "update_named_param", "at_exit"),
        script=_fail_second_declaration,
    ),
    Scenario("at exit after a sync", "single", (*_SYNC, "at_exit")),
)


@dataclasses.dataclass(frozen=True)
class _Call:
    method: str
    url: str
    params: dict | None
    body: object
    timeout: object
    session: requests.Session
    retry: Retry


class _RequestRecorder:
    """Records every ``requests`` call as it leaves the client, and stamps the request with its index."""

    def __init__(self):
        self.calls: list[_Call] = []
        self._lock = threading.Lock()

    @contextmanager
    def recording(self) -> Iterator[None]:
        send = requests.Session.request

        def recorded(session, method, url, **kwargs):
            retry = session.get_adapter(url).max_retries
            call = _Call(
                method.upper(), url, kwargs.get("params"), kwargs.get("json"), kwargs.get("timeout"), session, retry
            )
            with self._lock:
                call_id = str(len(self.calls))
                self.calls.append(call)
            headers = {**(kwargs.pop("headers", None) or {}), CALL_ID_HEADER: call_id}
            return send(session, method, url, headers=headers, **kwargs)

        with patch.object(requests.Session, "request", recorded):
            yield


def _normalized(value, server: FakeEngineServer):
    """``value`` with what the host picks written as what it is: the group port, the server's address."""
    if isinstance(value, dict):
        return {
            key: "<group port>" if key == "master_port" else _normalized(item, server) for key, item in value.items()
        }
    if isinstance(value, list):
        return [_normalized(item, server) for item in value]
    if isinstance(value, str):
        return value.replace(server.netloc, "<server>")
    return value


def _retry_spelling(retry: Retry) -> str:
    """``retry`` as every field it sets away from urllib3's defaults, by name; collections sorted."""
    fields = []
    for name in _RETRY_FIELDS:
        value = getattr(retry, name)
        if value == getattr(_RETRY_DEFAULTS, name):
            continue
        if isinstance(value, (list, tuple, set, frozenset)):
            value = f"[{','.join(str(item) for item in sorted(value))}]"
        fields.append(f"{name}={value}")
    return f"retry({','.join(fields)})"


def _request_line(call_id: int, call: _Call, client: WeightSyncClientBase | None, server: FakeEngineServer) -> str:
    url = urlsplit(call.url)
    query = "&".join(part for part in (url.query, urlencode(call.params or {})) if part)
    parts = [call.method, url.path + (f"?{query}" if query else "")]
    if call.body is not None:
        parts.append(json.dumps(_normalized(call.body, server), sort_keys=True, separators=(",", ":")))
    parts.append(
        f"timeout={call.timeout:g}" if isinstance(call.timeout, (int, float)) else f"timeout={call.timeout!r}"
    )
    kind = "session" if client is not None and call.session is client._session else "single-shot"
    parts.append(f"{kind} {_retry_spelling(call.retry)}")
    received = server.received(str(call_id))
    if received != 1:
        parts.append(f"received={received}")
    return " ".join(parts)


@contextmanager
def _failing_formation_thread(engine_asked_to_join: Callable[[], None]) -> Iterator[None]:
    """Fail the start of the second thread the caller starts: an init starts its request to the engine
    first, then its trainer-side group formation. It fails once ``engine_asked_to_join`` returns, so the
    record does not race the engine's request."""
    caller = threading.current_thread()
    start = threading.Thread.start
    started = 0

    def failing_start(thread):
        nonlocal started
        if threading.current_thread() is caller:
            started += 1
            if started == 2:
                engine_asked_to_join()
                raise RuntimeError("can't start new thread")
        return start(thread)

    with patch.object(threading.Thread, "start", failing_start):
        yield


class Driver:
    """The client's public lifecycle against one fake server: one method per scenario step, then the
    process the client runs in."""

    def __init__(self, engine: Engine, server: FakeEngineServer, scenario: Scenario):
        self.engine = engine
        self.server = server
        self.scenario = scenario
        self.client: WeightSyncClientBase | None = None
        # (fn, args, kwargs) per at-exit hook a weight-sync client registered, in order.
        self.at_exit_hooks: list[tuple[Callable, tuple, dict]] = []

    def construct(self) -> None:
        self.client = self.engine.client_class(base_url=self.server.url)

    def init_communicator(self) -> None:
        failing = self.scenario.formation_thread_fails
        with _failing_formation_thread(self.engine_asked_to_join) if failing else nullcontext():
            self.client.init_communicator(device=DEVICE)

    def update_named_param(self) -> None:
        for name, dtype, shape in PAYLOADS[self.scenario.payload]:
            self.client.update_named_param(name, torch.zeros(shape, dtype=dtype))

    def finish_weight_update(self) -> None:
        self.client.finish_weight_update()

    def abort_weight_update(self) -> None:
        self.client.abort_weight_update()

    def close_communicator(self) -> None:
        self.client.close_communicator()

    def at_exit(self) -> None:
        for fn, args, kwargs in reversed(list(self.at_exit_hooks)):
            fn(*args, **kwargs)

    def engine_asked_to_join(self) -> None:
        """Block until the engine received the client's request to join the group."""
        self.server.wait_for_request(f"POST {self.engine.init_route}")

    @contextmanager
    def process(self, transport: _Transport) -> Iterator[None]:
        """The client's process: its transport on the CPU, its at-exit hooks captured, the chunk budget at
        :data:`CHUNK_BYTES`, and no override of the group address or its listener in the environment."""
        register_at_exit, unregister_at_exit = atexit.register, atexit.unregister

        def registers_a_client(fn) -> bool:
            return isinstance(getattr(fn, "__self__", None), WeightSyncClientBase)

        def register(fn, *args, **kwargs):
            if not registers_a_client(fn):
                return register_at_exit(fn, *args, **kwargs)
            self.at_exit_hooks.append((fn, args, kwargs))
            return fn

        def unregister(fn) -> None:
            if registers_a_client(fn):
                self.at_exit_hooks[:] = [hook for hook in self.at_exit_hooks if hook[0] != fn]
            else:
                unregister_at_exit(fn)

        patches = [
            patch.dict(os.environ),
            patch.object(atexit, "register", register),
            patch.object(atexit, "unregister", unregister),
            # SGLang binds the budget by name: it sizes the staging arena.
            *(patch.object(module, "WEIGHT_SYNC_CHUNK_BYTES", CHUNK_BYTES) for module in (client_base, sglang_client)),
            *self.engine.transport_patches(transport),
        ]
        with ExitStack() as stack:
            for patcher in patches:
                stack.enter_context(patcher)
            for name in (self.engine.client_class.GROUP_HOST_ENV, rendezvous._BIND_ALL_ENV):
                os.environ.pop(name, None)
            yield


def scenario_record(engine_key: str, scenario: Scenario) -> list[dict]:
    """Run ``scenario`` against ``engine_key``'s fake server: one record per step, with its outcome (``ok``
    or the type it raised), its requests, its transport events and the at-exit hooks held after it."""
    engine = ENGINES[engine_key]
    server = fake_server_class(engine.client_class)()
    driver = Driver(engine, server, scenario)
    transport = _Transport(scenario.formation, driver.engine_asked_to_join)
    recorder = _RequestRecorder()
    scenario.script(engine, server)
    record = []
    try:
        with driver.process(transport), recorder.recording():
            for step in scenario.steps:
                first_call, first_event = len(recorder.calls), len(transport.events)
                try:
                    getattr(driver, step)()
                    outcome = "ok"
                except Exception as error:  # the outcome is part of the record
                    outcome = type(error).__name__
                calls = range(first_call, len(recorder.calls))
                record.append(
                    {
                        "step": step,
                        "outcome": outcome,
                        "requests": [_request_line(i, recorder.calls[i], driver.client, server) for i in calls],
                        "transport": transport.events[first_event:],
                        "at_exit_hooks": len(driver.at_exit_hooks),
                    }
                )
    finally:
        transport.release()
        server.close()
    return record


def render_golden() -> str:
    # The scenarios that fail on purpose log their failures, which the record already holds as outcomes.
    previous = logging.root.manager.disable
    logging.disable(logging.ERROR)
    try:
        record = {
            engine: {scenario.key: scenario_record(engine, scenario) for scenario in SCENARIOS} for engine in ENGINES
        }
    finally:
        logging.disable(previous)
    return json.dumps(record, indent=2) + "\n"


if __name__ == "__main__":
    regen_main(GOLDEN_PATH, render_golden)
