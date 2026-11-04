"""The SGLang weight-sync client's teardown contract. Under NCCL's cuMem transports each side's finalize waits for the
other, so the engine-side release runs during the local destroy. At exit the destroy is skipped, as it would block."""

import atexit
import logging
import threading
from types import SimpleNamespace

import pytest
import requests

import halo.weight_sync.clients.base as client_base
import halo.weight_sync.clients.sglang as sglang_module
from halo.weight_sync.clients.sglang import SGLangWeightSyncClient
from halo.weight_sync.transport import c10d_group
from tests.common.weight_sync import offline_client


def _teardown_probe(client, monkeypatch) -> list[tuple[str, object]]:
    calls: list[tuple[str, object]] = []
    monkeypatch.setattr(
        sglang_module,
        "release_weight_update_group",
        lambda group, *, destroy: calls.append(("destroy" if destroy else "drop", group)),
    )
    monkeypatch.setattr(client, "_destroy_remote_group", lambda: calls.append(("remote", None)))
    monkeypatch.setattr(client, "_engine_answers", lambda: True)
    return calls


def test_the_pause_is_lifted_on_sglangs_own_route_with_a_body(monkeypatch):
    """``/continue_generation`` parses a request dataclass, so an empty body is rejected, which surfaces only as a
    warning from ``_lift_pause`` and leaves the engine quiesced."""
    client = offline_client(SGLangWeightSyncClient)
    client._paused = True
    posts: list[tuple[str, dict]] = []
    monkeypatch.setattr(client, "request", lambda method, path, **kwargs: posts.append((path, kwargs)))

    client._lift_pause(timeout=1.0, context="test")

    assert [path for path, _ in posts] == ["/continue_generation"], posts
    assert posts[0][1].get("json") == {}, "the handler rejects an empty body — RESUME_REQUEST must ship {}"
    assert client._paused is False


def test_a_paused_engine_is_reported_as_paused_and_not_as_unreachable(monkeypatch):
    """A pause holds SGLang's ``/health``, which generates through the scheduler, and the wait for a server would run
    out its whole deadline and then advise waiting longer. ``/v1/models`` stays up, so the generation probe names it."""
    client = offline_client(SGLangWeightSyncClient)

    def request(method, path, **kwargs):
        if path in ("/health", "/v1/completions"):
            raise requests.exceptions.ReadTimeout(path)
        return SimpleNamespace(status_code=200, json=lambda: {"data": [{"id": "served"}]})

    monkeypatch.setattr(client, "request", request)

    client.check_server(0.0)
    with pytest.raises(RuntimeError, match="paused or wedged"):
        client.probe_generation()


class _ColdEngine:
    """SGLang while its warm-up generate runs: ``/v1/models`` answers, ``/health`` refuses at once, and a generation
    probe waits past its timeout behind that first forward. The warm-up ends after ``warmup_polls`` looks at it."""

    def __init__(self, warmup_polls: int):
        self.polls_left = warmup_polls
        self.health_polls = 0

    def request(self, method, path, **kwargs):
        if path == "/health":
            self.health_polls += 1
            self.polls_left -= 1
            return SimpleNamespace(status_code=503 if self.polls_left >= 0 else 200)
        if path == "/v1/completions" and self.polls_left >= 0:
            raise requests.exceptions.ReadTimeout(path)
        return SimpleNamespace(status_code=200, json=lambda: {"data": [{"id": "served"}]})


@pytest.fixture
def no_sleep(monkeypatch):
    monkeypatch.setattr(client_base.time, "sleep", lambda _seconds: None)


def test_a_server_in_its_warm_up_is_waited_for_and_not_reported_as_wedged(monkeypatch, no_sleep):
    """``/v1/models`` answers once the HTTP server is up, so stopping there sends the generation probe into the warm-up's
    first forward, and a cold kernel cache reads as a pause a dead trainer left."""
    client = offline_client(SGLangWeightSyncClient)
    engine = _ColdEngine(warmup_polls=3)
    monkeypatch.setattr(client, "request", engine.request)

    client.check_server(60.0)
    client.probe_generation()

    assert engine.health_polls == 4


def test_a_warm_up_past_the_deadline_says_the_server_is_still_starting(monkeypatch, no_sleep):
    client = offline_client(SGLangWeightSyncClient)
    monkeypatch.setattr(client, "request", _ColdEngine(warmup_polls=10**6).request)

    with pytest.raises(ConnectionError, match="its start has not finished.*needs a longer wait"):
        client.check_server(0.0)


def test_a_health_route_that_stops_connecting_does_not_read_as_started(monkeypatch, no_sleep):
    client = offline_client(SGLangWeightSyncClient)
    answers = iter([requests.exceptions.ConnectionError("refused"), SimpleNamespace(status_code=200)])

    def request(method, path, **kwargs):
        if path != "/health":
            return SimpleNamespace(status_code=200)
        answer = next(answers)
        if isinstance(answer, Exception):
            raise answer
        return answer

    monkeypatch.setattr(client, "request", request)

    client.check_server(60.0)

    assert next(answers, None) is None, "the wait stopped at the refused connection"


@pytest.mark.parametrize(("answers_after_s", "answers"), [(15.0, True), (25.0, True), (100.0, False)])
def test_an_engine_answering_inside_its_own_health_window_reads_as_alive(monkeypatch, answers_after_s, answers):
    """The server's ``/health`` allows a scheduler step 20 s. A shorter wait reads a busy engine as dead, the client
    aborts, and the engine's half of the destroy stays blocked."""
    client = offline_client(SGLangWeightSyncClient)

    def request(method, path, *, timeout, **kwargs):
        if timeout < answers_after_s:
            raise requests.exceptions.ReadTimeout(path)
        return SimpleNamespace(status_code=200)

    monkeypatch.setattr(client, "request", request)

    assert client._engine_answers() is answers


def test_the_engine_is_asked_before_its_half_of_the_destroy_starts(monkeypatch):
    """SGLang's ``/health`` generates through the scheduler the remote destroy occupies. Asked during it, a healthy
    engine reads as dead, the client aborts, and the engine's half of the destroy stays blocked for good."""
    client = offline_client(SGLangWeightSyncClient)
    client._transport = (object(), None)
    _teardown_probe(client, monkeypatch)
    order: list[str] = []
    monkeypatch.setattr(client, "_destroy_remote_group", lambda: order.append("remote"))
    monkeypatch.setattr(client, "_engine_answers", lambda: order.append("health") or True)

    client._release_transport(at_exit=False)

    assert order == ["health", "remote"]


def test_explicit_close_keeps_the_full_local_destroy(monkeypatch):
    client = offline_client(SGLangWeightSyncClient)
    group = object()
    client._transport = (group, None)
    calls = _teardown_probe(client, monkeypatch)

    client.close_communicator()

    assert ("destroy", group) in calls, "the reconnect path must destroy the local group or its name/port leak"
    assert all(kind != "drop" for kind, _ in calls)
    assert ("remote", None) in calls
    assert client._transport is None


def test_the_engine_side_release_runs_while_the_local_destroy_blocks(monkeypatch):
    """The local destroy blocks until the engine drops its half, so the engine must already have been asked by then."""
    client = offline_client(SGLangWeightSyncClient)
    client._transport = (object(), None)
    monkeypatch.setattr(client, "_engine_answers", lambda: True)
    remote_started = threading.Event()
    monkeypatch.setattr(client, "_destroy_remote_group", remote_started.set)
    seen_during_destroy: list[bool] = []
    monkeypatch.setattr(
        sglang_module,
        "release_weight_update_group",
        lambda group, *, destroy: seen_during_destroy.append(destroy and remote_started.wait(timeout=2.0)),
    )

    client.close_communicator()

    assert seen_during_destroy == [True], (
        "the engine-side /destroy_weights_update_group did not start while the local destroy ran — "
        "under cuMem transports that order parks destroy_process_group forever"
    )


def _aborts(monkeypatch) -> list[object]:
    aborted: list[object] = []
    monkeypatch.setattr(sglang_module, "abort_weight_update_group", aborted.append)
    return aborted


def test_a_dead_engine_is_aborted_not_destroyed(monkeypatch):
    """A destroy waits for an engine that will never enter its half, so the failure already recorded never raises and
    the job sits until something external kills it."""
    client = offline_client(SGLangWeightSyncClient)
    group = object()
    client._transport = (group, None)
    calls = _teardown_probe(client, monkeypatch)
    aborted = _aborts(monkeypatch)
    monkeypatch.setattr(client, "_engine_answers", lambda: False)

    client.close_communicator()

    assert aborted == [group]
    assert all(kind != "destroy" for kind, _ in calls), "close entered the destroy with no engine to answer it"
    assert ("drop", group) in calls


def test_a_destroy_blocked_past_its_deadline_is_aborted(monkeypatch):
    """A wedged engine answers its health route and never enters the destroy."""
    client = offline_client(SGLangWeightSyncClient)
    group = object()
    client._transport = (group, None)
    monkeypatch.setattr(client, "_engine_answers", lambda: True)
    monkeypatch.setattr(client, "_destroy_remote_group", lambda: None)
    monkeypatch.setattr(sglang_module, "CLEANUP_TIMEOUT_S", 0.2)
    aborted = _aborts(monkeypatch)
    released = threading.Event()
    monkeypatch.setattr(sglang_module, "release_weight_update_group", lambda group, *, destroy: released.wait(20))

    finished = threading.Event()
    thread = threading.Thread(target=lambda: (client.close_communicator(), finished.set()), daemon=True)
    thread.start()
    try:
        assert finished.wait(10), "the close stayed parked with the destroy"
    finally:
        released.set()
    assert aborted == [group]


def test_a_destroy_that_returns_is_not_aborted(monkeypatch):
    client = offline_client(SGLangWeightSyncClient)
    client._transport = (object(), None)
    _teardown_probe(client, monkeypatch)
    aborted = _aborts(monkeypatch)

    client.close_communicator()

    assert aborted == []


def test_an_unreachable_engine_reads_as_not_answering(monkeypatch):
    client = offline_client(SGLangWeightSyncClient, "http://127.0.0.1:9")
    assert client._engine_answers() is False


def test_an_aborted_group_is_dropped_not_destroyed(monkeypatch):
    """Torch has already removed its bookkeeping, so a destroy raises ``Invalid process group``."""
    client = offline_client(SGLangWeightSyncClient)
    group = object()
    client._transport = (group, None)
    client._aborted = True
    calls = _teardown_probe(client, monkeypatch)

    client.close_communicator()

    assert all(kind != "destroy" for kind, _ in calls), "close entered dist.destroy_process_group on an aborted group"
    assert ("drop", group) in calls
    assert ("remote", None) in calls


def test_a_client_that_never_formed_its_group_does_not_ask_the_engine_to_release_one(monkeypatch):
    """A failed or never-attempted formation already released the engine side; a second release delays the teardown."""
    client = offline_client(SGLangWeightSyncClient)
    calls = _teardown_probe(client, monkeypatch)

    client.close_communicator()

    assert calls == [], f"close touched the group machinery with no group formed: {calls}"


def test_the_atexit_invocation_skips_the_local_nccl_destroy(monkeypatch):
    client = offline_client(SGLangWeightSyncClient)
    group = object()
    client._transport = (group, None)
    calls = _teardown_probe(client, monkeypatch)

    client.close_communicator(at_exit=True)

    assert all(kind != "destroy" for kind, _ in calls), (
        "the atexit path entered dist.destroy_process_group — against a dead engine that blocks "
        "instead of raising and wedges interpreter exit"
    )
    assert ("drop", group) in calls, "the rank-map bookkeeping must still be dropped"
    assert ("remote", None) in calls, "skipping the engine-side release leaves the server refusing every future join"
    assert client._transport is None


def test_init_registers_the_no_destroy_close_at_atexit(monkeypatch):
    """The exit hook decides whether interpreter exit can wedge, so its registration is the contract."""
    client = offline_client(SGLangWeightSyncClient, "http://127.0.0.1:30000")
    registered: list[tuple[object, tuple, dict]] = []
    monkeypatch.setattr(atexit, "register", lambda fn, *args, **kwargs: registered.append((fn, args, kwargs)))
    monkeypatch.setattr(client, "_fetch_engine_world_size", lambda: 0)
    monkeypatch.setattr(client, "_destroy_remote_group", lambda: None)
    monkeypatch.setattr(client, "request", lambda method, path, **kwargs: None)
    monkeypatch.setattr(sglang_module, "build_weight_update_group", lambda **kwargs: (object(), None))

    client.init_communicator(device="cpu")

    assert registered == [(client.close_communicator, (), {"at_exit": True})], (
        f"init_communicator registered {registered} at exit, not the destroy-free close"
    )


def test_a_release_without_destroy_never_enters_the_nccl_destroy(monkeypatch):
    entered: list[object] = []
    monkeypatch.setattr(c10d_group.dist, "destroy_process_group", lambda group: entered.append(group))

    group = object()
    c10d_group.c10d._world.pg_group_ranks[group] = {0: 0}
    try:
        c10d_group.release_weight_update_group(group, destroy=False)
        assert entered == [], "a release without destroy entered dist.destroy_process_group"
        assert group not in c10d_group.c10d._world.pg_group_ranks, "the rank-map entry outlived the release"
    finally:
        c10d_group.c10d._world.pg_group_ranks.pop(group, None)


def test_a_failing_destroy_warns_and_still_drops_the_rank_map_entry(monkeypatch, caplog):
    """The engine may already be gone; the failure must not mask the error that triggered the teardown."""

    def failing_destroy(group):
        raise RuntimeError("Invalid process group")

    monkeypatch.setattr(c10d_group.dist, "destroy_process_group", failing_destroy)
    group = object()
    c10d_group.c10d._world.pg_group_ranks[group] = {0: 0}
    try:
        with caplog.at_level(logging.WARNING, logger=c10d_group.logger.name):
            c10d_group.release_weight_update_group(group, destroy=True)
        assert group not in c10d_group.c10d._world.pg_group_ranks, "the rank-map entry outlived a failed destroy"
        assert any("Invalid process group" in record.getMessage() for record in caplog.records)
    finally:
        c10d_group.c10d._world.pg_group_ranks.pop(group, None)


def test_a_failing_abort_warns_instead_of_raising(monkeypatch, caplog):
    def failing_abort(group):
        raise RuntimeError("abort refused by NCCL")

    monkeypatch.setattr(c10d_group.c10d, "_abort_process_group", failing_abort)

    with caplog.at_level(logging.WARNING, logger=c10d_group.logger.name):
        c10d_group.abort_weight_update_group(object())

    warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert any("abort refused by NCCL" in message for message in warnings), (
        f"a failed abort must surface as a warning naming the error, got {warnings}"
    )


def test_aborting_a_group_that_never_formed_does_not_enter_torch(monkeypatch):
    entered: list[object] = []
    monkeypatch.setattr(c10d_group.c10d, "_abort_process_group", entered.append)

    c10d_group.abort_weight_update_group(None)

    assert entered == [], "abort_weight_update_group(None) called torch's abort"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
