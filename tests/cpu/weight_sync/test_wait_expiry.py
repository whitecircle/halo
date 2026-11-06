#!/usr/bin/env python
"""Every bounded wait of a sync names its server, purpose and bound when it starts, and says once if the bound ran
out; a wait that ends early says nothing more."""

import logging
import threading
import time
from types import SimpleNamespace

import pytest
import requests

import halo.weight_sync.clients.base as client_base
import halo.weight_sync.clients.sglang as sglang_module
import halo.weight_sync.group as group_module
from halo.weight_sync.clients.server_calls import AsyncCall, announced_wait, wait_announced
from halo.weight_sync.clients.sglang import SGLangWeightSyncClient
from halo.weight_sync.clients.vllm import VLLMWeightSyncClient
from halo.weight_sync.group import WeightSyncGroup
from halo.weight_sync.transport import c10d_group
from tests.common.weight_sync import offline_client

_BOUND_S = 0.2


def _gave_up(caplog) -> list[str]:
    return [r.getMessage() for r in caplog.records if r.getMessage().startswith("Gave up waiting")]


def _started(caplog) -> list[str]:
    return [r.getMessage() for r in caplog.records if r.getMessage().startswith("Waiting on ")]


def _read_timeout(*args, **kwargs):
    raise requests.exceptions.ReadTimeout("no answer")


@pytest.fixture
def parked():
    release = threading.Event()
    yield release
    release.set()


@pytest.fixture
def short_bounds(monkeypatch):
    for module in (client_base, sglang_module, group_module):
        for name in ("CLEANUP_TIMEOUT_S", "SERVER_ERROR_GRACE_S", "_ENGINE_ANSWER_TIMEOUT_S"):
            if hasattr(module, name):
                monkeypatch.setattr(module, name, _BOUND_S)


def _expect_one_expiry(caplog, client, *fragments: str) -> None:
    started, gave_up = _started(caplog), _gave_up(caplog)
    assert len(started) == 1 and len(gave_up) == 1, (started, gave_up)
    for message in (started[0], gave_up[0]):
        assert client.base_url in message and all(f in message for f in fragments), message
    assert gave_up[0].endswith(f"after {_BOUND_S:.0f}s"), gave_up


def test_a_local_release_that_never_returns_says_it_gave_up(monkeypatch, caplog, parked, short_bounds):
    client = offline_client(SGLangWeightSyncClient)
    client._transport = (object(), None)
    monkeypatch.setattr(client, "_engine_answers", lambda: True)
    monkeypatch.setattr(client, "group_releases_remote_half", True)
    monkeypatch.setattr(sglang_module, "release_weight_update_group", lambda group, *, destroy: parked.wait(30))
    monkeypatch.setattr(client, "abort_transport", lambda: None)
    with caplog.at_level(logging.INFO):
        client._release_transport(at_exit=False)
    _expect_one_expiry(caplog, client, "trainer's half", "released")


def test_a_release_that_ends_on_its_own_deadline_says_it_gave_up(caplog):
    """A call whose deadline equals the wait's bound returns just as the join does, still dead to ``is_alive``."""
    call = AsyncCall(lambda: time.sleep(_BOUND_S * 0.99))
    with caplog.at_level(logging.INFO):
        wait_announced(call, "http://s", "its release", _BOUND_S)
    assert len(_gave_up(caplog)) == 1, _gave_up(caplog)


def test_an_abort_that_never_returns_says_it_gave_up(monkeypatch, caplog, parked, short_bounds):
    client = offline_client(SGLangWeightSyncClient)
    client._transport = (object(), None)
    monkeypatch.setattr(c10d_group.c10d, "_abort_process_group", lambda group: parked.wait(30))
    with caplog.at_level(logging.INFO):
        client.abort_transport()
    _expect_one_expiry(caplog, client, "abort")


def test_a_remote_destroy_that_gets_no_answer_says_it_gave_up(monkeypatch, caplog, short_bounds):
    client = offline_client(SGLangWeightSyncClient)
    monkeypatch.setattr(client, "request", _read_timeout)
    with caplog.at_level(logging.INFO):
        client.destroy_remote_group()
    _expect_one_expiry(caplog, client, "to be destroyed")


def test_a_vllm_engine_shutdown_that_gets_no_answer_says_it_gave_up(monkeypatch, caplog, short_bounds):
    client = offline_client(VLLMWeightSyncClient)
    monkeypatch.setattr(client_base, "CLEANUP_TIMEOUT_S", _BOUND_S)
    monkeypatch.setattr("halo.weight_sync.clients.vllm.CLEANUP_TIMEOUT_S", _BOUND_S)
    monkeypatch.setattr(client, "request", _read_timeout)
    with caplog.at_level(logging.INFO):
        client.destroy_remote_group()
    _expect_one_expiry(caplog, client, "weight-transfer engine")


def test_a_health_probe_that_gets_no_answer_says_it_gave_up(monkeypatch, caplog, short_bounds):
    client = offline_client(SGLangWeightSyncClient)
    monkeypatch.setattr(client, "request", _read_timeout)
    with caplog.at_level(logging.INFO):
        assert client._engine_answers() is False
    _expect_one_expiry(caplog, client, "/health")


def test_a_session_close_before_a_join_that_gets_no_answer_says_it_gave_up(monkeypatch, caplog, short_bounds):
    client = offline_client(SGLangWeightSyncClient)
    monkeypatch.setattr(client, "_destroy_group_named", lambda name: None)
    monkeypatch.setattr(client, "request", _read_timeout)
    with caplog.at_level(logging.INFO), pytest.raises(RuntimeError, match="only a restart frees"):
        client.request_engine_join("127.0.0.1", 1, 2, 1, None)
    _expect_one_expiry(caplog, client, "session close before it joins")


def test_an_interrupted_session_close_that_gets_no_answer_says_it_gave_up(monkeypatch, caplog, short_bounds):
    client = offline_client(SGLangWeightSyncClient)
    client._update_session_open = True
    monkeypatch.setattr(client, "request", _read_timeout)
    with caplog.at_level(logging.INFO):
        client.abandon_update_session()
    _expect_one_expiry(caplog, client, "interrupted update session")


def test_a_layout_read_that_gets_no_answer_says_it_gave_up(monkeypatch, caplog):
    client = offline_client(SGLangWeightSyncClient)
    monkeypatch.setattr(SGLangWeightSyncClient, "layout_read_bound_s", _BOUND_S)
    monkeypatch.setattr(client, "read_engine_layout", _read_timeout)
    group = WeightSyncGroup([client], nodes=["a"], group_port=0, group_host=None)
    with caplog.at_level(logging.INFO), pytest.raises(Exception, match="layout"):
        group.init_communicator("cpu")
    _expect_one_expiry(caplog, client, "parallel layout")


def test_a_server_that_never_answers_a_failed_chunk_says_it_gave_up(monkeypatch, caplog, parked, short_bounds):
    client = offline_client(SGLangWeightSyncClient)
    monkeypatch.setattr(client, "abort_transport", lambda: None)
    group = WeightSyncGroup([client], nodes=["a"], group_port=0, group_host=None)
    monkeypatch.setattr(group, "_mark_unusable", lambda *args, **kwargs: None)
    declared = [AsyncCall(lambda: parked.wait(30), name="declare")]
    with caplog.at_level(logging.INFO), pytest.raises(RuntimeError, match="send failed"):
        group._fail_group(RuntimeError("send failed"), declared)
    _expect_one_expiry(caplog, client, "failed chunk")


def test_a_local_formation_that_never_ends_says_it_gave_up(monkeypatch, caplog, parked, short_bounds):
    client = offline_client(SGLangWeightSyncClient)
    monkeypatch.setattr(client, "_release_transport", lambda *, at_exit: None)
    formed = AsyncCall(lambda: parked.wait(30), name="formation")
    with caplog.at_level(logging.INFO):
        client.discard_transport(formed)
    _expect_one_expiry(caplog, client, "local formation")


def test_negative_control_waits_that_end_early_log_no_expiry(monkeypatch, caplog):
    client = offline_client(SGLangWeightSyncClient)
    monkeypatch.setattr(client, "request", lambda *args, **kwargs: SimpleNamespace(status_code=200))
    call = AsyncCall(lambda: None)
    with caplog.at_level(logging.INFO):
        client.destroy_remote_group()
        assert client._engine_answers() is True
        wait_announced(call, client.base_url, "a quick call", 30.0)
    assert len(_started(caplog)) == 3 and _gave_up(caplog) == []


def test_a_non_timeout_failure_inside_a_wait_is_not_reported_as_expiry(caplog):
    with caplog.at_level(logging.INFO), pytest.raises(ValueError):
        with announced_wait("http://s", "an answer", 30.0):
            raise ValueError("refused")
    assert _gave_up(caplog) == []


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
