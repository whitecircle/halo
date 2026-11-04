"""A vLLM session unloads the model at its open and a failed chunk ends it, so resuming then kills the engine at its next
request. A server restarted between syncs has to be caught before the open, and a session no close can end must keep its
pause."""

from types import SimpleNamespace

import pytest
import requests

from halo.weight_sync.clients.base import ServerFlags
from halo.weight_sync.clients.sglang import SGLangWeightSyncClient
from halo.weight_sync.clients.vllm import _IDENTITY_KEY, VLLMWeightSyncClient
from tests.common.weight_sync import offline_client


def _recording_client(client_cls, monkeypatch, *, refuses=()):
    client = offline_client(client_cls)
    client._transport = SimpleNamespace(aborted=False) if client_cls is VLLMWeightSyncClient else (object(), None)
    posts: list[str] = []

    def request(method, path, **kwargs):
        posts.append(path)
        if path in refuses:
            raise requests.exceptions.HTTPError(f"500 on {path}")
        return SimpleNamespace(status_code=200, json=lambda: {})

    monkeypatch.setattr(client, "request", request)
    return client, posts


def test_a_restarted_engine_is_refused_before_it_is_touched(monkeypatch):
    client, posts = _recording_client(VLLMWeightSyncClient, monkeypatch)
    client._group_instance = "first"
    monkeypatch.setattr(client, "read_server_flags", lambda: ServerFlags(toolkit_image=True, instance_id="second"))

    with pytest.raises(RuntimeError, match="restarted after this client's group formed"):
        client._open_update()

    assert posts == []
    assert client.can_replay_sync, "the pool reconnects and replays only a sync that sent nothing"


def test_the_same_engine_opens_as_before(monkeypatch):
    client, posts = _recording_client(VLLMWeightSyncClient, monkeypatch)
    client._group_instance = "first"
    monkeypatch.setattr(client, "read_server_flags", lambda: ServerFlags(toolkit_image=True, instance_id="first"))

    client._open_update()

    assert posts == [VLLMWeightSyncClient.PAUSE_ENDPOINT, VLLMWeightSyncClient.SESSION_OPEN_ENDPOINT]


def test_a_server_that_reports_no_instance_is_not_read_at_each_open(monkeypatch):
    client, posts = _recording_client(VLLMWeightSyncClient, monkeypatch)

    def unread():
        raise AssertionError("no instance was recorded, so there is nothing to compare")

    monkeypatch.setattr(client, "read_server_flags", unread)

    client._open_update()

    assert posts == [VLLMWeightSyncClient.PAUSE_ENDPOINT, VLLMWeightSyncClient.SESSION_OPEN_ENDPOINT]


def test_group_formation_records_the_instance(monkeypatch):
    client = offline_client(VLLMWeightSyncClient)
    info = {"vllm_env": {_IDENTITY_KEY: "1"}, "vllm_config": {"instance_id": "launch-1"}}

    def request(method, path, **kwargs):
        return SimpleNamespace(status_code=200, json=lambda: info if path == "/server_info" else {"world_size": 1})

    monkeypatch.setattr(client, "request", request)
    monkeypatch.setattr(client, "_check_server_version", lambda: None)
    monkeypatch.setattr(client, "probe_generation", lambda: None)

    client._fetch_engine_world_size()

    assert client._group_instance == "launch-1"


def test_vllm_keeps_the_pause_when_no_close_can_end_the_session(monkeypatch):
    client, posts = _recording_client(
        VLLMWeightSyncClient, monkeypatch, refuses=(VLLMWeightSyncClient.SESSION_CLOSE_ENDPOINT,)
    )
    client._open_update()

    client.abort_weight_update()

    assert VLLMWeightSyncClient.RESUME_ENDPOINT not in posts, "resuming here kills the engine at its next request"
    assert client._unusable_reason is not None
    assert "no model" in client._unusable_reason, "nothing landed, so it holds neither old nor new weights"
    assert "part old" not in client._unusable_reason
    with pytest.raises(RuntimeError, match="RESTART"):
        client._open_update()


def test_vllm_resumes_when_a_retried_close_restores_the_model(monkeypatch):
    client, posts = _recording_client(VLLMWeightSyncClient, monkeypatch)
    client._open_update()
    calls = []
    plain = client.request

    def fails_once(method, path, **kwargs):
        if path == VLLMWeightSyncClient.SESSION_CLOSE_ENDPOINT and not calls:
            calls.append(path)
            raise requests.exceptions.ReadTimeout(path)
        return plain(method, path, **kwargs)

    monkeypatch.setattr(client, "request", fails_once)

    client.abort_weight_update()

    assert VLLMWeightSyncClient.RESUME_ENDPOINT in posts
    assert client._unusable_reason is None


def test_vllm_lifts_the_pause_when_the_session_never_opened(monkeypatch):
    client, posts = _recording_client(
        VLLMWeightSyncClient,
        monkeypatch,
        refuses=(VLLMWeightSyncClient.SESSION_OPEN_ENDPOINT, VLLMWeightSyncClient.SESSION_CLOSE_ENDPOINT),
    )

    with pytest.raises(requests.exceptions.HTTPError):
        client._open_update()

    assert VLLMWeightSyncClient.RESUME_ENDPOINT in posts
    assert client._unusable_reason is None


def test_sglang_keeps_its_model_through_a_session_that_cannot_close(monkeypatch):
    client, posts = _recording_client(
        SGLangWeightSyncClient, monkeypatch, refuses=(SGLangWeightSyncClient.SESSION_CLOSE_ENDPOINT,)
    )
    client._open_update()

    client.abort_weight_update()

    assert SGLangWeightSyncClient.RESUME_ENDPOINT in posts
    assert client._unusable_reason is None


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
