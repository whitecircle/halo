#!/usr/bin/env python
"""A server launched by hand can miss a flag the sync or the log-probs need, or run upstream's image, and every one of
those trains silently wrong. Both engines report the facts on ``/server_info``, so the preflight refuses before training
and a client refuses before forming a group."""

import ast
import logging

import pytest

from halo.trainers.grpo.rollout.engine_preflight import ServerRequirements, verify_server_flags
from halo.weight_sync.clients.sglang import SGLangWeightSyncClient
from halo.weight_sync.clients.vllm import _IDENTITY_KEY, VLLMWeightSyncClient
from tests.common.fake_engines import FakeSGLangServer, FakeVLLMServer
from tests.common.ports import free_port
from tests.common.repo import REPO_ROOT, load_script_module

_ENV_SCRIPT = REPO_ROOT / "scripts/training/grpo/environmental.py"
_ONLINE_SCRIPT = REPO_ROOT / "scripts/training/grpo/online.py"


@pytest.fixture
def engine(request):
    server_cls, client_cls = request.param
    server = server_cls()
    yield server, client_cls
    server.close()


def _verify(server, client_cls, **needs) -> None:
    verify_server_flags(
        [server.url], ServerRequirements(**needs), backend=client_cls.BACKEND_KEY, connection_timeout=5.0
    )


_ENGINES_FIXTURE = pytest.mark.parametrize(
    "engine",
    [(FakeVLLMServer, VLLMWeightSyncClient), (FakeSGLangServer, SGLangWeightSyncClient)],
    ids=["vllm", "sglang"],
    indirect=True,
)


@_ENGINES_FIXTURE
def test_the_compose_recipe_passes(engine):
    server, client_cls = engine
    server.reasoning_parser, server.captures_routing = "family", True
    _verify(
        server,
        client_cls,
        is_moe=True,
        sends_tools=True,
        captures_routing=True,
        reasoning_consumers=("reasoning_price",),
        sampled_token_ids=True,
    )


@_ENGINES_FIXTURE
def test_an_upstream_server_is_refused_by_the_preflight_and_by_group_formation(engine):
    """Upstream takes the same routes at the same version, and its first update reverts the experts behind a 200."""
    server, client_cls = engine
    server.toolkit_image = False
    with pytest.raises(ValueError, match="toolkit's image"):
        _verify(server, client_cls, is_moe=False)
    client = client_cls(base_url=server.url)
    with pytest.raises(RuntimeError, match="toolkit's image"):
        client._fetch_engine_world_size()


def test_a_report_that_cannot_be_read_does_not_stop_a_vllm_group_forming(caplog):
    """The route is not one a sync needs, so its failure refuses nothing, as in the preflights."""
    server = FakeVLLMServer()
    try:
        server.status["/server_info"] = 500
        client = VLLMWeightSyncClient(base_url=server.url)
        with caplog.at_level(logging.WARNING):
            assert client._fetch_engine_world_size() == 1
        assert "its image goes unchecked" in caplog.text
    finally:
        server.close()


@pytest.mark.parametrize("client_cls", [VLLMWeightSyncClient, SGLangWeightSyncClient], ids=["vllm", "sglang"])
def test_a_server_past_its_wait_names_the_knob(client_cls):
    """A server still loading past the wait is the common way to land here."""
    with pytest.raises(ConnectionError, match="rollout_connection_timeout, or vllm_server_timeout"):
        client_cls(base_url=f"http://127.0.0.1:{free_port()}", connection_timeout=0.0)


@_ENGINES_FIXTURE
def test_a_repacking_moe_backend_is_refused_on_a_moe_and_passes_on_a_dense_model(engine):
    server, client_cls = engine
    server.moe_backend = "auto"
    with pytest.raises(ValueError, match="repacks expert weights"):
        _verify(server, client_cls, is_moe=True)
    _verify(server, client_cls, is_moe=False)


def test_strict_tool_calling_is_refused_only_where_a_request_carries_tools():
    server = FakeVLLMServer()
    try:
        server.strict_tool_calling = True
        with pytest.raises(ValueError, match="VLLM_ENFORCE_STRICT_TOOL_CALLING"):
            _verify(server, VLLMWeightSyncClient, is_moe=False, sends_tools=True)
        _verify(server, VLLMWeightSyncClient, is_moe=False)
    finally:
        server.close()


@_ENGINES_FIXTURE
def test_a_server_with_no_tool_parser_is_refused_only_where_the_environment_sends_tools(engine):
    """vLLM answers a tool-bearing request 400 without one; SGLang's auto disables it with one log line, and every
    episode then scores alike with no error."""
    server, client_cls = engine
    server.tool_call_parser = ""
    with pytest.raises(ValueError, match="serves no tool parser"):
        _verify(server, client_cls, is_moe=False, sends_tools=True)
    _verify(server, client_cls, is_moe=False)


@_ENGINES_FIXTURE
def test_a_missing_reasoning_parser_is_refused_only_where_the_run_reads_reasoning(engine):
    server, client_cls = engine
    with pytest.raises(ValueError, match="no reasoning parser.*reasoning_floor"):
        _verify(server, client_cls, is_moe=False, reasoning_consumers=("reasoning_floor",))
    _verify(server, client_cls, is_moe=False)


@_ENGINES_FIXTURE
def test_routing_replay_needs_the_capture_flag(engine):
    server, client_cls = engine
    with pytest.raises(ValueError, match="--enable-return-routed-experts"):
        _verify(server, client_cls, is_moe=True, captures_routing=True)
    _verify(server, client_cls, is_moe=True)


def test_routing_replay_under_torch_compile_is_refused():
    server = FakeSGLangServer()
    try:
        server.captures_routing, server.torch_compile = True, True
        with pytest.raises(ValueError, match="--enable-torch-compile"):
            _verify(server, SGLangWeightSyncClient, is_moe=True, captures_routing=True)
        _verify(server, SGLangWeightSyncClient, is_moe=True)
    finally:
        server.close()


def test_a_vllm_server_without_token_ids_is_refused_under_train_on_sampled_tokens():
    server = FakeVLLMServer()
    try:
        server.returns_token_ids = False
        with pytest.raises(ValueError, match="--return-tokens-as-token-ids"):
            _verify(server, VLLMWeightSyncClient, is_moe=False, sampled_token_ids=True)
        _verify(server, VLLMWeightSyncClient, is_moe=False)
    finally:
        server.close()


@_ENGINES_FIXTURE
def test_a_reply_missing_a_flag_passes_with_it_unread(engine, monkeypatch):
    """A renamed upstream field must not refuse a good server; the image identity is the one field a server owes."""
    server, client_cls = engine
    original = server.answer

    def bare(method, path, body):
        if path != "/server_info":
            return original(method, path, body)
        identity = {"vllm_env": {_IDENTITY_KEY: "x"}} if client_cls is VLLMWeightSyncClient else {"halo_patches": True}
        return 200, {"tp_size": 1, **identity}

    monkeypatch.setattr(server, "answer", bare)
    _verify(server, client_cls, is_moe=True, reasoning_consumers=("reasoning_floor",), sends_tools=True)


def test_the_identity_key_is_the_one_the_image_declares():
    patch = load_script_module("docker/vllm/patches/vllm_image_identity_patch.py")
    assert patch.IDENTITY_KEY == _IDENTITY_KEY


@pytest.mark.parametrize("script", [_ENV_SCRIPT, _ONLINE_SCRIPT], ids=["environmental", "online"])
def test_each_script_runs_the_flag_preflight(script):
    calls = [
        node
        for node in ast.walk(ast.parse(script.read_text()))
        if isinstance(node, ast.Call) and ast.unparse(node.func) == "verify_server_flags"
    ]
    assert len(calls) == 1


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
