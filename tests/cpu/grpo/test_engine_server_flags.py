#!/usr/bin/env python
"""A server launched by hand can miss a flag the sync or the log-probs need, or run upstream's image, and every one of
those trains silently wrong. Both engines report the facts on ``/server_info``, so the preflight refuses before training
and a client refuses before forming a group."""

import ast
import json
import logging
import threading

import pytest
from huggingface_hub import constants

from halo.trainers.grpo.rollout.engine_preflight import ServerRequirements, verify_group_servers, verify_server_flags
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


def _checkpoint(directory, **config) -> str:
    """A checkpoint directory holding only ``config`` as its ``config.json``, for a fake server to report serving."""
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "config.json").write_text(json.dumps({"model_type": "qwen3_5_moe", **config}))
    return str(directory)


_FP8_CHECKPOINT = {"quantization_config": {"quant_method": "fp8", "weight_block_size": [128, 128]}}


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
        client.require_engine_world_size()


def test_a_report_that_cannot_be_read_does_not_stop_a_vllm_group_forming(caplog):
    """The route is not one a sync needs, so its failure refuses nothing, as in the preflights."""
    server = FakeVLLMServer()
    try:
        server.status["/server_info"] = 500
        client = VLLMWeightSyncClient(base_url=server.url)
        with caplog.at_level(logging.WARNING):
            assert client.require_engine_world_size() == 1
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


@_ENGINES_FIXTURE
def test_a_quantized_server_is_refused_on_every_model(engine):
    server, client_cls = engine
    server.quantization = "fp8"
    for is_moe in (True, False):
        with pytest.raises(ValueError, match=r"weights are quantized \(fp8\)"):
            _verify(server, client_cls, is_moe=is_moe)


@_ENGINES_FIXTURE
def test_a_bf16_run_names_the_fp8_remedy_when_it_meets_a_quantized_server(engine):
    server, client_cls = engine
    server.quantization = "fp8"
    with pytest.raises(ValueError, match="rollout_weight_precision: fp8_block"):
        _verify(server, client_cls, is_moe=True)


@_ENGINES_FIXTURE
def test_an_fp8_run_accepts_an_fp8_server_and_refuses_every_other_format(engine, tmp_path):
    server, client_cls = engine
    server.model_path = _checkpoint(tmp_path)
    server.quantization = "fp8"
    _verify(server, client_cls, is_moe=True, weight_precision="fp8_block")
    for served, wording in (("", "serves bf16"), ("awq", "serves awq")):
        server.quantization = served
        with pytest.raises(ValueError, match=wording):
            _verify(server, client_cls, is_moe=True, weight_precision="fp8_block")


def _sglang_on(model_path: str) -> FakeSGLangServer:
    """An SGLang server launched without --quantization, which it reports as null, on the checkpoint at
    ``model_path``."""
    server = FakeSGLangServer()
    server.quantization, server.model_path = "", model_path
    return server


def test_sglang_reports_the_format_of_a_quantized_checkpoint_it_was_launched_on_without_the_flag(tmp_path):
    server = _sglang_on(_checkpoint(tmp_path, **_FP8_CHECKPOINT))
    try:
        with pytest.raises(ValueError, match=r"weights are quantized \(fp8\)"):
            _verify(server, SGLangWeightSyncClient, is_moe=True)
        _verify(server, SGLangWeightSyncClient, is_moe=True, weight_precision="fp8_block")
    finally:
        server.close()


def test_sglang_finds_the_checkpoint_of_a_snapshot_path_in_this_hosts_hub_cache(tmp_path, monkeypatch):
    """The server's cache sits elsewhere: the same repo and commit are read from this host's."""
    commit = "a" * 40
    _checkpoint(tmp_path / "models--org--policy" / "snapshots" / commit, **_FP8_CHECKPOINT)
    monkeypatch.setattr(constants, "HF_HUB_CACHE", str(tmp_path))
    server = _sglang_on(f"/server/hf/hub/models--org--policy/snapshots/{commit}")
    try:
        with pytest.raises(ValueError, match=r"weights are quantized \(fp8\)"):
            _verify(server, SGLangWeightSyncClient, is_moe=True)
    finally:
        server.close()


def test_an_sglang_format_this_host_cannot_read_passes_a_bf16_run_and_refuses_an_fp8_one_naming_the_flag():
    server = _sglang_on("/server/only/checkpoint")
    try:
        _verify(server, SGLangWeightSyncClient, is_moe=True)
        with pytest.raises(ValueError, match="Launch it with --quantization fp8"):
            _verify(server, SGLangWeightSyncClient, is_moe=True, weight_precision="fp8_block")
        server.quantization = "fp8"
        _verify(server, SGLangWeightSyncClient, is_moe=True, weight_precision="fp8_block")
    finally:
        server.close()


@_ENGINES_FIXTURE
def test_an_fp8_run_still_needs_the_triton_moe_backend(engine):
    server, client_cls = engine
    server.quantization, server.moe_backend = "fp8", "auto"
    with pytest.raises(ValueError, match="repacks expert weights"):
        _verify(server, client_cls, is_moe=True, weight_precision="fp8_block")


@pytest.mark.parametrize(
    ("flag", "other"),
    [("dtype", "float16"), ("quantization", "fp8"), ("moe_backend", "auto"), ("reasoning_parser", "other")],
)
@pytest.mark.parametrize("server_cls", [FakeVLLMServer, FakeSGLangServer], ids=["vllm", "sglang"])
def test_servers_of_one_group_must_agree_on_every_launch_flag_that_changes_how_the_weights_load(
    server_cls, flag, other, tmp_path
):
    servers = [server_cls(), server_cls()]
    try:
        for server in servers:
            server.model_path = _checkpoint(tmp_path)
        urls = [server.url for server in servers]
        backend = server_cls.CLIENT_CLASS.BACKEND_KEY
        verify_group_servers(urls, backend=backend, connection_timeout=5.0)
        setattr(servers[1], flag, other)
        with pytest.raises(ValueError, match=f"disagree on their {flag}"):
            verify_group_servers(urls, backend=backend, connection_timeout=5.0)
    finally:
        for server in servers:
            server.close()


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


_LAUNCH_FACTS = {
    FakeVLLMServer: {
        "tensor_parallel_size": 8,
        "data_parallel_size": 1,
        "pipeline_parallel_size": 1,
        "expert_parallel": True,
        "max_num_seqs": 256,
        "max_num_batched_tokens": 16384,
        "attention_backend": "FLASH_ATTN",
        "compilation_mode": "VLLM_COMPILE",
        "cudagraph_mode": "FULL_AND_PIECEWISE",
    },
    FakeSGLangServer: {
        "tp_size": 1,
        "dp_size": 1,
        "pp_size": 1,
        "max_running_requests": 256,
        "attention_backend": "flashinfer",
    },
}


@pytest.mark.parametrize("server_cls", [FakeVLLMServer, FakeSGLangServer], ids=["vllm", "sglang"])
def test_the_launch_facts_a_server_reports_are_read_and_a_fact_it_lacks_is_left_out(server_cls):
    server = server_cls()
    try:
        flags = server_cls.CLIENT_CLASS(base_url=server.url).read_server_flags()
    finally:
        server.close()
    assert dict(flags.extras) == _LAUNCH_FACTS[server_cls]


@_ENGINES_FIXTURE
def test_the_launch_line_carries_every_servers_facts_and_names_one_that_did_not_answer(engine, caplog):
    server, client_cls = engine
    silent = type(server)()
    silent.status["/server_info"] = 500
    try:
        with caplog.at_level(logging.INFO):
            verify_server_flags(
                [server.url, silent.url],
                ServerRequirements(is_moe=False),
                backend=client_cls.BACKEND_KEY,
                connection_timeout=5.0,
            )
    finally:
        silent.close()
    launch_lines = [
        r.getMessage()
        for r in caplog.records
        if r.getMessage().startswith("Rollout server") and "launch flags" in r.getMessage()
    ]
    [served] = [line for line in launch_lines if server.url in line]
    [unanswered] = [line for line in launch_lines if silent.url in line]
    assert "extras=" in served and "'attention_backend'" in served
    assert "unread" in unanswered


@_ENGINES_FIXTURE
def test_the_servers_are_probed_at_once(engine):
    """Every server's /server_info is reached while the first is still unanswered, which a loop over the servers
    cannot do."""
    server, client_cls = engine
    others = [type(server)() for _ in range(2)]
    servers = [server, *others]
    for each in servers:
        each.hold.add("/server_info")
    failures: list[BaseException] = []

    def run() -> None:
        try:
            verify_server_flags(
                [each.url for each in servers],
                ServerRequirements(is_moe=False),
                backend=client_cls.BACKEND_KEY,
                connection_timeout=5.0,
            )
        except BaseException as e:  # reported below, on the test thread
            failures.append(e)

    thread = threading.Thread(target=run)
    thread.start()
    try:
        for each in servers:
            each.wait_for_request("GET /server_info", timeout=10.0)
    finally:
        for each in servers:
            each.release()
        thread.join(timeout=30.0)
        for each in others:
            each.close()
    assert not failures, failures


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
