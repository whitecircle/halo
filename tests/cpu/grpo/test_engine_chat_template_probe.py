#!/usr/bin/env python
"""Rows train on the engine's prompt ids, so a server left on the checkpoint's template fails nothing, and the reasoning
terms then charge for a contract the model never read. Byte-comparing renders cannot refuse, as an engine derives toggles
the trainer's render lacks; a prompt that ignores the level can be read off the server alone."""

import logging
from types import SimpleNamespace

import pytest

from halo.environments.rollout_config import RolloutConfig
from halo.trainers.grpo.rollout.engine_preflight import verify_server_chat_template
from halo.weight_sync.clients.sglang import SGLangWeightSyncClient
from halo.weight_sync.clients.vllm import VLLMWeightSyncClient
from tests.common.fake_engines import fake_server_class

_MESSAGES = [{"role": "user", "content": "Probe."}]
_LOW = SimpleNamespace(level="low")
_HIGH = SimpleNamespace(level="high")
_LEVEL_IDS = {"low": 11, "high": 12, None: 13}


def _trainer_ids(effort) -> list[int]:
    return [1, _LEVEL_IDS[effort.level if effort else None], 2]


def _ignores_the_level(effort) -> list[int]:
    return [1, 2]


def _level_of(body: dict) -> str | None:
    return body.get("reasoning_effort")


@pytest.fixture(params=[VLLMWeightSyncClient, SGLangWeightSyncClient], ids=["vllm", "sglang"])
def engine(request):
    server = fake_server_class(request.param)()
    yield server, request.param
    server.close()


def _verify(server, client_cls, trainer=_trainer_ids, efforts=(_LOW, _HIGH)) -> None:
    rollout = RolloutConfig(
        model_name="m",
        backend=client_cls.BACKEND_KEY,
        capture_routed_experts=True,
        max_tokens=4096,
        max_thinking_tokens=2048,
    )
    verify_server_chat_template([server.url], trainer, _MESSAGES, list(efforts), rollout, connection_timeout=5.0)


def test_a_server_whose_prompt_ignores_the_level_is_refused(engine):
    server, client_cls = engine
    with pytest.raises(ValueError, match="ignores the level") as refused:
        _verify(server, client_cls)
    assert client_cls.CHAT_TEMPLATE_FLAG in str(refused.value)


def test_the_probe_sends_each_contract_and_no_tools(engine):
    server, client_cls = engine
    server.render_prompt = lambda body: [1, _LEVEL_IDS[_level_of(body)], 2]
    _verify(server, client_cls)
    bodies = [body for path, body in server.posted if path == "/v1/chat/completions"]
    assert [_level_of(body) for body in bodies] == ["low", "high"]
    assert not any("tools" in body or "return_routed_experts" in body for body in bodies)
    assert not any("thinking_token_budget" in body for body in bodies)
    assert bodies[0]["max_tokens"] == 1
    assert all("reasoning_budget" not in body.get("chat_template_kwargs", {}) for body in bodies), (
        "the template is told the level alone"
    )


def test_a_matching_server_passes_with_no_warning(engine, caplog):
    server, client_cls = engine
    server.render_prompt = lambda body: [1, _LEVEL_IDS[_level_of(body)], 2]
    with caplog.at_level(logging.WARNING):
        _verify(server, client_cls)
    assert "differently" not in caplog.text


def test_a_render_that_differs_only_warns_since_an_engine_derives_a_toggle(engine, caplog):
    server, client_cls = engine
    server.render_prompt = lambda body: [1, _LEVEL_IDS[_level_of(body)], 2, 99]
    with caplog.at_level(logging.WARNING):
        _verify(server, client_cls)
    assert "first at token 3 of 4/3" in caplog.text


def test_a_trainer_template_with_no_level_cannot_refuse(engine):
    """A stock template on both sides: nothing moved with the level anywhere."""
    server, client_cls = engine
    server.render_prompt = lambda body: [1, 2]
    _verify(server, client_cls, trainer=_ignores_the_level)


def test_a_trainer_render_that_raises_passes_with_a_warning(engine, caplog):
    """The no-level contract of a fixed-level run is one the run never draws, so a template that raises on it is no fault
    of the server's."""
    server, client_cls = engine

    def raises(effort):
        raise ValueError("this template needs a level")

    with caplog.at_level(logging.WARNING):
        _verify(server, client_cls, trainer=raises, efforts=(_LOW, None))
    assert "this template needs a level" in caplog.text


def test_a_fixed_level_is_compared_with_no_level(engine):
    server, client_cls = engine
    with pytest.raises(ValueError, match="ignores the level"):
        _verify(server, client_cls, efforts=(_LOW, None))


def test_a_run_with_no_level_reads_only_the_render(engine, caplog):
    server, client_cls = engine
    server.render_prompt = lambda body: [1, 13, 2]
    with caplog.at_level(logging.WARNING):
        _verify(server, client_cls, efforts=(None,))
    assert "differently" not in caplog.text


def test_a_reply_without_prompt_ids_passes_unread(engine):
    server, client_cls = engine
    server.render_prompt = lambda body: None
    _verify(server, client_cls)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
