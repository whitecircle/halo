#!/usr/bin/env python
"""The OpenAI-compatible client: the request one call sends, the retry of a reply without choices, and the key the
client refuses to build without."""

import asyncio
import inspect
import types

import pytest

from halo.inference.openai_client import (
    UPSTREAM_RETRIES,
    EmptyChoicesError,
    build_openai_client,
    chat_completion,
    generate_openai_response,
)
from tests.common.fake_chat_client import FakeChatClient, RecordingChatClient, fake_completion


def test_generate_requires_an_explicit_keyword_only_client():
    """Omitting the client is a TypeError naming it, never a mid-run 401 from an endpoint the caller did not choose."""
    parameter = inspect.signature(generate_openai_response).parameters["client"]
    assert parameter.kind is inspect.Parameter.KEYWORD_ONLY
    assert parameter.default is inspect.Parameter.empty
    with pytest.raises(TypeError, match="keyword-only argument: 'client'"):
        asyncio.run(generate_openai_response("m", "hi"))
    assert asyncio.run(generate_openai_response("m", "hi", client=RecordingChatClient())).answer == "ok"


def test_generate_sends_the_default_temperature_and_omits_a_none():
    """``None`` keeps the served default, which reasoning models require; the default still pins 0.0."""
    client = RecordingChatClient()
    asyncio.run(generate_openai_response("m", "hi", client=client))
    assert client.captured["temperature"] == 0.0
    asyncio.run(generate_openai_response("m", "hi", client=client, temperature=None))
    assert "temperature" not in client.captured


def test_generate_sends_tools_only_when_given():
    tools = [{"type": "function", "function": {"name": "search"}}]
    client = RecordingChatClient()
    asyncio.run(generate_openai_response("m", "hi", client=client))
    assert "tools" not in client.captured
    asyncio.run(generate_openai_response("m", "hi", client=client, tools=tools))
    assert client.captured["tools"] == tools


def _empty_completion(error=None):
    """A 200 reply with no choices; ``error`` is the body an aggregator puts beside them, absent when
    None (the shape an SDK object has when the server sent no error either)."""
    completion = types.SimpleNamespace(choices=[])
    if error is not None:
        completion.error = error
    return completion


class _ScriptedClient(FakeChatClient):
    """AsyncOpenAI stand-in answering each create() with the next scripted completion, the last one repeating,
    counting requests."""

    def __init__(self, *completions):
        super().__init__()
        self._completions = completions
        self.requests = 0

    async def _create(self, **_kwargs):
        self.requests += 1
        return self._completions[min(self.requests, len(self._completions)) - 1]


def _record_backoff(monkeypatch) -> list[float]:
    waits: list[float] = []

    async def no_sleep(seconds):
        waits.append(seconds)

    monkeypatch.setattr("halo.inference.openai_client.asyncio.sleep", no_sleep)
    return waits


def test_chat_completion_retries_an_empty_reply_on_a_retryable_code_with_backoff(monkeypatch):
    """An aggregator answers 200 with no choices and the upstream code in the body: a rate limit the SDK's own retry
    never sees."""
    waits = _record_backoff(monkeypatch)
    throttled = _empty_completion({"code": 429, "message": "upstream rate limit"})
    spelled = _empty_completion({"code": "429", "message": "upstream rate limit"})
    client = _ScriptedClient(throttled, spelled, fake_completion("late"))

    completion = asyncio.run(chat_completion(client, model="m", messages=[]))

    assert completion.choices[0].message.content == "late"
    assert client.requests == 3
    assert waits == [2.0, 4.0]


@pytest.mark.parametrize("error", [{"code": 400, "message": "bad request"}, None], ids=["non-retryable", "no-body"])
def test_chat_completion_raises_at_once_on_a_non_retryable_empty_reply(monkeypatch, error):
    waits = _record_backoff(monkeypatch)
    client = _ScriptedClient(_empty_completion(error))

    with pytest.raises(EmptyChoicesError) as excinfo:
        asyncio.run(chat_completion(client, model="judge", messages=[]))

    assert client.requests == 1
    assert waits == []
    assert "'judge'" in str(excinfo.value) and repr(error) in str(excinfo.value)
    assert excinfo.value.error == error


def test_chat_completion_gives_up_after_the_retry_budget(monkeypatch):
    waits = _record_backoff(monkeypatch)
    client = _ScriptedClient(_empty_completion({"code": 503, "message": "upstream unavailable"}))

    with pytest.raises(EmptyChoicesError, match="'m'.*503"):
        asyncio.run(chat_completion(client, model="m", messages=[]))

    assert client.requests == UPSTREAM_RETRIES + 1
    assert waits == [2.0, 4.0, 8.0, 16.0]


@pytest.mark.parametrize("exported", [None, ""])
def test_the_client_refuses_to_build_without_a_key(monkeypatch, exported):
    """A blank OPENAI_API_KEY (``.env.example`` ships one) is no key: the SDK would re-read it and every request
    would fail as an opaque 401."""
    if exported is None:
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    else:
        monkeypatch.setenv("OPENAI_API_KEY", exported)
    with pytest.raises(ValueError, match="OPENAI_API_KEY"):
        build_openai_client(api_key_override="")
    monkeypatch.setenv("OPENAI_API_KEY", "sdk-key")
    assert build_openai_client().api_key == "sdk-key"
    assert build_openai_client(api_key_override="override").api_key == "override"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
