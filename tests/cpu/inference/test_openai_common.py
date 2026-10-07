import asyncio
import inspect
import json
import tempfile
import types

import pytest

from src.inference.batch_requests import parallel_openai_requests, resolve_checkpoint_file, resolve_request_tools
from src.inference.openai_client import (
    UPSTREAM_RETRIES,
    EmptyChoicesError,
    chat_completion,
    create_openai_client,
    generate_openai_response,
    parse_json_object,
)
from src.inference.response import OpenAIResponse
from src.inference.resume_store import append_openai_checkpoint, load_openai_checkpoint


def _fake_completion(content: str = "ok", **choice_extras):
    """Minimal stand-in for a chat-completion response (choices[0].message + usage); ``choice_extras``
    are attributes outside the OpenAI schema that an engine attaches to the choice."""
    message = types.SimpleNamespace(content=content, tool_calls=None, reasoning=None, reasoning_content=None)
    choice = types.SimpleNamespace(message=message, finish_reason="stop", stop_reason=None, **choice_extras)
    usage = types.SimpleNamespace(prompt_tokens=1, completion_tokens=2, total_tokens=3)
    return types.SimpleNamespace(choices=[choice], usage=usage)


class _RecordingClient:
    """AsyncOpenAI stand-in that records the create() kwargs it is called with (the last call in
    ``captured``, every call in ``calls``) and answers with ``completion`` (a plain
    ``_fake_completion()`` when omitted)."""

    def __init__(self, completion=None):
        self.captured: dict = {}
        self.calls: list[dict] = []
        self._completion = completion
        self.chat = types.SimpleNamespace(completions=types.SimpleNamespace(create=self._create))

    async def _create(self, **kwargs):
        self.captured = kwargs
        self.calls.append(kwargs)
        return self._completion if self._completion is not None else _fake_completion()


def test_generate_forwards_extra_body_verbatim():
    """Fields outside the OpenAI chat schema ride the body as given: this helper is generic, and the
    rollout engines' spellings are owned by ``engine_wire.generation_control_fields``, not restated
    here (a second owner would let the training and eval paths send different ones)."""
    client = _RecordingClient()
    fields = {"reasoning_effort": "high", "thinking_token_budget": 4096}
    resp = asyncio.run(generate_openai_response("m", "hi", client=client, extra_body=fields))
    assert client.captured["extra_body"] == fields
    assert resp.answer == "ok"


def test_generate_omits_extra_body_when_there_is_nothing_to_add():
    """Without extra fields, nothing is sent — the served model's own defaults apply."""
    client = _RecordingClient()
    asyncio.run(generate_openai_response("m", "hi", client=client))
    assert "extra_body" not in client.captured
    asyncio.run(generate_openai_response("m", "hi", client=client, extra_body={}))
    assert "extra_body" not in client.captured


def test_generate_keeps_the_sampled_ids_an_engine_attaches_to_the_choice():
    """vLLM's ``return_token_ids`` puts the sampled ids on the choice outside the OpenAI schema; the SDK
    keeps them as an extra attribute and the response carries them (the overlong charge reads a turn's
    reasoning count off them). Anything but a list is not that capture and reads as absent."""
    with_ids = _RecordingClient(completion=_fake_completion(token_ids=[1, 2, 3]))
    assert asyncio.run(generate_openai_response("m", "hi", client=with_ids)).token_ids == [1, 2, 3]
    not_a_list = _RecordingClient(completion=_fake_completion(token_ids="1,2,3"))
    assert asyncio.run(generate_openai_response("m", "hi", client=not_a_list)).token_ids is None
    assert asyncio.run(generate_openai_response("m", "hi", client=_RecordingClient())).token_ids is None


class _FlakyClient:
    """AsyncOpenAI stand-in that fails create() for designated message contents."""

    def __init__(self, fail_on: set[str]):
        self._fail_on = fail_on
        self.chat = types.SimpleNamespace(completions=types.SimpleNamespace(create=self._create))

    async def _create(self, **kwargs):
        content = kwargs["messages"][-1]["content"]
        if content in self._fail_on:
            raise RuntimeError(f"simulated API failure for {content!r}")
        return _fake_completion(content=f"echo:{content}")


def test_parallel_requests_failed_request_yields_none_without_clobbering_others():
    """A raising request must produce None at ITS index only: a recovery that resolved every failure
    to index -1 would silently overwrite the LAST result."""
    client = _FlakyClient(fail_on={"m1"})
    results = asyncio.run(
        parallel_openai_requests(
            "m",
            ["m0", "m1", "m2"],
            client=client,
            disable_checkpoints=True,
        )
    )
    assert results[0].answer == "echo:m0"
    assert results[1] is None  # the failed request, at its own index
    assert results[2] is not None and results[2].answer == "echo:m2"  # last result NOT clobbered


def test_checkpoint_round_trip_restores_the_sampled_token_ids(tmp_path):
    """A resumed row must carry the ids it was sampled with, as the live response did: a caller that
    asked for them (``return_token_ids``) reads ``None`` as "the engine returned none"."""
    checkpoint_file = str(tmp_path / "requests.jsonl")
    response = OpenAIResponse(answer="a", reasoning=None, finish_reason="stop", tool_calls=None, token_ids=[5, 6, 7])
    append_openai_checkpoint(checkpoint_file, [(0, response)])

    checkpoint = load_openai_checkpoint(checkpoint_file, result_count=1)

    assert checkpoint.results[0] == response


def test_checkpoint_loader_retries_a_record_whose_answer_is_not_text(tmp_path):
    """An answer is the reply's text: a record holding anything else (a parsed JSON object) is
    re-requested rather than handed back as an answer no caller reads."""
    checkpoint_file = tmp_path / "requests.jsonl"
    checkpoint_file.write_text(
        json.dumps({"index": 0, "result": {"answer": {"value": 1}, "finish_reason": "stop"}}) + "\n",
        encoding="utf-8",
    )

    checkpoint = load_openai_checkpoint(str(checkpoint_file), result_count=1)

    assert checkpoint.processed_indices == set()
    assert checkpoint.skipped_records == 1
    assert checkpoint.results[0] is None


def test_checkpoint_loader_skips_bad_records_and_retries_none_results(tmp_path):
    checkpoint_file = tmp_path / "requests.jsonl"
    checkpoint_file.write_text(
        "\n".join(
            [
                "not-json",
                json.dumps({"index": 0, "result": None}),
                json.dumps({"index": 9, "result": {"answer": "outside", "finish_reason": "stop"}}),
                json.dumps({"index": 1, "result": {"answer": "done", "finish_reason": "stop"}}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    checkpoint = load_openai_checkpoint(str(checkpoint_file), result_count=2)

    assert checkpoint.processed_indices == {1}
    assert checkpoint.skipped_records == 3
    assert checkpoint.results[0] is None
    assert checkpoint.results[1] == OpenAIResponse(
        answer="done",
        reasoning=None,
        finish_reason="stop",
        tool_calls=None,
    )


def test_request_tools_distinguishes_common_and_per_message_shapes():
    common_tools = [{"type": "function", "function": {"name": "search"}}]
    common = resolve_request_tools(common_tools, message_count=2)
    assert common.per_message is False
    assert common.for_message(0) == common_tools

    per_message_tools = [common_tools, None]
    per_message = resolve_request_tools(per_message_tools, message_count=2)
    assert per_message.per_message is True
    assert per_message.for_message(0) == common_tools
    assert per_message.for_message(1) is None


def test_request_tools_rejects_per_message_length_mismatch():
    with pytest.raises(ValueError, match="same length"):
        resolve_request_tools([[{"type": "function"}]], message_count=2)


def test_checkpoint_file_hash_is_stable_for_same_request_shape():
    request_tools = resolve_request_tools(None, message_count=1)
    first = resolve_checkpoint_file(
        model="model", messages=["hello"], temperature=0.0, max_tokens=32, request_tools=request_tools
    )
    second = resolve_checkpoint_file(
        model="model", messages=["hello"], temperature=0.0, max_tokens=32, request_tools=request_tools
    )
    assert first == second
    assert first.endswith(".jsonl")


def test_checkpoint_file_hash_differs_for_different_request_shape():
    request_tools = resolve_request_tools(None, message_count=1)
    base_kwargs = {
        "model": "model",
        "messages": ["hello"],
        "temperature": 0.0,
        "max_tokens": 32,
        "request_tools": request_tools,
    }
    baseline = resolve_checkpoint_file(**base_kwargs)
    # Every hashed field must change the resolved filename.
    assert resolve_checkpoint_file(**{**base_kwargs, "model": "other"}) != baseline
    assert resolve_checkpoint_file(**{**base_kwargs, "temperature": 0.7}) != baseline
    assert resolve_checkpoint_file(**{**base_kwargs, "max_tokens": 64}) != baseline
    assert resolve_checkpoint_file(**{**base_kwargs, "messages": ["world"]}) != baseline


def _checkpoint_file_for(messages):
    return resolve_checkpoint_file(
        model="model",
        messages=messages,
        temperature=0.0,
        max_tokens=32,
        request_tools=resolve_request_tools(None, message_count=len(messages)),
    )


def test_checkpoint_file_hash_separates_a_resumed_subset_from_the_full_set():
    """The resume path passes the PENDING rows, not the dataset, so a subset and its full set are two
    different request sets and must key to two different files. A shared checkpoint file replays one
    run's results onto the other's rows by index — and a long shared system prompt is what makes the
    two sets look alike for as far as any prefix of them reaches."""
    system = "S" * 1200  # one row is longer on its own than any fixed prefix of the set
    full = [[{"role": "system", "content": system}, {"role": "user", "content": f"q{i}"}] for i in range(10)]
    pending = full[7:]

    assert _checkpoint_file_for(full) != _checkpoint_file_for(pending)


def test_checkpoint_file_hash_covers_rows_past_the_first_few():
    """An edit to any row changes the request set, including rows past the leading few."""
    base = [[{"role": "user", "content": "x" * 400}] for _ in range(8)]
    edited = [list(row) for row in base]
    edited[7] = [{"role": "user", "content": "a different final prompt"}]

    assert _checkpoint_file_for(base) != _checkpoint_file_for(edited)


def test_tools_hash_covers_every_row():
    """The tools half of the key covers every row too: unchanged prompts with a tool schema edited on
    any row are a different request set, and must not resolve to the unedited run's checkpoint."""
    schema = {"type": "function", "function": {"name": "search", "description": "d" * 600}}
    base = [[schema] for _ in range(6)]
    edited = [list(row) for row in base]
    edited[5] = [{**schema, "function": {**schema["function"], "strict": True}}]

    assert resolve_request_tools(base, 6).hash_sample() != resolve_request_tools(edited, 6).hash_sample()


def test_tools_hash_covers_a_common_schema_to_its_full_length():
    """A single shared tool list is covered to its full length, however long its descriptions run."""

    def tools(tail):
        return [{"type": "function", "function": {"name": "search", "description": "d" * 600 + tail}}]

    assert resolve_request_tools(tools(""), 1).hash_sample() != resolve_request_tools(tools("!"), 1).hash_sample()


def test_load_checkpoint_missing_file_returns_empty_slots():
    checkpoint = load_openai_checkpoint("/nonexistent/path/requests.jsonl", result_count=3)
    assert checkpoint.results == [None, None, None]
    assert checkpoint.processed_indices == set()
    assert checkpoint.skipped_records == 0


def test_append_empty_records_is_noop(tmp_path):
    checkpoint_file = tmp_path / "requests.jsonl"
    append_openai_checkpoint(str(checkpoint_file), [])
    assert not checkpoint_file.exists()


def test_append_round_trip_via_loader(tmp_path):
    # append → load is the real on-disk contract; a plain string answer survives intact.
    checkpoint_file = tmp_path / "requests.jsonl"
    append_openai_checkpoint(
        str(checkpoint_file),
        [(0, OpenAIResponse(answer="hi", reasoning=None, finish_reason="stop", tool_calls=None, total_tokens=4))],
    )
    loaded = load_openai_checkpoint(str(checkpoint_file), result_count=1)
    assert loaded.processed_indices == {0}
    assert loaded.results[0].answer == "hi"
    assert loaded.results[0].total_tokens == 4


def test_request_tools_none_is_not_per_message():
    rt = resolve_request_tools(None, message_count=3)
    assert rt.per_message is False
    assert rt.for_message(0) is None
    assert rt.hash_sample() is None


def test_request_tools_empty_list_is_not_per_message():
    rt = resolve_request_tools([], message_count=0)
    assert rt.per_message is False


def test_generate_requires_an_explicit_keyword_only_client():
    """There is no implicit default client: the argument is keyword-only with no default, so omitting
    it is a TypeError naming it rather than a mid-run 401 from an endpoint the caller never chose."""
    parameter = inspect.signature(generate_openai_response).parameters["client"]
    assert parameter.kind is inspect.Parameter.KEYWORD_ONLY
    assert parameter.default is inspect.Parameter.empty
    with pytest.raises(TypeError, match="keyword-only argument: 'client'"):
        asyncio.run(generate_openai_response("m", "hi"))
    assert asyncio.run(generate_openai_response("m", "hi", client=_RecordingClient())).answer == "ok"


def _empty_completion(error=None):
    """A 200 reply with no choices; ``error`` is the body an aggregator puts beside them, absent when
    None (the shape an SDK object has when the server sent no error either)."""
    completion = types.SimpleNamespace(choices=[])
    if error is not None:
        completion.error = error
    return completion


class _ScriptedClient:
    """AsyncOpenAI stand-in answering each create() with the next scripted completion, the last one
    repeating, and counting the requests it took."""

    def __init__(self, *completions):
        self._completions = completions
        self.requests = 0
        self.chat = types.SimpleNamespace(completions=types.SimpleNamespace(create=self._create))

    async def _create(self, **_kwargs):
        self.requests += 1
        return self._completions[min(self.requests, len(self._completions)) - 1]


def _record_backoff(monkeypatch) -> list[float]:
    """Replace the retry sleep with a recorder of the waits it was asked for."""
    waits: list[float] = []

    async def no_sleep(seconds):
        waits.append(seconds)

    monkeypatch.setattr("src.inference.openai_client.asyncio.sleep", no_sleep)
    return waits


def test_chat_completion_retries_an_empty_reply_on_a_retryable_code_with_backoff(monkeypatch):
    """An aggregator answers 200 with no choices and the upstream code in the body — a rate limit
    the SDK's own retry never sees, so this is the one retry owner."""
    waits = _record_backoff(monkeypatch)
    throttled = _empty_completion({"code": 429, "message": "upstream rate limit"})
    spelled = _empty_completion({"code": "429", "message": "upstream rate limit"})
    client = _ScriptedClient(throttled, spelled, _fake_completion("late"))

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


def test_generate_sends_the_default_temperature_and_omits_a_none():
    """``None`` keeps the served default, which reasoning models require; the default still pins 0.0."""
    client = _RecordingClient()
    asyncio.run(generate_openai_response("m", "hi", client=client))
    assert client.captured["temperature"] == 0.0
    asyncio.run(generate_openai_response("m", "hi", client=client, temperature=None))
    assert "temperature" not in client.captured


@pytest.mark.parametrize(
    ("field", "value"), [("top_p", 0.9), ("tools", [{"type": "function", "function": {"name": "search"}}])]
)
def test_generate_sends_an_optional_request_field_only_when_given(field, value):
    client = _RecordingClient()
    asyncio.run(generate_openai_response("m", "hi", client=client))
    assert field not in client.captured
    asyncio.run(generate_openai_response("m", "hi", client=client, **{field: value}))
    assert client.captured[field] == value


def test_parse_json_object_reads_the_value_or_the_object_embedded_in_prose():
    assert parse_json_object('{"a": 1}') == {"a": 1}
    assert parse_json_object("[1, 2]") == [1, 2]
    assert parse_json_object('Sure: {"a": {"b": 1}} — done') == {"a": {"b": 1}}
    assert parse_json_object("no json here") is None


def test_parallel_requests_checkpoints_the_rows_that_succeeded_and_retries_the_failed_one(tmp_path, monkeypatch):
    """A failed row stays out of the store, so a re-run requests it and only it; the rows beside it
    are replayed from the store."""
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    prompts = ["m0", "m1", "m2"]

    first = asyncio.run(parallel_openai_requests("m", prompts, client=_FlakyClient(fail_on={"m1"})))
    assert first[1] is None
    (store,) = tmp_path.glob("openai_requests_*.jsonl")
    stored = load_openai_checkpoint(str(store), result_count=3)
    assert stored.processed_indices == {0, 2}

    retry = _RecordingClient(completion=_fake_completion("echo:m1"))
    second = asyncio.run(parallel_openai_requests("m", prompts, client=retry))
    assert [call["messages"][-1]["content"] for call in retry.calls] == ["m1"]
    assert [response.answer for response in second] == ["echo:m0", "echo:m1", "echo:m2"]


class _OverlappingClient:
    """AsyncOpenAI stand-in whose create() yields to the loop while in flight, recording the most
    requests ever in flight at once."""

    def __init__(self):
        self.in_flight = 0
        self.peak = 0
        self.chat = types.SimpleNamespace(completions=types.SimpleNamespace(create=self._create))

    async def _create(self, **_kwargs):
        self.in_flight += 1
        self.peak = max(self.peak, self.in_flight)
        for _ in range(3):
            await asyncio.sleep(0)
        self.in_flight -= 1
        return _fake_completion()


@pytest.mark.parametrize("max_workers", [2, 5])
def test_parallel_requests_keeps_at_most_max_workers_in_flight(max_workers):
    """Five overlapping requests: the cap is the peak, and at five the fake proves it overlaps."""
    client = _OverlappingClient()
    asyncio.run(
        parallel_openai_requests(
            "m", [f"q{i}" for i in range(5)], client=client, disable_checkpoints=True, max_workers=max_workers
        )
    )
    assert client.peak == max_workers


def test_parallel_requests_forwards_common_tools_to_every_row_and_per_message_tools_to_theirs():
    tool = {"type": "function", "function": {"name": "search"}}

    def tools_sent(client):
        return {call["messages"][-1]["content"]: call.get("tools") for call in client.calls}

    common = _RecordingClient()
    asyncio.run(parallel_openai_requests("m", ["a", "b"], client=common, disable_checkpoints=True, tools=[tool]))
    assert tools_sent(common) == {"a": [tool], "b": [tool]}

    per_message = _RecordingClient()
    asyncio.run(
        parallel_openai_requests("m", ["a", "b"], client=per_message, disable_checkpoints=True, tools=[[tool], None])
    )
    assert tools_sent(per_message) == {"a": [tool], "b": None}
    assert all("tools" not in call for call in per_message.calls if call["messages"][-1]["content"] == "b")


@pytest.mark.parametrize("exported", [None, ""])
def test_the_client_refuses_to_build_without_a_key(monkeypatch, exported):
    """A blank OPENAI_API_KEY (``.env.example`` ships one) is no key: the SDK would re-read it and build a
    client whose every request fails as an opaque 401, so construction names the missing key instead."""
    if exported is None:
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    else:
        monkeypatch.setenv("OPENAI_API_KEY", exported)
    with pytest.raises(ValueError, match="OPENAI_API_KEY"):
        create_openai_client(api_key_override="")
    monkeypatch.setenv("OPENAI_API_KEY", "sdk-key")
    assert create_openai_client().api_key == "sdk-key"
    assert create_openai_client(api_key_override="override").api_key == "override"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
