import asyncio
import json
import types

import pytest
from pydantic import BaseModel

from src.inference.openai_client import (
    generate_openai_response,
    parallel_openai_requests,
    resolve_checkpoint_file,
    resolve_request_tools,
)
from src.inference.response import OpenAIResponse
from src.inference.resume_store import append_openai_checkpoint, load_openai_checkpoint


def _fake_completion(content: str = "ok"):
    """Minimal stand-in for a chat-completion response (choices[0].message + usage)."""
    message = types.SimpleNamespace(content=content, tool_calls=None, reasoning=None, reasoning_content=None)
    choice = types.SimpleNamespace(message=message, finish_reason="stop", stop_reason=None)
    usage = types.SimpleNamespace(prompt_tokens=1, completion_tokens=2, total_tokens=3)
    return types.SimpleNamespace(choices=[choice], usage=usage)


class _RecordingClient:
    """AsyncOpenAI stand-in that records the create() kwargs it is called with."""

    def __init__(self):
        self.captured: dict = {}
        self.chat = types.SimpleNamespace(completions=types.SimpleNamespace(create=self._create))

    async def _create(self, **kwargs):
        self.captured = kwargs
        return _fake_completion()


def test_generate_forwards_extra_body_verbatim():
    """Fields outside the OpenAI chat schema ride the body as given: this helper is generic, and the
    rollout engines' spellings are owned by ``engine_wire.generation_control_fields``, not restated
    here (a second owner is how the training and eval paths came to send different ones)."""
    client = _RecordingClient()
    fields = {"reasoning_effort": "high", "thinking_token_budget": 4096}
    resp = asyncio.run(generate_openai_response("m", "hi", custom_client=client, extra_body=fields))
    assert client.captured["extra_body"] == fields
    assert resp.answer == "ok"


def test_generate_omits_extra_body_when_there_is_nothing_to_add():
    """Without extra fields, nothing is sent — the served model's own defaults apply."""
    client = _RecordingClient()
    asyncio.run(generate_openai_response("m", "hi", custom_client=client))
    assert "extra_body" not in client.captured
    asyncio.run(generate_openai_response("m", "hi", custom_client=client, extra_body={}))
    assert "extra_body" not in client.captured


def test_parallel_requests_forwards_request_timeout_to_each_call():
    """The batch path has to carry the caller's timeout down to the HTTP request.

    Forwarding only ``reasoning_effort`` pins every batch caller to the 180s default however short a
    deadline it enforces around the batch — an outer ``wait_for`` gives up while the socket it
    abandoned stays live for another minute.
    """
    client = _RecordingClient()
    asyncio.run(
        parallel_openai_requests("m", ["hi"], custom_client=client, disable_checkpoints=True, request_timeout=12.5)
    )
    assert client.captured["timeout"] == 12.5


def test_parallel_requests_default_timeout_matches_the_single_request_path():
    """Plumbing only: an omitted timeout still resolves to the same default a direct call gets."""
    batch, single = _RecordingClient(), _RecordingClient()
    asyncio.run(parallel_openai_requests("m", ["hi"], custom_client=batch, disable_checkpoints=True))
    asyncio.run(generate_openai_response("m", "hi", custom_client=single))
    assert batch.captured["timeout"] == single.captured["timeout"]


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
    """A raising request must produce None at ITS index only. The removed as_completed recovery
    branch resolved every failure to index -1, silently overwriting the LAST result — this pins the
    (idx, None) contract that replaced it."""
    client = _FlakyClient(fail_on={"m1"})
    results = asyncio.run(
        parallel_openai_requests(
            "m",
            ["m0", "m1", "m2"],
            custom_client=client,
            disable_checkpoints=True,
        )
    )
    assert results[0].answer == "echo:m0"
    assert results[1] is None  # the failed request, at its own index
    assert results[2] is not None and results[2].answer == "echo:m2"  # last result NOT clobbered


class StructuredAnswer(BaseModel):
    value: int


def test_checkpoint_round_trip_reconstructs_response_format(tmp_path):
    checkpoint_file = tmp_path / "requests.jsonl"
    append_openai_checkpoint(
        str(checkpoint_file),
        [
            (
                1,
                OpenAIResponse(
                    answer=StructuredAnswer(value=7),
                    reasoning="ok",
                    finish_reason="stop",
                    tool_calls=None,
                    prompt_tokens=2,
                    completion_tokens=3,
                    total_tokens=5,
                ),
            )
        ],
    )

    checkpoint = load_openai_checkpoint(
        str(checkpoint_file),
        result_count=3,
        response_format=StructuredAnswer,
    )

    assert checkpoint.processed_indices == {1}
    assert checkpoint.skipped_records == 0
    result = checkpoint.results[1]
    assert result is not None
    assert result.answer == StructuredAnswer(value=7)
    assert result.total_tokens == 5


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

    checkpoint = load_openai_checkpoint(
        str(checkpoint_file),
        result_count=2,
        response_format=None,
    )

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
        None,
        model="model",
        user_messages=["hello"],
        system_prompt=None,
        temperature=0.0,
        max_tokens=32,
        request_tools=request_tools,
        response_format=None,
    )
    second = resolve_checkpoint_file(
        None,
        model="model",
        user_messages=["hello"],
        system_prompt=None,
        temperature=0.0,
        max_tokens=32,
        request_tools=request_tools,
        response_format=None,
    )
    assert first == second
    assert first.endswith(".jsonl")


def test_checkpoint_file_hash_differs_for_different_request_shape():
    request_tools = resolve_request_tools(None, message_count=1)
    base_kwargs = {
        "checkpoint_file": None,
        "model": "model",
        "user_messages": ["hello"],
        "system_prompt": None,
        "temperature": 0.0,
        "max_tokens": 32,
        "request_tools": request_tools,
        "response_format": None,
    }
    baseline = resolve_checkpoint_file(**base_kwargs)
    # Every hashed field must change the resolved filename.
    assert resolve_checkpoint_file(**{**base_kwargs, "model": "other"}) != baseline
    assert resolve_checkpoint_file(**{**base_kwargs, "temperature": 0.7}) != baseline
    assert resolve_checkpoint_file(**{**base_kwargs, "max_tokens": 64}) != baseline
    assert resolve_checkpoint_file(**{**base_kwargs, "user_messages": ["world"]}) != baseline
    assert resolve_checkpoint_file(**{**base_kwargs, "response_format": StructuredAnswer}) != baseline


def _checkpoint_file_for(user_messages):
    return resolve_checkpoint_file(
        None,
        model="model",
        user_messages=user_messages,
        system_prompt=None,
        temperature=0.0,
        max_tokens=32,
        request_tools=resolve_request_tools(None, message_count=len(user_messages)),
        response_format=None,
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


def test_tools_hash_covers_every_row_past_the_truncated_prefix():
    """The tools half of the key covers every row too: unchanged prompts with a tool schema edited on
    any row are a different request set, and must not resolve to the unedited run's checkpoint."""
    schema = {"type": "function", "function": {"name": "search", "description": "d" * 600}}
    base = [[schema] for _ in range(6)]
    edited = [list(row) for row in base]
    edited[5] = [{**schema, "function": {**schema["function"], "strict": True}}]

    assert resolve_request_tools(base, 6).hash_sample() != resolve_request_tools(edited, 6).hash_sample()


def test_tools_hash_covers_a_common_schema_past_the_old_character_cutoff():
    """A single shared tool list is covered to its full length, however long its descriptions run."""

    def tools(tail):
        return [{"type": "function", "function": {"name": "search", "description": "d" * 600 + tail}}]

    assert resolve_request_tools(tools(""), 1).hash_sample() != resolve_request_tools(tools("!"), 1).hash_sample()


def test_load_checkpoint_missing_file_returns_empty_slots():
    checkpoint = load_openai_checkpoint("/nonexistent/path/requests.jsonl", result_count=3, response_format=None)
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
    loaded = load_openai_checkpoint(str(checkpoint_file), result_count=1, response_format=None)
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


class _SchemaCrashingFormat(BaseModel):
    """A response_format whose parse blows up for a reason that is NOT a schema mismatch."""

    value: int

    @classmethod
    def model_validate_json(cls, *_args, **_kwargs):
        raise RuntimeError("schema construction bug")


def test_structured_parse_does_not_swallow_non_validation_errors():
    """A parse failure that is not a schema/JSON mismatch must surface, not be reported as bad JSON.

    The fallback chain (native parse -> regex extraction -> ValueError) exists for payloads that are
    not the requested model's JSON. Catching everything there turns a real defect — a broken schema,
    a caller passing a non-model — into "Response does not contain valid JSON", blaming the served
    model.
    """

    class _JsonClient:
        def __init__(self):
            self.chat = types.SimpleNamespace(completions=types.SimpleNamespace(create=self._create))

        async def _create(self, **_kwargs):
            return _fake_completion('{"value": 1}')

    with pytest.raises(RuntimeError, match="schema construction bug"):
        asyncio.run(
            generate_openai_response("m", "hi", response_format=_SchemaCrashingFormat, custom_client=_JsonClient())
        )


def test_structured_parse_still_falls_back_to_regex_extraction():
    """A schema mismatch on the native parse still falls through to regex extraction (unchanged)."""

    class _ProseWrappedClient:
        def __init__(self):
            self.chat = types.SimpleNamespace(completions=types.SimpleNamespace(create=self._create))

        async def _create(self, **_kwargs):
            return _fake_completion('here you go: {"value": 7} hope that helps')

    resp = asyncio.run(
        generate_openai_response("m", "hi", response_format=StructuredAnswer, custom_client=_ProseWrappedClient())
    )
    assert resp.answer == StructuredAnswer(value=7)


def test_structured_parse_raises_when_no_json_present():
    """No JSON at all in the content still raises the caller-facing ValueError (unchanged)."""

    class _ProseClient:
        def __init__(self):
            self.chat = types.SimpleNamespace(completions=types.SimpleNamespace(create=self._create))

        async def _create(self, **_kwargs):
            return _fake_completion("no json here")

    with pytest.raises(ValueError, match="does not contain valid JSON"):
        asyncio.run(
            generate_openai_response("m", "hi", response_format=StructuredAnswer, custom_client=_ProseClient())
        )


def test_generate_requires_an_explicit_client():
    """There is no implicit default client — omitting one is a config error, not a mid-run 401.

    The previous fallback built an OpenRouter client from whatever keys happened to be in the
    environment (empty string when none), so a caller that forgot to pass a client got a 401 from a
    third-party endpoint it never chose instead of a message naming the missing argument.
    """
    with pytest.raises(ValueError, match="custom_client is required"):
        asyncio.run(generate_openai_response("m", "hi"))


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
