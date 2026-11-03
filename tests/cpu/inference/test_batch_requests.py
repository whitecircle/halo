#!/usr/bin/env python
"""The parallel resumable batch: one response per row at its own index, at most ``max_workers`` in flight, and the
JSONL resume store a re-run replays."""

import asyncio
import json
from pathlib import Path

import pytest

from halo import env
from halo.inference import batch_requests
from halo.inference.batch_requests import (
    append_resume_records,
    load_resume_records,
    parallel_openai_requests,
    resolve_request_tools,
    resume_store_path,
)
from halo.inference.response import OpenAIResponse
from tests.common.fake_chat_client import FakeChatClient, RecordingChatClient, fake_completion

_TOOL = {"type": "function", "function": {"name": "lookup", "parameters": {"type": "object"}}}
# A store in the on-disk line format, written in two flushes, rows in completion order: a tool-call turn
# with a null answer, a turn with reasoning and sampled ids, and a non-ASCII turn cut at its token cap.
_STORE_LINES = (
    '{"index": 2, "result": {"answer": null, "reasoning": null, "finish_reason": "tool_calls", "tool_calls": '
    '[{"id": "call_0", "function": {"arguments": "{\\"q\\": \\"Zürich\\"}", "name": "lookup"}, "type": "function"}], '
    '"prompt_tokens": 20, "completion_tokens": 5, "total_tokens": 25, "token_ids": null}}\n'
    '{"index": 0, "result": {"answer": "Paris", "reasoning": "The capital of France.", "finish_reason": "stop", '
    '"tool_calls": null, "prompt_tokens": 12, "completion_tokens": 3, "total_tokens": 15, "token_ids": [9, 8, 7]}}\n'
    '{"index": 1, "result": {"answer": "Déjà — cut", "reasoning": null, "finish_reason": "length", "tool_calls": '
    'null, "prompt_tokens": 0, "completion_tokens": 64, "total_tokens": 0, "token_ids": null}}\n'
)
# The batch without its store; the flush interval paces only the store, so its value is unused.
_WITHOUT_STORE = {"resume": False, "flush_interval": 1}


def _stored_rows() -> list[OpenAIResponse]:
    call = {"id": "call_0", "type": "function", "function": {"name": "lookup", "arguments": '{"q": "Zürich"}'}}
    return [
        OpenAIResponse(
            answer="Paris",
            reasoning="The capital of France.",
            finish_reason="stop",
            tool_calls=None,
            prompt_tokens=12,
            completion_tokens=3,
            total_tokens=15,
            token_ids=[9, 8, 7],
        ),
        OpenAIResponse(
            answer="Déjà — cut", reasoning=None, finish_reason="length", tool_calls=None, completion_tokens=64
        ),
        OpenAIResponse(
            answer=None,
            reasoning=None,
            finish_reason="tool_calls",
            tool_calls=[call],
            prompt_tokens=20,
            completion_tokens=5,
            total_tokens=25,
        ),
    ]


class _FlakyClient(FakeChatClient):
    """AsyncOpenAI stand-in that fails create() for designated message contents."""

    def __init__(self, fail_on: set[str]):
        super().__init__()
        self._fail_on = fail_on

    async def _create(self, **kwargs):
        content = kwargs["messages"][-1]["content"]
        if content in self._fail_on:
            raise RuntimeError(f"simulated API failure for {content!r}")
        return fake_completion(content=f"echo:{content}")


class _OverlappingClient(FakeChatClient):
    """AsyncOpenAI stand-in whose create() yields to the loop while in flight, recording the most
    requests ever in flight at once."""

    def __init__(self):
        super().__init__()
        self.in_flight = 0
        self.peak = 0

    async def _create(self, **_kwargs):
        self.in_flight += 1
        self.peak = max(self.peak, self.in_flight)
        for _ in range(3):
            await asyncio.sleep(0)
        self.in_flight -= 1
        return fake_completion()


def _store_path_for(messages, tools=None):
    return resume_store_path(
        model="model",
        messages=messages,
        temperature=0.0,
        max_tokens=32,
        request_tools=resolve_request_tools(tools, message_count=len(messages)),
    )


def test_parallel_requests_failed_request_yields_none_without_clobbering_others():
    """A recovery that resolved every failure to index -1 would silently overwrite the last result."""
    client = _FlakyClient(fail_on={"m1"})
    results = asyncio.run(parallel_openai_requests("m", ["m0", "m1", "m2"], client=client, **_WITHOUT_STORE))
    assert results[0].answer == "echo:m0"
    assert results[1] is None
    assert results[2] is not None and results[2].answer == "echo:m2"


def test_parallel_requests_stores_the_rows_that_succeeded_and_retries_the_failed_one(tmp_path, monkeypatch):
    """A failed row stays out of the store, so a re-run requests only it. The store lives under HALO_DATA_ROOT: in
    the temp dir, it is gone with a container whose TMPDIR is not mounted."""
    monkeypatch.setattr(env, "HALO_DATA_ROOT", str(tmp_path))
    prompts = ["m0", "m1", "m2"]

    first = asyncio.run(parallel_openai_requests("m", prompts, client=_FlakyClient(fail_on={"m1"}), flush_interval=10))
    assert first[1] is None
    (store,) = tmp_path.glob("inference/resume/openai_requests_*.jsonl")
    stored = load_resume_records(str(store), result_count=3)
    assert stored.processed_indices == {0, 2}

    retry = RecordingChatClient(completion=fake_completion("echo:m1"))
    second = asyncio.run(parallel_openai_requests("m", prompts, client=retry, flush_interval=10))
    assert [call["messages"][-1]["content"] for call in retry.calls] == ["m1"]
    assert [response.answer for response in second] == ["echo:m0", "echo:m1", "echo:m2"]


def test_the_store_is_appended_every_flush_interval_completions_and_once_at_the_end(tmp_path, monkeypatch):
    """A failed row counts toward the interval but is never stored. The fake answers without yielding, so rows
    complete in order."""
    monkeypatch.setattr(env, "HALO_DATA_ROOT", str(tmp_path))
    flushes: list[list[int]] = []
    monkeypatch.setattr(
        batch_requests, "append_resume_records", lambda path, records: flushes.append([i for i, _ in records])
    )
    prompts = [f"m{i}" for i in range(8)]

    asyncio.run(parallel_openai_requests("m", prompts, client=_FlakyClient(fail_on={"m4"}), flush_interval=3))
    assert flushes == [[0, 1, 2], [3, 5], [6, 7]]

    flushes.clear()
    asyncio.run(parallel_openai_requests("m", prompts, client=_FlakyClient(set()), **_WITHOUT_STORE))
    assert flushes == []


@pytest.mark.parametrize(
    ("request_fields", "tools", "name"),
    [
        pytest.param(
            {"model": "gen-model", "messages": ["hello", "world"], "temperature": 0.0, "max_tokens": 512},
            None,
            "openai_requests_17ac2b8bf4772e142fa5f5a620c06f4a.jsonl",
            id="no-tools",
        ),
        pytest.param(
            {
                "model": "gen-model",
                "messages": [[{"role": "user", "content": "q0"}], [{"role": "user", "content": "q1"}]],
                "temperature": None,
                "max_tokens": 64,
            },
            [_TOOL],
            "openai_requests_1d6bf86a2fd2faae4a979390dc182a10.jsonl",
            id="common-tools",
        ),
        pytest.param(
            {
                "model": "gen-model",
                "messages": [
                    [{"role": "system", "content": "S"}, {"role": "user", "content": "q0"}],
                    [{"role": "user", "content": "q1"}],
                ],
                "temperature": 0.7,
                "max_tokens": 64,
            },
            [[_TOOL], None],
            "openai_requests_83ab43056b0ad2d82dd63b721c282883.jsonl",
            id="per-message-tools",
        ),
    ],
)
def test_the_store_name_is_pinned_per_request_identity(request_fields, tools, name):
    """A renamed store is one a re-run would miss after an interrupted run wrote it."""
    request_tools = resolve_request_tools(tools, len(request_fields["messages"]))
    assert Path(resume_store_path(**request_fields, request_tools=request_tools)).name == name


def test_a_store_in_the_line_format_replays_its_rows_and_is_written_line_for_line(tmp_path):
    store = tmp_path / "store.jsonl"
    store.write_text(_STORE_LINES, encoding="utf-8")

    loaded = load_resume_records(str(store), result_count=4)

    assert loaded.results == [*_stored_rows(), None]
    assert loaded.processed_indices == {0, 1, 2} and loaded.skipped_records == 0

    rewritten = tmp_path / "rewritten.jsonl"
    rows = _stored_rows()
    append_resume_records(str(rewritten), [(2, rows[2]), (0, rows[0])])
    append_resume_records(str(rewritten), [(1, rows[1])])
    text = rewritten.read_text(encoding="utf-8")
    assert [json.loads(line) for line in text.splitlines()] == [json.loads(line) for line in _STORE_LINES.splitlines()]
    assert "Déjà — cut" in text, "non-ASCII text is written as itself, not escaped"


def test_store_round_trip_restores_the_sampled_token_ids(tmp_path):
    """A caller that asked for ids (``return_token_ids``) reads ``None`` as the engine returning none."""
    store = str(tmp_path / "requests.jsonl")
    response = OpenAIResponse(answer="a", reasoning=None, finish_reason="stop", tool_calls=None, token_ids=[5, 6, 7])
    append_resume_records(store, [(0, response)])

    assert load_resume_records(store, result_count=1).results[0] == response


def test_store_loader_retries_a_record_whose_answer_is_not_text(tmp_path):
    store = tmp_path / "requests.jsonl"
    store.write_text(
        json.dumps({"index": 0, "result": {"answer": {"value": 1}, "finish_reason": "stop"}}) + "\n",
        encoding="utf-8",
    )

    loaded = load_resume_records(str(store), result_count=1)

    assert loaded.processed_indices == set()
    assert loaded.skipped_records == 1
    assert loaded.results[0] is None


def test_store_loader_skips_bad_records_and_retries_none_results(tmp_path):
    store = tmp_path / "requests.jsonl"
    store.write_text(
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

    loaded = load_resume_records(str(store), result_count=2)

    assert loaded.processed_indices == {1}
    assert loaded.skipped_records == 3
    assert loaded.results[0] is None
    assert loaded.results[1] == OpenAIResponse(answer="done", reasoning=None, finish_reason="stop", tool_calls=None)


def test_load_of_a_missing_store_returns_empty_slots():
    loaded = load_resume_records("/nonexistent/path/requests.jsonl", result_count=3)
    assert loaded.results == [None, None, None]
    assert loaded.processed_indices == set()
    assert loaded.skipped_records == 0


def test_append_empty_records_is_noop(tmp_path):
    store = tmp_path / "requests.jsonl"
    append_resume_records(str(store), [])
    assert not store.exists()


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


def test_request_tools_none_or_empty_is_not_per_message():
    none = resolve_request_tools(None, message_count=3)
    assert none.per_message is False
    assert none.for_message(0) is None
    assert resolve_request_tools([], message_count=0).per_message is False


def test_store_path_differs_for_a_different_request():
    request_tools = resolve_request_tools(None, message_count=1)
    base_kwargs = {
        "model": "model",
        "messages": ["hello"],
        "temperature": 0.0,
        "max_tokens": 32,
        "request_tools": request_tools,
    }
    baseline = resume_store_path(**base_kwargs)
    assert resume_store_path(**{**base_kwargs, "model": "other"}) != baseline
    assert resume_store_path(**{**base_kwargs, "temperature": 0.7}) != baseline
    assert resume_store_path(**{**base_kwargs, "max_tokens": 64}) != baseline
    assert resume_store_path(**{**base_kwargs, "messages": ["world"]}) != baseline


def test_store_path_separates_a_resumed_subset_from_the_full_set():
    """The resume path passes the pending rows, so a subset and its full set must key to two files; a shared store
    replays one run's results onto the other's rows by index."""
    system = "S" * 1200  # one row is longer on its own than any fixed prefix of the set
    full = [[{"role": "system", "content": system}, {"role": "user", "content": f"q{i}"}] for i in range(10)]
    pending = full[7:]

    assert _store_path_for(full) != _store_path_for(pending)


def test_store_path_covers_rows_past_the_first_few():
    base = [[{"role": "user", "content": "x" * 400}] for _ in range(8)]
    edited = [list(row) for row in base]
    edited[7] = [{"role": "user", "content": "a different final prompt"}]

    assert _store_path_for(base) != _store_path_for(edited)


def test_store_path_covers_every_row_of_the_tools():
    schema = {"type": "function", "function": {"name": "search", "description": "d" * 600}}
    base = [[schema] for _ in range(6)]
    edited = [list(row) for row in base]
    edited[5] = [{**schema, "function": {**schema["function"], "strict": True}}]

    prompts = [f"q{i}" for i in range(6)]
    assert _store_path_for(prompts, base) != _store_path_for(prompts, edited)


def test_store_path_covers_a_common_tool_schema_to_its_full_length():
    def tools(tail):
        return [{"type": "function", "function": {"name": "search", "description": "d" * 600 + tail}}]

    assert _store_path_for(["q"], tools("")) != _store_path_for(["q"], tools("!"))


@pytest.mark.parametrize("max_workers", [2, 5])
def test_parallel_requests_keeps_at_most_max_workers_in_flight(max_workers):
    """Five overlapping requests: the cap is the peak, and at five the fake proves it overlaps."""
    client = _OverlappingClient()
    prompts = [f"q{i}" for i in range(5)]
    asyncio.run(parallel_openai_requests("m", prompts, client=client, max_workers=max_workers, **_WITHOUT_STORE))
    assert client.peak == max_workers


def test_parallel_requests_forwards_common_tools_to_every_row_and_per_message_tools_to_theirs():
    def tools_sent(client):
        return {call["messages"][-1]["content"]: call.get("tools") for call in client.calls}

    common = RecordingChatClient()
    asyncio.run(parallel_openai_requests("m", ["a", "b"], client=common, tools=[_TOOL], **_WITHOUT_STORE))
    assert tools_sent(common) == {"a": [_TOOL], "b": [_TOOL]}

    per_message = RecordingChatClient()
    asyncio.run(parallel_openai_requests("m", ["a", "b"], client=per_message, tools=[[_TOOL], None], **_WITHOUT_STORE))
    assert tools_sent(per_message) == {"a": [_TOOL], "b": None}
    assert all("tools" not in call for call in per_message.calls if call["messages"][-1]["content"] == "b")


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
