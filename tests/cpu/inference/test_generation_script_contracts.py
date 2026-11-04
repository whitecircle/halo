#!/usr/bin/env python
"""Contracts of ``generation/openai_batched_generation.py``, the checkpointing S3 generation CLI."""

import asyncio
import inspect
import sys
import types

import pytest
from datasets import Dataset

from halo.data.sources import s3_client
from halo.entrypoints import resumable_generation
from scripts.inference.generation import openai_batched_generation as batched

_ROWS = [{"id": i, "prompt": [{"role": "user", "content": f"q{i}"}]} for i in range(2)]

_BATCHED_ARGV = ["prog", "--model", "gen-model", "--dataset", "in", "--output_path", "out"]


def _response(answer: str):
    return types.SimpleNamespace(
        answer=answer, tool_calls=None, finish_reason="stop", prompt_tokens=1, completion_tokens=1
    )


_TOOL_CALL = {"id": "call_0", "type": "function", "function": {"name": "lookup", "arguments": "{}"}}


def _tool_call_response():
    """A pure tool-call turn: the API leaves ``content`` unset, the call itself carries the content."""
    return types.SimpleNamespace(
        answer=None,
        tool_calls=[types.SimpleNamespace(model_dump=lambda: dict(_TOOL_CALL))],
        finish_reason="tool_calls",
        prompt_tokens=1,
        completion_tokens=1,
    )


def _stub_module(monkeypatch, module, *, rows, existing, request_impl):
    """Stub a generation run's S3 + API edges; returns the captured request kwargs and pushes."""
    calls: list[dict] = []
    pushed: list = []
    real_requests = inspect.signature(module.parallel_openai_requests)

    async def fake_requests(**kwargs):
        real_requests.bind(**kwargs)
        calls.append(kwargs)
        return request_impl(kwargs["messages"])

    monkeypatch.setattr(module, "load_prompts_with_resume", lambda args, output_uri: (rows, existing))
    monkeypatch.setattr(module, "build_openai_client", lambda **kwargs: object())
    monkeypatch.setattr(module, "parallel_openai_requests", fake_requests)
    monkeypatch.setattr(module, "build_s3_uri", lambda path, subfolder, *, given_as: f"s3://bucket/{subfolder}/{path}")
    monkeypatch.setattr(module, "push_dataset_to_s3_uri", lambda dataset, uri: pushed.append(dataset))
    return calls, pushed


@pytest.mark.parametrize(("module", "argv"), [(batched, _BATCHED_ARGV)], ids=["batched"])
def test_subfolder_none_sentinel_parses_to_a_real_none(module, argv, monkeypatch):
    """A script declaring its own --subfolder would parse the sentinel as a directory name and resume against
    s3://bucket/None/<key>, never finding its output."""
    monkeypatch.setattr(sys, "argv", [*argv, "--subfolder", "None"])
    assert module.parse_args().subfolder is None, (
        "the string 'None' reads and writes s3://bucket/None/<key>, so a resume never finds its own output"
    )


def test_batched_generation_raises_when_every_request_failed(monkeypatch):
    """Exiting 0 on a dead endpoint reports a total failure as complete, and on a resume republishes a previous
    run's rows as the whole job."""
    _, pushed = _stub_module(
        monkeypatch,
        resumable_generation,
        rows=_ROWS,
        existing=[],
        request_impl=lambda messages: [None] * len(messages),
    )
    monkeypatch.setattr(sys, "argv", list(_BATCHED_ARGV))

    with pytest.raises(RuntimeError, match="No usable result"):
        asyncio.run(batched.main())
    assert pushed == []


def test_batched_generation_resolves_its_output_dataset_once(monkeypatch):
    """The resume read, the push and the empty-run refusal name one dataset; a second resolution can land elsewhere."""
    resolved: list[str] = []
    read_from: list[str] = []
    pushed_to: list[str] = []

    def numbered_uri(path, subfolder, *, given_as):
        if given_as == "--output_path":
            resolved.append(f"s3://bucket/run-{len(resolved)}/{path}")
            return resolved[-1]
        return f"s3://bucket/{path}"

    _stub_module(
        monkeypatch,
        resumable_generation,
        rows=_ROWS,
        existing=[],
        request_impl=lambda messages: [_response("a")] * len(messages),
    )
    monkeypatch.setattr(resumable_generation, "build_s3_uri", numbered_uri)
    monkeypatch.setattr(
        resumable_generation,
        "load_prompts_with_resume",
        lambda args, output_uri: read_from.append(output_uri) or (_ROWS, []),
    )
    monkeypatch.setattr(resumable_generation, "push_dataset_to_s3_uri", lambda dataset, uri: pushed_to.append(uri))
    monkeypatch.setattr(sys, "argv", list(_BATCHED_ARGV))
    asyncio.run(batched.main())
    assert len(resolved) == 1 and read_from == pushed_to == resolved

    resolved.clear()
    _stub_module(
        monkeypatch,
        resumable_generation,
        rows=_ROWS,
        existing=[],
        request_impl=lambda messages: [None] * len(messages),
    )
    monkeypatch.setattr(resumable_generation, "build_s3_uri", numbered_uri)
    with pytest.raises(RuntimeError, match="nothing written to s3://bucket/run-0/"):
        asyncio.run(batched.main())


def test_batched_generation_refuses_a_per_row_response_format(monkeypatch):
    """No request carries a schema, so a row's ``response_format`` would be generated as plain text and then popped
    from the record."""
    rows = [{**_ROWS[0], "response_format": {"type": "json_object"}}, _ROWS[1]]
    calls, pushed = _stub_module(
        monkeypatch,
        resumable_generation,
        rows=rows,
        existing=[],
        request_impl=lambda messages: [_response("a")] * len(messages),
    )
    monkeypatch.setattr(sys, "argv", list(_BATCHED_ARGV))

    with pytest.raises(ValueError, match="1 of 2 rows carry a structured response_format"):
        asyncio.run(batched.main())
    assert calls == [] and pushed == []


def test_batched_generation_accepts_a_plain_text_response_format(monkeypatch):
    """``{"type": "text"}`` (the reward-model scripts' default) asks for the plain text this tool produces."""
    rows = [{**row, "response_format": {"type": "text", "json_schema": None}} for row in _ROWS]
    _, pushed = _stub_module(
        monkeypatch,
        resumable_generation,
        rows=rows,
        existing=[],
        request_impl=lambda messages: [_response("a")] * len(messages),
    )
    monkeypatch.setattr(sys, "argv", list(_BATCHED_ARGV))

    asyncio.run(batched.main())
    assert len(pushed) == 1 and pushed[0].num_rows == 2


def test_one_surviving_row_is_the_other_side_of_that_boundary(monkeypatch):
    """The abort keys on every request failing; keying on any failure would throw the run away over one bad row."""
    _, pushed = _stub_module(
        monkeypatch,
        resumable_generation,
        rows=_ROWS,
        existing=[],
        request_impl=lambda messages: [None, _response("second")],
    )
    monkeypatch.setattr(sys, "argv", list(_BATCHED_ARGV))

    asyncio.run(batched.main())
    assert len(pushed) == 1
    assert pushed[0].num_rows == 1


def test_the_checkpoint_interval_flag_is_the_resume_store_flush_interval(monkeypatch):
    """The fake binds each call to the real signature, so a stale keyword fails here."""
    row = {**_ROWS[0], "follow_up_prompt": [{"role": "user", "content": "and now?"}]}
    calls, _ = _stub_module(
        monkeypatch,
        resumable_generation,
        rows=[row],
        existing=[],
        request_impl=lambda messages: [_response("a")] * len(messages),
    )
    monkeypatch.setattr(sys, "argv", [*_BATCHED_ARGV, "--checkpoint_interval", "7"])

    asyncio.run(batched.main())

    assert [call["flush_interval"] for call in calls] == [7, 7]
    assert all(call.get("resume", True) for call in calls)


def test_tool_call_only_first_turn_is_replayed_with_a_null_content(monkeypatch):
    """``str(None)`` would hand the model the literal text None as its own previous answer while the saved record
    says None."""
    row = {
        "id": 0,
        "prompt": [{"role": "user", "content": "q0"}],
        "follow_up_prompt": [{"role": "user", "content": "and now?"}],
    }
    passes: list = []

    def request_impl(messages):
        passes.append(messages)
        return [_tool_call_response() if len(passes) == 1 else _response("done")]

    calls, pushed = _stub_module(monkeypatch, resumable_generation, rows=[row], existing=[], request_impl=request_impl)
    monkeypatch.setattr(sys, "argv", list(_BATCHED_ARGV))

    asyncio.run(batched.main())

    assert len(calls) == 2, "the follow-up pass is the request that replays the first turn"
    sent = next(msg for msg in calls[1]["messages"][0] if msg["role"] == "assistant")
    assert sent["content"] is None, 'str(None) sends the literal text "None" as the assistant\'s answer'
    assert sent["tool_calls"] == [_TOOL_CALL], "the call the turn actually carried must survive the replay"

    saved = pushed[0][0]["conversation"]
    recorded = next(msg for msg in saved if msg["role"] == "assistant" and msg["tool_calls"])
    assert recorded["content"] == sent["content"], "the saved conversation must be the one the model saw"


def test_a_resume_over_another_record_layout_keeps_the_fresh_batch_columns(monkeypatch):
    """The reloaded rows lead the merged list and ``Dataset.from_list`` keys its schema on leading records, so
    without the union in ``save_results_to_s3`` this run's generations are silently dropped."""
    existing = [
        {
            "id": 99,
            "prompt": [{"role": "user", "content": "old"}],
            "chosen": [{"role": "assistant", "content": "a"}],
            "rejected": [{"role": "assistant", "content": "b"}],
        }
    ]
    _, pushed = _stub_module(
        monkeypatch,
        resumable_generation,
        rows=_ROWS,
        existing=existing,
        request_impl=lambda messages: [_response(f"fresh-{i}") for i in range(len(messages))],
    )
    monkeypatch.setattr(sys, "argv", list(_BATCHED_ARGV))

    asyncio.run(batched.main())

    saved = pushed[0]
    assert saved.num_rows == len(existing) + len(_ROWS)
    assert {"chosen", "rejected", "conversation", "finish_reason"} <= set(saved.column_names)
    fresh = saved[len(existing)]
    assert fresh["conversation"][-1]["content"] == "fresh-0", "the fresh batch's own columns must survive the merge"
    assert fresh["chosen"] is None, "a column only the resumed rows carry is filled, not dropped"


def _stub_s3_run(monkeypatch, *, default_bucket: str | None) -> dict[str, list[str]]:
    """Stub the batched run's S3 and API edges, recording every URI it loads, probes and pushes. The output already
    holds row 0, so the resume is real."""
    monkeypatch.setattr(s3_client, "DEFAULT_BUCKET", default_bucket)
    seen: dict[str, list[str]] = {"loaded": [], "probed": [], "pushed": []}
    stored = iter([Dataset.from_list(_ROWS), Dataset.from_list(_ROWS[:1])])

    def fake_load(uri):
        seen["loaded"].append(uri)
        return next(stored)

    class _Client:
        def __init__(self, bucket=None):
            self.bucket = bucket

        def exists(self, key, subfolder=None):
            seen["probed"].append(f"s3://{self.bucket}/{s3_client.S3Client.full_key(key, subfolder)}")
            return True

    async def fake_requests(**kwargs):
        return [_response("a")] * len(kwargs["messages"])

    monkeypatch.setattr(resumable_generation, "load_dataset_from_s3_uri", fake_load)
    monkeypatch.setattr(resumable_generation, "S3Client", _Client)
    monkeypatch.setattr(
        resumable_generation, "push_dataset_to_s3_uri", lambda dataset, uri: seen["pushed"].append(uri)
    )
    monkeypatch.setattr(resumable_generation, "build_openai_client", lambda **kwargs: object())
    monkeypatch.setattr(resumable_generation, "parallel_openai_requests", fake_requests)
    return seen


_URIS = ("s3://in-bucket/prompts", "s3://out-bucket/runs/responses")


@pytest.mark.parametrize(
    ("default_bucket", "paths", "expected"),
    [
        (None, _URIS, _URIS),
        ("team-bucket", _URIS, _URIS),
        (
            "team-bucket",
            ("prompts", "responses"),
            ("s3://team-bucket/datasets/prompts", "s3://team-bucket/datasets/responses"),
        ),
    ],
    ids=["uris_default_unset", "uris_default_set", "bare_keys"],
)
def test_dataset_paths_resolve_to_the_uri_they_name(monkeypatch, default_bucket, paths, expected):
    """A full URI is its own address whatever the default bucket and ``--subfolder`` say; a bare key keeps its place
    under both. The resume probe and push target the output the reload read, or a resumed run regenerates rows
    and publishes them elsewhere."""
    seen = _stub_s3_run(monkeypatch, default_bucket=default_bucket)
    dataset, output_path = paths
    monkeypatch.setattr(
        sys, "argv", ["prog", "--model", "gen-model", "--dataset", dataset, "--output_path", output_path]
    )

    asyncio.run(batched.main())

    in_uri, out_uri = expected
    assert seen == {"loaded": [in_uri, out_uri], "probed": [out_uri], "pushed": [out_uri]}


def test_a_bare_key_without_a_default_bucket_names_its_flag(monkeypatch):
    """The refusal names the flag and value that needed a bucket and the URI spelling that supplies one, and lands
    before any S3 read."""
    seen = _stub_s3_run(monkeypatch, default_bucket=None)
    argv = ["prog", "--model", "gen-model", "--dataset", "s3://in-bucket/prompts", "--output_path", "responses"]
    monkeypatch.setattr(sys, "argv", argv)

    with pytest.raises(ValueError) as excinfo:
        asyncio.run(batched.main())
    message = str(excinfo.value)
    assert "--output_path 'responses'" in message and "s3://<bucket>/<key>" in message
    assert "--bucket" not in message
    assert seen == {"loaded": [], "probed": [], "pushed": []}


def test_an_empty_follow_up_is_no_follow_up():
    """An empty list or a null sends no second request, which would follow the answer with a second assistant turn."""
    turn = [{"role": "user", "content": "And why?"}]
    assert resumable_generation.follow_up_messages({"follow_up_prompt": turn}, "follow_up_prompt") == turn
    for row in ({"follow_up_prompt": []}, {"follow_up_prompt": None}, {}):
        assert resumable_generation.follow_up_messages(row, "follow_up_prompt") is None


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
