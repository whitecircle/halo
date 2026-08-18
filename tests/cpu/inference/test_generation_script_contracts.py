#!/usr/bin/env python
"""Contracts of the checkpointing S3 generation CLI.

``generation/openai_batched_generation.py`` reads a prompt dataset from S3, resumes against a
partially written output dataset, and pushes the merged result back. Three ways that silently
corrupts or throws away a run:

* ``--subfolder None`` is the documented "no subfolder" sentinel. Left as the STRING ``"None"`` it
  reads and writes ``s3://bucket/None/<key>``, so the resume never matches its own output.
* ``Dataset.from_list`` infers its schema from the leading records and drops keys absent there — a
  resume that mixes record layouts writes the fresh batch's own columns away as if they never were.
* A tool-call-only turn carries ``content: None``. Spelled ``str(answer)`` into the follow-up
  request, the model is told its own previous answer was the literal text "None" — and the record
  written to S3 for that same message disagrees with the conversation the model actually saw.

Run: python tests/cpu/inference/test_generation_script_contracts.py  (or pytest)
"""

import asyncio
import sys
import types

import pytest

from scripts.inference import _common
from scripts.inference.generation import openai_batched_generation as batched

_ROWS = [{"id": i, "prompt": [{"role": "user", "content": f"q{i}"}]} for i in range(2)]

_BATCHED_ARGV = ["prog", "--model_name", "gen-model", "--input_path", "in", "--output_path", "out"]


def _response(answer: str):
    """Minimal stand-in for an ``OpenAIResponse`` as the generation scripts consume one."""
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
    """Stub a generation script's S3 + API edges; returns the captured request kwargs and pushes."""
    calls: list[dict] = []
    pushed: list = []

    async def fake_requests(**kwargs):
        calls.append(kwargs)
        return request_impl(kwargs["user_messages"])

    monkeypatch.setattr(module, "load_prompts_with_resume", lambda args: (rows, existing))
    monkeypatch.setattr(module, "create_openai_client", lambda **kwargs: object())
    monkeypatch.setattr(module, "parallel_openai_requests", fake_requests)
    # The merge-and-push tail is shared, so the S3 edge is stubbed in the shared module — the
    # scripts reach it through their own main().
    monkeypatch.setattr(_common, "build_s3_uri", lambda path, subfolder: f"s3://bucket/{subfolder}/{path}")
    monkeypatch.setattr(_common, "push_dataset_to_s3_uri", lambda dataset, uri: pushed.append(dataset))
    return calls, pushed


# --- `--subfolder None` sentinel -------------------------------------------------------------


@pytest.mark.parametrize(("module", "argv"), [(batched, _BATCHED_ARGV)], ids=["batched"])
def test_subfolder_none_sentinel_parses_to_a_real_none(module, argv, monkeypatch):
    """Driven through the script rather than on the shared parser alone: a script declaring its own
    --subfolder would parse the sentinel as a literal directory name and resume against
    s3://bucket/None/<key> — its own output, never found."""
    monkeypatch.setattr(sys, "argv", [*argv, "--subfolder", "None"])
    assert module.parse_args().subfolder is None, (
        "the string 'None' reads and writes s3://bucket/None/<key>, so a resume never finds its own output"
    )


# --- batched generation: an all-failed run is not a success ----------------------------------


def test_batched_generation_raises_when_every_request_failed(monkeypatch):
    """Exiting 0 on a dead endpoint reports a total failure as a completed generation run — and on
    a resume it republishes the rows a previous run wrote as if they were the whole job."""
    _, pushed = _stub_module(
        monkeypatch, batched, rows=_ROWS, existing=[], request_impl=lambda messages: [None] * len(messages)
    )
    monkeypatch.setattr(sys, "argv", list(_BATCHED_ARGV))

    with pytest.raises(RuntimeError, match="No usable result"):
        asyncio.run(batched.main())
    assert pushed == []


def test_one_surviving_row_is_the_other_side_of_that_boundary(monkeypatch):
    """The abort keys on EVERY request failing, so exactly one usable response still publishes:
    a guard that keyed on "any failure" would throw the run away over a single bad row."""
    _, pushed = _stub_module(
        monkeypatch,
        batched,
        rows=_ROWS,
        existing=[],
        request_impl=lambda messages: [None, _response("second")],
    )
    monkeypatch.setattr(sys, "argv", list(_BATCHED_ARGV))

    asyncio.run(batched.main())
    assert len(pushed) == 1
    assert pushed[0].num_rows == 1


# --- batched generation: the follow-up request replays the first turn verbatim -----------------


def test_tool_call_only_first_turn_is_replayed_with_a_null_content(monkeypatch):
    """The follow-up request re-sends the first assistant turn. A tool-call-only turn has no
    content, and ``str(None)`` hands the model the literal text "None" as its own previous answer —
    while the record saved for that same message says None, so the two disagree."""
    row = {
        "id": 0,
        "prompt": [{"role": "user", "content": "q0"}],
        "follow_up_prompt": [{"role": "user", "content": "and now?"}],
    }
    passes: list = []

    def request_impl(messages):
        passes.append(messages)
        return [_tool_call_response() if len(passes) == 1 else _response("done")]

    calls, pushed = _stub_module(monkeypatch, batched, rows=[row], existing=[], request_impl=request_impl)
    monkeypatch.setattr(sys, "argv", list(_BATCHED_ARGV))

    asyncio.run(batched.main())

    assert len(calls) == 2, "the follow-up pass is the request that replays the first turn"
    sent = next(msg for msg in calls[1]["user_messages"][0] if msg["role"] == "assistant")
    assert sent["content"] is None, 'str(None) sends the literal text "None" as the assistant\'s answer'
    assert sent["tool_calls"] == [_TOOL_CALL], "the call the turn actually carried must survive the replay"

    saved = pushed[0][0]["conversation"]
    recorded = next(msg for msg in saved if msg["role"] == "assistant" and msg["tool_calls"])
    assert recorded["content"] == sent["content"], "the saved conversation must be the one the model saw"


# --- the resumed push unions the record keys ---------------------------------------------------


def test_a_resume_over_another_record_layout_keeps_the_fresh_batch_columns(monkeypatch):
    """A reused --output_path can already hold rows in another training format, and the reloaded rows
    lead the merged list. ``Dataset.from_list`` keys its schema on those leading records, so without
    the union in ``save_results_to_s3`` this run's own generations are written away as if the pass
    had never happened — a silent data loss the run reports as success."""
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
        batched,
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


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
