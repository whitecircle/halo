#!/usr/bin/env python
"""CPU tests: the scoring sample and the views of it a scorer reads — how a transcript, a digest and
a final answer render, how a long view is cut, and what a chat-template scorer is handed.

Run: python tests/cpu/rewards/test_samples.py  (or pytest)
"""

import json

import pytest

from src.rewards.samples import (
    CUT_MARKER,
    DIGEST_CHARS,
    DIGEST_INLINE_CHARS,
    NO_FINAL_ANSWER,
    WIRE_KEYS,
    ScoringSample,
    cut_middle,
    final_assistant_text,
    render_digest,
    render_final_answer,
    render_tools,
    render_transcript,
    samples_from_completions,
    scored_messages,
    task_text,
    view_text,
)
from src.rewards.terms import View

CALL = {"id": "call_1", "type": "function", "function": {"name": "run", "arguments": '{"code": "print(1)", "n": 2}'}}
TURNS = [
    {"role": "assistant", "reasoning_content": "Let me run it.", "content": "Running.", "tool_calls": [CALL]},
    {"role": "tool", "name": "run", "tool_call_id": "call_1", "content": "1"},
    {"role": "assistant", "content": "The answer is 1."},
]
PROMPT = [{"role": "system", "content": "Be terse."}, {"role": "user", "content": "Run it."}]


def test_cut_middle_keeps_the_head_and_the_tail_with_the_marker():
    text = "H" * 50 + "T" * 50
    cut = cut_middle(text, 30)
    assert cut == "H" * 20 + CUT_MARKER.format(dropped=70) + "T" * 10
    assert "[70 chars cut here]" in cut
    assert cut_middle(text, 100) == text and cut_middle("short", 30) == "short"


def test_render_transcript_numbers_turns_and_sets_reasoning_apart():
    shown = render_transcript(TURNS, include_reasoning=True)
    assert shown.startswith(
        "[1] assistant\n<reasoning>\nLet me run it.\n</reasoning>\nRunning.\n→ run (call call_1)\n  code: "
    )
    hidden = render_transcript(TURNS, include_reasoning=False)
    assert hidden.startswith("[1] assistant\nRunning.\n→ run") and "<reasoning>" not in hidden
    assert "Let me run it." not in hidden
    # vLLM spells the reasoning ``reasoning``; SGLang ``reasoning_content``.
    vllm = [{"role": "assistant", "reasoning": "r", "content": "c"}]
    assert render_transcript(vllm, include_reasoning=True) == "[1] assistant\n<reasoning>\nr\n</reasoning>\nc"


def test_render_transcript_shows_tool_calls_with_their_id_and_results_under_the_call():
    assert render_transcript(TURNS, include_reasoning=False) == (
        "[1] assistant\nRunning.\n→ run (call call_1)\n  code: print(1)\n  n: 2\n\n"
        "[2] tool run (call call_1)\n1\n\n"
        "[3] assistant\nThe answer is 1."
    )
    # Arguments that are not JSON stay verbatim; a call without an id carries no call tag.
    raw = [{"role": "assistant", "tool_calls": [{"function": {"name": "sh", "arguments": "ls -la"}}]}]
    assert render_transcript(raw, include_reasoning=False) == "[1] assistant\n→ sh: ls -la"
    # A multi-line argument stands on its own lines, verbatim: a quote from a program matches it as written.
    code = 'import sys\nprint("hi")'
    submit = [
        {
            "role": "assistant",
            "tool_calls": [
                {"function": {"name": "submit", "arguments": json.dumps({"code": code, "language": "python"})}}
            ],
        }
    ]
    assert (
        render_transcript(submit, include_reasoning=False)
        == f"[1] assistant\n→ submit\n  code:\n{code}\n  language: python"
    )


def test_render_transcript_marks_flagged_turns_with_their_notes():
    turns = [
        {"role": "assistant", "content": "Let me th", "truncated": True},
        {"role": "assistant", "content": "", "empty": True},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"function": {"name": "nope", "arguments": "{}"}}],
            "calls_rejected": True,
            "truncated": True,
        },
    ]
    assert render_transcript(turns, include_reasoning=True) == (
        "[1] assistant  — cut by the engine at its length limit\nLet me th\n\n"
        "[2] assistant  — ended with neither visible text nor a tool call\n\n"
        "[3] assistant  — cut by the engine at its length limit; every tool call named a tool that does not exist, was refused unrun or ran and showed nothing\n"
        "→ nope"
    )


def test_render_digest_caps_each_item_and_ends_with_the_final_answer():
    reasoning, text, result, long_arg = "r" * 1000, "t" * 1000, "o" * 1000, "a" * 300
    arguments = json.dumps({"code": long_arg, "n": 2, "flag": "on"})
    sample = ScoringSample(
        prompt=PROMPT,
        completion=[
            {
                "role": "assistant",
                "reasoning_content": reasoning,
                "content": text,
                "tool_calls": [{"id": "c1", "function": {"name": "run", "arguments": arguments}}],
            },
            {"role": "tool", "name": "run", "tool_call_id": "c1", "content": result},
        ],
        final_answer="f" * 1000,
    )
    digest = render_digest(sample, include_reasoning=True)
    assert f"<reasoning>\n{'r' * DIGEST_CHARS}…[600 more chars]\n</reasoning>" in digest
    assert f"\n{'t' * DIGEST_CHARS}…[600 more chars]\n→ run (call c1)\n" in digest
    assert f"  code: {'a' * DIGEST_INLINE_CHARS}…[140 more chars]\n  n: 2\n  flag: on\n" in digest
    head = int(DIGEST_CHARS * 2 / 3)
    tail = DIGEST_CHARS - head
    assert f"[2] tool run (call c1)\n{'o' * head}{CUT_MARKER.format(dropped=600)}{'o' * tail}" in digest
    assert digest.endswith(f"Final answer:\n{'f' * 1000}"), "the final answer, the artifact an audit reads, is whole"
    assert "r" * (DIGEST_CHARS + 1) not in digest and "t" * (DIGEST_CHARS + 1) not in digest
    assert "<reasoning>" not in render_digest(sample, include_reasoning=False)
    unanswered = ScoringSample(prompt=PROMPT, completion=sample.completion)
    assert render_digest(unanswered, include_reasoning=False).endswith(f"Final answer:\n{NO_FINAL_ANSWER}")


def test_render_final_answer_with_and_without_a_final_answer():
    assert render_final_answer(ScoringSample(prompt=PROMPT, completion=TURNS, final_answer="1")) == "1"
    capped = ScoringSample(prompt=PROMPT, completion=TURNS)
    assert render_final_answer(capped) == f"{NO_FINAL_ANSWER}\nLast assistant turn:\nThe answer is 1."
    silent = ScoringSample(prompt=PROMPT, completion=[{"role": "tool", "content": "x"}])
    assert render_final_answer(silent) == NO_FINAL_ANSWER
    # An empty final answer is an answer: the episode delivered nothing, it did not end without one.
    assert render_final_answer(ScoringSample(prompt=PROMPT, completion=TURNS, final_answer="")) == ""


def test_view_text_dispatches_per_view_and_keeps_the_end_of_a_long_transcript():
    long_turns = [{"role": "assistant", "content": "x" * 5000}, {"role": "assistant", "content": "ANSWER=42"}]
    sample = ScoringSample(prompt=PROMPT, completion=long_turns, final_answer="ANSWER=42")
    full = view_text(sample, View.FULL, include_reasoning=True, max_chars=300)
    assert full.startswith("[1] assistant\n" + "x" * 150) and full.endswith("[2] assistant\nANSWER=42")
    dropped = len(render_transcript(long_turns, include_reasoning=True)) - 300
    assert CUT_MARKER.format(dropped=dropped) in full
    assert view_text(sample, View.FINAL, include_reasoning=True, max_chars=300) == "ANSWER=42"
    rendered = ScoringSample(prompt=PROMPT, completion=TURNS, final_answer="1")
    digest = view_text(rendered, View.DIGEST, include_reasoning=False, max_chars=10_000)
    assert digest == render_digest(rendered, include_reasoning=False)
    with pytest.raises(ValueError, match="unknown view"):
        view_text(sample, "tail", include_reasoning=True, max_chars=300)


def test_scored_messages_drops_sample_only_keys_and_uses_the_final_answer():
    sample = ScoringSample(
        prompt=PROMPT, completion=[{**TURNS[0], "truncated": True}, *TURNS[1:]], final_answer="FINAL"
    )
    full = scored_messages(sample, View.FULL)
    assert all(set(message) <= set(WIRE_KEYS) for message in full)
    assert full == [
        *PROMPT,
        {"role": "assistant", "content": "Running.", "tool_calls": [CALL]},
        {"role": "tool", "name": "run", "tool_call_id": "call_1", "content": "1"},
        {"role": "assistant", "content": "The answer is 1."},
    ]
    assert scored_messages(sample, View.FINAL) == [*PROMPT, {"role": "assistant", "content": "FINAL"}]
    # No final answer: an empty answer, never the last turn's fragment; a content-less tool-call turn
    # still carries an (empty) content on the wire.
    unanswered = ScoringSample(prompt=PROMPT, completion=sample.completion)
    assert scored_messages(unanswered, View.FINAL) == [*PROMPT, {"role": "assistant", "content": ""}]
    call_only = ScoringSample(prompt=PROMPT, completion=[{"role": "assistant", "content": None, "tool_calls": [CALL]}])
    assert scored_messages(call_only, View.FULL)[-1] == {"role": "assistant", "content": "", "tool_calls": [CALL]}


def test_render_tools_lists_each_tool_with_its_parameters_and_description_head():
    tools = [
        {
            "type": "function",
            "function": {
                "name": "run",
                "description": "Run a program.",
                "parameters": {"type": "object", "properties": {"code": {"type": "string"}, "timeout": {}}},
            },
        },
        {"type": "function", "function": {"name": "noop"}},
        {"name": "flat", "description": "d" * (DIGEST_INLINE_CHARS + 5), "parameters": {"properties": {"q": {}}}},
    ]
    assert render_tools(tools) == (
        "- run(code, timeout): Run a program.\n- noop()\n" + f"- flat(q): {'d' * DIGEST_INLINE_CHARS}…[5 more chars]"
    )


def test_samples_from_completions_sets_the_final_answer_to_the_last_assistant_text():
    first, second = samples_from_completions(
        ["plain prompt", [{"role": "user", "content": "q"}]],
        [
            [
                {"role": "assistant", "content": "a"},
                {"role": "tool", "content": "t"},
                {"role": "assistant", "content": "b"},
            ],
            "plain answer",
        ],
        references=["r1", None],
    )
    assert first.prompt == [{"role": "user", "content": "plain prompt"}]
    assert first.final_answer == "b" and first.reference == "r1"
    assert second.prompt == [{"role": "user", "content": "q"}]
    assert second.completion == [{"role": "assistant", "content": "plain answer"}]
    assert second.final_answer == "plain answer"
    assert second.reference is None and second.tools is None
    with pytest.raises(ValueError):
        samples_from_completions(["a", "b"], ["x"])


def test_task_text_and_final_text_read_user_turns_and_content_parts():
    parts = [
        {"type": "text", "text": "Part A"},
        {"type": "image_url", "image_url": {"url": "x"}},
        {"type": "text", "text": "Part B"},
    ]
    assert task_text([*PROMPT, {"role": "user", "content": parts}]) == "Run it.\n\nPart A\nPart B"
    last = [{"role": "assistant", "content": [{"type": "text", "text": "done"}]}, {"role": "tool", "content": "x"}]
    assert final_assistant_text(last) == "done"
    assert final_assistant_text([{"role": "user", "content": "q"}]) == ""


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
