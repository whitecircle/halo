#!/usr/bin/env python
"""A wrong tool parser leaves calls as text, each turn reads as a final answer and every episode ties with no error. The
watch reads the replies, since a vLLM server reports no parser identity to compare."""

import types

import pytest
from jinja2 import Template

from halo.configs.grpo_environmental import AsyncRolloutConfig
from halo.trainers.grpo.environmental import tool_call_watch
from halo.trainers.grpo.environmental.tool_call_watch import (
    ToolCallParserWatch,
    build_tool_call_watch,
    rendered_tool_call_marker,
)
from tests.common.tokenizers import load_cached_tokenizer

_MARKER = "<tool_call>"


def _episode(*turns, truncated=False):
    messages = [
        types.SimpleNamespace(
            role="assistant", content=content, tool_calls=calls, cut_tool_calls=None, truncated=truncated
        )
        for content, calls in turns
    ]
    return types.SimpleNamespace(trajectory=types.SimpleNamespace(messages=messages))


@pytest.fixture(autouse=True)
def _single_process(monkeypatch):
    monkeypatch.setattr(tool_call_watch, "world_sums", lambda values: values)


def test_two_rounds_of_unparsed_calls_raise_naming_the_parser():
    watch = ToolCallParserWatch(_MARKER)
    round_ = [_episode(("<tool_call>draw()</tool_call>", None))]
    watch.observe(round_)
    with pytest.raises(RuntimeError, match="VLLM_TOOL_PARSER"):
        watch.observe(round_)


def test_one_malformed_call_in_a_policy_that_rarely_calls_is_no_verdict():
    """Any parser leaves a malformed call as text, so a round of it and a silent round say nothing about the parser: a
    wrong one leaves every round that tries a call."""
    watch = ToolCallParserWatch(_MARKER)
    watch.observe([_episode(("<tool_call>broken", None))])
    for _ in range(3):
        watch.observe([_episode(("the answer is 4", None))])
    assert watch.armed
    with pytest.raises(RuntimeError, match="does not read this model's format"):
        watch.observe([_episode(("<tool_call>broken again", None))])


def test_one_parsed_call_disarms_for_the_run_so_a_later_malformed_call_never_trips():
    watch = ToolCallParserWatch(_MARKER)
    watch.observe([_episode(("<tool_call>broken", None), ("", [{"function": {"name": "draw"}}]))])
    assert not watch.armed
    # The trainer stops calling a disarmed watch; the state is what it reads.
    assert watch._parsed == 1


def test_a_cut_turns_call_counts_as_parsed():
    watch = ToolCallParserWatch(_MARKER)
    cut = types.SimpleNamespace(
        role="assistant", content="", tool_calls=None, cut_tool_calls=[{"name": "draw"}], truncated=True
    )
    watch.observe([types.SimpleNamespace(trajectory=types.SimpleNamespace(messages=[cut]))])
    assert not watch.armed


def test_a_call_the_engine_cut_mid_way_is_no_verdict():
    """A parser needs the whole call: a right one leaves a call cut at rollout_max_tokens as text with no salvage, so
    arguments longer than the turn limit would read as a wrong parser."""
    watch = ToolCallParserWatch(_MARKER)
    for _ in range(4):
        watch.observe([_episode(('<tool_call>{"name": "run", "arguments": {"code": "print(', None), truncated=True)])
    assert watch.armed


def test_the_verdict_takes_the_configured_round_count():
    watch = ToolCallParserWatch(_MARKER, rounds_before_verdict=3)
    round_ = [_episode(("<tool_call>draw()</tool_call>", None))]
    watch.observe(round_)
    watch.observe(round_)
    with pytest.raises(RuntimeError, match="tool_parser_watch_rounds"):
        watch.observe(round_)


def test_no_call_in_either_form_stays_armed_with_no_verdict():
    """Episodes that never try a call say nothing about the parser; an untrained policy may run this way for a while."""
    watch = ToolCallParserWatch(_MARKER)
    for _ in range(4):
        watch.observe([_episode(("the answer is 4", None)), types.SimpleNamespace(trajectory=None)])
    assert watch.armed


def test_the_verdict_is_taken_off_world_sums(monkeypatch):
    """A rank whose own episodes hold no call must raise with the rank whose episodes do."""
    monkeypatch.setattr(tool_call_watch, "world_sums", lambda values: [0, 5])
    watch = ToolCallParserWatch(_MARKER)
    watch.observe([])
    with pytest.raises(RuntimeError, match="10 assistant turns"):
        watch.observe([])


# Added tokens and renders of the real families, from what each template writes for an assistant call.
_HERMES_TEMPLATE = (
    "{% for m in messages %}{% if m.tool_calls %}<|im_start|>assistant\n<tool_call>\n"
    '{"name": "{{ m.tool_calls[0].function.name }}"}\n</tool_call>{% else %}<|im_start|>{{ m.role }}\n'
    "{{ m.content }}<|im_end|>\n{% endif %}{% endfor %}{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"
)
_TEXT_MARKER_TEMPLATE = (
    "{% for m in messages %}{% if m.tool_calls %}<|start|>assistant to=functions.{{ m.tool_calls[0].function.name }}"
    "<|call|>{% else %}<|start|>{{ m.role }}<|message|>{{ m.content }}<|end|>{% endif %}{% endfor %}"
    "{% if add_generation_prompt %}<|start|>assistant{% endif %}"
)
_STRADDLING_TEMPLATE = (
    "{% for m in messages %}{% if m.tool_calls %}<|turn>model\n<|tool_call>call:{{ m.tool_calls[0].function.name }}"
    "<tool_call|>{% else %}<|turn>{{ m.role }}\n{{ m.content }}<turn|>\n{% endif %}{% endfor %}"
    "{% if add_generation_prompt %}<|turn>model\n<|channel>{% endif %}"
)


@pytest.mark.parametrize(
    ("template", "added", "expected"),
    [
        (_HERMES_TEMPLATE, ["<|im_start|>", "<|im_end|>", "<tool_call>"], '<tool_call>\n{"name": "'),
        (_TEXT_MARKER_TEMPLATE, ["<|start|>", "<|end|>", "<|message|>", "<|call|>"], " to=functions."),
        (
            _STRADDLING_TEMPLATE,
            ["<|turn>", "<turn|>", "<|tool_call>", "<tool_call|>", "<|channel>"],
            "<|tool_call>call:",
        ),
    ],
    ids=[
        "hermes-opens-on-an-added-token",
        "gpt-oss-opens-on-plain-text",
        "gemma4-shares-the-prompts-first-characters",
    ],
)
def test_the_marker_reads_how_the_template_opens_a_call(template, added, expected):
    def render(messages, *, tools, tokenize, add_generation_prompt):
        return Template(template).render(messages=messages, add_generation_prompt=add_generation_prompt)

    tokenizer = types.SimpleNamespace(
        apply_chat_template=render, get_added_vocab=lambda: {token: i for i, token in enumerate(added)}
    )
    assert rendered_tool_call_marker(tokenizer, {}) == expected


@pytest.mark.parametrize(
    ("model_id", "expected"),
    [
        ("Qwen/Qwen3-30B-A3B-Instruct-2507", '<tool_call>\n{"name": "'),
        ("Qwen/Qwen3.6-35B-A3B", "<tool_call>\n<function="),
        ("unsloth/gpt-oss-20b-BF16", " to=functions."),
        ("zai-org/GLM-4.7-Flash", "<tool_call>"),
        ("LiquidAI/LFM2-24B-A2B", "<|tool_call_start|>["),
        ("google/gemma-4-26B-A4B-it", "<|tool_call>call:"),
    ],
)
def test_the_marker_on_the_real_templates(model_id, expected):
    assert rendered_tool_call_marker(load_cached_tokenizer(model_id), {}) == expected


def test_a_template_that_cannot_render_a_call_stands_the_watch_down():
    def refuse(*args, **kwargs):
        raise ValueError("this template raises on a tool call")

    tokenizer = types.SimpleNamespace(apply_chat_template=refuse, get_added_vocab=dict)
    assert rendered_tool_call_marker(tokenizer, {}) is None


@pytest.mark.parametrize(
    "template",
    [
        "{% for m in messages %}<start>{{ m.role }}\n{{ m.content }}<end>{% endfor %}"
        "{% if add_generation_prompt %}<start>model\n{% endif %}",
        "{% for t in tools %}{{ t.function.name }}{% endfor %}{% for m in messages %}<start>{{ m.role }}\n{{ m.content }}"
        "<end>{% endfor %}{% if add_generation_prompt %}<start>model\n{% endif %}",
    ],
    ids=["no-tools-block", "names-the-tool-only-in-its-tools-block"],
)
def test_a_template_that_drops_the_call_stands_the_watch_down(template):
    """Gemma 3, Qwen2.5-VL and a template with no tool support render no call: the turn's close is no call opening."""

    def render(messages, *, tools, tokenize, add_generation_prompt):
        return Template(template).render(messages=messages, tools=tools, add_generation_prompt=add_generation_prompt)

    tokenizer = types.SimpleNamespace(apply_chat_template=render, get_added_vocab=lambda: {"<start>": 0, "<end>": 1})
    assert rendered_tool_call_marker(tokenizer, {}) is None


def test_an_environment_with_no_tools_gets_no_watch():
    env = types.SimpleNamespace(get_tools_schema=lambda: None, TOOLS_ARRIVE_ON_RESET=False)
    assert build_tool_call_watch(env, None, {}) is None


def test_a_null_round_count_turns_the_watch_off(monkeypatch):
    monkeypatch.setattr(tool_call_watch, "rendered_tool_call_marker", lambda tokenizer, kwargs: _MARKER)
    env = types.SimpleNamespace(get_tools_schema=lambda: [{"name": "x"}], TOOLS_ARRIVE_ON_RESET=False)
    assert build_tool_call_watch(env, None, {}, None) is None
    assert build_tool_call_watch(env, None, {}, 1) is not None


@pytest.mark.parametrize("rounds", [0, -1])
def test_a_round_count_that_could_never_hold_is_refused(rounds):
    with pytest.raises(ValueError, match="tool_parser_watch_rounds"):
        AsyncRolloutConfig(tool_parser_watch_rounds=rounds)


def test_a_tool_environment_gets_one_mcp_included(monkeypatch):
    monkeypatch.setattr(tool_call_watch, "rendered_tool_call_marker", lambda tokenizer, kwargs: _MARKER)
    for env in (
        types.SimpleNamespace(get_tools_schema=lambda: [{"name": "x"}], TOOLS_ARRIVE_ON_RESET=False),
        types.SimpleNamespace(get_tools_schema=lambda: None, TOOLS_ARRIVE_ON_RESET=True),
    ):
        assert build_tool_call_watch(env, None, {}) is not None


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
