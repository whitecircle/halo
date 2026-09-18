#!/usr/bin/env python
"""CPU tests for ``jinja-templates/gemma4/gemma4-reasoning-effort.jinja``: the Gemma 4 rollout template that states the
episode's reasoning effort and thinking budget in the system turn, keeps thinking on without a template variable,
and otherwise renders the hub template's thinking-enabled form token-for-token (tool declarations, tool calls,
tool responses, carried reasoning).

Run: python tests/cpu/config/test_gemma4_effort_template.py  (or pytest)
"""

from pathlib import Path

import pytest
from huggingface_hub import try_to_load_from_cache
from transformers.utils.chat_template_utils import _compile_jinja_template

REPO_ROOT = Path(__file__).resolve().parents[3]
TEMPLATE_PATH = REPO_ROOT / "jinja-templates" / "gemma4" / "gemma4-reasoning-effort.jinja"
STOCK_TEMPLATE_REPO = "google/gemma-4-26B-A4B-it"

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "run_code",
            "description": "Run a program.",
            "parameters": {
                "type": "object",
                "properties": {"code": {"type": "string"}, "language": {"type": "string"}},
                "required": ["code", "language"],
            },
        },
    }
]

CONVERSATION = [
    {"role": "system", "content": "You are an expert competitive programmer."},
    {"role": "user", "content": "Print 1."},
    {
        "role": "assistant",
        "content": "",
        "reasoning_content": "A one-liner will do.",
        "tool_calls": [
            {"id": "c0", "function": {"name": "run_code", "arguments": {"code": "print(1)", "language": "python"}}}
        ],
    },
    {"role": "tool", "tool_call_id": "c0", "content": "1"},
    {"role": "assistant", "content": "Done.", "reasoning_content": "Output matches."},
    {"role": "user", "content": "Now print 2."},
]

EFFORT_LINE = (
    "Reasoning effort: high. Think for at most 16384 tokens per turn; reasoning past that budget is cut off, "
    "so finish the turn within it."
)


def _render(template_text: str, messages, **kwargs) -> str:
    return _compile_jinja_template(template_text).render(messages=messages, bos_token="<bos>", **kwargs)


@pytest.fixture(scope="module")
def template() -> str:
    return TEMPLATE_PATH.read_text()


def _stock_template() -> str | None:
    """The hub template at the revision ``from_pretrained`` loads, from the local cache only.

    A glob over the snapshots would pick whichever revision sorts first (a stale one left in the cache,
    or another checkpoint's) and compare against a template this one was never cut from.
    """
    path = try_to_load_from_cache(STOCK_TEMPLATE_REPO, "chat_template.jinja")
    return Path(path).read_text() if isinstance(path, str) else None


def test_effort_line_sits_after_the_system_text_before_the_tools(template):
    text = _render(
        template,
        CONVERSATION,
        tools=TOOLS,
        add_generation_prompt=True,
        reasoning_effort="high",
        reasoning_budget=16384,
    )
    system_turn = text.split("<turn|>\n", 1)[0]
    assert system_turn.startswith(
        "<bos><|turn>system\n<|think|>\nYou are an expert competitive programmer.\n\n" + EFFORT_LINE
    )
    assert system_turn.index(EFFORT_LINE) < system_turn.index("<|tool>declaration:run_code")
    assert text.count("Reasoning effort") == 1
    assert text.endswith("<|turn>model\n")


def test_thinking_is_on_without_a_template_variable(template):
    text = _render(template, [{"role": "user", "content": "q"}], add_generation_prompt=True)
    assert text == "<bos><|turn>system\n<|think|>\n<turn|>\n<|turn>user\nq<turn|>\n<|turn>model\n"


def test_level_without_a_budget_and_no_effort(template):
    only_level = _render(template, CONVERSATION[:2], add_generation_prompt=False, reasoning_effort="low")
    assert "You are an expert competitive programmer.\n\nReasoning effort: low.<turn|>" in only_level
    assert "tokens per turn" not in only_level
    assert "Reasoning effort" not in _render(template, CONVERSATION[:2], tools=TOOLS)


def test_assistant_turns_render_reasoning_tool_calls_and_tool_responses(template):
    text = _render(template, CONVERSATION, tools=TOOLS, add_generation_prompt=False)
    assert (
        "<|turn>model\n<|channel>thought\nA one-liner will do.\n<channel|>"
        '<|tool_call>call:run_code{code:<|"|>print(1)<|"|>,language:<|"|>python<|"|>}<tool_call|>'
        '<|tool_response>response:run_code{value:<|"|>1<|"|>}<tool_response|>'
    ) in text
    # The turn after a tool response continues the same model turn, as in the hub template.
    assert (
        "<tool_response|><|channel>thought\nOutput matches.\n<channel|>Done.<turn|>\n<|turn>user\nNow print 2.<turn|>\n"
        in text
    )


def test_media_items_are_not_rendered(template):
    text = _render(
        template, [{"role": "user", "content": [{"type": "text", "text": "hi"}, {"type": "image", "image": "x"}]}]
    )
    assert text.endswith("<|turn>user\nhi<turn|>\n")
    assert "<|image|>" not in text


def _as_content_parts(conversation):
    """The OpenAI content-parts form a vLLM server renders for this template; the trainer renders strings."""
    return [
        {**message, "content": [{"type": "text", "text": message["content"]}]}
        if isinstance(message.get("content"), str)
        else message
        for message in conversation
    ]


def test_string_and_content_parts_render_the_same_prompt(template):
    """vLLM wraps every message into content parts before rendering and the trainer renders strings; a
    system turn that renders differently between the two scores a prompt the policy never saw."""
    kwargs = {"tools": TOOLS, "add_generation_prompt": True, "reasoning_effort": "high", "reasoning_budget": 16384}
    assert _render(template, _as_content_parts(CONVERSATION), **kwargs) == _render(template, CONVERSATION, **kwargs)


def test_matches_the_hub_template_with_thinking_on_apart_from_the_effort_line(template):
    stock = _stock_template()
    if stock is None:
        pytest.skip("hub Gemma 4 template not in the local HF cache")
    # Up to the last user message: the hub template renders a turn's reasoning only after that message
    # (or on a tool-call turn under preserve_thinking); this template renders whatever a turn carries,
    # which the render test above pins for the turn before a later user message.
    conversation = CONVERSATION[:-1]
    kwargs = {"tools": TOOLS, "add_generation_prompt": True}
    theirs = _render(stock, conversation, enable_thinking=True, preserve_thinking=True, **kwargs)
    assert _render(template, conversation, **kwargs) == theirs
    ours = _render(template, conversation, reasoning_effort="high", reasoning_budget=16384, **kwargs)
    assert ours.replace("\n\n" + EFFORT_LINE, "") == theirs


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
