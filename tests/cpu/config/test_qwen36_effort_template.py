#!/usr/bin/env python
"""``jinja-templates/qwen3/qwen3.6-reasoning-effort.jinja`` states the episode's reasoning effort level and what it
asks for in the system block, never a token count, and otherwise renders the hub template's ``preserve_thinking=true`` form token-for-token."""

import re
from pathlib import Path

import pytest
from huggingface_hub import try_to_load_from_cache
from transformers.utils.chat_template_utils import _compile_jinja_template

from tests.common.effort_templates import LEVEL_LINES
from tests.common.models import QWEN3_6_MOE_35B
from tests.common.repo import REPO_ROOT
from tests.common.tokenizers import skip_uncached

TEMPLATE_PATH = REPO_ROOT / "jinja-templates" / "qwen3" / "qwen3.6-reasoning-effort.jinja"
STOCK_TEMPLATE_REPO = QWEN3_6_MOE_35B

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
        "tool_calls": [{"function": {"name": "run_code", "arguments": {"code": "print(1)", "language": "python"}}}],
    },
    {"role": "tool", "content": "1"},
    {"role": "assistant", "content": "Done.", "reasoning_content": "Output matches."},
    {"role": "user", "content": "Now print 2."},
]


def _render(template_text: str, messages, **kwargs) -> str:
    return _compile_jinja_template(template_text).render(messages=messages, **kwargs)


@pytest.fixture(scope="module")
def template() -> str:
    return TEMPLATE_PATH.read_text()


def _stock_template() -> str | None:
    """The hub template at the revision ``from_pretrained`` loads, from the local cache only. A glob over the
    snapshots could compare against a template this one was never cut from."""
    path = try_to_load_from_cache(STOCK_TEMPLATE_REPO, "chat_template.jinja")
    return Path(path).read_text() if isinstance(path, str) else None


@pytest.mark.parametrize("level", sorted(LEVEL_LINES))
def test_each_level_is_stated_once_in_the_system_block(template, level):
    text = _render(template, CONVERSATION, tools=TOOLS, add_generation_prompt=True, reasoning_effort=level)
    system_block, rest = text.split("<|im_end|>\n", 1)
    assert system_block.startswith("<|im_start|>system\n# Tools\n\nYou have access to the following functions:")
    assert system_block.endswith("You are an expert competitive programmer.\n\n" + LEVEL_LINES[level])
    assert "Reasoning effort" not in rest, "the effort line belongs to the system block only"
    assert text.endswith("<|im_start|>assistant\n<think>\n")


@pytest.mark.parametrize("level", sorted(LEVEL_LINES))
def test_the_line_states_no_token_count_whatever_the_caller_passes(template, level):
    """The model is told its level, never a count: the line carries no digit, and a stale budget variable a
    caller still passes changes nothing in the render."""
    text = _render(template, CONVERSATION[:2], add_generation_prompt=True, reasoning_effort=level)
    effort_line = next(line for line in text.splitlines() if line.startswith("Reasoning effort"))
    assert not re.search(r"\d", effort_line), effort_line
    assert "tokens" not in text
    stale = _render(
        template,
        CONVERSATION[:2],
        add_generation_prompt=True,
        reasoning_effort=level,
        reasoning_budget=16384,
        reasoning_budget_soft=True,
    )
    assert stale == text


def test_an_unknown_level_states_the_level_alone(template):
    text = _render(template, CONVERSATION[:2], add_generation_prompt=True, reasoning_effort="extreme")
    assert text.startswith(
        "<|im_start|>system\nYou are an expert competitive programmer.\n\nReasoning effort: extreme.<|im_end|>\n"
    )


def test_no_effort_renders_no_effort_line(template):
    text = _render(template, CONVERSATION[:2], tools=TOOLS, add_generation_prompt=True)
    assert "Reasoning effort" not in text
    assert text.count("<|im_start|>system\n") == 1


def test_effort_without_a_system_message_opens_a_system_block(template):
    text = _render(template, CONVERSATION[1:2], add_generation_prompt=True, reasoning_effort="medium")
    assert text.startswith("<|im_start|>system\n" + LEVEL_LINES["medium"] + "<|im_end|>\n<|im_start|>user\nPrint 1.")
    assert text.count("<|im_start|>system\n") == 1


def test_assistant_turns_render_reasoning_tool_calls_and_tool_responses(template):
    text = _render(template, CONVERSATION, tools=TOOLS, add_generation_prompt=False)
    assert (
        "<|im_start|>assistant\n<think>\nA one-liner will do.\n</think>\n\n<tool_call>\n<function=run_code>\n"
        "<parameter=code>\nprint(1)\n</parameter>\n<parameter=language>\npython\n</parameter>\n</function>\n</tool_call><|im_end|>\n"
        "<|im_start|>user\n<tool_response>\n1\n</tool_response><|im_end|>\n"
        "<|im_start|>assistant\n<think>\nOutput matches.\n</think>\n\nDone.<|im_end|>\n"
        "<|im_start|>user\nNow print 2.<|im_end|>\n"
    ) in text
    # A turn that carries no reasoning still renders the (empty) think block the hub's preserved form emits.
    plain = _render(template, [{"role": "user", "content": "q"}, {"role": "assistant", "content": "a"}])
    assert plain.endswith("<|im_start|>assistant\n<think>\n\n</think>\n\na<|im_end|>\n")


def test_text_item_lists_render_and_other_items_are_refused(template):
    text = _render(template, [{"role": "user", "content": [{"type": "text", "text": "hi"}]}])
    assert text == "<|im_start|>user\nhi<|im_end|>\n"
    with pytest.raises(Exception, match="Unexpected item type"):
        _render(template, [{"role": "user", "content": [{"type": "image", "image": "x"}]}])


def test_matches_the_hub_template_apart_from_the_effort_line(template):
    stock = _stock_template()
    if stock is None:
        skip_uncached(STOCK_TEMPLATE_REPO, "chat template of")
    ours = _render(template, CONVERSATION, tools=TOOLS, add_generation_prompt=True, reasoning_effort="high")
    theirs = _render(stock, CONVERSATION, tools=TOOLS, add_generation_prompt=True, preserve_thinking=True)
    assert ours.replace("\n\n" + LEVEL_LINES["high"], "") == theirs
    assert _render(template, CONVERSATION, tools=TOOLS, add_generation_prompt=True) == theirs


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
