#!/usr/bin/env python
"""CPU tests: ``jinja-templates/glm/glm5-chat.jinja`` renders a GLM-5 conversation as the hub template renders it.

The SFT template trains GLM-5 Next (GLM-5.3-Flash) on the model's own non-thinking turn, so a served model has to be
prompted with what it trained on. No GLM-5 snapshot sits in the HF cache, so the hub template is vendored byte for byte
from ``zai-org/GLM-5.3-Flash`` at revision ``46a050bec13002ad29ee795c16e61cf4fa3c4a9c``, with its MIT license
(``glm-5.3-flash-LICENSE.txt`` beside it). The SFT template departs from it in two places, by design:

* it leaves out the hub's default ``<|system|>Reasoning Effort: Max`` line, which budgets a reasoning block these
  turns do not have;
* it ends the final assistant turn with ``<|endoftext|>``, the token a trained turn stops on.

Every prefix of each conversation renders as the hub's does with those two differences, and the generation prompt
followed by the answer is the hub's rendering of the answered conversation. A tool definition, tool call or tool
result raises instead of vanishing from the rendered row.

Run: python tests/cpu/data/test_glm5_chat_template.py  (or pytest)
"""

import hashlib

import pytest
from jinja2.exceptions import TemplateError
from transformers.utils.chat_template_utils import _compile_jinja_template

from tests.common.repo import REPO_ROOT

TEMPLATE_PATH = REPO_ROOT / "jinja-templates" / "glm" / "glm5-chat.jinja"
HUB_TEMPLATE_PATH = REPO_ROOT / "tests" / "common" / "hub_templates" / "glm-5.3-flash.jinja"
HUB_TEMPLATE_SHA256 = "2614a3c896d53e265f564c23701173c0049f985dacfbae79fc4bcd9f5ba083f4"

PREAMBLE = "[gMASK]<sop>"
HUB_EFFORT_LINE = "<|system|>Reasoning Effort: Max"
FINAL_TURN_END = "<|endoftext|>"

CONVERSATIONS = {
    "single-turn": [
        {"role": "user", "content": "What is 2 + 2?"},
        {"role": "assistant", "content": "4."},
    ],
    "system-multi-turn": [
        {"role": "system", "content": "You are a concise assistant."},
        {"role": "user", "content": "Name a prime."},
        {"role": "assistant", "content": "Seven."},
        {"role": "user", "content": "And an even one?"},
        {"role": "assistant", "content": "Two is the only even prime."},
    ],
    # The hub strips each answer; user turns keep their whitespace.
    "padded-answers": [
        {"role": "user", "content": "  Spell cat.\n"},
        {"role": "assistant", "content": "\n  C-A-T. \n\n"},
        {"role": "user", "content": "Backwards?"},
        {"role": "assistant", "content": " T-A-C."},
    ],
}


# Tool data the template refuses: (the render's extra arguments, the messages), each beside a plain exchange.
TOOL_DATA = {
    "tool-definitions": (
        {"tools": [{"type": "function", "function": {"name": "add", "parameters": {"type": "object"}}}]},
        CONVERSATIONS["single-turn"],
    ),
    "tool-call": (
        {},
        [
            {"role": "user", "content": "What is 2 + 2?"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [{"type": "function", "function": {"name": "add", "arguments": {"a": 2, "b": 2}}}],
            },
        ],
    ),
    "tool-result": (
        {},
        [{"role": "user", "content": "What is 2 + 2?"}, {"role": "tool", "content": "4"}],
    ),
}


def _render(template: str, messages: list[dict], *, add_generation_prompt: bool = False, **kwargs) -> str:
    return _compile_jinja_template(template).render(
        messages=messages, add_generation_prompt=add_generation_prompt, **kwargs
    )


@pytest.fixture(scope="module")
def template() -> str:
    return TEMPLATE_PATH.read_text()


@pytest.fixture(scope="module")
def hub_template() -> str:
    text = HUB_TEMPLATE_PATH.read_text()
    assert hashlib.sha256(text.encode()).hexdigest() == HUB_TEMPLATE_SHA256, "the vendored hub template was edited"
    return text


def _without_effort_line(rendered: str) -> str:
    assert rendered.startswith(PREAMBLE + HUB_EFFORT_LINE), rendered[:80]
    return PREAMBLE + rendered[len(PREAMBLE + HUB_EFFORT_LINE) :]


@pytest.mark.parametrize("name", sorted(CONVERSATIONS))
def test_glm5_chat_template_renders_like_the_hub(template, hub_template, name):
    messages = CONVERSATIONS[name]
    for end in range(1, len(messages) + 1):
        prefix = messages[:end]
        expected = _without_effort_line(_render(hub_template, prefix))
        if prefix[-1]["role"] == "assistant":
            expected += FINAL_TURN_END
        assert _render(template, prefix) == expected, f"{name}, first {end} messages"


@pytest.mark.parametrize("name", sorted(CONVERSATIONS))
def test_the_generation_prompt_opens_the_turn_the_hub_renders(template, hub_template, name):
    """A model served with this template answers into the hub's non-thinking turn; served with the hub's own, it is
    prompted with the start of that turn."""
    messages = CONVERSATIONS[name]
    for end, answer in enumerate(messages):
        if answer["role"] != "assistant":
            continue
        prompt = _render(template, messages[:end], add_generation_prompt=True)
        answered = _without_effort_line(_render(hub_template, messages[: end + 1]))
        assert prompt + answer["content"].strip() == answered, f"{name}, answer {end}"
        hub_prompt = _without_effort_line(_render(hub_template, messages[:end], add_generation_prompt=True))
        assert prompt.startswith(hub_prompt)


@pytest.mark.parametrize("name", sorted(TOOL_DATA))
def test_tool_data_is_refused_rather_than_dropped(template, name):
    extra, messages = TOOL_DATA[name]
    with pytest.raises(TemplateError, match="renders system, user and assistant text turns only"):
        _render(template, messages, **extra)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
