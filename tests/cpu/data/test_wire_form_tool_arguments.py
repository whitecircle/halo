#!/usr/bin/env python3
"""A tool call stored in the OpenAI wire form (``arguments`` as JSON text) renders through the text render seam: parsed
into a mapping under a template that raises on the text or re-encodes it, and left as written under one that writes the
text."""

import json
import zlib

import pytest
from transformers.utils.chat_template_utils import _compile_jinja_template

from halo.data.chat_render import render_generation_prompt, resolve_template_conversation, supports_wire_form_arguments
from halo.data.conversation import has_wire_form_arguments, parse_tool_call_arguments
from halo.data.pipeline.text_rows import render_row_conversation
from tests.common.models import QWEN3_0_6B
from tests.common.repo import REPO_ROOT
from tests.common.tokenizers import load_cached_tokenizer

TEMPLATES = REPO_ROOT / "jinja-templates"
# Templates that render a call's arguments as a mapping and raise on its text.
MAPPING_ONLY = (
    "qwen3/qwen3.5-native.jinja",
    "qwen3/qwen3.6-reasoning-effort.jinja",
    "glm/glm-native.jinja",
    "gemma4/gemma4-reasoning-effort.jinja",
)
# Templates that write wire-form text as it is stored.
TEXT_WRITING = ("qwen3/qwen3-native.jinja", "gpt-oss/gpt-oss-harmony.jinja")
# One that drops a call from its render, and one that raises on any multi-turn conversation.
CALL_BLIND = ("qwen3/qwen3-multiturn.jinja", "qwen3/qwen3-instruct.jinja")
# What applying ``tojson`` to a call does to wire-form text: a quoted, escaped string in place of the object.
RE_ENCODING_TEMPLATE = (
    "{%- for message in messages %}<{{ message.role }}>{{ message.content or '' }}"
    "{%- for call in message.tool_calls or [] %}<call>{{ call.function | tojson }}</call>{%- endfor %}"
    "</{{ message.role }}>{%- endfor %}"
)

# One that writes a call's arguments as they are stored.
TEXT_WRITING_TEMPLATE = (
    "{%- for message in messages %}<{{ message.role }}>{{ message.content or '' }}"
    "{%- for call in message.tool_calls or [] %}<call>{{ call.function.name }} {{ call.function.arguments }}</call>"
    "{%- endfor %}</{{ message.role }}>{%- endfor %}"
)

ARGUMENTS = {"pattern": "a|b", "ignore_case": True, "context_lines": 2}
WIRE_ARGUMENTS = json.dumps(ARGUMENTS, separators=(",", ":"))
TOOLS = [{"type": "function", "function": {"name": "grep", "parameters": {"type": "object", "properties": {}}}}]


class TemplateTokenizer:
    """Renders one template, a bundled one by its path; a token per whitespace-separated word, an id per spelling, and
    no special tokens of its own."""

    bos_token = None
    bos_token_id = None

    def __init__(self, template: str):
        self.chat_template = (TEMPLATES / template).read_text() if template.endswith(".jinja") else template
        self.renders = 0

    def apply_chat_template(self, conversation, tokenize=False, add_generation_prompt=False, **kwargs):
        self.renders += 1
        return _compile_jinja_template(self.chat_template).render(
            messages=conversation, add_generation_prompt=add_generation_prompt, **kwargs
        )

    def __call__(self, text, add_special_tokens=True, truncation=False, padding=False, max_length=None):
        ids = [zlib.crc32(word.encode()) for word in text.split()]
        return {"input_ids": ids, "attention_mask": [1] * len(ids)}


def _conversation(arguments, *, flat: bool = False) -> list[dict]:
    function = {"name": "grep", "arguments": arguments}
    call = function if flat else {"id": "call_1", "type": "function", "function": function}
    return [
        {"role": "user", "content": "find it"},
        {"role": "assistant", "content": None, "tool_calls": [call]},
        {"role": "tool", "tool_call_id": "call_1", "content": "2 lines"},
        {"role": "assistant", "content": "found"},
    ]


def _row(arguments) -> dict:
    return {"messages": json.dumps(_conversation(arguments)), "tools": json.dumps(TOOLS)}


def _render_row(tokenizer, arguments) -> str:
    return render_row_conversation(_row(arguments), tokenizer, conversation_field="messages", tools_field="tools")


@pytest.mark.parametrize("flat", [False, True])
def test_parse_turns_wire_form_text_into_the_mapping_and_leaves_the_input(flat):
    conversation = _conversation(WIRE_ARGUMENTS, flat=flat)
    parsed = parse_tool_call_arguments(conversation)
    assert parsed == _conversation(ARGUMENTS, flat=flat)
    assert conversation == _conversation(WIRE_ARGUMENTS, flat=flat)


def test_parse_reads_empty_text_as_no_arguments_and_keeps_a_mapping():
    assert parse_tool_call_arguments(_conversation("")) == _conversation({})
    assert parse_tool_call_arguments(_conversation(" \n")) == _conversation({})
    assert parse_tool_call_arguments(_conversation(ARGUMENTS)) == _conversation(ARGUMENTS)


@pytest.mark.parametrize("text", ['{"pattern": "a', "[1, 2]", '"a|b"', "grep a|b"])
def test_parse_refuses_text_that_is_no_json_object(text):
    with pytest.raises(ValueError, match="Tool call 'grep' carries arguments that are not a JSON object"):
        parse_tool_call_arguments(_conversation(text))


def test_parse_reaches_every_call_of_every_turn():
    first, second = _conversation(WIRE_ARGUMENTS)[:3], _conversation(ARGUMENTS)[:3]
    second[1]["tool_calls"].append(
        {"id": "call_2", "type": "function", "function": {"name": "grep", "arguments": "{}"}}
    )
    twin = _conversation(ARGUMENTS)[:3] + _conversation(ARGUMENTS)[:3]
    twin[4]["tool_calls"].append({"id": "call_2", "type": "function", "function": {"name": "grep", "arguments": {}}})
    assert has_wire_form_arguments(first + second) and has_wire_form_arguments(second + first)
    assert parse_tool_call_arguments(first + second) == twin


def test_only_text_arguments_count_as_wire_form():
    assert has_wire_form_arguments(_conversation(WIRE_ARGUMENTS))
    assert has_wire_form_arguments(_conversation(WIRE_ARGUMENTS, flat=True))
    assert not has_wire_form_arguments(_conversation(ARGUMENTS))
    assert not has_wire_form_arguments([{"role": "user", "content": '{"arguments": "{}"}'}])
    assert not has_wire_form_arguments("a prompt")


@pytest.mark.parametrize("template", MAPPING_ONLY)
def test_a_template_that_iterates_arguments_does_not_take_wire_form(template):
    assert not supports_wire_form_arguments(TemplateTokenizer(template), with_tools=True)


@pytest.mark.parametrize("template", [*TEXT_WRITING, *CALL_BLIND])
def test_a_template_that_writes_the_text_or_no_call_at_all_takes_wire_form(template):
    assert supports_wire_form_arguments(TemplateTokenizer(template), with_tools=True)


def test_a_template_that_re_encodes_the_text_does_not_take_wire_form():
    tokenizer = TemplateTokenizer(RE_ENCODING_TEMPLATE)
    assert not supports_wire_form_arguments(tokenizer, with_tools=True)
    rendered = _render_row(tokenizer, WIRE_ARGUMENTS)
    assert rendered == _render_row(tokenizer, ARGUMENTS)
    assert '"arguments": {"pattern": "a|b"' in rendered


def test_the_verdict_is_probed_once_and_again_for_another_template():
    tokenizer = TemplateTokenizer(MAPPING_ONLY[0])
    assert not supports_wire_form_arguments(tokenizer, with_tools=True)
    probes = tokenizer.renders
    assert not supports_wire_form_arguments(tokenizer, with_tools=True)
    assert tokenizer.renders == probes
    tokenizer.chat_template = (TEMPLATES / TEXT_WRITING[0]).read_text()
    assert supports_wire_form_arguments(tokenizer, with_tools=True)


@pytest.mark.parametrize("template", MAPPING_ONLY)
def test_a_wire_form_row_renders_as_its_mapping_twin(template):
    tokenizer = TemplateTokenizer(template)
    rendered = _render_row(tokenizer, WIRE_ARGUMENTS)
    assert rendered == _render_row(tokenizer, ARGUMENTS)
    assert "a|b" in rendered and WIRE_ARGUMENTS not in rendered


def test_qwen36_writes_each_argument_of_a_wire_form_call_as_its_own_parameter():
    rendered = _render_row(TemplateTokenizer("qwen3/qwen3.6-reasoning-effort.jinja"), WIRE_ARGUMENTS)
    call = (
        "<tool_call>\n<function=grep>\n<parameter=pattern>\na|b\n</parameter>\n<parameter=ignore_case>\ntrue\n"
        "</parameter>\n<parameter=context_lines>\n2\n</parameter>\n</function>\n</tool_call>"
    )
    assert call in rendered


@pytest.mark.parametrize("template", TEXT_WRITING)
def test_a_template_that_writes_the_text_keeps_the_rows_own_spelling(template):
    tokenizer = TemplateTokenizer(template)
    assert WIRE_ARGUMENTS in _render_row(tokenizer, WIRE_ARGUMENTS)
    assert resolve_template_conversation(tokenizer, _conversation(WIRE_ARGUMENTS), with_tools=True) == _conversation(
        WIRE_ARGUMENTS
    )


def test_a_row_without_wire_form_arguments_costs_no_probe():
    tokenizer = TemplateTokenizer(MAPPING_ONLY[0])
    conversation = _conversation(ARGUMENTS)
    assert resolve_template_conversation(tokenizer, conversation, with_tools=True) is conversation
    assert tokenizer.renders == 0


def test_named_templates_are_probed_under_the_rows_own_tools_declaration():
    """A tokenizer holding named templates renders a row that declares tools with ``tool_use`` and any other with
    ``default``, and the two may take a call's arguments differently."""
    tokenizer = load_cached_tokenizer(QWEN3_0_6B)
    tokenizer.chat_template = {"default": RE_ENCODING_TEMPLATE, "tool_use": TEXT_WRITING_TEMPLATE}
    declaring, bare = _row(WIRE_ARGUMENTS), {"messages": json.dumps(_conversation(WIRE_ARGUMENTS))}
    # Alternating, so one row's verdict must not serve the other.
    for _ in range(2):
        assert WIRE_ARGUMENTS in render_row_conversation(
            declaring, tokenizer, conversation_field="messages", tools_field="tools"
        )
        rendered = render_row_conversation(bare, tokenizer, conversation_field="messages", tools_field="tools")
        assert '"arguments": {"pattern": "a|b"' in rendered


def test_malformed_wire_form_arguments_raise_under_a_mapping_only_template():
    with pytest.raises(ValueError, match="not a JSON object"):
        _render_row(TemplateTokenizer(MAPPING_ONLY[0]), '{"pattern": "a')


def test_generation_prompts_parse_wire_form_history():
    tokenizer = TemplateTokenizer(MAPPING_ONLY[0])
    history, twin = _conversation(WIRE_ARGUMENTS)[:3], _conversation(ARGUMENTS)[:3]
    prompt = render_generation_prompt(tokenizer, history, tools=TOOLS)
    assert prompt == render_generation_prompt(tokenizer, twin, tools=TOOLS)
    assert "<parameter=pattern>\na|b\n</parameter>" in prompt
    # The length check tokenizes the same parsed render.
    assert render_generation_prompt(tokenizer, history, tools=TOOLS, max_prompt_length=4) is None


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
