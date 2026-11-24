#!/usr/bin/env python
"""A VLM row keeps every key its messages carry (tool calls, reasoning, a tool reply's call id and name) through the
row map's pinned schema into the render, so a tool-use conversation renders on the VLM path as the text path renders
it, the image placeholder aside."""

import copy
import json

import numpy as np
import pytest
from datasets import Dataset
from PIL import Image

from halo.data.collators.vlm import VLMDataCollator
from halo.data.pipeline.text_rows import render_row_conversation
from halo.data.pipeline.vlm_rows import _VLM_SIGNATURE_COLUMNS, build_vlm_row_processor, vlm_map_features
from halo.data.vlm_render import extract_conversation_images, render_vlm_text, vlm_row_template_kwargs
from tests.common.models import GLM5_3_FLASH, PINNED_REVISIONS, QWEN3_VL_2B
from tests.common.tokenizers import load_cached_processor

# What each processor's template writes for one image, where the text path writes nothing.
IMAGE_PLACEHOLDERS = {
    QWEN3_VL_2B: "<|vision_start|><|image_pad|><|vision_end|>",
    GLM5_3_FLASH: "<|begin_of_image|><|image|><|end_of_image|>",
}
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "read",
            "description": "Read a file.",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}, "ratio": {"type": "number"}},
                "required": ["path"],
            },
        },
    }
]
# A tool reply holding JSON, which the mapped schema must not read back as its value.
REPLY = '{"text": "hello"}'
# A float the datasets Json feature would round to ten digits, and a row ending on a call.
CALLS = (
    {
        "id": "call_0",
        "type": "function",
        "function": {"name": "read", "arguments": {"path": "a", "ratio": 0.1234567890123}},
    },
    {"id": "call_1", "type": "function", "function": {"name": "read", "arguments": {"path": "b", "ratio": 2.5}}},
)


@pytest.fixture(autouse=True)
def _isolated_datasets_cache(tmp_path, monkeypatch):
    """Fresh coordinated-op cache per test: a stale cache hit would skip the map under test."""
    monkeypatch.setenv("HF_DATASETS_CACHE", str(tmp_path / "hf_datasets"))


def _conversation(*, image: bool, wire_form: bool) -> list[dict]:
    calls = copy.deepcopy(CALLS)
    if wire_form:
        calls[0]["function"]["arguments"] = json.dumps(calls[0]["function"]["arguments"])
    first = [{"type": "image"}] if image else []
    return [
        # A null key stands for Arrow's padding of one a sibling message carries.
        {"role": "user", "content": [*first, {"type": "text", "text": "What does a say?"}], "tool_calls": None},
        {"role": "assistant", "content": "", "reasoning_content": "Read it first.", "tool_calls": [calls[0]]},
        {"role": "tool", "tool_call_id": "call_0", "name": "read", "content": REPLY},
        {"role": "assistant", "content": "It says hello.", "reasoning_content": "Done."},
        {"role": "user", "content": "Read b too."},
        {"role": "assistant", "content": "", "tool_calls": [calls[1]]},
    ]


def _mapped_row(conversation: list[dict]) -> dict:
    """The row as the runtime VLM map writes it, through the pinned schema."""
    image = Image.fromarray(np.random.RandomState(0).randint(0, 255, (64, 80, 3), dtype=np.uint8))
    dataset = Dataset.from_list(
        [{"conversation": json.dumps(conversation), "tools": json.dumps(TOOLS), "images": [image]}]
    )
    mapped = dataset.map(
        build_vlm_row_processor(conversation_field="conversation", tools_field="tools", images_field="images"),
        remove_columns=sorted(set(dataset.column_names) - set(_VLM_SIGNATURE_COLUMNS)),
        features=vlm_map_features(),
        load_from_cache_file=False,
    )
    return mapped[0]


class _RecordingProcessor:
    def __init__(self):
        self.conversations: list[list[dict]] = []

    def apply_chat_template(self, conversation, **_kwargs):
        self.conversations.append(conversation)
        return ""


def test_every_message_key_reaches_the_template():
    """As the row holds them, text in parts form: a null key is no field, and an empty or null content stays."""
    processor = _RecordingProcessor()
    conversation = _conversation(image=True, wire_form=False)
    conversation[-1]["content"] = None
    row = _mapped_row(conversation)

    render_vlm_text(processor, row["history"], **vlm_row_template_kwargs(row))

    calls = copy.deepcopy(CALLS)
    assert processor.conversations == [
        [
            {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "What does a say?"}]},
            {"role": "assistant", "content": "", "reasoning_content": "Read it first.", "tool_calls": [calls[0]]},
            {"role": "tool", "content": [{"type": "text", "text": REPLY}], "tool_call_id": "call_0", "name": "read"},
            {
                "role": "assistant",
                "content": [{"type": "text", "text": "It says hello."}],
                "reasoning_content": "Done.",
            },
            {"role": "user", "content": [{"type": "text", "text": "Read b too."}]},
            {"role": "assistant", "content": None, "tool_calls": [calls[1]]},
        ]
    ]


def test_a_key_json_cannot_write_is_refused_by_name():
    """A value the row map cannot carry raises, naming the message's keys, rather than leaving its key out."""
    with pytest.raises(ValueError, match=r"\['blob', 'name'\] beside its role and content.*bytes"):
        extract_conversation_images([{"role": "tool", "content": "r", "name": "read", "blob": b"raw"}])


@pytest.mark.parametrize("model_name", sorted(IMAGE_PLACEHOLDERS))
@pytest.mark.parametrize("wire_form", [False, True], ids=["mapping", "wire-form"])
def test_a_tool_call_row_renders_as_on_the_text_path(model_name, wire_form):
    """Qwen3-VL writes a call's text arguments as stored and GLM-5.3-Flash one block per argument, so the wire form
    takes both of the text path's branches; GLM renders the reasoning too."""
    processor = load_cached_processor(model_name, revision=PINNED_REVISIONS[model_name])
    row = _mapped_row(_conversation(image=True, wire_form=wire_form))
    text_row = {"conversation": json.dumps(_conversation(image=False, wire_form=wire_form)), "tools": TOOLS}

    rendered = render_vlm_text(processor, row["history"], **vlm_row_template_kwargs(row))
    expected = render_row_conversation(
        text_row, processor.tokenizer, conversation_field="conversation", tools_field="tools"
    )

    placeholder = IMAGE_PLACEHOLDERS[model_name]
    assert rendered.count(placeholder) == 1
    assert rendered.replace(placeholder, "") == expected
    collator = VLMDataCollator(processor, processor.tokenizer, 8192, train_on_completions_only=False)
    last_exchange = expected[expected.rindex("Read b too.") :]
    assert last_exchange in processor.tokenizer.decode(collator([row])["input_ids"][0])


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
