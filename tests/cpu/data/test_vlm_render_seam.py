#!/usr/bin/env python
"""Every VLM render goes through ``render_vlm_text``: ``add_generation_prompt=False``, and a ``tools`` branch where
``is None`` and falsiness differ. A re-spelled call opts out of both, so one conversation can tokenize
differently in a preference pair than in SFT. Pins the kwargs, which a stub can see."""

import pytest
from PIL import Image

from halo.data.pipeline.preferences import render_vlm_preference_row, tokenize_vlm_preference_row
from halo.data.vlm_render import render_vlm_text

IMAGE_TOKEN = "<image>"


class _RecordingProcessor:
    """Records every ``apply_chat_template`` call's kwargs; renders a stable, prefix-safe string."""

    def __init__(self):
        self.calls: list[dict] = []

    def apply_chat_template(self, messages, **kwargs):
        self.calls.append(kwargs)
        rendered = []
        for message in messages:
            content = message["content"]
            if isinstance(content, list):
                content = " ".join(
                    IMAGE_TOKEN if part.get("type") == "image" else part.get("text", "") for part in content
                )
            rendered.append(f"[{message['role']}] {content} [end]\n")
        return "".join(rendered)


class _RecordingTokenizer:
    """The surface SMPO's tokenize step reads off the resolved tokenizer."""

    bos_token_id = None
    eos_token_id = 2
    pad_token_id = 0

    def __call__(self, text, add_special_tokens=True, **_kwargs):
        return {"input_ids": [len(text)]}

    def decode(self, token_ids, **_kwargs):
        return " ".join(f"t{token_id}" for token_id in token_ids)


def _row():
    return {
        "prompt": [{"role": "user", "content": "Describe the picture"}],
        "chosen": [{"role": "assistant", "content": "A red square"}],
        "rejected": [{"role": "assistant", "content": "A blue circle"}],
        "images": [Image.new("RGB", (7, 5))],
    }


def _seam_contract(processor):
    """The kwargs the seam passes for a conversation carrying no tools."""
    processor.calls.clear()
    render_vlm_text(processor, [{"role": "user", "content": "x"}])
    return processor.calls[0]


def test_the_preference_map_renders_through_the_seam():
    """The reward head scores a complete conversation, so a generation prompt on one side is a silent difference."""
    processor = _RecordingProcessor()
    contract = _seam_contract(processor)

    processor.calls.clear()
    render_vlm_preference_row(_row(), processor)

    assert processor.calls, "the preference map must render through the shared seam"
    for call in processor.calls:
        assert call == contract, f"preference render used {call}, the seam passes {contract}"
    assert contract["add_generation_prompt"] is False
    assert "tools" not in contract, "a toolless row must pass NO tools key (templates test `is defined`)"


def test_the_smpo_vlm_row_renders_through_the_seam(monkeypatch):
    """Otherwise the prefix-strip invariant compares two dialects of one conversation."""
    processor = _RecordingProcessor()
    contract = _seam_contract(processor)
    monkeypatch.setattr("halo.data.pipeline.preferences.resolve_tokenizer", lambda _p: _RecordingTokenizer())

    processor.calls.clear()
    tokenize_vlm_preference_row(
        _row(),
        processor,
        max_prompt_length=None,
        max_completion_length=None,
        truncation_mode="keep_end",
    )

    assert len(processor.calls) == 3, "prompt + chosen + rejected renders"
    for call in processor.calls:
        assert call == contract, f"SMPO VLM render used {call}, the seam passes {contract}"


def test_the_seam_declares_tools_only_when_the_row_has_them():
    """An explicit ``tools=[]`` still declares the key, so the row renders under the template's tool branch."""
    processor = _RecordingProcessor()

    render_vlm_text(processor, [{"role": "user", "content": "x"}], tools=[])
    render_vlm_text(processor, [{"role": "user", "content": "x"}], tools=None)

    assert processor.calls[0]["tools"] == []
    assert "tools" not in processor.calls[1]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
