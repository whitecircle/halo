#!/usr/bin/env python3
"""normalize_vlm_conversation: hub-shape adaptation for the VLM SFT pipeline.

Covers the two accepted conversation shapes (native role/content and the HuggingFaceM4
paired {user, assistant} ``texts`` shape shared by FineVision / the_cauldron / Docmatix),
top-level image injection into the first user turn (``images_field``), input immutability,
and the generation-branch slicing through ``create_vlm_processor``.

Usage:
    python tests/cpu/data/test_vlm_normalize_conversation.py
"""

import copy
import sys

import numpy as np
import pytest
from PIL import Image

from src.data.pipeline.row_processors import create_vlm_processor, normalize_vlm_conversation
from src.data.vlm import process_vlm_conversation


def _image(color: int = 128) -> Image.Image:
    return Image.fromarray(np.full((8, 8, 3), color, dtype=np.uint8))


def test_role_content_passthrough():
    conv = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}]
    out = normalize_vlm_conversation(conv)
    assert out == conv
    assert out[0] is not conv[0], "must return new dicts, not aliases into the row"


def test_m4_paired_turns_expand_in_order():
    conv = [{"user": "q1", "assistant": "a1", "source": "x"}, {"user": "q2", "assistant": "a2"}]
    out = normalize_vlm_conversation(conv)
    assert [m["role"] for m in out] == ["user", "assistant", "user", "assistant"]
    assert [m["content"] for m in out] == ["q1", "a1", "q2", "a2"]


def test_single_image_injected_before_first_user_text():
    img = _image()
    conv = [{"user": "transcribe this", "assistant": "text"}]
    out = normalize_vlm_conversation(conv, img)
    content = out[0]["content"]
    assert content[0] == {"type": "image", "image": img}
    assert content[1] == {"type": "text", "text": "transcribe this"}
    assert out[1] == {"role": "assistant", "content": "text"}


def test_image_list_injected_only_into_first_user_turn():
    imgs = [_image(1), _image(2)]
    conv = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "page 1?"},
        {"role": "assistant", "content": "yes"},
        {"role": "user", "content": "page 2?"},
    ]
    out = normalize_vlm_conversation(conv, imgs)
    assert [p["type"] for p in out[1]["content"]] == ["image", "image", "text"]
    assert out[3]["content"] == "page 2?", "later user turns must stay untouched"


def test_images_with_list_content_prepend():
    img = _image()
    conv = [{"role": "user", "content": [{"type": "text", "text": "look"}]}]
    out = normalize_vlm_conversation(conv, [img])
    assert [p["type"] for p in out[0]["content"]] == ["image", "text"]


def test_existing_placeholders_filled_not_doubled():
    """TRL vision convention: bare {"type": "image"} placeholders pair with the images column by
    order — filling must not add a second placeholder (double-injection starves the processor's
    replacement iterator, surfacing as a swallowed StopIteration = silent zero-step training)."""
    img = _image()
    conv = [
        {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "What digit?"}]},
        {"role": "assistant", "content": "3"},
    ]
    out = normalize_vlm_conversation(conv, [img])
    parts = out[0]["content"]
    assert [p["type"] for p in parts] == ["image", "text"], "no second placeholder may be injected"
    assert parts[0]["image"] is img, "the placeholder must be filled with the column image"
    assert conv[0]["content"][0] == {"type": "image"}, "the source row must stay unmutated"


def test_placeholder_count_mismatch_raises():
    conv = [{"role": "user", "content": [{"type": "image"}, {"type": "image"}, {"type": "text", "text": "q"}]}]
    with pytest.raises(ValueError, match="counts must match"):
        normalize_vlm_conversation(conv, [_image()])


def test_input_rows_never_mutated():
    img = _image()
    conv = [{"role": "user", "content": "q"}, {"role": "assistant", "content": "a"}]
    snapshot = copy.deepcopy(conv)
    normalize_vlm_conversation(conv, [img])
    assert conv == snapshot


def test_no_user_turn_with_images_raises():
    with pytest.raises(ValueError, match="no user turn"):
        normalize_vlm_conversation([{"role": "assistant", "content": "a"}], [_image()])


def test_unknown_turn_shape_raises():
    with pytest.raises(ValueError, match="Unrecognized conversation turn shape"):
        normalize_vlm_conversation([{"speaker": "user", "text": "hi"}])


def test_empty_image_list_is_noop():
    conv = [{"user": "q", "assistant": "a"}]
    out = normalize_vlm_conversation(conv, [])
    assert out[0]["content"] == "q"


def test_normalized_output_feeds_process_vlm_conversation():
    img = _image()
    conv = normalize_vlm_conversation([{"user": "read the page", "assistant": "the page text"}], [img])
    history, images = process_vlm_conversation(conv)
    assert images == [img]
    assert history[0]["content"][0] == {"type": "image"}, "processor placeholder must replace inline image data"
    # Uniform parts-form: string contents become text parts so the Arrow column schema is stable.
    assert history[1] == {"role": "assistant", "content": [{"type": "text", "text": "the page text"}]}


@pytest.mark.parametrize("payload", [b"", ""], ids=["empty-bytes", "empty-path"])
def test_a_falsy_but_present_image_payload_fails_loud(payload):
    """A truthiness test here dropped the payload while still emitting the placeholder, handing the
    processor a prompt with an image placeholder and no image — which surfaces later as a
    placeholder-count mismatch, if at all. A present payload must reach ``process_image``.
    """
    conv = [{"role": "user", "content": [{"type": "image", "image": payload}]}]
    with pytest.raises((OSError, ValueError)):
        process_vlm_conversation(conv)


def test_an_absent_image_payload_is_still_a_bare_placeholder():
    """The other side: an unfilled placeholder (``images_field`` pairing, or the ``image: None`` key
    Arrow adds to every text part of a mixed dataset) carries no payload and must pass through."""
    history, images = process_vlm_conversation([{"role": "user", "content": [{"type": "image"}]}])
    assert images == []
    assert history[0]["content"] == [{"type": "image"}]


def test_create_vlm_processor_images_field_and_generation_slice():
    img = _image()
    row = {"texts": [{"user": "describe", "assistant": "a cat"}], "images": [img]}

    train_fn = create_vlm_processor(conversation_field="texts", images_field="images")
    example = train_fn(dict(row))
    assert [m["role"] for m in example["history"]] == ["user", "assistant"]
    assert example["images"] == [img]

    generate_fn = create_vlm_processor(conversation_field="texts", images_field="images", add_generation_prompt=True)
    prompt_example = generate_fn(dict(row))
    assert [m["role"] for m in prompt_example["history"]] == ["user"], (
        "generation branch must drop only the final assistant message, not the whole user+assistant pair"
    )
    assert prompt_example["images"] == [img], "the image lives in the kept user turn"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
