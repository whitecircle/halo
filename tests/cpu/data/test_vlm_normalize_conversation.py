#!/usr/bin/env python3
"""``normalize_vlm_conversation``: the native and HuggingFaceM4 paired conversation shapes, top-level image
injection into the first user turn, and input immutability."""

import base64
import copy

import numpy as np
import pytest
from PIL import Image

from halo.data.conversation import reject_image_content
from halo.data.pipeline.vlm_rows import build_vlm_row_processor, normalize_vlm_conversation
from halo.data.vlm_render import extract_conversation_images


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
    """Bare placeholders pair with the images column by order; a second placeholder starves the processor's
    replacement iterator, surfacing as a swallowed StopIteration and silent zero-step training."""
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


def test_normalized_output_feeds_extract_conversation_images():
    img = _image()
    conv = normalize_vlm_conversation([{"user": "read the page", "assistant": "the page text"}], [img])
    history, images = extract_conversation_images(conv)
    assert images == [img]
    assert history[0]["content"][0] == {"type": "image"}, "processor placeholder must replace inline image data"
    # Uniform parts-form: string contents become text parts so the Arrow column schema is stable.
    assert history[1] == {"role": "assistant", "content": [{"type": "text", "text": "the page text"}]}


@pytest.mark.parametrize("payload", [b"", ""], ids=["empty-bytes", "empty-path"])
def test_a_falsy_but_present_image_payload_fails_loud(payload):
    """A truthiness test would drop the payload while still emitting the placeholder, a later placeholder-count
    mismatch."""
    conv = [{"role": "user", "content": [{"type": "image", "image": payload}]}]
    with pytest.raises((OSError, ValueError)):
        extract_conversation_images(conv)


@pytest.mark.parametrize(
    "part",
    [{"type": "image"}, {"type": "image", "image": None, "path": None, "url": None}],
    ids=["bare", "arrow-null"],
)
def test_an_image_part_nothing_filled_has_no_pixels_and_is_refused(part):
    """The images column fills placeholders before this point, so a part still empty would train its placeholder
    tokens with no pixels."""
    with pytest.raises(ValueError, match="carries no source"):
        extract_conversation_images([{"role": "user", "content": [part]}])


@pytest.mark.parametrize(
    "part",
    [
        {"type": "video", "video": "/data/a.mp4"},
        {"type": "video", "path": "/data/a.mp4"},
        {"type": "audio", "audio": "/data/a.wav"},
        {"type": "image_url", "image_url": {"url": "https://x/y.png"}},
        {"type": "input_image", "image_url": "https://x/y.png"},
        {"type": "video_url", "video_url": {"url": "https://x/y.mp4"}},
        {"type": "audio_url", "audio_url": {"url": "https://x/y.wav"}},
        {"type": "input_audio", "input_audio": {"data": "AAAA", "format": "wav"}},
    ],
    ids=lambda part: part["type"],
)
@pytest.mark.parametrize("pipeline", ["vlm", "text"])
def test_a_media_part_no_collator_feeds_a_processor_is_refused(part, pipeline):
    conversation = [{"role": "user", "content": [part, {"type": "text", "text": "Describe."}]}]
    with pytest.raises(ValueError, match="no Halo collator feeds a processor"):
        if pipeline == "vlm":
            extract_conversation_images(conversation)
        else:
            reject_image_content(conversation, "the prompt")


@pytest.fixture
def image_file(tmp_path):
    path = tmp_path / "page.png"
    _image(77).save(path)
    return path


def _base64_of(path) -> str:
    return base64.b64encode(path.read_bytes()).decode()


@pytest.mark.parametrize(
    "spell",
    [
        lambda path: {"path": str(path)},
        lambda path: {"url": str(path)},
        lambda path: {"url": f"file://{path}"},
        lambda path: {"url": f"data:image/png;base64,{_base64_of(path)}"},
        lambda path: {"base64": _base64_of(path)},
        lambda path: {"base64": f"data:image/png;base64,{_base64_of(path)}"},
        # Arrow's union pads the keys a part lacks.
        lambda path: {"image": None, "path": str(path), "url": None, "base64": None},
    ],
    ids=["path", "url-path", "url-file", "url-data", "base64", "base64-data", "arrow-null-keys"],
)
def test_the_chat_format_spellings_are_payloads_that_decode_to_the_pixels(image_file, spell):
    conversation = [{"role": "user", "content": [{"type": "image", **spell(image_file)}]}]
    history, images = extract_conversation_images(conversation)
    assert history[0]["content"] == [{"type": "image"}]
    assert np.array_equal(np.asarray(images[0].convert("RGB")), np.asarray(_image(77)))


@pytest.mark.parametrize("url", ["https://x/y.png", "http://x/y.png"])
def test_a_remote_url_is_refused_since_training_fetches_nothing(url):
    with pytest.raises(ValueError, match="training fetches nothing"):
        extract_conversation_images([{"role": "user", "content": [{"type": "image", "url": url}]}])


def test_a_chat_format_part_is_no_placeholder_for_the_images_column_to_fill(image_file):
    filled = {"type": "image", "path": str(image_file)}
    conversation = [{"role": "user", "content": [filled, {"type": "image"}]}]
    out = normalize_vlm_conversation(conversation, [_image(5)])
    assert out[0]["content"][0] == filled, "a part with a source keeps it"
    assert np.array_equal(np.asarray(out[0]["content"][1]["image"]), np.asarray(_image(5)))


def test_build_vlm_row_processor_images_field():
    img = _image()
    row = {"texts": [{"user": "describe", "assistant": "a cat"}], "images": [img]}

    example = build_vlm_row_processor(conversation_field="texts", images_field="images")(dict(row))
    assert [m["role"] for m in example["history"]] == ["user", "assistant"]
    assert example["images"] == [img]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
