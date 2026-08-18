#!/usr/bin/env python3
"""Offline-baked VLM tokenization must equal what the runtime VLM path renders for the same row.

The two paths exist so a run can skip tokenization at startup — the artifact is an OPTIMIZATION of
the runtime render, so a row prepared offline must train on exactly the ids the untouched runtime
path would produce from the same row with the same processor. Both sides here are the production
entry points (``tokenize_vlm_dataset`` offline; ``create_vlm_processor`` + ``VLMDataCollator`` at
runtime) driven over the same Arrow dataset: re-implementing either side would certify nothing, and
the Arrow round-trip is load-bearing — it unions the content-parts struct over the split, so a text
part gains a null ``image`` key as soon as any row anywhere carries an image.

Image rows have no baked form to compare against: the stored schema holds four processor keys, and
every current VLM processor also emits ``mm_token_type_ids``, so the offline path refuses image rows
by design. What must hold there is that the refusal is REACHED — the row's images have to be
extracted by the same extractor the runtime uses, not by a vision scan that aborts the map first.

Run: pytest tests/cpu/data/test_vlm_render_equivalence.py
"""

import base64
import io
import sys

import numpy as np
import pytest
from datasets import Dataset, Features, Value
from PIL import Image

from src.data.collators.vlm import VLMDataCollator
from src.data.pipeline.preprocessed_metadata import PreprocessingConfig
from src.data.pipeline.preprocessing import tokenize_vlm_dataset
from src.data.pipeline.row_processors import create_vlm_processor
from src.data.pipeline.vlm_dataset import _vlm_extra_columns, vlm_map_features
from tests.common.tokenizers import load_cached_processor

# Processor-only snapshots are enough (no weights are loaded); revisions are pinned because hub main
# can drift a chat template out from under the comparison. Two template dialects, so a divergence
# only one of them expresses cannot pass unnoticed.
PROCESSORS = [
    ("Qwen/Qwen2.5-VL-3B-Instruct", "66285546d2b821cf421d4f5eb2576359d3770cd3"),
    ("Qwen/Qwen3-VL-2B-Instruct", "89644892e4d85e24eaac8bacfd4f463576704203"),
]

MAX_LENGTH = 8192

# A conversation column whose content parts declare the image key every part of a mixed dataset
# carries after the Arrow round-trip — text parts included, where it is null.
_UNIONED_CONVERSATION_FEATURES = Features(
    {
        "conversation": [
            {
                "role": Value("string"),
                "content": [{"type": Value("string"), "text": Value("string"), "image": Value("string")}],
            }
        ]
    }
)


@pytest.fixture(autouse=True)
def _isolated_datasets_cache(tmp_path, monkeypatch):
    """Fresh coordinated-op cache per test: a stale cache hit would skip the map under test."""
    monkeypatch.setenv("HF_DATASETS_CACHE", str(tmp_path / "hf_datasets"))


def _image(seed: int, height: int = 100, width: int = 140) -> Image.Image:
    return Image.fromarray(np.random.RandomState(seed).randint(0, 255, (height, width, 3), dtype=np.uint8))


def _data_uri(image: Image.Image) -> str:
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return "data:image;base64," + base64.b64encode(buf.getvalue()).decode()


def _string_content_dataset() -> Dataset:
    return Dataset.from_list(
        [
            {
                "conversation": [
                    {"role": "user", "content": "What is 2 + 2?"},
                    {"role": "assistant", "content": "2 + 2 equals 4."},
                ]
            }
        ]
    )


def _parts_content_dataset() -> Dataset:
    return Dataset.from_list(
        [
            {
                "conversation": [
                    {"role": "user", "content": [{"type": "text", "text": "What is 2 + 2?"}]},
                    {"role": "assistant", "content": [{"type": "text", "text": "2 + 2 equals 4."}]},
                ]
            }
        ]
    )


def _arrow_unioned_text_dataset() -> Dataset:
    """Text rows carrying the null ``image`` key Arrow gives every part of a mixed split."""
    return Dataset.from_list(
        [
            {
                "conversation": [
                    {"role": "user", "content": [{"type": "text", "text": "What is 2 + 2?", "image": None}]},
                    {"role": "assistant", "content": [{"type": "text", "text": "2 + 2 equals 4.", "image": None}]},
                ]
            }
        ],
        features=_UNIONED_CONVERSATION_FEATURES,
    )


TEXT_DATASETS = {
    "arrow_unioned_parts": _arrow_unioned_text_dataset,
    "parts_content": _parts_content_dataset,
    "string_content": _string_content_dataset,
}


def _embedded_image_dataset() -> Dataset:
    return Dataset.from_list(
        [
            {
                "conversation": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": _data_uri(_image(0))},
                            {"type": "text", "text": "What is it?"},
                        ],
                    },
                    {"role": "assistant", "content": [{"type": "text", "text": "A noise patch."}]},
                ]
            }
        ]
    )


def _multi_image_dataset() -> Dataset:
    return Dataset.from_list(
        [
            {
                "conversation": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": _data_uri(_image(1))},
                            {"type": "image", "image": _data_uri(_image(2, 224, 224))},
                            {"type": "text", "text": "Compare them."},
                        ],
                    },
                    {"role": "assistant", "content": [{"type": "text", "text": "Both are noise."}]},
                ]
            }
        ]
    )


def _images_column_dataset() -> Dataset:
    """The hub VLM shape: images live in their own column, not in the message content."""
    return Dataset.from_list(
        [
            {
                "conversation": [
                    {"role": "user", "content": "Describe the picture."},
                    {"role": "assistant", "content": "Noise."},
                ],
                "images": [_image(3)],
            }
        ]
    )


IMAGE_DATASETS = {
    "embedded_image": (_embedded_image_dataset, None),
    "images_column": (_images_column_dataset, "images"),
    "multi_image": (_multi_image_dataset, None),
}


def _preprocessing_config(images_field: str | None) -> PreprocessingConfig:
    return PreprocessingConfig(
        model_name_or_path="local/processor",
        max_length=MAX_LENGTH,
        conversation_field="conversation",
        images_field=images_field,
        is_vlm=True,
        num_proc=1,
    )


def _runtime_batch(processor, dataset: Dataset, images_field: str | None) -> dict:
    """The batch an untouched runtime VLM SFT run builds from this dataset's single row."""
    mapped = dataset.map(
        create_vlm_processor(conversation_field="conversation", images_field=images_field),
        remove_columns=_vlm_extra_columns(dataset),
        features=vlm_map_features(),
        load_from_cache_file=False,
    )
    collator = VLMDataCollator(processor, processor.tokenizer, MAX_LENGTH)
    return collator([mapped[0]])


@pytest.mark.parametrize("model_name,revision", PROCESSORS)
@pytest.mark.parametrize("case", sorted(TEXT_DATASETS))
def test_offline_bake_matches_runtime_render(model_name, revision, case):
    """Same row, same processor: the baked ids and labels must equal the runtime render's.

    A mismatch means a preprocessed dataset trains on a different tokenization than the config it
    was prepared from renders at runtime — silently, since both artifacts are well-formed.
    """
    processor = load_cached_processor(model_name, revision=revision)
    dataset = TEXT_DATASETS[case]()

    baked = tokenize_vlm_dataset(dataset, processor, _preprocessing_config(None), "train")
    assert len(baked) == 1, f"{model_name} / {case}: the offline bake DROPPED the row — nothing was compared"
    batch = _runtime_batch(processor, dataset, None)

    runtime_ids = batch["input_ids"][0].tolist()
    assert baked[0]["input_ids"] == runtime_ids, (
        f"{model_name} / {case}: offline baked {len(baked[0]['input_ids'])} ids, runtime rendered {len(runtime_ids)}"
    )
    assert baked[0]["labels"] == batch["labels"][0].tolist(), f"{model_name} / {case}: baked labels diverge"
    assert baked[0]["pixel_values"] is None, f"{model_name} / {case}: a text row baked pixels"
    assert "pixel_values" not in batch, f"{model_name} / {case}: a text row collated pixels"


@pytest.mark.parametrize("model_name,revision", PROCESSORS)
@pytest.mark.parametrize("case", sorted(IMAGE_DATASETS))
def test_image_rows_reach_the_schema_capability_refusal(model_name, revision, case):
    """An image row must reach the stored-schema refusal, whatever shape it arrives in.

    The refusal ("this family emits keys the artifact cannot store") is the offline path's only
    defined outcome for an image row today, and it fires per row AFTER the images are extracted —
    so it is reachable only if extraction handles every shape the runtime path handles. An
    extractor keying on an ``image`` key's presence instead of the part TYPE trips over the null key
    Arrow adds to text parts and aborts the whole map with a third-party ``AttributeError`` before
    any refusal, turning a precise capability message into an opaque crash.
    """
    processor = load_cached_processor(model_name, revision=revision)
    build_dataset, images_field = IMAGE_DATASETS[case]

    with pytest.raises(NotImplementedError, match="the preprocessed VLM schema cannot store"):
        tokenize_vlm_dataset(build_dataset(), processor, _preprocessing_config(images_field), "train")


@pytest.mark.parametrize("model_name,revision", PROCESSORS)
@pytest.mark.parametrize("case", sorted(IMAGE_DATASETS))
def test_image_rows_render_and_collate_at_runtime(model_name, revision, case):
    """The runtime path must render every image shape into a batch whose vision tensors line up.

    The counterpart to the refusal above: the shapes the artifact cannot store are exactly the ones
    the runtime path does carry, so a regression there would leave image training with no path at
    all. The expanded image placeholders are what pair the text with ``pixel_values``.
    """
    processor = load_cached_processor(model_name, revision=revision)
    build_dataset, images_field = IMAGE_DATASETS[case]

    batch = _runtime_batch(processor, build_dataset(), images_field)

    expected_images = 2 if case == "multi_image" else 1
    assert batch["image_grid_thw"].shape[0] == expected_images, f"{model_name} / {case}: image count diverges"
    # image_grid_thw counts patches; merge_size**2 of them collapse into one placeholder token.
    merge_size = processor.image_processor.merge_size
    expected_pads = int(batch["image_grid_thw"].prod(dim=-1).sum()) // merge_size**2
    image_token_id = processor.tokenizer.convert_tokens_to_ids(processor.image_token)
    assert int((batch["input_ids"] == image_token_id).sum()) == expected_pads, (
        f"{model_name} / {case}: expanded image placeholders do not match the image grid"
    )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
