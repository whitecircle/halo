#!/usr/bin/env python
"""The raw-VLM data paths of the training scripts keep mixed-modality rows as the row processor emits them.

An image row and a text-only row infer different Arrow types for the mapped ``history`` content, so
a ``num_proc > 1`` map whose shards split the modalities either fails to align them or coerces every
image part into the text-part struct (``{"type": "image", "text": None}``) — unless the map pins
``vlm_map_features``. Each script path that maps raw VLM rows with no extra kept column is driven
here end to end through ``prepare_vlm_dataset``.

Run: pytest tests/cpu/config/test_vlm_script_map_schema.py
"""

from types import SimpleNamespace
from unittest import mock

import numpy as np
import pytest
from datasets import Dataset, DatasetDict
from PIL import Image

from tests.common.utils import load_script_module

_ROWS_PER_MODALITY = 4


class _Processor:
    """Renders a history to a short string, enough for the over-length filter to measure."""

    def apply_chat_template(self, history, tokenize=False, add_generation_prompt=False, **kwargs):
        return " ".join(str(message["content"]) for message in history)


class _Tokenizer:
    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": text.split()}


def _mixed_split() -> Dataset:
    """Image rows first, text rows second, so a two-worker map gives each worker one modality."""
    image = Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8))
    return Dataset.from_dict(
        {
            "texts": [[{"user": f"question {i}", "assistant": f"answer {i}"}] for i in range(2 * _ROWS_PER_MODALITY)],
            "images": [[image]] * _ROWS_PER_MODALITY + [[]] * _ROWS_PER_MODALITY,
        }
    )


def _args() -> SimpleNamespace:
    return SimpleNamespace(
        dataset="dummy/mixed",
        conversation_field="texts",
        images_field="images",
        system_prompt=None,
        model_supports_system_role=True,
        tools_field=None,
        interleaved_thinking=False,
        assistant_message_template=None,
        train_on_completions_only=False,
        train_on_last_assistant_only=False,
    )


def _training_config() -> SimpleNamespace:
    return SimpleNamespace(
        dataset_num_proc=2, max_length=10_000, remove_unused_columns=True, packing=False, padding_free=False
    )


def _sft_vlm_path(ds):
    module = load_script_module("scripts/training/sft.py", "halo_test_vlm_schema_sft")
    with mock.patch.object(module, "VLMDataCollator"):
        train, test, _, _ = module._prepare_vlm_data(
            ds, False, _args(), _training_config(), None, _Processor(), _Tokenizer(), None
        )
    return train, test


def _teacher_distill_vlm_path(ds):
    module = load_script_module(
        "scripts/training/distillation/teacher_distill.py", "halo_test_vlm_schema_teacher_distill"
    )
    with mock.patch.object(module, "VLMDataCollator"):
        train, test, _ = module._prepare_vlm_distill_data(
            ds, _args(), _training_config(), _Processor(), _Tokenizer(), None
        )
    return train, test


@pytest.mark.parametrize("prepare", [_sft_vlm_path, _teacher_distill_vlm_path], ids=["sft", "teacher_distill"])
def test_raw_vlm_map_keeps_mixed_modality_rows_intact(prepare):
    train, test = prepare(DatasetDict({"train": _mixed_split(), "test": _mixed_split()}))
    for split in (train, test):
        assert len(split) == 2 * _ROWS_PER_MODALITY
        image_row, text_row = split[0], split[-1]
        assert image_row["history"][0]["content"][0] == {"type": "image"}, "the image part gained a text slot"
        assert isinstance(image_row["images"][0], Image.Image)
        assert text_row["history"][0]["content"] == [{"type": "text", "text": f"question {len(split) - 1}"}]
        assert text_row["images"] == []


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
