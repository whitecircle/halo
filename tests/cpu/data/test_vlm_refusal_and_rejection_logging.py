#!/usr/bin/env python
"""VLM capability-refusal propagation + high-rejection-rate logging.

Two silent-failure modes:

1. ``tokenize_vlm_dataset``'s per-row map deliberately raises ``NotImplementedError`` when the
   processor emits vision keys the preprocessed schema cannot store, and the per-row
   ``except Exception`` degrade must not swallow it: swallowing silently drops every IMAGE row of a
   mixed dataset (the 100%-drop guard never fires when text rows survive) and trains text-only. The
   refusal must abort the map.

2. ``process_dataset_with_map_and_filter`` must not log the rejected fraction at INFO regardless of
   magnitude — a template/config bug rejecting most of the corpus then scrolls by unnoticed. At or
   above ``_HIGH_REJECTION_WARN_FRACTION`` it must WARN (never raise — legitimately high drop rates
   exist). Same threshold for the classification trainer's over-length filter, which otherwise logs
   nothing at all.

Run: pytest tests/cpu/data/test_vlm_refusal_and_rejection_logging.py
"""

import base64
import io
import logging
import sys

import numpy as np
import pytest
import torch
from datasets import Dataset
from jinja2 import TemplateError
from PIL import Image

from src.data.pipeline.preprocessed_metadata import PreprocessingConfig
from src.data.pipeline.preprocessing import tokenize_vlm_dataset
from src.data.pipeline.processing import (
    _HIGH_REJECTION_WARN_FRACTION,
    process_dataset_with_map_and_filter,
)
from tests.common.vlm_fakes import FakeVLMProcessorBase

_PROCESSING_LOGGER = "src.data.pipeline.processing"


@pytest.fixture(autouse=True)
def _isolated_datasets_cache(tmp_path, monkeypatch):
    """Fresh coordinated-op cache per test: a stale cache hit would skip the map under test."""
    monkeypatch.setenv("HF_DATASETS_CACHE", str(tmp_path / "hf_datasets"))


class _UnsupportedKeysVLMProcessor(FakeVLMProcessorBase):
    """Whitespace 'processor' that emits an extra vision key (``image_sizes``) for image rows —
    the shape that must trip the schema-capability refusal, not a silent per-row drop."""

    def __call__(self, text, return_tensors="pt", images=None, videos=None, add_special_tokens=True, **kwargs):
        result = self.encode_batch(text)
        if images:
            result["pixel_values"] = torch.zeros(len(images), 4)
            # The key the preprocessed schema cannot store — must trigger the refusal.
            result["image_sizes"] = torch.tensor([[8, 8]] * len(images))
        return result


class _ImageTolerantVLMProcessor(_UnsupportedKeysVLMProcessor):
    """The same fake minus the unstorable vision key, so an image row bakes instead of refusing."""

    def __call__(self, text, **kwargs):
        result = super().__call__(text, **kwargs)
        result.pop("image_sizes", None)
        return result


def _image_data_uri() -> str:
    img = Image.fromarray(np.zeros((28, 28, 3), dtype=np.uint8))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return "data:image;base64," + base64.b64encode(buf.getvalue()).decode()


_TEXT_ROW = {
    "conversation": [
        {"role": "user", "content": [{"type": "text", "text": "hi there"}]},
        {"role": "assistant", "content": [{"type": "text", "text": "ok"}]},
    ]
}


def _image_row(payload: str) -> dict:
    """One image-carrying conversation; ``payload`` is whatever ``process_image`` is handed."""
    return {
        "conversation": [
            {
                "role": "user",
                "content": [{"type": "image", "image": payload}, {"type": "text", "text": "what is this"}],
            },
            {"role": "assistant", "content": [{"type": "text", "text": "a square"}]},
        ]
    }


def _mixed_vlm_dataset(num_text: int, num_image: int) -> Dataset:
    return Dataset.from_list([_TEXT_ROW] * num_text + [_image_row(_image_data_uri())] * num_image)


def _plain_text_dataset(good: int, bad: int) -> Dataset:
    """String-content rows; the ``BOOM`` ones are what the row-refusing fakes below key on."""
    good_row = {"conversation": [{"role": "user", "content": "hi there"}]}
    bad_row = {"conversation": [{"role": "user", "content": "BOOM"}]}
    return Dataset.from_list([good_row] * good + [bad_row] * bad)


def _vlm_config() -> PreprocessingConfig:
    return PreprocessingConfig(
        model_name_or_path="fake/vlm",
        max_length=64,
        conversation_field="conversation",
        is_vlm=True,
        num_proc=1,
    )


def test_unsupported_vision_keys_abort_mixed_dataset_tokenization():
    """A processor emitting unsupported vision keys on a MIXED dataset must raise. A per-row handler
    that swallows the deliberate ``NotImplementedError`` drops every image row and returns a
    text-only dataset — the 100%-drop guard cannot fire while text rows survive."""
    dataset = _mixed_vlm_dataset(num_text=4, num_image=4)

    with pytest.raises(NotImplementedError, match="vision keys"):
        tokenize_vlm_dataset(dataset, _UnsupportedKeysVLMProcessor(), _vlm_config(), split_name="train")


def test_bad_rows_still_degrade_per_row():
    """The per-row degrade must survive genuinely bad rows (here: a row the fake template refuses to
    render) — only the capability refusal escalates."""

    class _CrashingTemplateProcessor(_UnsupportedKeysVLMProcessor):
        # Keyed on the RENDERED text, not the message shape: the render seam is handed the
        # placeholder history, whose content is uniformly parts-form.
        def apply_chat_template(self, history, **kwargs):
            rendered = super().apply_chat_template(history, **kwargs)
            if "BOOM" in rendered:
                raise ValueError("bad row")
            return rendered

    tokenized = tokenize_vlm_dataset(
        _plain_text_dataset(good=3, bad=1), _CrashingTemplateProcessor(), _vlm_config(), split_name="train"
    )
    assert len(tokenized) == 3


def test_a_template_row_refusal_drops_the_row():
    """Every single-turn template in ``jinja-templates/`` rejects a row it cannot render with
    ``raise_exception``, which transformers raises as ``jinja2.TemplateError``. That is a per-ROW
    data refusal: left out of the row-data tuple it aborts a 2M-row job over one bad conversation."""

    class _RefusingTemplateProcessor(_ImageTolerantVLMProcessor):
        def apply_chat_template(self, history, **kwargs):
            rendered = super().apply_chat_template(history, **kwargs)
            if "BOOM" in rendered:
                raise TemplateError("This template only supports single-turn data")
            return rendered

    tokenized = tokenize_vlm_dataset(
        _plain_text_dataset(good=3, bad=1), _RefusingTemplateProcessor(), _vlm_config(), split_name="train"
    )
    assert len(tokenized) == 3, "the template's per-row refusal did not drop just that row"


def test_an_unreadable_image_path_drops_the_row():
    """``process_image`` opens a path row-side, so a missing file is a property of the ROW.

    Named explicitly (``FileNotFoundError``/``UnidentifiedImageError``) rather than caught via their
    shared ``OSError`` base — the base would also swallow a full disk or an I/O fault.
    """
    dataset = Dataset.from_list([_TEXT_ROW] * 2 + [_image_row("/nonexistent/missing.png")])
    tokenized = tokenize_vlm_dataset(dataset, _ImageTolerantVLMProcessor(), _vlm_config(), split_name="train")
    assert len(tokenized) == 2, "an unreadable image path did not drop just its own row"


def test_an_oversized_image_drops_the_row(monkeypatch, tmp_path):
    """PIL refuses a decompression bomb with ``DecompressionBombError``, which subclasses plain
    ``Exception`` — outside the tuple it aborted the whole map over one oversized image."""
    dataset = _mixed_vlm_dataset(num_text=2, num_image=1)

    baseline = tokenize_vlm_dataset(dataset, _ImageTolerantVLMProcessor(), _vlm_config(), split_name="train")
    assert len(baseline) == 3, "the image row must bake when its pixel count is within budget"

    # The pixel budget is a PIL global, invisible to the map's cache key — the run below would
    # otherwise replay the baseline's cached rows and assert nothing.
    monkeypatch.setenv("HF_DATASETS_CACHE", str(tmp_path / "hf_datasets_bomb"))
    # Under the row's 28x28: PIL raises above 2x MAX_IMAGE_PIXELS, warns between.
    monkeypatch.setattr(Image, "MAX_IMAGE_PIXELS", 16)
    tokenized = tokenize_vlm_dataset(dataset, _ImageTolerantVLMProcessor(), _vlm_config(), split_name="train")
    assert len(tokenized) == 2, "the oversized image did not drop just its own row"


def test_a_filesystem_error_aborts_the_map():
    """An OS fault is not a bad row: counting ENOSPC/EIO as one thins the corpus silently, exactly
    the failure the blanket catch was narrowed to prevent."""

    class _OutOfSpaceProcessor(_ImageTolerantVLMProcessor):
        def apply_chat_template(self, history, **kwargs):
            raise OSError(28, "No space left on device")

    dataset = Dataset.from_list([{"conversation": [{"role": "user", "content": "hi there"}]}] * 3)
    with pytest.raises(OSError, match="No space left on device"):
        tokenize_vlm_dataset(dataset, _OutOfSpaceProcessor(), _vlm_config(), split_name="train")


def test_non_data_errors_abort_the_map():
    """Only errors that are a property of the ROW may degrade to a drop. A bug or environment fault
    (here a ``RuntimeError`` from the processor) must abort the map instead of quietly thinning the
    corpus one row at a time — the same failure mode as the swallowed capability refusal above."""

    class _BuggyProcessor(_UnsupportedKeysVLMProcessor):
        def apply_chat_template(self, history, **kwargs):
            raise RuntimeError("processor bug")

    dataset = Dataset.from_list([{"conversation": [{"role": "user", "content": "hi there"}]}] * 3)
    config = PreprocessingConfig(
        model_name_or_path="fake/vlm",
        max_length=64,
        conversation_field="conversation",
        is_vlm=True,
        num_proc=1,
    )

    with pytest.raises(RuntimeError, match="processor bug"):
        tokenize_vlm_dataset(dataset, _BuggyProcessor(), config, split_name="train")


def _reject_fraction_process_fn(row):
    if row["value"] == 0:
        return {"input_ids": [0], "attention_mask": [0]}  # typed rejection sentinel
    return {"input_ids": [1, 2], "attention_mask": [1, 1]}


def _run_filter(reject: int, keep: int, caplog):
    dataset = Dataset.from_dict({"value": [0] * reject + [1] * keep})
    with caplog.at_level(logging.INFO, logger=_PROCESSING_LOGGER):
        result = process_dataset_with_map_and_filter(
            dataset,
            _reject_fraction_process_fn,
            remove_columns=["value"],
            desc="rejection logging",
            num_proc=1,
        )
    assert len(result) == keep
    return [r for r in caplog.records if "rejected during processing" in r.getMessage()]


def test_high_rejection_fraction_warns(caplog):
    """Above _HIGH_REJECTION_WARN_FRACTION the filter stats must land at WARNING and name the
    likely causes — not scroll by at INFO."""
    assert _HIGH_REJECTION_WARN_FRACTION <= 0.6, "test drops 60%; keep it above the threshold"
    records = _run_filter(reject=6, keep=4, caplog=caplog)
    assert records, "no rejection-stats record emitted"
    assert records[-1].levelno == logging.WARNING
    assert "config bug" in records[-1].getMessage()


def test_low_rejection_fraction_stays_info(caplog):
    """A modest drop rate is normal operation — INFO, no WARNING."""
    records = _run_filter(reject=2, keep=8, caplog=caplog)
    assert records, "no rejection-stats record emitted"
    assert all(r.levelno == logging.INFO for r in records)


def test_classification_over_length_filter_logs(caplog):
    """The classification trainer's over-length filter must report the dropped count through the
    shared reporter — WARNING at its threshold, INFO below it. A silent filter hides how much of the
    split it removed, and a private copy of the report would drift from that threshold."""
    from src.trainers.reward.classification import _filter_over_length

    long_ids = list(range(32))
    short_ids = [1, 2, 3]
    dataset = Dataset.from_dict({"input_ids": [long_ids] * 3 + [short_ids]})

    with caplog.at_level(logging.INFO, logger=_PROCESSING_LOGGER):
        kept = _filter_over_length(dataset, max_length=8, num_proc=None, split_name="train")
    assert len(kept) == 1
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING and "3/4 examples" in r.getMessage()]
    assert warnings, "75% over-length drop did not warn"

    caplog.clear()
    dataset_low = Dataset.from_dict({"input_ids": [long_ids] + [short_ids] * 9})
    with caplog.at_level(logging.INFO, logger=_PROCESSING_LOGGER):
        kept_low = _filter_over_length(dataset_low, max_length=8, num_proc=None, split_name="eval")
    assert len(kept_low) == 9
    infos = [r for r in caplog.records if "1/10 examples" in r.getMessage()]
    assert infos and all(r.levelno == logging.INFO for r in infos)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
