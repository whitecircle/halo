#!/usr/bin/env python
"""VLM capability-refusal propagation and high-rejection-rate logging. The per-row degrade must not swallow
``tokenize_vlm_dataset``'s ``NotImplementedError``, or a mixed dataset trains text-only; at
``_HIGH_REJECTION_WARN_FRACTION`` the stats warn and never raise."""

import base64
import io
import logging

import numpy as np
import pytest
import torch
from datasets import Dataset
from jinja2 import TemplateError
from PIL import Image

from halo.data.pipeline.coordinated_map import coordinated_map_and_filter
from halo.data.pipeline.rejected_rows import _HIGH_REJECTION_WARN_FRACTION
from halo.data.preprocessed.metadata import PreprocessingConfig
from halo.data.preprocessed.tokenization import tokenize_vlm_dataset
from halo.trainers.reward.classification import _filter_over_length
from tests.common.vlm_fakes import FakeVLMProcessorBase

_REJECTION_LOGGER = "halo.data.pipeline.rejected_rows"


@pytest.fixture(autouse=True)
def _isolated_datasets_cache(tmp_path, monkeypatch):
    """Fresh coordinated-op cache per test: a stale cache hit would skip the map under test."""
    monkeypatch.setenv("HF_DATASETS_CACHE", str(tmp_path / "hf_datasets"))


class _UnsupportedKeysVLMProcessor(FakeVLMProcessorBase):
    """Emits an extra vision key (``image_sizes``) for image rows, which must trip the schema-capability refusal."""

    def __call__(self, text, return_tensors="pt", images=None, videos=None, add_special_tokens=True, **kwargs):
        result = self.encode_batch(text)
        images = [image for row_images in images or () for image in row_images]
        if images:
            result["pixel_values"] = torch.zeros(len(images), 4)
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
    """One image-carrying conversation; ``payload`` is whatever ``decode_image`` is handed."""
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
    """A handler swallowing the ``NotImplementedError`` returns a text-only dataset, and the 100%-drop guard cannot
    fire while text rows survive."""
    dataset = _mixed_vlm_dataset(num_text=4, num_image=4)

    with pytest.raises(NotImplementedError, match="vision keys"):
        tokenize_vlm_dataset(dataset, _UnsupportedKeysVLMProcessor(), _vlm_config(), split_name="train")


def test_bad_rows_still_degrade_per_row():
    """Only the capability refusal escalates."""

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
    """Single-turn templates reject a row with ``raise_exception``, a ``jinja2.TemplateError``; outside the row-data
    tuple it aborts a 2M-row job over one bad conversation."""

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
    """Named explicitly, since the shared ``OSError`` base would also swallow a full disk or an I/O fault."""
    dataset = Dataset.from_list([_TEXT_ROW] * 2 + [_image_row("/nonexistent/missing.png")])
    tokenized = tokenize_vlm_dataset(dataset, _ImageTolerantVLMProcessor(), _vlm_config(), split_name="train")
    assert len(tokenized) == 2, "an unreadable image path did not drop just its own row"


def test_an_oversized_image_drops_the_row(monkeypatch, tmp_path):
    """``DecompressionBombError`` subclasses plain ``Exception``, so outside the row-error tuple it would abort the
    whole map."""
    dataset = _mixed_vlm_dataset(num_text=2, num_image=1)

    baseline = tokenize_vlm_dataset(dataset, _ImageTolerantVLMProcessor(), _vlm_config(), split_name="train")
    assert len(baseline) == 3, "the image row must bake when its pixel count is within budget"

    # The pixel budget is a PIL global invisible to the map's cache key, so the run below would otherwise replay the
    # baseline's rows.
    monkeypatch.setenv("HF_DATASETS_CACHE", str(tmp_path / "hf_datasets_bomb"))
    # Under the row's 28x28: PIL raises above 2x MAX_IMAGE_PIXELS, warns between.
    monkeypatch.setattr(Image, "MAX_IMAGE_PIXELS", 16)
    tokenized = tokenize_vlm_dataset(dataset, _ImageTolerantVLMProcessor(), _vlm_config(), split_name="train")
    assert len(tokenized) == 2, "the oversized image did not drop just its own row"


def test_a_filesystem_error_aborts_the_map():
    """An OS fault is not a bad row: counting ENOSPC/EIO as one thins the corpus silently."""

    class _OutOfSpaceProcessor(_ImageTolerantVLMProcessor):
        def apply_chat_template(self, history, **kwargs):
            raise OSError(28, "No space left on device")

    dataset = Dataset.from_list([{"conversation": [{"role": "user", "content": "hi there"}]}] * 3)
    with pytest.raises(OSError, match="No space left on device"):
        tokenize_vlm_dataset(dataset, _OutOfSpaceProcessor(), _vlm_config(), split_name="train")


def test_non_data_errors_abort_the_map():
    """A bug or environment fault must abort the map and not quietly thin the corpus."""

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
    with caplog.at_level(logging.INFO, logger=_REJECTION_LOGGER):
        result = coordinated_map_and_filter(
            dataset,
            _reject_fraction_process_fn,
            remove_columns=["value"],
            desc="rejection logging",
            num_proc=1,
        )
    assert len(result) == keep
    return [r for r in caplog.records if "rejected during processing" in r.getMessage()]


def test_high_rejection_fraction_warns(caplog):
    """Filter stats land at WARNING and name the likely causes."""
    assert _HIGH_REJECTION_WARN_FRACTION <= 0.6, "test drops 60%; keep it above the threshold"
    records = _run_filter(reject=6, keep=4, caplog=caplog)
    assert records, "no rejection-stats record emitted"
    assert records[-1].levelno == logging.WARNING
    assert "config bug" in records[-1].getMessage()


def test_low_rejection_fraction_stays_info(caplog):
    records = _run_filter(reject=2, keep=8, caplog=caplog)
    assert records, "no rejection-stats record emitted"
    assert all(r.levelno == logging.INFO for r in records)


def test_classification_over_length_filter_logs(caplog):
    """A private copy of the report would drift from the shared threshold; a silent filter hides how much it
    removed."""
    long_ids = list(range(32))
    short_ids = [1, 2, 3]
    dataset = Dataset.from_dict({"input_ids": [long_ids] * 3 + [short_ids]})

    with caplog.at_level(logging.INFO, logger=_REJECTION_LOGGER):
        kept = _filter_over_length(dataset, max_length=8, num_proc=None, split_name="train")
    assert len(kept) == 1
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING and "3/4 examples" in r.getMessage()]
    assert warnings, "75% over-length drop did not warn"

    caplog.clear()
    dataset_low = Dataset.from_dict({"input_ids": [long_ids] + [short_ids] * 9})
    with caplog.at_level(logging.INFO, logger=_REJECTION_LOGGER):
        kept_low = _filter_over_length(dataset_low, max_length=8, num_proc=None, split_name="eval")
    assert len(kept_low) == 9
    infos = [r for r in caplog.records if "1/10 examples" in r.getMessage()]
    assert infos and all(r.levelno == logging.INFO for r in infos)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
