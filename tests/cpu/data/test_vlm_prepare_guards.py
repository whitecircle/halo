#!/usr/bin/env python3
"""Config-time guards on ``prepare_vlm_dataset``, so every consumer inherits them: a mistyped ``images_field``,
``interleaved_thinking`` on a VLM template, and a pinned map schema beside kept columns."""

import types

import pytest
from datasets import Dataset, DatasetDict

from halo.data.pipeline import vlm_rows
from halo.data.pipeline.vlm_rows import prepare_vlm_dataset, vlm_map_features
from halo.entrypoints import model_setup


def _args(**overrides) -> types.SimpleNamespace:
    defaults = {
        "dataset": "the loaded dataset",
        "conversation_field": "conversation",
        "images_field": None,
        "system_prompt": None,
        "model_supports_system_role": True,
        "tools_field": None,
        "interleaved_thinking": False,
        "train_on_completions_only": False,
        "assistant_message_template": None,
    }
    return types.SimpleNamespace(**{**defaults, **overrides})


def _dataset() -> DatasetDict:
    return DatasetDict({"train": Dataset.from_dict({"conversation": ["a"], "images": ["i"]})})


def test_missing_images_field_column_raises_before_any_processing():
    """Both guards fire before the tokenizer/processor is touched, so None stand-ins prove it."""
    with pytest.raises(ValueError, match="images_field='pictures' names a column the dataset"):
        prepare_vlm_dataset(_dataset(), _args(images_field="pictures"), None, None, 128, 1)


def test_matching_images_field_passes_the_guard(monkeypatch):
    class _ReachedProcessing(Exception):
        pass

    def _boom(**_kwargs):
        raise _ReachedProcessing

    monkeypatch.setattr(vlm_rows, "build_vlm_row_processor", _boom)
    with pytest.raises(_ReachedProcessing):
        prepare_vlm_dataset(_dataset(), _args(images_field="images"), None, None, 128, 1)


def test_completion_masking_without_a_marker_is_refused_before_the_map():
    """The VLM collators refuse the pair too, but only after this map has paid the whole tokenization."""
    args = _args(train_on_completions_only=True, assistant_message_template=None)
    with pytest.raises(ValueError, match="requires assistant_message_template"):
        prepare_vlm_dataset(_dataset(), args, None, None, 128, 1)


def test_interleaved_thinking_still_refused_on_the_vlm_path():
    with pytest.raises(ValueError, match="interleaved_thinking is text-only"):
        prepare_vlm_dataset(_dataset(), _args(interleaved_thinking=True), None, None, 128, 1)


def test_pinned_schema_with_kept_columns_is_refused_at_the_seam():
    """``features`` pins exactly the mapped columns; unguarded the pair dies on an Arrow cast inside a map worker,
    naming a type."""
    with pytest.raises(ValueError, match="pinned Arrow schema"):
        prepare_vlm_dataset(
            _dataset(),
            _args(images_field="images"),
            None,
            None,
            128,
            1,
            keep=("solution",),
            features=vlm_map_features(),
        )


def test_the_vlm_loader_takes_the_modality_verdict_instead_of_re_probing(monkeypatch):
    """``probe_checkpoint`` enters the ``vlm_probe`` store phase and only multimodal runs reach this loader, so
    probing here makes the phase count rank-dependent, leaving peers waiting on a key nobody writes."""

    def _must_not_probe(*args, **kwargs):
        raise AssertionError("_load_vlm_model_and_processor re-entered the vlm_probe store phase")

    class _Sentinel(Exception):
        pass

    def _processor_reached(*args, **kwargs):
        raise _Sentinel

    monkeypatch.setattr(model_setup, "probe_checkpoint", _must_not_probe)
    # The processor load is the loader's first step; reaching it proves no probe ran.
    monkeypatch.setattr(model_setup, "load_vlm_processor", _processor_reached)
    model_config = types.SimpleNamespace(model_name_or_path="Qwen/Qwen3-4B", model_revision=None)
    with pytest.raises(_Sentinel):
        model_setup._load_vlm_model_and_processor(None, None, model_config, None, vlm_run=True, attn_default=None)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
