#!/usr/bin/env python3
"""Config-time guards on the shared raw-VLM preparation seam (``prepare_vlm_dataset``).

The guards live on the seam, not per script, so every consumer (SFT and both distillation scripts)
inherits them: a mistyped ``images_field`` names no dataset column and would otherwise train
text-only in silence; ``interleaved_thinking`` renders nothing on any supported VLM template; a
pinned map schema alongside kept columns can only fail, and does so inside a map worker.

Run: pytest tests/cpu/data/test_vlm_prepare_guards.py
"""

import sys
import types

import pytest
from datasets import Dataset, DatasetDict

from src.data.pipeline import vlm_dataset
from src.data.pipeline.vlm_dataset import prepare_vlm_dataset, vlm_map_features
from src.distributed.loading import vlm_setup


def _args(**overrides) -> types.SimpleNamespace:
    defaults = {
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
    """Anti-over-rejection: a real column must reach the processing pipeline."""

    class _ReachedProcessing(Exception):
        pass

    def _boom(**_kwargs):
        raise _ReachedProcessing

    monkeypatch.setattr(vlm_dataset, "create_vlm_processor", _boom)
    with pytest.raises(_ReachedProcessing):
        prepare_vlm_dataset(_dataset(), _args(images_field="images"), None, None, 128, 1)


def test_completion_masking_without_a_marker_is_refused_before_the_map():
    """The VLM collators refuse the pair too, but they are built AFTER this map — a run missing its
    marker would otherwise pay the whole VLM tokenization before the refusal it could not avoid."""
    args = _args(train_on_completions_only=True, assistant_message_template=None)
    with pytest.raises(ValueError, match="requires assistant_message_template"):
        prepare_vlm_dataset(_dataset(), args, None, None, 128, 1)


def test_interleaved_thinking_still_refused_on_the_vlm_path():
    with pytest.raises(ValueError, match="interleaved_thinking is text-only"):
        prepare_vlm_dataset(_dataset(), _args(interleaved_thinking=True), None, None, 128, 1)


def test_pinned_schema_with_kept_columns_is_refused_at_the_seam():
    """``features`` pins exactly the mapped columns, so a ``keep`` column has no slot in it.

    Unguarded the pair dies on an Arrow cast deep inside a map worker, naming a type rather than the
    two arguments that cannot be combined — and the docstring's constraint stays advisory.
    """
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
    """``is_vlm_model`` enters the ``vlm_probe`` c10d-store phase, and only multimodal runs reach
    ``load_vlm_model_and_processor``. Probing again from inside that branch makes the phase count
    rank-dependent the moment two ranks disagree on the verdict, leaving the peers waiting on a key
    nobody writes. The caller owns the verdict, so this loader must reach the processor without it."""

    def _must_not_probe(*args, **kwargs):
        raise AssertionError("load_vlm_model_and_processor re-entered the vlm_probe store phase")

    class _Sentinel(Exception):
        pass

    def _processor_reached(*args, **kwargs):
        raise _Sentinel

    monkeypatch.setattr(vlm_setup, "is_vlm_model", _must_not_probe)
    # The processor load is the first thing after the (removed) verdict; reaching it proves no probe ran.
    monkeypatch.setattr(vlm_setup, "load_vlm_processor", _processor_reached)
    model_config = types.SimpleNamespace(model_name_or_path="Qwen/Qwen3-4B", model_revision=None)
    with pytest.raises(_Sentinel):
        vlm_setup.load_vlm_model_and_processor(model_config, None, None)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
