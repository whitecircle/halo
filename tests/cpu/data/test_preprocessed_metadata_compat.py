#!/usr/bin/env python
"""Startup guards on the preprocessed-dataset path run per rank in front of a consensus all-reduce, so an
over-strict verdict kills the job at step 0. A foreign ``metadata.json`` loads raw, only a render knob the YAML
states is a claim, and ``tools_field`` may be absent from some sources of a list."""

import json
import os
from dataclasses import asdict, dataclass

import pytest
from datasets import Dataset, DatasetDict

from halo.data.preprocessed.metadata import (
    PreprocessedDatasetMetadata,
    PreprocessingConfig,
    _verify_render_knobs,
    is_preprocessed_dataset,
    load_preprocessed_metadata,
)
from halo.data.preprocessed.shard_index import IncompatiblePreprocessedDataset
from halo.data.sources.corpus import _reject_tools_field_nowhere
from halo.data.sources.paths import METADATA_FILE


def _write_metadata(directory, payload: dict) -> str:
    os.makedirs(directory, exist_ok=True)
    with open(os.path.join(directory, METADATA_FILE), "w") as f:
        json.dump(payload, f)
    return str(directory)


def test_a_foreign_metadata_json_loads_raw(tmp_path):
    """Judged against the stamp schema it would raise on every rank that read it, right before the consensus
    all-reduce, blocking ranks whose read flaked until the watchdog."""
    path = _write_metadata(tmp_path / "hub-corpus", {"license": "mit", "language": ["en"]})
    assert is_preprocessed_dataset(path) is False


def test_a_directory_without_metadata_loads_raw(tmp_path):
    (tmp_path / "plain").mkdir()
    assert is_preprocessed_dataset(str(tmp_path / "plain")) is False


def test_loading_a_foreign_metadata_as_a_stamp_names_the_real_problem(tmp_path):
    path = _write_metadata(tmp_path / "hub-corpus", {"license": "mit"})
    with pytest.raises(FileNotFoundError, match="not a toolkit preprocessing stamp"):
        load_preprocessed_metadata(path)


def test_our_stamp_at_a_wrong_version_still_reports_preprocessed(tmp_path):
    """The dataset is pre-tokenized, so False would re-tokenize baked rows. True keeps every rank on one branch and
    lets the load raise the version error together."""
    path = _write_metadata(tmp_path / "future", {"preprocessed": True, "version": "9.9"})
    assert is_preprocessed_dataset(path) is True
    with pytest.raises(IncompatiblePreprocessedDataset, match="version"):
        load_preprocessed_metadata(path)


def test_our_current_stamp_is_detected(tmp_path):
    path = _write_metadata(tmp_path / "prepared", asdict(PreprocessedDatasetMetadata()))
    assert is_preprocessed_dataset(path) is True
    assert load_preprocessed_metadata(path).preprocessed is True


@dataclass
class _RenderArgs:
    """Stand-in for the training script's ConversationRenderArguments, defaults included."""

    conversation_field: str = "prompt"
    system_prompt: str | None = None
    assistant_message_template: str | None = None


def _artifact(**config_overrides) -> PreprocessedDatasetMetadata:
    config = PreprocessingConfig(model_name_or_path="m", **config_overrides)
    return PreprocessedDatasetMetadata(config=config.__dict__.copy())


def test_an_artifact_baked_under_another_default_is_not_rejected(caplog):
    """Rendering is baked, so the run's value is inert and the prepared value is what trained."""
    metadata = _artifact(conversation_field="conversation")
    with caplog.at_level("WARNING"):
        _verify_render_knobs(metadata, _RenderArgs())
    assert any("does not state" in record.message for record in caplog.records), (
        "the unstated mismatch must still be reported, just not fatally"
    )


def test_a_stated_render_knob_that_contradicts_the_artifact_still_raises():
    """A YAML saying conversation_field=messages over an artifact baked from 'conversation' does not do what it
    says."""
    metadata = _artifact(conversation_field="conversation")
    with pytest.raises(ValueError, match="do not match the training config"):
        _verify_render_knobs(metadata, _RenderArgs(conversation_field="messages"))


def test_a_matching_stated_knob_passes():
    metadata = _artifact(conversation_field="messages")
    _verify_render_knobs(metadata, _RenderArgs(conversation_field="messages"))


def test_a_stated_system_prompt_against_a_prompt_less_artifact_raises():
    metadata = _artifact(conversation_field="prompt", system_prompt=None)
    with pytest.raises(ValueError, match="system_prompt"):
        _verify_render_knobs(metadata, _RenderArgs(system_prompt="You are helpful."))


def _split_dict(columns: dict) -> DatasetDict:
    rows = {name: [value] for name, value in columns.items()}
    return DatasetDict({"train": Dataset.from_dict(rows), "test": Dataset.from_dict(rows)})


def test_tools_field_present_in_only_some_sources_is_allowed(caplog):
    """A tool-use corpus concatenated with plain chat: the plain source renders without tools, which is what those
    rows are."""
    with_tools = _split_dict({"prompt": [{"role": "user", "content": "hi"}], "tools": ["[]"]})
    without = _split_dict({"prompt": [{"role": "user", "content": "hi"}]})
    with caplog.at_level("WARNING"):
        _reject_tools_field_nowhere(["a", "b"], [with_tools, without], "tools")
    assert any("render without tools" in record.message for record in caplog.records)


def test_tools_field_absent_everywhere_raises():
    """Every row would silently render toolless."""
    without = _split_dict({"prompt": [{"role": "user", "content": "hi"}]})
    with pytest.raises(ValueError, match="NONE of the 2 configured dataset"):
        _reject_tools_field_nowhere(["a", "b"], [without, without], "tools")


def test_tools_field_present_everywhere_is_silent(caplog):
    with_tools = _split_dict({"prompt": [{"role": "user", "content": "hi"}], "tools": ["[]"]})
    with caplog.at_level("WARNING"):
        _reject_tools_field_nowhere(["a", "b"], [with_tools, with_tools], "tools")
    assert not [r for r in caplog.records if "tools" in r.message]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
