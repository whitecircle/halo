#!/usr/bin/env python
"""Startup-blocking guards on the preprocessed-dataset path must not fire on legitimate corpora.

Every check here runs during dataset loading — per rank, in front of a consensus all-reduce — so an
over-strict verdict is not a warning the operator reads later, it is a job that dies (or hangs) at
step 0:

* ``metadata.json`` is a common file name. A hub dataset shipping its own must load RAW, not be
  judged against this build's stamp schema (which would raise on every rank that managed to read it).
* An artifact prepared before a default moved carries the OLD default in its recorded render config.
  Comparing it against the training args' own default rejects every such artifact over a knob that
  cannot affect a preprocessed run at all — only a knob the YAML actually states is a claim.
* A dataset LIST is a mixed corpus by design: ``tools_field`` present in some sources and absent in
  others is the normal tool-use + plain-chat mix, not a typo.

    python tests/cpu/data/test_preprocessed_metadata_compat.py
"""

import json
import os
from dataclasses import dataclass

import pytest
from accelerate import PartialState
from datasets import Dataset, DatasetDict

PartialState()  # sources.loading warns through accelerate's logger

from src.data.pipeline.preprocessed_metadata import (
    PreprocessedDatasetMetadata,
    PreprocessingConfig,
    _validate_render_compatibility,
    is_preprocessed_dataset,
    load_preprocessed_metadata,
)
from src.data.shard_index import IncompatiblePreprocessedDataset
from src.data.sources.loading import _require_tools_field_somewhere
from src.data.sources.paths import METADATA_FILE


def _write_metadata(directory, payload: dict) -> str:
    os.makedirs(directory, exist_ok=True)
    with open(os.path.join(directory, METADATA_FILE), "w") as f:
        json.dump(payload, f)
    return str(directory)


# --- a foreign metadata.json is a raw dataset, not a broken preprocessed one ---


def test_a_foreign_metadata_json_loads_raw(tmp_path):
    """A hub dataset's own metadata.json ({"license": ...}) has no 'preprocessed' field. Judging it
    against the stamp schema raised on every rank that read it — right before the consensus
    all-reduce, so ranks whose read flaked blocked there until the watchdog fired."""
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
    """Not a downgrade to raw: the dataset IS pre-tokenized, so reporting False would silently
    re-tokenize baked rows. Reporting True keeps every rank on the same branch and lets the LOAD
    raise the version error on all of them together."""
    path = _write_metadata(tmp_path / "future", {"preprocessed": True, "version": "9.9"})
    assert is_preprocessed_dataset(path) is True
    with pytest.raises(IncompatiblePreprocessedDataset, match="version"):
        load_preprocessed_metadata(path)


def test_our_current_stamp_is_detected(tmp_path):
    path = _write_metadata(tmp_path / "prepared", PreprocessedDatasetMetadata().to_dict())
    assert is_preprocessed_dataset(path) is True
    assert load_preprocessed_metadata(path).preprocessed is True


# --- render-knob compatibility: only what the YAML STATES is a claim about the artifact ---


@dataclass
class _RenderArgs:
    """Stand-in for the training script's ConversationRenderArguments, defaults included."""

    conversation_field: str = "prompt"
    system_prompt: str | None = None
    assistant_message_template: str | None = None


def _artifact(**config_overrides) -> PreprocessedDatasetMetadata:
    config = PreprocessingConfig(model_name_or_path="m", **config_overrides)
    return PreprocessedDatasetMetadata(config=config.__dict__.copy())


def test_an_artifact_prepared_under_an_older_default_is_not_rejected(caplog):
    """The prep-side --conversation-field default moved (conversation -> prompt). An artifact
    prepared before that plus a YAML that states nothing must still train: rendering is baked, so
    the run's value is inert and the prepared value is what trained."""
    metadata = _artifact(conversation_field="conversation")
    with caplog.at_level("WARNING"):
        _validate_render_compatibility(metadata, _RenderArgs())
    assert any("does not state" in record.message for record in caplog.records), (
        "the grandfathered mismatch must still be reported, just not fatally"
    )


def test_a_stated_render_knob_that_contradicts_the_artifact_still_raises():
    """The check's real job: a YAML that SAYS conversation_field=messages while the artifact was
    baked from 'conversation' is a config that does not do what it says."""
    metadata = _artifact(conversation_field="conversation")
    with pytest.raises(ValueError, match="do not match the training config"):
        _validate_render_compatibility(metadata, _RenderArgs(conversation_field="messages"))


def test_a_matching_stated_knob_passes():
    metadata = _artifact(conversation_field="messages")
    _validate_render_compatibility(metadata, _RenderArgs(conversation_field="messages"))


def test_a_stated_system_prompt_against_a_prompt_less_artifact_raises():
    metadata = _artifact(conversation_field="prompt", system_prompt=None)
    with pytest.raises(ValueError, match="system_prompt"):
        _validate_render_compatibility(metadata, _RenderArgs(system_prompt="You are helpful."))


# --- tools_field is a claim about the dataset LIST, not about each source ---


def _split_dict(columns: dict) -> DatasetDict:
    rows = {name: [value] for name, value in columns.items()}
    return DatasetDict({"train": Dataset.from_dict(rows), "test": Dataset.from_dict(rows)})


def test_tools_field_present_in_only_some_sources_is_allowed(caplog):
    """A tool-use corpus concatenated with plain chat: the plain source renders without tools,
    which is what those rows are. Rejecting the run makes the mixture unexpressible."""
    with_tools = _split_dict({"prompt": [{"role": "user", "content": "hi"}], "tools": ["[]"]})
    without = _split_dict({"prompt": [{"role": "user", "content": "hi"}]})
    with caplog.at_level("WARNING"):
        _require_tools_field_somewhere(["a", "b"], [with_tools, without], "tools")
    assert any("render without tools" in record.message for record in caplog.records)


def test_tools_field_absent_everywhere_raises():
    """A knob no source can honour is a typo — and the failure it would otherwise cause is silent
    (every row renders toolless), so it has to be loud here."""
    without = _split_dict({"prompt": [{"role": "user", "content": "hi"}]})
    with pytest.raises(ValueError, match="NONE of the 2 configured dataset"):
        _require_tools_field_somewhere(["a", "b"], [without, without], "tools")


def test_tools_field_present_everywhere_is_silent(caplog):
    with_tools = _split_dict({"prompt": [{"role": "user", "content": "hi"}], "tools": ["[]"]})
    with caplog.at_level("WARNING"):
        _require_tools_field_somewhere(["a", "b"], [with_tools, with_tools], "tools")
    assert not [r for r in caplog.records if "tools" in r.message]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
