#!/usr/bin/env python3
"""Schema-normalization logging: one accurate warning, and no module-level INFO silencing.

Two properties pinned here:

* the module logger must not be pinned to WARNING — that silences its load-bearing INFO lines (which
  columns were kept, split sizes after normalization) on every run;
* ``_normalize_dataset_schema`` warning once per "essential" column absent from the common set gives
  five spurious warnings on a perfectly normal multi-dataset run, naming columns NO dataset ever had.
  The warning must fire once, and only for columns actually lost from a dataset that had them.

Run: pytest tests/cpu/data/test_loading_schema_warnings.py
"""

import logging
import sys

import pytest
from accelerate import PartialState
from datasets import Dataset

# The module logs through the accelerate logger, which requires an initialized state.
PartialState()

from src.data.sources.loading import _normalize_dataset_schema

_LOADING_LOGGER = "src.data.sources.loading"


def test_loading_logger_is_pinned_at_info():
    """The WARNING pin silenced the column-drop notice and the post-normalization dataset columns —
    the run's only record of what data actually trained.

    The logger's OWN level, not the effective one: accelerate's ``get_logger(..., log_level="INFO")``
    sets it on this logger, so reading the inherited level would pass on any ambient root
    configuration (pytest's, or a sibling test's ``caplog``) and certify nothing about this module."""
    assert logging.getLogger(_LOADING_LOGGER).level == logging.INFO


def test_schema_warning_names_only_columns_actually_lost(caplog):
    """Two datasets sharing 'conversation', one also carrying 'labels': exactly one warning, naming
    'labels' and none of the essential columns no dataset ever had."""
    ds_with_labels = Dataset.from_dict({"conversation": ["a", "b"], "labels": ["x", "y"]})
    ds_without = Dataset.from_dict({"conversation": ["c"]})

    with caplog.at_level(logging.WARNING, logger=_LOADING_LOGGER):
        normalized = _normalize_dataset_schema([ds_with_labels, ds_without])

    assert all(ds.column_names == ["conversation"] for ds in normalized)
    warnings = [r for r in caplog.records if "ssential" in r.getMessage()]
    assert len(warnings) == 1, [r.getMessage() for r in caplog.records]
    message = warnings[0].getMessage()
    assert "labels" in message
    for never_present in ("target", "input_ids", "attention_mask", "prompt"):
        assert never_present not in message, message


def test_schema_warning_names_the_runs_own_declared_column(caplog):
    """The warned-about set follows the run's configuration, not a fixed literal.

    A custom ``conversation_field`` is pinned through the concatenation, so the only way to lose it
    is a type mismatch across the entries — and that must be named. The hardcoded tuple knew only
    the default spellings, so a column named by the YAML vanished without a word.
    """
    ds_a = Dataset.from_dict({"dialogue": [[{"role": "user", "content": "hi"}]], "row_id": [0]})
    ds_b = Dataset.from_dict({"dialogue": ["hi"], "row_id": [1]})

    with caplog.at_level(logging.WARNING, logger=_LOADING_LOGGER):
        normalized = _normalize_dataset_schema([ds_a, ds_b], declared_columns=("dialogue",))

    assert all(ds.column_names == ["row_id"] for ds in normalized)
    warnings = [r.getMessage() for r in caplog.records if "ssential" in r.getMessage()]
    assert len(warnings) == 1, warnings
    assert "dialogue" in warnings[0], warnings[0]


def test_declared_column_absent_from_one_dataset_is_pinned_not_dropped(caplog):
    """A declared column only one entry carries is union-filled with nulls instead of being dropped
    from all of them — and a pinned column is not "lost", so it must not warn either."""
    ds_with = Dataset.from_dict({"conversation": ["a"], "tools": [[{"name": "f"}]]})
    ds_without = Dataset.from_dict({"conversation": ["b"]})

    with caplog.at_level(logging.WARNING, logger=_LOADING_LOGGER):
        normalized = _normalize_dataset_schema([ds_with, ds_without], declared_columns=("tools",))

    assert all("tools" in ds.column_names for ds in normalized)
    assert normalized[1]["tools"] == [None]
    assert not [r for r in caplog.records if r.levelno >= logging.WARNING], [r.getMessage() for r in caplog.records]


def test_no_schema_warning_when_nothing_is_lost(caplog):
    """The normal multi-dataset run (identical schemas) must warn about nothing — a per-column
    loop fires five times here, training operators to ignore this module's warnings."""
    ds_a = Dataset.from_dict({"conversation": ["a"]})
    ds_b = Dataset.from_dict({"conversation": ["b"]})

    with caplog.at_level(logging.WARNING, logger=_LOADING_LOGGER):
        _normalize_dataset_schema([ds_a, ds_b])

    assert not [r for r in caplog.records if r.levelno >= logging.WARNING], [r.getMessage() for r in caplog.records]


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
