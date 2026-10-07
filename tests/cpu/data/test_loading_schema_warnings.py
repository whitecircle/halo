#!/usr/bin/env python3
"""Schema-normalization logging: one accurate warning, and no module-level INFO silencing.

Two properties pinned here:

* the module logger must not be pinned to WARNING — that silences its load-bearing INFO lines (which
  columns were kept, split sizes after normalization) on every run; the same holds for the S3 and
  shard-cache modules beside it, whose INFO lines are the run's only record of what data moved;
* ``_normalize_dataset_schema`` warning once per "essential" column absent from the common set gives
  five spurious warnings on a perfectly normal multi-dataset run, naming columns NO dataset ever had.
  The warning must fire once, and only for columns actually lost from a dataset that had them.

Run: pytest tests/cpu/data/test_loading_schema_warnings.py
"""

import json
import logging
import os
import subprocess
import sys
from pathlib import Path

import pytest
from accelerate import PartialState
from datasets import Dataset

# The module logs through the accelerate logger, which requires an initialized state.
PartialState()

import src.data.sources.dataset_cache
import src.data.sources.s3_client
import src.data.sources.sharded_dataset  # noqa: F401  each module sets its logger level at import
from src.data.sources.loading import _normalize_dataset_schema

_LOADING_LOGGER = "src.data.sources.loading"


@pytest.mark.parametrize(
    "name",
    [
        _LOADING_LOGGER,
        "src.data.sources.dataset_cache",
        "src.data.sources.s3_client",
        "src.data.sources.sharded_dataset",
    ],
)
def test_data_source_loggers_are_pinned_at_info(name):
    """``src`` pins the root to WARNING, so a module logger left unset silences the column-drop
    notice, the cache and transfer records and each rank's loaded-split size — the run's only record
    of what data actually trained.

    The logger's OWN level, not the effective one: reading the inherited level would pass on any
    ambient root configuration (pytest's, or a sibling test's ``caplog``) and certify nothing about
    the module."""
    assert logging.getLogger(name).level == logging.INFO


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
    is a type mismatch across the entries — and that must be named. A hardcoded tuple would know only
    the default spellings, so a column named by the YAML would vanish without a word.
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


_COLUMN_ORDER_PROBE = """
import json
from accelerate import PartialState
PartialState()
from datasets import Dataset
from src.data.sources.loading import _normalize_dataset_schema
columns = ["prompt", "messages", "source", "id", "tools", "label", "score"]
first = Dataset.from_dict({c: ["x"] for c in columns})
second = Dataset.from_dict({c: ["y"] for c in reversed(columns)} | {"extra": ["z"]})
print(json.dumps(_normalize_dataset_schema([first, second])[0].column_names))
"""


def test_normalized_column_order_is_the_same_in_every_process():
    """Every rank normalizes on its own, and the column order it lands on reaches each map that keys
    or removes by ``column_names``: ranks ordering the columns differently key different caches for
    one map, so each re-runs it instead of reading the load rank's. String-set order changes with the
    per-process hash seed, so two seeds must agree — on the first dataset's order."""
    orders = []
    for seed in ("1", "2"):
        result = subprocess.run(
            [sys.executable, "-c", _COLUMN_ORDER_PROBE],
            capture_output=True,
            text=True,
            check=True,
            cwd=Path(__file__).resolve().parents[3],
            env={**os.environ, "PYTHONHASHSEED": seed},
        )
        orders.append(json.loads(result.stdout.strip().splitlines()[-1]))
    assert orders[0] == orders[1] == ["prompt", "messages", "source", "id", "tools", "label", "score"], orders


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
