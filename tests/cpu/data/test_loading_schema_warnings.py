#!/usr/bin/env python3
"""Schema-normalization logging: the module loggers stay pinned at INFO, and ``_normalize_dataset_schema`` warns
once, only for columns lost from a dataset that had them."""

import json
import logging
import os
import subprocess
import sys

import pytest
from datasets import Dataset

import halo.data.sources.dataset_cache
import halo.data.sources.loading
import halo.data.sources.s3_client
import halo.data.sources.s3_dataset_store
import halo.data.sources.sharded_dataset  # noqa: F401  each module sets its logger level at import
from halo.data.sources.corpus import _normalize_dataset_schema
from tests.common.repo import REPO_ROOT

_CORPUS_LOGGER = "halo.data.sources.corpus"


@pytest.mark.parametrize(
    "name",
    [
        "halo.data.sources.loading",
        _CORPUS_LOGGER,
        "halo.data.sources.dataset_cache",
        "halo.data.sources.s3_client",
        "halo.data.sources.s3_dataset_store",
        "halo.data.sources.sharded_dataset",
    ],
)
def test_data_source_loggers_are_pinned_at_info(name):
    """``halo`` pins the root to WARNING, so an unset module logger silences the run's only record of what data
    trained. The logger's own level, since the effective one passes on any ambient root configuration."""
    assert logging.getLogger(name).level == logging.INFO


def test_schema_warning_names_only_columns_actually_lost(caplog):
    ds_with_labels = Dataset.from_dict({"conversation": ["a", "b"], "labels": ["x", "y"]})
    ds_without = Dataset.from_dict({"conversation": ["c"]})

    with caplog.at_level(logging.WARNING, logger=_CORPUS_LOGGER):
        normalized = _normalize_dataset_schema([ds_with_labels, ds_without])

    assert all(ds.column_names == ["conversation"] for ds in normalized)
    warnings = [r for r in caplog.records if "ssential" in r.getMessage()]
    assert len(warnings) == 1, [r.getMessage() for r in caplog.records]
    message = warnings[0].getMessage()
    assert "labels" in message
    for never_present in ("target", "input_ids", "attention_mask", "prompt"):
        assert never_present not in message, message


def test_schema_warning_names_the_runs_own_declared_column(caplog):
    """A custom ``conversation_field`` is pinned through the concatenation, so only a type mismatch can lose it, and
    a hardcoded tuple would not name it."""
    ds_a = Dataset.from_dict({"dialogue": [[{"role": "user", "content": "hi"}]], "row_id": [0]})
    ds_b = Dataset.from_dict({"dialogue": ["hi"], "row_id": [1]})

    with caplog.at_level(logging.WARNING, logger=_CORPUS_LOGGER):
        normalized = _normalize_dataset_schema([ds_a, ds_b], declared_columns=("dialogue",))

    assert all(ds.column_names == ["row_id"] for ds in normalized)
    warnings = [r.getMessage() for r in caplog.records if "ssential" in r.getMessage()]
    assert len(warnings) == 1, warnings
    assert "dialogue" in warnings[0], warnings[0]


def test_declared_column_absent_from_one_dataset_is_pinned_not_dropped(caplog):
    """A declared column only one entry carries is union-filled with nulls, and a pinned column is not lost."""
    ds_with = Dataset.from_dict({"conversation": ["a"], "tools": [[{"name": "f"}]]})
    ds_without = Dataset.from_dict({"conversation": ["b"]})

    with caplog.at_level(logging.WARNING, logger=_CORPUS_LOGGER):
        normalized = _normalize_dataset_schema([ds_with, ds_without], declared_columns=("tools",))

    assert all("tools" in ds.column_names for ds in normalized)
    assert normalized[1]["tools"] == [None]
    assert not [r for r in caplog.records if r.levelno >= logging.WARNING], [r.getMessage() for r in caplog.records]


def test_no_schema_warning_when_nothing_is_lost(caplog):
    ds_a = Dataset.from_dict({"conversation": ["a"]})
    ds_b = Dataset.from_dict({"conversation": ["b"]})

    with caplog.at_level(logging.WARNING, logger=_CORPUS_LOGGER):
        _normalize_dataset_schema([ds_a, ds_b])

    assert not [r for r in caplog.records if r.levelno >= logging.WARNING], [r.getMessage() for r in caplog.records]


_COLUMN_ORDER_PROBE = """
import json
from datasets import Dataset
from halo.data.sources.corpus import _normalize_dataset_schema
columns = ["prompt", "messages", "source", "id", "tools", "label", "score"]
first = Dataset.from_dict({c: ["x"] for c in columns})
second = Dataset.from_dict({c: ["y"] for c in reversed(columns)} | {"extra": ["z"]})
print(json.dumps(_normalize_dataset_schema([first, second])[0].column_names))
"""


def test_normalized_column_order_is_the_same_in_every_process():
    """Ranks ordering the columns differently key different caches for one map and each re-runs it. String-set order
    changes with the hash seed, so two seeds must agree on the first dataset's order."""
    orders = []
    for seed in ("1", "2"):
        result = subprocess.run(
            [sys.executable, "-c", _COLUMN_ORDER_PROBE],
            capture_output=True,
            text=True,
            check=True,
            cwd=REPO_ROOT,
            env={**os.environ, "PYTHONHASHSEED": seed},
        )
        orders.append(json.loads(result.stdout.strip().splitlines()[-1]))
    assert orders[0] == orders[1] == ["prompt", "messages", "source", "id", "tools", "label", "score"], orders


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
