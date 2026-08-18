#!/usr/bin/env python
"""
Tests for log_dataset_examples decoded-sample dumping.

On-disk decoded-sample writing is opt-in via the ``write_decoded_samples`` flag
(wired from the ``log_decoded_samples`` YAML field on CommonScriptArguments). It
is off by default so ordinary training runs do not emit sample files. These tests
cover the gate, file naming, the input_ids skip, the back-compat (no-kwargs)
path, DatasetDict handling, the FS-aware writer gate (the same one ``run.log``
uses, so a non-shared filesystem gets the samples on every node), and the config
default.

Run: python tests/cpu/data/test_log_dataset_examples.py
"""

import os
import tempfile

import pytest
from datasets import Dataset, DatasetDict

from src.args.common_script_args import CommonScriptArguments
from src.data.pipeline import processing
from src.data.pipeline.processing import log_dataset_examples


class FakeTokenizer:
    """Offline stand-in — log_dataset_examples only calls .decode()."""

    def decode(self, input_ids, skip_special_tokens=False):
        assert skip_special_tokens is False, "decoded samples must keep special tokens"
        return " ".join(f"t{i}" for i in input_ids)


def _ids_dataset(rows=2):
    return Dataset.from_list([{"input_ids": [1, 2, 3, 4]} for _ in range(rows)])


def _sample_path(output_dir, fname):
    return os.path.join(output_dir, "log", fname)


def test_disabled_by_default():
    with tempfile.TemporaryDirectory() as d:
        log_dataset_examples({"train": _ids_dataset()}, tokenizer=FakeTokenizer(), output_dir=d)
        assert not os.path.exists(os.path.join(d, "log")), "decoded samples written while disabled"


def test_enabled_writes_file():
    with tempfile.TemporaryDirectory() as d:
        log_dataset_examples(
            {"train": _ids_dataset()},
            num_examples=2,
            tokenizer=FakeTokenizer(),
            output_dir=d,
            write_decoded_samples=True,
        )
        p = _sample_path(d, "train_sample.txt")
        assert os.path.exists(p), "decoded samples not written when enabled"
        with open(p) as f:
            content = f.read()
        assert "=== Sample 1 (4 tokens) ===" in content
        assert "=== Sample 2 (4 tokens) ===" in content
        assert "t1 t2 t3 t4" in content, "decoded token text missing"


def test_test_split_maps_to_eval_filename():
    with tempfile.TemporaryDirectory() as d:
        log_dataset_examples(
            {"test": _ids_dataset()}, tokenizer=FakeTokenizer(), output_dir=d, write_decoded_samples=True
        )
        assert os.path.exists(_sample_path(d, "eval_sample.txt"))


def test_custom_name_filename():
    with tempfile.TemporaryDirectory() as d:
        log_dataset_examples(
            {"generate": _ids_dataset()}, tokenizer=FakeTokenizer(), output_dir=d, write_decoded_samples=True
        )
        assert os.path.exists(_sample_path(d, "generate_sample.txt"))


def test_skips_dataset_without_input_ids():
    with tempfile.TemporaryDirectory() as d:
        log_dataset_examples(
            {"meta": Dataset.from_list([{"text": "x"}])},
            tokenizer=FakeTokenizer(),
            output_dir=d,
            write_decoded_samples=True,
        )
        assert not os.path.exists(os.path.join(d, "log")), "dataset without input_ids must be skipped"


def test_backcompat_no_tokenizer_no_write():
    """Enabled with an output_dir but no tokenizer: there is nothing to decode with, so the dump
    must be skipped rather than written from raw ids."""
    with tempfile.TemporaryDirectory() as d:
        log_dataset_examples({"train": _ids_dataset()}, output_dir=d, write_decoded_samples=True)
        assert not os.path.exists(os.path.join(d, "log")), "a missing tokenizer must not produce a sample dump"


def test_datasetdict_uses_first_split():
    with tempfile.TemporaryDirectory() as d:
        dd = DatasetDict({"first": _ids_dataset(), "second": _ids_dataset()})
        log_dataset_examples({"train": dd}, tokenizer=FakeTokenizer(), output_dir=d, write_decoded_samples=True)
        assert os.path.exists(_sample_path(d, "train_sample.txt"))


def test_off_the_fs_save_rank_no_write():
    """A rank that is not the FS-aware save rank writes nothing — else N ranks stampede one path."""
    original = processing.fs_aware_save_rank
    processing.fs_aware_save_rank = lambda: False
    try:
        with tempfile.TemporaryDirectory() as d:
            log_dataset_examples(
                {"train": _ids_dataset()}, tokenizer=FakeTokenizer(), output_dir=d, write_decoded_samples=True
            )
            assert not os.path.exists(os.path.join(d, "log"))
    finally:
        processing.fs_aware_save_rank = original


def test_non_main_fs_save_rank_still_writes():
    """The write gate is the FS-aware save rank, NOT global rank 0.

    Under ``DIST_SHARED_FILESYSTEM=0`` each node's local main is the save rank and owns that node's
    ``<output_dir>/log/`` (where ``run.log`` is also teed). Gating the sample dump on global rank 0
    instead leaves every other node's log dir without the samples it is supposed to sit beside.
    """
    original_main, original_save = processing.is_global_main_process, processing.fs_aware_save_rank
    processing.is_global_main_process = lambda: False
    processing.fs_aware_save_rank = lambda: True
    try:
        with tempfile.TemporaryDirectory() as d:
            log_dataset_examples(
                {"train": _ids_dataset()}, tokenizer=FakeTokenizer(), output_dir=d, write_decoded_samples=True
            )
            assert os.path.exists(_sample_path(d, "train_sample.txt")), (
                "a non-rank-0 FS save rank must still write its node's decoded samples"
            )
    finally:
        processing.is_global_main_process = original_main
        processing.fs_aware_save_rank = original_save


def test_config_field_defaults_off():
    assert CommonScriptArguments().log_decoded_samples is False


ALL_TESTS = [
    ("disabled_by_default", test_disabled_by_default),
    ("enabled_writes_file", test_enabled_writes_file),
    ("test_split_maps_to_eval_filename", test_test_split_maps_to_eval_filename),
    ("custom_name_filename", test_custom_name_filename),
    ("skips_dataset_without_input_ids", test_skips_dataset_without_input_ids),
    ("backcompat_no_tokenizer_no_write", test_backcompat_no_tokenizer_no_write),
    ("datasetdict_uses_first_split", test_datasetdict_uses_first_split),
    ("off_the_fs_save_rank_no_write", test_off_the_fs_save_rank_no_write),
    ("non_main_fs_save_rank_still_writes", test_non_main_fs_save_rank_still_writes),
    ("config_field_defaults_off", test_config_field_defaults_off),
]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
