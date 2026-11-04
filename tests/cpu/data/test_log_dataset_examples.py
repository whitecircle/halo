#!/usr/bin/env python
"""``log_dataset_examples``' opt-in decoded-sample dump (``write_decoded_samples``): the gate, file naming, the
skips, DatasetDict handling and the FS-aware writer gate."""

import os
import tempfile

import pytest
from datasets import Dataset, DatasetDict

from halo.configs.script_arguments import CommonScriptArguments
from halo.entrypoints import data_setup
from halo.entrypoints.data_setup import log_dataset_examples


class FakeTokenizer:
    """Offline stand-in: log_dataset_examples only calls .decode()."""

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


def test_no_tokenizer_no_write():
    with tempfile.TemporaryDirectory() as d:
        log_dataset_examples({"train": _ids_dataset()}, output_dir=d, write_decoded_samples=True)
        assert not os.path.exists(os.path.join(d, "log")), "a missing tokenizer must not produce a sample dump"


def test_datasetdict_uses_first_split():
    with tempfile.TemporaryDirectory() as d:
        dd = DatasetDict({"first": _ids_dataset(), "second": _ids_dataset()})
        log_dataset_examples({"train": dd}, tokenizer=FakeTokenizer(), output_dir=d, write_decoded_samples=True)
        assert os.path.exists(_sample_path(d, "train_sample.txt"))


def test_off_the_fs_save_rank_no_write():
    """Otherwise N ranks stampede one path."""
    original = data_setup.is_fs_save_rank
    data_setup.is_fs_save_rank = lambda: False
    try:
        with tempfile.TemporaryDirectory() as d:
            log_dataset_examples(
                {"train": _ids_dataset()}, tokenizer=FakeTokenizer(), output_dir=d, write_decoded_samples=True
            )
            assert not os.path.exists(os.path.join(d, "log"))
    finally:
        data_setup.is_fs_save_rank = original


def test_non_main_fs_save_rank_still_writes():
    """Under ``DIST_SHARED_FILESYSTEM=0`` each node's local main owns that node's ``<output_dir>/log/``; gating on
    global rank 0 leaves other nodes without the samples."""
    original_main, original_save = data_setup.is_global_main_process, data_setup.is_fs_save_rank
    data_setup.is_global_main_process = lambda: False
    data_setup.is_fs_save_rank = lambda: True
    try:
        with tempfile.TemporaryDirectory() as d:
            log_dataset_examples(
                {"train": _ids_dataset()}, tokenizer=FakeTokenizer(), output_dir=d, write_decoded_samples=True
            )
            assert os.path.exists(_sample_path(d, "train_sample.txt")), (
                "a non-rank-0 FS save rank must still write its node's decoded samples"
            )
    finally:
        data_setup.is_global_main_process = original_main
        data_setup.is_fs_save_rank = original_save


def test_config_field_defaults_off():
    assert CommonScriptArguments().log_decoded_samples is False


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
