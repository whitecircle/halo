#!/usr/bin/env python3
"""
Tests for dataset path parsing and source detection.

These tests verify the path parsing utilities that distinguish between
S3 URIs, HuggingFace Hub IDs, and local paths.

Usage:
    python tests/data/test_path_parsing.py
"""

import sys

import pytest

from src.data.sources.paths import (
    parse_dataset_destination,
    parse_dataset_source,
    parse_s3_uri,
)


def test_parse_s3_uri():
    """Test S3 URI parsing."""
    print("Testing parse_s3_uri...")

    bucket, key = parse_s3_uri("s3://my-bucket/path/to/data")
    assert bucket == "my-bucket", f"Expected 'my-bucket', got '{bucket}'"
    assert key == "path/to/data", f"Expected 'path/to/data', got '{key}'"

    bucket, key = parse_s3_uri("s3://my-bucket/sft_datasets/merged")
    assert bucket == "my-bucket"
    assert key == "sft_datasets/merged"

    bucket, key = parse_s3_uri("s3://bucket/path/")
    assert key == "path", f"Expected 'path', got '{key}'"

    bucket, key = parse_s3_uri("s3://bucket/path//")
    assert (bucket, key) == ("bucket", "path")

    with pytest.raises(ValueError, match="s3://"):
        parse_s3_uri("not-s3-uri")

    with pytest.raises(ValueError, match="(?i)key|path"):
        parse_s3_uri("s3://bucket-only")

    with pytest.raises(ValueError, match="(?i)bucket"):
        parse_s3_uri("s3:///key")

    with pytest.raises(ValueError, match="(?i)key|path"):
        parse_s3_uri("s3://bucket/")

    print("  parse_s3_uri: PASSED")


def test_parse_dataset_source_s3():
    """Test S3 source detection."""
    print("Testing parse_dataset_source (S3)...")

    source, bucket, key = parse_dataset_source("s3://my-bucket/path/to/data")
    assert source == "s3", f"Expected 's3', got '{source}'"
    assert bucket == "my-bucket"
    assert key == "path/to/data"

    source, bucket, key = parse_dataset_source("s3://my-bucket/sft_datasets/merged")
    assert source == "s3"
    assert bucket == "my-bucket"
    assert key == "sft_datasets/merged"

    print("  S3 source detection: PASSED")


def test_parse_dataset_source_hf_hub():
    """Test HuggingFace Hub source detection."""
    print("Testing parse_dataset_source (HF Hub)...")

    source, bucket, key = parse_dataset_source("HuggingFaceH4/ultrachat_200k")
    assert source == "hf_hub", f"Expected 'hf_hub', got '{source}'"
    assert bucket is None
    assert key == "HuggingFaceH4/ultrachat_200k"

    source, bucket, key = parse_dataset_source("someorg/sft-po-dataset")
    assert source == "hf_hub"

    # The loader strips a ':config' / '@split' suffix itself, so it stays an HF Hub id here.
    for spec in ("openai/gsm8k:main", "HuggingFaceH4/ultrachat_200k@train_sft"):
        source, _, _ = parse_dataset_source(spec)
        assert source == "hf_hub", f"{spec!r} should be hf_hub, got {source!r}"

    source, _, _ = parse_dataset_source("org/sub/dataset")
    assert source == "local", f"3-segment path should be local, got {source!r}"

    # A leading '.' in either segment disqualifies an otherwise Hub-shaped id.
    source, _, _ = parse_dataset_source(".hidden/dataset")
    assert source == "local"

    print("  HF Hub source detection: PASSED")


def test_parse_dataset_source_local():
    """Test local path source detection."""
    print("Testing parse_dataset_source (local)...")

    source, bucket, key = parse_dataset_source("/path/to/dataset")
    assert source == "local", f"Expected 'local', got '{source}'"
    assert bucket is None
    assert key == "/path/to/dataset"

    source, bucket, key = parse_dataset_source("./relative/path")
    assert source == "local"
    assert key == "./relative/path"

    source, bucket, key = parse_dataset_source("../parent/path")
    assert source == "local"

    source, bucket, key = parse_dataset_source("file:///absolute/path")
    assert source == "local"
    assert key == "/absolute/path"

    source, bucket, key = parse_dataset_source("data/train.jsonl")
    assert source == "local", f"Expected 'local', got '{source}'"

    source, bucket, key = parse_dataset_source("dataset.parquet")
    assert source == "local"

    source, bucket, key = parse_dataset_source("dataset")
    assert source == "local", f"Bare single word should be 'local', got '{source}'"

    print("  Local source detection: PASSED")


def test_parse_dataset_destination_bare_path_is_local():
    """OUTPUT classification must never guess the Hub: a relative path that happens to look like
    'org/name' (e.g. --output preprocessed/my_dataset) is a local directory, not an upload target.
    The input-side parse_dataset_source keeps its Hub guess — reads are harmless, writes are not."""
    for path in ("preprocessed/my_dataset", "a/b", "dataset", "out/dir/deep"):
        dest, bucket, key = parse_dataset_destination(path)
        assert dest == "local", f"{path!r} classified {dest!r}; outputs must default to local"
        assert bucket is None
        assert key == path

    # The same bare "a/b" IS hf_hub on the input side — the two classifiers intentionally differ.
    assert parse_dataset_source("a/b")[0] == "hf_hub"


def test_parse_dataset_destination_explicit_schemes():
    assert parse_dataset_destination("hf://org/name") == ("hf_hub", None, "org/name")
    assert parse_dataset_destination("s3://bucket/path/to/out") == ("s3", "bucket", "path/to/out")
    assert parse_dataset_destination("/abs/path") == ("local", None, "/abs/path")
    assert parse_dataset_destination("file:///abs/path") == ("local", None, "/abs/path")

    with pytest.raises(ValueError, match="hf://org/name"):
        parse_dataset_destination("hf://only-org")
    with pytest.raises(ValueError, match="hf://org/name"):
        parse_dataset_destination("hf://org/name/extra")


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
