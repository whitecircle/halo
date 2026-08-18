#!/usr/bin/env python
"""Hard corner cases of the data-loading and cache seams.

Each test here is an edge a healthy run never shows but a degraded one does: a corrupt mirror on a
flaky disk, a read-only cache volume during an outage, a dataset at the bucket root, a bare
``Dataset`` push (single load gate), a dataset name ending in a hyphen (staging-sibling boundary), a
pathological object in a map closure, and the arrow-cache reuse/invalidation contract itself —
the map must run exactly once per identity and re-run on any identity change.

    python tests/cpu/data/test_dataloading_corner_cases.py
"""

import json
import os
from unittest.mock import patch

import pytest
from datasets import Dataset
from transformers import ProcessorMixin

from src.data.pipeline.preprocessed_metadata import is_preprocessed_dataset
from src.data.pipeline.processing import _get_closure_fingerprint, _tokenizer_identity, coordinated_map
from src.data.sources.s3_client import (
    _PUSH_COMPLETE_MARKER,
    _control_json_mirror_path,
    read_control_json_with_cache,
)
from src.data.sources.sharded_dataset import ShardedDatasetLoader
from tests.cpu.data.test_s3_staged_push import (
    BUCKET,
    FakeBoto,
    _client,
    _mock_dataset,
    _staging_keys,
)

OUTAGE = OSError("Could not connect to the endpoint URL")


@pytest.fixture()
def cache_root(tmp_path):
    with patch("src.data.sources.s3_client.HALO_S3_DATASET_CACHE_DIR", str(tmp_path)):
        yield tmp_path


# Mirror-cache degradation


def _seed_mirror(bucket: str, key: str, payload) -> str:
    path = _control_json_mirror_path(bucket, key)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(payload if isinstance(payload, str) else json.dumps(payload))
    return path


def test_corrupt_mirror_during_outage_degrades_loud_not_crash(cache_root):
    """A truncated mirror (disk corruption — the atomic publish cannot produce one) plus an outage:
    the probes must degrade with their normal warnings, never crash a rank ahead of the consensus
    all-reduce."""
    _seed_mirror("b", "d/train/shard_index.json", '{"num_shards": 4, "sha')
    with patch("src.data.sources.s3_client.read_json_from_s3", side_effect=OUTAGE):
        assert ShardedDatasetLoader.is_sharded_dataset("s3://b/d") is False

    _seed_mirror("b", "raw/metadata.json", "not json at all")
    with patch("src.data.sources.s3_client.read_json_from_s3", side_effect=OUTAGE):
        assert is_preprocessed_dataset("s3://b/raw") is False


def test_absence_contract_survives_a_read_only_cache_volume(cache_root, monkeypatch):
    """The mirror drop on a live 404 must stay best-effort — on a read-only volume the unlink's
    PermissionError must not replace the FileNotFoundError the callers key on."""
    _seed_mirror("b", "d/metadata.json", {"preprocessed": True})

    def _read_only_unlink(path):
        raise PermissionError(30, "Read-only file system")

    monkeypatch.setattr("src.data.sources.s3_client.os.unlink", _read_only_unlink)
    with patch("src.data.sources.s3_client.read_json_from_s3", side_effect=FileNotFoundError("404")):
        with pytest.raises(FileNotFoundError):
            read_control_json_with_cache("b", "d/metadata.json")


def test_distinct_control_files_never_share_a_mirror(cache_root):
    with patch("src.data.sources.s3_client.read_json_from_s3", return_value={"split": "train"}):
        read_control_json_with_cache("b", "d/train/shard_index.json")
    with patch("src.data.sources.s3_client.read_json_from_s3", return_value={"split": "test"}):
        read_control_json_with_cache("b", "d/test/shard_index.json")
    with patch("src.data.sources.s3_client.read_json_from_s3", side_effect=OUTAGE):
        assert read_control_json_with_cache("b", "d/train/shard_index.json") == {"split": "train"}
        assert read_control_json_with_cache("b", "d/test/shard_index.json") == {"split": "test"}


# Staged push at the namespace edges


def _fake_bare_dataset_save(fake: FakeBoto, rows_tag: bytes = b"new"):
    """A save_to_disk side effect writing a bare-Dataset tree (single root state.json gate)."""

    def _save(uri, storage_options=None, **kwargs):
        prefix = uri[len(f"s3://{BUCKET}/") :]
        for rel in ("data-00000-of-00001.arrow", "state.json", "dataset_info.json"):
            fake._write(f"{prefix}/{rel}", rows_tag + b":" + rel.encode())

    return _save


def test_bucket_root_dataset_stages_and_promotes():
    """A dataset keyed at the bucket root has no parent prefix — the staging sibling must still be
    a valid dot-prefixed key, promote, and sweep."""
    client, fake = _client()
    client.push_dataset(_mock_dataset(_fake_bare_dataset_save(fake)), "rootset")
    assert "rootset/state.json" in fake.objects
    assert not _staging_keys(fake)
    assert all(not k.startswith("/") for k in fake.objects), "no key may begin with a slash"


def test_bare_dataset_promote_gates_on_its_root_state_json():
    """A bare Dataset's only load gate is the root state.json — an interrupted promote must leave
    the destination without it (loudly unreadable), never a readable old/new mixture."""
    client, fake = _client()
    for rel in ("data-00000-of-00001.arrow", "state.json", "dataset_info.json"):
        fake._write(f"bare/{rel}", b"old:" + rel.encode())

    original_copy = fake.copy
    calls = {"n": 0}

    def _failing_copy(CopySource, Bucket, Key, Config=None):
        calls["n"] += 1
        if calls["n"] == 2:
            raise OSError("mid-promote")
        original_copy(CopySource, Bucket, Key, Config)

    fake.copy = _failing_copy
    with pytest.raises(OSError, match="mid-promote"):
        client.push_dataset(_mock_dataset(_fake_bare_dataset_save(fake)), "bare")

    assert "bare/state.json" not in fake.objects, "the root gate must be deleted first and copied last"
    assert client.exists("bare"), "the destination must never empty out"
    assert any(k.endswith(_PUSH_COMPLETE_MARKER) for k in fake.objects), "the sealed staging must survive"

    fake.copy = original_copy
    client.push_dataset(_mock_dataset(_fake_bare_dataset_save(fake, b"new2")), "bare")
    ordered_ops = [key for op, key in fake.ops if op == "copy" and key.startswith("bare/")]
    assert ordered_ops[-1] == "bare/state.json", "the root gate must be the last copy of a bare-Dataset promote"


def test_trailing_hyphen_dataset_name_keeps_sibling_isolation():
    """'x-' stages as .staging-x--<hex>; scanning for sibling 'x' sees remainder '-<hex>/', which
    the 8-hex anchor must reject."""
    client, fake = _client()
    hyphen_staging = client._new_staging_key("datasets/x-")
    _fake_bare_dataset_save(fake)(f"s3://{BUCKET}/{hyphen_staging}")
    fake.put_object(Bucket=BUCKET, Key=f"{hyphen_staging}/{_PUSH_COMPLETE_MARKER}", Body=b"")

    assert client._latest_complete_staging("datasets/x") is None
    assert client._latest_complete_staging("datasets/x-") == hyphen_staging


# Fingerprint pathology


def test_raising_attribute_object_in_a_closure_lands_in_the_skip_warning():
    """An object whose attribute access raises (a lazy proxy, a property with a guard) must fall to
    the unfingerprintable-cell skip, not crash cache naming for the whole run."""

    class Booby:
        @property
        def tokenizer(self):
            raise RuntimeError("not initialized yet")

        name_or_path = property(tokenizer.fget)

    booby = Booby()

    def make_fn(obj):
        def fn(row):
            return obj

        return fn

    assert isinstance(_get_closure_fingerprint(make_fn(booby)), str)


def test_nested_processor_descends_one_level_only():
    """A processor whose .tokenizer is itself processor-like must not recurse (or loop): identity
    comes from its own template, and two different outer templates still key apart."""

    class Inner:
        tokenizer = None
        chat_template = "inner"

    class Outer(ProcessorMixin):
        def __init__(self, template):
            self.tokenizer = Inner()
            self.chat_template = template

    sig_a = _tokenizer_identity(Outer("A"))
    sig_b = _tokenizer_identity(Outer("B"))
    assert sig_a is not None and sig_a != sig_b


# The arrow-cache contract itself: run once per identity, re-run on any identity change.


def _counting_fn(row, tokenizer=None, marker_dir=None):
    with open(os.path.join(marker_dir, f"call_{os.getpid()}_{len(os.listdir(marker_dir))}"), "w") as f:
        f.write("x")
    return {"n": row["value"] + 1}


class _TokStub:
    def __init__(self, eos):
        self.name_or_path = "org/model"
        self.vocab_size = 8
        self.chat_template = "t"
        self.eos_token_id = eos
        self.bos_token_id = 1
        self.pad_token_id = 0
        self.padding_side = "right"
        self.truncation_side = "right"

    def __len__(self):
        return 8


def _run_map(ds, tmp_path, marker_dir, eos):
    with patch.dict(os.environ, {"HF_DATASETS_CACHE": str(tmp_path / "hf_cache")}):
        return coordinated_map(
            ds,
            _counting_fn,
            desc="corner cache probe",
            num_proc=1,
            fn_kwargs={"tokenizer": _TokStub(eos), "marker_dir": str(marker_dir)},
        )


def test_map_runs_once_per_identity_and_reruns_on_special_token_change(tmp_path):
    """The end-to-end cache contract: identical identity → the second call reuses the arrow cache
    and never re-executes the map; an in-vocab eos change → full re-execution."""
    ds = Dataset.from_dict({"value": [1, 2, 3]})
    marker_dir = tmp_path / "calls"
    marker_dir.mkdir()

    result = _run_map(ds, tmp_path, marker_dir, eos=2)
    assert result["n"] == [2, 3, 4]
    calls_after_first = len(os.listdir(marker_dir))
    assert calls_after_first > 0, "the first map must actually execute"

    result2 = _run_map(ds, tmp_path, marker_dir, eos=2)
    assert result2["n"] == [2, 3, 4]
    assert len(os.listdir(marker_dir)) == calls_after_first, (
        "an identical identity must serve the arrow cache without re-executing the map"
    )

    _run_map(ds, tmp_path, marker_dir, eos=5)
    assert len(os.listdir(marker_dir)) > calls_after_first, (
        "an in-vocab special-token change must invalidate the cache and re-execute"
    )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
