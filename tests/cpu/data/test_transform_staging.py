#!/usr/bin/env python
"""A coordinated whole-dataset transform stages its intermediate maps beside the published copy.

``coordinated_dataset_transform`` runs TRL's preparation with HF's map cache off. ``datasets`` then
writes every map over an on-disk input into one process-lifetime ``$TMPDIR/hf_datasets-*`` directory
and keeps each intermediate there until the process exits — on the root filesystem when a run moved
only ``HF_DATASETS_CACHE`` to the large volume. Proven with real ``datasets`` maps over an on-disk
source, on success and on a failing transform: no map writes under ``TMPDIR``, every map writes one
staging directory inside ``HF_DATASETS_CACHE``, and nothing but the published copy survives.

    python tests/cpu/data/test_transform_staging.py
"""

import os

import datasets
import pytest
from datasets import Dataset, load_from_disk
from datasets import fingerprint as hf_fingerprint

from src.data.pipeline.processing import coordinated_dataset_transform

ROWS = 8
MAPS = 3


@pytest.fixture
def dirs(tmp_path, monkeypatch):
    """``(source, tmpdir, cache)``: an on-disk source, an empty TMPDIR, and the datasets cache."""
    tmpdir, cache = tmp_path / "tmpdir", tmp_path / "hf_datasets"
    tmpdir.mkdir()
    monkeypatch.setenv("TMPDIR", str(tmpdir))
    monkeypatch.setenv("HF_DATASETS_CACHE", str(cache))
    # No directory opened yet: an unredirected uncached map would open its own under this TMPDIR.
    monkeypatch.setattr(hf_fingerprint, "_TEMP_DIR_FOR_TEMP_CACHE_FILES", None)
    Dataset.from_dict({"x": list(range(ROWS))}).save_to_disk(str(tmp_path / "source"))
    return load_from_disk(str(tmp_path / "source")), tmpdir, cache


def _mapping_transform(source: Dataset, written: list[str], *, fail: bool = False):
    """``MAPS`` chained maps over ``source``, recording the file each one wrote."""

    def transform() -> Dataset:
        rows = source
        for _ in range(MAPS):
            rows = rows.map(lambda row: {"x": row["x"] + 1})
            written.extend(entry["filename"] for entry in rows.cache_files)
        if fail:
            raise ValueError("preparation failed")
        return rows

    return transform


def _unlocked_entries(cache) -> list[str]:
    """What the datasets cache holds besides the publish lock (which filelock may remove on release)."""
    return sorted(name for name in os.listdir(cache) if not name.endswith(".lock"))


def _assert_staged_and_removed(written: list[str], tmpdir, cache) -> None:
    assert len(written) == MAPS, f"every map over an on-disk input writes a file: {written}"
    staging = {os.path.dirname(path) for path in written}
    assert len(staging) == 1, f"the maps of one transform share one staging directory: {staging}"
    (staging_dir,) = staging
    assert os.path.dirname(staging_dir) == str(cache), f"staged outside HF_DATASETS_CACHE: {staging_dir}"
    assert os.path.basename(staging_dir).startswith("transform-"), staging_dir
    assert not os.path.exists(staging_dir), f"the staging directory outlived the transform: {staging_dir}"
    assert list(tmpdir.iterdir()) == [], f"the maps wrote under TMPDIR: {list(tmpdir.iterdir())}"
    assert datasets.is_caching_enabled(), "the map cache stays off past the transform"
    assert hf_fingerprint._TEMP_DIR_FOR_TEMP_CACHE_FILES is None, "datasets' own temp directory was not restored"


def test_a_transform_stages_its_maps_in_the_datasets_cache_and_keeps_only_the_published_copy(dirs):
    source, tmpdir, cache = dirs
    written: list[str] = []

    prepared = coordinated_dataset_transform(source, _mapping_transform(source, written), "staging", {})

    assert prepared["x"] == [x + MAPS for x in range(ROWS)]
    _assert_staged_and_removed(written, tmpdir, cache)
    assert _unlocked_entries(cache) == [prepared._toolkit_cache_key], os.listdir(cache)


def test_a_failing_transform_leaves_no_staged_maps_behind(dirs):
    source, tmpdir, cache = dirs
    written: list[str] = []

    with pytest.raises(ValueError, match="preparation failed"):
        coordinated_dataset_transform(source, _mapping_transform(source, written, fail=True), "staging", {})

    _assert_staged_and_removed(written, tmpdir, cache)
    assert _unlocked_entries(cache) == [], os.listdir(cache)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
