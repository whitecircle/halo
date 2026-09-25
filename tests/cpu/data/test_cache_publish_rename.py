#!/usr/bin/env python
"""A stale cache tree that cannot be moved aside must fail the re-download, not be served.

``publish_cached_download`` renames the stale tree aside, then renames the fresh download into its
place. Only an absent tree means "nothing to replace". Reading any other failure to move it (a
permission or busy error) that way would leave the stale tree in place: the second rename then fails
as if another writer had published first, the fresh download is discarded and the stale rows are
served.

Run: pytest tests/cpu/data/test_cache_publish_rename.py
"""

import os

import pytest

from src.data.sources import dataset_cache
from src.data.sources.dataset_cache import DOWNLOAD_COMPLETE_MARKER, _write_download_marker, publish_cached_download


def _write_tree(path: str, payload: str) -> None:
    os.makedirs(path, exist_ok=True)
    with open(os.path.join(path, "data.txt"), "w") as fh:
        fh.write(payload)


def _publish(tmp_path, fingerprint: str | None) -> bool:
    return publish_cached_download(
        str(tmp_path / "cache"),
        str(tmp_path / "cache.lock"),
        "s3://bucket/key",
        lambda tmp: _write_tree(tmp, "fresh"),
        live_fingerprint=fingerprint,
        fresh_fingerprint=lambda: fingerprint,
    )


def test_a_stale_tree_that_cannot_be_moved_aside_raises(monkeypatch, tmp_path):
    cache = str(tmp_path / "cache")
    _write_tree(cache, "stale")
    _write_download_marker(os.path.join(cache, DOWNLOAD_COMPLETE_MARKER), "s3://bucket/key", "old")

    real_rename = os.rename

    def rename(src, dst):
        if src == cache:
            raise PermissionError(13, "Permission denied", src)
        real_rename(src, dst)

    monkeypatch.setattr(dataset_cache.os, "rename", rename)

    with pytest.raises(PermissionError):
        _publish(tmp_path, "new")
    assert not list(tmp_path.glob("cache.tmp-*")), "the refused download must not stay on the cache volume"


def test_an_absent_tree_is_published_fresh(tmp_path):
    assert _publish(tmp_path, "new") is False
    with open(tmp_path / "cache" / "data.txt") as fh:
        assert fh.read() == "fresh"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
