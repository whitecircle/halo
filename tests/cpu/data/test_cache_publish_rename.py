#!/usr/bin/env python
"""The download cache's publish: a stale tree that cannot be moved aside fails the re-download, since reading a
permission or busy error as an absent tree would leave the stale rows served; a source that changes while it downloads
publishes nothing, and the marker holds the listing taken before the fetch; and, both without the lock as on a read-only
mount, a stale tree the caller keeps is refused, not replaced, and a complete entry is served."""

import errno
import os

import pytest

from halo.data.sources import dataset_cache
from halo.data.sources.dataset_cache import (
    DOWNLOAD_COMPLETE_MARKER,
    _read_marker_fingerprint,
    _write_download_marker,
    publish_cached_download,
    serve_cached_tree,
)


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


def _publish_listing(tmp_path, listings: list[str | None]) -> None:
    """A cold publish whose source lists as each of ``listings`` in turn."""
    remaining = iter(listings)
    publish_cached_download(
        str(tmp_path / "cache"),
        str(tmp_path / "cache.lock"),
        "s3://bucket/key",
        lambda tmp: _write_tree(tmp, "fetched"),
        live_fingerprint=None,
        fresh_fingerprint=lambda: next(remaining),
    )


def test_a_source_changed_during_the_fetch_publishes_nothing(tmp_path):
    """The tree may hold files of either listing, so a marker of either would serve it as current for good."""
    with pytest.raises(RuntimeError, match="s3://bucket/key changed while it was downloaded"):
        _publish_listing(tmp_path, ["v1", "v2"])
    assert not (tmp_path / "cache").exists()
    assert not list(tmp_path.glob("cache.tmp-*"))


def test_the_marker_holds_the_listing_taken_before_the_fetch(tmp_path):
    """A listing that fails after the fetch proves nothing: the one taken before is what the tree was fetched from."""
    _publish_listing(tmp_path, ["v1", None])
    assert _read_marker_fingerprint(str(tmp_path / "cache" / DOWNLOAD_COMPLETE_MARKER)) == "v1"


def _cached_entry(tmp_path, fingerprint: str) -> str:
    cache = str(tmp_path / "entry" / "tree")
    _write_tree(cache, "cached")
    _write_download_marker(os.path.join(cache, DOWNLOAD_COMPLETE_MARKER), "s3://bucket/key", fingerprint)
    return cache


def _never_fetched(_tmp: str) -> None:
    raise AssertionError("a served cache must not be fetched again")


def test_a_listing_that_fails_before_the_fetch_takes_the_one_after(tmp_path):
    """A marker without a fingerprint is stamped with whatever the source lists next, so the tree would pass as any
    later state, and a stale tree the caller keeps would be served instead of refused."""
    _publish_listing(tmp_path, [None, "v1"])
    with pytest.raises(ValueError, match="keep the cached copy"):
        serve_cached_tree(
            str(tmp_path / "cache"),
            "s3://bucket/key",
            _never_fetched,
            lambda: "v2",
            stale_message="keep the cached copy",
            replace_stale=False,
        )


def _read_only_lock(monkeypatch) -> None:
    """A pre-staged cache on a read-only mount cannot open its lock file for writing."""

    def read_only_lock(path, **_kwargs):
        raise OSError(errno.EROFS, "Read-only file system", path)

    monkeypatch.setattr(dataset_cache.filelock, "FileLock", read_only_lock)


def test_a_stale_tree_the_caller_keeps_is_refused_and_left_in_place(tmp_path, monkeypatch):
    """Refused before the lock, so a read-only mount gets the refusal and its remedies, not a bare EROFS."""
    _read_only_lock(monkeypatch)
    cache = _cached_entry(tmp_path, "old")
    with pytest.raises(ValueError, match="keep the cached copy"):
        serve_cached_tree(
            cache,
            "s3://bucket/key",
            _never_fetched,
            lambda: "new",
            stale_message="keep the cached copy",
            replace_stale=False,
        )
    with open(os.path.join(cache, "data.txt")) as fh:
        assert fh.read() == "cached"


@pytest.mark.parametrize("live", ["fp", None], ids=["matching", "unlisted"])
def test_a_complete_entry_is_served_without_the_lock(tmp_path, monkeypatch, live):
    _read_only_lock(monkeypatch)
    cache = _cached_entry(tmp_path, "fp")
    assert serve_cached_tree(cache, "s3://bucket/key", _never_fetched, lambda: live, stale_message="stale") is True


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
