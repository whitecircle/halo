"""Local cache-publish protocol for downloaded dataset trees.

Dependency-light (``os``/``shutil``/``filelock``) with the fetch injected, so the S3 dataset cache
and the per-shard cache of a sharded dataset share the same crash and staleness semantics. The
fingerprint is what an S3 listing can produce (key/ETag/size), so a cached tree can be validated
against a live prefix without downloading it.
"""

import hashlib
import logging
import os
import re
import shutil
import time
import uuid
from collections.abc import Callable

import filelock

from src.env import (
    data_path,
    env_positive_int,
    env_str,
    resolve_store_timeout_hours,
)

# ``src`` pins the root level to WARNING, so without the child level this module's INFO record of
# what the cache did is dropped. Plain logging, not the accelerate adapter: entry points that
# initialize no accelerate state (the S3 CLI, ``scripts/inference/*``) reach here too.
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Explicit env override wins, else the toolkit scratch root (<root>/s3_datasets). ``or``-chained
# because a set-but-empty override would resolve every cache path against the CWD, scattering
# multi-GB downloads into the repo.
HALO_S3_DATASET_CACHE_DIR = env_str("HALO_S3_DATASET_CACHE_DIR") or data_path("s3_datasets")

_MARKER_FINGERPRINT_PREFIX = "etag_fingerprint="

# How long a rank waits for the peer downloading the same entry, sized off the c10d store budget
# that bounds the peers on the other side of that download. Positive-int because ``filelock`` reads
# the extremes as modes (0 fails immediately, negative waits forever).
_CACHE_LOCK_TIMEOUT_SECONDS = env_positive_int(
    "HALO_S3_CACHE_LOCK_TIMEOUT_SECONDS", resolve_store_timeout_hours() * 3600
)

# Written inside a published cache dir; its presence is what makes the cache servable. Shared with
# the sharded-dataset shard cache, which follows the same publish protocol.
DOWNLOAD_COMPLETE_MARKER = ".download_complete"

# Older publish-temp dirs are orphans from crashed writers; anything younger may belong to a live writer.
_STALE_PUBLISH_TMP_MAX_AGE_SECONDS = 24 * 60 * 60


def s3_cache_key(bucket: str, key: str) -> str:
    """The cache-directory name for an S3 object: ``md5("<bucket>/<key>")``.

    Shared by every cache on this volume: dataset trees, per-shard caches, control-file mirrors. A
    second spelling would not fail outright; it would miss the cache a prior run pre-populated, so
    the run would need live AWS where it was meant to survive an outage.
    """
    return hashlib.md5(f"{bucket}/{key}".encode()).hexdigest()


def compute_etag_fingerprint(entries: list[tuple[str, str, int]]) -> str | None:
    """Order-insensitive md5 over (key, etag, size) triples from an S3 listing, so a cached download
    can be validated against live S3 (an in-place overwrite changes the ETags). None for an empty
    listing."""
    lines = sorted(f"{key}\t{etag}\t{size}" for key, etag, size in entries)
    if not lines:
        return None
    return hashlib.md5("\n".join(lines).encode()).hexdigest()


def _write_download_marker(marker_path: str, source_uri: str, fingerprint: str | None) -> None:
    """Write a download-completion marker: the source URI plus (when known) its content fingerprint."""
    lines = [source_uri]
    if fingerprint is not None:
        lines.append(f"{_MARKER_FINGERPRINT_PREFIX}{fingerprint}")
    with open(marker_path, "w") as f:
        f.write("\n".join(lines))


def _read_marker_fingerprint(marker_path: str) -> str | None:
    """Content fingerprint stored in a completion marker; None for fingerprint-less or unreadable ones."""
    try:
        with open(marker_path) as f:
            for line in f:
                if line.startswith(_MARKER_FINGERPRINT_PREFIX):
                    return line[len(_MARKER_FINGERPRINT_PREFIX) :].strip() or None
    except OSError:
        return None
    return None


def publish_cached_download(
    cache_path: str,
    lock_path: str,
    source_uri: str,
    fetch: Callable[[str], None],
    live_fingerprint: str | None,
    fresh_fingerprint: Callable[[], str | None],
    *,
    use_cache: bool = True,
    stale_message: str | None = None,
) -> bool:
    """Serve ``cache_path`` from cache or (re)download into it, publishing atomically. Returns cache-hit.

    Shared by the S3 dataset cache and the sharded-dataset shard cache so their staleness and crash
    semantics cannot drift:

    * a ``FileLock`` serializes same-node writers; the completion marker is re-checked under it;
    * ``live_fingerprint`` (probed by the caller outside the lock, since it is network I/O) is
      compared with the marker's stored one, so an in-place re-push to the same URI invalidates the
      cache. ``None`` means the cache cannot be validated and is served as-is; a fingerprint-less
      marker is upgraded in place;
    * ``fetch(tmp_path)`` writes a unique temp dir published by ``os.rename``: flock does not reliably
      exclude across NFS clients, so cross-node writers race the rename rather than the bytes.
    """
    complete_marker = os.path.join(cache_path, DOWNLOAD_COMPLETE_MARKER)
    lock = filelock.FileLock(lock_path, timeout=_CACHE_LOCK_TIMEOUT_SECONDS)
    with lock:
        # Under the lock same-node writers are excluded; the age threshold covers cross-node ones.
        _cleanup_stale_publish_tmp_dirs(cache_path)

        cache_valid = use_cache and os.path.exists(complete_marker)
        if cache_valid and live_fingerprint is not None:
            stored_fingerprint = _read_marker_fingerprint(complete_marker)
            if stored_fingerprint is None:
                # Fingerprint-less marker (the source could not be identified when it was written):
                # serve the cache, and stamp the live identity so the next probe can validate it.
                _write_download_marker(complete_marker, source_uri, live_fingerprint)
            elif stored_fingerprint != live_fingerprint:
                logger.warning(
                    stale_message
                    or f"{source_uri} changed since it was cached (content fingerprint mismatch); re-downloading."
                )
                cache_valid = False

        if cache_valid:
            return True

        tmp_path = f"{cache_path}.tmp-{uuid.uuid4().hex}"
        fetch(tmp_path)
        _write_download_marker(os.path.join(tmp_path, DOWNLOAD_COMPLETE_MARKER), source_uri, fresh_fingerprint())
        # Replaced only once the new tree is fully staged and marked, and by renaming the old tree
        # aside rather than deleting it in place: the tree at cache_path may be a complete cache
        # another node is loading (loads run outside the lock, and flock does not exclude across NFS
        # clients), and deleting it under that reader raises FileNotFoundError mid-load. Renaming
        # keeps the reader's already-open files valid and narrows the window where cache_path is
        # absent to one rename; a crash in it leaves the path absent (re-downloaded next run) rather
        # than a half-written tree behind a completion marker. The superseded tree wears the same
        # `.tmp-<uuid>` name a crashed writer's does, so one reaper covers both.
        superseded = f"{cache_path}.tmp-{uuid.uuid4().hex}"
        try:
            os.rename(cache_path, superseded)
        except OSError:  # nothing to replace
            superseded = None
        try:
            os.rename(tmp_path, cache_path)
        except OSError:  # another writer (possibly another node) published first; keep theirs
            shutil.rmtree(tmp_path, ignore_errors=True)
        if superseded is not None:
            shutil.rmtree(superseded, ignore_errors=True)
        return False


def _cleanup_stale_publish_tmp_dirs(
    cache_path: str, max_age_seconds: float = _STALE_PUBLISH_TMP_MAX_AGE_SECONDS
) -> list[str]:
    """Best-effort removal of orphaned ``<cache_path>.tmp-<uuid>`` dirs left by crashed writers.

    Sibling dirs matching the exact temp pattern and older than ``max_age_seconds`` are removed; a
    live writer's dir is always fresh, so there is no race. Returns the removed paths.
    """
    pattern = re.compile(re.escape(os.path.basename(cache_path)) + r"\.tmp-[0-9a-f]{32}$")
    cutoff = time.time() - max_age_seconds
    removed: list[str] = []
    try:
        entries = list(os.scandir(os.path.dirname(cache_path)))
    except OSError:
        return removed
    for entry in entries:
        try:
            if not pattern.match(entry.name) or not entry.is_dir(follow_symlinks=False):
                continue
            if entry.stat(follow_symlinks=False).st_mtime >= cutoff:
                continue
            shutil.rmtree(entry.path, ignore_errors=True)
            removed.append(entry.path)
        except OSError:
            continue
    if removed:
        logger.info(f"Removed {len(removed)} stale cache publish temp dir(s) from crashed writers: {removed}")
    return removed
