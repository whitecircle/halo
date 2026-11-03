"""Local cache-publish protocol for downloaded dataset trees, with the fetch injected. The fingerprint is what an S3
listing produces (key/ETag/size), so a cached tree validates against a live prefix unread."""

import hashlib
import os
import re
import shutil
import time
import uuid
from collections.abc import Callable

import filelock

from halo.env import data_path, env_positive_int, env_str, store_timeout
from halo.log import info_logger

logger = info_logger(__name__)

# ``or``-chained: a set-but-empty override would resolve every cache path against the CWD.
HALO_S3_DATASET_CACHE_DIR = env_str("HALO_S3_DATASET_CACHE_DIR") or data_path("s3_datasets")

_MARKER_FINGERPRINT_PREFIX = "etag_fingerprint="

# The wait for a peer downloading the same entry; positive because filelock reads 0 and negatives as modes.
_CACHE_LOCK_TIMEOUT_SECONDS = env_positive_int(
    "HALO_S3_CACHE_LOCK_TIMEOUT_SECONDS", int(store_timeout().total_seconds())
)

# Written inside a published cache dir; its presence is what makes the cache servable.
DOWNLOAD_COMPLETE_MARKER = ".download_complete"

# Older publish-temp dirs are orphans from crashed writers; anything younger may belong to a live writer.
_STALE_PUBLISH_TMP_MAX_AGE_SECONDS = 24 * 60 * 60


def s3_cache_key(bucket: str, key: str) -> str:
    """``md5("<bucket>/<key>")``, for every cache on this volume: another spelling would miss a pre-populated cache."""
    return hashlib.md5(f"{bucket}/{key}".encode()).hexdigest()


def compute_etag_fingerprint(entries: list[tuple[str, str, int]]) -> str | None:
    """Order-insensitive md5 over an S3 listing's ``(key, etag, size)`` triples; None for an empty listing."""
    lines = sorted(f"{key}\t{etag}\t{size}" for key, etag, size in entries)
    if not lines:
        return None
    return hashlib.md5("\n".join(lines).encode()).hexdigest()


def _write_download_marker(marker_path: str, source_uri: str, fingerprint: str | None) -> None:
    lines = [source_uri]
    if fingerprint is not None:
        lines.append(f"{_MARKER_FINGERPRINT_PREFIX}{fingerprint}")
    with open(marker_path, "w") as f:
        f.write("\n".join(lines))


def _read_marker_fingerprint(marker_path: str) -> str | None:
    """None for fingerprint-less or unreadable markers."""
    try:
        with open(marker_path) as f:
            for line in f:
                if line.startswith(_MARKER_FINGERPRINT_PREFIX):
                    return line[len(_MARKER_FINGERPRINT_PREFIX) :].strip() or None
    except OSError:
        return None
    return None


def _cleanup_stale_publish_tmp_dirs(
    cache_path: str, max_age_seconds: float = _STALE_PUBLISH_TMP_MAX_AGE_SECONDS
) -> list[str]:
    """Remove ``<cache_path>.tmp-<uuid>`` siblings older than ``max_age_seconds``, which crashed writers left; a live
    writer's is always fresh."""
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


def _publish_tree(tmp_path: str, cache_path: str) -> None:
    """Swap ``tmp_path`` in, keeping another writer's tree if it published first. The old tree is renamed aside, as
    another node may load it; a crash between renames leaves ``cache_path`` absent and a ``.tmp-<uuid>`` to reap."""
    superseded = f"{cache_path}.tmp-{uuid.uuid4().hex}"
    try:
        os.rename(cache_path, superseded)
    except FileNotFoundError:  # nothing to replace
        superseded = None
    except OSError:
        shutil.rmtree(tmp_path, ignore_errors=True)
        raise
    try:
        os.rename(tmp_path, cache_path)
    except OSError:  # another writer (possibly another node) published first; keep theirs
        shutil.rmtree(tmp_path, ignore_errors=True)
    if superseded is not None:
        shutil.rmtree(superseded, ignore_errors=True)


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
    """Serve from cache or (re)download with ``fetch(tmp_path)`` and publish atomically; returns cache-hit. Writers
    serialize on ``lock_path``; ``live_fingerprint`` (``None`` serves unvalidated) invalidates a cache whose source
    changed and upgrades a fingerprint-less marker. A raising fetch leaves no partial tree."""
    complete_marker = os.path.join(cache_path, DOWNLOAD_COMPLETE_MARKER)
    with filelock.FileLock(lock_path, timeout=_CACHE_LOCK_TIMEOUT_SECONDS):
        # Under the lock same-node writers are excluded; the age threshold covers cross-node ones.
        _cleanup_stale_publish_tmp_dirs(cache_path)

        cache_valid = use_cache and os.path.exists(complete_marker)
        if cache_valid and live_fingerprint is not None:
            stored_fingerprint = _read_marker_fingerprint(complete_marker)
            if stored_fingerprint is None:
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
        try:
            fetch(tmp_path)
        except BaseException:
            shutil.rmtree(tmp_path, ignore_errors=True)
            raise
        _write_download_marker(os.path.join(tmp_path, DOWNLOAD_COMPLETE_MARKER), source_uri, fresh_fingerprint())
        _publish_tree(tmp_path, cache_path)
        return False


def serve_cached_tree(
    cache_path: str,
    source_uri: str,
    fetch: Callable[[str], None],
    fingerprint: Callable[[], str | None],
    *,
    stale_message: str,
    use_cache: bool = True,
) -> bool:
    """Locked by ``<cache dir>/.lock``, validating a complete cache against ``fingerprint()``. Returns cache-hit."""
    cache_dir = os.path.dirname(cache_path)
    os.makedirs(cache_dir, exist_ok=True)
    # Probed outside the lock (network I/O), and only for a cache that could be served.
    complete = use_cache and os.path.exists(os.path.join(cache_path, DOWNLOAD_COMPLETE_MARKER))
    return publish_cached_download(
        cache_path,
        os.path.join(cache_dir, ".lock"),
        source_uri,
        fetch,
        fingerprint() if complete else None,
        fingerprint,
        use_cache=use_cache,
        stale_message=stale_message,
    )
