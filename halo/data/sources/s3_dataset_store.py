"""Dataset trees on S3: the push, the cached load and the mirrored reads of their small control files. A
push lands whole beside the destination and is promoted gates-last, so a complete copy exists at every
instant; a load serves the local cache and falls back to the newest sealed push when the destination is torn."""

import contextlib
import copy
import json
import os
import re
import uuid
from typing import Any

import s3fs
from datasets import Dataset, DatasetDict, load_from_disk

from halo.checkpoint.atomic import staged_file
from halo.data.sources.dataset_cache import (
    DOWNLOAD_COMPLETE_MARKER,
    HALO_S3_DATASET_CACHE_DIR,
    s3_cache_key,
    serve_cached_tree,
)
from halo.data.sources.paths import parse_s3_uri
from halo.data.sources.s3_client import S3Client, parallel_s3_transfer
from halo.log import info_logger
from halo.runtime.collectives import broadcast_from_rank0
from halo.runtime.ranks import is_global_main_process, is_local_main_process

logger = info_logger(__name__)

# A push uploads to a dot-prefixed sibling prefix, outside the destination's anchored listings.
_PUSH_INFIX = ".staging-"
_PUSH_COMPLETE_MARKER = ".push_complete"
# The scan prefix also matches a sibling dataset ("base" vs "base-2"); the 8-hex push id anchors it.
_PUSH_ID_RE = re.compile(r"^[0-9a-f]{8}/")

# The files whose absence makes load_from_disk refuse a dataset directory: the promote's ordering gates.
_DATASET_DICT_GATE = "dataset_dict.json"
_SPLIT_GATE_BASENAME = "state.json"

# The control-file reads :func:`share_control_files` made for this launch: ``(found, payload)`` by (bucket, key).
_shared_control_reads: dict[tuple[str, str], tuple[bool, Any]] = {}


def read_json_from_s3(bucket: str, key: str) -> Any:
    """A small JSON control file (``metadata.json``, ``shard_index.json``) read through s3fs on the default
    credential chain; ``FileNotFoundError`` for an absent key, other exceptions for an unreachable S3."""
    fs = s3fs.S3FileSystem()
    fs.invalidate_cache()  # s3fs instances are cached; a stale dircache would serve a deleted key
    with fs.open(f"{bucket}/{key}", "r") as f:
        return json.load(f)


def _control_json_mirror_path(bucket: str, key: str) -> str:
    """Local mirror path for an S3 control file, under the S3 dataset cache volume."""
    return os.path.join(HALO_S3_DATASET_CACHE_DIR, "control", f"{s3_cache_key(bucket, key)}.json")


def _drop_control_json_mirror(bucket: str, key: str) -> None:
    # A racing rank's unlink or a read-only volume must not replace the absence signal with its own error.
    with contextlib.suppress(OSError):
        os.unlink(_control_json_mirror_path(bucket, key))


def _write_control_json_mirror(bucket: str, key: str, payload: Any) -> None:
    mirror_dir, mirror_name = os.path.split(_control_json_mirror_path(bucket, key))
    try:
        os.makedirs(mirror_dir, exist_ok=True)
        with staged_file(mirror_dir, mirror_name, durable=False) as staged, open(staged, "w") as f:
            json.dump(payload, f)
    except OSError as e:
        logger.warning(f"Could not mirror control file s3://{bucket}/{key} under {mirror_dir}: {e}")


def read_control_json_with_cache(bucket: str, key: str) -> Any:
    """:func:`read_json_from_s3`, mirrored locally so a warm-cache relaunch survives an S3/SSO outage.

    A live read refreshes the mirror; an absent key drops it and raises, so a re-push without the file
    cannot keep serving it; any other failure serves the mirror with a warning, or re-raises without one.
    A key :func:`share_control_files` read for this launch is served from a copy of that read, so no caller sees
    another's edits.
    """
    if (bucket, key) in _shared_control_reads:
        found, payload = _shared_control_reads[(bucket, key)]
        if not found:
            raise FileNotFoundError(f"s3://{bucket}/{key}")
        return copy.deepcopy(payload)
    mirror_path = _control_json_mirror_path(bucket, key)
    try:
        payload = read_json_from_s3(bucket, key)
    except FileNotFoundError:
        _drop_control_json_mirror(bucket, key)
        raise
    except Exception as e:
        if not os.path.exists(mirror_path):
            raise
        try:
            with open(mirror_path) as f:
                payload = json.load(f)
        except (OSError, ValueError) as mirror_error:
            # A corrupt mirror is no mirror: drop it and surface the outage, the actionable error.
            logger.warning(
                f"Local mirror for s3://{bucket}/{key} at {mirror_path} is unreadable "
                f"({type(mirror_error).__name__}: {mirror_error}); dropping it."
            )
            with contextlib.suppress(OSError):
                os.unlink(mirror_path)
            raise e from mirror_error
        logger.warning(
            f"S3 unreachable for control file s3://{bucket}/{key} ({type(e).__name__}: {e}); "
            f"serving the local mirror at {mirror_path}."
        )
        return payload
    _write_control_json_mirror(bucket, key, payload)
    return payload


def share_control_files(bucket: str, keys: list[str]) -> None:
    """COLLECTIVE: read ``keys`` (:func:`read_control_json_with_cache`) on global rank 0 alone and serve every rank's
    later reads of them this launch from that read; each node's local rank 0 mirrors it. A key rank 0 could not
    read (an outage its mirror could not serve) is left to every rank's own read."""
    for key in keys:  # a key rank 0 cannot read now must not serve an earlier call's read in this process
        _shared_control_reads.pop((bucket, key), None)
    reads = {}
    if is_global_main_process():
        for key in keys:
            try:
                reads[key] = (True, read_control_json_with_cache(bucket, key))
            except FileNotFoundError:
                reads[key] = (False, None)
            except Exception as e:  # each rank then reads it itself, with its own mirror
                logger.warning(f"Reading s3://{bucket}/{key} for every rank failed ({type(e).__name__}: {e})")
    for key, (found, payload) in broadcast_from_rank0(reads).items():
        _shared_control_reads[(bucket, key)] = (found, payload)
        if is_local_main_process() and not is_global_main_process():
            if found:
                _write_control_json_mirror(bucket, key, payload)
            else:
                _drop_control_json_mirror(bucket, key)


def download_dataset_tree(client: S3Client, full_key: str, local_path: str) -> None:
    """Download the ``save_to_disk`` tree at ``full_key`` straight into ``local_path`` and open it there, so a tree
    missing its load gates (a torn promote) raises ``FileNotFoundError`` before any cache publishes it."""
    client.download_folder(full_key, local_path)
    load_from_disk(local_path)


def _describe(dataset: Dataset | DatasetDict) -> str:
    """``DatasetDict (train: N rows, ...)`` or ``Dataset (N rows)``, as the transfer records name a tree."""
    if isinstance(dataset, DatasetDict):
        return f"DatasetDict ({', '.join(f'{k}: {len(v)} rows' for k, v in dataset.items())})"
    return f"Dataset ({len(dataset)} rows)"


def _push_scan_prefix(full_key: str) -> str:
    """Listing prefix matching every pending push of this dataset (dot-prefixed sibling names)."""
    parent, _, base = full_key.rpartition("/")
    return f"{parent}/{_PUSH_INFIX}{base}-" if parent else f"{_PUSH_INFIX}{base}-"


def _own_pending_pushes(client: S3Client, full_key: str) -> list[dict[str, Any]]:
    """Objects of THIS dataset's pending pushes only, never a hyphen-sibling dataset's (``_PUSH_ID_RE``)."""
    scan_prefix = _push_scan_prefix(full_key)
    return [
        obj for obj in client.list_prefix_objects(scan_prefix) if _PUSH_ID_RE.match(obj["Key"][len(scan_prefix) :])
    ]


def _latest_complete_push(client: S3Client, full_key: str) -> str | None:
    """Newest sealed push of this dataset (by its sentinel's timestamp), or None."""
    sentinels = [
        obj for obj in _own_pending_pushes(client, full_key) if obj["Key"].endswith(f"/{_PUSH_COMPLETE_MARKER}")
    ]
    if not sentinels:
        return None
    # The key tie-breaks S3's 1-second LastModified granularity deterministically.
    newest = max(sentinels, key=lambda obj: (obj["LastModified"], obj["Key"]))
    return newest["Key"][: -len(f"/{_PUSH_COMPLETE_MARKER}")]


def _delete_pending_pushes(client: S3Client, full_key: str) -> None:
    """Remove every pending push of this dataset once a promote completed, sentinel first (listing order), so
    a crash mid-delete leaves a tree unsealed and never recovered from."""
    keys = [obj["Key"] for obj in _own_pending_pushes(client, full_key)]
    client.delete_keys(keys)
    if keys:
        logger.info(f"Removed {len(keys)} staged object(s) for s3://{client.bucket}/{full_key}")


def _promote_pushed_dataset(client: S3Client, push_key: str, full_key: str) -> None:
    """Server-side copy of a sealed push onto the destination: existing gates deleted first, the bulk in
    parallel, the split ``state.json`` gates after it and ``dataset_dict.json`` last, so the destination is
    always the old tree, loudly unreadable, or the new tree; stale objects go only after the gates landed.
    """
    push_prefix = f"{push_key}/"
    new_files: dict[str, int] = {}
    for obj in client.list_prefix_objects(push_prefix):
        rel = obj["Key"][len(push_prefix) :]
        if rel != _PUSH_COMPLETE_MARKER:
            new_files[rel] = obj["Size"]
    split_gates = sorted(rel for rel in new_files if os.path.basename(rel) == _SPLIT_GATE_BASENAME)
    if not split_gates:
        raise RuntimeError(
            f"Staged dataset at s3://{client.bucket}/{push_key} has no {_SPLIT_GATE_BASENAME}; "
            f"refusing to promote an ungated tree."
        )
    gate_set = set(split_gates) | ({_DATASET_DICT_GATE} if _DATASET_DICT_GATE in new_files else set())
    dest_prefix = f"{full_key.rstrip('/')}/"
    existing = {obj["Key"][len(dest_prefix) :] for obj in client.list_prefix_objects(dest_prefix)}
    for rel in sorted(gate_set & existing, key=lambda r: (r != _DATASET_DICT_GATE, r)):
        client.boto_client.delete_object(Bucket=client.bucket, Key=f"{dest_prefix}{rel}")

    def _copy_one(rel: str) -> int:
        client.boto_client.copy(
            {"Bucket": client.bucket, "Key": f"{push_prefix}{rel}"},
            client.bucket,
            f"{dest_prefix}{rel}",
            Config=client.transfer_config,
        )
        return new_files[rel]

    bulk = [rel for rel in new_files if rel not in gate_set]
    parallel_s3_transfer(bulk, _copy_one, desc=f"Promoting {full_key}", show_progress=False)
    for rel in split_gates:  # sequential and gate-ordered — the parallel bulk gives no ordering
        _copy_one(rel)
    if _DATASET_DICT_GATE in gate_set:
        _copy_one(_DATASET_DICT_GATE)
    for rel in sorted(existing - set(new_files)):
        client.boto_client.delete_object(Bucket=client.bucket, Key=f"{dest_prefix}{rel}")
    _delete_pending_pushes(client, full_key)


def push_dataset(
    client: S3Client,
    dataset: Dataset | DatasetDict,
    key: str,
    subfolder: str | None = None,
    overwrite: bool = True,
) -> str:
    """Save a Dataset/DatasetDict to S3, returning the S3 URI: upload beside the destination, seal, promote.

    Never delete-destination-first: the generation CLIs push their accumulated results here every
    checkpoint interval, so a crash anywhere leaves a complete copy at the destination or in the sealed
    push :func:`load_dataset` falls back to. Raises FileExistsError for an existing dataset unless
    ``overwrite=True``. One writer per destination at a time: concurrent pushers interleave promotes.
    """
    full_key = client.full_key(key, subfolder)
    s3_uri = client.s3_uri(key, subfolder)

    if not overwrite and client.exists(key, subfolder):
        raise FileExistsError(f"Dataset already exists: {s3_uri}. Use overwrite=True to replace.")

    # Unique per push, never reused: the previous push's sealed tree may be the only complete copy.
    push_key = f"{_push_scan_prefix(full_key)}{uuid.uuid4().hex[:8]}"
    dataset.save_to_disk(f"s3://{client.bucket}/{push_key}", storage_options=client.get_storage_options())
    client.boto_client.put_object(
        Bucket=client.bucket, Key=f"{push_key}/{_PUSH_COMPLETE_MARKER}", Body=full_key.encode()
    )
    _promote_pushed_dataset(client, push_key, full_key)

    logger.info(f"Uploaded {_describe(dataset)} to {s3_uri}")
    return s3_uri


def _recover_from_pending_push(
    client: S3Client, full_key: str, keep_in_memory: bool | None, use_cache: bool, cause: BaseException
) -> Dataset | DatasetDict | None:
    """The newest sealed push an interrupted promote left behind, or None when there is none, when the
    listing fails (the caller's error is the diagnostic one) or when this load already targets a push."""
    if _PUSH_INFIX in os.path.basename(full_key):
        return None
    try:
        push_key = _latest_complete_push(client, full_key)
    except Exception:  # best-effort recovery; the caller re-raises the real cause
        return None
    if push_key is None:
        return None
    logger.warning(
        f"Dataset s3://{client.bucket}/{full_key} is unreadable ({cause}), but an interrupted push "
        f"left a sealed staged copy at s3://{client.bucket}/{push_key}; serving the staged copy. "
        f"The next successful push_dataset to this destination heals it."
    )
    return load_dataset(client, push_key, subfolder=None, keep_in_memory=keep_in_memory, use_cache=use_cache)


def load_dataset(
    client: S3Client,
    key: str,
    subfolder: str | None = None,
    keep_in_memory: bool | None = None,
    use_cache: bool = True,
) -> Dataset | DatasetDict:
    """Load a Dataset from S3 through the local cache keyed by md5(bucket/key) (:func:`serve_cached_tree`):
    a complete cache is served without an S3 probe and re-downloaded when its content fingerprint no longer
    matches live S3, or always with ``use_cache=False``. A torn destination is served from the newest sealed push.
    """
    full_key = client.full_key(key, subfolder)
    s3_uri = client.s3_uri(key, subfolder)
    cache_path = os.path.join(HALO_S3_DATASET_CACHE_DIR, s3_cache_key(client.bucket, full_key), "dataset")
    # Probe S3 only when we would really fetch, so a complete cache survives an outage / expired SSO.
    if not (use_cache and os.path.exists(os.path.join(cache_path, DOWNLOAD_COMPLETE_MARKER))):
        try:
            found = client.exists(key, subfolder)
        except Exception as e:
            # The raw NoCredentialsError names neither the dataset nor the way out.
            raise FileNotFoundError(
                f"Dataset {s3_uri} is not in the local cache ({cache_path}) and S3 is unreachable "
                f"({type(e).__name__}: {e}). Restore S3 access (see agent-docs/infrastructure/aws-auth.md) "
                f"or pre-stage the cache from a node that has it."
            ) from e
        if not found:
            raise FileNotFoundError(f"Dataset not found: {s3_uri}")

    def _fetch(tmp_path: str) -> None:
        logger.info(f"Downloading dataset from S3: {s3_uri}")
        download_dataset_tree(client, full_key, tmp_path)

    try:
        cache_hit = serve_cached_tree(
            cache_path,
            s3_uri,
            _fetch,
            lambda: client.content_fingerprint(full_key),
            use_cache=use_cache,
            stale_message=(
                f"S3 dataset {s3_uri} changed since it was cached (content fingerprint mismatch); "
                f"re-downloading to {cache_path}."
            ),
        )
    except FileNotFoundError as e:
        # A destination torn by an interrupted promote has data objects but no load gate.
        recovered = _recover_from_pending_push(client, full_key, keep_in_memory, use_cache, cause=e)
        if recovered is not None:
            return recovered
        raise
    if cache_hit:
        logger.info(f"Loading dataset from cache: {cache_path}")

    # Load outside the lock so other processes can read concurrently.
    dataset = load_from_disk(cache_path, **({} if keep_in_memory is None else {"keep_in_memory": keep_in_memory}))
    logger.info(f"Loaded {_describe(dataset)} from {s3_uri}")
    return dataset


def load_dataset_from_s3_uri(
    s3_uri: str, keep_in_memory: bool | None = None, use_cache: bool = True
) -> Dataset | DatasetDict:
    """:func:`load_dataset` of a full S3 URI, the recommended entry point."""
    bucket, key = parse_s3_uri(s3_uri)
    return load_dataset(S3Client(bucket=bucket), key, keep_in_memory=keep_in_memory, use_cache=use_cache)


def push_dataset_to_s3_uri(dataset: Dataset | DatasetDict, s3_uri: str, overwrite: bool = True) -> str:
    """:func:`push_dataset` to a full S3 URI, the recommended entry point."""
    bucket, key = parse_s3_uri(s3_uri)
    return push_dataset(S3Client(bucket=bucket), dataset, key, overwrite=overwrite)
