"""S3 transport: the boto3 ``S3Client``, small control-file reads, and the default-bucket helpers.

``S3Client`` takes keys relative to a bucket; the module-level ``*_from_s3_uri`` helpers take full
URIs and build their own client. Dataset trees move through the staged-push protocol described at
:data:`_STAGING_INFIX` and are served locally through :mod:`src.data.sources.dataset_cache`.
"""

import contextlib
import json
import logging
import os
import re
import shutil
import uuid
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any

import boto3
import s3fs
from boto3.s3.transfer import TransferConfig
from botocore.config import Config
from botocore.exceptions import ClientError
from datasets import Dataset, DatasetDict, load_from_disk
from tqdm import tqdm

from src.data.sources.dataset_cache import (
    DOWNLOAD_COMPLETE_MARKER,
    HALO_S3_DATASET_CACHE_DIR,
    compute_etag_fingerprint,
    publish_cached_download,
    s3_cache_key,
)
from src.data.sources.paths import METADATA_FILE, parse_s3_uri
from src.env import env_int, env_str

# ``src`` pins the ROOT level to WARNING, so without the child level this module's INFO record of
# what data moved lands nowhere. Plain logging, not the accelerate adapter: entry points that
# initialize no accelerate state (the S3 CLI, ``scripts/inference/*``) reach here too.
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

__all__ = [
    "DEFAULT_BUCKET",
    "S3Client",
    "build_s3_uri",
    "exists",
    "has_control_json_mirror",
    "load_dataset_from_s3_uri",
    "push_dataset_to_s3_uri",
    "read_control_json_with_cache",
]

DEFAULT_BUCKET = env_str("HALO_S3_DEFAULT_BUCKET") or "my-bucket"

# Bounded so concurrency × each file's multipart connections stays within max_pool_connections=50.
# Clamped, not ``or``-defaulted: an explicit 0 would silently become 16 instead of the "no workers"
# it literally asks for, and env_int already turns unset/malformed into the default.
_S3_FOLDER_CONCURRENCY = max(1, env_int("HALO_S3_MAX_FOLDER_CONCURRENCY", 16))

# Staged-push protocol (push_dataset): the tree uploads whole to a dot-prefixed SIBLING prefix
# (outside the destination's anchored listings), is sealed with the sentinel, then promoted with the
# load gates deleted first and copied last. A complete copy therefore exists at every instant, an
# interrupted promote leaves the destination loudly unreadable, and the sealed tree is what
# ``load_dataset`` recovers from.
_STAGING_INFIX = ".staging-"
_PUSH_COMPLETE_MARKER = ".push_complete"
# The staging scan prefix is a plain string prefix, so a sibling dataset name extending this one
# ("base" vs "base-2") matches it too; anchoring the remainder to exactly the 8-hex staging id keeps
# a sweep from destroying the sibling's only complete copy and a recovery from serving its rows.
_STAGING_ID_RE = re.compile(r"^[0-9a-f]{8}/")

# The files whose absence makes load_from_disk refuse a dataset directory — the promote's ordering
# gates. dataset_dict.json is the DatasetDict root gate and goes last of all.
_DATASET_DICT_GATE = "dataset_dict.json"
_SPLIT_GATE_BASENAME = "state.json"


def _parallel_s3_transfer(items: list, transfer_one: Callable[[Any], int], *, desc: str, show_progress: bool) -> None:
    """Run ``transfer_one`` over ``items`` on a bounded thread pool, summing returned byte counts.

    Shared body of the folder upload/download loops. The first per-file error propagates via
    ``future.result``. No-op on an empty list.
    """
    if not items:
        return
    pbar = tqdm(total=len(items), desc=desc, unit="file") if show_progress else None
    moved = 0
    with ThreadPoolExecutor(max_workers=min(_S3_FOLDER_CONCURRENCY, len(items))) as pool:
        for future in as_completed(pool.submit(transfer_one, item) for item in items):
            moved += future.result()  # re-raises the first per-file transfer error
            if pbar is not None:
                pbar.update(1)
                pbar.set_postfix({"size": f"{moved / (1024 * 1024):.1f}MB"})
    if pbar is not None:
        pbar.close()


def read_json_from_s3(bucket: str, key: str) -> Any:
    """Read and parse a small JSON control file (``metadata.json``, ``shard_index.json``) from S3.

    ``key`` is the object path within ``bucket``. Uses s3fs on the default credential chain — a
    single small read needs none of ``S3Client``'s caching/transfer machinery. Raises
    ``FileNotFoundError`` for an absent key (authoritative), other exceptions for an unreachable S3.
    """
    fs = s3fs.S3FileSystem()
    fs.invalidate_cache()  # s3fs instances are cached; a stale dircache would serve a deleted key
    with fs.open(f"{bucket}/{key}", "r") as f:
        return json.load(f)


def _control_json_mirror_path(bucket: str, key: str) -> str:
    """Local mirror path for an S3 control file, under the S3 dataset cache volume."""
    return os.path.join(HALO_S3_DATASET_CACHE_DIR, "control", f"{s3_cache_key(bucket, key)}.json")


def has_control_json_mirror(bucket: str, key: str) -> bool:
    """Whether a local mirror of this control file exists (a prior run reached it live)."""
    return os.path.exists(_control_json_mirror_path(bucket, key))


def read_control_json_with_cache(bucket: str, key: str) -> Any:
    """Read a small S3 JSON control file, mirroring it locally; serve the mirror when S3 is down.

    Control files (``metadata.json``, ``shard_index.json``) are what training needs from S3 *before*
    the dataset bytes, which the completion-marker caches already serve offline — without this a
    warm-cache relaunch dies on the first control read during an S3/SSO outage. Live wins and
    refreshes the mirror. ``FileNotFoundError`` is an authoritative ABSENCE: the mirror is dropped and
    the error propagates, so a re-push without the control file cannot keep serving a stale mirror.
    Any other failure serves the mirror with a warning, or re-raises when there is none. The mirror
    write is temp-file + atomic ``os.replace``, so a crashed writer never leaves a torn one.
    """
    mirror_path = _control_json_mirror_path(bucket, key)
    try:
        payload = read_json_from_s3(bucket, key)
    except FileNotFoundError:
        # Best-effort drop, OSError-wide: a concurrent rank racing the same unlink or a read-only
        # cache volume must not replace the authoritative-absence signal with its own error.
        with contextlib.suppress(OSError):
            os.unlink(mirror_path)
        raise
    except Exception as e:
        if not os.path.exists(mirror_path):
            raise
        try:
            with open(mirror_path) as f:
                payload = json.load(f)
        except (OSError, ValueError) as mirror_error:
            # A torn/corrupt mirror (disk fault — the atomic publish cannot produce one) is NO
            # mirror: drop it so probes stop trusting its existence, and surface the outage, which
            # is the actionable error.
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
    os.makedirs(os.path.dirname(mirror_path), exist_ok=True)
    tmp_path = f"{mirror_path}.tmp-{uuid.uuid4().hex}"
    try:
        with open(tmp_path, "w") as f:
            json.dump(payload, f)
        os.replace(tmp_path, mirror_path)
    except OSError as e:
        # The live payload is in hand — a full/read-only cache volume must not fail the read itself.
        logger.warning(f"Could not mirror control file s3://{bucket}/{key} to {mirror_path}: {e}")
        with contextlib.suppress(OSError):
            os.unlink(tmp_path)
    return payload


@dataclass
class S3Client:
    """S3 client wrapper for pandas and HuggingFace dataset operations.

    Credentials default to the standard AWS chain (env/credentials file) when unset.
    endpoint_url targets S3-compatible services.
    """

    bucket: str = DEFAULT_BUCKET
    aws_access_key_id: str | None = None
    aws_secret_access_key: str | None = None
    endpoint_url: str | None = None
    region_name: str | None = None
    _client: Any = field(default=None, init=False, repr=False)
    _transfer_config: Any = field(default=None, init=False, repr=False)

    def __post_init__(self):
        """Initialize the boto3 client and its transfer config."""
        session_kwargs = {}
        if self.aws_access_key_id:
            session_kwargs["aws_access_key_id"] = self.aws_access_key_id
        if self.aws_secret_access_key:
            session_kwargs["aws_secret_access_key"] = self.aws_secret_access_key
        if self.region_name:
            session_kwargs["region_name"] = self.region_name

        boto_config = Config(
            max_pool_connections=50,
            retries={"max_attempts": 3, "mode": "standard"},
        )

        # 5 threads/transfer, not boto3's 10: parallel workers must stay within max_pool_connections=50.
        self._transfer_config = TransferConfig(
            max_concurrency=5,
        )

        client_kwargs = {"config": boto_config}
        if self.endpoint_url:
            client_kwargs["endpoint_url"] = self.endpoint_url

        session = boto3.Session(**session_kwargs)
        self._client = session.client("s3", **client_kwargs)

    def get_storage_options(self) -> dict:
        """Translate this client's credentials/endpoint into s3fs storage_options for
        datasets load_from_disk/save_to_disk. Empty dict under the default credential chain.
        """
        opts = {}
        if self.aws_access_key_id:
            opts["key"] = self.aws_access_key_id
        if self.aws_secret_access_key:
            opts["secret"] = self.aws_secret_access_key
        client_kwargs = {}
        if self.endpoint_url:
            client_kwargs["endpoint_url"] = self.endpoint_url
        if self.region_name:
            client_kwargs["region_name"] = self.region_name
        if client_kwargs:
            opts["client_kwargs"] = client_kwargs
        return opts

    def _get_full_key(self, key: str, subfolder: str | None = None) -> str:
        """Join an optional subfolder prefix onto key, normalizing surrounding slashes."""
        if subfolder is not None:
            subfolder = subfolder.strip("/")
            key = key.strip("/")
            return f"{subfolder}/{key}"
        return key.strip("/")

    def _get_s3_uri(self, key: str, subfolder: str | None = None) -> str:
        """Get full S3 URI for a key."""
        full_key = self._get_full_key(key, subfolder)
        return f"s3://{self.bucket}/{full_key}"

    def object_exists(self, key: str, subfolder: str | None = None) -> bool:
        """Whether an object exists at EXACTLY this key.

        Unlike :meth:`exists`, a folder-like prefix with children but no object of its own is False.
        Callers that must not treat a prefix as a deletable object need this distinction: S3's
        DeleteObject on a prefix succeeds as a no-op, so acting on :meth:`exists` reports success
        having done nothing.
        """
        full_key = self._get_full_key(key, subfolder)
        try:
            self._client.head_object(Bucket=self.bucket, Key=full_key)
            return True
        except ClientError as e:
            if e.response["Error"]["Code"] != "404":
                raise
            return False

    def exists(self, key: str, subfolder: str | None = None) -> bool:
        """Check if key exists as an exact object or as a (folder-like) prefix.

        Only an authoritative absence of the KEY reads as False: expired credentials, a 403 or a
        throttle answer "unknown", and reporting that as absence makes a push overwrite-check pass and
        a delete claim there was nothing there. A missing BUCKET is a misspelled bucket or a wrong
        region, and says so rather than sending every caller after the key.
        """
        if self.object_exists(key, subfolder):
            return True

        full_key = self._get_full_key(key, subfolder)
        try:
            response = self._client.list_objects_v2(Bucket=self.bucket, Prefix=full_key.rstrip("/") + "/", MaxKeys=1)
            return response.get("KeyCount", 0) > 0
        except ClientError as e:
            code = e.response["Error"]["Code"]
            if code == "NoSuchBucket":
                raise FileNotFoundError(
                    f"S3 bucket '{self.bucket}' does not exist (looking for '{full_key}'). Check the "
                    f"bucket name and the region this client resolved."
                ) from e
            if code not in ("404", "NoSuchKey"):
                raise
            return False

    def list_objects(
        self, prefix: str = "", subfolder: str | None = None, recursive: bool = True, max_keys: int = 1000
    ) -> list[dict[str, Any]]:
        """List objects under prefix as metadata dicts (Key, Size, LastModified, Type).

        recursive=False lists only immediate children (ls-style, via a "/" delimiter).
        """
        full_prefix = self._get_full_key(prefix, subfolder)
        if full_prefix and not full_prefix.endswith("/"):
            full_prefix += "/"

        objects = []
        kwargs = {
            "Bucket": self.bucket,
            "Prefix": full_prefix,
            "MaxKeys": max_keys,
        }

        if not recursive:
            kwargs["Delimiter"] = "/"

        paginator = self._client.get_paginator("list_objects_v2")

        for page in paginator.paginate(**kwargs):
            for obj in page.get("Contents", []):
                display_key = obj["Key"]
                if subfolder is not None:
                    display_key = display_key[len(full_prefix) :]
                objects.append(
                    {
                        "Key": display_key,
                        "FullKey": obj["Key"],
                        "Size": obj["Size"],
                        "LastModified": obj["LastModified"],
                        "Type": "file",
                    }
                )

            for prefix_info in page.get("CommonPrefixes", []):
                display_key = prefix_info["Prefix"]
                if subfolder is not None:
                    display_key = display_key[len(full_prefix) :]
                objects.append({"Key": display_key.rstrip("/"), "FullKey": prefix_info["Prefix"], "Type": "directory"})

            if len(objects) >= max_keys:
                break

        return objects

    def delete(self, key: str, subfolder: str | None = None, recursive: bool = True) -> bool:
        """Delete a single object, or with recursive=True every object under the prefix
        (anchored with a trailing slash; deleted in batches of 1000, the S3 API limit).
        """
        full_key = self._get_full_key(key, subfolder)

        if recursive:
            # The trailing slash anchors the prefix to children: without it "run1" also matches "run10/...".
            prefix = full_key.rstrip("/") + "/" if full_key else full_key
            objects_to_delete = []
            paginator = self._client.get_paginator("list_objects_v2")

            for page in paginator.paginate(Bucket=self.bucket, Prefix=prefix):
                for obj in page.get("Contents", []):
                    objects_to_delete.append({"Key": obj["Key"]})

            if objects_to_delete:
                for i in range(0, len(objects_to_delete), 1000):
                    batch = objects_to_delete[i : i + 1000]
                    self._client.delete_objects(Bucket=self.bucket, Delete={"Objects": batch})
                logger.info(f"Deleted {len(objects_to_delete)} objects from s3://{self.bucket}/{full_key}")
            # False when the prefix matched nothing: a caller that reports success off this would
            # tell the user their data is gone while it is still there.
            return bool(objects_to_delete)
        else:
            try:
                self._client.delete_object(Bucket=self.bucket, Key=full_key)
                logger.info(f"Deleted s3://{self.bucket}/{full_key}")
                return True
            except ClientError as e:
                logger.error(f"Failed to delete s3://{self.bucket}/{full_key}: {e}")
                return False

    def content_entries(self, full_key: str) -> list[tuple[str, str, int]]:
        """(key, ETag, size) of every object under the prefix, raising on any listing failure.

        The raising half of :meth:`content_fingerprint`: both an unreachable S3 and a prefix that
        genuinely holds nothing fingerprint as ``None``, so a caller that must tell those apart
        lists here instead.
        """
        return [
            (obj["Key"], obj.get("ETag", ""), obj["Size"])
            for obj in self._list_prefix_keys(full_key.rstrip("/") + "/")
        ]

    def content_fingerprint(self, full_key: str) -> str | None:
        """Best-effort content identity of every object under the prefix (aggregate ETag fingerprint).

        None when S3 is unreachable (no credentials, expired SSO, network outage) or the prefix lists
        empty — callers treat None as "cannot validate" and keep serving the local cache, so offline
        cache use keeps working.
        """
        try:
            return compute_etag_fingerprint(self.content_entries(full_key))
        except Exception as e:  # any botocore/network error → cannot validate, keep the cache usable
            logger.info(
                f"S3 unreachable for cache validation of s3://{self.bucket}/{full_key} "
                f"({type(e).__name__}: {e}); serving the local cache as-is."
            )
            return None

    def _staging_scan_prefix(self, full_key: str) -> str:
        """Listing prefix matching every staging tree of this dataset (dot-prefixed sibling names)."""
        parent, _, base = full_key.rpartition("/")
        return f"{parent}/{_STAGING_INFIX}{base}-" if parent else f"{_STAGING_INFIX}{base}-"

    def _new_staging_key(self, full_key: str) -> str:
        """A fresh unique staging prefix for this dataset. Unique per push, never reused: the
        previous push's sealed staging tree may be the only complete copy while the destination is
        torn, so it must never be overwritten before a new complete copy exists."""
        return f"{self._staging_scan_prefix(full_key)}{uuid.uuid4().hex[:8]}"

    def _list_prefix_keys(self, prefix: str) -> list[dict[str, Any]]:
        """Raw object listing under an exact string prefix (no trailing-slash forcing)."""
        entries = []
        paginator = self._client.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=self.bucket, Prefix=prefix):
            entries.extend(page.get("Contents", []))
        return entries

    def _own_staging_entries(self, full_key: str) -> list[dict[str, Any]]:
        """Objects of THIS dataset's staging trees only — remainder anchored to the 8-hex staging id
        (see ``_STAGING_ID_RE``), never a hyphen-sibling dataset's trees."""
        scan_prefix = self._staging_scan_prefix(full_key)
        return [
            obj for obj in self._list_prefix_keys(scan_prefix) if _STAGING_ID_RE.match(obj["Key"][len(scan_prefix) :])
        ]

    def _latest_complete_staging(self, full_key: str) -> str | None:
        """Newest sealed staging tree for this dataset (by its sentinel's timestamp), or None."""
        sentinels = [
            obj for obj in self._own_staging_entries(full_key) if obj["Key"].endswith(f"/{_PUSH_COMPLETE_MARKER}")
        ]
        if not sentinels:
            return None
        # The key tie-breaks S3's 1-second LastModified granularity deterministically.
        newest = max(sentinels, key=lambda obj: (obj["LastModified"], obj["Key"]))
        return newest["Key"][: -len(f"/{_PUSH_COMPLETE_MARKER}")]

    def _delete_staging_trees(self, full_key: str) -> None:
        """Remove every staging tree of this dataset, once a promote has completed and they are garbage.

        Deletion follows lexicographic listing order, so the dot-prefixed sentinel goes before its
        tree's data keys: a crash mid-delete leaves the tree UNSEALED and unpickable by recovery.
        """
        keys = [{"Key": obj["Key"]} for obj in self._own_staging_entries(full_key)]
        for i in range(0, len(keys), 1000):
            self._client.delete_objects(Bucket=self.bucket, Delete={"Objects": keys[i : i + 1000]})
        if keys:
            logger.info(f"Removed {len(keys)} staged object(s) for s3://{self.bucket}/{full_key}")

    def _promote_staged_dataset(self, staging_key: str, full_key: str) -> None:
        """Server-side copy of a sealed staging tree onto the destination, load-gates last.

        Ordering: the destination's existing gate files are deleted first, the bulk copies in
        parallel, per-split ``state.json`` files after it, ``dataset_dict.json`` last — so at every
        instant the destination is either the old complete tree, loudly unreadable (a gate missing),
        or the new complete tree; never a readable old/new mixture. Stale objects the new tree lacks
        are swept only after the gates landed.
        """
        staging_prefix = f"{staging_key}/"
        new_files: dict[str, int] = {}
        for obj in self._list_prefix_keys(staging_prefix):
            rel = obj["Key"][len(staging_prefix) :]
            if rel != _PUSH_COMPLETE_MARKER:
                new_files[rel] = obj["Size"]
        split_gates = sorted(rel for rel in new_files if os.path.basename(rel) == _SPLIT_GATE_BASENAME)
        if not split_gates:
            # save_to_disk not writing state.json means the layout contract changed under us; promoting
            # would leave a tree with no gate ordering at all.
            raise RuntimeError(
                f"Staged dataset at s3://{self.bucket}/{staging_key} has no {_SPLIT_GATE_BASENAME}; "
                f"refusing to promote an ungated tree."
            )
        gate_set = set(split_gates) | ({_DATASET_DICT_GATE} if _DATASET_DICT_GATE in new_files else set())

        dest_prefix = f"{full_key.rstrip('/')}/"
        existing = {obj["Key"][len(dest_prefix) :] for obj in self._list_prefix_keys(dest_prefix)}

        for rel in sorted(gate_set & existing, key=lambda r: (r != _DATASET_DICT_GATE, r)):
            self._client.delete_object(Bucket=self.bucket, Key=f"{dest_prefix}{rel}")

        def _copy_one(rel: str) -> int:
            self._client.copy(
                {"Bucket": self.bucket, "Key": f"{staging_prefix}{rel}"},
                self.bucket,
                f"{dest_prefix}{rel}",
                Config=self._transfer_config,
            )
            return new_files[rel]

        _parallel_s3_transfer(
            [rel for rel in new_files if rel not in gate_set],
            _copy_one,
            desc=f"Promoting {full_key}",
            show_progress=False,
        )
        for rel in split_gates:  # sequential and gate-ordered — the parallel bulk gives no ordering
            _copy_one(rel)
        if _DATASET_DICT_GATE in gate_set:
            _copy_one(_DATASET_DICT_GATE)

        for rel in sorted(existing - set(new_files)):
            self._client.delete_object(Bucket=self.bucket, Key=f"{dest_prefix}{rel}")
        self._delete_staging_trees(full_key)

    def push_dataset(
        self,
        dataset: Dataset | DatasetDict,
        key: str,
        subfolder: str | None = None,
        overwrite: bool = True,
    ) -> str:
        """Save a Dataset/DatasetDict to S3 through the staged-push protocol, returning the S3 URI.

        Never delete-destination-first: the generation CLIs push their accumulated results through
        here every checkpoint interval, and a crash between a prefix delete and the re-upload would
        erase all of it. A crash anywhere in this protocol instead leaves a complete copy either at
        the destination or in the sealed staging tree :meth:`load_dataset` falls back to. Raises
        FileExistsError for an existing dataset unless ``overwrite=True``. One writer per destination
        at a time — concurrent pushers interleave promotes.
        """
        full_key = self._get_full_key(key, subfolder)
        s3_uri = f"s3://{self.bucket}/{full_key}"

        if not overwrite and self.exists(key, subfolder):
            raise FileExistsError(f"Dataset already exists: {s3_uri}. Use overwrite=True to replace.")

        staging_key = self._new_staging_key(full_key)
        dataset.save_to_disk(f"s3://{self.bucket}/{staging_key}", storage_options=self.get_storage_options())
        self._client.put_object(
            Bucket=self.bucket, Key=f"{staging_key}/{_PUSH_COMPLETE_MARKER}", Body=full_key.encode()
        )
        self._promote_staged_dataset(staging_key, full_key)

        if isinstance(dataset, DatasetDict):
            splits_info = ", ".join(f"{k}: {len(v)} rows" for k, v in dataset.items())
            logger.info(f"Uploaded DatasetDict ({splits_info}) to {s3_uri}")
        else:
            logger.info(f"Uploaded Dataset ({len(dataset)} rows) to {s3_uri}")

        return s3_uri

    def _recover_from_staging(
        self, full_key: str, keep_in_memory: bool | None, use_cache: bool, cause: BaseException
    ) -> Dataset | DatasetDict | None:
        """Torn-destination fallback: the newest sealed staging tree an interrupted push left behind.

        None when there is none, when the listing itself fails (offline — the original error is the
        diagnostic one), or when this load already targets a staging tree (one recovery level).
        """
        if _STAGING_INFIX in os.path.basename(full_key):
            return None
        try:
            staging_key = self._latest_complete_staging(full_key)
        except Exception:  # best-effort recovery; the caller re-raises the real cause
            return None
        if staging_key is None:
            return None
        logger.warning(
            f"Dataset s3://{self.bucket}/{full_key} is unreadable ({cause}), but an interrupted push "
            f"left a sealed staged copy at s3://{self.bucket}/{staging_key}; serving the staged copy. "
            f"The next successful push_dataset to this destination heals it."
        )
        return self.load_dataset(staging_key, subfolder=None, keep_in_memory=keep_in_memory, use_cache=use_cache)

    def load_dataset(
        self,
        key: str,
        subfolder: str | None = None,
        keep_in_memory: bool | None = None,
        use_cache: bool = True,
    ) -> Dataset | DatasetDict:
        """Load a Dataset from S3 with local caching keyed by md5(bucket/key).

        Downloads once, then serves from HALO_S3_DATASET_CACHE_DIR. A FileLock coordinates ranks (one
        downloads, others wait). With a complete cache the S3 existence probe is skipped, so training
        survives transient S3 outages. A best-effort staleness probe compares the marker's content
        fingerprint against live S3 (an in-place re-push to the same URI changes the ETags) and
        re-downloads on mismatch; when S3 is unreachable the cache is served as-is. use_cache=False
        forces a re-download.
        """
        full_key = self._get_full_key(key, subfolder)
        s3_uri = f"s3://{self.bucket}/{full_key}"

        cache_dir = os.path.join(HALO_S3_DATASET_CACHE_DIR, s3_cache_key(self.bucket, full_key))
        cache_path = os.path.join(cache_dir, "dataset")
        lock_path = os.path.join(cache_dir, ".lock")
        complete_marker = os.path.join(cache_path, DOWNLOAD_COMPLETE_MARKER)

        os.makedirs(cache_dir, exist_ok=True)

        # Probe S3 only when we would really fetch, so a complete cache survives an outage / expired SSO.
        cache_complete = use_cache and os.path.exists(complete_marker)
        if not cache_complete:
            try:
                found = self.exists(key, subfolder)
            except Exception as e:
                # A cold cache with unreachable S3 is unloadable either way — but the raw
                # NoCredentialsError names neither the dataset nor the way out.
                raise FileNotFoundError(
                    f"Dataset {s3_uri} is not in the local cache ({cache_path}) and S3 is unreachable "
                    f"({type(e).__name__}: {e}). Restore S3 access (see agent-docs/infrastructure/aws-auth.md) "
                    f"or pre-stage the cache from a node that has it."
                ) from e
            if not found:
                raise FileNotFoundError(f"Dataset not found: {s3_uri}")

        # Staleness probe outside the lock (network I/O); None (unreachable / empty) skips validation.
        live_fingerprint = self.content_fingerprint(full_key) if cache_complete else None

        def _fetch(tmp_path: str) -> None:
            logger.info(f"Downloading dataset from S3: {s3_uri}")
            load_from_disk(s3_uri, storage_options=self.get_storage_options()).save_to_disk(tmp_path)

        try:
            cache_hit = publish_cached_download(
                cache_path,
                lock_path,
                s3_uri,
                _fetch,
                live_fingerprint,
                lambda: self.content_fingerprint(full_key),
                use_cache=use_cache,
                stale_message=(
                    f"S3 dataset {s3_uri} changed since it was cached (content fingerprint mismatch); "
                    f"re-downloading to {cache_path}."
                ),
            )
        except FileNotFoundError as e:
            # A destination torn by an interrupted promote has data objects but no load gate; the
            # sealed staging tree the push left behind is the complete copy.
            recovered = self._recover_from_staging(full_key, keep_in_memory, use_cache, cause=e)
            if recovered is not None:
                return recovered
            raise
        if cache_hit:
            logger.info(f"Loading dataset from cache: {cache_path}")

        # Load outside the lock so other processes can read concurrently.
        load_kwargs = {}
        if keep_in_memory is not None:
            load_kwargs["keep_in_memory"] = keep_in_memory

        dataset = load_from_disk(cache_path, **load_kwargs)

        if isinstance(dataset, DatasetDict):
            splits_info = ", ".join(f"{k}: {len(v)} rows" for k, v in dataset.items())
            logger.info(f"Loaded DatasetDict ({splits_info}) from {s3_uri}")
        else:
            logger.info(f"Loaded Dataset ({len(dataset)} rows) from {s3_uri}")

        return dataset

    def push_folder(
        self,
        local_path: str,
        key: str,
        subfolder: str | None = None,
        overwrite: bool = True,
        show_progress: bool = False,
    ) -> str:
        """Upload a local directory tree to S3, returning the S3 URI.

        An existing prefix raises FileExistsError unless ``overwrite=True``. Overwrite never clears
        the prefix up front — that leaves no copy anywhere until the first byte lands. The top-level
        ``metadata.json`` (the preprocessed-dataset marker) is deleted FIRST and re-uploaded LAST, so
        a mid-failure leaves a prefix the preprocessed probe rejects rather than one it accepts.
        """
        local_path = os.path.abspath(local_path)
        if not os.path.isdir(local_path):
            raise ValueError(f"Local path is not a directory: {local_path}")

        full_key = self._get_full_key(key, subfolder)
        s3_uri = f"s3://{self.bucket}/{full_key}"

        existing_keys: set[str] = set()
        if self.exists(key, subfolder):
            if not overwrite:
                raise FileExistsError(f"Folder already exists: {s3_uri}. Use overwrite=True to replace.")
            logger.info(f"Overwriting existing folder at {s3_uri}")
            # Uncapped: the stale sweep below deletes what this set does NOT contain, so a listing
            # truncated at any max_keys would silently re-upload-and-keep objects it never saw.
            existing_keys = {entry["Key"] for entry in self._list_prefix_keys(f"{full_key}/")}

        files_to_upload = []
        total_size = 0
        for root, _dirs, files in os.walk(local_path):
            for file in files:
                local_file = os.path.join(root, file)
                relative_path = os.path.relpath(local_file, local_path)
                file_size = os.path.getsize(local_file)
                files_to_upload.append((local_file, relative_path, file_size))
                total_size += file_size

        marker_items = [item for item in files_to_upload if item[1] == METADATA_FILE]
        bulk_items = [item for item in files_to_upload if item[1] != METADATA_FILE]
        marker_key = f"{full_key}/{METADATA_FILE}"
        if marker_key in existing_keys:
            self._client.delete_object(Bucket=self.bucket, Key=marker_key)

        def _upload_one(item):
            local_file, relative_path, file_size = item
            self._client.upload_file(
                local_file, self.bucket, f"{full_key}/{relative_path}", Config=self._transfer_config
            )
            return file_size

        _parallel_s3_transfer(bulk_items, _upload_one, desc=f"Uploading to {key}", show_progress=show_progress)
        for item in marker_items:  # sequential and last — the parallel bulk gives no ordering
            _upload_one(item)

        # Only after the complete upload: a stale shard numbering left behind would otherwise ride
        # into the new dataset (an index-less reader globs the prefix).
        stale = sorted(existing_keys - {f"{full_key}/{rel}" for _, rel, _ in files_to_upload})
        for stale_key in stale:
            self._client.delete_object(Bucket=self.bucket, Key=stale_key)
        if stale:
            logger.info(f"Swept {len(stale)} stale object(s) from {s3_uri}")

        size_mb = total_size / (1024 * 1024)
        logger.info(f"Uploaded folder ({len(files_to_upload)} files, {size_mb:.2f} MB) to {s3_uri}")

        return s3_uri

    def download_folder(
        self,
        key: str,
        local_path: str,
        subfolder: str | None = None,
        overwrite: bool = True,
        show_progress: bool = False,
    ) -> str:
        """Download an S3 prefix to local_path, returning that path.

        An existing local_path is removed first when overwrite is True, else raises
        FileExistsError. Per-file transfers use limited concurrency (self._transfer_config)
        to avoid connection-pool overflow under parallel workers.
        """
        full_key = self._get_full_key(key, subfolder)
        s3_uri = f"s3://{self.bucket}/{full_key}"

        if not self.exists(key, subfolder):
            raise FileNotFoundError(f"Folder not found: {s3_uri}")

        local_path = os.path.abspath(local_path)

        if os.path.exists(local_path):
            if overwrite:
                logger.info(f"Overwriting existing local folder: {local_path}")
                shutil.rmtree(local_path)
            else:
                raise FileExistsError(f"Local folder already exists: {local_path}. Use overwrite=True to replace.")

        os.makedirs(local_path, exist_ok=True)

        prefix = full_key.rstrip("/") + "/"
        paginator = self._client.get_paginator("list_objects_v2")

        files_to_download = []
        total_size = 0
        for page in paginator.paginate(Bucket=self.bucket, Prefix=prefix):
            for obj in page.get("Contents", []):
                s3_key = obj["Key"]
                relative_path = s3_key[len(prefix) :]
                file_size = obj["Size"]
                files_to_download.append((s3_key, relative_path, file_size))
                total_size += file_size

        # Pre-create the tree single-threaded so the workers never race on os.makedirs of a shared parent.
        for _s3_key, relative_path, _size in files_to_download:
            os.makedirs(os.path.dirname(os.path.join(local_path, relative_path)), exist_ok=True)

        def _download_one(item):
            s3_key, relative_path, file_size = item
            self._client.download_file(
                self.bucket, s3_key, os.path.join(local_path, relative_path), Config=self._transfer_config
            )
            return file_size

        _parallel_s3_transfer(files_to_download, _download_one, desc=f"Downloading {key}", show_progress=show_progress)

        size_mb = total_size / (1024 * 1024)
        logger.info(f"Downloaded folder ({len(files_to_download)} files, {size_mb:.2f} MB) to {local_path}")

        return local_path


_default_client: S3Client | None = None


def _get_default_client() -> S3Client:
    """Get or create the default S3 client."""
    global _default_client
    if _default_client is None:
        _default_client = S3Client()
    return _default_client


def build_s3_uri(key: str, subfolder: str | None = None) -> str:
    """Build a full S3 URI from key (+optional subfolder) under DEFAULT_BUCKET, without
    constructing a client."""
    parts = [p for p in [subfolder, key] if p]
    return f"s3://{DEFAULT_BUCKET}/{'/'.join(parts)}"


def exists(key: str, subfolder: str | None = None) -> bool:
    """Check if key exists (object or prefix) under the default client."""
    return _get_default_client().exists(key, subfolder)


def load_dataset_from_s3_uri(
    s3_uri: str,
    keep_in_memory: bool | None = None,
    use_cache: bool = True,
) -> Dataset | DatasetDict:
    """Load a Dataset from a full S3 URI (recommended entry point) with local caching.

    Wraps S3Client.load_dataset; HALO_S3_DATASET_CACHE_DIR overrides the cache location.
    """
    bucket, key = parse_s3_uri(s3_uri)
    client = S3Client(bucket=bucket)
    return client.load_dataset(key, subfolder=None, keep_in_memory=keep_in_memory, use_cache=use_cache)


def push_dataset_to_s3_uri(dataset: Dataset | DatasetDict, s3_uri: str, overwrite: bool = True) -> str:
    """Push a Dataset/DatasetDict to a full S3 URI (recommended entry point)."""
    bucket, key = parse_s3_uri(s3_uri)
    client = S3Client(bucket=bucket)
    return client.push_dataset(dataset, key, subfolder=None, overwrite=overwrite)
