"""S3 transport: the boto3 ``S3Client`` of one bucket (existence, listing, deletion, folder transfers), the
bounded parallel transfer pool and the default-bucket helpers. Dataset trees go through
:mod:`halo.data.sources.s3_dataset_store`."""

import os
import shutil
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any

import boto3
from boto3.s3.transfer import TransferConfig
from botocore.config import Config
from botocore.exceptions import ClientError
from tqdm import tqdm

from halo.data.sources.dataset_cache import compute_etag_fingerprint
from halo.data.sources.paths import METADATA_FILE
from halo.env import env_int, env_str
from halo.log import info_logger

logger = info_logger(__name__)

# No fallback name: a placeholder would address a real, globally named bucket someone else owns.
DEFAULT_BUCKET = env_str("HALO_S3_DEFAULT_BUCKET")

# The client's connection pool and the threads one multipart transfer takes from it (boto3 takes 10).
_S3_MAX_POOL_CONNECTIONS = 50
_S3_TRANSFER_THREADS = 5
# Past pool // threads workers a large-file folder queues on the pool; clamped, so an explicit 0 means one.
_S3_FOLDER_CONCURRENCY = max(1, env_int("HALO_S3_MAX_FOLDER_CONCURRENCY", 16))

# DeleteObjects' per-request key limit.
_S3_DELETE_BATCH = 1000


def default_bucket() -> str:
    """The bucket a key-only S3 path lives in: ``HALO_S3_DEFAULT_BUCKET``, raising when it is unset."""
    if not DEFAULT_BUCKET:
        raise ValueError(
            "No S3 bucket given: use a full s3://<bucket>/... URI (or --bucket), or set "
            "HALO_S3_DEFAULT_BUCKET to your own bucket."
        )
    return DEFAULT_BUCKET


def parallel_s3_transfer(items: list, transfer_one: Callable[[Any], int], *, desc: str, show_progress: bool) -> None:
    """Run ``transfer_one`` over ``items`` on a bounded thread pool, re-raising the first per-file error;
    the byte counts it returns feed the progress bar."""
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


def _local_files(root_path: str) -> list[tuple[str, str, int]]:
    """``(path, path relative to root_path, size)`` of every file under ``root_path``, in walk order."""
    files = []
    for root, _dirs, names in os.walk(root_path):
        for name in names:
            local_file = os.path.join(root, name)
            files.append((local_file, os.path.relpath(local_file, root_path), os.path.getsize(local_file)))
    return files


@dataclass
class S3Client:
    """One bucket's boto3 client: existence, listing, deletion and folder transfers.

    Credentials default to the standard AWS chain when unset; ``endpoint_url`` targets S3-compatible services.
    """

    bucket: str | None = None  # None takes default_bucket()
    aws_access_key_id: str | None = None
    aws_secret_access_key: str | None = None
    endpoint_url: str | None = None
    region_name: str | None = None
    boto_client: Any = field(default=None, init=False, repr=False)
    transfer_config: Any = field(default=None, init=False, repr=False)

    def __post_init__(self):
        if self.bucket is None:
            self.bucket = default_bucket()
        session_kwargs = {}
        if self.aws_access_key_id:
            session_kwargs["aws_access_key_id"] = self.aws_access_key_id
        if self.aws_secret_access_key:
            session_kwargs["aws_secret_access_key"] = self.aws_secret_access_key
        if self.region_name:
            session_kwargs["region_name"] = self.region_name
        boto_config = Config(
            max_pool_connections=_S3_MAX_POOL_CONNECTIONS, retries={"max_attempts": 3, "mode": "standard"}
        )
        self.transfer_config = TransferConfig(max_concurrency=_S3_TRANSFER_THREADS)
        client_kwargs = {"config": boto_config}
        if self.endpoint_url:
            client_kwargs["endpoint_url"] = self.endpoint_url
        self.boto_client = boto3.Session(**session_kwargs).client("s3", **client_kwargs)

    def get_storage_options(self) -> dict:
        """This client's credentials and endpoint as s3fs ``storage_options`` (empty under the default chain)."""
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

    @staticmethod
    def full_key(key: str, subfolder: str | None = None) -> str:
        """``key`` under an optional ``subfolder``, surrounding slashes normalized."""
        if subfolder:
            subfolder = subfolder.strip("/")
            key = key.strip("/")
            return f"{subfolder}/{key}"
        return key.strip("/")

    def s3_uri(self, key: str, subfolder: str | None = None) -> str:
        """The full ``s3://`` URI of ``key`` (under ``subfolder``) in this client's bucket."""
        return f"s3://{self.bucket}/{self.full_key(key, subfolder)}"

    def object_exists(self, key: str, subfolder: str | None = None) -> bool:
        """Whether an object exists at EXACTLY this key; a prefix with children but no object of its own is
        False, since S3's DeleteObject on a prefix succeeds as a no-op."""
        full_key = self.full_key(key, subfolder)
        try:
            self.boto_client.head_object(Bucket=self.bucket, Key=full_key)
            return True
        except ClientError as e:
            if e.response["Error"]["Code"] != "404":
                raise
            return False

    def exists(self, key: str, subfolder: str | None = None) -> bool:
        """Whether ``key`` exists as an object or a prefix. Only an authoritative absence of the key is False:
        expired credentials, a 403 or a throttle raise, and a missing bucket raises naming the bucket."""
        if self.object_exists(key, subfolder):
            return True

        full_key = self.full_key(key, subfolder)
        try:
            response = self.boto_client.list_objects_v2(
                Bucket=self.bucket, Prefix=full_key.rstrip("/") + "/", MaxKeys=1
            )
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
        """Objects under ``prefix`` as ``Key``/``FullKey``/``Size``/``LastModified``/``Type`` dicts; immediate
        children only when not ``recursive``."""
        full_prefix = self.full_key(prefix, subfolder)
        if full_prefix and not full_prefix.endswith("/"):
            full_prefix += "/"

        objects = []
        kwargs = {"Bucket": self.bucket, "Prefix": full_prefix, "MaxKeys": max_keys}
        if not recursive:
            kwargs["Delimiter"] = "/"
        for page in self.boto_client.get_paginator("list_objects_v2").paginate(**kwargs):
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
        """Delete one object, or with ``recursive`` every object under the slash-anchored prefix; False when
        nothing was deleted."""
        full_key = self.full_key(key, subfolder)

        if recursive:
            # The trailing slash anchors the prefix to children: without it "run1" also matches "run10/...".
            prefix = full_key.rstrip("/") + "/" if full_key else full_key
            keys = [obj["Key"] for obj in self.list_prefix_objects(prefix)]
            if keys:
                self.delete_keys(keys)
                logger.info(f"Deleted {len(keys)} objects from s3://{self.bucket}/{full_key}")
            return bool(keys)
        else:
            try:
                self.boto_client.delete_object(Bucket=self.bucket, Key=full_key)
                logger.info(f"Deleted s3://{self.bucket}/{full_key}")
                return True
            except ClientError as e:
                logger.error(f"Failed to delete s3://{self.bucket}/{full_key}: {e}")
                return False

    def content_entries(self, full_key: str) -> list[tuple[str, str, int]]:
        """``(key, ETag, size)`` of every object under the prefix, raising on any listing failure (where
        :meth:`content_fingerprint` answers ``None`` alike for an outage and an empty prefix)."""
        return [
            (obj["Key"], obj.get("ETag", ""), obj["Size"])
            for obj in self.list_prefix_objects(full_key.rstrip("/") + "/")
        ]

    def content_fingerprint(self, full_key: str) -> str | None:
        """The aggregate ETag fingerprint of the prefix, or ``None`` ("cannot validate", so the local cache is
        served) when S3 is unreachable or the prefix lists empty."""
        try:
            return compute_etag_fingerprint(self.content_entries(full_key))
        except Exception as e:  # any botocore/network error → cannot validate, keep the cache usable
            logger.info(
                f"S3 unreachable for cache validation of s3://{self.bucket}/{full_key} "
                f"({type(e).__name__}: {e}); serving the local cache as-is."
            )
            return None

    def delete_keys(self, keys: list[str]) -> None:
        """Delete ``keys`` in listing order, batched to :data:`_S3_DELETE_BATCH` per request."""
        for start in range(0, len(keys), _S3_DELETE_BATCH):
            batch = [{"Key": key} for key in keys[start : start + _S3_DELETE_BATCH]]
            self.boto_client.delete_objects(Bucket=self.bucket, Delete={"Objects": batch})

    def list_prefix_objects(self, prefix: str) -> list[dict[str, Any]]:
        """Raw object listing under an exact string prefix (no trailing-slash forcing)."""
        entries = []
        paginator = self.boto_client.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=self.bucket, Prefix=prefix):
            entries.extend(page.get("Contents", []))
        return entries

    def push_folder(
        self,
        local_path: str,
        key: str,
        subfolder: str | None = None,
        overwrite: bool = True,
        show_progress: bool = False,
    ) -> str:
        """Upload a local directory tree to S3, returning the S3 URI; an existing prefix raises unless
        ``overwrite``. Overwriting never clears the prefix up front, and the ``metadata.json`` marker is
        deleted first and uploaded last, so a torn upload reads as no preprocessed dataset."""
        local_path = os.path.abspath(local_path)
        if not os.path.isdir(local_path):
            raise ValueError(f"Local path is not a directory: {local_path}")

        full_key = self.full_key(key, subfolder)
        s3_uri = self.s3_uri(key, subfolder)

        existing_keys: set[str] = set()
        if self.exists(key, subfolder):
            if not overwrite:
                raise FileExistsError(f"Folder already exists: {s3_uri}. Use overwrite=True to replace.")
            logger.info(f"Overwriting existing folder at {s3_uri}")
            # Uncapped: the stale sweep below deletes what this set does NOT contain, so a listing
            # truncated at any max_keys would silently re-upload-and-keep objects it never saw.
            existing_keys = {entry["Key"] for entry in self.list_prefix_objects(f"{full_key}/")}

        files_to_upload = _local_files(local_path)
        marker_items = [item for item in files_to_upload if item[1] == METADATA_FILE]
        bulk_items = [item for item in files_to_upload if item[1] != METADATA_FILE]
        marker_key = f"{full_key}/{METADATA_FILE}"
        if marker_key in existing_keys:
            self.boto_client.delete_object(Bucket=self.bucket, Key=marker_key)

        def _upload_one(item):
            local_file, relative_path, file_size = item
            self.boto_client.upload_file(
                local_file, self.bucket, f"{full_key}/{relative_path}", Config=self.transfer_config
            )
            return file_size

        parallel_s3_transfer(bulk_items, _upload_one, desc=f"Uploading to {key}", show_progress=show_progress)
        for item in marker_items:  # sequential and last — the parallel bulk gives no ordering
            _upload_one(item)

        # Only after the complete upload: a stale shard numbering left behind would otherwise ride
        # into the new dataset (an index-less reader globs the prefix).
        stale = sorted(existing_keys - {f"{full_key}/{rel}" for _, rel, _ in files_to_upload})
        for stale_key in stale:
            self.boto_client.delete_object(Bucket=self.bucket, Key=stale_key)
        if stale:
            logger.info(f"Swept {len(stale)} stale object(s) from {s3_uri}")

        size_mb = sum(size for _, _, size in files_to_upload) / (1024 * 1024)
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
        """Download an S3 prefix to ``local_path``, returning it; an existing ``local_path`` is replaced when
        ``overwrite``, else raises."""
        full_key = self.full_key(key, subfolder)
        s3_uri = self.s3_uri(key, subfolder)

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
        files_to_download = [
            (obj["Key"], obj["Key"][len(prefix) :], obj["Size"]) for obj in self.list_prefix_objects(prefix)
        ]
        total_size = sum(size for _key, _relative_path, size in files_to_download)

        # Pre-create the tree single-threaded so the workers never race on os.makedirs of a shared parent.
        for _s3_key, relative_path, _size in files_to_download:
            os.makedirs(os.path.dirname(os.path.join(local_path, relative_path)), exist_ok=True)

        def _download_one(item):
            s3_key, relative_path, file_size = item
            self.boto_client.download_file(
                self.bucket, s3_key, os.path.join(local_path, relative_path), Config=self.transfer_config
            )
            return file_size

        parallel_s3_transfer(files_to_download, _download_one, desc=f"Downloading {key}", show_progress=show_progress)
        size_mb = total_size / (1024 * 1024)
        logger.info(f"Downloaded folder ({len(files_to_download)} files, {size_mb:.2f} MB) to {local_path}")
        return local_path


def build_s3_uri(key: str, subfolder: str | None = None) -> str:
    """Build a full S3 URI from key (+optional subfolder) under :func:`default_bucket`, without
    constructing a client."""
    return f"s3://{default_bucket()}/{S3Client.full_key(key, subfolder)}"
