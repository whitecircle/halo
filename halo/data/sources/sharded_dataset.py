"""Sharded dataset loader: each rank loads only its assigned shards (Megatron-LM style)."""

import hashlib
import json
import os

from datasets import Dataset, DatasetDict, concatenate_datasets, load_from_disk

from halo.data.preprocessed.shard_index import SHARD_INDEX_FILE, ShardIndex, ShardInfo
from halo.data.sources.dataset_cache import (
    HALO_S3_DATASET_CACHE_DIR,
    compute_etag_fingerprint,
    s3_cache_key,
    serve_cached_tree,
)
from halo.data.sources.paths import parse_dataset_source, s3_object_key
from halo.data.sources.s3_client import S3Client
from halo.data.sources.s3_dataset_store import download_dataset_tree, read_control_json_with_cache
from halo.log import info_logger

logger = info_logger(__name__)


class ShardedDatasetLoader:
    """A data-parallel rank's slice of a sharded dataset: shards spread evenly, the remainder to the first
    ranks (10 shards over 3 ranks -> [4, 3, 3])."""

    def __init__(self, dataset_path: str, dp_rank: int, dp_size: int):
        self.dataset_path = dataset_path
        self.dp_rank = dp_rank
        self.dp_size = dp_size
        self.source_type, self.bucket, self.key = parse_dataset_source(dataset_path)
        self._shard_indices: dict[str, ShardIndex] = {}
        # Latched by an UNREACHABLE S3 only: a fully cached offline run must not pay a connection
        # timeout per shard. An empty listing is a real absence, not a transport fault.
        self._s3_fingerprint_unavailable = False
        # Built on first use: an S3Client opens boto3 sessions a local-source loader never needs.
        self._s3_client: S3Client | None = None

    @staticmethod
    def is_sharded_dataset(path: str) -> bool:
        """Whether ``path`` has a ``train/shard_index.json``, read on S3 through the control-file mirror.

        Never raises (the caller agrees it across ranks, so a rank-local raise would hang the peers): a
        failure the mirror cannot serve errs toward "non-sharded" with a warning.
        """
        source_type, bucket, key = parse_dataset_source(path)
        if source_type == "s3":
            index_key = s3_object_key(key, "train", SHARD_INDEX_FILE)
            try:
                read_control_json_with_cache(bucket, index_key)
                return True
            except FileNotFoundError:
                return False
            except Exception as e:  # transient creds/403/throttle must not kill one rank of a job
                logger.warning(
                    f"S3 sharded-dataset probe for s3://{bucket}/{index_key} failed ({type(e).__name__}: {e}); "
                    f"assuming non-sharded. Will attempt local cache fallback."
                )
                return False
        if source_type == "local":
            return os.path.exists(os.path.join(path, "train", SHARD_INDEX_FILE))
        return False

    def load(self) -> DatasetDict:
        """This rank's train and test splits; a split without a shard index is skipped, a listed shard the
        source lacks raises."""
        result = DatasetDict()
        try:
            self._load_shard_index("train")
        except FileNotFoundError:
            logger.warning("Train split not found")
        else:
            result["train"] = self.load_split("train")
        try:
            self._load_shard_index("test")
        except FileNotFoundError:
            logger.info("Test split not found (this is often expected)")
        else:
            result["test"] = self.load_split("test")
        return result

    def load_split(self, split: str = "train") -> Dataset:
        """Load and concatenate this rank's assigned shards for the split."""
        # Both verdicts are read off the index, which every rank loads itself — the same answer on
        # every rank, so a refusal here cannot skew a collective.
        index = self._load_shard_index(split)
        if index.num_shards == 0 or not index.shards:
            self._reject_unusable_split(
                split,
                f"Sharded dataset '{split}' at {self.dataset_path} lists no shards "
                f"(num_shards={index.num_shards}, {len(index.shards)} shard entries): there is "
                f"nothing to load. Re-preprocess the dataset with "
                f"scripts/before_training/prepare_dataset.py.",
            )
        # Fewer shards than ranks starves a rank, and the train length-equalizer all_reduce(MIN)s to 0.
        elif index.num_shards < self.dp_size:
            self._reject_unusable_split(
                split,
                f"Sharded dataset '{split}' has {index.num_shards} shards but "
                f"data_parallel_size={self.dp_size}: ranks >= {index.num_shards} would "
                f"receive zero examples. Re-preprocess with --num_shards >= {self.dp_size} "
                f"(the data-parallel degree), or leave --num_shards at its default of 1 for an "
                f"unsharded dataset that every rank loads whole.",
            )

        assigned_shards = self.get_assigned_shards(split)
        if not assigned_shards:
            return self._starved_split(split, index)
        logger.info(
            f"Rank {self.dp_rank}/{self.dp_size} loading {len(assigned_shards)} "
            f"shards for {split} (shards {assigned_shards[0].id}-{assigned_shards[-1].id})"
        )

        shard_datasets = [self._load_shard(shard) for shard in assigned_shards]
        return shard_datasets[0] if len(shard_datasets) == 1 else concatenate_datasets(shard_datasets)

    def get_assigned_shards(self, split: str = "train") -> list[ShardInfo]:
        """ShardInfo list assigned to this rank for the split."""
        index = self._load_shard_index(split)
        start, end = self._compute_shard_assignment(index.num_shards)
        return index.shards[start:end]

    def total_examples(self, split: str = "train") -> int:
        """Total examples in a split across all ranks."""
        return self._load_shard_index(split).total_examples

    def shard_index_digests(self) -> dict[str, str]:
        """Content digest of every shard index this loader read, by split — what all ranks must share."""
        return {
            split: hashlib.md5(json.dumps(index.to_dict(), sort_keys=True).encode()).hexdigest()
            for split, index in sorted(self._shard_indices.items())
        }

    def _load_shard_index(self, split: str) -> ShardIndex:
        """Load shard index for a split."""
        if split in self._shard_indices:
            return self._shard_indices[split]

        if self.source_type == "s3":
            # The mirrored control-file read: a warm-cache relaunch during an S3/SSO outage takes the local mirror.
            index_key = s3_object_key(self.key, split, SHARD_INDEX_FILE)
            index = ShardIndex.from_dict(read_control_json_with_cache(self.bucket, index_key))
        else:
            index = ShardIndex.load(os.path.join(self.dataset_path, split, SHARD_INDEX_FILE))

        self._shard_indices[split] = index
        return index

    def _reject_unusable_split(self, split: str, message: str) -> None:
        """Raise for ``train``, which the length equalizer would silently zero, and warn for an eval split."""
        if split == "train":
            raise ValueError(message)
        logger.warning(message)

    def _starved_split(self, split: str, index: ShardIndex) -> Dataset:
        """An empty split WITH the split's schema (its first shard, no rows) for a rank the assignment left
        with nothing: downstream ``column_names`` tests gate coordinated operations and must answer alike."""
        logger.warning(
            f"Rank {self.dp_rank} has no shards assigned for {split} "
            f"({index.num_shards} shards over {self.dp_size} ranks); serving an empty split."
        )
        if not index.shards:
            return Dataset.from_dict({})
        return self._load_shard(index.shards[0]).select([])

    def _load_shard(self, shard: ShardInfo) -> Dataset:
        """One shard, from whichever source this dataset lives on."""
        if self.source_type == "s3":
            return self._load_shard_from_s3(shard)
        return load_from_disk(os.path.join(self.dataset_path, shard.path))

    def _load_shard_from_s3(self, shard: ShardInfo) -> Dataset:
        """One shard through its own cache keyed by md5(bucket/key/shard.path) (:func:`serve_cached_tree`):
        identical shards download once, and a shard re-preprocessed in place on S3 is re-downloaded."""
        shard_key = f"{self.key.rstrip('/')}/{shard.path}"
        shard_uri = f"s3://{self.bucket}/{shard_key}"
        cache_path = os.path.join(HALO_S3_DATASET_CACHE_DIR, "shards", s3_cache_key(self.bucket, shard_key), "shard")
        serve_cached_tree(
            cache_path,
            shard_uri,
            lambda tmp_path: download_dataset_tree(self._s3(), shard_key, tmp_path),
            lambda: self._shard_fingerprint(shard_key),
            stale_message=(
                f"Cached shard {shard_uri} no longer matches its S3 content fingerprint (the shard changed on S3, "
                f"or it was cached by a build that fingerprinted shards differently); re-downloading."
            ),
        )
        # Memory-map, NOT keep_in_memory: at scale a rank holds tens of shards and would blow host RAM.
        return load_from_disk(cache_path)

    def _s3(self) -> S3Client:
        """The source bucket's client, built on first use."""
        if self._s3_client is None:
            self._s3_client = S3Client(bucket=self.bucket)
        return self._s3_client

    def _shard_fingerprint(self, shard_key: str) -> str | None:
        """The aggregate ETag fingerprint of a shard prefix, or ``None`` ("cannot validate"); an unreachable
        S3, unlike an empty listing, stops the remaining probes."""
        if self._s3_fingerprint_unavailable:
            return None
        try:
            # Client construction is inside the probe's guard: it resolves credentials and a region,
            # so a warm-cache relaunch without either would otherwise die here instead of serving.
            return compute_etag_fingerprint(self._s3().content_entries(shard_key))
        except Exception as e:  # any transport/credential fault, cache stays servable
            self._s3_fingerprint_unavailable = True
            logger.warning(
                f"S3 unreachable for shard cache validation of s3://{self.bucket}/{shard_key} "
                f"({type(e).__name__}: {e}); serving cached shards as-is and skipping the fingerprint "
                f"probe for the rest of this process."
            )
            return None

    def _compute_shard_assignment(self, num_shards: int) -> tuple[int, int]:
        """Shard range (start, end-exclusive) for this rank."""
        shards_per_rank = num_shards // self.dp_size
        remainder = num_shards % self.dp_size

        if self.dp_rank < remainder:
            start = self.dp_rank * (shards_per_rank + 1)
            end = start + shards_per_rank + 1
        else:
            start = remainder * (shards_per_rank + 1) + (self.dp_rank - remainder) * shards_per_rank
            end = start + shards_per_rank

        return (start, end)
