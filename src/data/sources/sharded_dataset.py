"""Sharded dataset loader: each rank loads only its assigned shards (Megatron-LM style)."""

import logging
import os

from datasets import Dataset, DatasetDict, concatenate_datasets, load_from_disk

from src.data.shard_index import SHARD_INDEX_FILE, ShardIndex, ShardInfo
from src.data.sources.dataset_cache import (
    DOWNLOAD_COMPLETE_MARKER,
    HALO_S3_DATASET_CACHE_DIR,
    compute_etag_fingerprint,
    publish_cached_download,
    s3_cache_key,
)
from src.data.sources.paths import parse_dataset_source
from src.data.sources.s3_client import S3Client, has_control_json_mirror, read_control_json_with_cache

logger = logging.getLogger(__name__)


class ShardedDatasetLoader:
    """Load a rank's slice of a sharded dataset.

    Shards are distributed evenly across ranks, remainder going to the first
    ranks (10 shards, 3 ranks -> [4, 3, 3]).
    """

    def __init__(
        self,
        dataset_path: str,
        global_rank: int,
        world_size: int,
    ):
        self.dataset_path = dataset_path
        self.global_rank = global_rank
        self.world_size = world_size

        self.source_type, self.bucket, self.key = parse_dataset_source(dataset_path)

        self._shard_indices: dict[str, ShardIndex] = {}
        self._datasets: dict[str, Dataset] = {}
        # Latched only by an unreachable S3, so a fully cached offline run does not pay a connection
        # timeout per shard. An empty listing is a real absence, not a transport fault.
        self._s3_fingerprint_unavailable = False
        # Built on first use: an S3Client opens boto3 sessions a local-source loader never needs.
        self._s3_client: S3Client | None = None

    def _load_shard_index(self, split: str) -> ShardIndex:
        """Load shard index for a split."""
        if split in self._shard_indices:
            return self._shard_indices[split]

        if self.source_type == "s3":
            index = self._load_shard_index_from_s3(split)
        else:
            index = self._load_shard_index_from_local(split)

        self._shard_indices[split] = index
        return index

    def _load_shard_index_from_s3(self, split: str) -> ShardIndex:
        """Load shard index from S3 through the mirrored control-file read: a warm-cache relaunch
        during an S3/SSO outage takes the local mirror instead of dying on the index fetch."""
        data = read_control_json_with_cache(self.bucket, f"{self.key.rstrip('/')}/{split}/{SHARD_INDEX_FILE}")
        return ShardIndex.from_dict(data)

    def _load_shard_index_from_local(self, split: str) -> ShardIndex:
        """Load shard index from local path."""
        index_path = os.path.join(self.dataset_path, split, SHARD_INDEX_FILE)
        if not os.path.exists(index_path):
            raise FileNotFoundError(f"Shard index not found at {index_path}")
        return ShardIndex.load(index_path)

    def _compute_shard_assignment(self, num_shards: int) -> tuple[int, int]:
        """Shard range (start, end-exclusive) for this rank."""
        if num_shards == 0:
            return (0, 0)

        shards_per_rank = num_shards // self.world_size
        remainder = num_shards % self.world_size

        if self.global_rank < remainder:
            start = self.global_rank * (shards_per_rank + 1)
            end = start + shards_per_rank + 1
        else:
            start = remainder * (shards_per_rank + 1) + (self.global_rank - remainder) * shards_per_rank
            end = start + shards_per_rank

        return (start, end)

    def get_assigned_shards(self, split: str = "train") -> list[ShardInfo]:
        """ShardInfo list assigned to this rank for the split."""
        index = self._load_shard_index(split)
        start, end = self._compute_shard_assignment(index.num_shards)
        return index.shards[start:end]

    def _shard_fingerprint(self, shard_key: str) -> str | None:
        """Best-effort content identity (aggregate ETag fingerprint) of a shard prefix on S3.

        The dataset cache's own probe, so a shard and a whole-dataset download are validated by
        identical fingerprints. ``None`` means the cache cannot be validated (S3 unreachable, or an
        empty listing) and is served as-is; only an unreachable S3 stops the remaining probes.
        """
        if self._s3_fingerprint_unavailable:
            return None
        try:
            # Client construction is inside the probe's guard: it resolves credentials and a region,
            # so a warm-cache relaunch without either would otherwise fail here instead of serving.
            if self._s3_client is None:
                self._s3_client = S3Client(bucket=self.bucket)
            return compute_etag_fingerprint(self._s3_client.content_entries(shard_key))
        except Exception as e:  # any transport/credential fault, cache stays servable
            self._s3_fingerprint_unavailable = True
            logger.warning(
                f"S3 unreachable for shard cache validation of s3://{self.bucket}/{shard_key} "
                f"({type(e).__name__}: {e}); serving cached shards as-is and skipping the fingerprint "
                f"probe for the rest of this process."
            )
            return None

    def _load_shard_from_s3(self, shard: ShardInfo) -> Dataset:
        """Load a single shard from S3 with a local per-shard cache.

        Keyed by md5(bucket/key/shard.path) under a FileLock plus completion marker: disjoint shards
        download in parallel, identical shards (TP/CP siblings, reruns) download once and are reused,
        and a complete cache lets a relaunch survive an S3/SSO outage offline. A best-effort
        staleness probe compares the marker's content fingerprint against live S3 (in-place
        re-preprocessing to the same URI changes the ETags) and re-downloads on mismatch.
        """
        shard_key = f"{self.key.rstrip('/')}/{shard.path}"
        shard_uri = f"s3://{self.bucket}/{shard_key}"
        cache_dir = os.path.join(HALO_S3_DATASET_CACHE_DIR, "shards", s3_cache_key(self.bucket, shard_key))
        cache_path = os.path.join(cache_dir, "shard")
        complete_marker = os.path.join(cache_path, DOWNLOAD_COMPLETE_MARKER)
        os.makedirs(cache_dir, exist_ok=True)

        live_fingerprint = self._shard_fingerprint(shard_key) if os.path.exists(complete_marker) else None
        publish_cached_download(
            cache_path,
            os.path.join(cache_dir, ".lock"),
            shard_uri,
            lambda tmp_path: load_from_disk(shard_uri).save_to_disk(tmp_path),
            live_fingerprint,
            lambda: self._shard_fingerprint(shard_key),
            stale_message=(
                f"Cached shard {shard_uri} no longer matches its S3 content fingerprint (the shard changed on S3, "
                f"or it was cached by a build that fingerprinted shards differently); re-downloading."
            ),
        )
        # Memory-map rather than keep_in_memory: at scale a rank holds tens of shards, which would
        # exhaust host RAM.
        return load_from_disk(cache_path)

    def _load_shard_from_local(self, shard: ShardInfo) -> Dataset:
        """Load a single shard from local path."""
        shard_path = os.path.join(self.dataset_path, shard.path)
        return load_from_disk(shard_path)

    def _load_shard(self, shard: ShardInfo) -> Dataset:
        """One shard, from whichever source this dataset lives on."""
        if self.source_type == "s3":
            return self._load_shard_from_s3(shard)
        return self._load_shard_from_local(shard)

    def _refuse_unusable_split(self, split: str, message: str) -> None:
        """Raise for ``train``, warn for any other split.

        A small eval split whose shards do not reach every rank is legitimate; a train split in the
        same state zeroes the world through the length equalizer.
        """
        if split == "train":
            raise ValueError(message)
        logger.warning(message)

    def _starved_split(self, split: str, index: ShardIndex) -> Dataset:
        """Empty split, carrying the split's schema, for a rank the shard assignment left with nothing.

        A column-less ``Dataset.from_dict({})`` makes every downstream ``column_names`` test answer
        differently here than on its peers, and those tests gate coordinated operations. Reading the
        split's first shard and selecting no rows keeps the shape rank-uniform; a split listing no
        shards has no schema to preserve and starves every rank identically.
        """
        logger.warning(
            f"Rank {self.global_rank} has no shards assigned for {split} "
            f"({index.num_shards} shards over {self.world_size} ranks); serving an empty split."
        )
        if not index.shards:
            return Dataset.from_dict({})
        return self._load_shard(index.shards[0]).select([])

    def load_split(self, split: str = "train") -> Dataset:
        """Load and concatenate this rank's assigned shards for the split."""
        if split in self._datasets:
            return self._datasets[split]

        # Both verdicts are read off the index, which every rank loads itself, so a refusal here is
        # reached on every rank and cannot skew a collective.
        index = self._load_shard_index(split)
        if index.num_shards == 0 or not index.shards:
            self._refuse_unusable_split(
                split,
                f"Sharded dataset '{split}' at {self.dataset_path} lists no shards "
                f"(num_shards={index.num_shards}, {len(index.shards)} shard entries): there is "
                f"nothing to load. Re-preprocess the dataset with "
                f"scripts/before_training/prepare_dataset.py.",
            )
        # Fewer shards than ranks starves a rank, and the train length equalizer all_reduce(MIN)s to 0.
        elif index.num_shards < self.world_size:
            self._refuse_unusable_split(
                split,
                f"Sharded dataset '{split}' has {index.num_shards} shards but "
                f"data_parallel_size={self.world_size}: ranks >= {index.num_shards} would "
                f"receive zero examples. Re-preprocess with --num-shards >= {self.world_size} "
                f"(the data-parallel degree), or use a non-sharded dataset.",
            )

        assigned_shards = self.get_assigned_shards(split)

        if not assigned_shards:
            return self._starved_split(split, index)

        logger.info(
            f"Rank {self.global_rank}/{self.world_size} loading {len(assigned_shards)} "
            f"shards for {split} (shards {assigned_shards[0].id}-{assigned_shards[-1].id})"
        )

        shard_datasets = [self._load_shard(shard) for shard in assigned_shards]

        dataset = shard_datasets[0] if len(shard_datasets) == 1 else concatenate_datasets(shard_datasets)

        self._datasets[split] = dataset
        logger.info(f"Rank {self.global_rank} loaded {len(dataset)} examples for {split}")

        return dataset

    def load(self) -> DatasetDict:
        """Load train and test splits for this rank (missing splits are skipped)."""
        result = DatasetDict()

        try:
            result["train"] = self.load_split("train")
        except FileNotFoundError:
            logger.warning("Train split not found")

        try:
            result["test"] = self.load_split("test")
        except FileNotFoundError:
            logger.info("Test split not found (this is often expected)")

        return result

    @staticmethod
    def is_sharded_dataset(path: str) -> bool:
        """True if path has a train/shard_index.json.

        The S3 probe reads the index through the mirrored control-file read, so a warm run relaunched
        during an outage still classifies as sharded off the local mirror. A live 404 is False and
        drops the mirror; any other failure without a mirror errs toward non-sharded with a warning.
        This probe never raises: the caller reconciles across ranks, and a rank-local raise would
        skip that all-reduce and hang its peers.
        """
        source_type, bucket, key = parse_dataset_source(path)

        if source_type == "s3":
            index_key = f"{key.rstrip('/')}/train/{SHARD_INDEX_FILE}"
            try:
                read_control_json_with_cache(bucket, index_key)
                return True
            except FileNotFoundError:
                return False
            except Exception as e:  # a transient creds/403/throttle must not kill one rank of a job
                if has_control_json_mirror(bucket, index_key):
                    logger.warning(
                        f"S3 unreachable for sharded-dataset probe of s3://{bucket}/{index_key} "
                        f"({type(e).__name__}: {e}); the local control-file mirror says sharded."
                    )
                    return True
                logger.warning(
                    f"S3 sharded-dataset probe for s3://{bucket}/{index_key} failed ({type(e).__name__}: {e}); "
                    f"assuming non-sharded. Will attempt local cache fallback."
                )
                return False

        elif source_type == "local":
            index_path = os.path.join(path, "train", SHARD_INDEX_FILE)
            return os.path.exists(index_path)

        return False

    def get_total_examples(self, split: str = "train") -> int:
        """Total examples in a split across all ranks."""
        index = self._load_shard_index(split)
        return index.total_examples
