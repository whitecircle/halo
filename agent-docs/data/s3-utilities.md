# S3 Utilities

Reading and writing HuggingFace `Dataset`/`DatasetDict` objects and arbitrary folders in S3, with a
local cache for distributed training and a CLI. The default bucket comes from
`HALO_S3_DEFAULT_BUCKET` — set it to your own bucket. Four modules plus the CLI and its commands, split so a consumer
takes on only the dependency it needs:

| File | Holds |
|---|---|
| `halo/data/sources/paths.py` | source/destination classification — string rules, no boto3 |
| `halo/data/sources/dataset_cache.py` | the local cache-publish protocol — `os`/`shutil`/`filelock`, the fetch injected |
| `halo/data/sources/s3_client.py` | S3 transport: the boto3 `S3Client`, the bounded parallel transfer pool, the default-bucket helpers |
| `halo/data/sources/s3_dataset_store.py` | dataset trees on S3: the push protocol, the cached load, the mirrored control-file reads |
| `halo/data/sources/s3_commands.py` | the CLI's commands — push, download, list, exists, delete — over one `S3Client` |
| `scripts/before_training/s3.py` | the CLI's flags |

## Dataset path formats

`parse_dataset_source` (`halo/data/sources/paths.py` — string rules, no boto3, so the
loader and the scripts classify a path without an S3 stack) sorts a path into
`('s3' | 'hf_hub' | 'local', bucket, key)`:

- **S3 URI** (recommended): `"s3://my-bucket/datasets/my_dataset/train"`.
- **HuggingFace Hub**: `"HuggingFaceH4/ultrachat_200k"` — exactly one slash, no part starting with a dot. Optional
  suffixes: `:config` picks a dataset config (`"openai/gsm8k:main"`), `@split` picks one split when
  the repo has no plain `train` (`"HuggingFaceH4/ultrachat_200k@train_sft"`). The key is returned
  **verbatim including the suffixes**; they are stripped later, at load.
- **Local**: everything else — an absolute, `./` or `../` path, a `file://` URI (prefix stripped), any
  path ending in `.jsonl` / `.json` / `.parquet` / `.arrow` / `.csv`, and a relative path with no slash or
  more than one.

A Hub-shaped path that also names a local directory (`preprocessed/my_dataset` beside a directory of
that name) is refused by every reader, training, `prepare_dataset.py --input`, the eval runner and the
deduplication tool alike, rather than guessed: write `./preprocessed/my_dataset` or an absolute path for the directory, without a
`:config` / `@split` suffix (a saved split is its own subdirectory).
`reject_ambiguous_hub_source` is the check; on training it runs inside the rank-joined load and probe, so
a directory present on some nodes only fails every rank with the same cause.

`parse_dataset_destination` is its write-side twin and classifies a bare `org/name` as **local**:
guessing the Hub is acceptable for a read, but a write only targets the Hub on an explicit
`hf://org/name`. Combining sources and required schemas: [Dataset Formats](dataset-formats.md).

## Caching

S3 datasets are cached locally to avoid re-downloading. Cache directory:
`$HALO_DATA_ROOT/s3_datasets` (default `~/.cache/halo/s3_datasets`); point `HALO_DATA_ROOT` at a
large mount, or override this cache alone with `HALO_S3_DATASET_CACHE_DIR`:

```bash
HALO_S3_DATASET_CACHE_DIR="$LARGE_VOLUME"/s3_datasets torchrun --nproc_per_node=8 \
    scripts/training/sft.py examples/sft/qwen3/qwen3-4b-ultrachat.yaml
```

How it works:

- **Cache key**: MD5 of `bucket/key` → one subdirectory per dataset.

- **File locking**: `filelock.FileLock` prevents duplicate downloads across processes.

- **Completion marker**: a `.download_complete` file marks a valid cache; it records the source URI and an ETag content fingerprint of the S3 prefix. Downloads stage into a unique temp dir and publish with an atomic rename, so a crashed writer never leaves a half-valid cache. A stale tree that cannot be renamed aside (a permission error, say) raises and discards the fresh download, rather than serving the stale rows.

- **One copy on disk**: the tree downloads straight into that temp dir through the client's parallel transfer (`download_dataset_tree`), so no second copy waits under `TMPDIR` for the process to exit. A tree missing its load gates (a torn promote) raises there and its partial download is removed, so it is never published.

- **Staleness check**: on a cache hit the marker's fingerprint is re-validated against live S3 — an in-place re-push to the same URI changes the ETags, and a mismatch triggers a re-download. A marker carrying no fingerprint is trusted and upgraded in place.

- **Offline-resilient hit**: a complete cache skips the existence probe and the download, and the staleness probe fails soft.

    An unreachable S3 (no credentials, expired SSO, outage) reads as "cannot validate" and the cache is served. A *cold* cache with S3 unreachable raises instead, naming the cache path and the way out.

- **Control-file mirror**: `metadata.json` and `shard_index.json` reads (`read_control_json_with_cache`) write a local mirror under `$HALO_S3_DATASET_CACHE_DIR/control/` on every live read and serve it when S3 is unreachable.

    Preprocessed/sharded classification therefore survives an outage too, not just the dataset bytes. Live wins; a live 404 drops the mirror, so a re-pushed raw dataset cannot keep its old classification offline.

    A training script reads them once per launch: global rank 0 reads the stamp and both shard indexes (`share_s3_control_files`), every rank's later probes and loads use that read, and each node's local rank 0 writes the mirror. A file rank 0 cannot read is left to every rank's own read.

The per-shard cache used by sharded pre-processed datasets (`halo/data/sources/sharded_dataset.py`) uses the same marker + fingerprint scheme. The first probe that finds S3 unreachable logs a warning and skips the remaining shard probes, so a fully cached offline run pays a few control-read timeouts (the sharded probe and per-split index each try live first), not one per shard.

An empty prefix listing is a real absence, not a transport fault, and does not disarm the probes.

Each rank's shard loads join across the world on the c10d store (`DIST_STORE_TIMEOUT_HOURS`), so one rank's failed or slow first-run download surfaces its real cause on every rank instead of an NCCL watchdog dump.

Force a re-download with `load_dataset_from_s3_uri(..., use_cache=False)`. Clear it by removing the resolved cache directory (`$HALO_DATA_ROOT/s3_datasets` unless overridden above).

On multi-node with a shared filesystem (NFS/Lustre) the cache is shared — only one node downloads. With per-node local storage each node caches independently; set `DIST_SHARED_FILESYSTEM=0` (see [Filesystem Handling](filesystem-handling.md)). Since a node can serve its own stale cache while S3 is unreachable from it, the load agrees the rows across ranks and fails the launch on a disagreement ([Distributed Data Loading](../parallelism/data-loading.md)).

## Authentication

`S3Client` uses boto3's default credential chain (env vars, `~/.aws/credentials`, SSO) — bring your
own account and bucket; setup and credential failures are in [AWS Auth](../infrastructure/aws-auth.md). For an S3-compatible
service or non-default bucket, pass `aws_access_key_id` / `aws_secret_access_key` / `endpoint_url` /
`region_name` directly to `S3Client(...)`.

Retries follow the AWS configuration: `AWS_MAX_ATTEMPTS` / `AWS_RETRY_MODE`, or the profile's
`max_attempts` / `retry_mode`. A key neither sets takes the toolkit default, 4 total attempts in `standard`
mode; raise both for a many-node fetch that meets S3 throttling (`SlowDown`).

## Python API

Dataset trees go through `halo.data.sources.s3_dataset_store`; everything else is `halo.data.sources.s3_client`
(see the modules for full signatures and defaults):

| Function | Purpose |
|----------|---------|
| `load_dataset_from_s3_uri(s3_uri, keep_in_memory=None, use_cache=True)` | Load HF dataset by URI; cached with file locking |
| `push_dataset_to_s3_uri(dataset, s3_uri, overwrite=True)` | Push HF dataset by URI: the tree uploads whole to a `.staging-*` sibling prefix, is sealed, then promoted by server-side copy with the load gates (`state.json`/`dataset_dict.json`) deleted first and copied last. A crash leaves either a complete copy at the destination or the sealed push, which `load_dataset_from_s3_uri` falls back to; the next successful push heals and sweeps. One writer per destination at a time |
| `push_dataset(client, dataset, key, ...)` / `load_s3_dataset(client, key, ...)` | The same on an explicit `S3Client` (the URI functions above wrap these) |
| `build_s3_uri(path, subfolder=None, *, given_as)` (`s3_client`) | The S3 URI a CLI path names, without constructing a client: a full `s3://` URI as given, a bare key (+ optional subfolder) under `HALO_S3_DEFAULT_BUCKET`. A bare key with the variable unset raises naming `given_as`, the flag the path came from |

`subfolder` is prepended to a bare key; key paths may be nested. Everything else — folder transfers, listings,
deletes — is an `S3Client` method, called on a client built for the bucket in question.

### S3Client

For a non-default bucket or an S3-compatible service:

```python
from halo.data.sources.s3_client import S3Client
from halo.data.sources.s3_dataset_store import push_dataset

client = S3Client(bucket="<bucket>")
push_dataset(client, dataset, "data")

# MinIO / S3-compatible
client = S3Client(
    bucket="<bucket>",
    endpoint_url="http://localhost:9000",
    aws_access_key_id="minioadmin",
    aws_secret_access_key="minioadmin",
)
```

`S3Client` carries the transport surface: `exists` / `object_exists`, `list_objects`, `delete`, and the
folder transfers:

| Method | Purpose |
|--------|---------|
| `push_folder(local_path, key, subfolder=None, overwrite=True, show_progress=False)` | Upload a local folder. Overwrites **in place** (no prefix delete): a top-level `metadata.json` is removed first and re-uploaded last, and objects the new tree lacks are swept only after the upload succeeds |
| `download_folder(key, local_path, subfolder=None, overwrite=True, show_progress=False)` | Download a folder |

Both transfer files concurrently (a many-shard dataset or checkpoint moves in parallel, not
file-by-file); `HALO_S3_MAX_FOLDER_CONCURRENCY` sets the files-in-flight bound (default 16).

Training scripts call `load_datasets_auto()`, which probes for a pre-processed dataset first, then
delegates to `load_datasets()`: detect the source, download to cache if needed, file-lock against
duplicate downloads, memory-map the Arrow files.

## CLI

```bash
python scripts/before_training/s3.py push ./data project/experiment/data
python scripts/before_training/s3.py download project/experiment/data ./data
python scripts/before_training/s3.py list my_project/ --recursive
python scripts/before_training/s3.py delete my_folder/data.json --yes
python scripts/before_training/s3.py delete my_folder --recursive --yes
python scripts/before_training/s3.py download s3://my-bucket/project/experiment/data ./data
```

Commands: `push`, `download`, `list`, `exists`, `delete`. Every command takes `--subfolder/-s`,
`--bucket/-b` (default `HALO_S3_DEFAULT_BUCKET`; a bare key needs one of the two) and `--verbose/-v`. A
key (or `list`'s prefix) given as a full `s3://bucket/key` URI names its own bucket, so it refuses
`--subfolder` and a different `--bucket`. The other flags are per command:

| Flag | Commands | Notes |
|---|---|---|
| `--no_overwrite` | push, download | |
| `--quiet/-q` | push, download | Suppresses the transfer progress bar |
| `--recursive/-r` | list, delete | Opt-in on `delete`: removes every object under the prefix. The Python `delete()` above defaults the other way (`recursive=True`) |
| `--yes/-y` | delete | Skips the confirmation |
| `--max_keys/-n` | list | Default 100 here, unlike `list_objects`'s 1000 |

CLI `delete` removes ONE object; a non-recursive delete aimed at a prefix is refused rather than
reported as a no-op success. That refusal, a `--recursive` delete aimed at a single object, and a
failed delete all exit non-zero; a key with nothing behind it is reported and exits 0.
