# Filesystem Handling

When multiple ranks download a model, load a dataset, or create a directory, one rank goes first and the others read from the result. The toolkit picks the coordination scope from whether nodes share a filesystem, with separate control over the read and write sides.

## Env vars

| Var | Governs | Default |
|-----|---------|---------|
| `DIST_SHARED_FILESYSTEM` | Umbrella both sides fall back to | `1` (shared) |
| `DIST_INPUT_SHARED_FILESYSTEM` | Read side: model/dataset downloads, dataset map/pack, HF caches | umbrella |
| `DIST_OUTPUT_SHARED_FILESYSTEM` | Write side: checkpoints, `run.log`, dumped artifacts | umbrella |

Each is read through `env_flag`: **true** for `1`/`true`/`yes`/`on` (case-insensitive), **false** for
any other value — including a typo. Empty counts as unset and yields the default, so a side var set to
`""` still falls back to the umbrella; any other value wins over it in either direction.

Shared: global rank 0 goes first, all other ranks wait then read the result — for shared NFS / Lustre
/ GPFS. Non-shared: each node's local rank 0 operates independently, in parallel — for per-node local
NVMe (cloud, RunPod).

All three must be the **same on every rank**: they pick the coordination scope, so a split value would
put participants of one tag under two different scopes where they never see each other's keys.
`init_distributed()` therefore broadcasts global rank 0's *resolved* values for all three
(`resolve_shared_filesystem_consensus()`) and memoizes them, and a rank whose own env disagreed warns.

The **output** declaration is additionally checked against the filesystem itself at startup, on
multi-node runs only (`verify_output_filesystem_sharing`): global rank 0 writes a sentinel under
`output_dir` and every rank looks for it — waiting up to 60 s when shared is declared, not at all
when per-node is. Either contradiction raises and names the var to set. Declared shared but invisible
to some rank means only global rank 0 writes `trainer_state.json`, so the other nodes resume at step
0; declared per-node but visible everywhere means every node's local rank 0 writes the same
checkpoint paths concurrently.

The split exists because the two sides can want opposite settings on a multi-node run over a slow or
flaky shared mount (NFS/EFS). Input wants per-node: global rank 0 writing the HF/dataset cache and
remote ranks reading those same inodes is a read-after-cross-node-write that NFS surfaces as
`OSError: [Errno 116] Stale file handle`.

Output is fine shared — checkpoints have a single writer, per-rank dumps are distinct files, and a
shared mount gives one authoritative copy with no post-hoc gather. That combination is
`DIST_INPUT_SHARED_FILESYSTEM=0` with the umbrella left at its shared default.

## Coordination primitives

Owning files: `src/distributed/filesystem.py` (the store-coordinated primitives) and its
one-way dependency `src/distributed/runtime.py` (the shared-FS flags, the writer/reader
rank predicates, `fs_aware_makedirs` and `reject_across_ranks`).

- `fs_aware_main_first(tag, timeout=None)` — context manager that orders the body main-rank-first.
  **Every rank runs the body**: the main rank runs it alone first and populates a cache, then the
  others run the same code and hit it. Scope follows the **input** flag — shared → the main rank is
  global rank 0 and all other ranks wait; non-shared → it is each node's local rank 0 and only that
  node's ranks wait, nodes proceeding in parallel. Wrap any download / cache-writing block in it.

    Waiters block on a c10d key-value store key rather than in a collective, because the body is
    unbounded single-rank work (a 100B+ `snapshot_download`, whole-corpus packing) that would trip the
    NCCL watchdog (`DIST_NCCL_TIMEOUT_MINUTES`) on peers held in a collective. The wait is bounded by
    `DIST_STORE_TIMEOUT_HOURS` (default 4 h) instead, and covers the **main rank's** work only: there
    is no trailing join, so the main rank's own wait for the peers is whatever collective comes next
    — the watchdog, minutes not hours. That is why a **misdeclared input flag** is expensive rather
    than merely slow: declare shared over per-node storage and every peer redoes the whole
    download/pack against the watchdog, not the store bound.

    Two rules the caller owns. The body must issue **no collective**, directly or through a helper
    (`ensure_cache_dir()` and `fs_aware_makedirs()` both barrier) — while the main rank runs it, the
    peers are on a store key, not in a matching collective, so the main rank would block alone. And
    every rank must reach a given `tag` the **same number of times, in the same order**: keep one tag
    per call site and never enter one from a rank-dependent branch.
- `hub_metadata_main_first(tag, fetch)` — `fs_aware_main_first` around a single hub-metadata read
  (`config.json`, a processor config), returning `fetch()`'s result. Tags live under `hub_meta/`, and
  the bound is 30 minutes rather than the hours-scale store default — a metadata fetch that slow is a
  dead endpoint, not slow work. It resolves the shared-filesystem consensus itself if that has not
  happened yet, since these reads can precede `init_distributed()` and a split scope would put the
  tag's participants where they never see each other's keys.
- `sequential_load_within_node(tag, max_concurrent)` — store-coordinated throttle that admits at most
  `max_concurrent` ranks per node at a time; backs `max_concurrent_loading` at model load. Same
  `DIST_STORE_TIMEOUT_HOURS` bound.
- `is_shared_filesystem()` / `is_input_shared_filesystem()` / `is_output_shared_filesystem()` — the
  resolved modes, for picking `is_global_main_process()` vs `is_local_main_process()` directly.
- `fs_aware_load_rank()` — whether this rank performs shared read-side work (downloads, cache fill,
  the coordinated dataset map). Follows the **input** flag.
- `fs_aware_save_rank()` — whether this rank writes shared files. Follows the **output** flag: shared
  → only global rank 0 writes (avoids the NFS write race); non-shared → each node's local rank 0
  writes its own copy. Every gathered/sharded save, the `run.log` tee, and `fs_aware_makedirs` by
  default route their writer choice through it.
- `fs_aware_makedirs(path, writer_rank=fs_aware_save_rank)` — creates a directory with the correct
  rank, then barriers. Pass `writer_rank=fs_aware_load_rank` for a read-side cache dir (what
  `src/training/environment.py` does for `HF_DATASETS_CACHE`).
- `reject_across_ranks(local_reason, what, exc_type=RuntimeError)` — the collective that joins a
  rank-gated body. Every rank calls it with its own reason or `None`; if **any** is non-`None`, every
  rank raises the same `exc_type`, naming how many ranks failed and the first one's reason. Pass
  `exc_type` to keep a caller's own error contract (a config gate documented as `ValueError`). Use it
  in place of a barrier wherever one rank does work the others wait on, so a failure there aborts the
  job instead of parking the peers in the barrier until the watchdog.
- `store_reject_across_ranks(tag, local_reason, what, exc_type=RuntimeError, timeout=None)` — the
  same contract carried over the c10d store instead of a collective, for joins whose preceding work
  is unbounded single-rank time (a fresh-cache dataset map, a first-run shard download). Peers wait
  on store keys bounded by `DIST_STORE_TIMEOUT_HOURS` (or the explicit `timeout`), not the NCCL
  watchdog. World-scoped and collective-equivalent: every rank, same tags, same order. Each rank
  writes one key and reads the world's in a single `multi_get`, so a join costs O(world) store
  requests in total rather than O(world) per rank. The coordinated dataset ops and the
  sharded-dataset loads join through it. The tradeoff: a rank that dies *between* joins parks its
  peers on the store for the full bound rather than the watchdog minutes — the timeout diagnostic
  names the tag and the knob, and the launcher's elastic agent usually reaps the group first.

```python
from src.distributed.filesystem import fs_aware_main_first

with fs_aware_main_first("teacher_model"):
    snapshot_download(teacher_repo_id)   # one rank fetches; the rest read the populated cache
```

!!! warning "`coordinated_map` / `coordinated_filter` are already ordered — never nest them"
    Every coordinated dataset op runs through `coordinated_dataset_operation`
    (`src/data/pipeline/processing.py`), which **is** the rank ordering: it picks the writer via
    `fs_aware_load_rank()`, runs the op there, joins on `store_reject_across_ranks`, then runs it on
    the peers against the cache and joins again. The store joins let either side wait out the other's
    unbounded work — the writer's fresh-cache map, or a peer's full recompute under a misdeclared
    input flag — under `DIST_STORE_TIMEOUT_HOURS` instead of the NCCL watchdog.

Wrapping a call in a main-first block (`local_main_process_first()`, `fs_aware_main_first`) breaks it.
Those hold peers *outside* the body, so the op's `ensure_cache_dir` barrier and store joins land at a
different sequence position on the main rank than on its peers: the barrier pairs with the wrong
collective (or blocks alone against a store-key wait), and the joins go permanently off-by-one on the
equal-entry invariant. `tests/cpu/data/test_coordinated_op_not_main_first.py` fails if any call site
nests one.

## Where it is applied

| Area | Side | File | Notes |
|------|------|------|-------|
| Model downloads | input | `src/distributed/loading/model_loading.py` | `_ensure_model_downloaded()` wraps the `snapshot_download` hub fetch (not `from_pretrained`) in `fs_aware_main_first`; per-rank `from_pretrained` then reads the populated cache |
| Dataset loading | input | `src/data/sources/loading.py` | One `fs_aware_main_first` block covers the source load (S3, Hub, or local) and the train/test split. A sharded dataset returns before it, coordinates per shard with a file lock, and joins the world over the store so one rank's shard failure aborts every rank with the cause |
| Dataset map / filter | input | `src/data/pipeline/processing.py` | `coordinated_map` / `coordinated_filter` — self-ordering, see the warning above |
| Dataset packing | input | `pack_dataset_coordinated` (`processing.py`) | `fs_aware_main_first` around a deterministic `cache_file_name`. Unguarded, `pack_dataset` materializes a full corpus copy **per rank** |
| EP / PP lazy model loading | input | `src/distributed/expert_parallel/lazy_loader.py`, `src/distributed/pipeline_parallel/lazy_loader.py` | Both share one prologue that orders the meta-device instantiation main-rank-first; each rank then slices its expert range (EP) or stage layers (PP) from the safetensors shards |
| Checkpoint saves | output | `src/distributed/expert_parallel/saving.py`, `src/distributed/tensor_parallel/checkpoint.py`, `src/trainers/mixins/checkpointing.py` | Writer-rank selection via `fs_aware_save_rank()`; a non-shared output FS also forces `save_on_each_node` and rejects a multi-node sharded-EP save (the per-rank expert shards are keyed by global rank and would scatter across nodes' local disks with no gather path) |
| `run.log` tee | output | `src/training/run_logging.py` | Same `fs_aware_save_rank()` predicate as the checkpoint writers, so two nodes can never both hold one `run.log` |
| Directory creation | either | `fs_aware_makedirs()` | Output dirs by default; read-side cache dirs pass `writer_rank=fs_aware_load_rank` |

## Storage layout

On a shared-storage cluster, put read-only models and datasets there and keep the HF cache and temp files on local NVMe.

```bash
export DIST_SHARED_FILESYSTEM=1          # 0 for per-node local storage
export HF_HOME=/local/hf_cache
export HF_DATASETS_CACHE=/local/hf_cache/datasets
export TMPDIR=/local/tmp
export TORCH_EXTENSIONS_DIR=/local/torch_extensions
```

For a 20B MoE budget roughly: weights ~40 GB (shared, read-only), HF cache ~80 GB (local), temp
scratch ~100 GB (local NVMe).

`HF_DATASETS_CACHE` grows without bound across config sweeps: every distinct
(function, tokenizer, knobs, dataset, library-version) tuple leaves a permanent `cache-*.arrow` set,
plus `packed_*.arrow` per packing shape, and `hf cache prune` reclaims only the hub cache, not
these. Reclaim by deleting the directory — every cache name is deterministic, so the next run
rebuilds exactly what it needs.

To avoid shared-FS bottlenecks, pre-stage models and datasets to local NVMe before training —
`snapshot_download(repo_id, local_dir=...)` per node (parallel under `DIST_SHARED_FILESYSTEM=0`), or
`rsync`. With `local_dir` set, `huggingface_hub` writes real files, not symlinks (silencing NFS
symlink warnings otherwise needs `HF_HUB_DISABLE_SYMLINKS_WARNING=1`). Reclaim local space with
`hf cache prune` or `hf cache rm <repo>`.

Multi-node launches: [Launch Recipes](../parallelism/launch-recipes.md).
