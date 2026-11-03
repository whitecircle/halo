#!/usr/bin/env python
"""Sharded-dataset loading under real torch.distributed: probe agreement, exact partition, equal lengths.

A per-rank S3 credential or throttling flake must not split ranks onto sharded and full paths, which hangs. Unequal
lengths hang the next grad all-reduce.
"""

import os
import shutil
from collections import Counter

import torch.distributed as dist
from datasets import Dataset

from halo.data.preprocessed.preparation import shard_dataset
from halo.data.preprocessed.shard_index import SHARD_INDEX_FILE, write_stamped_sidecar
from halo.data.sources.loading import is_sharded_dataset_across_ranks
from halo.data.sources.sharded_dataset import ShardedDatasetLoader
from halo.trainers.dataloaders import DataParallelLoaderMixin
from tests.common.distributed import shared_scratch_dir
from tests.common.harness import gpu_test_main
from tests.common.rank_log import log

N_EXAMPLES = 100
N_SHARDS = 17  # Above world_size at every nproc (2/4/8), so the partition stays non-trivial.


def _broadcast_path(rank: int, world_size: int, path: str | None) -> str:
    gathered = [None] * world_size
    dist.all_gather_object(gathered, path if rank == 0 else None)
    return gathered[0]


def test_coordinated_probe(rank: int, world_size: int, path: str) -> bool:
    if not is_sharded_dataset_across_ranks(path):
        log(f"FAIL: rank {rank} did not detect a sharded dataset at {path}", rank=rank)
        return False

    # Rank 0's probe flakes to False; all_reduce(MAX) must still yield True, so no rank diverges onto the full path.
    orig = ShardedDatasetLoader.is_sharded_dataset
    try:
        if rank == 0:
            ShardedDatasetLoader.is_sharded_dataset = staticmethod(lambda p: False)
        recovered = is_sharded_dataset_across_ranks(path)
    finally:
        ShardedDatasetLoader.is_sharded_dataset = orig
    if not recovered:
        log("FAIL: all_reduce(MAX) did not recover 'sharded' after rank-0 flake", rank=rank)
        return False
    log("PASS: coordinated probe agrees + recovers from a single-rank flake")
    return True


def test_exact_partition(rank: int, world_size: int, path: str) -> bool:
    """Every id exactly once: no gap, no overlap."""
    loader = ShardedDatasetLoader(dataset_path=path, dp_rank=rank, dp_size=world_size)
    ds = loader.load_split("train")
    my_ids = sorted(int(x) for x in ds["example_id"]) if len(ds) else []

    gathered = [None] * world_size
    dist.all_gather_object(gathered, my_ids)
    cov = Counter()
    for lst in gathered:
        cov.update(lst)

    dups = {k: v for k, v in cov.items() if v != 1}
    if dups:
        log(f"FAIL: partition overlap — ids loaded by >1 rank: {dict(list(dups.items())[:10])}", rank=rank)
        return False
    if set(cov.keys()) != set(range(N_EXAMPLES)):
        missing = set(range(N_EXAMPLES)) - set(cov.keys())
        log(f"FAIL: partition gap — {len(missing)} ids never loaded, e.g. {sorted(missing)[:10]}", rank=rank)
        return False
    counts = [len(g) for g in gathered]
    log(f"PASS: exact partition — {N_EXAMPLES} ids over {N_SHARDS} shards loaded once each; per-rank counts {counts}")
    return True


def test_equalize_to_global_min(rank: int, world_size: int) -> bool:
    lengths = [10 + r for r in range(world_size)]
    local = Dataset.from_dict({"x": list(range(lengths[rank]))})
    equalized = DataParallelLoaderMixin._equalize_presharded_length(None, local)
    if len(equalized) != min(lengths):
        log(f"FAIL: equalized to {len(equalized)}, expected global min {min(lengths)}", rank=rank)
        return False
    log(f"PASS: _equalize_presharded_length truncated all ranks to global min {min(lengths)}")
    return True


def test_zero_rank_raises(rank: int, world_size: int) -> bool:
    n = 0 if rank == world_size - 1 else 5
    ds = Dataset.from_dict({"x": list(range(n))})
    try:
        DataParallelLoaderMixin._equalize_presharded_length(None, ds)
    except ValueError:
        log("PASS: zero-example rank triggers ValueError on every rank")
        return True
    log("FAIL: expected ValueError for zero-example rank, none raised", rank=rank)
    return False


def run(ctx) -> dict:
    base = shared_scratch_dir("test_sharded_dist_load")
    path = os.path.join(base, "ds") if ctx.rank == 0 else None
    path = _broadcast_path(ctx.rank, ctx.world_size, path)

    # Rank 0 writes an id-tagged sharded dataset; all ranks then load disjoint slices.
    if ctx.rank == 0:
        ctx.on_teardown(lambda: shutil.rmtree(base, ignore_errors=True))
        shutil.rmtree(base, ignore_errors=True)
        synth = Dataset.from_dict({"example_id": list(range(N_EXAMPLES)), "v": list(range(N_EXAMPLES))})
        index = shard_dataset(synth, output_dir=path, split_name="train", num_shards=N_SHARDS)
        write_stamped_sidecar(os.path.join(path, "train", SHARD_INDEX_FILE), index)
    ctx.barrier()

    checks = {
        "coordinated_probe": test_coordinated_probe(ctx.rank, ctx.world_size, path),
        "exact_partition": test_exact_partition(ctx.rank, ctx.world_size, path),
        "equalize_to_global_min": test_equalize_to_global_min(ctx.rank, ctx.world_size),
        "zero_rank_raises": test_zero_rank_raises(ctx.rank, ctx.world_size),
    }
    ctx.barrier()
    return {"checks": checks}


main = gpu_test_main(min_world_size=2, prefix="sharded_distributed_load")(run)

if __name__ == "__main__":
    main()
