#!/usr/bin/env python
"""Non-shared-filesystem save + resume: two "nodes", two writers, two per-node checkpoint dirs.

With ``DIST_OUTPUT_SHARED_FILESYSTEM=0`` the checkpoint writer is EVERY node's local rank 0
(:func:`fs_aware_save_rank`), each writing to its own local disk. A single-node run never reaches
that shape — ``is_local_main_process()`` and ``is_global_main_process()`` agree there, so the writer
set collapses back to global rank 0 and the divergent-writer path is proven only by the unit tests
of the guards around it. Two 1-rank "nodes" (``LOCAL_WORLD_SIZE=1``) on one gloo group restore the
divergence: both ranks are writers, and each :class:`OptimizerShardStore` save/load pair sees only
its OWN directory.

Driven on the real writer and the real reader, this pins:

1. **One meta per node.** Each node's dir must hold its own ``optimizer_meta.pt`` beside its own
   globally-named ``optimizer_shard_XXXXX.pt``. A rank-0-only meta write — the natural spelling —
   leaves every other node's shards ungated:
   ``_read_saved_meta`` is rank-local, so a missing meta is a FAILED topology gate there, and the
   gate is all-or-nothing across ranks. The whole world then warm-restarts and every node resumes
   with reset Adam moments — node 0's own intact shard included — at exit code 0.
2. **Round trip.** Resuming from the node's own dir restores the pre-save optimizer state
   bit-exactly — moments and step counter included.
3. **Rank distinctness.** The two nodes hold DIFFERENT optimizer state, so a rank→dir mix-up cannot
   pass the round trip by comparing two identical states.

    python tests/cpu/checkpoint/test_nonshared_fs_resume.py
"""

import copy
import datetime
import hashlib
import os
import pathlib

import pytest
import torch

from src.distributed import runtime
from src.distributed.checkpoint.fingerprint import OptimizerStateFingerprint
from src.distributed.checkpoint.optimizer import OptimizerShardStore
from src.distributed.runtime import fs_aware_makedirs, fs_aware_save_rank
from tests.common.checkpoint_io import (
    shard_round_trip_model,
    shard_round_trip_optimizer,
    shard_store_context,
    step_on_seeded_data,
)
from tests.common.gloo import run_gloo_ranks
from tests.common.parallelism import make_parallelism_config
from tests.common.utils import assert_optimizer_state_bit_exact

WORLD_SIZE = 2  # two 1-rank "nodes"
SEED = 20260817
TRAIN_STEPS = 3

# The on-disk contract the resume reads and the merge/inspection tools glob for, spelled literally:
# importing the writer's own private constants would make a rename invisible here.
SHARD_FILE_FMT = "optimizer_shard_{rank:05d}.pt"
SHARD_FILE_GLOB = "optimizer_shard_*.pt"
META_FILE = "optimizer_meta.pt"

# A rank that diverges from a collective records gloo's error in its verdict at this bound, instead of
# waiting out gloo's 30-minute default until the join deadline kills it.
PG_TIMEOUT = datetime.timedelta(seconds=120)

# One rank per "node": every rank is its node's local main, hence a checkpoint writer.
TWO_NODE_ENV = {"LOCAL_RANK": "0", "LOCAL_WORLD_SIZE": "1", "DIST_OUTPUT_SHARED_FILESYSTEM": "0"}


def _node_dir(root: str, rank: int) -> str:
    """This node's OWN output dir — on a non-shared FS a node sees only its own local disk."""
    return os.path.join(root, f"node_{rank}", "checkpoint-1")


def _state_signature(state_dict: dict) -> str:
    """Content digest of an ``optimizer.state_dict()`` payload's per-param state."""
    digest = hashlib.sha256()
    for idx in sorted(state_dict["state"]):
        entry = state_dict["state"][idx]
        for key in sorted(entry):
            value = entry[key]
            digest.update(key.encode())
            digest.update(
                value.detach().contiguous().numpy().tobytes() if torch.is_tensor(value) else repr(value).encode()
            )
    return digest.hexdigest()


def _worker(rank: int, root: str) -> None:
    problems: list[str] = []
    try:
        runtime.resolve_shared_filesystem_consensus()  # what init_distributed does for a real run
        if runtime.is_output_shared_filesystem():
            problems.append("premise: the output filesystem resolved as shared")
        if not fs_aware_save_rank():
            problems.append("premise: this rank is not a checkpoint writer, so the writers do not diverge")

        out_dir = _node_dir(root, rank)
        fs_aware_makedirs(out_dir)
        config = make_parallelism_config(world_size=WORLD_SIZE, gpus_per_node=1, rank=rank)

        # Rank-specific data and no gradient sync: identical state on both ranks would let a rank->dir
        # mix-up pass every comparison below.
        model = shard_round_trip_model(SEED)
        optimizer = shard_round_trip_optimizer(model)
        step_on_seeded_data(model, optimizer, SEED + 1 + rank, TRAIN_STEPS)
        saved = copy.deepcopy(optimizer.state_dict())
        with open(os.path.join(root, f"signature_{rank}.txt"), "w") as fh:
            fh.write(_state_signature(saved))

        OptimizerShardStore(shard_store_context(model, optimizer, config)).save(out_dir)

        fresh_model = shard_round_trip_model(SEED)
        fresh_optimizer = shard_round_trip_optimizer(fresh_model)
        OptimizerShardStore(shard_store_context(fresh_model, fresh_optimizer, config)).load(out_dir)

        assert_optimizer_state_bit_exact(saved, fresh_optimizer.state_dict())
    except Exception as e:  # incl. the AssertionError above — the peer must still get a verdict file
        problems.append(f"{type(e).__name__}: {str(e).splitlines()[0][:200]}")

    with open(os.path.join(root, f"result_{rank}.txt"), "w") as fh:
        fh.write("PASS" if not problems else "FAIL: " + "; ".join(problems))
    runtime.reset_shared_filesystem_consensus()


@pytest.fixture(scope="module")
def two_node_run(tmp_path_factory):
    """One 2-process gloo save→resume run; the tests below read the artifacts it left behind."""
    root = tmp_path_factory.mktemp("nonshared_fs_resume")
    run_gloo_ranks(_worker, WORLD_SIZE, str(root), pg_timeout=PG_TIMEOUT, env=TWO_NODE_ENV)
    return root


def test_each_node_resumes_its_own_optimizer_shard_bit_exactly(two_node_run):
    """A per-node save and a per-node resume must round-trip the optimizer state exactly.

    Both nodes write and read only their own directory, so nothing here is covered by a single-node
    run: rank 1's shard is the one no shared-FS topology ever produces.
    """
    for rank in range(WORLD_SIZE):
        result = (two_node_run / f"result_{rank}.txt").read_text()
        assert result == "PASS", f"rank {rank}: {result}"


def test_the_two_nodes_hold_distinct_optimizer_state(two_node_run):
    """Anti-vacuity for the round trip: identical state on both nodes would make a rank→dir mix-up
    (rank 1 restoring rank 0's shard) indistinguishable from a correct resume."""
    signatures = [(two_node_run / f"signature_{rank}.txt").read_text() for rank in range(WORLD_SIZE)]
    assert len(set(signatures)) == WORLD_SIZE, f"the nodes' optimizer states are not distinct: {signatures}"


def test_every_node_writes_its_own_shard_and_meta(two_node_run):
    """Each node's dir holds its own globally-named shard AND its own ``optimizer_meta.pt``.

    One writer per NODE, not one per world: a rank-0-only meta write leaves node 1's shards ungated,
    which ``_read_saved_meta`` reads as a failed topology gate — and since that gate is
    all-or-nothing, EVERY node then warm-restarts with reset moments. The shard name stays keyed by
    GLOBAL rank (a per-node numbering still round-trips, so only this assertion catches it), so node 1
    writes ``optimizer_shard_00001.pt``; renumbering per node would make every node's dir look like
    rank 0's to the resume.
    """
    for rank in range(WORLD_SIZE):
        node_dir = pathlib.Path(_node_dir(str(two_node_run), rank))
        shards = sorted(path.name for path in node_dir.glob(SHARD_FILE_GLOB))
        assert shards == [SHARD_FILE_FMT.format(rank=rank)], f"node {rank} holds the wrong shard set: {shards}"

        meta_path = node_dir / META_FILE
        assert meta_path.is_file(), f"node {rank} has no {META_FILE} — its shards would resume ungated"
        meta = torch.load(meta_path, map_location="cpu", weights_only=False)
        assert meta["num_ranks"] == WORLD_SIZE, f"node {rank} recorded num_ranks={meta['num_ranks']}"
        assert OptimizerStateFingerprint.from_dict(meta.get("fingerprint")) is not None, (
            f"node {rank}'s {META_FILE} carries no complete fingerprint, so its resume falls back to "
            f"the num_ranks-only gate"
        )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
