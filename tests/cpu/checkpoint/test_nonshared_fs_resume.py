#!/usr/bin/env python
"""Non-shared-filesystem save + resume on two 1-rank nodes, both writers, each seeing only its own directory. A
single-node run collapses the writer set to global rank 0 and never reaches this."""

import copy
import datetime
import hashlib
import json
import os
import pathlib

import pytest
import torch

from halo.checkpoint.parallel.optimizer import OptimizerShardStore
from halo.runtime import filesystem
from halo.runtime.filesystem import fs_aware_makedirs, is_fs_save_rank
from tests.common.checkpoint_io import (
    shard_round_trip_model,
    shard_round_trip_optimizer,
    shard_store_context,
    step_on_seeded_data,
)
from tests.common.distributed import reset_shared_filesystem_consensus
from tests.common.gloo import run_gloo_ranks
from tests.common.optimizer_state import assert_optimizer_state_bit_exact
from tests.common.parallelism import make_parallelism_config

WORLD_SIZE = 2
TORN_NODE = 1
SEED = 20260817
TRAIN_STEPS = 3

# Spelled literally: importing the writer's private constants would make a rename invisible here.
SHARD_FILE_FMT = "optimizer_shard_{rank:05d}.pt"
SHARD_FILE_GLOB = "optimizer_shard_*.pt"
META_FILE = "optimizer_meta.pt"

# Records gloo's error at this bound instead of gloo's 30-minute default.
PG_TIMEOUT = datetime.timedelta(seconds=120)

# Every rank is its node's local main, hence a checkpoint writer.
TWO_NODE_ENV = {"LOCAL_RANK": "0", "LOCAL_WORLD_SIZE": "1", "DIST_OUTPUT_SHARED_FILESYSTEM": "0"}


def _node_dir(root: str, rank: int) -> str:
    return os.path.join(root, f"node_{rank}", "checkpoint-1")


def _state_signature(state_dict: dict) -> str:
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
        filesystem.resolve_shared_filesystem_consensus()  # What init_distributed does for a real run.
        if filesystem.is_output_shared_filesystem():
            problems.append("premise: the output filesystem resolved as shared")
        if not is_fs_save_rank():
            problems.append("premise: this rank is not a checkpoint writer, so the writers do not diverge")

        out_dir = _node_dir(root, rank)
        fs_aware_makedirs(out_dir)
        config = make_parallelism_config(world_size=WORLD_SIZE, gpus_per_node=1, rank=rank)

        # Rank-specific data and no gradient sync: identical state would let a rank->dir mix-up pass.
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
    except Exception as e:  # Incl. the AssertionError above: the peer must still get a verdict file.
        problems.append(f"{type(e).__name__}: {str(e).splitlines()[0][:200]}")

    with open(os.path.join(root, f"result_{rank}.txt"), "w") as fh:
        fh.write("PASS" if not problems else "FAIL: " + "; ".join(problems))
    reset_shared_filesystem_consensus()


def _torn_node_worker(rank: int, root: str) -> None:
    """Node 1 loses its meta, as a save killed between its shard and its meta."""
    outcome = {"raised": None, "restored_entries": None}
    try:
        filesystem.resolve_shared_filesystem_consensus()
        out_dir = _node_dir(root, rank)
        fs_aware_makedirs(out_dir)
        config = make_parallelism_config(world_size=WORLD_SIZE, gpus_per_node=1, rank=rank)
        model = shard_round_trip_model(SEED)
        optimizer = shard_round_trip_optimizer(model)
        step_on_seeded_data(model, optimizer, SEED + 1 + rank, TRAIN_STEPS)
        OptimizerShardStore(shard_store_context(model, optimizer, config)).save(out_dir)
        if rank == TORN_NODE:
            os.remove(os.path.join(out_dir, META_FILE))

        fresh_model = shard_round_trip_model(SEED)
        fresh_optimizer = shard_round_trip_optimizer(fresh_model)
        try:
            OptimizerShardStore(shard_store_context(fresh_model, fresh_optimizer, config)).load(out_dir)
        except RuntimeError as e:
            outcome["raised"] = str(e)
        outcome["restored_entries"] = len(fresh_optimizer.state)
    except Exception as e:  # The setup failing must still leave a verdict file.
        outcome["raised"] = f"setup {type(e).__name__}: {e}"
    with open(os.path.join(root, f"torn_{rank}.json"), "w") as fh:
        json.dump(outcome, fh)
    reset_shared_filesystem_consensus()


@pytest.fixture(scope="module")
def two_node_run(tmp_path_factory):
    root = tmp_path_factory.mktemp("nonshared_fs_resume")
    run_gloo_ranks(_worker, WORLD_SIZE, str(root), pg_timeout=PG_TIMEOUT, env=TWO_NODE_ENV)
    return root


def test_each_node_resumes_its_own_optimizer_shard_bit_exactly(two_node_run):
    """Rank 1's shard is the one no shared-FS topology ever produces."""
    for rank in range(WORLD_SIZE):
        result = (two_node_run / f"result_{rank}.txt").read_text()
        assert result == "PASS", f"rank {rank}: {result}"


def test_the_two_nodes_hold_distinct_optimizer_state(two_node_run):
    """Anti-vacuity: identical state would make a rank→dir mix-up indistinguishable."""
    signatures = [(two_node_run / f"signature_{rank}.txt").read_text() for rank in range(WORLD_SIZE)]
    assert len(set(signatures)) == WORLD_SIZE, f"the nodes' optimizer states are not distinct: {signatures}"


def test_every_node_writes_its_own_shard_and_meta(two_node_run):
    """A rank-0-only meta leaves node 1's shards reading as torn, so every node refuses. Global-rank names matter:
    per-node numbering still round-trips."""
    for rank in range(WORLD_SIZE):
        node_dir = pathlib.Path(_node_dir(str(two_node_run), rank))
        shards = sorted(path.name for path in node_dir.glob(SHARD_FILE_GLOB))
        assert shards == [SHARD_FILE_FMT.format(rank=rank)], f"node {rank} holds the wrong shard set: {shards}"

        meta_path = node_dir / META_FILE
        assert meta_path.is_file(), f"node {rank} has no {META_FILE} — its shards would resume ungated"
        meta = torch.load(meta_path, map_location="cpu", weights_only=False)
        assert meta["num_ranks"] == WORLD_SIZE, f"node {rank} recorded num_ranks={meta['num_ranks']}"
        assert isinstance(meta.get("fingerprint"), dict), (
            f"node {rank}'s {META_FILE} carries no fingerprint, so its resume is refused as a "
            f"pre-fingerprint checkpoint"
        )


def test_one_nodes_torn_save_refuses_the_resume_on_every_node(tmp_path):
    run_gloo_ranks(_torn_node_worker, WORLD_SIZE, str(tmp_path), pg_timeout=PG_TIMEOUT, env=TWO_NODE_ENV)
    for rank in range(WORLD_SIZE):
        outcome = json.loads((tmp_path / f"torn_{rank}.json").read_text())
        raised = outcome["raised"]
        assert raised is not None, f"rank {rank} resumed over node {TORN_NODE}'s torn shard set"
        assert not raised.startswith("setup"), raised
        assert "torn set of an interrupted save" in raised, f"rank {rank}: {raised}"
        assert "allow_optimizer_warm_restart" in raised, f"rank {rank} did not name the opt-in: {raised}"
        assert outcome["restored_entries"] == 0, f"rank {rank} restored moments its peer could not"
    own = json.loads((tmp_path / f"torn_{TORN_NODE}.json").read_text())["raised"]
    assert f"{META_FILE} is missing" in own, own


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
