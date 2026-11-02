#!/usr/bin/env python
"""The optimizer-shard save offloads and writes under the node's load throttle, as the resume reads.

A sharded save copies each rank's optimizer state to host RAM (``cpu_offload``) and keeps it until its shard is
written. Unthrottled, a node holds ``local_world_size`` copies at once; the throttle admits
``max_concurrent_loading`` ranks at a time, so a node holds that many. The low-RAM warning estimates the copies the
node holds together, not one rank's.

Two ranks, real gloo, ``max_concurrent_loading=1``: each rank's host copy, from the offload to the end of its
shard write, must not overlap the other's, and a rank that cannot produce its state still fails the save on every
rank, joined over the store after the throttle.

    python tests/cpu/checkpoint/test_optimizer_shard_save_throttle.py
"""

from __future__ import annotations

import datetime
import os
import time
import types

import pytest
import torch
import torch.distributed as dist
import torch.nn as nn

import halo.checkpoint.parallel.optimizer as optimizer_mod
from halo.checkpoint.parallel.optimizer import OptimizerShardStore
from tests.common.checkpoint_io import load_context
from tests.common.gloo import run_gloo_ranks
from tests.common.parallelism import make_parallelism_config

WORLD_SIZE = 2
# How long each rank's host copy lives; long enough that two concurrent copies overlap unmistakably.
HOLD_SECONDS = 0.4
PG_TIMEOUT_SEC = 60


def _store(rank: int, max_concurrent_loading: int = 1) -> OptimizerShardStore:
    model = nn.Linear(2, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    config = make_parallelism_config(
        world_size=WORLD_SIZE, gpus_per_node=WORLD_SIZE, max_concurrent_loading=max_concurrent_loading
    )
    return OptimizerShardStore(load_context(model, config, fsdp_wrapped=True, optimizer=optimizer))


def _hold_windows(rank: int, tmp_dir: str) -> None:
    """Save once, recording when this rank's host copy came to exist and when its shard write ended."""
    window = []

    def offload(*_args, **_kwargs):
        window.append(time.monotonic())
        time.sleep(HOLD_SECONDS)
        return {"state": {}, "param_groups": []}

    write = torch.save

    def timed_write(*args, **kwargs):
        write(*args, **kwargs)
        window.append(time.monotonic())

    optimizer_mod.get_optimizer_state_dict = offload
    optimizer_mod.torch.save = timed_write
    output = os.path.join(tmp_dir, "checkpoint")
    os.makedirs(output, exist_ok=True)
    _store(rank).save(output)
    dist.barrier()
    with open(os.path.join(tmp_dir, f"window_{rank}.txt"), "w") as fh:
        fh.write(f"{window[0]}|{window[1]}")


def _one_rank_cannot_produce_its_state(rank: int, tmp_dir: str) -> None:
    def offload(*_args, **_kwargs):
        if rank == 1:
            raise RuntimeError("unevenly-sharded DTensors have no state_dict")
        return {"state": {}, "param_groups": []}

    optimizer_mod.get_optimizer_state_dict = offload
    try:
        _store(rank).save(tmp_dir)
        outcome = "saved"
    except RuntimeError as e:
        outcome = str(e)
    dist.barrier()
    with open(os.path.join(tmp_dir, f"outcome_{rank}.txt"), "w") as fh:
        fh.write(outcome)


def test_a_nodes_ranks_hold_their_host_copies_one_batch_at_a_time(tmp_path):
    run_gloo_ranks(_hold_windows, WORLD_SIZE, str(tmp_path), pg_timeout=datetime.timedelta(seconds=PG_TIMEOUT_SEC))
    windows = sorted(
        tuple(float(t) for t in (tmp_path / f"window_{rank}.txt").read_text().split("|")) for rank in range(WORLD_SIZE)
    )
    (first, second) = windows
    assert first[1] <= second[0], (
        f"the two ranks' optimizer host copies coexisted ({first} vs {second}): the shard save offloads outside "
        f"the node load throttle, so every local rank holds its copy at once."
    )


def test_a_rank_that_cannot_produce_its_state_fails_the_save_on_every_rank(tmp_path):
    run_gloo_ranks(
        _one_rank_cannot_produce_its_state, WORLD_SIZE, str(tmp_path), pg_timeout=datetime.timedelta(seconds=30)
    )
    for rank in range(WORLD_SIZE):
        outcome = (tmp_path / f"outcome_{rank}.txt").read_text()
        assert "save_only_model" in outcome and "unevenly-sharded" in outcome, f"rank {rank}: {outcome}"
    assert not (tmp_path / "optimizer_meta.pt").exists(), "a meta vouches for a shard set missing rank 1's shard"


class _OneGigabyteOfMoments:
    """A parameter's data the warning sizes at 1 GB of optimizer state: two moments of 125M fp32 elements."""

    def numel(self) -> int:
        return 125_000_000

    def element_size(self) -> int:
        return 4


@pytest.mark.parametrize(
    ("available_gb", "warns"),
    [(0.9, True), (1.9, True), (2.1, False)],
    ids=["below-one-copy", "below-the-admitted-copies", "above-them"],
)
def test_the_low_ram_warning_counts_the_ranks_admitted_together(monkeypatch, caplog, available_gb, warns):
    """1 GB copies on an 8-rank node at ``max_concurrent_loading=2``: the node holds two at once."""
    parameter = types.SimpleNamespace(data=_OneGigabyteOfMoments())
    optimizer = types.SimpleNamespace(param_groups=[{"params": [parameter]}])
    monkeypatch.setattr(optimizer_mod, "get_local_world_size", lambda: 8)
    monkeypatch.setattr(optimizer_mod, "available_host_ram_bytes", lambda: int(available_gb * 1e9))
    with caplog.at_level("WARNING", logger=optimizer_mod.logger.name):
        optimizer_mod._warn_if_low_host_ram(optimizer, "the shard save", 2)
    assert bool(caplog.records) is warns, [record.getMessage() for record in caplog.records]
    if warns:
        assert "2 rank(s)" in caplog.records[0].getMessage()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
