#!/usr/bin/env python
"""``get_local_rank`` / ``get_local_world_size`` resolution — including the PRE-INIT path.

``init_distributed`` calls ``get_local_rank()`` *before* ``init_process_group`` to pick this rank's
CUDA device (``torch.cuda.set_device`` + ``device_id=``). At that moment ``dist`` is not initialized
and the launcher may have set only ``RANK`` (``LOCAL_RANK`` and ``SLURM_LOCALID`` absent), so a
resolver that only consults those two plus an *initialized* process group answers 0 on EVERY rank —
binding the whole node to ``cuda:0``. Derivation from ``RANK`` is what makes the pre-init answer
usable. The local-world-size last resort is armed on the pre-init path too, so that every consumer
of node-local rank math (``ParallelismConfig``'s ``gpus_per_node``, the store coordination scopes)
gets the same answer and the same warning before and after init.

CUDA is forced unavailable throughout: the device-count branch would otherwise mask the derivation
under test on a GPU host.

Run: pytest tests/cpu/parallelism/test_local_rank_resolution.py
"""

import logging
import sys

import pytest
import torch

from src.distributed import runtime

_RANK_ENV_VARS = ("LOCAL_RANK", "SLURM_LOCALID", "LOCAL_WORLD_SIZE", "SLURM_NTASKS_PER_NODE", "RANK", "WORLD_SIZE")


@pytest.fixture()
def pre_init(monkeypatch):
    """A launcher-started process before ``init_process_group``, on a host with no visible GPUs."""
    for name in _RANK_ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(runtime.dist, "is_initialized", lambda: False)
    monkeypatch.setattr(runtime, "_LWS_FALLBACK_ANNOUNCED", set())


def test_rank_only_launch_does_not_collapse_every_rank_to_zero(pre_init, monkeypatch):
    """``RANK=3 WORLD_SIZE=4`` with no LOCAL_* must resolve to local rank 3, not 0."""
    monkeypatch.setenv("RANK", "3")
    monkeypatch.setenv("WORLD_SIZE", "4")
    assert runtime.get_local_world_size() == 4
    assert runtime.get_local_rank() == 3
    assert runtime.is_local_main_process() is False


def test_multi_node_rank_wraps_within_the_node(pre_init, monkeypatch):
    """Rank 11 of a 2×8 job is local rank 3 on node 1 — the modulo must use the NODE's size."""
    monkeypatch.setenv("RANK", "11")
    monkeypatch.setenv("WORLD_SIZE", "16")
    monkeypatch.setenv("LOCAL_WORLD_SIZE", "8")
    assert runtime.get_local_rank() == 3
    assert runtime.get_node_rank() == 1


def test_explicit_launcher_vars_win_over_derivation(pre_init, monkeypatch):
    monkeypatch.setenv("RANK", "3")
    monkeypatch.setenv("WORLD_SIZE", "4")
    monkeypatch.setenv("LOCAL_RANK", "1")
    assert runtime.get_local_rank() == 1
    monkeypatch.delenv("LOCAL_RANK")
    monkeypatch.setenv("SLURM_LOCALID", "2")
    assert runtime.get_local_rank() == 2


def test_slurm_ntasks_per_node_sizes_the_node(pre_init, monkeypatch):
    """SLURM's compact form ("8(x2)") sizes the node, so rank 9 is local rank 1."""
    monkeypatch.setenv("RANK", "9")
    monkeypatch.setenv("WORLD_SIZE", "16")
    monkeypatch.setenv("SLURM_NTASKS_PER_NODE", "8(x2)")
    assert runtime.get_local_world_size() == 8
    assert runtime.get_local_rank() == 1


def test_pre_init_last_resort_warns(pre_init, monkeypatch, caplog):
    """The single-node assumption must be LOUD on the pre-init path too.

    A silent ``return 1`` here is where a real multi-node job builds wrong node-local EP/CP groups
    and wrong store-coordination scopes with no diagnostic at all.
    """
    monkeypatch.setenv("RANK", "3")
    monkeypatch.setenv("WORLD_SIZE", "4")
    with caplog.at_level(logging.WARNING, logger=runtime.logger.name):
        runtime.get_local_world_size()
    assert any("local world size" in r.message for r in caplog.records), "pre-init fallback warned nothing"


def test_single_process_run_is_rank_zero_and_silent(pre_init, caplog):
    """No launcher vars at all: local rank 0 of a 1-process node, and no scary warning."""
    with caplog.at_level(logging.WARNING, logger=runtime.logger.name):
        assert runtime.get_local_world_size() == 1
        assert runtime.get_local_rank() == 0
    assert caplog.records == []


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
