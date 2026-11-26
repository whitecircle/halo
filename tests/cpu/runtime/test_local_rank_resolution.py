#!/usr/bin/env python
"""``get_local_rank`` / ``get_local_world_size``, including pre-init, where a resolver not deriving from ``RANK``
binds the whole node to ``cuda:0``. CUDA is forced unavailable."""

import logging

import pytest
import torch

from halo.runtime import ranks

_RANK_ENV_VARS = (
    "LOCAL_RANK",
    "SLURM_LOCALID",
    "SLURM_PROCID",
    "LOCAL_WORLD_SIZE",
    "SLURM_STEP_TASKS_PER_NODE",
    "SLURM_STEP_NUM_TASKS",
    "RANK",
    "WORLD_SIZE",
)


@pytest.fixture()
def pre_init(monkeypatch):
    for name in _RANK_ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(ranks.dist, "is_initialized", lambda: False)
    monkeypatch.setattr(ranks, "_local_world_size_fallback_warned", set())


def test_rank_only_launch_does_not_collapse_every_rank_to_zero(pre_init, monkeypatch):
    monkeypatch.setenv("RANK", "3")
    monkeypatch.setenv("WORLD_SIZE", "4")
    assert ranks.get_local_world_size() == 4
    assert ranks.get_local_rank() == 3
    assert ranks.is_local_main_process() is False


def test_multi_node_rank_wraps_within_the_node(pre_init, monkeypatch):
    """The modulo must use the node's size."""
    monkeypatch.setenv("RANK", "11")
    monkeypatch.setenv("WORLD_SIZE", "16")
    monkeypatch.setenv("LOCAL_WORLD_SIZE", "8")
    assert ranks.get_local_rank() == 3
    assert ranks.get_node_rank() == 1


def test_explicit_launcher_vars_win_over_derivation(pre_init, monkeypatch):
    monkeypatch.setenv("RANK", "3")
    monkeypatch.setenv("WORLD_SIZE", "4")
    monkeypatch.setenv("LOCAL_RANK", "1")
    assert ranks.get_local_rank() == 1
    monkeypatch.delenv("LOCAL_RANK")
    monkeypatch.setenv("SLURM_LOCALID", "2")
    assert ranks.get_local_rank() == 2


def test_the_step_tasks_per_node_size_the_node(pre_init, monkeypatch):
    """SLURM's compact form ("8(x2)") sizes the node."""
    monkeypatch.setenv("SLURM_PROCID", "9")
    monkeypatch.setenv("SLURM_STEP_NUM_TASKS", "16")
    monkeypatch.setenv("SLURM_STEP_TASKS_PER_NODE", "8(x2)")
    assert ranks.get_local_world_size() == 8
    assert ranks.get_local_rank() == 1


def test_an_uneven_step_is_refused(pre_init, monkeypatch):
    """Its leading count is one node's, and would size the other nodes with it."""
    monkeypatch.setenv("SLURM_PROCID", "5")
    monkeypatch.setenv("SLURM_STEP_NUM_TASKS", "6")
    monkeypatch.setenv("SLURM_STEP_TASKS_PER_NODE", "4,2")
    with pytest.raises(ValueError, match="SLURM_STEP_TASKS_PER_NODE=4,2"):
        ranks.get_local_world_size()


def test_ranks_another_launcher_forked_inside_a_step_are_not_sized_by_it(pre_init, monkeypatch):
    """The step counts the launcher's parent, one task on the node, while eight ranks share the node's devices."""
    monkeypatch.setenv("RANK", "3")
    monkeypatch.setenv("WORLD_SIZE", "8")
    monkeypatch.setenv("SLURM_STEP_TASKS_PER_NODE", "1")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 8)
    assert ranks.get_local_world_size() == 8


def test_pre_init_last_resort_warns(pre_init, monkeypatch, caplog):
    """A silent ``return 1`` builds wrong node-local EP/CP groups and store scopes on a real multi-node job."""
    monkeypatch.setenv("RANK", "3")
    monkeypatch.setenv("WORLD_SIZE", "4")
    with caplog.at_level(logging.WARNING, logger=ranks.logger.name):
        ranks.get_local_world_size()
    assert any("local world size" in r.message for r in caplog.records), "pre-init fallback warned nothing"


def test_single_process_run_is_rank_zero_and_silent(pre_init, caplog):
    with caplog.at_level(logging.WARNING, logger=ranks.logger.name):
        assert ranks.get_local_world_size() == 1
        assert ranks.get_local_rank() == 0
    assert caplog.records == []


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
