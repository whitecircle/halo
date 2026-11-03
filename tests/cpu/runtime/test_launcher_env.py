#!/usr/bin/env python
"""Rank and world under a bare ``srun``, which exports ``SLURM_PROCID`` / ``SLURM_NTASKS`` only. Reading torchrun's
spellings gives rank 0 of world 1 everywhere, all writing one ``output_dir``."""

import pytest
import torch

from halo.runtime import process_group, ranks

_LAUNCHER_VARS = (
    "RANK",
    "WORLD_SIZE",
    "LOCAL_RANK",
    "LOCAL_WORLD_SIZE",
    "SLURM_PROCID",
    "SLURM_NTASKS",
    "SLURM_LOCALID",
    "SLURM_NTASKS_PER_NODE",
    "MASTER_ADDR",
    "MASTER_PORT",
)


@pytest.fixture()
def pre_init(monkeypatch):
    for name in _LAUNCHER_VARS:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(ranks.dist, "is_initialized", lambda: False)
    monkeypatch.setattr(ranks, "_lws_fallback_announced", set())


def _bare_srun(monkeypatch, *, procid: int, ntasks: int, per_node: int) -> None:
    monkeypatch.setenv("SLURM_PROCID", str(procid))
    monkeypatch.setenv("SLURM_NTASKS", str(ntasks))
    monkeypatch.setenv("SLURM_LOCALID", str(procid % per_node))
    monkeypatch.setenv("SLURM_NTASKS_PER_NODE", str(per_node))


def test_bare_srun_reports_the_real_world(pre_init, monkeypatch):
    _bare_srun(monkeypatch, procid=11, ntasks=16, per_node=8)
    assert ranks.launcher_global_rank() == 11
    assert ranks.launcher_global_world_size() == 16
    assert ranks.get_num_nodes() == 2
    assert ranks.get_node_rank() == 1


def test_a_procid_only_launcher_does_not_bind_the_node_to_cuda0(pre_init, monkeypatch):
    """A global reader stuck at rank 0 makes every process local rank 0 and binds them all to cuda:0."""
    monkeypatch.setenv("SLURM_PROCID", "5")
    monkeypatch.setenv("SLURM_NTASKS", "8")
    assert ranks.get_local_world_size() == 8
    assert ranks.get_local_rank() == 5
    assert ranks.is_local_main_process() is False


def test_torchrun_vars_win_over_slurm(pre_init, monkeypatch):
    """SLURM counts one task per node there, so taking it reports world 2 for a 16-rank job and shards the data 8x
    too coarsely."""
    _bare_srun(monkeypatch, procid=1, ntasks=2, per_node=1)
    monkeypatch.setenv("RANK", "11")
    monkeypatch.setenv("WORLD_SIZE", "16")
    monkeypatch.setenv("LOCAL_WORLD_SIZE", "8")
    assert ranks.launcher_global_rank() == 11
    assert ranks.launcher_global_world_size() == 16


def test_no_launcher_at_all_is_still_a_single_process(pre_init):
    assert ranks.launcher_global_rank() == 0
    assert ranks.launcher_global_world_size() == 1


def test_an_allocation_task_count_without_a_task_rank_is_not_a_world(pre_init, monkeypatch):
    """``sbatch``/``salloc`` export the task count without launching tasks; only ``srun`` adds ``SLURM_PROCID``."""
    monkeypatch.setenv("SLURM_NTASKS", "16")
    monkeypatch.setenv("SLURM_NTASKS_PER_NODE", "8")
    assert ranks.launcher_global_world_size() == 1
    assert ranks.launcher_global_rank() == 0

    # Anti-vacuity: the same NTASKS is a world once the task rank appears beside it.
    monkeypatch.setenv("SLURM_PROCID", "3")
    assert ranks.launcher_global_world_size() == 16


def test_a_single_python_launch_inside_an_allocation_demands_no_rendezvous(pre_init, monkeypatch):
    """Under the NTASKS-alone reading, a single-GPU debug run in an ``salloc`` shell dies at startup."""
    monkeypatch.setenv("SLURM_NTASKS", "16")
    monkeypatch.setenv("SLURM_NTASKS_PER_NODE", "8")
    assert process_group.init_distributed() is False


def test_bare_srun_without_rendezvous_fails_with_a_recipe(pre_init, monkeypatch):
    """Returning False runs N unsynchronized single-GPU trainings sharing one output directory."""
    _bare_srun(monkeypatch, procid=3, ntasks=16, per_node=8)
    with pytest.raises(RuntimeError) as excinfo:
        process_group.init_distributed()
    message = str(excinfo.value)
    assert "MASTER_ADDR" in message, f"the missing variable must be named: {message}"
    assert "srun" in message and "torchrun" in message, f"the launch recipe must be given: {message}"
    assert "bash -c" in message and "\\$SLURM_NODEID" in message, f"unquoted, every node expands rank 0: {message}"
    assert "world_size=16" in message, f"the world it detected must be reported: {message}"


def test_a_provided_rendezvous_is_not_refused(pre_init, monkeypatch):
    """Exporting MASTER_ADDR/PORT on every task is a supported bare-srun launch."""
    _bare_srun(monkeypatch, procid=3, ntasks=16, per_node=8)
    monkeypatch.setenv("MASTER_ADDR", "10.0.0.1")
    monkeypatch.setenv("MASTER_PORT", "29500")
    process_group._require_rendezvous_env()


def test_bare_srun_with_rendezvous_hands_c10d_the_launcher_rank_and_world(pre_init, monkeypatch):
    """c10d's ``env://`` handler reads ``RANK`` / ``WORLD_SIZE`` itself and raises when absent, so the SLURM-derived
    pair is passed explicitly."""
    _bare_srun(monkeypatch, procid=3, ntasks=16, per_node=8)
    monkeypatch.setenv("MASTER_ADDR", "10.0.0.1")
    monkeypatch.setenv("MASTER_PORT", "29500")
    captured: dict = {}
    monkeypatch.setattr(ranks.dist, "init_process_group", lambda **kwargs: captured.update(kwargs))
    monkeypatch.setattr(process_group, "_apply_default_pg_timeout", lambda timeout: None)
    monkeypatch.setattr(process_group, "resolve_shared_filesystem_consensus", lambda: None)

    assert process_group.init_distributed(backend="gloo") is True
    assert captured["rank"] == 3 and captured["world_size"] == 16


def test_torchrun_launch_leaves_rank_and_world_to_the_env_handler(pre_init, monkeypatch):
    """Passing them as well would only let a stale SLURM value contradict them."""
    _bare_srun(monkeypatch, procid=1, ntasks=2, per_node=1)
    monkeypatch.setenv("RANK", "11")
    monkeypatch.setenv("WORLD_SIZE", "16")
    monkeypatch.setenv("MASTER_ADDR", "10.0.0.1")
    monkeypatch.setenv("MASTER_PORT", "29500")
    captured: dict = {}
    monkeypatch.setattr(ranks.dist, "init_process_group", lambda **kwargs: captured.update(kwargs))
    monkeypatch.setattr(process_group, "_apply_default_pg_timeout", lambda timeout: None)
    monkeypatch.setattr(process_group, "resolve_shared_filesystem_consensus", lambda: None)

    assert process_group.init_distributed(backend="gloo") is True
    assert "rank" not in captured and "world_size" not in captured


def test_single_task_srun_is_not_forced_into_a_process_group(pre_init, monkeypatch):
    """SLURM_PROCID is set but the world is 1."""
    _bare_srun(monkeypatch, procid=0, ntasks=1, per_node=1)
    assert process_group.init_distributed() is False


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
