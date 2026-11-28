#!/usr/bin/env python
"""Rank and world under each Slurm launch. A bare ``srun`` step exports none of torchrun's spellings, and reading only
those gives rank 0 of world 1 everywhere, all writing one ``output_dir``. An ``sbatch`` script and an ``salloc`` shell
hold the job's task count for one process, which is no world."""

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
    "SLURM_TASKS_PER_NODE",
    "SLURM_STEP_NUM_TASKS",
    "SLURM_STEP_TASKS_PER_NODE",
    "MASTER_ADDR",
    "MASTER_PORT",
)
# What Slurm 23.11 exports to a process started once inside a job submitted with ``-N2 --ntasks-per-node=4``.
_JOB = {"SLURM_NTASKS": "8", "SLURM_NTASKS_PER_NODE": "4", "SLURM_TASKS_PER_NODE": "4(x2)"}
_SINGLE_LAUNCHES = {
    "sbatch script": {**_JOB, "SLURM_PROCID": "0", "SLURM_LOCALID": "0"},
    "salloc shell": _JOB,
    "salloc interactive step": {
        **_JOB,
        "SLURM_PROCID": "0",
        "SLURM_LOCALID": "0",
        "SLURM_STEP_NUM_TASKS": "1",
        "SLURM_STEP_TASKS_PER_NODE": "1",
    },
}


@pytest.fixture()
def pre_init(monkeypatch):
    for name in _LAUNCHER_VARS:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(ranks.dist, "is_initialized", lambda: False)
    monkeypatch.setattr(ranks, "_local_world_size_fallback_warned", set())


def _bare_srun(monkeypatch, *, procid: int, ntasks: int, per_node: int, job_per_node: int | None = None) -> None:
    """One task of an ``srun`` step of ``ntasks`` tasks, ``per_node`` to a node, in block order. The step inherits its
    job's ``--ntasks-per-node``, ``job_per_node``."""
    nodes = ntasks // per_node
    shape = f"{per_node}(x{nodes})" if nodes > 1 else str(per_node)
    monkeypatch.setenv("SLURM_PROCID", str(procid))
    monkeypatch.setenv("SLURM_LOCALID", str(procid % per_node))
    monkeypatch.setenv("SLURM_NTASKS", str(ntasks))
    monkeypatch.setenv("SLURM_NTASKS_PER_NODE", str(job_per_node or per_node))
    monkeypatch.setenv("SLURM_TASKS_PER_NODE", shape)
    monkeypatch.setenv("SLURM_STEP_NUM_TASKS", str(ntasks))
    monkeypatch.setenv("SLURM_STEP_TASKS_PER_NODE", shape)


def test_bare_srun_reports_the_real_world(pre_init, monkeypatch):
    _bare_srun(monkeypatch, procid=11, ntasks=16, per_node=8)
    assert ranks.launcher_global_rank() == 11
    assert ranks.launcher_global_world_size() == 16
    assert ranks.get_num_nodes() == 2
    assert ranks.get_node_rank() == 1


def test_a_step_smaller_than_its_job_takes_its_own_node_size(pre_init, monkeypatch):
    """``srun -n2`` in a job of four tasks per node puts one task on each node, under the job's ``NTASKS_PER_NODE``."""
    _bare_srun(monkeypatch, procid=1, ntasks=2, per_node=1, job_per_node=4)
    assert ranks.launcher_global_world_size() == 2
    assert ranks.get_local_world_size() == 1
    assert ranks.get_num_nodes() == 2
    assert ranks.get_node_rank() == 1


def test_a_step_with_no_node_size_does_not_bind_the_node_to_cuda0(pre_init, monkeypatch):
    """A global reader stuck at rank 0 makes every process local rank 0 and binds them all to cuda:0."""
    monkeypatch.setenv("SLURM_PROCID", "5")
    monkeypatch.setenv("SLURM_STEP_NUM_TASKS", "8")
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
    assert ranks.get_local_world_size() == 8
    assert ranks.get_num_nodes() == 2


def test_no_launcher_at_all_is_still_a_single_process(pre_init):
    assert ranks.launcher_global_rank() == 0
    assert ranks.launcher_global_world_size() == 1


@pytest.mark.parametrize("launch", _SINGLE_LAUNCHES)
def test_a_python_started_once_inside_a_job_is_one_process(pre_init, monkeypatch, launch):
    """Read as a world of the job's eight tasks, a single-GPU run dies at startup asking for a rendezvous."""
    for name, value in _SINGLE_LAUNCHES[launch].items():
        monkeypatch.setenv(name, value)
    assert ranks.launcher_global_world_size() == 1
    assert ranks.launcher_global_rank() == 0
    assert ranks.get_local_world_size() == 1
    assert ranks.get_num_nodes() == 1
    assert process_group.init_distributed() is False


def test_the_same_job_is_a_world_under_srun(pre_init, monkeypatch):
    _bare_srun(monkeypatch, procid=3, ntasks=8, per_node=4)
    assert ranks.launcher_global_world_size() == 8


def test_a_task_rank_without_its_step_world_is_refused(pre_init, monkeypatch):
    """A container passed ``SLURM_PROCID`` and the job's ``SLURM_NTASKS`` alone reads a world of 1, and every task
    would train by itself on one ``output_dir``."""
    monkeypatch.setenv("SLURM_PROCID", "5")
    monkeypatch.setenv("SLURM_NTASKS", "8")
    monkeypatch.setenv("MASTER_ADDR", "10.0.0.1")
    monkeypatch.setenv("MASTER_PORT", "29500")
    # A reading that takes the job's count for the world reaches the rendezvous: stubbed so it fails here, at once.
    monkeypatch.setattr(ranks.dist, "init_process_group", lambda **kwargs: None)
    monkeypatch.setattr(process_group, "_apply_default_pg_timeout", lambda timeout: None)
    monkeypatch.setattr(process_group, "resolve_shared_filesystem_consensus", lambda: None)
    with pytest.raises(RuntimeError, match="rank 5 in a world of 1.*SLURM_STEP_NUM_TASKS"):
        process_group.init_distributed(backend="gloo")


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
