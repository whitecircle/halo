#!/usr/bin/env python
"""Launcher-derived rank/world under a BARE ``srun`` — the shape that silently collapses the world.

``srun --ntasks-per-node=8`` starts eight processes per node and exports ``SLURM_PROCID`` /
``SLURM_NTASKS`` — never ``RANK`` / ``WORLD_SIZE``. Reading only the torchrun spellings makes every
one of those processes report rank 0 of world 1: no process group is built, ``is_global_main_process``
is True everywhere, and all 16 ranks of a 2-node job write one ``output_dir`` (interleaved
``trainer_state.json``, racing checkpoint shards) while the run reports itself as single-GPU.

The sibling readers already consult SLURM (``SLURM_LOCALID`` for the local rank,
``SLURM_NTASKS_PER_NODE`` for the node width), so the two global readers must do the same. Once they
see the real world, ``init_distributed`` must attempt the process group and say what is missing —
a bare srun provides no ``env://`` rendezvous, and the raw c10d error names neither the launcher nor
the fix.

The mirror-image failure is over-reading SLURM: ``sbatch``/``salloc`` export ``SLURM_NTASKS`` for the
whole ALLOCATION without ever launching those tasks, so the count is only a world when the task rank
``SLURM_PROCID`` is present beside it.

Run: pytest tests/cpu/parallelism/test_launcher_env.py
"""

import sys

import pytest
import torch

from src.distributed import runtime

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
    """A launcher-started process before ``init_process_group``, on a host with no visible GPUs."""
    for name in _LAUNCHER_VARS:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(runtime.dist, "is_initialized", lambda: False)
    monkeypatch.setattr(runtime, "_LWS_FALLBACK_ANNOUNCED", set())


def _bare_srun(monkeypatch, *, procid: int, ntasks: int, per_node: int) -> None:
    """Exactly what ``srun --ntasks-per-node=<per_node>`` exports — no RANK, no WORLD_SIZE."""
    monkeypatch.setenv("SLURM_PROCID", str(procid))
    monkeypatch.setenv("SLURM_NTASKS", str(ntasks))
    monkeypatch.setenv("SLURM_LOCALID", str(procid % per_node))
    monkeypatch.setenv("SLURM_NTASKS_PER_NODE", str(per_node))


def test_bare_srun_reports_the_real_world(pre_init, monkeypatch):
    """Rank 11 of a 2x8 bare-srun job must be rank 11 of world 16, not rank 0 of world 1."""
    _bare_srun(monkeypatch, procid=11, ntasks=16, per_node=8)
    assert runtime.launcher_global_rank() == 11
    assert runtime.launcher_global_world_size() == 16
    assert runtime.get_num_nodes() == 2
    assert runtime.get_node_rank() == 1


def test_a_procid_only_launcher_does_not_bind_the_node_to_cuda0(pre_init, monkeypatch):
    """Some srun configs export only ``SLURM_PROCID``/``SLURM_NTASKS``. The node width then falls
    back through the global reader, and the local rank through it — so a global reader stuck at
    rank 0 / world 1 makes every process on the node local rank 0 and binds them all to cuda:0."""
    monkeypatch.setenv("SLURM_PROCID", "5")
    monkeypatch.setenv("SLURM_NTASKS", "8")
    assert runtime.get_local_world_size() == 8
    assert runtime.get_local_rank() == 5
    assert runtime.is_local_main_process() is False


def test_torchrun_vars_win_over_slurm(pre_init, monkeypatch):
    """``srun --ntasks-per-node=1 torchrun ...`` sets BOTH: SLURM counts one task per NODE there, so
    taking it would report world 2 for a 16-rank job and shard the data 8x too coarsely."""
    _bare_srun(monkeypatch, procid=1, ntasks=2, per_node=1)
    monkeypatch.setenv("RANK", "11")
    monkeypatch.setenv("WORLD_SIZE", "16")
    monkeypatch.setenv("LOCAL_WORLD_SIZE", "8")
    assert runtime.launcher_global_rank() == 11
    assert runtime.launcher_global_world_size() == 16


def test_no_launcher_at_all_is_still_a_single_process(pre_init):
    """A plain ``python script.py`` must keep resolving to rank 0 of world 1."""
    assert runtime.launcher_global_rank() == 0
    assert runtime.launcher_global_world_size() == 1


def test_an_allocation_task_count_without_a_task_rank_is_not_a_world(pre_init, monkeypatch):
    """``SLURM_NTASKS`` counts only alongside ``SLURM_PROCID``.

    ``sbatch``/``salloc`` export the allocation's task count into the shell without launching those
    tasks; only ``srun`` adds the per-task ``SLURM_PROCID``. Counting NTASKS on its own makes a plain
    ``python train.py`` in an allocation shell report world 16 from one process — which then demands
    an ``env://`` rendezvous nothing set up (see the test below).
    """
    monkeypatch.setenv("SLURM_NTASKS", "16")
    monkeypatch.setenv("SLURM_NTASKS_PER_NODE", "8")
    assert runtime.launcher_global_world_size() == 1
    assert runtime.launcher_global_rank() == 0

    # Anti-vacuity: the same NTASKS is a world the moment the task rank appears beside it.
    monkeypatch.setenv("SLURM_PROCID", "3")
    assert runtime.launcher_global_world_size() == 16


def test_a_single_python_launch_inside_an_allocation_demands_no_rendezvous(pre_init, monkeypatch):
    """The consequence at the entry point: no group is built and no MASTER_ADDR is required.

    Under the NTASKS-alone reading this raises the bare-srun rendezvous error instead, so an
    ordinary single-GPU debug run inside an ``salloc`` shell dies at startup.
    """
    monkeypatch.setenv("SLURM_NTASKS", "16")
    monkeypatch.setenv("SLURM_NTASKS_PER_NODE", "8")
    assert runtime.init_distributed() is False


def test_bare_srun_without_rendezvous_fails_with_a_recipe(pre_init, monkeypatch):
    """``init_distributed`` must REACH the group build and refuse it by name.

    Silently returning False (RANK is unset) is the failure mode: the job runs to completion as N
    unsynchronized single-GPU trainings sharing one output directory.
    """
    _bare_srun(monkeypatch, procid=3, ntasks=16, per_node=8)
    with pytest.raises(RuntimeError) as excinfo:
        runtime.init_distributed()
    message = str(excinfo.value)
    assert "MASTER_ADDR" in message, f"the missing variable must be named: {message}"
    assert "srun" in message and "torchrun" in message, f"the launch recipe must be given: {message}"
    assert "world_size=16" in message, f"the world it detected must be reported: {message}"


def test_a_provided_rendezvous_is_not_refused(pre_init, monkeypatch):
    """Anti-over-rejection: exporting MASTER_ADDR/PORT on every task is a supported bare-srun launch,
    so the guard must let it through to the real ``init_process_group``."""
    _bare_srun(monkeypatch, procid=3, ntasks=16, per_node=8)
    monkeypatch.setenv("MASTER_ADDR", "10.0.0.1")
    monkeypatch.setenv("MASTER_PORT", "29500")
    runtime.require_rendezvous_env()  # must not raise


def test_bare_srun_with_rendezvous_hands_c10d_the_launcher_rank_and_world(pre_init, monkeypatch):
    """The remedy the guard names must actually work: c10d's ``env://`` handler reads ``RANK`` /
    ``WORLD_SIZE`` itself and raises when they are absent, so a bare srun that exported only
    ``MASTER_ADDR``/``MASTER_PORT`` still dies inside torch unless the SLURM-derived pair is passed
    explicitly."""
    _bare_srun(monkeypatch, procid=3, ntasks=16, per_node=8)
    monkeypatch.setenv("MASTER_ADDR", "10.0.0.1")
    monkeypatch.setenv("MASTER_PORT", "29500")
    captured: dict = {}
    monkeypatch.setattr(runtime.dist, "init_process_group", lambda **kwargs: captured.update(kwargs))
    monkeypatch.setattr(runtime, "apply_default_pg_timeout", lambda timeout: None)
    monkeypatch.setattr(runtime, "resolve_shared_filesystem_consensus", lambda: {})

    assert runtime.init_distributed(backend="gloo") is True
    assert captured["rank"] == 3 and captured["world_size"] == 16


def test_torchrun_launch_leaves_rank_and_world_to_the_env_handler(pre_init, monkeypatch):
    """Under torchrun ``RANK``/``WORLD_SIZE`` are the env handler's own inputs; passing them as well
    would only let a stale SLURM value contradict them."""
    _bare_srun(monkeypatch, procid=1, ntasks=2, per_node=1)
    monkeypatch.setenv("RANK", "11")
    monkeypatch.setenv("WORLD_SIZE", "16")
    monkeypatch.setenv("MASTER_ADDR", "10.0.0.1")
    monkeypatch.setenv("MASTER_PORT", "29500")
    captured: dict = {}
    monkeypatch.setattr(runtime.dist, "init_process_group", lambda **kwargs: captured.update(kwargs))
    monkeypatch.setattr(runtime, "apply_default_pg_timeout", lambda timeout: None)
    monkeypatch.setattr(runtime, "resolve_shared_filesystem_consensus", lambda: {})

    assert runtime.init_distributed(backend="gloo") is True
    assert "rank" not in captured and "world_size" not in captured


def test_single_task_srun_is_not_forced_into_a_process_group(pre_init, monkeypatch):
    """``srun -n1 python train.py`` is an ordinary single-GPU run: SLURM_PROCID is set but the world
    is 1, so no group is built and nothing is demanded of the environment."""
    _bare_srun(monkeypatch, procid=0, ntasks=1, per_node=1)
    assert runtime.init_distributed() is False


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
