#!/usr/bin/env python
"""The output-filesystem reality probe: ``DIST_*_SHARED_FILESYSTEM`` vs what ``output_dir`` does.

The flags are pure declarations — nothing else in the toolkit checks them against a filesystem —
and both ways of getting them wrong are silent at 2+ nodes:

* declared SHARED on per-node storage: only global rank 0 writes ``trainer_state.json`` /
  ``scheduler.pt`` / ``rng_state``, so no checkpoint resumes on nodes 1..N
  (the mixin's ``save_on_each_node`` forcing keys off exactly this flag);
* declared PER-NODE on shared storage: every node's local rank 0 writes the SAME paths at once.

``verify_output_filesystem_sharing`` writes one sentinel from rank 0 and gathers who can see it;
``output_filesystem_contradiction`` is the whole verdict, identical on every rank because the
gathered list is. These tests drive the verdict directly with synthetic gathers, and drive the
collective wrapper against a fake process group to pin the skip rule and the cleanup.

Run: pytest tests/cpu/parallelism/test_output_fs_probe.py
"""

import os
from types import SimpleNamespace

import pytest

import src.trainers.mixins.base as base_mixin
from src.distributed import filesystem, runtime
from src.distributed.filesystem import output_filesystem_contradiction
from src.distributed.parallelism_config import ParallelismConfig


def test_declared_shared_on_per_node_storage_is_rejected():
    """16 ranks over 2 nodes; only node 0 sees rank 0's file."""
    seen = [True] * 8 + [False] * 8
    reason = output_filesystem_contradiction(declared_shared=True, seen=seen)
    assert reason is not None, "the resume-desync case must not pass silently"
    assert "rank 8" in reason, f"the first blind rank must be named: {reason}"
    assert "DIST_OUTPUT_SHARED_FILESYSTEM" in reason, f"the fix must be actionable: {reason}"
    assert "no checkpoint would resume" in reason, f"the consequence must be stated: {reason}"


def test_declared_per_node_on_shared_storage_is_rejected():
    """The mirror image: every node's local rank 0 would write the same checkpoint paths."""
    reason = output_filesystem_contradiction(declared_shared=False, seen=[True] * 16)
    assert reason is not None
    assert "DIST_OUTPUT_SHARED_FILESYSTEM" in reason, f"the fix must be actionable: {reason}"


def test_matching_declarations_pass():
    """Anti-over-rejection, both directions."""
    assert output_filesystem_contradiction(declared_shared=True, seen=[True] * 16) is None
    assert output_filesystem_contradiction(declared_shared=False, seen=[True] + [False] * 15) is None


def test_a_writer_that_cannot_read_its_own_sentinel_is_its_own_diagnosis():
    """``seen[0]`` False means the checkpoint writer's own directory is broken (full, read-only,
    stale handle) — reporting that as a sharing mismatch would send the operator to the wrong knob."""
    reason = output_filesystem_contradiction(declared_shared=True, seen=[False] * 16)
    assert reason is not None
    assert "rank 0 could not read back" in reason, f"the real fault must be named: {reason}"


def test_partial_visibility_is_not_shared():
    """A 4-node job where two nodes mount the share and two do not is still not a shared output."""
    assert output_filesystem_contradiction(declared_shared=True, seen=[True] * 24 + [False] * 8) is not None


def _fake_world(monkeypatch, *, world: int, local_world: int, rank: int, seen: list[bool]):
    """Drive the collective wrapper as ``rank`` of a ``world``-rank job whose gather returns ``seen``."""
    monkeypatch.setattr(runtime.dist, "is_available", lambda: True)
    monkeypatch.setattr(runtime.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(runtime.dist, "get_world_size", lambda *a, **k: world)
    monkeypatch.setattr(runtime.dist, "get_rank", lambda *a, **k: rank)
    monkeypatch.setattr(filesystem, "get_local_world_size", lambda: local_world)
    monkeypatch.setattr(filesystem, "get_num_nodes", lambda: max(1, world // local_world))
    monkeypatch.setattr(filesystem, "broadcast_from_rank0", lambda value: value)
    monkeypatch.setattr(filesystem, "reject_across_ranks", lambda *a, **k: None)
    # The NFS-lag poll budget is real wall time; a rank that sees nothing would sit it out here.
    monkeypatch.setattr(filesystem, "_OUTPUT_FS_PROBE_TIMEOUT_S", 0.0)

    def fake_gather(out_list, obj):
        del obj  # this rank's observation is supplied by the synthetic list
        out_list[:] = list(seen)

    monkeypatch.setattr(runtime.dist, "all_gather_object", fake_gather)


def test_a_single_node_job_is_skipped_entirely(monkeypatch, tmp_path):
    """Within one node every rank shares the mounts by construction, so the probe (and its poll)
    must not run — nor leave a sentinel behind."""
    _fake_world(monkeypatch, world=8, local_world=8, rank=0, seen=[True] * 8)
    monkeypatch.setattr(filesystem, "is_output_shared_filesystem", lambda: False)
    filesystem.verify_output_filesystem_sharing(str(tmp_path))
    assert os.listdir(tmp_path) == [], "a single-node run must write nothing"


def test_the_multi_node_probe_cleans_up_its_sentinel(monkeypatch, tmp_path):
    """A leftover dotfile in every checkpoint directory is its own bug."""
    _fake_world(monkeypatch, world=16, local_world=8, rank=0, seen=[True] * 16)
    monkeypatch.setattr(filesystem, "is_output_shared_filesystem", lambda: True)
    filesystem.verify_output_filesystem_sharing(str(tmp_path))
    assert os.listdir(tmp_path) == [], f"probe sentinel leaked: {os.listdir(tmp_path)}"


def test_the_multi_node_probe_raises_on_a_contradiction(monkeypatch, tmp_path):
    """Wiring check: the verdict must actually reach a raise, on every rank."""
    _fake_world(monkeypatch, world=16, local_world=8, rank=9, seen=[True] * 8 + [False] * 8)
    monkeypatch.setattr(filesystem, "is_output_shared_filesystem", lambda: True)
    with pytest.raises(RuntimeError, match="declared SHARED"):
        filesystem.verify_output_filesystem_sharing(str(tmp_path))


def test_a_per_node_declaration_never_waits_out_the_poll(monkeypatch, tmp_path):
    """Asymmetry that keeps the probe free on the common non-shared launch: only a rank that
    EXPECTS to see the sentinel pays the NFS-lag poll. Not seeing it agrees with the declaration,
    so there is nothing to wait for."""
    # Rank 0 sees its own write; no other node does — the ordinary per-node-NVMe shape.
    _fake_world(monkeypatch, world=16, local_world=8, rank=9, seen=[True] + [False] * 15)
    monkeypatch.setattr(filesystem, "is_output_shared_filesystem", lambda: False)
    monkeypatch.setattr(filesystem, "_OUTPUT_FS_PROBE_TIMEOUT_S", 30.0)
    budgets = []
    monkeypatch.setattr(filesystem, "_visible_within", lambda path, seconds: budgets.append(seconds) or False)
    filesystem.verify_output_filesystem_sharing(str(tmp_path))
    assert budgets == [0.0], f"a per-node declaration waited on a file it does not expect: {budgets}"


def test_a_directory_is_probed_once_per_process(monkeypatch, tmp_path):
    """The entry scripts probe before resume detection and every trainer probes again at construction;
    the second call must not write another sentinel or enter the gather again."""
    _fake_world(monkeypatch, world=16, local_world=8, rank=0, seen=[True] * 16)
    monkeypatch.setattr(filesystem, "is_output_shared_filesystem", lambda: True)
    monkeypatch.setattr(filesystem, "_PROBED_OUTPUT_DIRS", set())
    gathers = []
    real_gather = runtime.dist.all_gather_object
    monkeypatch.setattr(runtime.dist, "all_gather_object", lambda out, obj: gathers.append(1) or real_gather(out, obj))

    filesystem.verify_output_filesystem_sharing(str(tmp_path))
    filesystem.verify_output_filesystem_sharing(str(tmp_path))

    assert gathers == [1], f"the second probe of one directory gathered again: {gathers}"


def test_a_contradiction_is_never_remembered_as_probed(monkeypatch, tmp_path):
    _fake_world(monkeypatch, world=16, local_world=8, rank=9, seen=[True] * 8 + [False] * 8)
    monkeypatch.setattr(filesystem, "is_output_shared_filesystem", lambda: True)
    monkeypatch.setattr(filesystem, "_PROBED_OUTPUT_DIRS", set())
    for _ in range(2):
        with pytest.raises(RuntimeError, match="declared SHARED"):
            filesystem.verify_output_filesystem_sharing(str(tmp_path))


class _ProbeReached(Exception):
    """Carries control out of the trainer's shared init once the output directory was handed over."""


class _InitHost:
    """A bare trainer host for the shared init, stopped right after the output-filesystem probe."""

    def _configure_mixed_precision(self, kwargs, training_args):
        pass


def test_every_trainer_construction_probes_its_output_dir(monkeypatch):
    """A Python-API run never passes through the entry scripts' probe, so the trainer runs it itself."""
    probed = []
    monkeypatch.setattr(base_mixin, "verify_output_filesystem_sharing", probed.append)

    def stop(_training_args):
        raise _ProbeReached

    monkeypatch.setattr(base_mixin, "align_save_on_each_node", stop)
    args = SimpleNamespace(output_dir="/runs/api-launched", use_liger_kernel=False)
    kwargs = {"parallelism_config": ParallelismConfig(), "model": SimpleNamespace(config=None), "args": args}

    with pytest.raises(_ProbeReached):
        base_mixin.DistributedTrainerMixin._init_distributed_config(_InitHost(), kwargs)

    assert probed == ["/runs/api-launched"]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
