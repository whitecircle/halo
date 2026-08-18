#!/usr/bin/env python
"""The output-filesystem reality probe: ``DIST_*_SHARED_FILESYSTEM`` vs what ``output_dir`` does.

The flags are pure declarations — nothing else in the toolkit checks them against a filesystem —
and both ways of getting them wrong are silent at 2+ nodes:

* declared SHARED on per-node storage: only global rank 0 writes ``trainer_state.json`` /
  ``scheduler.pt`` / ``rng_state``, so nodes 1..N resume at ``global_step=0`` and the run desyncs
  (the mixin's ``save_on_each_node`` forcing keys off exactly this flag);
* declared PER-NODE on shared storage: every node's local rank 0 writes the SAME paths at once.

``verify_output_filesystem_sharing`` writes one sentinel from rank 0 and gathers who can see it;
``output_filesystem_contradiction`` is the whole verdict, identical on every rank because the
gathered list is. These tests drive the verdict directly with synthetic gathers, and drive the
collective wrapper against a fake process group to pin the skip rule and the cleanup.

Run: pytest tests/cpu/parallelism/test_output_fs_probe.py
"""

import os
import sys

import pytest

from src.distributed import filesystem, runtime
from src.distributed.filesystem import output_filesystem_contradiction


def test_declared_shared_on_per_node_storage_is_rejected():
    """16 ranks over 2 nodes; only node 0 sees rank 0's file."""
    seen = [True] * 8 + [False] * 8
    reason = output_filesystem_contradiction(declared_shared=True, seen=seen)
    assert reason is not None, "the resume-desync case must not pass silently"
    assert "rank 8" in reason, f"the first blind rank must be named: {reason}"
    assert "DIST_OUTPUT_SHARED_FILESYSTEM" in reason, f"the fix must be actionable: {reason}"
    assert "global_step=0" in reason, f"the consequence must be stated: {reason}"


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


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
