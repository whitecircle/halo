#!/usr/bin/env python
"""The output-filesystem reality probe: ``DIST_*_SHARED_FILESYSTEM`` vs what ``output_dir`` does.
The flags are pure declarations and both ways of being wrong are silent at 2+ nodes."""

import datetime
import os
from types import SimpleNamespace

import pytest

import halo.trainers.distributed_trainer as base_mixin
from halo.distributed.parallelism_config import ParallelismConfig
from halo.env import fs_probe_timeout
from halo.runtime import filesystem, ranks
from halo.runtime.filesystem import OUTPUT_SIDE, filesystem_contradiction


def _output_contradiction(*, declared_shared: bool, seen: list[bool]) -> str | None:
    return filesystem_contradiction(OUTPUT_SIDE, "output_dir", declared_shared, seen)


def test_declared_shared_on_per_node_storage_is_rejected():
    """16 ranks over 2 nodes; only node 0 sees rank 0's file."""
    seen = [True] * 8 + [False] * 8
    reason = _output_contradiction(declared_shared=True, seen=seen)
    assert reason is not None, "the resume-desync case must not pass silently"
    assert "rank 8" in reason, f"the first blind rank must be named: {reason}"
    assert "DIST_OUTPUT_SHARED_FILESYSTEM" in reason, f"the fix must be actionable: {reason}"
    assert "no checkpoint would resume" in reason, f"the consequence must be stated: {reason}"


def test_declared_per_node_on_shared_storage_is_rejected():
    """Every node's local rank 0 would write the same checkpoint paths."""
    reason = _output_contradiction(declared_shared=False, seen=[True] * 16)
    assert reason is not None
    assert "DIST_OUTPUT_SHARED_FILESYSTEM" in reason, f"the fix must be actionable: {reason}"


def test_matching_declarations_pass():
    assert _output_contradiction(declared_shared=True, seen=[True] * 16) is None
    assert _output_contradiction(declared_shared=False, seen=[True] + [False] * 15) is None


def test_a_writer_that_cannot_read_its_own_sentinel_is_its_own_diagnosis():
    """``seen[0]`` False means the writer's own directory is broken; reporting a sharing mismatch sends the operator
    to the wrong knob."""
    reason = _output_contradiction(declared_shared=True, seen=[False] * 16)
    assert reason is not None
    assert "rank 0 could not read back" in reason, f"the real fault must be named: {reason}"


def test_partial_visibility_is_not_shared():
    assert _output_contradiction(declared_shared=True, seen=[True] * 24 + [False] * 8) is not None


def _fake_world(monkeypatch, *, world: int, local_world: int, rank: int, seen: list[bool]):
    monkeypatch.setattr(ranks.dist, "is_available", lambda: True)
    monkeypatch.setattr(ranks.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(ranks.dist, "get_world_size", lambda *a, **k: world)
    monkeypatch.setattr(ranks.dist, "get_rank", lambda *a, **k: rank)
    monkeypatch.setattr(filesystem, "get_local_world_size", lambda: local_world)
    monkeypatch.setattr(filesystem, "get_num_nodes", lambda: max(1, world // local_world))
    # Off rank 0 the broadcast delivers what rank 0 wrote.
    monkeypatch.setattr(
        filesystem, "broadcast_from_rank0", lambda value: value if value is not None else ("0", [None])
    )
    # The NFS-lag poll budget is real wall time.
    monkeypatch.setattr(filesystem, "fs_probe_timeout", lambda: datetime.timedelta(0))

    def fake_gather(out_list, obj):
        del obj  # Supplied by the synthetic list.
        out_list[:] = [[rank_seen] for rank_seen in seen]

    monkeypatch.setattr(ranks.dist, "all_gather_object", fake_gather)


def test_a_single_node_job_is_skipped_entirely(monkeypatch, tmp_path):
    """Within one node every rank shares the mounts."""
    _fake_world(monkeypatch, world=8, local_world=8, rank=0, seen=[True] * 8)
    monkeypatch.setattr(filesystem, "is_output_shared_filesystem", lambda: False)
    filesystem.verify_output_filesystem_sharing(str(tmp_path))
    assert os.listdir(tmp_path) == [], "a single-node run must write nothing"


def test_the_multi_node_probe_cleans_up_its_sentinel(monkeypatch, tmp_path):
    _fake_world(monkeypatch, world=16, local_world=8, rank=0, seen=[True] * 16)
    monkeypatch.setattr(filesystem, "is_output_shared_filesystem", lambda: True)
    filesystem.verify_output_filesystem_sharing(str(tmp_path))
    assert os.listdir(tmp_path) == [], f"probe sentinel leaked: {os.listdir(tmp_path)}"


def test_the_multi_node_probe_raises_on_a_contradiction(monkeypatch, tmp_path):
    _fake_world(monkeypatch, world=16, local_world=8, rank=9, seen=[True] * 8 + [False] * 8)
    monkeypatch.setattr(filesystem, "is_output_shared_filesystem", lambda: True)
    with pytest.raises(RuntimeError, match="declared SHARED"):
        filesystem.verify_output_filesystem_sharing(str(tmp_path))


def test_a_per_node_declaration_never_waits_out_the_poll(monkeypatch, tmp_path):
    """Only a rank expecting to see the sentinel pays the poll, which keeps the probe free on the common non-shared
    launch."""
    # Rank 0 sees its own write; no other node does (per-node NVMe).
    _fake_world(monkeypatch, world=16, local_world=8, rank=9, seen=[True] + [False] * 15)
    monkeypatch.setattr(filesystem, "is_output_shared_filesystem", lambda: False)
    monkeypatch.setenv("DIST_FS_PROBE_TIMEOUT_SECONDS", "30")
    monkeypatch.setattr(filesystem, "fs_probe_timeout", fs_probe_timeout)
    budgets = []
    monkeypatch.setattr(
        filesystem, "_visible_within", lambda paths, seconds: budgets.append(seconds) or [False] * len(paths)
    )
    filesystem.verify_output_filesystem_sharing(str(tmp_path))
    assert budgets == [0.0], f"a per-node declaration waited on a file it does not expect: {budgets}"


def test_a_shared_declaration_polls_for_the_configured_budget(monkeypatch, tmp_path):
    _fake_world(monkeypatch, world=16, local_world=8, rank=9, seen=[True] * 16)
    monkeypatch.setattr(filesystem, "is_output_shared_filesystem", lambda: True)
    monkeypatch.setenv("DIST_FS_PROBE_TIMEOUT_SECONDS", "180")
    monkeypatch.setattr(filesystem, "fs_probe_timeout", fs_probe_timeout)
    budgets = []
    monkeypatch.setattr(
        filesystem, "_visible_within", lambda paths, seconds: budgets.append(seconds) or [True] * len(paths)
    )
    filesystem.verify_output_filesystem_sharing(str(tmp_path))
    assert budgets == [180.0], f"the probe ignored DIST_FS_PROBE_TIMEOUT_SECONDS: {budgets}"


def test_a_shared_declaration_the_probe_cannot_confirm_names_the_attribute_cache():
    """The refusal points at the probe budget first, since the per-node flag on that mount would race every node's
    writer."""
    reason = _output_contradiction(declared_shared=True, seen=[True] * 8 + [False] * 8)
    assert "DIST_FS_PROBE_TIMEOUT_SECONDS" in reason and "acdirmax" in reason, reason


def test_an_output_dir_rank_zero_cannot_write_raises_naming_it(monkeypatch, tmp_path):
    _fake_world(monkeypatch, world=16, local_world=8, rank=0, seen=[False] * 16)
    monkeypatch.setattr(filesystem, "is_output_shared_filesystem", lambda: True)
    blocking = tmp_path / "a_file"
    blocking.write_text("")
    with pytest.raises(OSError, match="Output-filesystem probe: could not write"):
        filesystem.verify_output_filesystem_sharing(str(blocking / "out"))


def test_a_directory_is_probed_once_per_process(monkeypatch, tmp_path):
    """Entry scripts probe before resume detection and every trainer probes again at construction."""
    _fake_world(monkeypatch, world=16, local_world=8, rank=0, seen=[True] * 16)
    monkeypatch.setattr(filesystem, "is_output_shared_filesystem", lambda: True)
    monkeypatch.setattr(filesystem, "_probed_output_dirs", set())
    gathers = []
    real_gather = ranks.dist.all_gather_object
    monkeypatch.setattr(ranks.dist, "all_gather_object", lambda out, obj: gathers.append(1) or real_gather(out, obj))

    filesystem.verify_output_filesystem_sharing(str(tmp_path))
    filesystem.verify_output_filesystem_sharing(str(tmp_path))

    assert gathers == [1], f"the second probe of one directory gathered again: {gathers}"


def test_a_contradiction_is_never_remembered_as_probed(monkeypatch, tmp_path):
    _fake_world(monkeypatch, world=16, local_world=8, rank=9, seen=[True] * 8 + [False] * 8)
    monkeypatch.setattr(filesystem, "is_output_shared_filesystem", lambda: True)
    monkeypatch.setattr(filesystem, "_probed_output_dirs", set())
    for _ in range(2):
        with pytest.raises(RuntimeError, match="declared SHARED"):
            filesystem.verify_output_filesystem_sharing(str(tmp_path))


class _ProbeReached(Exception):
    """Carries control out of the trainer's shared init once the output directory was handed over."""


class _InitHost:
    _apply_training_args_overrides = base_mixin.DistributedTrainerMixin._apply_training_args_overrides

    def _snapshot_model_config_state(self, model_config):
        pass


def test_every_trainer_construction_probes_its_output_dir(monkeypatch):
    """A Python-API run never passes through the entry scripts' probe."""
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
