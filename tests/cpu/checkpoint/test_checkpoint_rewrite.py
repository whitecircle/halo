#!/usr/bin/env python
"""A save rewriting a step directory in place replaces its per-rank resume files and clears earlier completion
markers. A merge hard-links them, so a truncating re-save would rewrite the merged copy."""

import datetime
import json
import os
from pathlib import Path

import pytest

from halo.checkpoint.atomic import link_or_copy_file
from halo.checkpoint.format import CHECKPOINT_COMPLETION_MARKERS, optimizer_shard_filename
from halo.checkpoint.parallel.completeness import prepare_checkpoint_rewrite
from halo.runtime import collectives, filesystem
from halo.runtime.filesystem import is_fs_save_rank
from halo.runtime.ranks import get_global_rank
from tests.common.checkpoint_resave import resave_in_place
from tests.common.distributed import reset_shared_filesystem_consensus
from tests.common.gloo import run_gloo_ranks

WORLD_SIZE = 2
PG_TIMEOUT = datetime.timedelta(seconds=60)
PER_NODE_ENV = {"LOCAL_RANK": "0", "LOCAL_WORLD_SIZE": "1", "DIST_OUTPUT_SHARED_FILESYSTEM": "0"}
SHARED_ENV = {"DIST_OUTPUT_SHARED_FILESYSTEM": "1"}


def _rank_files(rank: int) -> tuple[str, str]:
    return optimizer_shard_filename(rank), f"rng_state_{rank}.pth"


def _worker(rank: int, root: str, per_node: bool) -> None:
    outcome, phase = {}, "setup"
    try:
        filesystem.resolve_shared_filesystem_consensus()
        node_root = Path(root) / (f"node_{rank}" if per_node else "shared")
        step_dir, linked = node_root / "out" / "checkpoint-2", node_root / "merged"
        if is_fs_save_rank():
            step_dir.mkdir(parents=True)
            linked.mkdir()
            for saver in [rank] if per_node else range(WORLD_SIZE):
                for name in _rank_files(saver):
                    (step_dir / name).write_bytes(f"the earlier save's {name}".encode())
                    link_or_copy_file(step_dir / name, linked / name)
        collectives.barrier()
        before = {name: (linked / name).read_bytes() for name in os.listdir(linked)}

        phase = "re-save"
        resave_in_place(step_dir, world_size=WORLD_SIZE)

        outcome["rewritten"] = [name for name in _rank_files(rank) if (step_dir / name).read_bytes() != before[name]]
        outcome["changed"] = [name for name in sorted(before) if (linked / name).read_bytes() != before[name]]
    except Exception as e:  # A failing rank must still leave a verdict file, naming the phase that raised.
        outcome["error"] = f"{phase}: {type(e).__name__}: {e}"
    finally:
        reset_shared_filesystem_consensus()
    with open(os.path.join(root, f"result_{get_global_rank()}.json"), "w") as fh:
        json.dump(outcome, fh)


@pytest.mark.parametrize("per_node", [False, True], ids=["shared-output", "per-node-output"])
def test_a_resave_leaves_a_linked_copy_of_every_ranks_resume_files(tmp_path, per_node):
    run_gloo_ranks(
        _worker,
        WORLD_SIZE,
        str(tmp_path),
        per_node,
        pg_timeout=PG_TIMEOUT,
        env=PER_NODE_ENV if per_node else SHARED_ENV,
    )
    for rank in range(WORLD_SIZE):
        outcome = json.loads((tmp_path / f"result_{rank}.json").read_text())
        assert "error" not in outcome, f"rank {rank}, {outcome['error']}"
        assert outcome["rewritten"] == list(_rank_files(rank)), f"rank {rank}: premise — the re-save wrote its files"
        assert not outcome["changed"], f"rank {rank}: the re-save rewrote the linked {outcome['changed']}"


def test_the_completion_markers_go_whatever_the_listing_shows(tmp_path, monkeypatch):
    for name in CHECKPOINT_COMPLETION_MARKERS:
        (tmp_path / name).write_text("the earlier save's")
    monkeypatch.setattr(os, "listdir", lambda path: [])

    prepare_checkpoint_rewrite(str(tmp_path))

    left = [name for name in CHECKPOINT_COMPLETION_MARKERS if (tmp_path / name).exists()]
    assert not left, f"a marker the listing missed outlived the clearing and vouches for the re-save: {left}"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
