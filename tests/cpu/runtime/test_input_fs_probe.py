#!/usr/bin/env python
"""The input-filesystem probe: ``DIST_INPUT_SHARED_FILESYSTEM`` vs what the HF caches do, on two gloo ranks standing
for nodes. Both wrong declarations are silent at 2+ nodes: unfilled caches, or every node filling the same files."""

import datetime
import json
import os

import pytest

from halo.entrypoints import script
from halo.runtime import filesystem
from halo.runtime.filesystem import verify_input_filesystem_sharing
from tests.common.gloo import run_gloo_ranks

WORLD_SIZE = 2
PG_TIMEOUT = datetime.timedelta(seconds=60)
# A one-second poll, so an unseen sentinel costs a second, not the default minute.
TWO_NODE_ENV = {"LOCAL_WORLD_SIZE": "1", "LOCAL_RANK": "0", "DIST_FS_PROBE_TIMEOUT_SECONDS": "1"}


_CACHES = ("HF_HUB_CACHE", "HF_DATASETS_CACHE")


def _probe_worker(
    rank: int, root: str, node_local: tuple[str, ...], declared_shared: bool, unwritable_hub: bool
) -> None:
    os.environ["DIST_INPUT_SHARED_FILESYSTEM"] = "1" if declared_shared else "0"
    filesystem.resolve_shared_filesystem_consensus()
    caches = {name: os.path.join(root, f"node{rank}" if name in node_local else "shared", name) for name in _CACHES}
    if unwritable_hub:
        caches["HF_HUB_CACHE"] = os.path.join(root, "read_only_file", "hub")
    try:
        verify_input_filesystem_sharing(caches)
        outcome = "passed"
    except Exception as exc:  # The verdict on every rank is the subject.
        outcome = f"{type(exc).__name__}: {exc}"
    with open(os.path.join(root, f"outcome_{rank}.json"), "w") as fh:
        json.dump(outcome, fh)


def _outcomes(tmp_path, *, node_local: bool | str, declared_shared: bool, unwritable_hub: bool = False) -> list[str]:
    """``node_local``: both caches (``True``), neither (``False``), or the one named."""
    (tmp_path / "read_only_file").write_text("")
    run_gloo_ranks(
        _probe_worker,
        WORLD_SIZE,
        str(tmp_path),
        _CACHES if node_local is True else (node_local,) if node_local else (),
        declared_shared,
        unwritable_hub,
        pg_timeout=PG_TIMEOUT,
        env=TWO_NODE_ENV,
    )
    return [json.loads((tmp_path / f"outcome_{rank}.json").read_text()) for rank in range(WORLD_SIZE)]


def test_a_shared_declaration_over_node_local_caches_is_refused_on_every_rank(tmp_path):
    for outcome in _outcomes(tmp_path, node_local=True, declared_shared=True):
        assert outcome.startswith("RuntimeError: Input filesystem is declared SHARED"), outcome
        assert "HF_HUB_CACHE" in outcome and "rank 1" in outcome, outcome
        assert "DIST_INPUT_SHARED_FILESYSTEM=0" in outcome, outcome
        assert "point HF_HUB_CACHE at the shared mount" in outcome, outcome


def test_a_per_node_declaration_over_a_shared_cache_is_refused_on_every_rank(tmp_path):
    for outcome in _outcomes(tmp_path, node_local=False, declared_shared=False):
        assert outcome.startswith("RuntimeError: Input filesystem is declared PER-NODE"), outcome
        assert "DIST_INPUT_SHARED_FILESYSTEM=1" in outcome, outcome
        assert "put HF_HUB_CACHE on each node's local storage" in outcome, outcome


@pytest.mark.parametrize("declared_shared", [True, False], ids=["declared-shared", "declared-per-node"])
def test_caches_on_different_kinds_of_storage_are_refused_under_either_declaration(tmp_path, declared_shared):
    """Either declaration contradicts one cache, and the refusal says the two must match rather than flip the flag
    into the other refusal."""
    flagged = "HF_HUB_CACHE" if declared_shared else "HF_DATASETS_CACHE"
    remedy = f"point {flagged} at the shared mount" if declared_shared else f"put {flagged} on each node's local"
    for outcome in _outcomes(tmp_path, node_local="HF_HUB_CACHE", declared_shared=declared_shared):
        assert outcome.startswith("RuntimeError: Input filesystem is declared"), outcome
        assert remedy in outcome and "HF_HOME" not in outcome, outcome
        assert "both must be on the same kind of storage" in outcome, outcome


@pytest.mark.parametrize(("node_local", "declared_shared"), [(False, True), (True, False)], ids=["shared", "per-node"])
def test_a_matching_declaration_passes_and_leaves_no_sentinel(tmp_path, node_local, declared_shared):
    assert _outcomes(tmp_path, node_local=node_local, declared_shared=declared_shared) == ["passed"] * WORLD_SIZE
    leaked = [path for path, _, files in os.walk(tmp_path) for name in files if name.startswith(".halo_fs_probe_")]
    assert leaked == [], f"probe sentinels leaked: {leaked}"


def test_a_cache_rank_zero_cannot_write_is_skipped_and_the_rest_still_probed(tmp_path):
    """A read-only, pre-staged cache is filled by nobody."""
    for outcome in _outcomes(tmp_path, node_local=True, declared_shared=True, unwritable_hub=True):
        assert "point HF_DATASETS_CACHE at the shared mount" in outcome, outcome
        assert "declared SHARED but" in outcome and "HF_HUB_CACHE (" not in outcome, outcome
    assert _outcomes(tmp_path, node_local=False, declared_shared=True, unwritable_hub=True) == ["passed"] * WORLD_SIZE


class _RecordingParser:
    def __init__(self, dataclass_types, events):
        self.events = events

    def load_values(self):
        self.events.append("parse")
        return {}

    def parse_dict(self, values):
        return ["parsed"]


def test_a_training_script_probes_its_caches_after_the_group_starts_and_before_any_hub_read(monkeypatch):
    """A follower reads only the cache its main rank fills, so the declaration is verified first."""
    events: list = []
    monkeypatch.setattr(script, "init_distributed", lambda: events.append("group"))
    monkeypatch.setattr(script, "verify_input_filesystem_sharing", lambda caches: events.append(sorted(caches)))
    monkeypatch.setattr(script, "HaloArgumentParser", lambda types: _RecordingParser(types, events))
    monkeypatch.setattr(script, "install_log_tee", lambda output_dir: None)

    assert script.parse_script_args(object) == "parsed"
    assert events == ["group", ["HF_DATASETS_CACHE", "HF_HUB_CACHE"], "parse"]


def test_a_single_node_job_is_not_probed(tmp_path, monkeypatch):
    """Every rank shares the mounts by construction."""
    monkeypatch.setattr(filesystem, "get_global_world_size", lambda: 8)
    monkeypatch.setattr(filesystem, "get_local_world_size", lambda: 8)
    verify_input_filesystem_sharing({"HF_DATASETS_CACHE": str(tmp_path / "datasets")})
    assert list(tmp_path.iterdir()) == []


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
