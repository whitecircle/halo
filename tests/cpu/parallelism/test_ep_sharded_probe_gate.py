"""The EP-sharded checkpoint probe is a filesystem read in front of world collectives.

Every loader that can be pointed at a per-rank EP-sharded directory has to refuse it — its keys
match no expert pattern, so the lazy loaders would plan nothing and fail much later as "never
materialized" instead of "merge the shards". The refusal is decided by reading the safetensors
index, which is a per-rank read on a filesystem that need not be shared: a rank whose cache is
half-populated raises inside ``json.load`` while its peers are already in the next collective, and a
rank whose cache disagrees takes a different branch entirely. Both hang the job with no diagnostic.

So rank 0 decides for everyone and broadcasts, and a rank that cannot resolve the directory rank 0
found says so by name. The PP and EP+TP loaders share the EP loader's gate rather than probing
per-rank.

    python tests/cpu/parallelism/test_ep_sharded_probe_gate.py
"""

from __future__ import annotations

import ast
import json
import pathlib

import pytest
from accelerate import PartialState

import src.distributed.expert_parallel.loading as ep_loading
from src.checkpoint.format import SAFETENSORS_INDEX_FILE
from src.distributed.expert_parallel.loading import reject_ep_sharded_checkpoint
from tests.common.utils import REPO_ROOT, imports_name

PartialState()  # the loader logs through accelerate's logger

MODEL_LOADING = REPO_ROOT / "src/distributed/loading/model_loading.py"


def _checkpoint(tmp_path: pathlib.Path, fmt: str | None) -> str:
    metadata = {"format": fmt} if fmt else {}
    (tmp_path / SAFETENSORS_INDEX_FILE).write_text(json.dumps({"metadata": metadata, "weight_map": {}}))
    return str(tmp_path)


def test_rank0_refuses_a_sharded_checkpoint_with_the_merge_instructions(tmp_path):
    with pytest.raises(NotImplementedError, match="merge_ep_shards.py"):
        reject_ep_sharded_checkpoint(_checkpoint(tmp_path, "ep_sharded"), "checkpoints/run")


def test_a_plain_checkpoint_passes(tmp_path):
    assert reject_ep_sharded_checkpoint(_checkpoint(tmp_path, None), "checkpoints/run") is None


def test_an_unresolvable_path_passes():
    """Nothing to probe is not a refusal — the caller's own loader reports an unusable path."""
    assert reject_ep_sharded_checkpoint(None, "org/model") is None


def test_only_rank0_touches_the_filesystem(tmp_path, monkeypatch):
    """The whole point of the gate: a non-zero rank must not run the probe at all, so a torn or
    half-fetched index on its node cannot raise it out of the collective its peers are in."""

    def _explode(_path):
        raise AssertionError("a non-zero rank probed the checkpoint itself")

    monkeypatch.setattr(ep_loading, "get_global_rank", lambda: 1)
    monkeypatch.setattr(ep_loading, "is_global_main_process", lambda: False)
    monkeypatch.setattr(ep_loading, "_detect_checkpoint_format", _explode)
    assert reject_ep_sharded_checkpoint(_checkpoint(tmp_path, "ep_sharded"), "checkpoints/run") is None


def test_a_rank_that_cannot_resolve_rank0s_verdict_says_so(monkeypatch):
    """Rank 0 found the shards; this rank's cache does not have them. Naming that beats raising
    ``_load_ep_model_sharded``'s merge text against a directory this rank cannot see."""
    monkeypatch.setattr(ep_loading, "get_global_rank", lambda: 3)
    monkeypatch.setattr(ep_loading, "is_global_main_process", lambda: False)
    monkeypatch.setattr(ep_loading, "broadcast_from_rank0", lambda value: "ep_sharded")
    with pytest.raises(RuntimeError, match="non-shared filesystem"):
        reject_ep_sharded_checkpoint(None, "org/model")


def test_a_torn_index_on_rank0_is_deferred_past_the_collective(tmp_path, monkeypatch):
    """Rank 0's probe is a ``json.load`` sitting IN FRONT of the broadcast, so a torn or
    half-fetched index raising inside it strands every peer in that broadcast — the exact hang the
    gate exists to prevent, moved one line earlier. It has to be fenced: recorded locally, the
    broadcast still entered, and the real cause raised uniformly afterwards."""
    (tmp_path / SAFETENSORS_INDEX_FILE).write_text('{"metadata": {"format": "ep_sha')
    broadcast = ep_loading.broadcast_from_rank0
    entered: list = []
    monkeypatch.setattr(ep_loading, "broadcast_from_rank0", lambda v: entered.append(v) or broadcast(v))

    with pytest.raises(RuntimeError, match="JSONDecodeError"):
        reject_ep_sharded_checkpoint(str(tmp_path), "checkpoints/run")

    assert entered == [None], "the probe must still enter the broadcast, or every peer blocks in it"


def test_the_pp_and_ep_tp_loaders_hold_no_probe_of_their_own():
    """Drift pin: importing the raw detector is what a re-introduced per-rank probe would need."""
    assert not imports_name(MODEL_LOADING, "_detect_checkpoint_format")
    assert imports_name(MODEL_LOADING, "reject_ep_sharded_checkpoint")


@pytest.mark.parametrize("loader", ["_load_pp_stage_model", "_load_ep_tp_model"])
def test_every_loader_that_can_be_handed_a_sharded_dir_runs_the_shared_gate(loader):
    tree = ast.parse(MODEL_LOADING.read_text(encoding="utf-8"))
    function = next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == loader)
    called = {
        node.func.id for node in ast.walk(function) if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    assert "reject_ep_sharded_checkpoint" in called, f"{loader} does not refuse an EP-sharded checkpoint"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
