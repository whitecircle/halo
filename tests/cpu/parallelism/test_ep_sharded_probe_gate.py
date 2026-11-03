"""The EP-sharded checkpoint probe is a filesystem read in front of world collectives, so rank 0 decides and
broadcasts. A half-populated cache raising in ``json.load`` on one rank hangs the job with no diagnostic."""

from __future__ import annotations

import ast
import json
import pathlib

import pytest

import halo.model_loading.parallel.model_source as model_source
import halo.runtime.consensus as consensus
from halo.checkpoint.format import SAFETENSORS_INDEX_FILE
from halo.model_loading.parallel.lazy_plan import agree_lazy_load
from halo.model_loading.parallel.model_source import ModelSource, reject_ep_sharded_checkpoint
from tests.common.repo import REPO_ROOT
from tests.common.source_sweep import imports_name

# Each could hold a probe of its own, with the call through which it reaches the shared gate.
LOADER_MODULES = {
    "load_pp_stage_model": (REPO_ROOT / "halo/model_loading/parallel/pp.py", "agree_lazy_load"),
    "load_ep_tp_model": (REPO_ROOT / "halo/model_loading/parallel/tp.py", "load_ep_model"),
}


def _checkpoint(tmp_path: pathlib.Path, fmt: str | None) -> ModelSource:
    metadata = {"format": fmt} if fmt else {}
    (tmp_path / SAFETENSORS_INDEX_FILE).write_text(json.dumps({"metadata": metadata, "weight_map": {}}))
    return ModelSource("checkpoints/run", None, str(tmp_path))


def test_rank0_refuses_a_sharded_checkpoint_with_the_merge_instructions(tmp_path):
    with pytest.raises(ValueError, match="merge_ep_shards.py"):
        reject_ep_sharded_checkpoint(_checkpoint(tmp_path, "ep_sharded"))


def test_the_refusal_names_the_gathered_save_knob(tmp_path):
    """The YAML setting a user writes."""
    with pytest.raises(ValueError) as err:
        reject_ep_sharded_checkpoint(_checkpoint(tmp_path, "ep_sharded"))
    assert "save_sharded_ep: false" in str(err.value), err.value


def test_a_plain_checkpoint_passes(tmp_path):
    assert reject_ep_sharded_checkpoint(_checkpoint(tmp_path, None)) is None


def test_only_rank0_touches_the_filesystem(tmp_path, monkeypatch):
    """A torn index on a non-zero rank's node cannot raise it out of the collective."""

    def _explode(*_args, **_kwargs):
        raise AssertionError("a non-zero rank probed the checkpoint itself")

    monkeypatch.setattr(consensus, "is_global_main_process", lambda: False)
    monkeypatch.setattr(model_source, "read_checkpoint_index", _explode)
    assert reject_ep_sharded_checkpoint(_checkpoint(tmp_path, "ep_sharded")) is None


def test_a_torn_index_on_rank0_is_deferred_past_the_collective(tmp_path, monkeypatch):
    """Recorded locally, the broadcast still entered, the cause raised uniformly afterwards."""
    (tmp_path / SAFETENSORS_INDEX_FILE).write_text('{"metadata": {"format": "ep_sha')
    broadcast = consensus.broadcast_from_rank0
    entered: list = []
    monkeypatch.setattr(consensus, "broadcast_from_rank0", lambda v: entered.append(v) or broadcast(v))

    with pytest.raises(RuntimeError, match="JSONDecodeError"):
        reject_ep_sharded_checkpoint(ModelSource("checkpoints/run", None, str(tmp_path)))

    assert entered == [None], "the probe must still enter the broadcast, or every peer blocks in it"


def test_the_lazy_checkpoint_verdict_refuses_a_sharded_checkpoint(tmp_path):
    with pytest.raises(ValueError, match="merge_ep_shards.py"):
        agree_lazy_load(_checkpoint(tmp_path, "ep_sharded"))


@pytest.mark.parametrize("loader", sorted(LOADER_MODULES))
def test_every_loader_that_can_be_handed_a_sharded_dir_runs_the_shared_gate(loader):
    """Importing the index predicate is what a per-rank probe would need."""
    module, gate = LOADER_MODULES[loader]
    assert not imports_name(module, "index_declares_ep_shards")
    tree = ast.parse(module.read_text(encoding="utf-8"))
    function = next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == loader)
    called = {
        node.func.id for node in ast.walk(function) if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    assert gate in called, f"{loader} does not refuse an EP-sharded checkpoint"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
