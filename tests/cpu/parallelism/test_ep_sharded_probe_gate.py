"""The EP-sharded checkpoint probe is a filesystem read in front of world collectives.

Every loader that can be pointed at a per-rank EP-sharded directory has to refuse it — its keys
match no expert pattern, so the lazy loaders would plan nothing and fail much later as "never
materialized" instead of "merge the shards". The refusal is decided by reading the safetensors
index, which is a per-rank read on a filesystem that need not be shared: a rank whose cache is
half-populated raises inside ``json.load`` while its peers are already in the next collective, and a
rank whose cache disagrees takes a different branch entirely. Both hang the job with no diagnostic.

So rank 0 decides for everyone and broadcasts. The PP loader shares the EP loader's gate through the
lazy-checkpoint verdict, and the EP+TP loader loads through the EP loader itself; neither probes per-rank.

    python tests/cpu/parallelism/test_ep_sharded_probe_gate.py
"""

from __future__ import annotations

import ast
import json
import pathlib

import pytest

import halo.model_loading.parallel.model_source as model_source
import halo.runtime.consensus as consensus
from halo.checkpoint.format import SAFETENSORS_INDEX_FILE
from halo.model_loading.parallel.model_source import ModelSource, reject_ep_sharded_checkpoint, resolve_lazy_checkpoint
from tests.common.repo import REPO_ROOT
from tests.common.utils import imports_name

# The PP stage loader and the EP+TP loader, each a module that could hold a probe of its own, with the
# call through which each reaches the shared gate.
LOADER_MODULES = {
    "load_pp_stage_model": (REPO_ROOT / "halo/model_loading/parallel/pp.py", "resolve_lazy_checkpoint"),
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
    """The gathered-save alternative is the YAML setting a user writes, not an internal call."""
    with pytest.raises(ValueError) as err:
        reject_ep_sharded_checkpoint(_checkpoint(tmp_path, "ep_sharded"))
    assert "save_sharded_ep: false" in str(err.value), err.value


def test_a_plain_checkpoint_passes(tmp_path):
    assert reject_ep_sharded_checkpoint(_checkpoint(tmp_path, None)) is None


def test_only_rank0_touches_the_filesystem(tmp_path, monkeypatch):
    """The whole point of the gate: a non-zero rank must not run the probe at all, so a torn or
    half-fetched index on its node cannot raise it out of the collective its peers are in."""

    def _explode(*_args, **_kwargs):
        raise AssertionError("a non-zero rank probed the checkpoint itself")

    monkeypatch.setattr(consensus, "is_global_main_process", lambda: False)
    monkeypatch.setattr(model_source, "read_checkpoint_index", _explode)
    assert reject_ep_sharded_checkpoint(_checkpoint(tmp_path, "ep_sharded")) is None


def test_a_torn_index_on_rank0_is_deferred_past_the_collective(tmp_path, monkeypatch):
    """Rank 0's probe is a ``json.load`` sitting IN FRONT of the broadcast, so a torn or
    half-fetched index raising inside it strands every peer in that broadcast — the exact hang the
    gate exists to prevent, moved one line earlier. It has to be fenced: recorded locally, the
    broadcast still entered, and the real cause raised uniformly afterwards."""
    (tmp_path / SAFETENSORS_INDEX_FILE).write_text('{"metadata": {"format": "ep_sha')
    broadcast = consensus.broadcast_from_rank0
    entered: list = []
    monkeypatch.setattr(consensus, "broadcast_from_rank0", lambda v: entered.append(v) or broadcast(v))

    with pytest.raises(RuntimeError, match="JSONDecodeError"):
        reject_ep_sharded_checkpoint(ModelSource("checkpoints/run", None, str(tmp_path)))

    assert entered == [None], "the probe must still enter the broadcast, or every peer blocks in it"


def test_the_lazy_checkpoint_verdict_refuses_a_sharded_checkpoint(tmp_path):
    with pytest.raises(ValueError, match="merge_ep_shards.py"):
        resolve_lazy_checkpoint(_checkpoint(tmp_path, "ep_sharded"), lazy_requested=True, layout_supported=bool)


@pytest.mark.parametrize("loader", sorted(LOADER_MODULES))
def test_every_loader_that_can_be_handed_a_sharded_dir_runs_the_shared_gate(loader):
    """Drift pin: importing the index predicate is what a re-introduced per-rank probe would need."""
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
