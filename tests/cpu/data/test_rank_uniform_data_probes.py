#!/usr/bin/env python
"""CPU tests: the data-loading probes each rank answers off its OWN slice are agreed on the world.

Under a pre-sharded corpus every data-parallel rank holds a disjoint slice, so a probe of "does this
dataset declare images", "is this split empty", or "which splits did I get" is a per-rank fact. The
branch each one decides runs coordinated work — one ``ensure_cache_dir()`` barrier plus two
``dataset_op`` store phases per operation — and the two arms of the modality dispatch do not even run
the same NUMBER of them, so a split verdict pairs a barrier against a store wait and the phase
counters never realign. These tests pin the three properties that make that impossible:

- the verdict is routed through :func:`agree_probe_across_ranks` (a disagreeing rank adopts MAX);
- the number of collectives does not depend on the verdict or on which branch produced it;
- a starved shard assignment yields an empty split with the artifact's SCHEMA, so no downstream
  ``column_names`` test can answer differently on that rank.

Run: python tests/cpu/data/test_rank_uniform_data_probes.py  (or pytest)
"""

import os
import shutil
import tempfile
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
from datasets import Dataset, DatasetDict
from transformers.models.auto.configuration_auto import CONFIG_MAPPING

import src.data.probe_consensus as probe_consensus_mod
import src.data.sources.loading as loading_mod
from src.data.pipeline.preprocessing import shard_dataset
from src.data.shard_index import SHARD_INDEX_FILE
from src.data.sources.sharded_dataset import ShardedDatasetLoader
from src.data.vlm import dataset_declares_images, is_vlm_run
from tests.common.utils import load_script_module

VLM_CONFIG = CONFIG_MAPPING["gemma3"]()
TEXT_CONFIG = CONFIG_MAPPING["qwen3"]()

IMAGE_TURNS = [{"role": "user", "content": [{"type": "image", "image": "x"}, {"type": "text", "text": "hi"}]}]
TEXT_TURNS = [{"role": "user", "content": [{"type": "text", "text": "hi"}]}]


def _args(**overrides):
    return SimpleNamespace(**{"images_field": None, "conversation_field": "prompt", **overrides})


def _dataset(turns, extra=None):
    return DatasetDict({"train": Dataset.from_dict({"prompt": turns, **(extra or {})})})


class _FakeWorld:
    """A two-rank world whose PEER's probe answers ``peer``; counts every consensus collective."""

    def __init__(self, monkeypatch, peer: bool):
        self.calls = 0

        def fake_rank_consensus(local_ok: bool) -> tuple[bool, bool]:
            self.calls += 1
            votes = (bool(local_ok), peer)
            return all(votes), any(votes)

        monkeypatch.setattr(probe_consensus_mod, "rank_consensus", fake_rank_consensus)


def test_a_rank_whose_slice_shows_no_images_takes_the_peers_vlm_verdict(monkeypatch):
    """MAX consensus: a text-only shard must not send its rank down the text pipeline while a peer
    holding an image shard goes down the VLM one. Without the agreement the two arms run different
    numbers of coordinated operations and the store phases desynchronize permanently."""
    world = _FakeWorld(monkeypatch, peer=True)

    assert is_vlm_run(_args(), "Qwen/Qwen3.5-9B", _dataset([TEXT_TURNS]), config=VLM_CONFIG)
    assert world.calls == 1


def test_the_agreed_verdict_never_promotes_a_text_only_checkpoint(monkeypatch):
    """Anti-vacuity: consensus decides the DATA half only. A text checkpoint stays a text run even
    when a peer's slice declares images, because its local verdict is already False on both halves."""
    world = _FakeWorld(monkeypatch, peer=False)

    assert not is_vlm_run(_args(images_field="images"), "Qwen/Qwen3-8B", _dataset([IMAGE_TURNS]), config=TEXT_CONFIG)
    assert world.calls == 1


@pytest.mark.parametrize(
    ("args", "model", "config", "dataset"),
    [
        pytest.param(_args(), "Qwen/Qwen3-8B", TEXT_CONFIG, _dataset([TEXT_TURNS]), id="text-checkpoint"),
        pytest.param(
            _args(images_field="pictures"), "Qwen/Qwen3.5-9B", VLM_CONFIG, _dataset([TEXT_TURNS]), id="field"
        ),
        pytest.param(_args(), "Qwen/Qwen3.5-9B", VLM_CONFIG, _dataset([IMAGE_TURNS]), id="embedded"),
        pytest.param(_args(), "Qwen/Qwen3.5-9B", VLM_CONFIG, None, id="no-dataset"),
    ],
)
def test_the_dispatch_costs_exactly_one_collective_on_every_branch(monkeypatch, args, model, config, dataset):
    """The collective count must not depend on the verdict or on which term produced it.

    ``is_vlm_run`` short-circuits on a text config and on ``images_field``; agreeing only inside the
    dataset probe would leave those branches collective-free, and a rank taking a different branch
    (a hub config read that failed on one node) would then be one collective behind its peers.
    """
    world = _FakeWorld(monkeypatch, peer=False)

    is_vlm_run(args, model, dataset, config=config)

    assert world.calls == 1, f"expected exactly one agreement collective, got {world.calls}"


def test_the_image_column_guard_probe_is_agreed_too(monkeypatch):
    """``dataset_declares_images`` is read directly by KTO's embedded-image guard, whose raise sits
    in front of the coordinated column renames — so that entry point owes the same agreement."""
    world = _FakeWorld(monkeypatch, peer=True)

    assert dataset_declares_images(_dataset([TEXT_TURNS]), "prompt")
    assert world.calls == 1


def test_an_empty_split_on_one_rank_raises_on_all_of_them(monkeypatch):
    """SFT's post-tokenization emptiness gate is a per-rank length on a pre-sharded corpus, and it
    sits directly in front of ``pack_dataset_coordinated``. A rank-local raise there strands its
    peers in the pack's barrier and store phases; the verdict must be world-agreed."""
    sft = load_script_module("scripts/training/sft.py")
    _FakeWorld(monkeypatch, peer=True)

    with pytest.raises(ValueError, match="at least one data-parallel rank"):
        sft._reject_empty_split(Dataset.from_dict({"input_ids": [[1, 2]]}), "training", 8)


def test_a_split_only_some_ranks_loaded_is_rejected(monkeypatch):
    """A transient shard-index read failure drops a split on one rank alone. Every consumer of
    ``ds.get("test")`` then runs a different number of coordinated operations, so the divergence
    must raise — and raise on every rank, off the agreed MIN/MAX rather than the local shape."""

    def fake_all_reduce(tensor, op=None, **_kwargs):
        if op is dist.ReduceOp.MAX:
            tensor.fill_(1)  # some rank loaded both splits
        return None

    monkeypatch.setattr(loading_mod, "get_global_world_size", lambda: 2)
    monkeypatch.setattr(loading_mod, "current_device", lambda: torch.device("cpu"))
    monkeypatch.setattr(dist, "all_reduce", fake_all_reduce)

    only_train = DatasetDict({"train": Dataset.from_dict({"input_ids": [[1]]})})
    with pytest.raises(ValueError, match=r"loaded split\(s\) \['test'\]"):
        loading_mod._reject_divergent_split_presence(only_train, "s3://bucket/ds")


def test_a_starved_rank_gets_the_splits_schema_not_a_column_less_dataset():
    """The shard assignment leaves high ranks empty when a split has fewer shards than ranks. That
    empty split must still carry the artifact's columns: every downstream ``column_names`` test (the
    empty-conversation filter, the declared-render-column guard, the image probe) gates a coordinated
    operation, so a column-less split makes the starved rank skip a barrier its peers make."""
    temp_dir = tempfile.mkdtemp()
    try:
        dataset = Dataset.from_dict({"prompt": [[{"role": "user", "content": "a"}]] * 4, "example_id": [0, 1, 2, 3]})
        index = shard_dataset(dataset, output_dir=temp_dir, split_name="test", num_shards=2)
        index.save(os.path.join(temp_dir, "test", SHARD_INDEX_FILE))

        fed = ShardedDatasetLoader(temp_dir, global_rank=0, world_size=4).load_split("test")
        starved = ShardedDatasetLoader(temp_dir, global_rank=3, world_size=4).load_split("test")

        assert len(starved) == 0, "the starved rank holds no rows"
        assert starved.column_names == fed.column_names, (
            f"starved rank sees columns {starved.column_names}, its peers {fed.column_names} — every "
            f"column_names branch in front of a coordinated op would diverge"
        )
        assert starved.features == fed.features
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
