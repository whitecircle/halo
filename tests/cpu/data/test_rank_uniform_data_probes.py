#!/usr/bin/env python
"""The data-loading probes each rank answers off its own slice are agreed on the world. Each branch runs coordinated
work, and the two arms of the modality dispatch run different counts of it, so a split verdict desynchronizes the
phase counters for good. A starved shard assignment still yields an empty split with the artifact's schema."""

import os
import shutil
import tempfile
from types import SimpleNamespace

import pytest
from datasets import Dataset, DatasetDict
from transformers.models.auto.configuration_auto import CONFIG_MAPPING

from halo.data.preprocessed.preparation import shard_dataset
from halo.data.preprocessed.shard_index import SHARD_INDEX_FILE, write_stamped_sidecar
from halo.data.sources.sharded_dataset import ShardedDatasetLoader
from halo.data.vlm_detection import dataset_declares_images_across_ranks, is_vlm_run_across_ranks
from halo.runtime import consensus
from halo.runtime.consensus import RankConsensus
from tests.common.repo import load_script_module

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

        def fake_rank_consensus(local_ok: bool) -> RankConsensus:
            self.calls += 1
            votes = (bool(local_ok), peer)
            return RankConsensus(all(votes), any(votes))

        monkeypatch.setattr(consensus, "rank_consensus", fake_rank_consensus)


def test_a_rank_whose_slice_shows_no_images_takes_the_peers_vlm_verdict(monkeypatch):
    """A text-only shard must not send its rank down the text pipeline while a peer holding an image shard goes down
    the VLM one."""
    world = _FakeWorld(monkeypatch, peer=True)

    assert is_vlm_run_across_ranks(_args(), "Qwen/Qwen3.5-9B", _dataset([TEXT_TURNS]), config=VLM_CONFIG)
    assert world.calls == 1


def test_the_agreed_verdict_never_promotes_a_text_only_checkpoint(monkeypatch):
    """Consensus decides the data half only, so a text checkpoint stays a text run even when a peer's slice declares
    images."""
    world = _FakeWorld(monkeypatch, peer=False)

    assert not is_vlm_run_across_ranks(
        _args(images_field="images"), "Qwen/Qwen3-8B", _dataset([IMAGE_TURNS]), config=TEXT_CONFIG
    )
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
    """``is_vlm_run_across_ranks`` short-circuits on a text config and on ``images_field``; a rank taking another
    branch would be one collective behind."""
    world = _FakeWorld(monkeypatch, peer=False)

    is_vlm_run_across_ranks(args, model, dataset, config=config)

    assert world.calls == 1, f"expected exactly one agreement collective, got {world.calls}"


def test_the_image_column_guard_probe_is_agreed_too(monkeypatch):
    """Read directly by KTO's embedded-image guard, whose raise sits in front of the coordinated column renames."""
    world = _FakeWorld(monkeypatch, peer=True)

    assert dataset_declares_images_across_ranks(_dataset([TEXT_TURNS]), "prompt")
    assert world.calls == 1


def test_an_empty_split_on_one_rank_raises_on_all_of_them(monkeypatch):
    """The gate sits in front of ``coordinated_pack``, and a rank-local raise strands the peers in the pack's
    barrier and store phases."""
    sft = load_script_module("scripts/training/sft.py")
    _FakeWorld(monkeypatch, peer=True)

    with pytest.raises(ValueError, match="at least one data-parallel rank"):
        sft._reject_empty_split(Dataset.from_dict({"input_ids": [[1, 2]]}), "training", 8)


def test_a_starved_rank_gets_the_splits_schema_not_a_column_less_dataset():
    """Every downstream ``column_names`` test gates a coordinated operation, so a column-less split makes the
    starved rank skip a barrier its peers make."""
    temp_dir = tempfile.mkdtemp()
    try:
        dataset = Dataset.from_dict({"prompt": [[{"role": "user", "content": "a"}]] * 4, "example_id": [0, 1, 2, 3]})
        index = shard_dataset(dataset, output_dir=temp_dir, split_name="test", num_shards=2)
        write_stamped_sidecar(os.path.join(temp_dir, "test", SHARD_INDEX_FILE), index)

        fed = ShardedDatasetLoader(temp_dir, dp_rank=0, dp_size=4).load_split("test")
        starved = ShardedDatasetLoader(temp_dir, dp_rank=3, dp_size=4).load_split("test")

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
