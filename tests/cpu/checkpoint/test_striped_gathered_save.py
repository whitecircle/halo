#!/usr/bin/env python
"""A gathered EP save striped over several writers reproduces the single-writer save tensor for tensor: the same keys,
values, dtypes, index size and config, written by the ranks that own the layers. Real gloo ranks, the EP layers stubbed
to a collective-shaped gather."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
from transformers import Qwen3Config, Qwen3ForCausalLM

import halo.checkpoint.writing as writing
from halo.checkpoint.format import SAFETENSORS_INDEX_FILE
from halo.checkpoint.parallel.ep import save_ep_gathered_checkpoint
from halo.checkpoint.writing import (
    StreamedParts,
    StreamingShardWriter,
    WriterStripe,
    merge_streamed_parts,
    resolve_writer_stripe,
)
from tests.common.checkpoint_io import file_digests, safetensors_state_dict, save_context
from tests.common.ep_stubs import StubEPLayerBase
from tests.common.gloo import run_gloo_ranks

WORLD_SIZE = 4
LAYERS = 5
# Small enough that a writer's layers span several parts.
SHARD_SIZE = "40KB"
WRITERS_ENV = "HALO_CHECKPOINT_WRITERS_PER_NODE"
SHARED_ENV = "DIST_OUTPUT_SHARED_FILESYSTEM"
# (ranks per node, writers per node) and the writers that save has.
STRIPES = {
    "reference": (4, 1, 1),
    "two_nodes": (2, 1, 2),
    "three_per_node": (4, 3, 3),
    "two_nodes_two_each": (2, 2, 4),
}


class _GatherStub(StubEPLayerBase):
    """A decoder MLP posing as an EP layer: its gather is a collective every rank joins, the copy kept by ``retain``."""

    def __init__(self, mlp: torch.nn.Module):
        super().__init__()
        for name, child in mlp.named_children():
            self.add_module(name, child)

    def gather_layer_state_dict(self, layer_name, *, merge_lora=False, retain=True):
        dist.all_reduce(torch.zeros(1))
        return {f"{layer_name}.{name}": p.detach().clone() for name, p in self.named_parameters()} if retain else {}


def _model() -> Qwen3ForCausalLM:
    torch.manual_seed(7)
    config = Qwen3Config(
        vocab_size=96,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=LAYERS,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=16,
        tie_word_embeddings=False,
    )
    model = Qwen3ForCausalLM(config)
    for layer in model.model.layers:
        layer.mlp = _GatherStub(layer.mlp)
    return model


def _staged_keys_by_rank(patch) -> list[str]:
    """What this rank's writer staged, so the test sees who wrote what and not only what landed."""
    staged: list[str] = []
    real_add = StreamingShardWriter.add

    def recording_add(self, key, tensor):
        if self.enabled:
            staged.append(key)
        return real_add(self, key, tensor)

    patch.setattr(StreamingShardWriter, "add", recording_add)
    return staged


def _striped_worker(rank: int, root: str) -> None:
    outcome: dict[str, list[str]] = {}
    for name, (local_world_size, per_node, _writers) in STRIPES.items():
        # The harness launches one node; the simulated layout re-derives each rank's place on its node.
        os.environ.update(
            LOCAL_WORLD_SIZE=str(local_world_size),
            LOCAL_RANK=str(rank % local_world_size),
            **{WRITERS_ENV: str(per_node)},
        )
        output = os.path.join(root, name)
        with pytest.MonkeyPatch.context() as patch:
            staged = _staged_keys_by_rank(patch)
            save_ep_gathered_checkpoint(save_context(_model(), has_ep_layers=True, max_shard_size=SHARD_SIZE), output)
        outcome[name] = staged
    Path(root, f"staged_{rank}.json").write_text(json.dumps(outcome))


@pytest.fixture(scope="module")
def striped_run(tmp_path_factory) -> Path:
    root = tmp_path_factory.mktemp("striped")
    for name in STRIPES:
        os.makedirs(root / name)
    run_gloo_ranks(_striped_worker, WORLD_SIZE, str(root))
    return root


def _staged(root: Path, name: str) -> dict[int, set[str]]:
    return {rank: set(json.loads(Path(root, f"staged_{rank}.json").read_text())[name]) for rank in range(WORLD_SIZE)}


def _weight_files(directory: Path) -> set[str]:
    return {p.name for p in directory.glob("*.safetensors")}


@pytest.mark.parametrize("name", [n for n in STRIPES if n != "reference"])
def test_a_striped_save_holds_the_single_writer_tensors_and_index(striped_run, name):
    reference, striped = striped_run / "reference", striped_run / name
    want, got = safetensors_state_dict(reference), safetensors_state_dict(striped)
    assert all(f"model.layers.{layer}.mlp.up_proj.weight" in want for layer in range(LAYERS)), (
        "premise: the reference holds every layer's gathered experts"
    )
    assert set(got) == set(want), f"key sets differ: {sorted(set(got) ^ set(want))[:6]}"
    for key, tensor in want.items():
        assert got[key].dtype == tensor.dtype and torch.equal(got[key], tensor), f"{key} differs"
    index_want = json.loads((reference / SAFETENSORS_INDEX_FILE).read_text())
    index_got = json.loads((striped / SAFETENSORS_INDEX_FILE).read_text())
    assert index_got["metadata"] == index_want["metadata"]
    assert set(index_got["weight_map"]) == set(index_want["weight_map"])
    # The index names exactly the files on disk, each under HF's name: no streaming part survives the rename.
    assert set(index_got["weight_map"].values()) == _weight_files(striped)
    assert all(file.startswith("model-0") and "-of-" in file for file in _weight_files(striped))
    assert file_digests(striped)["config.json"] == file_digests(reference)["config.json"]


@pytest.mark.parametrize("name", [n for n in STRIPES if n != "reference"])
def test_each_writer_stages_only_the_chunks_it_owns(striped_run, name):
    """Chunk 0 (everything outside the layers) is the first writer's; layer ``i`` is chunk ``i + 1``, round-robin."""
    local_world_size, per_node, count = STRIPES[name]
    staged = _staged(striped_run, name)
    writer_ranks = sorted(
        node * local_world_size + local for node in range(WORLD_SIZE // local_world_size) for local in range(per_node)
    )
    assert len(writer_ranks) == count
    for rank in range(WORLD_SIZE):
        if rank not in writer_ranks:
            assert not staged[rank], f"rank {rank} is no writer yet staged {sorted(staged[rank])[:3]}"
    for layer in range(LAYERS):
        owner = writer_ranks[layer % count]
        holders = [
            rank for rank in writer_ranks if any(k.startswith(f"model.layers.{layer}.mlp.") for k in staged[rank])
        ]
        assert holders == [owner], f"layer {layer}: written by {holders}, owned by rank {owner}"
    assert {"model.embed_tokens.weight", "lm_head.weight"} <= staged[0]
    assert any(len(staged[rank]) for rank in writer_ranks[1:]), "premise: the stripe moved work off rank 0"


def test_the_single_writer_reference_is_written_by_rank_zero_alone(striped_run):
    staged = _staged(striped_run, "reference")
    assert staged[0] and not any(staged[rank] for rank in range(1, WORLD_SIZE))


def test_a_key_two_writers_claim_is_refused():
    one = StreamedParts(("p0.safetensors",), {"w": "p0.safetensors"}, 8)
    two = StreamedParts(("q0.safetensors",), {"w": "q0.safetensors"}, 8)
    with pytest.raises(RuntimeError, match="disagree on 'w'"):
        merge_streamed_parts([one, None, two])


def test_the_merged_parts_keep_writer_order_and_skip_idle_writers():
    one = StreamedParts(("p0.safetensors", "p1.safetensors"), {"a": "p0.safetensors", "b": "p1.safetensors"}, 20)
    two = StreamedParts(("q0.safetensors",), {"c": "q0.safetensors"}, 5)
    merged = merge_streamed_parts([one, None, StreamedParts((), {}, 0), two])
    assert merged.part_names == ("p0.safetensors", "p1.safetensors", "q0.safetensors")
    assert merged.weight_map == {"a": "p0.safetensors", "b": "p1.safetensors", "c": "q0.safetensors"}
    assert merged.nbytes == 25


def test_ownership_covers_every_chunk_exactly_once():
    for count in (1, 2, 3, 8):
        stripes = [WriterStripe(index, count) for index in range(count)]
        for chunk in range(40):
            assert sum(s.owns(chunk) for s in stripes) == 1, (count, chunk)
        assert stripes[0].owns(0)
        assert not WriterStripe(None, count).owns(0)


@pytest.mark.parametrize(
    ("shared", "local_world_size", "rank", "per_node", "expected"),
    [
        # Shared: the save rank first, then node-major.
        (True, 8, 0, 1, WriterStripe(0, 4)),
        (True, 8, 8, 1, WriterStripe(1, 4)),
        (True, 8, 9, 1, WriterStripe(None, 4)),
        (True, 8, 17, 2, WriterStripe(5, 8)),
        (True, 8, 18, 2, WriterStripe(None, 8)),
        # A count past the node's ranks is capped at them.
        (True, 8, 3, 99, WriterStripe(3, 32)),
        # Per-node storage: each node's first rank writes the whole checkpoint.
        (False, 8, 8, 4, WriterStripe(0, 1)),
        (False, 8, 9, 4, WriterStripe(None, 1)),
    ],
)
def test_the_stripe_follows_the_filesystem_and_node_layout(
    monkeypatch, shared, local_world_size, rank, per_node, expected
):
    monkeypatch.setenv(WRITERS_ENV, str(per_node))
    monkeypatch.setenv(SHARED_ENV, "1" if shared else "0")
    monkeypatch.setattr(writing, "get_num_nodes", lambda: 4)
    monkeypatch.setattr(writing, "get_local_world_size", lambda: local_world_size)
    monkeypatch.setattr(writing, "get_node_rank", lambda: rank // local_world_size)
    monkeypatch.setattr(writing, "get_local_rank", lambda: rank % local_world_size)
    is_save_rank = rank == 0 if shared else rank % local_world_size == 0
    assert resolve_writer_stripe(is_save_rank=is_save_rank) == expected


def test_a_writer_count_below_one_is_refused(monkeypatch):
    monkeypatch.setenv(WRITERS_ENV, "0")
    monkeypatch.setenv(SHARED_ENV, "1")
    with pytest.raises(ValueError, match="HALO_CHECKPOINT_WRITERS_PER_NODE=0"):
        resolve_writer_stripe(is_save_rank=True)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
