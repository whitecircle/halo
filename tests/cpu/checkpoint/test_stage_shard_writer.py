#!/usr/bin/env python
"""The pipeline stage writer must bound its own memory, not just split its output.

safetensors has no append handle, so honouring ``save_max_shard_size`` and bounding the writer's
host memory are one requirement: flush a part whenever the pending batch would overflow. A PP save
that accumulates the whole stage in a dict before a single ``save_file`` peaks at
``model_bytes / pp_size``, and on a single node the writers-per-node count cancels that divisor, so
the entire model lands in one host's RAM regardless of ``pp_size``.

The weakref assertion below is the honest one: an implementation that merely splits its OUTPUT but
still holds every tensor would pass a size check and fail this.

Run: pytest tests/cpu/checkpoint/test_stage_shard_writer.py
"""

import json
import os
import re
import weakref

import pytest
import torch
from safetensors.torch import safe_open

from src.checkpoint.format import save_sharded_state_dict, write_merged_index
from src.checkpoint.shard_writer import StageShardWriter
from src.distributed.checkpoint.write import merge_shard_index

PREFIX = "model-pp00000-of-00002"


def _tensor(mb: int) -> torch.Tensor:
    """A float32 tensor of roughly ``mb`` megabytes (1 MB = 250_000 float32 elements)."""
    return torch.zeros(250_000 * mb, dtype=torch.float32)


def test_writer_releases_each_tensor_once_its_part_is_flushed(tmp_path):
    """Peak memory must track the shard limit, not the total written.

    Every tensor handed in is dropped once the part containing it reaches disk, so the writer never
    holds more than one part's worth. An accumulating writer keeps all 20 alive to the end.
    """
    writer = StageShardWriter(str(tmp_path), PREFIX, "4MB", enabled=True)
    refs = []
    for i in range(20):
        tensor = _tensor(1)
        refs.append(weakref.ref(tensor))
        writer.add(f"w{i}", tensor)
        del tensor
    alive_before_close = sum(ref() is not None for ref in refs)
    assert alive_before_close <= 4, f"{alive_before_close} tensors still held; the writer is accumulating"

    writer.close()
    assert all(ref() is None for ref in refs), "close() must not retain the final part"


def test_parts_respect_the_size_limit_and_the_index_names_every_file(tmp_path):
    """Each part stays under the limit, and the returned weight map matches what is on disk."""
    writer = StageShardWriter(str(tmp_path), PREFIX, "4MB", enabled=True)
    for i in range(20):
        writer.add(f"w{i}", _tensor(1))
    weight_map, total_bytes = writer.close()

    files = sorted(p for p in os.listdir(tmp_path) if p.endswith(".safetensors"))
    assert len(files) > 1, "20 MB at a 4 MB limit must not land in one file"
    assert set(weight_map.values()) == set(files), "the weight map must name exactly the files written"
    assert set(weight_map) == {f"w{i}" for i in range(20)}
    # Reported bytes are the tensor payload (HF's total_size), not file size: parts carry a header.
    on_disk_payload = sum(t.numel() * t.element_size() for name in files for t in _tensors(tmp_path / name).values())
    assert total_bytes == on_disk_payload
    for name in files:
        payload = sum(t.numel() * t.element_size() for t in _tensors(tmp_path / name).values())
        assert payload <= 4_000_000, f"{name} exceeds the 4MB limit"


def test_a_single_oversized_tensor_still_gets_its_own_part(tmp_path):
    """A tensor larger than the limit cannot be split, so it becomes one over-limit part alone.

    Flushing BEFORE appending is what keeps it from dragging unrelated tensors over the limit
    with it.
    """
    writer = StageShardWriter(str(tmp_path), PREFIX, "2MB", enabled=True)
    writer.add("small", _tensor(1))
    writer.add("huge", _tensor(8))
    writer.add("after", _tensor(1))
    weight_map, _ = writer.close()

    assert weight_map["huge"] != weight_map["small"], "the oversized tensor must not share a part"
    assert weight_map["huge"] != weight_map["after"]
    assert len(_tensors(tmp_path / weight_map["huge"])) == 1


def test_disabled_writer_writes_nothing(tmp_path):
    """Non-writer ranks share the call sites; they must produce no files and an empty fragment."""
    writer = StageShardWriter(str(tmp_path), PREFIX, "4MB", enabled=False)
    writer.add("w", _tensor(1))
    assert writer.close() == ({}, 0)
    assert not [p for p in os.listdir(tmp_path) if p.endswith(".safetensors")]


@pytest.mark.parametrize("flush_between", [False, True], ids=["same-part", "already-flushed"])
def test_the_writer_refuses_a_key_staged_twice(tmp_path, flush_between):
    """A streaming writer cannot resolve a second claim on a key the way a state-dict build does.

    Once the first tensor is in a flushed part the checkpoint would carry the key twice at two
    shapes — readable through the index, wrong to every tool that walks the shard FILES — and
    ``total_size`` would count both. The pending batch is refused the same way: silently overwriting
    there would leave the byte count of the dropped tensor in the part's budget.
    """
    writer = StageShardWriter(str(tmp_path), PREFIX, "4MB", enabled=True)
    writer.add("w", _tensor(1))
    if flush_between:
        writer._flush()

    with pytest.raises(RuntimeError, match="staged twice"):
        writer.add("w", _tensor(1))


def test_index_merge_rejects_a_key_two_stages_both_claim():
    """A global parameter lives on exactly one stage.

    Two claims resolving last-writer-wins double-count ``total_size`` and yield a checkpoint that
    loads fine while holding the wrong stage's tensor.
    """
    stage0 = ({"model.layers.0.weight": "model-pp00000-of-00002-00001.safetensors"}, 100)
    stage1 = ({"model.layers.0.weight": "model-pp00001-of-00002-00001.safetensors"}, 100)

    with pytest.raises(RuntimeError, match="claimed by both"):
        merge_shard_index([stage0, stage1])


def test_index_merge_counts_each_file_once_across_node_writers():
    """On per-node storage every node of a stage writes the SAME filenames, so bytes must not re-fold.

    ``is_pp_shard_writer`` widens to one writer per NODE when the filesystem is not shared, and each
    of them emits its stage's shard under the identical ``model-pp{rank}-of-{size}`` name. Folding
    every fragment's bytes reported ``total_size`` multiplied by the nodes per stage — 8x on a
    64-node job at pp8 — in an index that otherwise round-trips.
    """
    fragment = ({"model.layers.0.weight": "model-pp00000-of-00002-00001.safetensors"}, 100)
    other_stage = ({"model.layers.9.weight": "model-pp00001-of-00002-00001.safetensors"}, 250)

    # Two node-writers of stage 0 plus two of stage 1, the shape a 2-node-per-stage job produces.
    weight_map, total = merge_shard_index([fragment, fragment, other_stage, other_stage])

    assert total == 350, "each distinct shard file contributes its bytes exactly once"
    assert len(weight_map) == 2


def test_index_merge_rejects_a_replicated_key_two_ep_ranks_claim():
    """The EP sharded save writes every replicated key from rank 0 alone (experts carry a per-rank
    ``.shard_N`` suffix), so two ranks claiming one name means a replica was written per-rank — the
    merged index would silently name one rank's copy while counting the bytes twice."""
    ep0 = ({"model.norm.weight": "model-00000-of-00002.safetensors"}, 100)
    ep1 = ({"model.norm.weight": "model-00001-of-00002.safetensors"}, 100)

    with pytest.raises(RuntimeError, match="claimed by both"):
        merge_shard_index([ep0, ep1])


def test_index_merge_sums_disjoint_stages():
    """The non-colliding case must still fold cleanly, including a non-writer's ``None``."""
    stage0 = ({"a": "f0.safetensors"}, 100)
    stage1 = ({"b": "f1.safetensors"}, 250)

    weight_map, total = merge_shard_index([stage0, None, stage1, None])

    assert weight_map == {"a": "f0.safetensors", "b": "f1.safetensors"}
    assert total == 350


def _tensors(path) -> dict[str, torch.Tensor]:
    with safe_open(path, framework="pt") as f:
        # safe_open exposes .keys() but is not iterable, so SIM118's rewrite does not apply here.
        return {k: f.get_tensor(k) for k in f.keys()}  # noqa: SIM118


def test_written_parts_round_trip(tmp_path):
    """Values must survive the split unchanged — the point of the writer is where bytes land."""
    writer = StageShardWriter(str(tmp_path), PREFIX, "4MB", enabled=True)
    expected = {f"w{i}": torch.randn(1000) for i in range(10)}
    for key, tensor in expected.items():
        writer.add(key, tensor.clone())
    weight_map, _ = writer.close()

    for key, tensor in expected.items():
        stored = _tensors(tmp_path / weight_map[key])[key]
        assert torch.equal(stored, tensor)


def _stream(dir_path, state_dict, max_shard_size):
    dir_path.mkdir()
    writer = StageShardWriter(str(dir_path), "model-streaming", max_shard_size, enabled=True)
    for key, tensor in state_dict.items():
        writer.add(key, tensor)
    writer.close_as_hf_checkpoint()
    return json.loads((dir_path / "model.safetensors.index.json").read_text())


def test_hf_finalize_is_readable_exactly_like_the_buffered_splitter(tmp_path):
    """The streamed save must be READABLE like ``save_sharded_state_dict``'s output — not identical.

    That helper is the one the gathered EP path calls after buffering the whole model in host
    RAM. What a reader needs is the HF naming, an index naming every key, the same total_size, and
    the same bytes reachable through it. Part BOUNDARIES are deliberately not guaranteed — see the
    next test — so asserting the same filenames or key→file map would pin a property the writer
    explicitly disclaims.
    """
    state_dict = {f"w{i}": torch.randn(250_000) for i in range(20)}  # 1 MB each, none over the cap

    buffered_dir = tmp_path / "buffered"
    buffered_dir.mkdir()
    save_sharded_state_dict(dict(state_dict), str(buffered_dir), max_shard_size="4MB")
    buffered_index = json.loads((buffered_dir / "model.safetensors.index.json").read_text())

    streamed_index = _stream(tmp_path / "streamed", state_dict, "4MB")

    assert set(streamed_index["weight_map"]) == set(buffered_index["weight_map"])
    assert streamed_index["metadata"]["total_size"] == buffered_index["metadata"]["total_size"]
    for name in streamed_index["weight_map"].values():
        assert re.fullmatch(r"model-\d{5}-of-\d{5}\.safetensors", name), f"not an HF shard name: {name}"
    for key, tensor in state_dict.items():
        assert torch.equal(_tensors(tmp_path / "streamed" / streamed_index["weight_map"][key])[key], tensor)


def test_an_oversized_tensor_diverges_from_hf_but_still_round_trips(tmp_path):
    """The one documented divergence, pinned so nobody "fixes" it back into a full-block peak.

    HF gives a tensor over the limit its own shard and leaves the current block OPEN, continuing to
    fill it; the writer flushes that block first. Keeping it open would hold a full block alongside
    an oversized tensor — the peak this writer exists to bound. So the two produce different part
    counts here, and both are correct. The fixture in the test above (uniform, all under the cap) is
    exactly the family where this cannot show up.
    """
    state_dict = {
        "small_a": torch.randn(250_000),
        "big": torch.randn(2_000_000),  # 8 MB — over the 4 MB cap
        "small_b": torch.randn(250_000),
    }

    buffered_dir = tmp_path / "buffered"
    buffered_dir.mkdir()
    save_sharded_state_dict(dict(state_dict), str(buffered_dir), max_shard_size="4MB")
    buffered_index = json.loads((buffered_dir / "model.safetensors.index.json").read_text())

    streamed_index = _stream(tmp_path / "streamed", state_dict, "4MB")

    streamed_parts = set(streamed_index["weight_map"].values())
    buffered_parts = set(buffered_index["weight_map"].values())
    assert len(streamed_parts) == 3, "the writer should flush, isolate the oversized tensor, then reopen"
    assert len(buffered_parts) == 2, "HF keeps its block open across the oversized tensor"

    assert set(streamed_index["weight_map"]) == set(buffered_index["weight_map"])
    assert streamed_index["metadata"]["total_size"] == buffered_index["metadata"]["total_size"]
    for key, tensor in state_dict.items():
        assert torch.equal(_tensors(tmp_path / "streamed" / streamed_index["weight_map"][key])[key], tensor)


def test_hf_finalize_writes_one_unsharded_file_and_no_index(tmp_path):
    """A checkpoint that fits one part is ``model.safetensors`` with no index, exactly as HF writes it.

    ``from_pretrained`` prefers that file over an index, so emitting a numbered part plus an index
    here would still load — but a later single-file save into the same directory would then be
    shadowed by this run's stale index.
    """
    writer = StageShardWriter(str(tmp_path), "model-streaming", "100MB", enabled=True)
    writer.add("w", _tensor(1))
    total = writer.close_as_hf_checkpoint()

    assert sorted(os.listdir(tmp_path)) == ["model.safetensors"]
    assert total == 1_000_000
    assert set(_tensors(tmp_path / "model.safetensors")) == {"w"}


def test_hf_finalize_removes_a_leftover_checkpoint_it_did_not_write(tmp_path):
    """A previous save's files must not survive into this one.

    ``from_pretrained`` prefers a single ``model.safetensors`` over any index, so a leftover from an
    earlier (smaller) save would be loaded INSTEAD of the sharded checkpoint just written — silently
    restoring stale weights. The streaming writer inherits the sweep HF's splitter does.
    """
    (tmp_path / "model.safetensors").write_bytes(b"stale")
    (tmp_path / "model-00001-of-00009.safetensors").write_bytes(b"stale from a 9-shard save")

    writer = StageShardWriter(str(tmp_path), "model-streaming", "4MB", enabled=True)
    for i in range(20):
        writer.add(f"w{i}", _tensor(1))
    writer.close_as_hf_checkpoint()

    remaining = sorted(os.listdir(tmp_path))
    assert "model.safetensors" not in remaining, "the stale single-file checkpoint must be deleted"
    assert "model-00001-of-00009.safetensors" not in remaining, "a stale shard count must not survive"
    index = json.loads((tmp_path / "model.safetensors.index.json").read_text())
    assert set(index["weight_map"].values()) == {n for n in remaining if n.endswith(".safetensors")}


def test_hf_finalize_refuses_to_replace_a_checkpoint_with_nothing(tmp_path):
    """An enabled writer that staged no tensors must raise, not sweep the directory clean.

    The stale sweep exists to stop a previous save from shadowing this one; with zero parts it would
    delete the previous checkpoint and write no index or weights in its place — turning a caller-side
    gather bug into the loss of the last good checkpoint.
    """
    (tmp_path / "model.safetensors").write_bytes(b"the previous save")
    writer = StageShardWriter(str(tmp_path), "model-streaming", "4MB", enabled=True)

    with pytest.raises(RuntimeError, match="empty checkpoint"):
        writer.close_as_hf_checkpoint()
    assert os.listdir(tmp_path) == ["model.safetensors"], "the previous checkpoint must survive"


def test_hf_finalize_on_a_disabled_writer_touches_nothing(tmp_path):
    """Non-writer ranks reach the same call site and must not delete the writer rank's output."""
    (tmp_path / "model.safetensors").write_bytes(b"written by the save rank")
    writer = StageShardWriter(str(tmp_path), "model-streaming", "4MB", enabled=False)
    writer.add("w", _tensor(1))

    assert writer.close_as_hf_checkpoint() == 0
    assert os.listdir(tmp_path) == ["model.safetensors"]


def test_the_index_writer_refuses_a_metadata_block_that_is_not_a_mapping(tmp_path):
    """``metadata`` is written into the index verbatim, so a scalar — this argument's previous shape
    was a bare ``total_size`` int — produces ``{"metadata": 12345}``: an index every reader parses
    without complaint and no reader can use."""
    with pytest.raises(TypeError, match="must be a mapping"):
        write_merged_index(str(tmp_path), {"w": "model-00001-of-00002.safetensors"}, 12345)
    assert not os.listdir(tmp_path), "a refused index must not have swept the directory first"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
