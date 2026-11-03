#!/usr/bin/env python
"""safetensors has no append handle, so honouring ``save_max_shard_size`` and bounding host memory are one requirement.
Accumulating a stage before ``save_file`` lands the whole model in one host's RAM on a single node."""

import json
import os
import re
import weakref

import pytest
import torch
from huggingface_hub import split_torch_state_dict_into_shards
from safetensors.torch import safe_open

import halo.checkpoint.writing as checkpoint_writing
from halo.checkpoint.format import STREAM_PART_PREFIX
from halo.checkpoint.writing import StreamingShardWriter, merge_shard_index, write_merged_index

PREFIX = "model-pp00000-of-00002"


def _tensor(mb: int) -> torch.Tensor:
    """1 MB = 250_000 float32 elements."""
    return torch.zeros(250_000 * mb, dtype=torch.float32)


def test_writer_releases_each_tensor_once_its_part_is_flushed(tmp_path):
    """An accumulating writer keeps all 20 tensors alive to the end, which a size check would miss."""
    writer = StreamingShardWriter(str(tmp_path), "4MB", enabled=True, filename_prefix=PREFIX)
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
    writer = StreamingShardWriter(str(tmp_path), "4MB", enabled=True, filename_prefix=PREFIX)
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
    """Flushing before appending keeps it from dragging unrelated tensors over the limit."""
    writer = StreamingShardWriter(str(tmp_path), "2MB", enabled=True, filename_prefix=PREFIX)
    writer.add("small", _tensor(1))
    writer.add("huge", _tensor(8))
    writer.add("after", _tensor(1))
    weight_map, _ = writer.close()

    assert weight_map["huge"] != weight_map["small"], "the oversized tensor must not share a part"
    assert weight_map["huge"] != weight_map["after"]
    assert len(_tensors(tmp_path / weight_map["huge"])) == 1


def test_disabled_writer_writes_nothing(tmp_path):
    """Non-writer ranks share the call sites."""
    writer = StreamingShardWriter(str(tmp_path), "4MB", enabled=False, filename_prefix=PREFIX)
    writer.add("w", _tensor(1))
    assert writer.close() == ({}, 0)
    assert not [p for p in os.listdir(tmp_path) if p.endswith(".safetensors")]


@pytest.mark.parametrize("flush_between", [False, True], ids=["same-part", "already-flushed"])
def test_the_writer_refuses_a_key_staged_twice(tmp_path, flush_between):
    """The checkpoint would carry the key twice at two shapes, wrong to every tool that walks the shard files, and
    ``total_size`` would count both."""
    writer = StreamingShardWriter(str(tmp_path), "4MB", enabled=True, filename_prefix=PREFIX)
    writer.add("w", _tensor(1))
    if flush_between:
        writer._flush()

    with pytest.raises(RuntimeError, match="staged twice"):
        writer.add("w", _tensor(1))


def test_index_merge_rejects_a_key_two_stages_both_claim():
    """Last-writer-wins double-counts ``total_size`` and loads the wrong stage's tensor without complaint."""
    stage0 = ({"model.layers.0.weight": "model-pp00000-of-00002-00001.safetensors"}, 100)
    stage1 = ({"model.layers.0.weight": "model-pp00001-of-00002-00001.safetensors"}, 100)

    with pytest.raises(RuntimeError, match="claimed by both"):
        merge_shard_index([stage0, stage1])


def test_index_merge_counts_each_file_once_across_node_writers():
    """On per-node storage every node writes the same filenames; folding all bytes reports ``total_size`` 8x on a
    64-node job at pp8."""
    fragment = ({"model.layers.0.weight": "model-pp00000-of-00002-00001.safetensors"}, 100)
    other_stage = ({"model.layers.9.weight": "model-pp00001-of-00002-00001.safetensors"}, 250)

    weight_map, total = merge_shard_index([fragment, fragment, other_stage, other_stage])

    assert total == 350, "each distinct shard file contributes its bytes exactly once"
    assert len(weight_map) == 2


def test_index_merge_rejects_a_replicated_key_two_ep_ranks_claim():
    """Experts carry a per-rank suffix, so two claims mean a replica written per-rank: one copy named, bytes counted
    twice."""
    ep0 = ({"model.norm.weight": "model-00000-of-00002.safetensors"}, 100)
    ep1 = ({"model.norm.weight": "model-00001-of-00002.safetensors"}, 100)

    with pytest.raises(RuntimeError, match="claimed by both"):
        merge_shard_index([ep0, ep1])


def test_index_merge_sums_disjoint_stages():
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
    writer = StreamingShardWriter(str(tmp_path), "4MB", enabled=True, filename_prefix=PREFIX)
    expected = {f"w{i}": torch.randn(1000) for i in range(10)}
    for key, tensor in expected.items():
        writer.add(key, tensor.clone())
    weight_map, _ = writer.close()

    for key, tensor in expected.items():
        stored = _tensors(tmp_path / weight_map[key])[key]
        assert torch.equal(stored, tensor)


def _stream(dir_path, state_dict, max_shard_size):
    dir_path.mkdir(exist_ok=True)
    writer = StreamingShardWriter(str(dir_path), max_shard_size, enabled=True)
    for key, tensor in state_dict.items():
        writer.add(key, tensor)
    writer.close_as_hf_checkpoint()
    return json.loads((dir_path / "model.safetensors.index.json").read_text())


def test_hf_finalize_is_readable_exactly_like_hfs_splitter(tmp_path):
    """Part boundaries are deliberately not guaranteed, so this asserts naming, index, total_size and bytes only."""
    state_dict = {f"w{i}": torch.randn(250_000) for i in range(20)}
    split = split_torch_state_dict_into_shards(state_dict, max_shard_size="4MB")

    streamed_index = _stream(tmp_path / "streamed", state_dict, "4MB")

    assert set(streamed_index["weight_map"]) == set(split.tensor_to_filename)
    assert streamed_index["metadata"]["total_size"] == split.metadata["total_size"]
    for name in streamed_index["weight_map"].values():
        assert re.fullmatch(r"model-\d{5}-of-\d{5}\.safetensors", name), f"not an HF shard name: {name}"
    for key, tensor in state_dict.items():
        assert torch.equal(_tensors(tmp_path / "streamed" / streamed_index["weight_map"][key])[key], tensor)


def test_an_oversized_tensor_diverges_from_hf_but_still_round_trips(tmp_path):
    """The one documented divergence, pinned so nobody restores a full-block peak: this writer flushes before an
    oversized tensor, HF keeps its block open."""
    state_dict = {
        "small_a": torch.randn(250_000),
        "big": torch.randn(2_000_000),
        "small_b": torch.randn(250_000),
    }
    split = split_torch_state_dict_into_shards(state_dict, max_shard_size="4MB")

    streamed_index = _stream(tmp_path / "streamed", state_dict, "4MB")

    assert len(set(streamed_index["weight_map"].values())) == 3, "the writer should flush, isolate, then reopen"
    assert len(split.filename_to_tensors) == 2, "HF keeps its block open across the oversized tensor"
    assert streamed_index["metadata"]["total_size"] == split.metadata["total_size"]
    for key, tensor in state_dict.items():
        assert torch.equal(_tensors(tmp_path / "streamed" / streamed_index["weight_map"][key])[key], tensor)


def test_hf_finalize_writes_one_unsharded_file_and_no_index(tmp_path):
    """A numbered part plus an index would load, but would later shadow a single-file save into the same directory."""
    writer = StreamingShardWriter(str(tmp_path), "100MB", enabled=True)
    writer.add("w", _tensor(1))
    total = writer.close_as_hf_checkpoint()

    assert sorted(os.listdir(tmp_path)) == ["model.safetensors"]
    assert total == 1_000_000
    assert set(_tensors(tmp_path / "model.safetensors")) == {"w"}


def test_hf_finalize_removes_a_leftover_checkpoint_it_did_not_write(tmp_path):
    """``from_pretrained`` prefers a single ``model.safetensors`` over any index, silently restoring stale weights."""
    (tmp_path / "model.safetensors").write_bytes(b"stale")
    (tmp_path / "model-00001-of-00009.safetensors").write_bytes(b"stale from a 9-shard save")

    writer = StreamingShardWriter(str(tmp_path), "4MB", enabled=True)
    for i in range(20):
        writer.add(f"w{i}", _tensor(1))
    writer.close_as_hf_checkpoint()

    remaining = sorted(os.listdir(tmp_path))
    assert "model.safetensors" not in remaining, "the stale single-file checkpoint must be deleted"
    assert "model-00001-of-00009.safetensors" not in remaining, "a stale shard count must not survive"
    index = json.loads((tmp_path / "model.safetensors.index.json").read_text())
    assert set(index["weight_map"].values()) == {n for n in remaining if n.endswith(".safetensors")}


def test_an_abandoned_save_never_passes_for_a_checkpoint_and_the_next_finalize_sweeps_it(tmp_path):
    """The prefix stays outside HF's shard names and inside the ``model*`` sweep."""
    abandoned = StreamingShardWriter(str(tmp_path), "4MB", enabled=True)
    for i in range(6):
        abandoned.add(f"w{i}", _tensor(1))
    abandoned._flush()
    leftovers = os.listdir(tmp_path)
    assert len(leftovers) > 1, "premise: the interrupted save left several parts behind"
    assert all(name.startswith(f"{STREAM_PART_PREFIX}-") for name in leftovers)
    assert not any(re.fullmatch(r"model(-\d{5}-of-\d{5})?\.safetensors", name) for name in leftovers)

    writer = StreamingShardWriter(str(tmp_path), "4MB", enabled=True)
    writer.add("w", _tensor(1))
    writer.close_as_hf_checkpoint()

    assert os.listdir(tmp_path) == ["model.safetensors"]


def test_a_crash_mid_resave_leaves_the_previous_checkpoint_intact(tmp_path, monkeypatch):
    """Otherwise a re-save dying after part *k* leaves new and old parts that ``from_pretrained`` reads as one
    healthy checkpoint."""
    out = tmp_path / "out"
    previous = _stream(out, {f"w{i}": torch.full((250_000,), 1.0) for i in range(3)}, "1MB")
    real_save = checkpoint_writing.save_safetensors_file
    written = []

    def crash_on_the_second_part(tensors, path, **kwargs):
        written.append(path)
        if len(written) == 2:
            raise OSError(28, "No space left on device")
        return real_save(tensors, path, **kwargs)

    monkeypatch.setattr(checkpoint_writing, "save_safetensors_file", crash_on_the_second_part)
    with pytest.raises(OSError):
        _stream(out, {f"w{i}": torch.full((250_000,), 2.0) for i in range(3)}, "1MB")
    assert len(written) == 2, "premise: the crash lands after one part of the re-save was written"

    assert json.loads((out / "model.safetensors.index.json").read_text()) == previous
    for key, part in previous["weight_map"].items():
        assert torch.equal(_tensors(out / part)[key], torch.full((250_000,), 1.0)), key


def test_hf_finalize_refuses_to_replace_a_checkpoint_with_nothing(tmp_path):
    """With zero parts the stale sweep turns a caller-side gather bug into the loss of the last good checkpoint."""
    (tmp_path / "model.safetensors").write_bytes(b"the previous save")
    writer = StreamingShardWriter(str(tmp_path), "4MB", enabled=True)

    with pytest.raises(RuntimeError, match="empty checkpoint"):
        writer.close_as_hf_checkpoint()
    assert os.listdir(tmp_path) == ["model.safetensors"], "the previous checkpoint must survive"


def test_hf_finalize_on_a_disabled_writer_touches_nothing(tmp_path):
    (tmp_path / "model.safetensors").write_bytes(b"written by the save rank")
    writer = StreamingShardWriter(str(tmp_path), "4MB", enabled=False)
    writer.add("w", _tensor(1))

    assert writer.close_as_hf_checkpoint() == 0
    assert os.listdir(tmp_path) == ["model.safetensors"]


def test_the_index_writer_refuses_a_metadata_block_that_is_not_a_mapping(tmp_path):
    """``{"metadata": 12345}`` is an index every reader parses and no reader can use."""
    with pytest.raises(TypeError, match="must be a mapping"):
        write_merged_index(str(tmp_path), {"w": "model-00001-of-00002.safetensors"}, 12345)
    assert not os.listdir(tmp_path), "a refused index must not have swept the directory first"


def test_written_files_claim_each_key_once(tmp_path):
    writer = StreamingShardWriter(str(tmp_path), "4MB", enabled=True, filename_prefix=PREFIX)
    writer.add("staged", _tensor(1))
    writer.add_written_files({"table.0": "model-table00-00001-of-00001.safetensors"}, 0)
    with pytest.raises(RuntimeError, match="written twice"):
        writer.add_written_files({"staged": "other.safetensors"}, 0)
    with pytest.raises(RuntimeError, match="written twice"):
        writer.add_written_files({"table.0": "other.safetensors"}, 0)
    with pytest.raises(RuntimeError, match="staged twice"):
        writer.add("table.0", _tensor(1))


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
