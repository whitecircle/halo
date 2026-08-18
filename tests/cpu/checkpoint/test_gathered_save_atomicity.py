#!/usr/bin/env python
"""A gathered save that dies mid-write must leave the previous checkpoint loadable.

``save_sharded_state_dict`` is the writer for EVERY gathered save (FSDP2, CP, gathered TP,
``write_gathered_checkpoint``). HF's splitter names shards deterministically from the model and
``max_shard_size``, so re-saving into a directory that already holds a same-shaped checkpoint reuses
the very filenames that are on disk: writing them directly and crashing after shard *k* leaves new
shards 1..k beside old shards k+1..N under the old — still valid, still same-named — index.
``from_pretrained`` reads that as a healthy checkpoint and loads half of each model.

The gathered writer therefore uses the same part+rename protocol as the EP/PP writers
(``StageShardWriter.close_as_hf_checkpoint`` writes numbered parts and ``os.replace``s them once the
count is known), so a crash before the renames costs nothing but stray parts, which the next
successful save sweeps.

    python tests/cpu/checkpoint/test_gathered_save_atomicity.py
"""

import glob
import json
import os

import pytest
import torch
from safetensors.torch import load_file

import src.checkpoint.format as checkpoint_format_mod
from src.checkpoint.format import (
    HF_STREAM_PART_PREFIX,
    SAFETENSORS_INDEX_FILE,
    save_sharded_state_dict,
)

# One tensor per shard at this cap: 512x512 fp32 = 1 MiB each, over the 1 MB (SI) limit.
SHARD_SIZE = "1MB"
TENSOR_SHAPE = (512, 512)


def _state(value: float, keys: int = 3) -> dict[str, torch.Tensor]:
    return {f"weight_{i}": torch.full(TENSOR_SHAPE, value) for i in range(keys)}


def _load_through_index(directory: str) -> dict[str, torch.Tensor]:
    """Read the directory the way ``from_pretrained`` does: index first, then the files it names."""
    with open(os.path.join(directory, SAFETENSORS_INDEX_FILE)) as handle:
        weight_map = json.load(handle)["weight_map"]
    return {key: load_file(os.path.join(directory, filename))[key] for key, filename in weight_map.items()}


def test_a_crash_mid_resave_leaves_the_previous_checkpoint_intact(tmp_path, monkeypatch):
    out = str(tmp_path)
    save_sharded_state_dict(_state(1.0), out, max_shard_size=SHARD_SIZE)
    previous = _load_through_index(out)
    assert len(set(previous)) == 3, "premise: the first save must have sharded, else there is no window"

    real_save_file = checkpoint_format_mod._safetensors_save_file
    written = []

    def crash_on_the_second_shard(tensors, path, **kwargs):
        written.append(path)
        if len(written) == 2:
            raise OSError(28, "No space left on device")
        return real_save_file(tensors, path, **kwargs)

    monkeypatch.setattr(checkpoint_format_mod, "_safetensors_save_file", crash_on_the_second_shard)
    with pytest.raises(OSError):
        save_sharded_state_dict(_state(2.0), out, max_shard_size=SHARD_SIZE)
    assert len(written) == 2, "premise: the crash must land after at least one shard was written"

    current = _load_through_index(out)
    assert set(current) == set(previous)
    for key, tensor in previous.items():
        assert torch.equal(current[key], tensor), (
            f"{key} came back from the half-written save — the directory now mixes two checkpoints "
            f"under an index that names them all"
        )


def test_the_next_successful_save_sweeps_the_abandoned_parts(tmp_path, monkeypatch):
    """The stray parts of a failed save are matched by the ``model*.safetensors`` stale sweep, so a
    successful re-save clears them; nothing accumulates across preemptions."""
    out = str(tmp_path)
    save_sharded_state_dict(_state(1.0), out, max_shard_size=SHARD_SIZE)

    real_save_file = checkpoint_format_mod._safetensors_save_file
    written = []

    def crash_on_the_second_shard(tensors, path, **kwargs):
        written.append(path)
        if len(written) == 2:
            raise OSError(28, "No space left on device")
        return real_save_file(tensors, path, **kwargs)

    monkeypatch.setattr(checkpoint_format_mod, "_safetensors_save_file", crash_on_the_second_shard)
    with pytest.raises(OSError):
        save_sharded_state_dict(_state(2.0), out, max_shard_size=SHARD_SIZE)
    assert glob.glob(os.path.join(out, f"{HF_STREAM_PART_PREFIX}-*.safetensors")), (
        "premise: the failed save must have left its in-flight part behind"
    )

    monkeypatch.setattr(checkpoint_format_mod, "_safetensors_save_file", real_save_file)
    save_sharded_state_dict(_state(3.0), out, max_shard_size=SHARD_SIZE)

    assert not glob.glob(os.path.join(out, f"{HF_STREAM_PART_PREFIX}-*.safetensors"))
    for tensor in _load_through_index(out).values():
        assert torch.equal(tensor, torch.full(TENSOR_SHAPE, 3.0))


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
