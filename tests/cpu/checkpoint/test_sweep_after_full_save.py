#!/usr/bin/env python
"""``sweep_after_full_save``: a COMPLETED save owns the output directory, a crashed one owns nothing.

Every conversion tool re-saves into a directory that may already hold a previous run's weights, and
``save_pretrained`` only deletes the shards its own ``-00001-of-00002`` numbering regex matches — a
leftover single ``model.safetensors`` or a stale index outlives the save. ``from_pretrained`` then
prefers that single file over the index just written, and the toolkit's index-first readers follow a
stale index to shards that are gone: either way the artifact served is not the one that was saved.

The sweep therefore runs AFTER the save and derives its keep-set from what is on disk NOW. That
ordering is the whole safety property: a save that died mid-write leaves neither a consistent index
nor a ``model.safetensors``, the keep-set comes out empty, and the sweep must then delete NOTHING —
sweeping before the save (or on an empty keep-set) would let a failed conversion destroy whatever
good checkpoint the output directory previously held.

When BOTH layouts are on disk and both are self-consistent, one of them is a previous run's
leftover and only the modification time says which. A writer that clears numbered shards itself
(``save_pretrained``) can leave a stale single file; a writer that clears nothing (``reset_sinks``'
safetensors branch) leaves a stale index AND its shards beside the one file it just wrote. Both
directions are pinned below: guessing wrong deletes the checkpoint that was just written and serves
the previous run's weights out of a directory the tool reported as saved.

    python tests/cpu/checkpoint/test_sweep_after_full_save.py
"""

import json
import os
import time
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

import src.checkpoint.format as checkpoint_format_mod
from src.checkpoint.format import sweep_after_full_save, write_gathered_checkpoint

INDEX = "model.safetensors.index.json"
SINGLE = "model.safetensors"


def _plant(directory, *names, index: dict | None = None, age: float = 0.0) -> None:
    """Materialise a directory state: ``names`` as files, ``index`` as the weight map of an index.

    ``age`` back-dates everything written by this call, so a test can state which write is the
    previous run's instead of relying on the resolution of two writes microseconds apart.
    """
    os.makedirs(directory, exist_ok=True)
    written = list(names)
    for name in names:
        with open(os.path.join(directory, name), "wb") as handle:
            handle.write(name.encode())
    if index is not None:
        with open(os.path.join(directory, INDEX), "w") as handle:
            json.dump({"metadata": {}, "weight_map": index}, handle)
        written.append(INDEX)
    if age:
        stamp = time.time() - age
        for name in written:
            os.utime(os.path.join(directory, name), (stamp, stamp))


def _files(directory) -> set[str]:
    return set(os.listdir(directory))


def test_a_sharded_save_sweeps_the_previous_runs_single_file(tmp_path):
    """The dangerous leftover: ``from_pretrained`` prefers ``model.safetensors`` over the index, so a
    surviving single file means the OLD weights win the lookup over the shards just written."""
    out = tmp_path / "out"
    _plant(out, SINGLE, age=10)  # the previous run
    _plant(
        out,
        "model-00001-of-00002.safetensors",
        "model-00002-of-00002.safetensors",
        index={"a.weight": "model-00001-of-00002.safetensors", "b.weight": "model-00002-of-00002.safetensors"},
    )

    sweep_after_full_save(str(out))

    assert _files(out) == {INDEX, "model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"}


def test_a_single_file_save_outlives_a_previous_runs_complete_shard_set(tmp_path):
    """The opposite direction, and the one a writer that clears nothing actually hits.

    ``reset_sinks``' safetensors branch copies the source directory in and writes ONE file, deleting
    nothing — so a complete previous sharded save is still there, index and shards intact, and both
    layouts describe a full checkpoint. Keeping the index because it is a consistent one deletes the
    file this save just wrote, and every later read resolves the previous run's shards instead.
    """
    out = tmp_path / "out"
    _plant(
        out,
        "model-00001-of-00002.safetensors",
        "model-00002-of-00002.safetensors",
        index={"a.weight": "model-00001-of-00002.safetensors", "b.weight": "model-00002-of-00002.safetensors"},
        age=10,  # the previous run
    )
    _plant(out, SINGLE)

    sweep_after_full_save(str(out))

    assert _files(out) == {SINGLE}, "the freshly written single file must survive its predecessor"


def test_a_single_file_save_sweeps_the_previous_runs_index(tmp_path):
    """Mirror case: the shards the stale index names are gone (``save_pretrained`` deletes numbered
    shards it did not write), so the index alone would send every index-first reader to nowhere."""
    out = tmp_path / "out"
    _plant(out, SINGLE, index={"a.weight": "model-00001-of-00002.safetensors"})

    sweep_after_full_save(str(out))

    assert _files(out) == {SINGLE}


def test_a_crashed_save_sweeps_nothing(tmp_path):
    """No index and no single file: the save never completed, so the directory's previous contents
    are the only checkpoint there is. Deleting them here is unrecoverable."""
    out = tmp_path / "out"
    _plant(out, "model-00001-of-00003.safetensors", "model-00002-of-00003.safetensors")
    before = _files(out)

    sweep_after_full_save(str(out))

    assert _files(out) == before


def test_an_index_whose_shards_are_missing_is_not_a_completed_save(tmp_path):
    """A torn index is not evidence of a completed write. Trusting it would keep the (missing) shards
    it names and delete the partial ones actually on disk — a directory with no weights at all."""
    out = tmp_path / "out"
    _plant(
        out,
        "model-00001-of-00002.safetensors",
        index={"a.weight": "model-00001-of-00002.safetensors", "b.weight": "model-00002-of-00002.safetensors"},
    )
    before = _files(out)

    sweep_after_full_save(str(out))

    assert _files(out) == before


def test_an_index_with_an_empty_weight_map_is_not_a_completed_save(tmp_path):
    """``all()`` over an empty shard set is vacuously true — an empty weight map must not read as a
    consistent index, or a torn write would sweep every real shard beside it."""
    out = tmp_path / "out"
    _plant(out, "model-00001-of-00002.safetensors", index={})
    before = _files(out)

    sweep_after_full_save(str(out))

    assert _files(out) == before


def test_non_weight_files_are_never_swept(tmp_path):
    """The sweep runs over a live checkpoint directory: config/tokenizer/resume sidecars and a
    co-located PEFT adapter are not this save's business, and an adapter deleted here is gone."""
    out = tmp_path / "out"
    _plant(
        out,
        SINGLE,
        "model-00001-of-00002.safetensors",
        "config.json",
        "tokenizer.json",
        "adapter_model.safetensors",
        "router_balancing_biases.pt",
    )

    sweep_after_full_save(str(out))

    assert _files(out) == {
        SINGLE,
        "config.json",
        "tokenizer.json",
        "adapter_model.safetensors",
        "router_balancing_biases.pt",
    }


def _plant_previous_sharded_run(out) -> set[str]:
    """A complete, back-dated sharded save — the checkpoint a failed re-save must not destroy."""
    _plant(
        out,
        "model-00001-of-00002.safetensors",
        "model-00002-of-00002.safetensors",
        index={"a.weight": "model-00001-of-00002.safetensors", "b.weight": "model-00002-of-00002.safetensors"},
        age=10,
    )
    return _files(out)


def test_save_sharded_state_dict_sweeps_only_after_a_completed_write(tmp_path, monkeypatch):
    """The shared writer must follow its own module's contract: sweeping the directory BEFORE
    writing leaves a write that dies (ENOSPC, preemption) with neither the previous checkpoint nor
    a new one — the inverse of everything this file pins about sweep_after_full_save."""
    out = tmp_path / "out"
    before = _plant_previous_sharded_run(out)

    def failing_write(*args, **kwargs):
        raise OSError("No space left on device")

    monkeypatch.setattr(checkpoint_format_mod, "_safetensors_save_file", failing_write)
    with pytest.raises(OSError):
        checkpoint_format_mod.save_sharded_state_dict({"w": torch.ones(2)}, str(out))

    assert _files(out) == before, "a failed save destroyed the directory's previous checkpoint"


def test_save_sharded_state_dict_still_sweeps_stale_files_after_success(tmp_path):
    """Anti-vacuity for the ordering test: the sweep still runs — after the write — so a completed
    save leaves exactly its own layout, never the previous run's beside it."""
    out = tmp_path / "out"
    _plant_previous_sharded_run(out)

    checkpoint_format_mod.save_sharded_state_dict({"w": torch.ones(2)}, str(out))

    assert _files(out) == {SINGLE}


def _hf_like_model():
    """The gathered writer branches on ``hasattr(model, "save_pretrained")`` and never calls it.

    A bare ``nn.Module`` (the writer's save-dtype keep-set walks the module tree) with no config —
    the config/tie steps skip, no norms/balancing keep anything — so the write ordering is what
    remains under test."""
    model = nn.Module()
    model.save_pretrained = None
    return model


def test_the_bin_fallback_writes_before_it_sweeps(tmp_path, monkeypatch):
    """The gathered writer's pytorch-format fallback must not delete every safetensors file BEFORE
    writing the .bin — a second failure (the likely case: the safetensors save already failed for a
    disk reason) then leaves the directory empty."""
    out = tmp_path / "out"
    before = _plant_previous_sharded_run(out)

    def failing(*args, **kwargs):
        raise OSError("No space left on device")

    monkeypatch.setattr(checkpoint_format_mod, "save_sharded_state_dict", failing)
    monkeypatch.setattr(checkpoint_format_mod.torch, "save", failing)
    ctx = SimpleNamespace(max_shard_size="5GB")
    with pytest.raises(OSError):
        write_gathered_checkpoint(_hf_like_model(), {"w": torch.ones(2)}, str(out), ctx.max_shard_size)

    assert _files(out) == before, "the fallback swept the previous checkpoint before its .bin existed"


def test_the_bin_fallback_replaces_the_weights_then_sweeps(tmp_path, monkeypatch):
    """Anti-vacuity: when the .bin write succeeds, the stale safetensors do go — resume prefers an
    index over the .bin, so leaving them would serve the previous run's weights."""
    out = tmp_path / "out"
    _plant_previous_sharded_run(out)

    def failing(*args, **kwargs):
        raise OSError("No space left on device")

    monkeypatch.setattr(checkpoint_format_mod, "save_sharded_state_dict", failing)
    ctx = SimpleNamespace(max_shard_size="5GB")
    write_gathered_checkpoint(_hf_like_model(), {"w": torch.ones(2)}, str(out), ctx.max_shard_size)

    assert _files(out) == {"pytorch_model.bin"}


def test_a_failed_sweep_after_the_bin_fallback_is_loud(tmp_path, monkeypatch, caplog):
    """The sweep exists because resume prefers an index over the .bin: if it fails, the directory
    resumes the PREVIOUS run's weights. That must be a warning, not a suppressed OSError."""
    out = tmp_path / "out"
    _plant_previous_sharded_run(out)

    def failing(*args, **kwargs):
        raise OSError("No space left on device")

    def failing_sweep(*args, **kwargs):
        raise OSError("Operation not permitted")

    monkeypatch.setattr(checkpoint_format_mod, "save_sharded_state_dict", failing)
    monkeypatch.setattr(checkpoint_format_mod, "remove_stale_checkpoint_files", failing_sweep)
    ctx = SimpleNamespace(max_shard_size="5GB")
    with caplog.at_level("WARNING", logger=checkpoint_format_mod.logger.name):
        write_gathered_checkpoint(_hf_like_model(), {"w": torch.ones(2)}, str(out), ctx.max_shard_size)

    assert "pytorch_model.bin" in _files(out), "the fallback .bin must survive a failed sweep"
    assert any("resume the OLD weights" in record.message for record in caplog.records), (
        "a failed stale sweep left the directory silently resuming the previous run's weights"
    )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
