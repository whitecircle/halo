#!/usr/bin/env python
"""``sweep_after_full_save``: a completed save owns the output directory, a crashed one owns nothing. When both
layouts are self-consistent, only the modification time says which is the leftover."""

import json
import os
import time

import pytest

from halo.checkpoint.writing import sweep_after_full_save

INDEX = "model.safetensors.index.json"
SINGLE = "model.safetensors"


def _plant(directory, *names, index: dict | None = None, age: float = 0.0) -> None:
    """``age`` back-dates the write, so a test can state which is the previous run's without relying on two writes
    microseconds apart."""
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
    """``from_pretrained`` prefers ``model.safetensors`` over the index, so the old weights would win."""
    out = tmp_path / "out"
    _plant(out, SINGLE, age=10)  # The previous run.
    _plant(
        out,
        "model-00001-of-00002.safetensors",
        "model-00002-of-00002.safetensors",
        index={"a.weight": "model-00001-of-00002.safetensors", "b.weight": "model-00002-of-00002.safetensors"},
    )

    sweep_after_full_save(str(out))

    assert _files(out) == {INDEX, "model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"}


def test_a_single_file_save_outlives_a_previous_runs_complete_shard_set(tmp_path):
    """``reset_sinks``' safetensors branch writes one file and deletes nothing; keeping the consistent index deletes
    the file just written."""
    out = tmp_path / "out"
    _plant(
        out,
        "model-00001-of-00002.safetensors",
        "model-00002-of-00002.safetensors",
        index={"a.weight": "model-00001-of-00002.safetensors", "b.weight": "model-00002-of-00002.safetensors"},
        age=10,  # The previous run.
    )
    _plant(out, SINGLE)

    sweep_after_full_save(str(out))

    assert _files(out) == {SINGLE}, "the freshly written single file must survive its predecessor"


def test_a_single_file_save_sweeps_the_previous_runs_index(tmp_path):
    """The index alone would send every index-first reader to nowhere."""
    out = tmp_path / "out"
    _plant(out, SINGLE, index={"a.weight": "model-00001-of-00002.safetensors"})

    sweep_after_full_save(str(out))

    assert _files(out) == {SINGLE}


def test_a_crashed_save_sweeps_nothing(tmp_path):
    """The directory's previous contents are the only checkpoint there is."""
    out = tmp_path / "out"
    _plant(out, "model-00001-of-00003.safetensors", "model-00002-of-00003.safetensors")
    before = _files(out)

    sweep_after_full_save(str(out))

    assert _files(out) == before


def test_an_index_whose_shards_are_missing_is_not_a_completed_save(tmp_path):
    """Trusting it would keep the missing shards and delete the partial ones, leaving no weights at all."""
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
    """``all()`` over an empty shard set is vacuously true."""
    out = tmp_path / "out"
    _plant(out, "model-00001-of-00002.safetensors", index={})
    before = _files(out)

    sweep_after_full_save(str(out))

    assert _files(out) == before


def test_non_weight_files_are_never_swept(tmp_path):
    """A co-located PEFT adapter deleted here is gone."""
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


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
