#!/usr/bin/env python
"""The single open-every-shard loop standalone checkpoint tools read through, plus ``detect_model_type``'s fallback."""

import contextlib
import json
from pathlib import Path

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import save_file

from halo.checkpoint import reading
from halo.checkpoint.reading import (
    detect_model_type,
    detect_model_types,
    iter_checkpoint_shard_entries,
    iter_checkpoint_tensors,
)

_SHARD_A = "model-00001-of-00002.safetensors"
_SHARD_B = "model-00002-of-00002.safetensors"


class _ReaderSpy:
    """Records the tensor reads a walk actually performs."""

    def __init__(self, reader, events: list[tuple[str, str]]):
        self._reader = reader
        self._events = events

    def keys(self):
        return self._reader.keys()

    def get_tensor(self, key: str):
        self._events.append(("read", key))
        return self._reader.get_tensor(key)


def _record_shard_io(monkeypatch) -> list[tuple[str, str]]:
    """``(open|close|read, shard name | key)`` at the walk's real seam."""
    events: list[tuple[str, str]] = []

    @contextlib.contextmanager
    def _recording_open(shard, framework):
        name = Path(shard).name
        events.append(("open", name))
        try:
            with safe_open(shard, framework=framework) as reader:
                yield _ReaderSpy(reader, events)
        finally:
            events.append(("close", name))

    monkeypatch.setattr(reading, "safe_open", _recording_open)
    return events


def _sharded_checkpoint(tmp_path: Path) -> Path:
    first = {"model.layers.0.self_attn.sinks": torch.zeros(4), "model.layers.0.mlp.weight": torch.ones(2, 3)}
    second = {"model.layers.1.self_attn.sinks": torch.zeros(4), "model.layers.1.mlp.weight": torch.ones(2, 3) * 2}
    save_file(first, str(tmp_path / _SHARD_A), metadata={"format": "pt"})
    save_file(second, str(tmp_path / _SHARD_B), metadata={"format": "pt"})
    weight_map = dict.fromkeys(first, _SHARD_A) | dict.fromkeys(second, _SHARD_B)
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {"total_size": 0}, "weight_map": weight_map})
    )
    return tmp_path


def test_entries_attribute_every_key_to_the_shard_that_actually_holds_it(tmp_path):
    """A key pointed at the wrong shard reads a different tensor and silently breaks merge_models."""
    root = _sharded_checkpoint(tmp_path)

    by_key = {key: Path(shard).name for shard, _reader, key in iter_checkpoint_shard_entries(str(root))}

    assert by_key == {
        "model.layers.0.self_attn.sinks": _SHARD_A,
        "model.layers.0.mlp.weight": _SHARD_A,
        "model.layers.1.self_attn.sinks": _SHARD_B,
        "model.layers.1.mlp.weight": _SHARD_B,
    }


def test_tensors_come_back_whole_across_shards(tmp_path):
    root = _sharded_checkpoint(tmp_path)

    tensors = dict(iter_checkpoint_tensors(str(root)))

    assert sorted(tensors) == [
        "model.layers.0.mlp.weight",
        "model.layers.0.self_attn.sinks",
        "model.layers.1.mlp.weight",
        "model.layers.1.self_attn.sinks",
    ]
    assert torch.equal(tensors["model.layers.1.mlp.weight"], torch.ones(2, 3) * 2)


def test_the_predicate_decides_before_the_tensor_is_read(tmp_path, monkeypatch):
    """Asserted on the reads: loading each shard whole and filtering afterwards returns the same two tensors."""
    root = _sharded_checkpoint(tmp_path)
    events = _record_shard_io(monkeypatch)
    offered: list[str] = []

    def is_sink(key: str) -> bool:
        offered.append(key)
        return key.endswith(".sinks")

    read = dict(iter_checkpoint_tensors(str(root), predicate=is_sink))

    assert len(offered) == 4, "every key must be offered to the predicate"
    assert sorted(read) == ["model.layers.0.self_attn.sinks", "model.layers.1.self_attn.sinks"]
    assert sorted(key for kind, key in events if kind == "read") == [
        "model.layers.0.self_attn.sinks",
        "model.layers.1.self_attn.sinks",
    ]


def test_a_predicate_matching_nothing_reads_nothing(tmp_path, monkeypatch):
    root = _sharded_checkpoint(tmp_path)
    events = _record_shard_io(monkeypatch)

    assert dict(iter_checkpoint_tensors(str(root), predicate=lambda key: False)) == {}
    assert [kind for kind, _ in events if kind == "read"] == []


def test_each_shard_is_opened_only_while_the_walk_is_on_it(tmp_path, monkeypatch):
    """The tools walk checkpoints far larger than host RAM."""
    root = _sharded_checkpoint(tmp_path)
    events = _record_shard_io(monkeypatch)

    walk = iter_checkpoint_tensors(str(root))
    assert events == [], "constructing the walk must open nothing"

    first_key, _first_tensor = next(walk)
    assert [name for kind, name in events if kind == "open"] == [_SHARD_A]
    assert first_key.startswith("model.layers.0."), "the index order decides which shard comes first"

    remaining = [key for key, _tensor in walk]

    assert [name for kind, name in events if kind == "open"] == [_SHARD_A, _SHARD_B], "one open per shard"
    assert events.index(("close", _SHARD_A)) < events.index(("open", _SHARD_B)), (
        "shard B was opened while shard A was still open — the walk is holding the checkpoint open"
    )
    assert len(remaining) == 3


def test_an_early_stop_opens_no_further_shard(tmp_path, monkeypatch):
    """A walk pre-opening every shard pays the whole checkpoint for one question."""
    root = _sharded_checkpoint(tmp_path)
    events = _record_shard_io(monkeypatch)

    shard, _reader, _key = next(iter(iter_checkpoint_shard_entries(str(root))))

    assert Path(shard).name == _SHARD_A
    assert [name for kind, name in events if kind == "open"] == [_SHARD_A]
    assert [kind for kind, _ in events if kind == "read"] == [], "a key-only scan reads no tensors"


def test_detect_model_type_falls_back_to_the_text_config(tmp_path):
    """An empty answer refuses a convertible checkpoint."""
    (tmp_path / "config.json").write_text(json.dumps({"architectures": ["X"], "text_config": {"model_type": "qwen3"}}))

    assert detect_model_type(str(tmp_path)) == "qwen3"


def test_a_declared_top_level_model_type_still_wins(tmp_path):
    (tmp_path / "config.json").write_text(
        json.dumps({"model_type": "gemma4", "text_config": {"model_type": "gemma4_text"}})
    )

    assert detect_model_type(str(tmp_path)) == "gemma4"


def test_detect_model_types_leads_with_the_language_family(tmp_path):
    """The fused-expert layout is the language family's; a ``null`` sub-config has none."""
    (tmp_path / "config.json").write_text(
        json.dumps({"model_type": "gemma4", "text_config": {"model_type": "gemma4_text"}})
    )
    assert detect_model_types(str(tmp_path)) == ["gemma4_text", "gemma4"]

    (tmp_path / "config.json").write_text(json.dumps({"model_type": "qwen3", "text_config": None}))
    assert detect_model_types(str(tmp_path)) == ["qwen3"]
    assert detect_model_types(str(tmp_path / "absent")) == []


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
