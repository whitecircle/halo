#!/usr/bin/env python
"""Safetensors-index classification and reading: ``is_ep_sharded_checkpoint`` + ``load_full_state_dict``. The predicate
sits in rank-0 decision paths, so a torn index classifies False: a rank-local raise would strand peers."""

import json
import os

import pytest
import torch
from safetensors.torch import save_file

from halo.checkpoint.format import EP_SHARDED_FORMAT, ep_sharded_index_metadata
from halo.checkpoint.parallel.reading import reject_ep_sharded_resume
from halo.checkpoint.reading import (
    ep_sharded_checkpoint_reason,
    is_ep_sharded_checkpoint,
    load_full_state_dict,
    reject_partial_checkpoint,
)
from halo.model_loading.parallel.model_source import ModelSource, reject_ep_sharded_checkpoint


def _write_index(tmp_path, payload: str) -> str:
    path = os.path.join(tmp_path, "model.safetensors.index.json")
    with open(path, "w") as fh:
        fh.write(payload)
    return str(tmp_path)


def test_gathered_index_without_format_marker_is_not_sharded(tmp_path):
    payload = json.dumps({"metadata": {"total_size": 4}, "weight_map": {"w": "model-00001.safetensors"}})
    assert is_ep_sharded_checkpoint(_write_index(tmp_path, payload)) is False


def test_sharded_format_marker_detected(tmp_path):
    payload = json.dumps({"metadata": {"format": EP_SHARDED_FORMAT}, "weight_map": {}})
    assert is_ep_sharded_checkpoint(_write_index(tmp_path, payload)) is True


def test_the_ep_sharded_index_metadata_is_what_the_predicate_reads(tmp_path):
    metadata = ep_sharded_index_metadata(total_size=12, ep_size=4)

    assert metadata == {"total_size": 12, "ep_size": 4, "format": "ep_sharded"}
    assert is_ep_sharded_checkpoint(_write_index(tmp_path, json.dumps({"metadata": metadata, "weight_map": {}})))


def test_pp_stage_index_is_not_per_rank_sharded(tmp_path):
    """Its tensors are complete, and it stamps no format marker."""
    payload = json.dumps(
        {"metadata": {"total_size": 8}, "weight_map": {"w": "model-pp00000-of-00002-00001.safetensors"}}
    )
    assert is_ep_sharded_checkpoint(_write_index(tmp_path, payload)) is False


def test_torn_truncated_index_returns_false_not_raise(tmp_path):
    """Never propagate the JSONDecodeError into the rank-0 decision path."""
    full = json.dumps({"metadata": {"format": "ep_sharded"}, "weight_map": {"w": "shard-0.safetensors"}})
    torn_dir = _write_index(tmp_path, full[: len(full) // 2])
    assert is_ep_sharded_checkpoint(torn_dir) is False


def test_empty_index_file_returns_false(tmp_path):
    assert is_ep_sharded_checkpoint(_write_index(tmp_path, "")) is False


@pytest.mark.parametrize("payload", ["", "{not json", '{"metadata": {"format": "ep_sha'])
def test_load_full_state_dict_diagnoses_a_torn_index(tmp_path, payload):
    """Only the peer holding the torn copy raises, so its traceback has to carry the diagnosis."""
    checkpoint_dir = _write_index(tmp_path, payload)
    with pytest.raises(RuntimeError, match="unreadable"):
        load_full_state_dict(checkpoint_dir)


def test_load_full_state_dict_rejects_per_rank_shards(tmp_path):
    """From the same parse."""
    payload = json.dumps({"metadata": {"format": "ep_sharded"}, "weight_map": {"w": "model-00001.safetensors"}})
    with pytest.raises(ValueError, match="merge_ep_shards"):
        load_full_state_dict(_write_index(tmp_path, payload))


def test_every_ep_sharded_refusal_gives_one_reason(tmp_path):
    """Four paths refuse the same artifact, so they tell the user one thing."""
    payload = json.dumps({"metadata": {"format": "ep_sharded"}, "weight_map": {"w": "model-00001.safetensors"}})
    checkpoint_dir = _write_index(tmp_path, payload)
    with pytest.raises(ValueError) as read:
        load_full_state_dict(checkpoint_dir)
    with pytest.raises(ValueError) as resume:
        reject_ep_sharded_resume(checkpoint_dir)
    with pytest.raises(ValueError) as load:
        reject_ep_sharded_checkpoint(ModelSource("checkpoints/run", None, checkpoint_dir))
    with pytest.raises(ValueError) as tool:
        reject_partial_checkpoint(checkpoint_dir)

    reason = ep_sharded_checkpoint_reason(checkpoint_dir)
    assert str(read.value) == str(resume.value) == str(load.value) == str(tool.value) == reason
    assert "per-rank sharded checkpoint" in str(tool.value) and "merge_ep_shards" in str(tool.value)


def test_load_full_state_dict_rejects_an_index_without_weight_map(tmp_path):
    """Not a KeyError deep in the read loop."""
    with pytest.raises(RuntimeError, match="weight_map"):
        load_full_state_dict(_write_index(tmp_path, json.dumps({"metadata": {"total_size": 4}})))


def test_load_full_state_dict_reads_a_gathered_index(tmp_path):
    tensors = {"a": torch.zeros(2), "b": torch.ones(3)}
    save_file({"a": tensors["a"]}, os.path.join(tmp_path, "model-00001.safetensors"), metadata={"format": "pt"})
    save_file({"b": tensors["b"]}, os.path.join(tmp_path, "model-00002.safetensors"), metadata={"format": "pt"})
    payload = json.dumps(
        {
            "metadata": {"total_size": 20},
            "weight_map": {"a": "model-00001.safetensors", "b": "model-00002.safetensors"},
        }
    )
    loaded = load_full_state_dict(_write_index(tmp_path, payload))
    assert loaded is not None and set(loaded) == {"a", "b"}
    assert torch.equal(loaded["a"], tensors["a"]) and torch.equal(loaded["b"], tensors["b"])


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
