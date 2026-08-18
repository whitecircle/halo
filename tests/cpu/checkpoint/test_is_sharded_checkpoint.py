#!/usr/bin/env python
"""Safetensors-index classification and reading: ``is_sharded_checkpoint`` + ``load_full_state_dict``.

The predicate gates resume paths inside rank-0 decision/broadcast patterns, so a torn/truncated
``model.safetensors.index.json`` must classify as ``False`` (not raise — a rank-local raise would
strand peer ranks in the collective; the downstream read-consensus surfaces the torn checkpoint
with the real cause).

``load_full_state_dict`` reads the SAME file on every rank (``CheckpointLoader`` TP resume calls it
world-wide after a rank-0 readability probe). It must therefore parse the index exactly once and
diagnose a torn one explicitly, instead of asking the tolerant predicate and then re-opening the
file for an unguarded ``json.load`` that surfaces a bare ``JSONDecodeError`` from whichever rank
holds the torn copy.

Run: pytest tests/cpu/checkpoint/test_is_sharded_checkpoint.py
"""

import json
import os
import sys

import pytest
import torch
from safetensors.torch import save_file

from src.checkpoint.format import EP_SHARDED_FORMAT, is_sharded_checkpoint, load_full_state_dict


def _write_index(tmp_path, payload: str) -> str:
    path = os.path.join(tmp_path, "model.safetensors.index.json")
    with open(path, "w") as fh:
        fh.write(payload)
    return str(tmp_path)


def test_gathered_index_without_format_marker_is_not_sharded(tmp_path):
    """A gathered HF index (plain weight_map, no format metadata) must stay directly loadable."""
    payload = json.dumps({"metadata": {"total_size": 4}, "weight_map": {"w": "model-00001.safetensors"}})
    assert is_sharded_checkpoint(_write_index(tmp_path, payload)) is False


def test_sharded_format_marker_detected(tmp_path):
    payload = json.dumps({"metadata": {"format": EP_SHARDED_FORMAT}, "weight_map": {}})
    assert is_sharded_checkpoint(_write_index(tmp_path, payload)) is True


def test_pp_stage_index_is_not_per_rank_sharded(tmp_path):
    """A PP save writes one shard per stage under GLOBAL names and deliberately stamps no format
    marker — its tensors are complete, so it must stay directly loadable."""
    payload = json.dumps(
        {"metadata": {"total_size": 8}, "weight_map": {"w": "model-pp00000-of-00002-00001.safetensors"}}
    )
    assert is_sharded_checkpoint(_write_index(tmp_path, payload)) is False


def test_torn_truncated_index_returns_false_not_raise(tmp_path):
    """A truncated index (killed mid-write) is unparseable JSON — the predicate must return False,
    never propagate the JSONDecodeError into the rank-0 decision path."""
    full = json.dumps({"metadata": {"format": "ep_sharded"}, "weight_map": {"w": "shard-0.safetensors"}})
    torn_dir = _write_index(tmp_path, full[: len(full) // 2])
    assert is_sharded_checkpoint(torn_dir) is False


def test_empty_index_file_returns_false(tmp_path):
    assert is_sharded_checkpoint(_write_index(tmp_path, "")) is False


@pytest.mark.parametrize("payload", ["", "{not json", '{"metadata": {"format": "ep_sha'])
def test_load_full_state_dict_diagnoses_a_torn_index(tmp_path, payload):
    """A torn index must raise ONE explicit error naming the cause, not a bare JSONDecodeError.

    ``load_full_state_dict`` runs on every rank; the peer that happens to hold the torn copy is the
    only one that raises, so its traceback has to carry the diagnosis (partial write / non-shared FS)
    rather than a decoder error from an unguarded re-parse.
    """
    checkpoint_dir = _write_index(tmp_path, payload)
    with pytest.raises(RuntimeError, match="unreadable"):
        load_full_state_dict(checkpoint_dir)


def test_load_full_state_dict_rejects_per_rank_shards(tmp_path):
    """The per-rank verdict must come from the same parse, and still point at the merge scripts."""
    payload = json.dumps({"metadata": {"format": "ep_sharded"}, "weight_map": {"w": "model-00001.safetensors"}})
    with pytest.raises(ValueError, match="merge_ep_shards"):
        load_full_state_dict(_write_index(tmp_path, payload))


def test_load_full_state_dict_rejects_an_index_without_weight_map(tmp_path):
    """Parseable JSON that is not an index must fail loud, not KeyError deep in the read loop."""
    with pytest.raises(RuntimeError, match="weight_map"):
        load_full_state_dict(_write_index(tmp_path, json.dumps({"metadata": {"total_size": 4}})))


def test_load_full_state_dict_reads_a_gathered_index(tmp_path):
    """The gathered/FSDP2 resume path is untouched by the single-parse rewrite."""
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
    sys.exit(pytest.main([__file__, "-v"]))
