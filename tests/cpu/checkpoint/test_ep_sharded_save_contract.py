#!/usr/bin/env python
"""What a per-rank EP save owes the tools that read it back; each contract fails silently when broken."""

from __future__ import annotations

import json
import logging
import os
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
from safetensors import safe_open
from safetensors.torch import save_file

from halo.checkpoint import reading
from halo.checkpoint.format import SAFETENSORS_INDEX_FILE
from halo.checkpoint.parallel.ep import save_ep_sharded_checkpoint
from halo.checkpoint.reading import checkpoint_shard_files, header_nbytes, reject_partial_checkpoint
from tests.common.checkpoint_io import save_context
from tests.common.ep_stubs import StubEPLayerBase

NUM_LOCAL_EXPERTS = 2
HIDDEN = 8
INTER = 6


class _StubConfig(SimpleNamespace):
    def save_pretrained(self, output_dir):
        pass


class _StubEPLayer(StubEPLayerBase):
    """The shape an fp32-master run holds at save time."""

    def __init__(self):
        super().__init__()
        self.gate_up_proj = nn.Parameter(torch.randn(NUM_LOCAL_EXPERTS, HIDDEN, 2 * INTER))
        self.down_proj = nn.Parameter(torch.randn(NUM_LOCAL_EXPERTS, INTER, HIDDEN))
        self.router = nn.Linear(HIDDEN, NUM_LOCAL_EXPERTS, bias=False)
        self.ep_config = SimpleNamespace(ep_size=1, ep_group_size=1, expert_tp_size=1, num_ep_groups=1)

    def expert_named_params(self):
        return [("gate_up_proj", self.gate_up_proj), ("down_proj", self.down_proj)]

    def gather_expert_state_dict(self, device: str = "cpu", merge_lora: bool = False) -> dict:
        return {}


def _stub_model() -> nn.Module:
    model = nn.Module()
    backbone = nn.Module()
    layer = nn.Module()
    layer.self_attn = nn.Linear(HIDDEN, HIDDEN, bias=False)
    layer.mlp = _StubEPLayer()
    backbone.layers = nn.ModuleList([layer])
    model.model = backbone
    model.config = _StubConfig(model_type="gpt_oss", auto_map=None, tie_word_embeddings=False)
    return model


def _shard_files(path) -> list[str]:
    return sorted(name for name in os.listdir(path) if name.endswith(".safetensors"))


def test_index_total_size_counts_the_bytes_actually_written(tmp_path):
    """Counting live parameters stamps up to 2x the bytes on disk on a bf16 checkpoint."""
    model = _stub_model()
    live_bytes = sum(p.numel() * p.element_size() for p in model.parameters())

    save_ep_sharded_checkpoint(save_context(model, is_save_rank=True), str(tmp_path))

    with open(tmp_path / SAFETENSORS_INDEX_FILE) as f:
        index = json.load(f)
    shard = _shard_files(tmp_path)[0]
    with safe_open(str(tmp_path / shard), framework="pt") as reader:
        on_disk = sum(header_nbytes(reader.get_slice(key)) for key in reader.keys())  # noqa: SIM118

    assert index["metadata"]["total_size"] == on_disk
    assert on_disk < live_bytes, "premise: fp32 masters were cast down on the way to disk"


def test_a_per_rank_shard_is_one_file_and_the_save_says_so(tmp_path, caplog):
    """The merge reads one file per rank, so splitting would break it and silence would imply a cap applied."""
    model = _stub_model()

    with caplog.at_level(logging.INFO, logger="halo.checkpoint.parallel.ep"):
        save_ep_sharded_checkpoint(save_context(model, is_save_rank=True, max_shard_size="1KB"), str(tmp_path))

    assert _shard_files(tmp_path) == ["model-00000-of-00001.safetensors"], "one shard per rank, cap or no cap"
    assert any("save_max_shard_size" in record.getMessage() for record in caplog.records), (
        "the save must name the knob it does not apply, or the user reads the single file as a bug"
    )


def _indexless_shard_dir(tmp_path) -> str:
    """The window between the shard writes and the index write."""
    save_file(
        {
            "model.layers.0.mlp.experts.gate_up_proj.shard_0": torch.zeros(2, 4),
            "model.embed_tokens.weight": torch.zeros(4, 8),
        },
        os.path.join(tmp_path, "model-00000-of-00002.safetensors"),
        metadata={"format": "pt"},
    )
    return str(tmp_path)


def test_indexless_per_rank_shards_are_refused(tmp_path):
    """Without the index the glob fallback would hand each rank's slice to a tool as the whole expert bank."""
    checkpoint = _indexless_shard_dir(tmp_path)

    with pytest.raises(ValueError, match="merge_ep_shards"):
        reject_partial_checkpoint(checkpoint)
    with pytest.raises(ValueError, match="no model.safetensors.index.json"):
        checkpoint_shard_files(checkpoint)


def test_an_unreadable_indexless_shard_is_left_to_the_callers_read(tmp_path):
    """The caller's own read reports it with its cause."""
    (tmp_path / "model-00000-of-00002.safetensors").write_bytes(b"x" * 64)
    reject_partial_checkpoint(str(tmp_path))


def test_a_fault_in_the_indexless_peek_raises_instead_of_waving_shards_through(tmp_path, monkeypatch):
    """Swallowing a guard bug would pass partial ``.shard_N`` tensors as whole weights."""
    checkpoint = _indexless_shard_dir(tmp_path)
    monkeypatch.setattr(reading, "EP_SHARD_KEY_RE", None)

    with pytest.raises(AttributeError):
        reject_partial_checkpoint(checkpoint)


def test_an_indexless_gathered_checkpoint_still_resolves(tmp_path):
    save_file(
        {"model.embed_tokens.weight": torch.zeros(4, 8)},
        os.path.join(tmp_path, "model-00001-of-00002.safetensors"),
        metadata={"format": "pt"},
    )
    assert checkpoint_shard_files(str(tmp_path)) == [str(tmp_path / "model-00001-of-00002.safetensors")]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
