#!/usr/bin/env python
"""What a per-rank EP save owes the tools that read it back.

Three contracts, each of which fails silently when broken:

* the index's ``total_size`` describes the artifact — counted off the tensors actually WRITTEN, so
  an fp32-master run does not stamp a figure up to 2x the bytes on disk;
* a per-rank shard is ONE file by construction (the merge reads exactly one per rank), so
  ``save_max_shard_size`` does not bound it — the save says so instead of appearing to honour it;
* the shards are unreadable as ordinary weights, and the index is written LAST — so a run killed in
  that window leaves ``model-*.safetensors`` full of ``.shard_N`` partial tensors with nothing
  declaring the format, which every after-training tool must refuse rather than read as whole.

    python tests/cpu/checkpoint/test_ep_sharded_save_contract.py
"""

from __future__ import annotations

import json
import logging
import os
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
from accelerate import PartialState
from safetensors import safe_open
from safetensors.torch import save_file

from src.checkpoint.format import SAFETENSORS_INDEX_FILE
from src.checkpoint.tool_io import checkpoint_shard_files, reject_sharded_checkpoint, stored_tensor_nbytes
from src.distributed.expert_parallel.saving import _save_ep_sharded
from tests.common.ep_stubs import StubEPLayerBase

PartialState()  # the EP save logs through accelerate's logger

NUM_LOCAL_EXPERTS = 2
HIDDEN = 8
INTER = 6


class _StubConfig(SimpleNamespace):
    def save_pretrained(self, output_dir):
        pass


class _StubEPLayer(StubEPLayerBase):
    """Local expert shards as fp32 params — the shape an fp32-master run holds at save time."""

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

    @classmethod
    def merge_shards_to_hf(cls, prefix: str, params: dict) -> dict:  # pragma: no cover - paired override
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
    """``metadata.total_size`` is read as the artifact's size. Counting the LIVE parameters instead
    of the cast ones puts an fp32-master run's figure — up to 2x the bytes on disk — on a bf16
    checkpoint, and the gathered writer counts what it writes."""
    model = _stub_model()
    live_bytes = sum(p.numel() * p.element_size() for p in model.parameters())

    _save_ep_sharded(model, str(tmp_path))

    with open(tmp_path / SAFETENSORS_INDEX_FILE) as f:
        index = json.load(f)
    shard = _shard_files(tmp_path)[0]
    with safe_open(str(tmp_path / shard), framework="pt") as reader:
        on_disk = sum(stored_tensor_nbytes(reader, key) for key in reader.keys())  # noqa: SIM118

    assert index["metadata"]["total_size"] == on_disk
    assert on_disk < live_bytes, "premise: fp32 masters were cast down on the way to disk"


def test_a_per_rank_shard_is_one_file_and_the_save_says_so(tmp_path, caplog):
    """``save_max_shard_size`` bounds the gathered/merged artifact, never these: the merge reads one
    file per rank. Splitting silently would break that reader; honouring it silently would leave the
    user believing a cap applied. The save writes one file and reports the knob it does not read."""
    model = _stub_model()

    with caplog.at_level(logging.INFO, logger="src.distributed.expert_parallel.saving"):
        _save_ep_sharded(model, str(tmp_path), max_shard_size="1KB")

    assert _shard_files(tmp_path) == ["model-00000-of-00001.safetensors"], "one shard per rank, cap or no cap"
    assert any("save_max_shard_size" in record.getMessage() for record in caplog.records), (
        "the save must name the knob it does not apply, or the user reads the single file as a bug"
    )


def _indexless_shard_dir(tmp_path) -> str:
    """The window between the shard writes and the index write: partial tensors, no index."""
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
    """Without the index nothing declares the format, so ``is_sharded_checkpoint`` reads False and
    the glob fallback would hand each rank's SLICE to a tool as the whole expert bank — the exact
    silent expert loss the indexed refusal exists to stop."""
    checkpoint = _indexless_shard_dir(tmp_path)

    with pytest.raises(ValueError, match="merge_ep_shards"):
        reject_sharded_checkpoint(checkpoint)
    with pytest.raises(ValueError, match="no model.safetensors.index.json"):
        checkpoint_shard_files(checkpoint)


def test_an_indexless_gathered_checkpoint_still_resolves(tmp_path):
    """Anti-over-rejection: the glob fallback exists for a gathered save written without its index,
    which carries no ``.shard_N`` keys and must keep working."""
    save_file(
        {"model.embed_tokens.weight": torch.zeros(4, 8)},
        os.path.join(tmp_path, "model-00001-of-00002.safetensors"),
        metadata={"format": "pt"},
    )
    assert checkpoint_shard_files(str(tmp_path)) == [str(tmp_path / "model-00001-of-00002.safetensors")]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
