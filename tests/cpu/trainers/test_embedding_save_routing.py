#!/usr/bin/env python
"""The embedding save path is the shared checkpoint ladder, not a fourth hand-rolled copy.

``EmbeddingTrainer`` owns one genuine difference from every other trainer: its top-level model is a
``SentenceTransformer`` ``nn.Sequential``, so the checkpoint context has to be re-pointed at the
``auto_model`` backbone. Everything downstream of that — save dtype, hub expert layout, shard size,
the ``.bin`` fallback, and the single retaining rank on a gathered save — must come from
``save_checkpoint`` and the shared gather/write leaves rather than a local re-implementation.

Run: pytest tests/cpu/trainers/test_embedding_save_routing.py
"""

import contextlib

import pytest
import torch
import torch.nn as nn

from src.trainers.embedding import trainer as embedding_module
from src.trainers.embedding.trainer import EmbeddingTrainer


class _Backbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 4, bias=False)


class _SentenceTransformerLike(nn.Module):
    """Stands in for the ST Sequential: the backbone is a CHILD, not the model itself."""

    def __init__(self, backbone: nn.Module):
        super().__init__()
        self.transformer = backbone


def _host(*, has_lora: bool = False, is_save_rank: bool = True, max_shard_size: str = "3GB"):
    backbone = _Backbone()
    if has_lora:
        # inject_adapter_in_model's in-place layout: "<module>.lora_A.<adapter>.weight".
        adapters = nn.Module()
        adapters.lora_A = nn.ModuleDict({"default": nn.Linear(4, 2, bias=False)})
        adapters.lora_B = nn.ModuleDict({"default": nn.Linear(2, 4, bias=False)})
        backbone.add_module("q_proj", adapters)
    top = _SentenceTransformerLike(backbone)

    host = object.__new__(EmbeddingTrainer)
    host.parallelism_config = _ParallelismStub()
    host.model = top
    host.args = _ArgsStub(max_shard_size)
    host.processing_class = _TokenizerStub()
    host._fsdp_wrapped = True
    host._accelerate_manages_fsdp = False
    host.save_sharded_ep = False
    host._top_level_model = lambda: top
    host._get_unwrapped_model = lambda: backbone
    host._get_tp_rank = lambda: 0
    host._find_cp_wrapper = lambda: None
    host._is_save_rank = is_save_rank
    return host, backbone, top


class _ParallelismStub:
    is_pp_mode = False
    is_cp_mode = False
    is_tp_mode = False
    is_ep_tp_mode = False
    is_ep_mode = False
    tp_size = 1
    merge_expert_lora_on_save = False


class _ArgsStub:
    def __init__(self, max_shard_size):
        self.save_max_shard_size = max_shard_size


class _TokenizerStub:
    def __init__(self):
        self.saved_to = None

    def save_pretrained(self, output_dir):
        self.saved_to = output_dir


def _context(host, monkeypatch):
    monkeypatch.setattr(embedding_module, "fs_aware_save_rank", lambda: host._is_save_rank)
    monkeypatch.setattr("src.trainers.mixins.checkpointing.fs_aware_save_rank", lambda: host._is_save_rank)
    return EmbeddingTrainer._checkpoint_context(host)


def test_checkpoint_context_points_at_the_backbone_not_the_st_sequential(monkeypatch):
    # The base factory snapshots _top_level_model(); a strategy handed the Sequential would gather
    # and write "transformer.*"-prefixed keys no loader accepts.
    host, backbone, top = _host()
    ctx = _context(host, monkeypatch)

    assert ctx.model is backbone
    assert ctx.model is not top
    assert ctx.tokenizer is host.processing_class


def test_non_lora_save_delegates_to_the_registry_with_the_configured_shard_size(monkeypatch, tmp_path):
    host, backbone, _ = _host(max_shard_size="1GB")
    ctx = _context(host, monkeypatch)
    seen = {}
    monkeypatch.setattr(
        embedding_module, "save_checkpoint", lambda ctx_arg, out: seen.update(ctx=ctx_arg, out=out) or True
    )

    EmbeddingTrainer._save_distributed_embedding_model(host, ctx, str(tmp_path))

    assert seen["ctx"] is ctx
    assert seen["out"] == str(tmp_path)
    # The hand-rolled EP/TP branches dropped max_shard_size on the floor; the ladder carries it.
    assert seen["ctx"].max_shard_size == "1GB"
    assert seen["ctx"].model is backbone


def test_gathered_lora_save_retains_only_on_the_writer(monkeypatch, tmp_path):
    # retain defaults True, so an unqualified gather leaves a full CPU state dict on EVERY rank.
    host, _, _ = _host(has_lora=True, is_save_rank=False)
    ctx = _context(host, monkeypatch)
    seen = {}

    def _gather(model, retain: bool = True):
        seen["retain"] = retain
        return {}

    monkeypatch.setattr(embedding_module, "gather_saveable_tensors", _gather)
    monkeypatch.setattr(embedding_module, "write_gathered_checkpoint", lambda *a, **k: seen.setdefault("wrote", True))

    EmbeddingTrainer._save_distributed_embedding_model(host, ctx, str(tmp_path))

    assert seen["retain"] is False
    assert "wrote" not in seen  # a non-writer rank runs the collective and nothing else


def test_gathered_lora_save_writes_through_the_shared_writer(monkeypatch, tmp_path):
    """The merged dict must go through ``write_gathered_checkpoint``.

    Writing it straight to ``save_sharded_state_dict`` skips ``normalize_gathered_state_dict``, so an
    fp32-master run exports fp32 while the same model under EP/TP exports the save dtype, a MoE
    backbone exports module-fused expert keys vLLM rejects, and there is no ``.bin`` recovery.
    """
    host, backbone, _ = _host(has_lora=True, is_save_rank=True)
    ctx = _context(host, monkeypatch)
    written = {}

    monkeypatch.setattr(
        embedding_module,
        "gather_saveable_tensors",
        lambda model, retain=True: {"linear.base_layer.weight": torch.zeros(4, 4)},
    )

    def _write(model, state_dict, output_dir, max_shard_size=None):
        written["model"] = model
        written["keys"] = sorted(state_dict)
        written["max_shard_size"] = max_shard_size

    monkeypatch.setattr(embedding_module, "write_gathered_checkpoint", _write)

    EmbeddingTrainer._save_distributed_embedding_model(host, ctx, str(tmp_path))

    assert written["model"] is backbone
    assert written["max_shard_size"] == ctx.max_shard_size
    assert written["keys"] == ["linear.weight"]  # adapters folded, base_layer spelling gone
    assert host.processing_class.saved_to == str(tmp_path)


def test_save_model_runs_every_writer_under_pristine_model_max_length(monkeypatch, tmp_path):
    # The run's sequence budget is pinned on the tokenizer as a truncation default; save_pretrained
    # would otherwise persist it as the exported model's served context.
    host, _, _ = _host()
    host._pristine_special_token_ids = []
    host._persist_router_balancing_biases = lambda _dir: None
    order = []

    @contextlib.contextmanager
    def _pristine(tokenizer):
        order.append("enter")
        yield
        order.append("exit")

    monkeypatch.setattr(embedding_module, "pristine_model_max_length", _pristine)
    monkeypatch.setattr(embedding_module, "fs_aware_save_rank", lambda: True)
    monkeypatch.setattr("src.trainers.mixins.checkpointing.fs_aware_save_rank", lambda: True)
    monkeypatch.setattr(
        EmbeddingTrainer,
        "_save_distributed_embedding_model",
        lambda self, ctx, output_dir, _internal_call=False: order.append("write"),
    )

    EmbeddingTrainer.save_model(host, str(tmp_path))

    assert order == ["enter", "write", "exit"]


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
