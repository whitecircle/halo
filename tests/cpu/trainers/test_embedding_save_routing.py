#!/usr/bin/env python
"""The embedding save path is the shared checkpoint ladder.

``EmbeddingTrainer`` owns one genuine difference from every other trainer: its top-level model is a
``SentenceTransformer`` ``nn.Sequential``, so the checkpoint context has to be re-pointed at the
``auto_model`` backbone. Everything downstream of that — save dtype, hub expert layout, shard size,
the ``.bin`` fallback, and the single retaining rank on a gathered save — must come from
``save_checkpoint`` and the shared gather/write leaves rather than a local re-implementation. A run
with nothing to gather is the mixin's save alone, so every save step (reshard, router-balancing
sidecar, writer) runs once on either branch.

Run: pytest tests/cpu/trainers/test_embedding_save_routing.py
"""

import contextlib

import pytest
import torch
import torch.nn as nn
from peft import LoraConfig, inject_adapter_in_model
from sentence_transformers import SentenceTransformerTrainer

import src.trainers.mixins.checkpointing as checkpointing_mod
from src.distributed.expert_parallel.base_layer import find_ep_layers
from src.trainers.embedding import trainer as embedding_module
from src.trainers.embedding.trainer import EmbeddingTrainer
from tests.common.ep_stubs import StubEPLayerBase
from tests.common.peft_helpers import randomize_adapters


class _Backbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 4, bias=False)


class _SharedExpertEPLayer(StubEPLayerBase):
    """An EP layer holding the block's shared experts, which the EP layers adopt as a child."""

    def __init__(self):
        super().__init__()
        self.shared_experts = nn.Linear(4, 4, bias=False)


class _SentenceTransformerLike(nn.Module):
    """Stands in for the ST Sequential: the backbone is a CHILD, not the model itself."""

    def __init__(self, backbone: nn.Module):
        super().__init__()
        self.transformer = backbone


def _host(*, has_lora: bool = False, is_save_rank: bool = True, max_shard_size: str = "3GB"):
    backbone = _Backbone()
    if has_lora:
        inject_adapter_in_model(LoraConfig(r=2, lora_alpha=8, target_modules=["linear"]), backbone)
        randomize_adapters(backbone, lora_b_only=True)
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
    # The ladder carries max_shard_size through to the EP/TP savers.
    assert seen["ctx"].max_shard_size == "1GB"
    assert seen["ctx"].model is backbone


def test_gathered_lora_save_retains_only_on_the_writer(monkeypatch, tmp_path):
    # retain defaults True, so an unqualified gather leaves a full CPU state dict on EVERY rank.
    host, _, _ = _host(has_lora=True, is_save_rank=False)
    ctx = _context(host, monkeypatch)
    seen = {}

    def _gather(model, retain: bool = True, items=None):
        seen["retain"] = retain
        return {}

    monkeypatch.setattr(embedding_module, "gather_saveable_tensors", _gather)
    monkeypatch.setattr(embedding_module, "write_gathered_checkpoint", lambda *a, **k: seen.setdefault("wrote", True))

    EmbeddingTrainer._save_distributed_embedding_model(host, ctx, str(tmp_path))

    assert seen["retain"] is False
    assert "wrote" not in seen  # a non-writer rank runs the collective and nothing else


def test_gathered_lora_save_writes_the_fold_through_the_shared_writer(monkeypatch, tmp_path):
    """The folded tensors must go through ``write_gathered_checkpoint``, under the plain names.

    Writing them straight to ``save_sharded_state_dict`` skips ``normalize_gathered_state_dict``, so an
    fp32-master run exports fp32 while the same model under EP/TP exports the save dtype, a MoE
    backbone exports module-fused expert keys vLLM rejects, and there is no ``.bin`` recovery.
    """
    host, backbone, _ = _host(has_lora=True, is_save_rank=True)
    ctx = _context(host, monkeypatch)
    layer = backbone.linear
    expected = layer.base_layer.weight + layer.get_delta_weight("default")
    assert not torch.equal(expected, layer.base_layer.weight), "premise: the adapter moves its base"
    written = {}

    def _write(model, state_dict, output_dir, max_shard_size=None):
        written["model"] = model
        written["state_dict"] = state_dict
        written["max_shard_size"] = max_shard_size

    monkeypatch.setattr(embedding_module, "write_gathered_checkpoint", _write)

    EmbeddingTrainer._save_distributed_embedding_model(host, ctx, str(tmp_path))

    assert written["model"] is backbone
    assert written["max_shard_size"] == ctx.max_shard_size
    assert sorted(written["state_dict"]) == ["linear.weight"]  # adapters folded, base_layer spelling gone
    assert torch.equal(written["state_dict"]["linear.weight"], expected.detach())
    assert host.processing_class.saved_to == str(tmp_path)


def test_injected_lora_on_a_module_an_ep_layer_adopted_still_counts():
    """An EP layer adopts the block's shared experts as children, so a target list that reaches only
    those puts every adapter inside it. The run still trains injected LoRA: it must be folded on save
    and refused under EP, which an EP-excluding scan would miss."""
    layer = _SharedExpertEPLayer()
    backbone = _Backbone()
    backbone.add_module("moe", layer)
    inject_adapter_in_model(LoraConfig(r=2, lora_alpha=4, target_modules=["shared_experts"]), backbone)
    assert find_ep_layers(backbone) == [("moe", layer)], "premise: the adapters sit inside an EP layer"

    assert EmbeddingTrainer._has_injected_lora(None, backbone)


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

    monkeypatch.setattr(checkpointing_mod, "pristine_model_max_length", _pristine)
    monkeypatch.setattr(embedding_module, "fs_aware_save_rank", lambda: True)
    monkeypatch.setattr("src.trainers.mixins.checkpointing.fs_aware_save_rank", lambda: True)
    monkeypatch.setattr(
        EmbeddingTrainer,
        "_save_distributed_embedding_model",
        lambda self, ctx, output_dir, _internal_call=False: order.append("write"),
    )

    EmbeddingTrainer.save_model(host, str(tmp_path))

    assert order == ["enter", "write", "exit"]


def _record_save_steps(host, monkeypatch) -> list[str]:
    """Record the save steps that must run once per save: the reshard, the balancing sidecar, each writer."""
    calls = []
    host._pristine_special_token_ids = []
    host._persist_router_balancing_biases = lambda _dir: calls.append("sidecar")
    for module in (embedding_module, checkpointing_mod):
        monkeypatch.setattr(module, "reshard_fsdp2_modules", lambda _model: calls.append("reshard"))
        monkeypatch.setattr(module, "fs_aware_save_rank", lambda: True)
    monkeypatch.setattr(
        EmbeddingTrainer,
        "_save_distributed_embedding_model",
        lambda self, ctx, output_dir, _internal_call=False: calls.append("distributed"),
    )
    monkeypatch.setattr(
        SentenceTransformerTrainer, "save_model", lambda self, output_dir, _internal_call=False: calls.append("st")
    )
    return calls


def test_a_plain_save_is_the_mixins_save_alone(monkeypatch, tmp_path):
    """Single GPU, DDP and accelerate FSDP: the mixin's save owns the reshard, the sidecar and the fence
    mark, with ST's writer as its base fallback. Wrapping it in a second copy of those steps runs each
    twice — a second sidecar write, a second reshard."""
    host, _, _ = _host()
    host._fsdp_wrapped = False
    host.args.should_save = False
    calls = _record_save_steps(host, monkeypatch)

    EmbeddingTrainer.save_model(host, str(tmp_path))

    assert calls == ["reshard", "sidecar", "st"]
    assert host._model_save_collectives_done


def test_a_distributed_save_writes_through_the_mixins_payload_hook(monkeypatch, tmp_path):
    """EP/TP/FSDP2 and injected LoRA replace only the payload: the mixin's save keeps the reshard, the
    sidecar and the fence mark, each once, and ST's writer never runs."""
    host, _, _ = _host()
    calls = _record_save_steps(host, monkeypatch)

    EmbeddingTrainer.save_model(host, str(tmp_path))

    assert calls == ["reshard", "sidecar", "distributed"]
    assert host._model_save_collectives_done


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
