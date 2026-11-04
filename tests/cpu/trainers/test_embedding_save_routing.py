#!/usr/bin/env python
"""The embedding save path is the shared checkpoint ladder, re-pointed at the ``auto_model`` backbone.
Every save step (reshard, router-balancing sidecar, writer) runs once on either branch."""

import contextlib
import os
import weakref
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
from peft import LoraConfig, inject_adapter_in_model
from peft.tuners.lora import LoraLayer
from sentence_transformers import SentenceTransformerTrainer

import halo.checkpoint.writing as writing_mod
import halo.models.peft_adapters as peft_adapters
import halo.trainers.checkpointing as checkpointing_mod
from halo.checkpoint.parallel.writing import conversion_chunk_key
from halo.checkpoint.reading import load_full_state_dict
from halo.checkpoint.writing import cast_to_save_dtype, normalize_gathered_state_dict
from halo.distributed.expert_parallel.moe_layer.layer import named_ep_layers
from halo.trainers.embedding import trainer as embedding_module
from halo.trainers.embedding.injected_lora import has_injected_lora
from halo.trainers.embedding.trainer import EmbeddingTrainer
from tests.common.adapters import randomize_adapters
from tests.common.ep_stubs import StubEPLayerBase
from tests.common.gloo import run_gloo_ranks

WORLD_SIZE = 2


class _Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 4, bias=False)


class _Backbone(nn.Module):
    def __init__(self, bare: bool = False):
        super().__init__()
        self.linear = nn.Linear(4, 4, bias=False)
        blocks = nn.ModuleList([_Block(), _Block()])
        if bare:
            self.layers = blocks
        else:
            self.encoder = nn.Module()
            self.encoder.layers = blocks


class _SharedExpertEPLayer(StubEPLayerBase):
    def __init__(self):
        super().__init__()
        self.shared_experts = nn.Linear(4, 4, bias=False)


class _SentenceTransformerLike(nn.Module):
    """The backbone is a child, not the model itself."""

    def __init__(self, backbone: nn.Module):
        super().__init__()
        self.transformer = backbone


def _host(*, has_lora: bool = False, is_save_rank: bool = True, max_shard_size: str = "3GB", bare: bool = False):
    backbone = _Backbone(bare)
    if has_lora:
        inject_adapter_in_model(LoraConfig(r=2, lora_alpha=8, target_modules=["linear"]), backbone)
        randomize_adapters(backbone, lora_b_only=True)
    top = _SentenceTransformerLike(backbone)

    host = object.__new__(EmbeddingTrainer)
    host.parallelism_config = _ParallelismStub()
    host.model = top
    host.args = SimpleNamespace()
    host.save_max_shard_size = max_shard_size
    host.processing_class = _TokenizerStub()
    host._fsdp_wrapped = True
    host._accelerate_manages_fsdp = False
    host.save_sharded_ep = False
    host._top_level_model = lambda: top
    host._backbone_model = lambda: backbone
    host._get_tp_rank = lambda: 0
    host._find_cp_wrapper = lambda: None
    host._is_save_rank = is_save_rank
    return host, backbone, top


class _ParallelismStub:
    is_pp_mode = False
    is_cp_mode = False
    is_tp_mode = False
    has_ep_group = False
    tp_size = 1
    merge_expert_lora_on_save = False


class _TokenizerStub:
    def __init__(self):
        self.saved_to = None

    def save_pretrained(self, output_dir):
        self.saved_to = output_dir


def _context(host, monkeypatch):
    monkeypatch.setattr(embedding_module, "is_fs_save_rank", lambda: host._is_save_rank)
    monkeypatch.setattr("halo.trainers.checkpointing.is_fs_save_rank", lambda: host._is_save_rank)
    return EmbeddingTrainer._checkpoint_context(host)


def test_checkpoint_context_points_at_the_backbone_not_the_st_sequential(monkeypatch):
    # A strategy handed the Sequential would gather and write "transformer.*"-prefixed keys no loader accepts.
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
    assert seen["ctx"].max_shard_size == "1GB"
    assert seen["ctx"].model is backbone


def test_gathered_lora_save_retains_only_on_the_writer(monkeypatch, tmp_path):
    # retain defaults True, so an unqualified gather leaves a full CPU state dict on every rank.
    host, _, _ = _host(has_lora=True, is_save_rank=False)
    ctx = _context(host, monkeypatch)
    seen = {}

    def _chunks(model, *, retain):
        seen["retain"] = retain
        return iter(())

    monkeypatch.setattr(embedding_module, "folded_backbone_chunks", _chunks)
    monkeypatch.setattr(
        embedding_module, "stream_gathered_checkpoint", lambda *a, is_save_rank, **k: seen.update(writes=is_save_rank)
    )

    EmbeddingTrainer._save_distributed_embedding_model(host, ctx, str(tmp_path))

    assert seen == {"retain": False, "writes": False}  # a non-writer rank runs the collectives and writes nothing


@pytest.mark.parametrize("bare", [False, True], ids=["prefixed-layers", "bare-layers"])
def test_injected_lora_save_streams_the_fold_one_conversion_chunk_at_a_time(monkeypatch, tmp_path, bare):
    """One conversion chunk (a decoder block) at a time, so the save rank never holds the whole backbone."""
    host, backbone, _ = _host(has_lora=True, bare=bare)
    ctx = _context(host, monkeypatch)
    expected = {
        f"{name}.weight": cast_to_save_dtype((module.base_layer.weight + module.get_delta_weight("default")).detach())
        for name, module in backbone.named_modules()
        if isinstance(module, LoraLayer)
    }
    chunks = []

    def recording(model, chunk, **kwargs):
        chunks.append({conversion_chunk_key(key) for key in chunk})
        return normalize_gathered_state_dict(model, chunk, **kwargs)

    monkeypatch.setattr(embedding_module, "normalize_gathered_state_dict", recording)

    EmbeddingTrainer._save_distributed_embedding_model(host, ctx, str(tmp_path))

    layers = "layers" if bare else "encoder.layers"
    assert chunks == [{""}, {f"{layers}.0"}, {f"{layers}.1"}]
    written = load_full_state_dict(str(tmp_path))
    assert written.keys() == expected.keys()  # adapters folded, base_layer spelling gone
    for key, tensor in expected.items():
        assert torch.equal(written[key], tensor), key
    assert host.processing_class.saved_to == str(tmp_path)


def test_the_injected_lora_save_folds_one_target_at_a_time(monkeypatch, tmp_path):
    """Each fold is a device copy of its target; folded up front, all would be alive at once."""
    host, _, _ = _host(has_lora=True)
    ctx = _context(host, monkeypatch)
    real_fold = peft_adapters.lora_folded
    alive = made = peak = 0

    def released():
        nonlocal alive
        alive -= 1

    def counting_fold(param, layers):
        nonlocal alive, made, peak
        folded = real_fold(param, layers)
        weakref.finalize(folded, released)
        alive, made = alive + 1, made + 1
        peak = max(peak, alive)
        return folded

    monkeypatch.setattr(peft_adapters, "lora_folded", counting_fold)

    EmbeddingTrainer._save_distributed_embedding_model(host, ctx, str(tmp_path))

    assert made == 3, "premise: one fold per adapted linear"
    assert peak == 1


def _enospc(*args, **kwargs):
    raise OSError(28, "No space left on device")


def _failing_write_worker(rank: int, tmp_dir: str) -> None:
    host, _, _ = _host(has_lora=True, is_save_rank=rank == 0)
    outcome = "NO RAISE"
    with pytest.MonkeyPatch.context() as patcher:
        ctx = _context(host, patcher)
        patcher.setattr(writing_mod, "save_safetensors_file", _enospc)
        try:
            EmbeddingTrainer._save_distributed_embedding_model(host, ctx, tmp_dir)
        except RuntimeError as error:
            outcome = str(error)
    with open(os.path.join(tmp_dir, f"result_{rank}.txt"), "w") as fh:
        fh.write(outcome)


def test_a_failed_injected_lora_write_raises_on_every_rank(tmp_path):
    """A ``pytorch_model.bin`` fallback would leave a format no other gathered save writes, or strand the peers."""
    run_gloo_ranks(_failing_write_worker, WORLD_SIZE, str(tmp_path))

    for rank in range(WORLD_SIZE):
        outcome = (tmp_path / f"result_{rank}.txt").read_text()
        assert "No space left on device" in outcome, f"rank {rank}: {outcome}"
    assert not (tmp_path / "pytorch_model.bin").exists()


def test_injected_lora_on_a_module_an_ep_layer_adopted_still_counts():
    """An EP layer adopts the block's shared experts as children, and the run still trains injected LoRA there.
    It must be folded on save and refused under EP, which an EP-excluding scan would miss."""
    layer = _SharedExpertEPLayer()
    backbone = _Backbone()
    backbone.add_module("moe", layer)
    inject_adapter_in_model(LoraConfig(r=2, lora_alpha=4, target_modules=["shared_experts"]), backbone)
    assert named_ep_layers(backbone) == [("moe", layer)], "premise: the adapters sit inside an EP layer"

    assert has_injected_lora(backbone)


def test_save_model_runs_every_writer_under_pristine_model_max_length(monkeypatch, tmp_path):
    # The run's sequence budget is pinned on the tokenizer, and save_pretrained would persist it as the served context.
    host, _, _ = _host()
    host._pristine_special_token_ids = []
    monkeypatch.setattr(checkpointing_mod, "save_balancing_biases", lambda *args, **kwargs: None)
    order = []

    @contextlib.contextmanager
    def _pristine(tokenizer):
        order.append("enter")
        yield
        order.append("exit")

    monkeypatch.setattr(checkpointing_mod, "pristine_model_max_length", _pristine)
    monkeypatch.setattr(embedding_module, "is_fs_save_rank", lambda: True)
    monkeypatch.setattr("halo.trainers.checkpointing.is_fs_save_rank", lambda: True)
    monkeypatch.setattr(
        EmbeddingTrainer,
        "_save_distributed_embedding_model",
        lambda self, ctx, output_dir, _internal_call=False: order.append("write"),
    )

    EmbeddingTrainer.save_model(host, str(tmp_path))

    assert order == ["enter", "write", "exit"]


def _record_save_steps(host, monkeypatch) -> list[str]:
    calls = []
    host._pristine_special_token_ids = []
    monkeypatch.setattr(checkpointing_mod, "save_balancing_biases", lambda *args, **kwargs: calls.append("sidecar"))
    for module in (embedding_module, checkpointing_mod):
        monkeypatch.setattr(module, "reshard_fsdp2_modules", lambda _model: calls.append("reshard"))
    monkeypatch.setattr(embedding_module, "is_fs_save_rank", lambda: True)
    monkeypatch.setattr("halo.trainers.checkpointing.is_fs_save_rank", lambda: True)
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
    """Single GPU, DDP, accelerate FSDP: a second copy of the mixin's steps would run the sidecar and reshard twice."""
    host, _, _ = _host()
    host._fsdp_wrapped = False
    host.args.should_save = False
    calls = _record_save_steps(host, monkeypatch)

    EmbeddingTrainer.save_model(host, str(tmp_path))

    assert calls == ["reshard", "sidecar", "st"]
    assert host._model_save_collectives_done


def test_a_one_process_save_of_ep_layers_writes_the_pipeline_config_beside_the_ladder(monkeypatch, tmp_path):
    """Only the ladder's EP saver writes hub-layout experts, but the mixin's payload adds no pipeline config beside it.
    The export would load as a bare backbone, mean-pooled and unnormalized."""
    host, backbone, _ = _host()
    backbone.add_module("moe", _SharedExpertEPLayer())
    host._fsdp_wrapped = False
    host.args.should_save = False
    calls = _record_save_steps(host, monkeypatch)
    monkeypatch.setattr(checkpointing_mod, "save_checkpoint", lambda ctx, output_dir: calls.append("ladder") or True)

    EmbeddingTrainer.save_model(host, str(tmp_path))

    assert calls == ["reshard", "sidecar", "distributed"]


def test_a_distributed_save_writes_through_the_mixins_payload_hook(monkeypatch, tmp_path):
    host, _, _ = _host()
    calls = _record_save_steps(host, monkeypatch)

    EmbeddingTrainer.save_model(host, str(tmp_path))

    assert calls == ["reshard", "sidecar", "distributed"]
    assert host._model_save_collectives_done


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
