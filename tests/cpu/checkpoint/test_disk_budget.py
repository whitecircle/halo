#!/usr/bin/env python
"""A training run refuses, before its first step, an output volume that cannot hold its rotation peak of checkpoints:
rotation deletes the oldest only after the new save, so ``save_total_limit`` + 1 coexist."""

import logging

import pytest
import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from transformers import Qwen3Config, Qwen3ForCausalLM
from trl import SFTConfig

import halo.checkpoint.parallel.disk_budget as disk_budget
from halo.checkpoint.parallel.disk_budget import (
    estimate_checkpoint_bytes,
    estimate_model_bytes,
    verify_checkpoint_disk_space,
)
from halo.trainers.sft import DistributedSFTTrainer
from tests.common.checkpoint_io import save_context
from tests.common.parallelism import make_parallelism_config
from tests.common.tiny_models import word_level_tokenizer

ITEMSIZE = 4  # fp32: the CPU runs here are not bf16
WORDS = ("solve", "the", "problem")


def _model() -> Qwen3ForCausalLM:
    config = Qwen3Config(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
    )
    return Qwen3ForCausalLM(config)


def _numel(model) -> int:
    return sum(p.numel() for p in model.parameters())


def test_the_model_is_sized_from_its_config_not_its_live_shards():
    model = _model()
    assert estimate_model_bytes(model, torch.float32) == ITEMSIZE * _numel(model)
    assert estimate_model_bytes(model, torch.bfloat16) == 2 * _numel(model)


def test_a_full_checkpoint_is_the_weights_plus_two_moments_per_trainable_parameter():
    model = _model()
    ctx = save_context(model)
    full = estimate_checkpoint_bytes(ctx, torch.float32, save_optimizer=True, writes_optimizer_shards=False)
    weights_only = estimate_checkpoint_bytes(ctx, torch.float32, save_optimizer=False, writes_optimizer_shards=False)
    assert weights_only == ITEMSIZE * _numel(model)
    assert full == 3 * weights_only


def test_frozen_parameters_carry_no_optimizer_state():
    model = _model()
    for param in model.model.layers[0].parameters():
        param.requires_grad = False
    frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    ctx = save_context(model)
    full = estimate_checkpoint_bytes(ctx, torch.float32, save_optimizer=True, writes_optimizer_shards=False)
    assert full == ITEMSIZE * (_numel(model) + 2 * (_numel(model) - frozen))


def test_an_adapter_save_counts_the_adapters_alone():
    model = get_peft_model(_model(), LoraConfig(r=2, target_modules=["q_proj"], task_type="CAUSAL_LM"))
    adapters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    ctx = save_context(model)
    assert estimate_checkpoint_bytes(ctx, torch.float32, save_optimizer=False, writes_optimizer_shards=False) == (
        ITEMSIZE * adapters
    )
    assert estimate_checkpoint_bytes(ctx, torch.float32, save_optimizer=True, writes_optimizer_shards=False) == (
        3 * ITEMSIZE * adapters
    )


def test_optimizer_shards_divide_over_the_nodes_of_a_node_local_filesystem(monkeypatch):
    model = _model()
    ctx = save_context(model)
    monkeypatch.setattr(disk_budget, "is_output_shared_filesystem", lambda: False)
    monkeypatch.setattr(disk_budget, "get_num_nodes", lambda: 4)
    weights = ITEMSIZE * _numel(model)
    sharded = estimate_checkpoint_bytes(ctx, torch.float32, save_optimizer=True, writes_optimizer_shards=True)
    whole = estimate_checkpoint_bytes(ctx, torch.float32, save_optimizer=True, writes_optimizer_shards=False)
    assert (sharded, whole) == (weights + 2 * weights // 4, 3 * weights)
    monkeypatch.setattr(disk_budget, "is_output_shared_filesystem", lambda: True)
    assert estimate_checkpoint_bytes(ctx, torch.float32, save_optimizer=True, writes_optimizer_shards=True) == whole


def _verify(model, tmp_path, monkeypatch, *, free, limit, is_save_rank=True, optimizer=True):
    monkeypatch.setattr(disk_budget, "free_disk_bytes", lambda _path: free)
    verify_checkpoint_disk_space(
        save_context(model, is_save_rank=is_save_rank),
        str(tmp_path),
        torch.float32,
        save_total_limit=limit,
        save_optimizer=optimizer,
        writes_optimizer_shards=False,
    )


def _one_checkpoint(model) -> int:
    return 3 * ITEMSIZE * _numel(model)


def test_the_rotation_peak_is_the_limit_plus_one(tmp_path, monkeypatch):
    model = _model()
    per = _one_checkpoint(model)
    _verify(model, tmp_path, monkeypatch, free=3 * per + 10**6, limit=2)
    with pytest.raises(ValueError, match=r"3 checkpoint\(s\).*ENOSPC"):
        _verify(model, tmp_path, monkeypatch, free=3 * per - 1, limit=2)


def test_without_a_limit_only_the_next_save_is_checked(tmp_path, monkeypatch):
    model = _model()
    per = _one_checkpoint(model)
    _verify(model, tmp_path, monkeypatch, free=per + 10**6, limit=None)
    with pytest.raises(ValueError, match=r"1 checkpoint\(s\)"):
        _verify(model, tmp_path, monkeypatch, free=per - 1, limit=None)


def test_a_thin_margin_warns_and_runs(tmp_path, monkeypatch, caplog):
    model = _model()
    per = _one_checkpoint(model)
    with caplog.at_level(logging.WARNING):
        _verify(model, tmp_path, monkeypatch, free=int(2 * per * 1.05), limit=1)
    assert any("margin" in record.getMessage() for record in caplog.records)
    caplog.clear()
    with caplog.at_level(logging.WARNING):
        _verify(model, tmp_path, monkeypatch, free=2 * per * 2, limit=1)
    assert not caplog.records


def test_save_only_model_drops_the_optimizer_from_the_bill(tmp_path, monkeypatch):
    model = _model()
    free = 2 * ITEMSIZE * _numel(model) + 16
    _verify(model, tmp_path, monkeypatch, free=free, limit=1, optimizer=False)
    with pytest.raises(ValueError):
        _verify(model, tmp_path, monkeypatch, free=free, limit=1)


def test_a_rank_that_is_no_filesystems_writer_checks_nothing(tmp_path, monkeypatch):
    _verify(_model(), tmp_path, monkeypatch, free=0, limit=5, is_save_rank=False)


def test_an_unsizable_model_warns_instead_of_failing_the_run(tmp_path, monkeypatch, caplog):
    def broken(*_args, **_kwargs):
        raise RuntimeError("no shell")

    monkeypatch.setattr(disk_budget, "estimate_checkpoint_bytes", broken)
    with caplog.at_level(logging.WARNING):
        _verify(_model(), tmp_path, monkeypatch, free=0, limit=5)
    assert any("Could not size" in record.getMessage() for record in caplog.records)


def _trainer(tmp_path, **config):
    tokenizer = word_level_tokenizer(WORDS)
    ids = tokenizer("solve the problem")["input_ids"]
    args = SFTConfig(
        output_dir=str(tmp_path / "out"),
        use_cpu=True,
        bf16=False,
        max_length=len(ids),
        max_steps=1,
        per_device_train_batch_size=1,
        gradient_checkpointing=False,
        use_liger_kernel=False,
        remove_unused_columns=False,
        dataloader_num_workers=0,
        eval_strategy="no",
        report_to="none",
        **config,
    )
    return DistributedSFTTrainer(
        model=_model(),
        args=args,
        train_dataset=Dataset.from_dict({"input_ids": [ids] * 2, "labels": [ids] * 2}),
        processing_class=tokenizer,
        parallelism_config=make_parallelism_config(world_size=1, gpus_per_node=1, use_grouped_gemm=False),
        moe_balancing="none",
    )


def test_the_trainer_refuses_a_volume_too_small_for_its_saves_at_construction(tmp_path, monkeypatch):
    monkeypatch.setattr(disk_budget, "free_disk_bytes", lambda _path: 1000)
    with pytest.raises(ValueError, match="ENOSPC"):
        _trainer(tmp_path, save_strategy="steps", save_steps=1, save_total_limit=2)


def test_a_run_that_never_saves_is_not_judged_on_disk(tmp_path, monkeypatch):
    monkeypatch.setattr(disk_budget, "free_disk_bytes", lambda _path: 1000)
    _trainer(tmp_path, save_strategy="no")


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
