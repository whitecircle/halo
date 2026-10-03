#!/usr/bin/env python
"""Pure TP adapter resume and best-model loads use the DTensor-aware adapter restorer."""

import json
from unittest.mock import Mock

import pytest
import torch
import torch.distributed as dist
from peft import LoraConfig, get_peft_model
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import DTensor, Shard, distribute_tensor
from transformers import AutoConfig, AutoModelForCausalLM

import src.distributed.checkpoint.loader as loader_mod
from src.checkpoint.format import ADAPTER_CONFIG_FILE, ADAPTER_SAFETENSORS_FILE
from src.distributed.checkpoint.context import CheckpointLoadContext
from src.distributed.checkpoint.loader import CheckpointLoader


@pytest.fixture
def single_rank_gloo(tmp_path):
    dist.init_process_group("gloo", rank=0, world_size=1, init_method=f"file://{tmp_path / 'pg'}")
    try:
        yield init_device_mesh("cpu", (1,), mesh_dim_names=("tp",))
    finally:
        dist.destroy_process_group()


def _model(*, alpha=8, mesh=None):
    config = AutoConfig.for_model(
        "qwen3",
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        tie_word_embeddings=False,
        attn_implementation="eager",
    )
    model = get_peft_model(
        AutoModelForCausalLM.from_config(config, dtype=torch.float32),
        LoraConfig(r=4, lora_alpha=alpha, target_modules=["q_proj", "o_proj"], task_type="CAUSAL_LM"),
    )
    if mesh is not None:
        attention = model.get_base_model().model.layers[0].self_attn
        for factor, dim in ((attention.q_proj.lora_B.default, 0), (attention.o_proj.lora_A.default, 1)):
            factor.weight = torch.nn.Parameter(distribute_tensor(factor.weight.detach(), mesh, [Shard(dim)]))
    return model


def _ctx(model, *, fsdp_wrapped=False):
    return CheckpointLoadContext(
        model=model,
        optimizer=None,
        lr_scheduler=None,
        parallelism_config=None,
        is_pp_mode=False,
        is_cp_mode=False,
        is_tp_mode=True,
        has_ep_layers=False,
        fsdp_wrapped=fsdp_wrapped,
        tp_rank=0,
        tp_size=2,
        super_load_from_checkpoint=Mock(side_effect=AssertionError("ordinary Trainer must not restore TP adapters")),
        super_load_optimizer_and_scheduler=Mock(),
    )


def _snapshot(model):
    return {
        name: (param.full_tensor() if isinstance(param, DTensor) else param).detach().clone()
        for name, param in model.named_parameters()
    }


def _save_changed_adapter(model, path, *, safe_serialization=True):
    with torch.no_grad():
        for index, param in enumerate(p for p in model.parameters() if p.requires_grad):
            param.fill_(0.125 * (index + 1))
    model.save_pretrained(path, safe_serialization=safe_serialization)


@pytest.mark.parametrize("for_best_model", [False, True], ids=["resume", "best"])
@pytest.mark.parametrize("safe_serialization", [False, True], ids=["bin", "safetensors"])
def test_tp_adapter_restore_writes_plain_and_dtensor_factors(
    tmp_path, single_rank_gloo, for_best_model, safe_serialization
):
    trained = _model()
    _save_changed_adapter(trained, tmp_path, safe_serialization=safe_serialization)
    expected = _snapshot(trained)
    fresh = _model(mesh=single_rank_gloo)
    before = _snapshot(fresh)
    assert sum(isinstance(param, DTensor) for param in fresh.parameters() if param.requires_grad) == 2

    CheckpointLoader(_ctx(fresh)).load_model(str(tmp_path), for_best_model=for_best_model)

    restored = _snapshot(fresh)
    for name, param in fresh.named_parameters():
        assert torch.equal(restored[name], expected[name] if param.requires_grad else before[name]), name
        if param.requires_grad:
            assert not torch.equal(restored[name], before[name]), f"{name} stayed at initialization"


@pytest.mark.parametrize("for_best_model", [False, True], ids=["resume", "best"])
def test_tp_adapter_scaling_mismatch_refuses_before_writing(tmp_path, single_rank_gloo, for_best_model):
    _save_changed_adapter(_model(), tmp_path)
    fresh = _model(alpha=16, mesh=single_rank_gloo)
    before = _snapshot(fresh)

    with pytest.raises(ValueError, match="another LoRA scaling"):
        CheckpointLoader(_ctx(fresh)).load_model(str(tmp_path), for_best_model=for_best_model)

    assert all(torch.equal(value, before[name]) for name, value in _snapshot(fresh).items())


def test_tp_adapter_malformed_scaling_config_is_not_bypassed(tmp_path, single_rank_gloo):
    _save_changed_adapter(_model(), tmp_path)
    (tmp_path / ADAPTER_CONFIG_FILE).write_text("{broken", encoding="utf-8")
    fresh = _model(mesh=single_rank_gloo)
    before = _snapshot(fresh)

    with pytest.raises(json.JSONDecodeError):
        CheckpointLoader(_ctx(fresh)).load_model(str(tmp_path))

    assert all(torch.equal(value, before[name]) for name, value in _snapshot(fresh).items())


def test_tp_dp_adapter_restore_is_refused(tmp_path):
    model = _model()
    _save_changed_adapter(model, tmp_path)
    with pytest.raises(RuntimeError, match="TP\\+DP adapter reload is unsupported"):
        CheckpointLoader(_ctx(model, fsdp_wrapped=True)).load_model(str(tmp_path))


def test_torn_tp_adapter_raises_instead_of_falling_back(tmp_path):
    (tmp_path / ADAPTER_SAFETENSORS_FILE).write_bytes(b"not a safetensors file")
    with pytest.raises(RuntimeError, match="torn/corrupt"):
        CheckpointLoader(_ctx(_model())).load_model(str(tmp_path))


def test_torn_model_file_beside_adapter_keeps_full_model_fallback(tmp_path):
    model = _model()
    _save_changed_adapter(model, tmp_path)
    (tmp_path / "model.safetensors").write_bytes(b"not a safetensors file")
    ctx = _ctx(model)
    ctx.super_load_from_checkpoint = Mock()

    CheckpointLoader(ctx).load_model(str(tmp_path))

    ctx.super_load_from_checkpoint.assert_called_once_with(str(tmp_path), model)


def test_non_rank0_enters_the_same_adapter_restore(tmp_path, monkeypatch):
    model = _model()
    monkeypatch.setattr(loader_mod, "is_global_main_process", lambda: False)
    decisions = iter([False, False, True])
    monkeypatch.setattr(loader_mod, "broadcast_from_rank0", lambda _local: next(decisions))
    restore = Mock(return_value="peer-adapter.safetensors")
    monkeypatch.setattr(loader_mod, "restore_adapters", restore)

    CheckpointLoader(_ctx(model)).load_model(str(tmp_path))

    restore.assert_called_once_with(str(tmp_path), model, is_cp_mode=False)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
