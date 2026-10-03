#!/usr/bin/env python
"""Real pre-wrap dense/CP/EP1 loads and 1-D TP replay retain off-BF16-grid checkpoint masters.

The actual loader, key planner and streamed reads run. Only GPU placement and native communication
buffers are substituted on CPU. Gloo ranks exercise native HF TP construction, packed placements,
tied owner replacement, and a one-rank read failure after the real node loading throttle.
"""

import datetime
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.nn as nn
from accelerate import PartialState
from safetensors.torch import save_file
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import DTensor, Replicate, Shard, distribute_tensor
from torch.distributed.tensor.placement_types import _StridedShard
from transformers import AutoConfig, Qwen3Config, Qwen3ForCausalLM

import src.distributed.context_parallel.loading as cp_loading
import src.distributed.filesystem as filesystem
import src.distributed.loading.master_weights as master_loading
import src.distributed.loading.model_loading as loading
from src.distributed.context_parallel.wrapper import patch_model_for_cp
from src.distributed.loading.precision import fp32_master_param_keys
from tests.common.gloo import run_gloo_ranks

PartialState()


def _dense_checkpoint(path: Path):
    torch.manual_seed(27)
    model = Qwen3ForCausalLM(
        Qwen3Config(
            vocab_size=64,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=1,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=8,
            tie_word_embeddings=True,
        )
    ).to(torch.bfloat16)
    for param in model.parameters():
        param.data = param.data.float() + 1e-5
    expected = {key: value.detach().clone() for key, value in model.state_dict(keep_vars=True).items()}
    assert expected and all(not torch.equal(value, value.bfloat16().float()) for value in expected.values())
    model.save_pretrained(path)
    model.config.dtype = torch.bfloat16
    model.config.save_pretrained(path)
    return expected


def _cpu_pretrained(verified):
    def load(model_class, path, **kwargs):
        kwargs["device_map"] = "cpu"
        return verified(model_class, path, **kwargs)

    return load


def _cpu_cp_wrap(model, config):
    model.config._attn_implementation = "flash_attention_2"
    return patch_model_for_cp(model, config)


def _dense_pc(*, keep=True):
    return SimpleNamespace(
        max_concurrent_loading=1,
        fp32_non_ep_params=keep,
        needs_ep_wrappers=False,
        cp_size=2,
        create_cp_config=lambda: SimpleNamespace(cp_size=2, cp_rank=0, process_group=None),
    )


@pytest.mark.parametrize("mode", ("dense", "cp"))
@pytest.mark.parametrize("strict", (False, True), ids=("fresh_stage", "strict_resume"))
@pytest.mark.parametrize("keep", (False, True), ids=("run_dtype", "fp32_masters"))
def test_real_dense_and_cp_construction_retains_checkpoint_masters(tmp_path, monkeypatch, mode, strict, keep):
    expected = _dense_checkpoint(tmp_path / "source")
    source = str(tmp_path / "source")
    monkeypatch.setattr(loading, "from_pretrained_verified", _cpu_pretrained(loading.from_pretrained_verified))
    monkeypatch.setattr(cp_loading, "move_model_to_local_device", lambda model: model)
    monkeypatch.setattr(cp_loading, "patch_model_for_cp", _cpu_cp_wrap)
    kwargs = {
        "config": AutoConfig.from_pretrained(source),
        "dtype": torch.bfloat16,
        "attn_implementation": "eager",
        "trust_remote_code": False,
        "preserve_checkpoint_precision": strict,
    }
    if mode == "cp":
        model = loading._load_cp_model(source, _dense_pc(keep=keep), Qwen3ForCausalLM, kwargs)
    else:
        model = loading._load_undistributed_model(
            source, _dense_pc(keep=keep), Qwen3ForCausalLM, kwargs, 0, ep_wrappers=False
        )
    actual = model.state_dict(keep_vars=True)
    for key, value in expected.items():
        oracle = value if keep else value.bfloat16()
        assert actual[key].dtype == oracle.dtype
        assert torch.equal(actual[key].detach(), oracle), key
    owners = model.model if mode == "cp" else model
    assert owners.lm_head.weight is owners.model.embed_tokens.weight


def _native_tp_rank(rank, source, strict, keep):
    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr(
            loading,
            "create_dp_tp_mesh",
            lambda tp_size, dp_size: init_device_mesh("cpu", (tp_size,), mesh_dim_names=("tp",)),
        )
        pc = SimpleNamespace(tp_size=2, data_parallel_size=1, fp32_non_ep_params=keep)
        model = loading._load_tp_model(
            source,
            pc,
            Qwen3ForCausalLM,
            {
                "config": AutoConfig.from_pretrained(source),
                "dtype": torch.bfloat16,
                "attn_implementation": "eager",
                "trust_remote_code": False,
                "preserve_checkpoint_precision": strict,
            },
        )
    oracle = Qwen3ForCausalLM.from_pretrained(source, dtype=torch.float32 if keep else torch.bfloat16)
    expected = oracle.state_dict(keep_vars=True)
    sharded = 0
    for key, param in model.state_dict(keep_vars=True).items():
        value = param.full_tensor() if isinstance(param, DTensor) else param
        sharded += isinstance(param, DTensor)
        assert value.dtype == expected[key].dtype
        if isinstance(param, DTensor):
            assert param.to_local().dtype == expected[key].dtype, f"{key}: stale local DTensor dtype"
        assert torch.equal(value.detach(), expected[key].detach()), key
    assert sharded > 0, "the real HF-native TP load must shard weights"
    assert model.lm_head.weight is model.model.embed_tokens.weight


@pytest.mark.parametrize("strict", (False, True), ids=("fresh_stage", "strict_resume"))
@pytest.mark.parametrize("keep", (False, True), ids=("run_dtype", "fp32_masters"))
def test_native_tp_load_preserves_checkpoint_masters_and_local_storage(tmp_path, strict, keep):
    source = tmp_path / "source"
    _dense_checkpoint(source)
    run_gloo_ranks(_native_tp_rank, 2, str(source), strict, keep, pg_timeout=datetime.timedelta(seconds=30))


def _placement_rank(rank, path):
    mesh = init_device_mesh("cpu", (2,), mesh_dim_names=("tp",))
    # Packed row order must differ from an ordinary contiguous Shard(0) oracle.
    expected = torch.arange(64).reshape(8, 8).float() / 16 + 1e-5
    for placement in (Shard(0), Shard(1), Replicate(), _StridedShard(0, split_factor=2)):
        model = nn.Module()
        model.config = Qwen3Config(tie_word_embeddings=False)
        model.weight = nn.Parameter(distribute_tensor(expected.bfloat16(), mesh, [placement], src_data_rank=None))
        model.alias = nn.Module()
        model.alias.weight = model.weight
        original = model.weight
        master_loading.restore_fp32_master_parameters(model, path, keep_non_ep=True, strict=True)
        assert model.alias.weight is model.weight and model.weight is not original
        assert model.weight.device_mesh == mesh and model.weight.placements == (placement,)
        assert model.weight.shape == expected.shape and model.weight.stride() == expected.stride()
        assert model.weight.dtype == model.weight.to_local().dtype == torch.float32
        if isinstance(placement, _StridedShard):
            chunks = expected.chunk(4, dim=0)
            local = torch.cat((chunks[rank], chunks[rank + 2]), dim=0)
        elif isinstance(placement, Shard):
            local = expected.chunk(2, dim=placement.dim)[rank]
        else:
            local = expected
        assert torch.equal(model.weight.to_local().detach(), local)
        if isinstance(placement, _StridedShard):
            # This synthetic 1-D packed placement has no TP/DP shard_order for full_tensor().
            # Reconstruct independently through Gloo: ranks own [chunk0, chunk2]/[chunk1, chunk3].
            gathered = [torch.empty_like(local) for _ in range(2)]
            dist.all_gather(gathered, model.weight.to_local().detach())
            left, right = (shard.chunk(2, dim=0) for shard in gathered)
            full = torch.cat((left[0], right[0], left[1], right[1]), dim=0)
        else:
            full = model.weight.full_tensor().detach()
        assert torch.equal(full, expected)


def test_tp_replay_preserves_each_placement_and_all_tied_owners(tmp_path):
    expected = torch.arange(64).reshape(8, 8).float() / 16 + 1e-5
    save_file({"weight": expected}, str(tmp_path / "model.safetensors"))
    run_gloo_ranks(_placement_rank, 2, str(tmp_path), pg_timeout=datetime.timedelta(seconds=30))


def _read_failure_rank(rank, source, outcomes):
    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr(loading, "from_pretrained_verified", _cpu_pretrained(loading.from_pretrained_verified))
        monkeypatch.setattr(filesystem, "get_store_timeout", lambda: datetime.timedelta(seconds=10))
        original_get = master_loading.StreamingCheckpointReader.get

        def get(reader, key):
            if rank == 1:
                raise OSError("injected rank-1 master read failure")
            return original_get(reader, key)

        monkeypatch.setattr(master_loading.StreamingCheckpointReader, "get", get)
        try:
            loading._load_undistributed_model(
                source,
                _dense_pc(),
                Qwen3ForCausalLM,
                {
                    "config": AutoConfig.from_pretrained(source),
                    "dtype": torch.bfloat16,
                    "attn_implementation": "eager",
                    "trust_remote_code": False,
                    "preserve_checkpoint_precision": True,
                },
                rank,
                ep_wrappers=False,
            )
        except RuntimeError as error:
            outcome = str(error)
        else:
            outcome = "incorrectly returned a constructed model"
        Path(outcomes, f"rank{rank}.txt").write_text(outcome)


def test_a_rank_local_master_read_failure_joins_after_the_loading_throttle(tmp_path):
    source = tmp_path / "source"
    _dense_checkpoint(source)
    run_gloo_ranks(
        _read_failure_rank,
        2,
        str(source),
        str(tmp_path),
        pg_timeout=datetime.timedelta(seconds=10),
    )
    outcomes = [Path(tmp_path, f"rank{rank}.txt").read_text() for rank in range(2)]
    assert all("injected rank-1 master read failure" in outcome for outcome in outcomes), outcomes
    assert all("Rank 1" in outcome or "rank 1" in outcome for outcome in outcomes), outcomes


def test_packed_float_4bit_storage_is_not_selected_as_a_parameter_master():
    model = nn.Linear(4, 4)
    model.weight.quant_state = None
    assert fp32_master_param_keys(model, keep_non_ep=True) == frozenset({"bias"})


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
