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
from bitsandbytes.nn import Linear4bit, Params4bit
from safetensors.torch import save_file
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import DTensor, Replicate, Shard, distribute_tensor
from torch.distributed.tensor.placement_types import _StridedShard
from transformers import AutoConfig, BitsAndBytesConfig, Qwen3Config, Qwen3ForCausalLM
from transformers.core_model_loading import Chunk, WeightConverter, WeightRenaming
from transformers.quantizers.quantizer_bnb_4bit import Bnb4BitHfQuantizer

import src.distributed.context_parallel.loading as cp_loading
import src.distributed.expert_parallel.master_weights as master_loading
import src.distributed.filesystem as filesystem
import src.distributed.loading.model_loading as loading
from src.distributed.context_parallel.wrapper import patch_model_for_cp
from src.distributed.expert_parallel.config import EPConfig
from src.distributed.expert_parallel.fp32_masters import fp32_master_param_keys
from src.distributed.expert_parallel.hub_conversion import resolve_loaded_conversion_steps
from src.distributed.expert_parallel.lazy_loader import ExpertFuser, build_family_key_mapping
from tests.common.gloo import run_gloo_ranks
from tests.common.tiny_models import module_with_weight_keys

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
        pc = SimpleNamespace(tp_size=2, data_parallel_size=1, fp32_non_ep_params=keep, max_concurrent_loading=None)
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


def _placement_rank(rank, path, stored_fp32):
    mesh = init_device_mesh("cpu", (2,), mesh_dim_names=("tp",))
    # Packed row order must differ from an ordinary contiguous Shard(0) oracle.
    expected = torch.arange(64).reshape(8, 8).float() / 16 + (1e-5 if stored_fp32 else 0)
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


@pytest.mark.parametrize("stored_fp32", (False, True), ids=("bf16_promotion", "fp32_replay"))
def test_tp_replay_preserves_each_placement_and_all_tied_owners(tmp_path, stored_fp32):
    expected = torch.arange(64).reshape(8, 8).float() / 16 + (1e-5 if stored_fp32 else 0)
    if not stored_fp32:
        expected = expected.bfloat16()
    save_file({"weight": expected}, str(tmp_path / "model.safetensors"))
    run_gloo_ranks(_placement_rank, 2, str(tmp_path), stored_fp32, pg_timeout=datetime.timedelta(seconds=30))


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


def test_bf16_stored_masters_skip_payload_reads_but_fp32_masters_are_replayed(tmp_path, monkeypatch):
    expected = {"bf16": torch.arange(8).bfloat16() / 16, "fp32": torch.arange(8).float() / 16 + 1e-5}
    save_file(expected, str(tmp_path / "model.safetensors"))
    model = nn.Module()
    model.config = Qwen3Config(tie_word_embeddings=False)
    for key, value in expected.items():
        model.register_parameter(key, nn.Parameter(value.bfloat16()))
    reads = []
    original_get = master_loading.StreamingCheckpointReader.get

    def get(reader, key):
        reads.append(key)
        return original_get(reader, key)

    monkeypatch.setattr(master_loading.StreamingCheckpointReader, "get", get)
    master_loading.restore_fp32_master_parameters(model, str(tmp_path), keep_non_ep=True, strict=True)
    assert reads == ["fp32"], "BF16 checkpoint values must not incur a second payload read"
    for key, value in expected.items():
        assert getattr(model, key).dtype == torch.float32
        assert torch.equal(getattr(model, key).detach(), value.float())


def test_bf16_replay_elision_cannot_exempt_a_missing_master_from_strict_coverage(tmp_path):
    model = nn.Module()
    model.config = Qwen3Config(tie_word_embeddings=False)
    model.present = nn.Parameter(torch.ones(4, dtype=torch.bfloat16))
    model.missing = nn.Parameter(torch.ones(4, dtype=torch.bfloat16))
    save_file({"present": model.present.detach()}, str(tmp_path / "model.safetensors"))
    with pytest.raises(RuntimeError, match="missing"):
        master_loading.restore_fp32_master_parameters(model, str(tmp_path), keep_non_ep=True, strict=True)


def test_fused_master_replay_preserves_live_device_instead_of_staging_device(tmp_path):
    model = nn.Module()
    model.experts = nn.Module()
    model.experts.down_proj = nn.Parameter(torch.zeros(2, 4, 3, dtype=torch.bfloat16))
    original = model.experts.down_proj
    expected = torch.arange(24).reshape(2, 4, 3).float() / 16 + 1e-5
    state = {f"experts.{rank}.down_proj.weight": expected[rank] for rank in range(2)}
    save_file(state, str(tmp_path / "model.safetensors"))
    tasks = [
        (
            "experts.down_proj",
            "down",
            {
                rank: {"down_proj.weight": (f"experts.{rank}.down_proj.weight", "model.safetensors")}
                for rank in range(2)
            },
        )
    ]
    with master_loading.StreamingCheckpointReader(str(tmp_path), state) as reader:
        # A distinct requested device catches relocation even on a CPU-only test runner.
        restored = ExpertFuser(0, 2).execute(
            tasks, model, str(tmp_path), torch.float32, "meta", reader=reader, preserve_parameters=True
        )
    assert restored == {"experts.down_proj"}
    assert model.experts.down_proj is original
    assert model.experts.down_proj.device.type == "cpu"
    assert model.experts.down_proj.dtype == torch.float32
    assert torch.equal(model.experts.down_proj.detach(), expected)


def test_eager_fp32_reread_uses_the_loaded_nested_rename_and_its_scope(tmp_path):
    vision_key = "model.vision_model.q_proj.weight"
    text_key = "model.language_model.wq_du.weight"
    disk_vision_key = "model.vision_model.wq_du.weight"
    model = module_with_weight_keys([vision_key, text_key]).to(torch.bfloat16)
    model.base_model_prefix = "model"
    model.config = SimpleNamespace(model_type="qwen3_moe")
    rename = WeightRenaming(source_patterns=r"^wq_du\.", target_patterns="q_proj.")
    rename.scope_prefix = "vision_model"
    model._weight_conversions = [rename]
    stored = {disk_vision_key: torch.tensor([1.00001]), text_key: torch.tensor([2.00001])}
    assert all(not torch.equal(value, value.bfloat16().float()) for value in stored.values())
    save_file(stored, str(tmp_path / "model.safetensors"))
    state = model.state_dict(keep_vars=True)
    for key, disk_key in ((vision_key, disk_vision_key), (text_key, text_key)):
        state[key].data.copy_(stored[disk_key])
    plain_mapping, _ = build_family_key_mapping(model, list(stored))
    assert plain_mapping[disk_vision_key] not in state, "the declared lazy family cannot do this nested rename"
    master_loading.restore_fp32_master_parameters(model, str(tmp_path), EPConfig(ep_size=1), keep_non_ep=True)
    restored = model.state_dict(keep_vars=True)
    for key, disk_key in ((vision_key, disk_vision_key), (text_key, text_key)):
        assert restored[key] is state[key]
        assert restored[key].dtype == torch.float32 and torch.equal(restored[key], stored[disk_key])


@pytest.mark.parametrize("deserialize_first", (False, True))
def test_prequantized_float_storage_keeps_codes_and_restores_plain_masters(tmp_path, deserialize_first):
    """The real HF broad deserializer also matches plain weights, which it leaves unchanged.

    Packed floating-storage Params4bit and their quantization statistics come from a real bnb
    serialization. The streamed master replay must leave them intact without dropping a separate
    recorded vendor rename or rounding the ordinary master's checkpoint value.
    """
    model = module_with_weight_keys(["norm.weight"]).to(torch.bfloat16)
    model.config = SimpleNamespace(model_type="qwen3")
    model.quantized = Linear4bit(64, 64, bias=False, quant_storage=torch.bfloat16, quant_type="nf4")
    model.quantized.weight = Params4bit(
        torch.randn(64, 64, dtype=torch.bfloat16),
        requires_grad=False,
        quant_storage=torch.bfloat16,
        quant_type="nf4",
    )
    model.quantized.to("cpu")
    packed = model.quantized.weight
    codes = packed.data.view(torch.uint8).clone()
    assert packed.bnb_quantized and packed.dtype == torch.bfloat16
    original = torch.tensor([1.00001])
    stored = {key: value.clone().contiguous() for key, value in model.quantized.state_dict().items()}
    assert any("quant_state" in key for key in stored), "the checkpoint must carry prequantized statistics"
    stored = {f"quantized.{key}": value for key, value in stored.items()}
    stored["vendor_norm.weight"] = original
    save_file(stored, str(tmp_path / "model.safetensors"))
    quantizer = Bnb4BitHfQuantizer(BitsAndBytesConfig(load_in_4bit=True), pre_quantized=True)
    (deserialize,) = quantizer.get_weight_conversions()
    rename = WeightRenaming(source_patterns=r"^vendor_norm\.", target_patterns="norm.")
    model._weight_conversions = [deserialize, rename] if deserialize_first else [rename, deserialize]
    model.norm.weight.data.copy_(original)
    assert not torch.equal(model.norm.weight.float(), original)
    # The actual op's identity branch is the premise for skipping only this storage conversion.
    assert deserialize.operations[0].convert({"weight": original})["weight"] is original

    master_loading.restore_fp32_master_parameters(model, str(tmp_path), keep_non_ep=True, strict=True)

    assert model.quantized.weight is packed and torch.equal(packed.data.view(torch.uint8), codes)
    assert model.norm.weight.dtype == torch.float32 and torch.equal(model.norm.weight, original)
    model._weight_conversions.append(
        WeightConverter(source_patterns=["a", "b"], target_patterns="c", operations=[Chunk(dim=0)])
    )
    with pytest.raises(ValueError, match="multi-source.*qwen3"):
        resolve_loaded_conversion_steps(model)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
