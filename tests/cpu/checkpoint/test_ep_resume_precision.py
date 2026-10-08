#!/usr/bin/env python
"""EP, EP+CP, EP+TP, TP-MoE and PP+EP checkpoint construction retains stored fp32 masters before any forward.

The real gathered writer, safetensors readers, expert fusion and EP/CP wrappers run on CPU without
any forward. Communication buffers and device placement are replaced, and CP's flash label is set
after CPU materialization. Fractional masters are off the bf16 grid, so widening a rounded load
cannot satisfy the exact-value oracle. Configured masters preserve checkpoint precision for both
fresh stages and resumes; only resumes require strict coverage of every configured master. A PP
stage's routers and experts load only at construction (the stage reload skips EP layers), so the
stage loader is held to the same oracle on each stage's own slice. The attention-TP loaders run on two
gloo ranks with a CPU mesh, so their sharded attention is compared whole.
"""

import datetime
import re
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
from accelerate import PartialState
from safetensors.torch import save_file
from torch.distributed.tensor import DTensor
from transformers import AutoConfig, Qwen3MoeForSequenceClassification

import src.distributed.context_parallel.loading as cp_loading
import src.distributed.expert_parallel.lazy_loader as lazy_loading
import src.distributed.expert_parallel.loading as eager_loading
import src.distributed.loading.model_loading as model_loading
import src.distributed.pipeline_parallel.lazy_loader as pp_lazy_loading
from src.checkpoint.config_export import LOADED_WEIGHTS_FROM_ATTR
from src.checkpoint.format import StreamingCheckpointReader, load_full_state_dict
from src.distributed.checkpoint.context import CheckpointLoadContext
from src.distributed.checkpoint.loader import CheckpointLoader, built_from_checkpoint, weights_read_from
from src.distributed.checkpoint.write import chunked_saveable_tensors, stream_gathered_checkpoint
from src.distributed.context_parallel.wrapper import UlyssesCPModelWrapper, patch_model_for_cp
from src.distributed.expert_parallel.base_layer import EPMoELayerBase
from src.distributed.expert_parallel.config import EPConfig
from src.distributed.expert_parallel.patching import MOE_LAYER_MAP, ep_claimed_blocks
from src.distributed.loading.model_loading import (
    _load_ep_cp_model,
    _load_ep_model,
    _load_ep_tp_model,
    _load_pp_stage_model,
    _load_tp_moe_model,
    _load_undistributed_model,
)
from src.distributed.mesh import create_dp_tp_mesh
from src.distributed.pipeline_parallel.stage import PP_STAGE_PARTITION_ATTR, build_pipeline_stage
from src.distributed.tensor_parallel.state_dict import tp_sharded_non_dtensor_suffixes
from src.trainers.mixins.ep_introspection import EpIntrospectionMixin
from tests.common.gloo import run_gloo_ranks
from tests.common.tiny_models import TINY_MOE_FAMILIES

PartialState()

FAMILIES = ("qwen3_moe", "gpt_oss")
PRECISION_CASES = (
    pytest.param(True, False, False, id="router"),
    pytest.param(False, True, False, id="experts"),
    pytest.param(False, False, True, id="non_ep_implies_router"),
    pytest.param(True, True, True, id="all_masters"),
)
TINY_OVERRIDES = {
    "vocab_size": 64,
    "hidden_size": 32,
    "intermediate_size": 32,
    "moe_intermediate_size": 16,
    "num_hidden_layers": 1,
    "num_experts": 4,
    "num_local_experts": 4,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "head_dim": 8,
}
MASTER_OFFSET = 1e-5
PP_SIZE = 2
# Every stage starts on a whole period of gpt-oss's alternating sliding/full layer_types.
PP_LAYERS = 4
_LAYER_KEY = re.compile(r"^model\.layers\.(\d+)\.(.+)$")


def _ownership(model):
    """Checkpoint scopes derived from the same family registry that constructs the EP wrappers."""
    routers, experts, blocks = [], [], []
    for path, block in ep_claimed_blocks(model):
        wrapper = MOE_LAYER_MAP[type(block).__name__]
        blocks.append(f"{path}.")
        if wrapper._ROUTER_ATTR is not None:
            routers.append(f"{path}.{wrapper._ROUTER_ATTR}.")
        container = wrapper._find_experts_container(block)
        container_path = next(name for name, child in block.named_modules() if child is container)
        experts.append(f"{path}.{container_path}.")
    assert blocks and routers and experts
    return tuple(routers), tuple(experts), tuple(blocks)


def _role(name, ownership):
    routers, experts, blocks = ownership
    if name.startswith(routers):
        return "router"
    if name.startswith(experts):
        return "experts"
    return "replicated_ep" if name.startswith(blocks) else "non_ep"


def _checkpoint(tmp_path, family, *, stored_fp32=True, tied=False, legacy_bin=False, layers=1):
    overrides = {**TINY_OVERRIDES, "tie_word_embeddings": tied, "num_hidden_layers": layers}
    model = TINY_MOE_FAMILIES[family].build(overrides).to(torch.bfloat16)
    model.config.dtype = torch.bfloat16
    ownership = _ownership(model)
    if stored_fp32:
        for param in model.parameters():
            param.data = param.data.float() + MASTER_OFFSET
    destination = tmp_path / family
    destination.mkdir()
    output_dir = str(destination)
    stream_gathered_checkpoint(
        model,
        chunked_saveable_tensors(model, retain=True),
        output_dir,
        is_save_rank=True,
        max_shard_size="64KB",
        keep_live_dtype=True,
    )
    state = load_full_state_dict(output_dir)
    if stored_fp32:
        assert state and all(
            tensor.dtype == torch.float32 and not torch.equal(tensor, tensor.to(torch.bfloat16).float())
            for tensor in state.values()
        ), "the checkpoint must contain precision a bf16 load cannot recover"
    else:
        assert state and all(tensor.dtype == torch.bfloat16 for tensor in state.values())
    if legacy_bin:
        torch.save(state, destination / "pytorch_model.bin")
        for path in destination.glob("*.safetensors"):
            path.unlink()
        (destination / "model.safetensors.index.json").unlink(missing_ok=True)
    return output_dir, state, ownership


def _patch_cpu_loading(monkeypatch):
    """No forward or distributed collective runs; the actual checkpoint materialization does."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(eager_loading, "move_model_to_local_device", lambda model: model)
    monkeypatch.setattr(model_loading, "create_ep_buffers", lambda model: 0)
    verified = model_loading.from_pretrained_verified

    def cpu_pretrained(model_class, source, **kwargs):
        kwargs["device_map"] = "cpu"
        return verified(model_class, source, **kwargs)

    monkeypatch.setattr(model_loading, "from_pretrained_verified", cpu_pretrained)
    for module in (eager_loading, lazy_loading, pp_lazy_loading):
        monkeypatch.setattr(module, "create_ep_buffers", lambda model: 0)
    monkeypatch.setattr(cp_loading, "patch_model_for_cp", _cpu_cp_wrap)


@pytest.fixture
def cpu_loading(monkeypatch):
    _patch_cpu_loading(monkeypatch)


def _cpu_cp_wrap(model, config):
    # HF materializes on CPU without flash; CP construction validates its own flash-only forward.
    # No attention forward runs in this weight-construction test.
    model.config._attn_implementation = "flash_attention_2"
    return patch_model_for_cp(model, config)


def _ep_config(*, fp32_router, fp32_experts, fp32_non_ep, managed_experts):
    return EPConfig(
        ep_size=1,
        world_size=1,
        gpus_per_node=1,
        fp32_router=fp32_router or fp32_non_ep,
        fp32_experts=fp32_experts,
        fsdp_shard_ep1_experts=managed_experts,
        use_grouped_gemm=False,
    )


def _construct(
    path,
    *,
    mode,
    lazy,
    fp32_router,
    fp32_experts,
    fp32_non_ep,
    preserve,
    managed_experts=False,
    model_class=None,
):
    ep_config = _ep_config(
        fp32_router=fp32_router, fp32_experts=fp32_experts, fp32_non_ep=fp32_non_ep, managed_experts=managed_experts
    )
    pc = SimpleNamespace(
        ep_size=1,
        cp_size=2,
        is_node_local_ep=True,
        ep_lazy_loading=lazy,
        max_concurrent_loading=1,
        fp32_non_ep_params=fp32_non_ep,
        create_ep_config=lambda: ep_config,
        create_cp_config=lambda: SimpleNamespace(cp_size=2, cp_rank=0, process_group=None),
    )
    loader = {"ep_cp": _load_ep_cp_model, "ep": _load_ep_model, "ep1": _load_undistributed_model}[mode]
    config = AutoConfig.from_pretrained(path)
    extra = {"local_rank": 0, "ep_wrappers": True} if mode == "ep1" else {}
    model = loader(
        path,
        pc,
        model_class or TINY_MOE_FAMILIES[config.model_type].load_class,
        {
            "config": config,
            "dtype": torch.bfloat16,
            "trust_remote_code": False,
            "preserve_checkpoint_precision": preserve,
        },
        **extra,
    )
    assert isinstance(model, UlyssesCPModelWrapper) == (mode == "ep_cp")
    return model


def _loaded_checkpoint_layout(model):
    """Inspect constructed parameters in their disk layout without a trainer restore or forward."""
    # Collective for the TP loaders' attention shards; every rank walks the same parameters.
    state = {
        name: tensor.full_tensor() if isinstance(tensor, DTensor) else tensor
        for name, tensor in model.named_parameters()
    }
    # Attention sinks stay plain tensors under TP, each rank holding its contiguous run of heads.
    for suffix in tp_sharded_non_dtensor_suffixes(model):
        for name in [name for name in state if name.endswith(suffix)]:
            shards = [torch.empty_like(state[name]) for _ in range(dist.get_world_size())]
            dist.all_gather(shards, state[name].detach().contiguous())
            state[name] = torch.cat(shards)
    inner = model.model if isinstance(model, UlyssesCPModelWrapper) else model
    layers = [(name, module) for name, module in inner.named_modules() if isinstance(module, EPMoELayerBase)]
    assert layers, "the real EP constructors must run"
    for path, layer in layers:
        state.update({f"{path}.{name}": tensor for name, tensor in layer.gather_expert_state_dict().items()})
    return state


def _assert_values(model, stored, ownership, *, router, experts, non_ep, managed_experts=False, local_names=None):
    """``local_names`` maps each checked checkpoint key to its name on ``model`` (all of them by default)."""
    actual = _loaded_checkpoint_layout(model)
    local_names = local_names or {name: name for name in stored}
    assert set(local_names.values()) <= set(actual), sorted(set(local_names.values()) - set(actual))
    enabled = {"router": router or non_ep, "experts": experts and not managed_experts, "non_ep": non_ep}
    checked_masters = set()
    for name, local in local_names.items():
        tensor = stored[name]
        role = _role(name, ownership)
        keeps_master = enabled.get(role, False)
        expected = tensor.float() if keeps_master else tensor.to(torch.bfloat16)
        assert actual[local].dtype == expected.dtype, f"{name}: stored fp32 dtype was lost at construction"
        assert torch.equal(actual[local].detach(), expected), f"{name}: stored fp32 precision was lost at construction"
        if keeps_master and tensor.dtype == torch.float32:
            checked_masters.add(role)
    expected_masters = {
        _role(name, ownership)
        for name in local_names
        if stored[name].dtype == torch.float32 and enabled.get(_role(name, ownership), False)
    }
    assert checked_masters == expected_masters


@pytest.mark.parametrize("family", FAMILIES)
@pytest.mark.parametrize("mode", ("ep", "ep_cp"))
@pytest.mark.parametrize("lazy", (False, True), ids=("eager", "lazy"))
@pytest.mark.parametrize("fp32_router,fp32_experts,fp32_non_ep", PRECISION_CASES)
def test_resume_construction_keeps_exact_fp32_masters(
    tmp_path, cpu_loading, family, mode, lazy, fp32_router, fp32_experts, fp32_non_ep
):
    path, stored, ownership = _checkpoint(tmp_path, family)
    model = _construct(
        path,
        mode=mode,
        lazy=lazy,
        fp32_router=fp32_router,
        fp32_experts=fp32_experts,
        fp32_non_ep=fp32_non_ep,
        preserve=True,
    )
    _assert_values(model, stored, ownership, router=fp32_router, experts=fp32_experts, non_ep=fp32_non_ep)


@pytest.mark.parametrize("family", FAMILIES)
@pytest.mark.parametrize("mode", ("ep", "ep_cp"))
@pytest.mark.parametrize("lazy", (False, True), ids=("eager", "lazy"))
@pytest.mark.parametrize("preserve", (False, True), ids=("fresh", "resume_without_fp32_flags"))
def test_fp32_sources_still_load_at_run_dtype_without_master_preservation(
    tmp_path, cpu_loading, family, mode, lazy, preserve
):
    path, stored, ownership = _checkpoint(tmp_path, family)
    model = _construct(
        path, mode=mode, lazy=lazy, fp32_router=False, fp32_experts=False, fp32_non_ep=False, preserve=preserve
    )
    _assert_values(model, stored, ownership, router=False, experts=False, non_ep=False)


@pytest.mark.parametrize("family", FAMILIES)
@pytest.mark.parametrize("lazy", (False, True), ids=("eager", "lazy"))
def test_a_fresh_stage_preserves_configured_fp32_checkpoint_masters(tmp_path, cpu_loading, family, lazy):
    path, stored, ownership = _checkpoint(tmp_path, family)
    model = _construct(
        path, mode="ep", lazy=lazy, fp32_router=True, fp32_experts=True, fp32_non_ep=True, preserve=False
    )
    _assert_values(model, stored, ownership, router=True, experts=True, non_ep=True)


@pytest.mark.parametrize("family", FAMILIES)
@pytest.mark.parametrize("lazy", (False, True), ids=("eager", "lazy"))
def test_a_bf16_checkpoint_keeps_exact_values_when_promoted_for_resume(tmp_path, cpu_loading, family, lazy):
    path, stored, ownership = _checkpoint(tmp_path, family, stored_fp32=False)
    for preserve in (False, True):
        model = _construct(
            path, mode="ep", lazy=lazy, fp32_router=True, fp32_experts=True, fp32_non_ep=True, preserve=preserve
        )
        _assert_values(model, stored, ownership, router=True, experts=True, non_ep=True)


@pytest.mark.parametrize("family", FAMILIES)
def test_eager_bf16_master_promotion_does_not_reread_expert_or_dense_payloads(
    tmp_path, cpu_loading, monkeypatch, family
):
    path, stored, ownership = _checkpoint(tmp_path, family, stored_fp32=False)

    def reject_second_read(reader, key):
        raise AssertionError(f"BF16 master payload reread: {key}")

    monkeypatch.setattr(StreamingCheckpointReader, "get", reject_second_read)
    model = _construct(
        path, mode="ep", lazy=False, fp32_router=True, fp32_experts=True, fp32_non_ep=True, preserve=True
    )
    _assert_values(model, stored, ownership, router=True, experts=True, non_ep=True)


@pytest.mark.parametrize("family", FAMILIES)
@pytest.mark.parametrize("preserve", (False, True), ids=("fresh_stage", "strict_resume"))
@pytest.mark.parametrize("fp32_router,fp32_experts,fp32_non_ep", PRECISION_CASES)
def test_undistributed_ep1_actual_route_keeps_configured_masters(
    tmp_path, cpu_loading, family, preserve, fp32_router, fp32_experts, fp32_non_ep
):
    path, stored, ownership = _checkpoint(tmp_path, family)
    model = _construct(
        path,
        mode="ep1",
        lazy=False,
        fp32_router=fp32_router,
        fp32_experts=fp32_experts,
        fp32_non_ep=fp32_non_ep,
        preserve=preserve,
    )
    _assert_values(model, stored, ownership, router=fp32_router, experts=fp32_experts, non_ep=fp32_non_ep)


@pytest.mark.parametrize("family", FAMILIES)
@pytest.mark.parametrize("preserve", (False, True), ids=("fresh_stage", "strict_resume"))
@pytest.mark.parametrize("router", (False, True))
def test_ep1_fsdp_managed_experts_do_not_gain_fp32_masters(tmp_path, cpu_loading, family, preserve, router):
    path, stored, ownership = _checkpoint(tmp_path, family)
    model = _construct(
        path,
        mode="ep1",
        lazy=False,
        fp32_router=router,
        fp32_experts=True,
        fp32_non_ep=False,
        preserve=preserve,
        managed_experts=True,
    )
    _assert_values(model, stored, ownership, router=router, experts=True, non_ep=False, managed_experts=True)


@pytest.mark.parametrize("lazy", (False, True), ids=("eager", "lazy"))
def test_tied_fp32_non_ep_master_survives_a_checkpoint_without_the_shadow_key(tmp_path, cpu_loading, lazy):
    path, stored, ownership = _checkpoint(tmp_path, "qwen3_moe", tied=True)
    assert "model.embed_tokens.weight" in stored and "lm_head.weight" not in stored
    model = _construct(
        path, mode="ep", lazy=lazy, fp32_router=False, fp32_experts=False, fp32_non_ep=True, preserve=True
    )
    _assert_values(model, stored, ownership, router=False, experts=False, non_ep=True)
    assert model.lm_head.weight is model.model.embed_tokens.weight


@pytest.mark.parametrize("lazy", (False, True), ids=("eager", "lazy"))
def test_explicitly_saved_distinct_head_is_not_retied_by_precision_restore(tmp_path, cpu_loading, lazy):
    path, stored, ownership = _checkpoint(tmp_path, "qwen3_moe")
    assert not torch.equal(stored["model.embed_tokens.weight"], stored["lm_head.weight"])
    config = AutoConfig.from_pretrained(path)
    config.tie_word_embeddings = True
    config.save_pretrained(path)
    model = _construct(
        path, mode="ep", lazy=lazy, fp32_router=False, fp32_experts=False, fp32_non_ep=True, preserve=True
    )
    _assert_values(model, stored, ownership, router=False, experts=False, non_ep=True)
    assert model.lm_head.weight is not model.model.embed_tokens.weight


@pytest.mark.parametrize("family", FAMILIES)
@pytest.mark.parametrize("lazy", (False, True), ids=("eager", "lazy_fallback"))
def test_legacy_bin_checkpoint_keeps_exact_fp32_masters(tmp_path, cpu_loading, family, lazy):
    path, stored, ownership = _checkpoint(tmp_path, family, legacy_bin=True)
    model = _construct(
        path, mode="ep", lazy=lazy, fp32_router=True, fp32_experts=True, fp32_non_ep=True, preserve=True
    )
    _assert_values(model, stored, ownership, router=True, experts=True, non_ep=True)


@pytest.mark.parametrize("lazy", (False, True), ids=("eager", "lazy"))
@pytest.mark.parametrize("misnamed", (False, True), ids=("missing_head", "misnamed_head"))
@pytest.mark.parametrize("class_ignore", (False, True), ids=("ordinary_class", "class_excuses_head"))
@pytest.mark.parametrize("env_override", (False, True), ids=("strict_env", "env_excuses_missing"))
def test_resume_missing_selected_master_is_never_randomly_initialized(
    tmp_path, cpu_loading, monkeypatch, lazy, misnamed, class_ignore, env_override
):
    path, stored, _ = _checkpoint(tmp_path, "qwen3_moe")
    head = stored.pop("lm_head.weight")
    if misnamed:
        stored["score.misnamed_weight"] = head
    for shard in Path(path).glob("*.safetensors"):
        shard.unlink()
    (Path(path) / "model.safetensors.index.json").unlink(missing_ok=True)
    save_file(stored, str(Path(path) / "model.safetensors"))
    assert "score.weight" not in load_full_state_dict(path)
    monkeypatch.setattr(
        Qwen3MoeForSequenceClassification,
        "_keys_to_ignore_on_load_missing",
        [r"^score\.weight$"] if class_ignore else None,
    )
    monkeypatch.setenv("HALO_ALLOW_MISSING_CHECKPOINT_KEYS", "1" if env_override else "0")
    kwargs = {
        "mode": "ep",
        "lazy": lazy,
        "fp32_router": False,
        "fp32_experts": False,
        "fp32_non_ep": True,
        "model_class": Qwen3MoeForSequenceClassification,
    }
    # A fresh classification load is allowed to add a trainable head; a saved-master resume is not.
    fresh = _construct(path, preserve=False, **kwargs)
    assert not fresh.score.weight.is_meta and torch.isfinite(fresh.score.weight).all()
    with pytest.raises(RuntimeError, match=r"FP32.master.*score\.weight"):
        _construct(path, preserve=True, **kwargs)


def _construct_pp_stage(
    path, pp_rank, *, fp32_router, fp32_experts, fp32_non_ep, preserve, managed_experts=False, model_class=None
):
    ep_config = _ep_config(
        fp32_router=fp32_router, fp32_experts=fp32_experts, fp32_non_ep=fp32_non_ep, managed_experts=managed_experts
    )
    pc = SimpleNamespace(
        pp_rank=pp_rank,
        pp_size=PP_SIZE,
        pp_split=None,
        ep_size=1,
        data_parallel_size=1,
        needs_ep_wrappers=True,
        fp32_non_ep_params=fp32_non_ep,
        create_ep_config=lambda: ep_config,
    )
    config = AutoConfig.from_pretrained(path)
    common = {
        "config": config,
        "dtype": torch.bfloat16,
        "trust_remote_code": False,
        "preserve_checkpoint_precision": preserve,
    }
    return _load_pp_stage_model(
        path, pc, model_class or TINY_MOE_FAMILIES[config.model_type].load_class, common, is_moe=True
    )


def _stage_names(stage, stored, pp_rank):
    """Checkpoint key → stage key for what ``pp_rank`` holds: its layers re-based to 0, plus the
    embeddings, final norm and head every stage materializes before the stage build drops them."""
    lo, hi = getattr(stage, PP_STAGE_PARTITION_ATTR)[pp_rank]
    names = {}
    for name in stored:
        match = _LAYER_KEY.match(name)
        if match is None:
            names[name] = name
        elif lo <= int(match[1]) < hi:
            names[name] = f"model.layers.{int(match[1]) - lo}.{match[2]}"
    assert any(_LAYER_KEY.match(name) for name in names), f"stage {pp_rank} holds no decoder layer"
    return names


@pytest.mark.parametrize("family", FAMILIES)
@pytest.mark.parametrize("pp_rank", range(PP_SIZE))
@pytest.mark.parametrize("preserve", (False, True), ids=("fresh_stage", "strict_resume"))
@pytest.mark.parametrize("fp32_router,fp32_experts,fp32_non_ep", PRECISION_CASES)
def test_pp_stage_construction_keeps_exact_fp32_masters(
    tmp_path, cpu_loading, family, pp_rank, preserve, fp32_router, fp32_experts, fp32_non_ep
):
    path, stored, ownership = _checkpoint(tmp_path, family, layers=PP_LAYERS)
    flags = {"router": fp32_router, "experts": fp32_experts, "non_ep": fp32_non_ep}
    stage = _construct_pp_stage(
        path, pp_rank, fp32_router=fp32_router, fp32_experts=fp32_experts, fp32_non_ep=fp32_non_ep, preserve=preserve
    )
    _assert_values(stage, stored, ownership, **flags, local_names=_stage_names(stage, stored, pp_rank))


@pytest.mark.parametrize("family", FAMILIES)
@pytest.mark.parametrize("pp_rank", range(PP_SIZE))
def test_pp_stage_without_fp32_flags_loads_at_run_dtype(tmp_path, cpu_loading, family, pp_rank):
    path, stored, ownership = _checkpoint(tmp_path, family, layers=PP_LAYERS)
    stage = _construct_pp_stage(path, pp_rank, fp32_router=False, fp32_experts=False, fp32_non_ep=False, preserve=True)
    _assert_values(
        stage,
        stored,
        ownership,
        router=False,
        experts=False,
        non_ep=False,
        local_names=_stage_names(stage, stored, pp_rank),
    )


@pytest.mark.parametrize("family", FAMILIES)
def test_pp_stage_fsdp_managed_experts_do_not_gain_fp32_masters(tmp_path, cpu_loading, family):
    path, stored, ownership = _checkpoint(tmp_path, family, layers=PP_LAYERS)
    stage = _construct_pp_stage(
        path, 1, fp32_router=True, fp32_experts=True, fp32_non_ep=False, preserve=True, managed_experts=True
    )
    _assert_values(
        stage,
        stored,
        ownership,
        router=True,
        experts=True,
        non_ep=False,
        managed_experts=True,
        local_names=_stage_names(stage, stored, 1),
    )


@pytest.mark.parametrize("family", FAMILIES)
@pytest.mark.parametrize("pp_rank", range(PP_SIZE))
@pytest.mark.parametrize("fp32_router,fp32_experts,fp32_non_ep", PRECISION_CASES)
def test_pp_resume_reload_of_a_constructed_stage_writes_exactly_its_values(
    tmp_path, cpu_loading, family, pp_rank, fp32_router, fp32_experts, fp32_non_ep
):
    """A Path-B PP resume skips the stage reload because the stage was built from the checkpoint. That
    is lossless only if the reload would write back exactly what construction read: the same values
    at the same dtypes, configured FP32 masters included. The stage's checkpoint-mapped state is
    zeroed first, so the comparison holds only if the reload really wrote every tensor."""
    path, _stored, _ = _checkpoint(tmp_path, family, layers=PP_LAYERS)
    model = _construct_pp_stage(
        path, pp_rank, fp32_router=fp32_router, fp32_experts=fp32_experts, fp32_non_ep=fp32_non_ep, preserve=True
    )
    setattr(model, LOADED_WEIGHTS_FROM_ATTR, path)  # load_distributed_model's stamp
    stage = build_pipeline_stage(model, pp_rank, PP_SIZE)
    assert built_from_checkpoint(weights_read_from(stage), path), "the stage must carry the construction source"
    if fp32_non_ep:
        # The trainer's own upcast runs between construction and resume; a master construction read
        # at bf16 would reach the comparison as rounded FP32.
        ep_param_ids = {id(param) for _, layer in stage.ep_moe_layers() for param in layer.parameters()}
        trainer = SimpleNamespace(_top_level_model=lambda: stage, _get_ep_param_ids=lambda: ep_param_ids)
        EpIntrospectionMixin._upcast_non_ep_params_to_fp32(trainer)
    constructed = {name: tensor.detach().clone() for name, tensor in stage.state_dict().items()}
    mapped = set(stage.checkpoint_name_map().values())
    assert mapped, "the reload must cover the stage's non-expert state"
    with torch.no_grad():
        for name, tensor in stage.state_dict(keep_vars=True).items():
            if name in mapped:
                tensor.zero_()

    context = CheckpointLoadContext(
        model=stage,
        optimizer=None,
        lr_scheduler=None,
        parallelism_config=SimpleNamespace(pp_size=PP_SIZE, pp_rank=pp_rank, tp_size=1, cp_size=1),
        is_pp_mode=True,
        is_cp_mode=False,
        is_tp_mode=False,
        has_ep_layers=True,
        fsdp_wrapped=False,
        tp_rank=0,
        tp_size=1,
        super_load_from_checkpoint=None,
        super_load_optimizer_and_scheduler=None,
    )
    # The read the resume skips; a best-model load takes it unconditionally.
    CheckpointLoader(context).load_model(path, for_best_model=True)

    reloaded = stage.state_dict()
    assert reloaded.keys() == constructed.keys()
    for name, tensor in constructed.items():
        assert reloaded[name].dtype == tensor.dtype, f"{name}: the reload would change the constructed dtype"
        assert torch.equal(reloaded[name], tensor), f"{name}: the reload would change the constructed value"


def test_pp_stage_resume_missing_selected_master_is_never_randomly_initialized(tmp_path, cpu_loading):
    path, stored, _ = _checkpoint(tmp_path, "qwen3_moe", layers=PP_LAYERS)
    stored.pop("lm_head.weight")
    for shard in Path(path).glob("*.safetensors"):
        shard.unlink()
    (Path(path) / "model.safetensors.index.json").unlink(missing_ok=True)
    save_file(stored, str(Path(path) / "model.safetensors"))
    kwargs = {
        "fp32_router": False,
        "fp32_experts": False,
        "fp32_non_ep": True,
        "model_class": Qwen3MoeForSequenceClassification,
    }
    # A fresh classification stage may add its trainable head; a saved-master resume may not.
    fresh = _construct_pp_stage(path, PP_SIZE - 1, preserve=False, **kwargs)
    assert not fresh.score.weight.is_meta and torch.isfinite(fresh.score.weight).all()
    with pytest.raises(RuntimeError, match=r"FP32.master.*score\.weight"):
        _construct_pp_stage(path, PP_SIZE - 1, preserve=True, **kwargs)


# The attention-TP loaders: EP+TP (lazy safetensors or sequential from_pretrained) and TP on a MoE.
TP_MODES = ("ep_tp_eager", "ep_tp_lazy", "tp_moe")
TP_SIZE = 2


def _patch_cpu_tp_loading(monkeypatch):
    """:func:`_patch_cpu_loading` for the TP loaders on gloo ranks, with the TP mesh laid on CPU devices."""
    _patch_cpu_loading(monkeypatch)
    monkeypatch.setattr(
        model_loading,
        "create_dp_tp_mesh",
        lambda tp_size, dp_size: create_dp_tp_mesh(tp_size, dp_size, device_type="cpu"),
    )


def _construct_tp(path, mode, *, fp32_router, fp32_experts, fp32_non_ep, preserve, model_class=None):
    ep_config = _ep_config(
        fp32_router=fp32_router, fp32_experts=fp32_experts, fp32_non_ep=fp32_non_ep, managed_experts=False
    )
    pc = SimpleNamespace(
        ep_size=1,
        tp_size=TP_SIZE,
        data_parallel_size=1,
        ep_lazy_loading=mode == "ep_tp_lazy",
        max_concurrent_loading=1,
        fp32_non_ep_params=fp32_non_ep,
        needs_ep_wrappers=True,
        create_ep_config=lambda: ep_config,
    )
    config = AutoConfig.from_pretrained(path)
    common = {
        "config": config,
        "dtype": torch.bfloat16,
        "trust_remote_code": False,
        "preserve_checkpoint_precision": preserve,
    }
    load_class = model_class or TINY_MOE_FAMILIES[config.model_type].load_class
    if mode == "tp_moe":
        model = _load_tp_moe_model(path, pc, load_class, common)
    else:
        model = _load_ep_tp_model(path, pc, load_class, common)
    assert any(isinstance(param, DTensor) for param in model.parameters()), "attention must be TP-sharded"
    return model


def _tp_masters_rank(rank, path, stored, ownership, mode):
    with pytest.MonkeyPatch.context() as monkeypatch:
        _patch_cpu_tp_loading(monkeypatch)
        for case in PRECISION_CASES:
            fp32_router, fp32_experts, fp32_non_ep = case.values
            flags = {"fp32_router": fp32_router, "fp32_experts": fp32_experts, "fp32_non_ep": fp32_non_ep}
            model = _construct_tp(path, mode, **flags, preserve=True)
            _assert_values(model, stored, ownership, router=fp32_router, experts=fp32_experts, non_ep=fp32_non_ep)


@pytest.mark.parametrize("family", FAMILIES)
@pytest.mark.parametrize("mode", TP_MODES)
def test_tp_loader_resume_construction_keeps_exact_fp32_masters(tmp_path, family, mode):
    path, stored, ownership = _checkpoint(tmp_path, family)
    run_gloo_ranks(_tp_masters_rank, TP_SIZE, path, stored, ownership, mode, pg_timeout=datetime.timedelta(seconds=60))


def _tp_missing_master_rank(rank, path, mode):
    with pytest.MonkeyPatch.context() as monkeypatch:
        _patch_cpu_tp_loading(monkeypatch)
        kwargs = {
            "fp32_router": False,
            "fp32_experts": False,
            "fp32_non_ep": True,
            "model_class": Qwen3MoeForSequenceClassification,
        }
        fresh = _construct_tp(path, mode, preserve=False, **kwargs)
        assert not fresh.score.weight.is_meta and torch.isfinite(fresh.score.weight).all()
        with pytest.raises(RuntimeError, match=r"FP32.master.*score\.weight"):
            _construct_tp(path, mode, preserve=True, **kwargs)


@pytest.mark.parametrize("mode", TP_MODES)
def test_tp_loader_resume_missing_selected_master_is_never_randomly_initialized(tmp_path, mode):
    path, stored, _ = _checkpoint(tmp_path, "qwen3_moe")
    stored.pop("lm_head.weight")
    for shard in Path(path).glob("*.safetensors"):
        shard.unlink()
    (Path(path) / "model.safetensors.index.json").unlink(missing_ok=True)
    save_file(stored, str(Path(path) / "model.safetensors"))
    run_gloo_ranks(_tp_missing_master_rank, TP_SIZE, path, mode, pg_timeout=datetime.timedelta(seconds=60))


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
