#!/usr/bin/env python
"""EP and EP+CP checkpoint construction retains stored fp32 masters before any forward.

The real gathered writer, safetensors readers, expert fusion and EP/CP wrappers run on CPU without
any forward. Communication buffers and device placement are replaced, and CP's flash label is set
after CPU materialization. Fractional masters are off the bf16 grid, so widening a rounded load
cannot satisfy the exact-value oracle. Configured masters preserve checkpoint precision for both
fresh stages and resumes; only resumes require strict coverage of every configured master.
"""

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from accelerate import PartialState
from safetensors.torch import save_file
from transformers import AutoConfig, Qwen3MoeForSequenceClassification

import src.distributed.context_parallel.loading as cp_loading
import src.distributed.expert_parallel.lazy_loader as lazy_loading
import src.distributed.expert_parallel.loading as eager_loading
import src.distributed.loading.model_loading as model_loading
from src.checkpoint.format import load_full_state_dict
from src.distributed.checkpoint.write import chunked_saveable_tensors, stream_gathered_checkpoint
from src.distributed.context_parallel.wrapper import UlyssesCPModelWrapper, patch_model_for_cp
from src.distributed.expert_parallel.base_layer import EPMoELayerBase
from src.distributed.expert_parallel.config import EPConfig
from src.distributed.expert_parallel.patching import MOE_LAYER_MAP, ep_claimed_blocks
from src.distributed.loading.model_loading import _load_ep_cp_model, _load_ep_model, _load_undistributed_model
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


def _checkpoint(tmp_path, family, *, stored_fp32=True, tied=False, legacy_bin=False):
    model = TINY_MOE_FAMILIES[family].build({**TINY_OVERRIDES, "tie_word_embeddings": tied}).to(torch.bfloat16)
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


@pytest.fixture
def cpu_loading(monkeypatch):
    """No forward or distributed collective runs; the actual checkpoint materialization does."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(eager_loading, "move_model_to_local_device", lambda model: model)
    monkeypatch.setattr(cp_loading, "move_model_to_local_device", lambda model: model)
    monkeypatch.setattr(model_loading, "create_ep_buffers", lambda model: 0)
    verified = model_loading.from_pretrained_verified

    def cpu_pretrained(model_class, source, **kwargs):
        kwargs["device_map"] = "cpu"
        return verified(model_class, source, **kwargs)

    monkeypatch.setattr(model_loading, "from_pretrained_verified", cpu_pretrained)
    for module in (eager_loading, lazy_loading):
        monkeypatch.setattr(module, "create_ep_buffers", lambda model: 0)
    monkeypatch.setattr(cp_loading, "patch_model_for_cp", _cpu_cp_wrap)


def _cpu_cp_wrap(model, config):
    # HF materializes on CPU without flash; CP construction validates its own flash-only forward.
    # No attention forward runs in this weight-construction test.
    model.config._attn_implementation = "flash_attention_2"
    return patch_model_for_cp(model, config)


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
    ep_config = EPConfig(
        ep_size=1,
        world_size=1,
        gpus_per_node=1,
        fp32_router=fp32_router or fp32_non_ep,
        fp32_experts=fp32_experts,
        fsdp_shard_ep1_experts=managed_experts,
        use_grouped_gemm=False,
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
    state = dict(model.named_parameters())
    inner = model.model if isinstance(model, UlyssesCPModelWrapper) else model
    layers = [(name, module) for name, module in inner.named_modules() if isinstance(module, EPMoELayerBase)]
    assert layers, "the real EP constructors must run"
    for path, layer in layers:
        state.update({f"{path}.{name}": tensor for name, tensor in layer.gather_expert_state_dict().items()})
    return state


def _assert_values(model, stored, ownership, *, router, experts, non_ep, preserve, managed_experts=False):
    actual = _loaded_checkpoint_layout(model)
    assert set(stored) <= set(actual), sorted(set(stored) - set(actual))
    enabled = {"router": router or non_ep, "experts": experts and not managed_experts, "non_ep": non_ep}
    checked_masters = set()
    for name, tensor in stored.items():
        role = _role(name, ownership)
        keeps_master = enabled.get(role, False)
        expected = tensor.float() if keeps_master else tensor.to(torch.bfloat16)
        assert actual[name].dtype == expected.dtype, f"{name}: stored fp32 dtype was lost at construction"
        assert torch.equal(actual[name].detach(), expected), f"{name}: stored fp32 precision was lost at construction"
        if keeps_master and tensor.dtype == torch.float32:
            checked_masters.add(role)
    expected_masters = {
        _role(name, ownership)
        for name, tensor in stored.items()
        if tensor.dtype == torch.float32 and enabled.get(_role(name, ownership), False)
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
    _assert_values(
        model, stored, ownership, router=fp32_router, experts=fp32_experts, non_ep=fp32_non_ep, preserve=True
    )


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
    _assert_values(model, stored, ownership, router=False, experts=False, non_ep=False, preserve=preserve)


@pytest.mark.parametrize("family", FAMILIES)
@pytest.mark.parametrize("lazy", (False, True), ids=("eager", "lazy"))
def test_a_fresh_stage_preserves_configured_fp32_checkpoint_masters(tmp_path, cpu_loading, family, lazy):
    path, stored, ownership = _checkpoint(tmp_path, family)
    model = _construct(
        path, mode="ep", lazy=lazy, fp32_router=True, fp32_experts=True, fp32_non_ep=True, preserve=False
    )
    _assert_values(model, stored, ownership, router=True, experts=True, non_ep=True, preserve=False)


@pytest.mark.parametrize("family", FAMILIES)
@pytest.mark.parametrize("lazy", (False, True), ids=("eager", "lazy"))
def test_a_bf16_checkpoint_keeps_exact_values_when_promoted_for_resume(tmp_path, cpu_loading, family, lazy):
    path, stored, ownership = _checkpoint(tmp_path, family, stored_fp32=False)
    for preserve in (False, True):
        model = _construct(
            path, mode="ep", lazy=lazy, fp32_router=True, fp32_experts=True, fp32_non_ep=True, preserve=preserve
        )
        _assert_values(model, stored, ownership, router=True, experts=True, non_ep=True, preserve=preserve)


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
    _assert_values(
        model, stored, ownership, router=fp32_router, experts=fp32_experts, non_ep=fp32_non_ep, preserve=preserve
    )


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
    _assert_values(
        model, stored, ownership, router=router, experts=True, non_ep=False, preserve=preserve, managed_experts=True
    )


@pytest.mark.parametrize("lazy", (False, True), ids=("eager", "lazy"))
def test_tied_fp32_non_ep_master_survives_a_checkpoint_without_the_shadow_key(tmp_path, cpu_loading, lazy):
    path, stored, ownership = _checkpoint(tmp_path, "qwen3_moe", tied=True)
    assert "model.embed_tokens.weight" in stored and "lm_head.weight" not in stored
    model = _construct(
        path, mode="ep", lazy=lazy, fp32_router=False, fp32_experts=False, fp32_non_ep=True, preserve=True
    )
    _assert_values(model, stored, ownership, router=False, experts=False, non_ep=True, preserve=True)
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
    _assert_values(model, stored, ownership, router=False, experts=False, non_ep=True, preserve=True)
    assert model.lm_head.weight is not model.model.embed_tokens.weight


@pytest.mark.parametrize("family", FAMILIES)
@pytest.mark.parametrize("lazy", (False, True), ids=("eager", "lazy_fallback"))
def test_legacy_bin_checkpoint_keeps_exact_fp32_masters(tmp_path, cpu_loading, family, lazy):
    path, stored, ownership = _checkpoint(tmp_path, family, legacy_bin=True)
    model = _construct(
        path, mode="ep", lazy=lazy, fp32_router=True, fp32_experts=True, fp32_non_ep=True, preserve=True
    )
    _assert_values(model, stored, ownership, router=True, experts=True, non_ep=True, preserve=True)


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


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
