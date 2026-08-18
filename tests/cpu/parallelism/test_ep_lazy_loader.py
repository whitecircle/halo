#!/usr/bin/env python
"""Unit tests for the EP lazy loader planner logic (CPU-only, no GPU required)."""

import json
import re
from types import SimpleNamespace

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import save_file

import src.distributed.expert_parallel.lazy_loader as lazy_loader  # noqa: E402
from src.distributed.expert_parallel.lazy_loader import (  # noqa: E402
    CheckpointFormat,
    EPWeightPlanner,
    ExpertFuser,
    SafetensorsWeightLoader,
    WeightAction,
    WeightPlan,
    _check_fused_experts_mapped,
    assign_tensor_to_model,
    build_family_key_mapping,
    lazy_loader_supports_checkpoint,
    resolve_safetensors_index,
)
from src.models.loading.lazy_safetensors.weights import has_safetensors_checkpoint  # noqa: E402
from tests.common.models import TINY_GPTOSS_CONFIG  # noqa: E402


class MockEPConfig:
    """Minimal mock EPConfig for planner tests."""

    def __init__(self, start=4, end=8, total=16):
        self.expert_start_idx = start
        self.expert_end_idx = end
        self.experts_per_rank = end - start
        self.num_experts = total


def _fused_expert_weight_map():
    """Weight map with fused 3D expert keys (GptOss-style)."""
    wm = {}
    for layer in range(2):
        wm[f"model.layers.{layer}.mlp.experts.gate_up_proj"] = "shard-00001.safetensors"
        wm[f"model.layers.{layer}.mlp.experts.down_proj"] = "shard-00001.safetensors"
        wm[f"model.layers.{layer}.mlp.router.weight"] = "shard-00001.safetensors"
        wm[f"model.layers.{layer}.self_attn.q_proj.weight"] = "shard-00002.safetensors"
        wm[f"model.layers.{layer}.self_attn.k_proj.weight"] = "shard-00002.safetensors"
        wm[f"model.layers.{layer}.input_layernorm.weight"] = "shard-00002.safetensors"
    wm["model.embed_tokens.weight"] = "shard-00001.safetensors"
    wm["lm_head.weight"] = "shard-00002.safetensors"
    return wm


def _individual_expert_weight_map():
    """Weight map with individual expert keys (Qwen3-style ModuleList)."""
    wm = {}
    for layer in range(2):
        for expert in range(16):
            wm[f"model.layers.{layer}.block_sparse_moe.experts.{expert}.gate_proj.weight"] = "shard-00001.safetensors"
            wm[f"model.layers.{layer}.block_sparse_moe.experts.{expert}.up_proj.weight"] = "shard-00001.safetensors"
            wm[f"model.layers.{layer}.block_sparse_moe.experts.{expert}.down_proj.weight"] = "shard-00001.safetensors"
        wm[f"model.layers.{layer}.block_sparse_moe.gate.weight"] = "shard-00001.safetensors"
        wm[f"model.layers.{layer}.self_attn.q_proj.weight"] = "shard-00002.safetensors"
    return wm


def test_fused_experts_classified_as_shard():
    cfg = MockEPConfig(4, 8, 16)
    planner = EPWeightPlanner(cfg)
    wm = _fused_expert_weight_map()
    identity = {k: k for k in wm}
    model_keys = set(wm.keys())

    plans = planner.build(wm, identity, model_keys)
    expert_plans = [p for p in plans if p.action == WeightAction.EXPERT_SHARD]

    assert len(expert_plans) == 4
    for p in expert_plans:
        assert p.shard_dim == 0
        assert p.shard_start == 4
        assert p.shard_end == 8
        assert "experts." in p.disk_key


def test_non_expert_classified_as_replicate():
    cfg = MockEPConfig(4, 8, 16)
    planner = EPWeightPlanner(cfg)
    wm = _fused_expert_weight_map()
    identity = {k: k for k in wm}
    model_keys = set(wm.keys())

    plans = planner.build(wm, identity, model_keys)
    replicate_keys = {p.disk_key for p in plans if p.action == WeightAction.REPLICATE}

    assert "model.embed_tokens.weight" in replicate_keys
    assert "lm_head.weight" in replicate_keys
    assert any("self_attn" in k for k in replicate_keys)
    assert any("router" in k for k in replicate_keys)


def test_individual_experts_local_replicate_remote_ignore():
    cfg = MockEPConfig(4, 8, 16)
    planner = EPWeightPlanner(cfg)
    wm = _individual_expert_weight_map()
    identity = {k: k for k in wm}
    model_keys = set(wm.keys())

    plans = planner.build(wm, identity, model_keys)

    for p in plans:
        if ".experts." in p.disk_key and "gate.weight" not in p.disk_key:
            m = re.search(r"\.experts\.(\d+)\.", p.disk_key)
            assert m
            idx = int(m.group(1))
            if 4 <= idx < 8:
                assert p.action == WeightAction.REPLICATE, f"Expert {idx} should be REPLICATE"
            else:
                assert p.action == WeightAction.IGNORE, f"Expert {idx} should be IGNORE"


def test_resolve_sharded_index(tmp_path):
    wm = {"param.a": "shard-001.safetensors", "param.b": "shard-002.safetensors"}
    index = {"metadata": {}, "weight_map": wm}
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps(index))

    resolved_map, shard_files = resolve_safetensors_index(str(tmp_path))
    assert resolved_map == wm
    assert sorted(shard_files) == ["shard-001.safetensors", "shard-002.safetensors"]


def test_resolve_single_file(tmp_path):
    tensors = {"layer.weight": torch.randn(4, 4), "layer.bias": torch.randn(4)}
    save_file(tensors, str(tmp_path / "model.safetensors"))

    resolved_map, shard_files = resolve_safetensors_index(str(tmp_path))
    assert set(resolved_map.keys()) == {"layer.weight", "layer.bias"}
    assert shard_files == ["model.safetensors"]


def test_resolve_missing_raises(tmp_path):
    try:
        resolve_safetensors_index(str(tmp_path))
        raise AssertionError("Should have raised")
    except FileNotFoundError:
        pass


def test_expert_shard_slicing(tmp_path):
    """Verify that EXPERT_SHARD only materializes the requested slice."""
    full_weight = torch.randn(16, 128, 256)
    save_file({"experts.gate_up_proj": full_weight}, str(tmp_path / "model.safetensors"))

    plan = WeightPlan(
        action=WeightAction.EXPERT_SHARD,
        shard_file="model.safetensors",
        disk_key="experts.gate_up_proj",
        model_key="experts.gate_up_proj",
        shard_dim=0,
        shard_start=4,
        shard_end=8,
    )

    loader = SafetensorsWeightLoader(str(tmp_path), ["model.safetensors"], device="cpu")
    loader._open()
    try:
        tensor = loader._materialize(plan)
        assert tensor.shape == (4, 128, 256)
        assert torch.allclose(tensor, full_weight[4:8])
    finally:
        loader._close()


def test_replicate_loads_full(tmp_path):
    """Verify that REPLICATE loads the full tensor."""
    weight = torch.randn(512, 512)
    save_file({"attn.weight": weight}, str(tmp_path / "model.safetensors"))

    plan = WeightPlan(
        action=WeightAction.REPLICATE,
        shard_file="model.safetensors",
        disk_key="attn.weight",
        model_key="attn.weight",
    )

    loader = SafetensorsWeightLoader(str(tmp_path), ["model.safetensors"], device="cpu")
    loader._open()
    try:
        tensor = loader._materialize(plan)
        assert tensor.shape == (512, 512)
        assert torch.allclose(tensor, weight)
    finally:
        loader._close()


def test_assign_replaces_parameter():
    import torch.nn as nn

    model = nn.Sequential(nn.Linear(4, 8))
    real_weight = torch.randn(8, 4)
    assign_tensor_to_model(model, "0.weight", real_weight)
    assert torch.equal(model[0].weight.data, real_weight)


def test_assign_nested_module():
    import torch.nn as nn

    class Inner(nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = nn.Linear(4, 8)

    class Outer(nn.Module):
        def __init__(self):
            super().__init__()
            self.layer = Inner()

    model = Outer()
    real_weight = torch.randn(8, 4)
    assign_tensor_to_model(model, "layer.proj.weight", real_weight)
    assert torch.equal(model.layer.proj.weight.data, real_weight)


def test_detect_fused_format():
    assert CheckpointFormat.detect(_fused_expert_weight_map()) == CheckpointFormat.FUSED


def test_detect_individual_format():
    assert CheckpointFormat.detect(_individual_expert_weight_map()) == CheckpointFormat.INDIVIDUAL


def test_detect_routed_experts_fused():
    """GLM4 uses ``routed_experts.gate_up_proj`` — still the fused 3D format."""
    wm = {"model.layers.0.mlp.routed_experts.gate_up_proj": "s.safetensors"}
    assert CheckpointFormat.detect(wm) == CheckpointFormat.FUSED


def test_detect_dense_defaults_fused():
    """A non-MoE (dense) checkpoint has no expert keys → treated as fused (no-op)."""
    wm = {"model.embed_tokens.weight": "s", "model.layers.0.self_attn.q_proj.weight": "s"}
    assert CheckpointFormat.detect(wm) == CheckpointFormat.FUSED


def test_detect_bias_keys_are_fused():
    """Fused expert bias keys (GptOss) classify as fused, not individual."""
    wm = {"model.layers.0.mlp.experts.gate_up_proj_bias": "s"}
    assert CheckpointFormat.detect(wm) == CheckpointFormat.FUSED


def _nested_expert_weight_map():
    """A layout nesting one module per expert under the expert container (as pre-5.14 Zaya
    checkpoints did: ``experts.local_experts.N.linear_fc{1,2}``) — an individual layout neither
    expert pattern matches, so detect() misclassifies it as FUSED (fuser skipped, plan dropped,
    params left on meta). The lazy loader must decline these checkpoints."""
    wm = {}
    for layer in range(2):
        for expert in range(8):
            wm[f"model.layers.{layer}.mlp.experts.local_experts.{expert}.linear_fc1.weight"] = "s.safetensors"
            wm[f"model.layers.{layer}.mlp.experts.local_experts.{expert}.linear_fc2.weight"] = "s.safetensors"
        wm[f"model.layers.{layer}.mlp.router.weight"] = "s.safetensors"
    return wm


def test_nested_expert_layout_misdetected_as_fused():
    """The nested-individual layout falls through detect() to FUSED (the silent-corruption
    trigger) — which is exactly why lazy_loader_supports_checkpoint must reject it."""
    assert CheckpointFormat.detect(_nested_expert_weight_map()) == CheckpointFormat.FUSED


def test_unmapped_fused_experts_fail_loud():
    """A fused checkpoint whose expert keys all miss the model namespace must raise, not train
    freshly-initialized experts."""
    wm = _fused_expert_weight_map()
    with pytest.raises(RuntimeError, match="none map"):
        _check_fused_experts_mapped(CheckpointFormat.FUSED, wm, 0)
    # Anti-vacuity: mapped experts, individual format and expertless maps must all pass.
    _check_fused_experts_mapped(CheckpointFormat.FUSED, wm, 4)
    _check_fused_experts_mapped(CheckpointFormat.INDIVIDUAL, _individual_expert_weight_map(), 0)
    _check_fused_experts_mapped(CheckpointFormat.FUSED, {"model.embed_tokens.weight": "s.safetensors"}, 0)


def test_load_ep_model_lazy_raises_on_unmapped_fused_experts(monkeypatch):
    """The guard must be wired into ``load_ep_model_lazy`` itself — a helper alone protects nothing."""
    wm = _fused_expert_weight_map()
    monkeypatch.setattr(lazy_loader, "resolve_safetensors_index", lambda path: (wm, ["s.safetensors"]))
    monkeypatch.setattr(lazy_loader, "instantiate_on_meta", lambda *a, **k: _MockModel(["unrelated.weight"]))
    # num_experts must be real: get_num_experts fails loud rather than guessing.
    config = SimpleNamespace(dtype=torch.bfloat16, num_experts=MockEPConfig().num_experts)
    with pytest.raises(RuntimeError, match="none map"):
        lazy_loader.load_ep_model_lazy("/nonexistent", MockEPConfig(), config=config)


def _write_index(tmp_path, weight_map):
    import json

    (tmp_path / "model.safetensors.index.json").write_text(json.dumps({"metadata": {}, "weight_map": weight_map}))
    return str(tmp_path)


def test_lazy_unsupported_for_nested_expert_layout(tmp_path):
    """``local_experts.N.linear_fc*`` keys → lazy loader declines (routes to HF path)."""
    path = _write_index(tmp_path, _nested_expert_weight_map())
    assert lazy_loader_supports_checkpoint(path) is False


def test_lazy_supported_for_fused_layout(tmp_path):
    """GptOss/GLM fused 3D experts are handled by the lazy loader."""
    path = _write_index(tmp_path, _fused_expert_weight_map())
    assert lazy_loader_supports_checkpoint(path) is True


def test_lazy_supported_for_individual_layout(tmp_path):
    """Qwen3/Bailing ``experts.N.gate_proj`` (digit directly after experts) is handled by the fuser."""
    path = _write_index(tmp_path, _individual_expert_weight_map())
    assert lazy_loader_supports_checkpoint(path) is True


def test_lazy_supported_ignores_shared_experts_index(tmp_path):
    """``shared_experts.0.*`` must NOT trip the guard: ``_experts`` has no preceding dot, so the
    unsupported-layout pattern (which anchors on ``.experts.``) does not match — no false fallback."""
    wm = {
        "model.layers.0.mlp.experts.gate_up_proj": "s.safetensors",
        "model.layers.0.mlp.shared_experts.0.gate_proj.weight": "s.safetensors",
        "model.layers.0.mlp.shared_experts.0.down_proj.weight": "s.safetensors",
    }
    path = _write_index(tmp_path, wm)
    assert lazy_loader_supports_checkpoint(path) is True


def test_lazy_unsupported_for_a_natively_quantized_checkpoint(tmp_path):
    """A checkpoint-native ``quantization_config`` must route to from_pretrained: the lazy loaders
    read weight tensors raw and map nothing for the scale tensors, so the coverage gate cannot fire
    and the experts load as garbage."""
    path = _write_index(tmp_path, _fused_expert_weight_map())
    (tmp_path / "config.json").write_text(
        json.dumps({"model_type": "gpt_oss", "quantization_config": {"quant_method": "mxfp4"}})
    )
    assert lazy_loader_supports_checkpoint(path) is False

    # Anti-vacuity: the same checkpoint without the key is the accepted branch, not a blanket refusal.
    (tmp_path / "config.json").write_text(json.dumps({"model_type": "gpt_oss"}))
    assert lazy_loader_supports_checkpoint(path) is True


def test_lazy_supported_missing_index_defaults_true(tmp_path):
    """Best-effort: an unreadable/missing index → True so the normal load path runs and diagnoses."""
    assert lazy_loader_supports_checkpoint(str(tmp_path)) is True


def test_has_checkpoint_index(tmp_path):
    (tmp_path / "model.safetensors.index.json").write_text("{}")
    assert has_safetensors_checkpoint(str(tmp_path))


def test_has_checkpoint_single(tmp_path):
    save_file({"w": torch.zeros(2)}, str(tmp_path / "model.safetensors"))
    assert has_safetensors_checkpoint(str(tmp_path))


def test_has_checkpoint_absent(tmp_path):
    assert not has_safetensors_checkpoint(str(tmp_path))


class _MockModel:
    """Minimal stand-in for the parts of nn.Module that build_family_key_mapping uses."""

    def __init__(self, keys, base_model_prefix="", model_type=""):
        self._keys = list(keys)
        self.base_model_prefix = base_model_prefix
        self.config = SimpleNamespace(model_type=model_type)

    def state_dict(self):
        return {k: None for k in self._keys}

    def named_modules(self):
        """No sub-model claims a conversion scope: these keys exercise alignment, not conversion."""
        return iter(())


def _family_key_mapping(model, disk_keys):
    """The disk→model half of the live mapping the EP/PP loaders build."""
    return build_family_key_mapping(model, disk_keys)[0]


def test_key_mapping_identity():
    model = _MockModel(["model.layers.0.mlp.experts.gate_up_proj"])
    mapping = _family_key_mapping(model, ["model.layers.0.mlp.experts.gate_up_proj"])
    assert mapping["model.layers.0.mlp.experts.gate_up_proj"] == "model.layers.0.mlp.experts.gate_up_proj"


def test_key_mapping_adds_base_prefix():
    """Checkpoint saved without the ``model.`` base prefix gets it added back."""
    model = _MockModel(["model.embed_tokens.weight"], base_model_prefix="model")
    mapping = _family_key_mapping(model, ["embed_tokens.weight"])
    assert mapping["embed_tokens.weight"] == "model.embed_tokens.weight"


def test_key_mapping_strips_base_prefix_for_a_bare_backbone():
    """Mirror of the add branch: a ``*ForCausalLM`` checkpoint (``model.layers.*``) loaded into the
    bare backbone ``AutoModel`` builds (``layers.*``) must have the prefix stripped."""
    model = _MockModel(["layers.0.self_attn.q_proj.weight"], base_model_prefix="model")
    disk_key = "model.layers.0.self_attn.q_proj.weight"
    assert _family_key_mapping(model, [disk_key])[disk_key] == "layers.0.self_attn.q_proj.weight"


def test_key_mapping_covers_every_key_of_a_bare_moe_backbone():
    """The failure the strip exists for, stated end-to-end on a real MoE pair.

    ``scripts/training/embedding.py`` loads with ``model_class=AutoModel`` under EP, so the shell is
    ``GptOssModel`` while the checkpoint on disk is ``GptOssForCausalLM``. A key that maps to nothing
    is dropped by :class:`EPWeightPlanner` without a word, so an unstripped prefix leaves the ENTIRE
    backbone unplanned — every weight random. Without the strip the intersection below is empty.
    """
    from transformers import GptOssConfig, GptOssForCausalLM, GptOssModel

    config = GptOssConfig(**TINY_GPTOSS_CONFIG)
    with torch.device("meta"):
        disk_keys = sorted(GptOssForCausalLM(config).state_dict())
        backbone = GptOssModel(config)

    model_keys = set(backbone.state_dict())
    mapping = _family_key_mapping(backbone, disk_keys)
    planner = EPWeightPlanner(MockEPConfig(0, 4, TINY_GPTOSS_CONFIG["num_local_experts"]))
    planned = {
        plan.model_key for plan in planner.build(dict.fromkeys(disk_keys, "s.safetensors"), mapping, model_keys)
    }

    assert planned == model_keys, sorted(model_keys - planned)
    assert any(plan_key.endswith("experts.gate_up_proj") for plan_key in planned)


def test_key_mapping_strips_vlm_language_model_segment():
    """A VLM checkpoint key ``model.language_model.layers...`` loaded as a text-only
    CausalLM has the ``language_model.`` segment stripped (the VLM-reuse case)."""
    model = _MockModel(["model.layers.0.self_attn.q_proj.weight"])
    disk_key = "model.language_model.layers.0.self_attn.q_proj.weight"
    mapping = _family_key_mapping(model, [disk_key])
    assert mapping[disk_key] == "model.layers.0.self_attn.q_proj.weight"


def test_key_mapping_strips_leading_language_model():
    model = _MockModel(["embed_tokens.weight"])
    mapping = _family_key_mapping(model, ["language_model.embed_tokens.weight"])
    assert mapping["language_model.embed_tokens.weight"] == "embed_tokens.weight"


def test_key_mapping_reorders_nested_vlm_language_model():
    """Public Mistral3 checkpoints and their instantiated VLM use different text namespaces."""
    model = _MockModel(
        [
            "model.language_model.layers.0.mlp.experts.gate_up_proj",
            "lm_head.weight",
        ],
        base_model_prefix="model",
    )
    expert_key = "language_model.model.layers.0.mlp.experts.gate_up_proj"
    head_key = "language_model.lm_head.weight"

    mapping = _family_key_mapping(model, [expert_key, head_key])

    assert mapping[expert_key] == "model.language_model.layers.0.mlp.experts.gate_up_proj"
    assert mapping[head_key] == "lm_head.weight"


def test_key_mapping_applies_the_family_hub_renames():
    """A family whose hub spelling differs from its module spelling (Laguna) is rewritten when the
    raw key misses. Unmapped keys are silently skipped by the planner, so a missed rename loads the
    module it belongs to as random weights."""
    model = _MockModel(["model.layers.1.mlp.shared_experts.gate_proj.weight"], model_type="laguna")
    disk_key = "model.layers.1.mlp.shared_expert.gate_proj.weight"
    mapping = _family_key_mapping(model, [disk_key])
    assert mapping[disk_key] == "model.layers.1.mlp.shared_experts.gate_proj.weight"


def test_key_mapping_leaves_a_family_without_renames_alone():
    """Anti-over-rejection: an unregistered or rename-free family keeps every key verbatim."""
    model = _MockModel(["model.layers.0.mlp.experts.gate_up_proj"], model_type="qwen3_moe")
    disk_key = "model.layers.0.mlp.shared_expert.gate_proj.weight"
    assert _family_key_mapping(model, [disk_key])[disk_key] == disk_key


def _individual_fuse_map(experts, layer=0):
    """A weight map with per-expert gate/up/down keys for one layer."""
    wm = {}
    for e in experts:
        wm[f"model.layers.{layer}.mlp.experts.{e}.gate_proj.weight"] = "model.safetensors"
        wm[f"model.layers.{layer}.mlp.experts.{e}.up_proj.weight"] = "model.safetensors"
        wm[f"model.layers.{layer}.mlp.experts.{e}.down_proj.weight"] = "model.safetensors"
    return wm


def _identity_mapping(weight_map):
    """disk_to_model for a checkpoint whose keys already match the model's."""
    return {k: k for k in weight_map}


def test_fuser_detect_only_local_experts():
    """detect_tasks emits fusion tasks only for the rank's local expert range."""
    wm = _individual_fuse_map(range(16))
    model_keys = {
        "model.layers.0.mlp.experts.gate_up_proj",
        "model.layers.0.mlp.experts.down_proj",
    }
    fuser = ExpertFuser(ep_start=4, ep_end=8)
    tasks = fuser.detect_tasks(wm, _identity_mapping(wm), model_keys)

    by_type = {t[1]: t for t in tasks}
    assert set(by_type) == {"gate_up", "down"}
    _, _, experts_data = by_type["gate_up"]
    assert sorted(experts_data.keys()) == [4, 5, 6, 7]


def test_fuser_raises_when_expert_group_resolves_to_nothing():
    """A layer the model DID build, whose experts match neither a fused nor a per-expert model
    param: they would stay on meta and only surface much later as "Cannot copy out of meta tensor",
    so this must fail loud here. The layer's non-expert params are what separates it from an absent
    MTP tail."""
    wm = _individual_fuse_map(range(4))
    built_layer = {"model.layers.0.input_layernorm.weight", "model.layers.0.self_attn.q_proj.weight"}
    fuser = ExpertFuser(ep_start=0, ep_end=4)
    with pytest.raises(RuntimeError, match="would never load"):
        fuser.detect_tasks(wm, _identity_mapping(wm), model_keys=built_layer)


def test_fuser_skips_experts_of_a_layer_the_model_never_builds():
    """GLM-4 declares num_hidden_layers=N plus num_nextn_predict_layers=1, so the checkpoint carries
    a trailing MTP layer transformers never instantiates. Its expert keys are legitimately dropped —
    a resolve guard that reads them as "these weights would never load" refuses to load
    GLM-4.5/4.6/4.7 at all."""
    wm = _individual_fuse_map(range(4), layer=0) | _individual_fuse_map(range(4), layer=1)
    # The model holds every parameter of layer 0 and NOTHING of layer 1 (the MTP tail).
    model_keys = {
        "model.layers.0.mlp.experts.gate_up_proj",
        "model.layers.0.mlp.experts.down_proj",
        "model.layers.0.input_layernorm.weight",
    }
    fuser = ExpertFuser(ep_start=0, ep_end=4)
    tasks = fuser.detect_tasks(wm, _identity_mapping(wm), model_keys)

    assert {t[0] for t in tasks} == {
        "model.layers.0.mlp.experts.gate_up_proj",
        "model.layers.0.mlp.experts.down_proj",
    }, "the built layer must still be fused"


def test_fuser_still_raises_for_the_built_sibling_of_an_absent_layer():
    """The absent-layer skip must not become a blanket amnesty: with an unresolvable target on a
    layer the model DID build, the raise still fires even though a later layer is absent."""
    wm = _individual_fuse_map(range(4), layer=0) | _individual_fuse_map(range(4), layer=1)
    model_keys = {"model.layers.0.input_layernorm.weight"}  # layer 0 built, no expert target
    fuser = ExpertFuser(ep_start=0, ep_end=4)
    with pytest.raises(RuntimeError, match="layout mismatch"):
        fuser.detect_tasks(wm, _identity_mapping(wm), model_keys)


def test_fuser_defers_to_planner_when_model_stores_individual_experts():
    """A model shell that keeps one module per expert is covered by EPWeightPlanner — no fusion task,
    and no raise."""
    wm = _individual_fuse_map(range(4))
    fuser = ExpertFuser(ep_start=0, ep_end=4)
    assert fuser.detect_tasks(wm, _identity_mapping(wm), model_keys=set(wm)) == []


def test_fuser_resolves_target_through_the_disk_to_model_mapping():
    """Families whose keys need the base_model_prefix or a declared hub rename spell the disk keys
    differently; the fusion target must follow that mapping, not string surgery on the disk key."""
    wm = _individual_fuse_map(range(2))
    disk_to_model = {k: f"model.{k}" for k in wm}
    model_keys = {
        "model.model.layers.0.mlp.experts.gate_up_proj",
        "model.model.layers.0.mlp.experts.down_proj",
    }
    fuser = ExpertFuser(ep_start=0, ep_end=2)
    tasks = fuser.detect_tasks(wm, disk_to_model, model_keys)
    assert {t[0] for t in tasks} == model_keys


def test_fuser_fuse_gate_up_values(tmp_path):
    """_fuse_gate_up stacks [gate; up] per expert into [E_local, 2M, H]."""
    H, M = 8, 5
    gate = {e: torch.randn(M, H) for e in range(2)}
    up = {e: torch.randn(M, H) for e in range(2)}
    tensors = {}
    for e in range(2):
        tensors[f"model.layers.0.mlp.experts.{e}.gate_proj.weight"] = gate[e]
        tensors[f"model.layers.0.mlp.experts.{e}.up_proj.weight"] = up[e]
        tensors[f"model.layers.0.mlp.experts.{e}.down_proj.weight"] = torch.randn(H, M)
    save_file(tensors, str(tmp_path / "model.safetensors"))

    wm = {k: "model.safetensors" for k in tensors}
    model_keys = {
        "model.layers.0.mlp.experts.gate_up_proj",
        "model.layers.0.mlp.experts.down_proj",
    }
    fuser = ExpertFuser(ep_start=0, ep_end=2)
    tasks = fuser.detect_tasks(wm, _identity_mapping(wm), model_keys)
    handles = {"model.safetensors": safe_open(str(tmp_path / "model.safetensors"), framework="pt", device="cpu")}
    gate_up_task = next(t for t in tasks if t[1] == "gate_up")
    fused = fuser._fuse_gate_up(gate_up_task[2], handles)
    assert fused.shape == (2, 2 * M, H)
    for e in range(2):
        assert torch.allclose(fused[e, :M], gate[e])
        assert torch.allclose(fused[e, M:], up[e])


def test_fuser_handles_routed_experts_prefix():
    """GLM4-style ``routed_experts.N`` keys fuse to the routed_experts target."""
    wm = {
        "model.layers.0.mlp.routed_experts.0.gate_proj.weight": "model.safetensors",
        "model.layers.0.mlp.routed_experts.0.up_proj.weight": "model.safetensors",
        "model.layers.0.mlp.routed_experts.0.down_proj.weight": "model.safetensors",
    }
    model_keys = {
        "model.layers.0.mlp.routed_experts.gate_up_proj",
        "model.layers.0.mlp.routed_experts.down_proj",
    }
    fuser = ExpertFuser(ep_start=0, ep_end=1)
    tasks = fuser.detect_tasks(wm, _identity_mapping(wm), model_keys)
    target_keys = {t[0] for t in tasks}
    assert "model.layers.0.mlp.routed_experts.gate_up_proj" in target_keys
    assert "model.layers.0.mlp.routed_experts.down_proj" in target_keys


def test_load_skips_ignore_plans(tmp_path):
    import torch.nn as nn

    save_file({"w": torch.randn(4, 4)}, str(tmp_path / "model.safetensors"))
    model = nn.Sequential(nn.Linear(4, 4, bias=False))
    # Sentinel proves IGNORE left the param untouched.
    sentinel = torch.full((4, 4), 7.0)
    model[0].weight.data.copy_(sentinel)

    plan = WeightPlan(
        action=WeightAction.IGNORE,
        shard_file="model.safetensors",
        disk_key="w",
        model_key="0.weight",
    )
    loader = SafetensorsWeightLoader(str(tmp_path), ["model.safetensors"], device="cpu")
    loader.load_into_model(model, [plan])
    assert torch.equal(model[0].weight.data, sentinel)


if __name__ == "__main__":
    import pytest

    pytest.main([__file__, "-v"])
