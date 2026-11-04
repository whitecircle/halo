#!/usr/bin/env python
import json
import re
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
from safetensors import safe_open
from safetensors.torch import save_file
from transformers import GptOssConfig, GptOssForCausalLM, GptOssModel

import halo.model_loading.parallel.ep as ep_loading  # noqa: E402
import halo.model_loading.parallel.lazy_plan as lazy_plan  # noqa: E402
from halo.checkpoint.reading import CheckpointWeights  # noqa: E402
from halo.model_loading.lazy_safetensors.conversion_selection import family_key_mapping  # noqa: E402
from halo.model_loading.lazy_safetensors.materialization import (  # noqa: E402
    SafetensorsWeightLoader,
    WeightAction,
    WeightPlan,
    assign_tensor_to_model,
)
from halo.model_loading.parallel.ep import _check_fused_experts_mapped  # noqa: E402
from halo.model_loading.parallel.expert_fuser import ExpertFuser  # noqa: E402
from halo.model_loading.parallel.lazy_plan import (  # noqa: E402
    ExpertKeyLayout,
    WeightPlanner,
    is_lazy_loadable,
)
from tests.common.models import TINY_GPTOSS_CONFIG  # noqa: E402
from tests.common.tiny_models import module_with_weight_keys  # noqa: E402


class MockEPConfig:
    def __init__(self, start=4, end=8, total=16):
        self.expert_start_idx = start
        self.expert_end_idx = end
        self.experts_per_rank = end - start
        self.num_experts = total


def _fused_expert_weight_map():
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
    planner = WeightPlanner(cfg)
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
    planner = WeightPlanner(cfg)
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
    planner = WeightPlanner(cfg)
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


def test_expert_shard_slicing(tmp_path):
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

    model = nn.Module()
    model.experts = nn.Module()
    model.experts.gate_up_proj = nn.Parameter(torch.empty(4, 128, 256))
    SafetensorsWeightLoader(str(tmp_path), {"experts.gate_up_proj": "model.safetensors"}).load_into_model(
        model, [plan]
    )
    assert torch.equal(model.experts.gate_up_proj, full_weight[4:8])


def test_replicate_loads_full(tmp_path):
    weight = torch.randn(512, 512)
    save_file({"attn.weight": weight}, str(tmp_path / "model.safetensors"))

    plan = WeightPlan(
        action=WeightAction.REPLICATE,
        shard_file="model.safetensors",
        disk_key="attn.weight",
        model_key="attn.weight",
    )

    model = nn.Module()
    model.attn = nn.Linear(512, 512, bias=False)
    SafetensorsWeightLoader(str(tmp_path), {"attn.weight": "model.safetensors"}).load_into_model(model, [plan])
    assert torch.equal(model.attn.weight, weight)


def test_assign_replaces_parameter():
    model = nn.Sequential(nn.Linear(4, 8))
    real_weight = torch.randn(8, 4)
    assign_tensor_to_model(model, "0.weight", real_weight)
    assert torch.equal(model[0].weight.data, real_weight)


def test_assign_nested_module():
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
    assert ExpertKeyLayout.detect(_fused_expert_weight_map()) == ExpertKeyLayout.FUSED


def test_detect_individual_format():
    assert ExpertKeyLayout.detect(_individual_expert_weight_map()) == ExpertKeyLayout.PER_EXPERT


def test_detect_routed_experts_fused():
    wm = {"model.layers.0.mlp.routed_experts.gate_up_proj": "s.safetensors"}
    assert ExpertKeyLayout.detect(wm) == ExpertKeyLayout.FUSED


def test_detect_dense_defaults_fused():
    wm = {"model.embed_tokens.weight": "s", "model.layers.0.self_attn.q_proj.weight": "s"}
    assert ExpertKeyLayout.detect(wm) == ExpertKeyLayout.FUSED


def test_detect_bias_keys_are_fused():
    wm = {"model.layers.0.mlp.experts.gate_up_proj_bias": "s"}
    assert ExpertKeyLayout.detect(wm) == ExpertKeyLayout.FUSED


def _nested_expert_weight_map():
    """As pre-5.14 Zaya checkpoints did: ``experts.local_experts.N.linear_fc{1,2}``. detect() misclassifies it as
    FUSED, so the lazy loader must decline."""
    wm = {}
    for layer in range(2):
        for expert in range(8):
            wm[f"model.layers.{layer}.mlp.experts.local_experts.{expert}.linear_fc1.weight"] = "s.safetensors"
            wm[f"model.layers.{layer}.mlp.experts.local_experts.{expert}.linear_fc2.weight"] = "s.safetensors"
        wm[f"model.layers.{layer}.mlp.router.weight"] = "s.safetensors"
    return wm


def test_nested_expert_layout_misdetected_as_fused():
    """Why is_lazy_loadable must reject it."""
    assert ExpertKeyLayout.detect(_nested_expert_weight_map()) == ExpertKeyLayout.FUSED


def test_unmapped_fused_experts_fail_loud():
    def check(layout, weight_map, *actions):
        plans = [SimpleNamespace(action=action) for action in actions]
        _check_fused_experts_mapped(SimpleNamespace(expert_layout=layout, weight_map=weight_map, plans=plans))

    wm = _fused_expert_weight_map()
    with pytest.raises(RuntimeError, match="none map"):
        check(ExpertKeyLayout.FUSED, wm, WeightAction.REPLICATE)
    # Anti-vacuity: mapped experts, individual format and expertless maps all pass.
    check(ExpertKeyLayout.FUSED, wm, WeightAction.REPLICATE, WeightAction.EXPERT_SHARD)
    check(ExpertKeyLayout.PER_EXPERT, _individual_expert_weight_map())
    check(ExpertKeyLayout.FUSED, {"model.embed_tokens.weight": "s.safetensors"})


def test_load_ep_model_lazy_raises_on_unmapped_fused_experts(monkeypatch):
    """A helper alone protects nothing."""
    wm = _fused_expert_weight_map()
    monkeypatch.setattr(
        lazy_plan,
        "resolve_checkpoint_weights",
        lambda path: CheckpointWeights(path, ("s.safetensors",), index={"weight_map": wm}),
    )
    shell = module_with_weight_keys(["unrelated.weight"])
    shell.config = SimpleNamespace(model_type="")
    monkeypatch.setattr(ep_loading, "instantiate_on_meta", lambda *a, **k: shell)
    # require_expert_count fails loud rather than guessing.
    config = SimpleNamespace(dtype=torch.bfloat16, num_experts=MockEPConfig().num_experts)
    with pytest.raises(RuntimeError, match="none map"):
        ep_loading.load_ep_model_lazy("/nonexistent", MockEPConfig(), config=config)


def _write_index(tmp_path, weight_map):
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps({"metadata": {}, "weight_map": weight_map}))
    return str(tmp_path)


def test_lazy_unsupported_for_nested_expert_layout(tmp_path):
    path = _write_index(tmp_path, _nested_expert_weight_map())
    assert is_lazy_loadable(path) is False


def test_lazy_supported_for_fused_layout(tmp_path):
    path = _write_index(tmp_path, _fused_expert_weight_map())
    assert is_lazy_loadable(path) is True


def test_lazy_supported_for_individual_layout(tmp_path):
    """Digit directly after experts is handled by the fuser."""
    path = _write_index(tmp_path, _individual_expert_weight_map())
    assert is_lazy_loadable(path) is True


def test_lazy_supported_ignores_shared_experts_index(tmp_path):
    """``_experts`` has no preceding dot, so the pattern anchored on ``.experts.`` does not match."""
    wm = {
        "model.layers.0.mlp.experts.gate_up_proj": "s.safetensors",
        "model.layers.0.mlp.shared_experts.0.gate_proj.weight": "s.safetensors",
        "model.layers.0.mlp.shared_experts.0.down_proj.weight": "s.safetensors",
    }
    path = _write_index(tmp_path, wm)
    assert is_lazy_loadable(path) is True


def test_lazy_unsupported_for_a_natively_quantized_checkpoint(tmp_path):
    """The lazy loaders read tensors raw and map nothing for the scale tensors, so the coverage gate cannot fire and
    experts load as garbage."""
    path = _write_index(tmp_path, _fused_expert_weight_map())
    (tmp_path / "config.json").write_text(
        json.dumps({"model_type": "gpt_oss", "quantization_config": {"quant_method": "mxfp4"}})
    )
    assert is_lazy_loadable(path) is False

    # Anti-vacuity: without the key it is accepted.
    (tmp_path / "config.json").write_text(json.dumps({"model_type": "gpt_oss"}))
    assert is_lazy_loadable(path) is True


def test_lazy_supported_missing_index_defaults_true(tmp_path):
    """Best-effort: True so the normal load path runs and diagnoses."""
    assert is_lazy_loadable(str(tmp_path)) is True


@pytest.mark.parametrize(
    ("disk_class", "shell_class"),
    ((GptOssForCausalLM, GptOssModel), (GptOssModel, GptOssForCausalLM)),
    ids=("strip", "add"),
)
def test_key_mapping_aligns_the_base_model_prefix(disk_class, shell_class):
    """``scripts/training/embedding.py`` uses ``model_class=AutoModel`` under EP; an unaligned prefix leaves the
    whole backbone unplanned, every weight random."""
    config = GptOssConfig(**TINY_GPTOSS_CONFIG)
    with torch.device("meta"):
        disk_keys = sorted(disk_class(config).state_dict())
        shell = shell_class(config)

    model_keys = set(shell.state_dict())
    mapping, fanout = family_key_mapping(shell, disk_keys)
    planner = WeightPlanner(MockEPConfig(0, 4, TINY_GPTOSS_CONFIG["num_local_experts"]))
    planned = {
        plan.model_key
        for plan in planner.build(dict.fromkeys(disk_keys, "s.safetensors"), mapping, model_keys, fanout=fanout)
    }

    assert planned == {key for key in model_keys if not key.startswith("lm_head.")}, sorted(model_keys - planned)
    assert any(plan_key.endswith("experts.gate_up_proj") for plan_key in planned)


def _individual_fuse_map(experts, layer=0):
    wm = {}
    for e in experts:
        wm[f"model.layers.{layer}.mlp.experts.{e}.gate_proj.weight"] = "model.safetensors"
        wm[f"model.layers.{layer}.mlp.experts.{e}.up_proj.weight"] = "model.safetensors"
        wm[f"model.layers.{layer}.mlp.experts.{e}.down_proj.weight"] = "model.safetensors"
    return wm


def _identity_mapping(weight_map):
    return {k: k for k in weight_map}


def test_fuser_detect_only_local_experts():
    wm = _individual_fuse_map(range(16))
    model_keys = {
        "model.layers.0.mlp.experts.gate_up_proj",
        "model.layers.0.mlp.experts.down_proj",
    }
    fuser = ExpertFuser(ep_start=4, ep_end=8)
    tasks = fuser.detect_tasks(wm, _identity_mapping(wm), model_keys)

    by_type = {t.kind: t for t in tasks}
    assert set(by_type) == {"gate_up", "down"}
    assert sorted(by_type["gate_up"].experts) == [4, 5, 6, 7]


def test_fuser_raises_when_expert_group_resolves_to_nothing():
    """They would stay on meta until a late "Cannot copy out of meta tensor". The layer's non-expert params separate
    it from an absent MTP tail."""
    wm = _individual_fuse_map(range(4))
    built_layer = {"model.layers.0.input_layernorm.weight", "model.layers.0.self_attn.q_proj.weight"}
    fuser = ExpertFuser(ep_start=0, ep_end=4)
    with pytest.raises(RuntimeError, match="would never load"):
        fuser.detect_tasks(wm, _identity_mapping(wm), model_keys=built_layer)


def test_fuser_skips_experts_of_a_layer_the_model_never_builds():
    """GLM-4's checkpoint carries a trailing MTP layer transformers never instantiates; raising refuses
    GLM-4.5/4.6/4.7 at all."""
    wm = _individual_fuse_map(range(4), layer=0) | _individual_fuse_map(range(4), layer=1)
    # Every parameter of layer 0 and nothing of layer 1 (the MTP tail).
    model_keys = {
        "model.layers.0.mlp.experts.gate_up_proj",
        "model.layers.0.mlp.experts.down_proj",
        "model.layers.0.input_layernorm.weight",
    }
    fuser = ExpertFuser(ep_start=0, ep_end=4)
    tasks = fuser.detect_tasks(wm, _identity_mapping(wm), model_keys)

    assert {t.model_key for t in tasks} == {
        "model.layers.0.mlp.experts.gate_up_proj",
        "model.layers.0.mlp.experts.down_proj",
    }, "the built layer must still be fused"


def test_fuser_still_raises_for_the_built_sibling_of_an_absent_layer():
    """The absent-layer skip must not become a blanket amnesty."""
    wm = _individual_fuse_map(range(4), layer=0) | _individual_fuse_map(range(4), layer=1)
    model_keys = {"model.layers.0.input_layernorm.weight"}  # Layer 0 built, no expert target.
    fuser = ExpertFuser(ep_start=0, ep_end=4)
    with pytest.raises(RuntimeError, match="layout mismatch"):
        fuser.detect_tasks(wm, _identity_mapping(wm), model_keys)


def test_fuser_defers_to_planner_when_model_stores_individual_experts():
    """WeightPlanner covers a shell keeping one module per expert."""
    wm = _individual_fuse_map(range(4))
    fuser = ExpertFuser(ep_start=0, ep_end=4)
    assert fuser.detect_tasks(wm, _identity_mapping(wm), model_keys=set(wm)) == []


def test_fuser_resolves_target_through_the_disk_to_model_mapping():
    """The target follows that mapping, not string surgery on the disk key."""
    wm = _individual_fuse_map(range(2))
    disk_to_model = {k: f"model.{k}" for k in wm}
    model_keys = {
        "model.model.layers.0.mlp.experts.gate_up_proj",
        "model.model.layers.0.mlp.experts.down_proj",
    }
    fuser = ExpertFuser(ep_start=0, ep_end=2)
    tasks = fuser.detect_tasks(wm, disk_to_model, model_keys)
    assert {t.model_key for t in tasks} == model_keys


def test_fuser_fuse_gate_up_values(tmp_path):
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
    gate_up_task = next(t for t in tasks if t.kind == "gate_up")
    fused = fuser._fuse_gate_up(gate_up_task, handles)
    assert fused.shape == (2, 2 * M, H)
    for e in range(2):
        assert torch.allclose(fused[e, :M], gate[e])
        assert torch.allclose(fused[e, M:], up[e])


_GATELESS = "mixer.experts"
_GATELESS_E, _GATELESS_L, _GATELESS_M = 4, 8, 6


def _gateless_keys(experts, names=("up_proj", "down_proj")):
    return [f"{_GATELESS}.{expert}.{name}.weight" for expert in experts for name in names]


def _gateless_tasks(fuser, keys):
    return fuser.detect_tasks(
        dict.fromkeys(keys, "shard.safetensors"),
        {key: key for key in keys},
        {f"{_GATELESS}.up_proj", f"{_GATELESS}.down_proj"},
    )


def test_fuser_stacks_a_gateless_experts_up_proj_alone(tmp_path):
    """Nemotron-H's experts: an ``up_proj`` key the map files under the GLU pair is the whole of a bank here."""
    up = {e: torch.randn(_GATELESS_M, _GATELESS_L) for e in range(_GATELESS_E)}
    down = {e: torch.randn(_GATELESS_L, _GATELESS_M) for e in range(_GATELESS_E)}
    tensors = {f"{_GATELESS}.{e}.up_proj.weight": up[e] for e in up} | {
        f"{_GATELESS}.{e}.down_proj.weight": down[e] for e in down
    }
    save_file(tensors, tmp_path / "shard.safetensors")
    model = nn.Module()
    model.mixer = nn.Module()
    model.mixer.experts = nn.Module()
    model.mixer.experts.up_proj = nn.Parameter(torch.empty(_GATELESS_E, _GATELESS_M, _GATELESS_L))
    model.mixer.experts.down_proj = nn.Parameter(torch.empty(_GATELESS_E, _GATELESS_L, _GATELESS_M))

    fuser = ExpertFuser(2, 4)
    tasks = _gateless_tasks(fuser, list(tensors))
    assert {(task.kind, task.model_key) for task in tasks} == {
        ("up", f"{_GATELESS}.up_proj"),
        ("down", f"{_GATELESS}.down_proj"),
    }
    assert fuser.execute(tasks, model, str(tmp_path), torch.float32, "cpu") == {task.model_key for task in tasks}
    assert torch.equal(model.mixer.experts.up_proj, torch.stack([up[2], up[3]]))
    assert torch.equal(model.mixer.experts.down_proj, torch.stack([down[2], down[3]]))


def test_fuser_keeps_the_pair_where_the_model_fuses_gate_and_up():
    """The branch reads the live model: a GLU family's ``up_proj`` keys stay a half of its ``gate_up_proj``."""
    keys = _gateless_keys(range(2), ("gate_proj", "up_proj"))
    tasks = ExpertFuser(0, 2).detect_tasks(
        dict.fromkeys(keys, "shard.safetensors"), {key: key for key in keys}, {f"{_GATELESS}.gate_up_proj"}
    )
    assert [(task.kind, task.model_key) for task in tasks] == [("gate_up", f"{_GATELESS}.gate_up_proj")]


def test_fuser_refuses_a_gate_half_on_a_gateless_expert():
    """Stacking whichever half came first would load a gate's weights into the bank at the same shape."""
    fuser = ExpertFuser(0, 2)
    [task] = _gateless_tasks(fuser, _gateless_keys(range(2), ("gate_proj", "up_proj")))
    assert task.kind == "up"
    with pytest.raises(RuntimeError, match="gateless"):
        fuser.validate_task(task)


def test_fuser_refuses_a_gap_in_a_gateless_bank():
    """The fuse stacks sorted indices into local slots, so a hole would shift every later expert."""
    fuser = ExpertFuser(0, 4)
    keys = [key for key in _gateless_keys(range(4)) if key != f"{_GATELESS}.1.up_proj.weight"]
    up = next(task for task in _gateless_tasks(fuser, keys) if task.kind == "up")
    with pytest.raises(RuntimeError, match=r"global expert index\(es\) \[1\] are absent"):
        fuser.validate_task(up)


@pytest.mark.parametrize("rank_start", [0, 4])
def test_a_gateless_bank_slices_to_this_ranks_experts(rank_start):
    """Planned whole, it would reach the layer as the first experts on every rank while ``down_proj`` is its own."""
    planner = WeightPlanner(MockEPConfig(start=rank_start, end=rank_start + 4, total=8))
    action, dim, start, end = planner._classify("backbone.layers.1.mixer.experts.up_proj")
    assert (action, dim, start, end) == (WeightAction.EXPERT_SHARD, 0, rank_start, rank_start + 4)
    assert ExpertKeyLayout.detect({"backbone.layers.1.mixer.experts.up_proj": "s"}) is ExpertKeyLayout.FUSED


@pytest.mark.parametrize(
    "key",
    [
        "backbone.layers.1.mixer.shared_experts.up_proj.weight",
        "backbone.layers.1.mixer.experts.3.up_proj.weight",
        "backbone.layers.1.mixer.up_proj",
    ],
)
def test_the_gateless_bank_is_told_from_its_neighbours(key):
    assert WeightPlanner(MockEPConfig())._classify(key)[0] is not WeightAction.EXPERT_SHARD


def test_fuser_handles_routed_experts_prefix():
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
    target_keys = {t.model_key for t in tasks}
    assert "model.layers.0.mlp.routed_experts.gate_up_proj" in target_keys
    assert "model.layers.0.mlp.routed_experts.down_proj" in target_keys


def test_load_skips_ignore_plans(tmp_path):
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
    loader = SafetensorsWeightLoader(str(tmp_path), {"w": "model.safetensors"}, device="cpu")
    loader.load_into_model(model, [plan])
    assert torch.equal(model[0].weight.data, sentinel)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
