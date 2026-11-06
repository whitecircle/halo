#!/usr/bin/env python
"""The lazy loaders' hub-conversion seam on hand-built entries. Real mappings are pinned by
``tests/cpu/parallelism/test_lazy_load_converted_families.py``."""

import json
from functools import cache
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file
from transformers import AutoConfig, AutoModel
from transformers.core_model_loading import (
    Chunk,
    Concatenate,
    GroupWeightRename,
    Interleave,
    MergeModulelist,
    PermuteForRope,
    PrefixChange,
    Transpose,
    WeightConverter,
    WeightRenaming,
)
from transformers.models.inkling.configuration_inkling import InklingTextConfig
from transformers.models.inkling.modeling_inkling import InklingForCausalLM

import halo.model_loading.lazy_safetensors.conversion_selection as conversion_selection
from halo.checkpoint.reading import resolve_checkpoint_weights
from halo.model_loading.checkpoint_keys import build_key_plan
from halo.model_loading.lazy_safetensors.conversion import (
    Concat,
    Deinterleave,
    PermuteRope,
    TakeChunk,
    lazy_key_mapping,
    translate_converter,
)
from halo.model_loading.lazy_safetensors.conversion_selection import family_key_mapping
from halo.model_loading.lazy_safetensors.materialization import (
    SafetensorsWeightLoader,
    WeightAction,
    WeightPlan,
)
from halo.model_loading.model_preparation import resolve_auto_model_class
from halo.model_loading.parallel.lazy_plan import WeightPlanner
from halo.models.moe.expert_layout import ExpertLayoutKind
from tests.common.models import TINY_INKLING_CONFIG
from tests.common.rosters import ep_layer_class_by_model_type
from tests.common.tiny_models import TINY_MOE_VLM_FAMILIES, build_tiny_family_checkpoint, module_with_weight_keys

E, M, H = 4, 6, 8


def _interleaved(gate: torch.Tensor, up: torch.Tensor, dim: int) -> torch.Tensor:
    stacked = torch.stack([gate, up], dim=dim + 1)
    shape = list(gate.shape)
    shape[dim] *= 2
    return stacked.reshape(shape)


def test_deinterleave_recovers_contiguous_halves():
    gate = torch.randn(E, M, H)
    up = torch.randn(E, M, H)
    disk = _interleaved(gate, up, dim=1)
    canonical = Deinterleave(dim=1)(disk)
    assert torch.equal(canonical, torch.cat([gate, up], dim=1))


def test_deinterleave_matches_transformers_op():
    """A reimplementation of transformers' Interleave forward, pinned equal to it."""
    tensor = torch.randn(E, 2 * M, H)
    theirs = Interleave(dim=1).convert({"src": tensor}, ["src"], ["tgt"])["tgt"]
    assert torch.equal(Deinterleave(dim=1)(tensor), theirs)


def test_take_chunk_fan_out():
    tensor = torch.randn(2 * M, H)
    assert torch.equal(TakeChunk(0, 0, 2)(tensor), tensor[:M])
    assert torch.equal(TakeChunk(0, 1, 2)(tensor), tensor[M:])


def test_permute_rope_matches_transformers_op():
    """A reimplementation of transformers' PermuteForRope forward, pinned equal to it."""
    n_heads, rows = 4, 32
    config = SimpleNamespace(vision_config=SimpleNamespace(num_attention_heads=n_heads))
    for tensor in (torch.randn(rows, H), torch.randn(rows)):
        for inverse in (False, True):
            theirs = PermuteForRope(subconfig_key="vision_config", permute_layer_names=["q_proj"], inverse=inverse)
            expected = theirs.convert({"attn.q_proj.weight": tensor}, [], [], config=config)["attn.q_proj.weight"]
            assert torch.equal(PermuteRope(n_heads, inverse=inverse)(tensor), expected)
    # Anti-vacuity: the permutation is not the identity.
    assert not torch.equal(
        PermuteRope(n_heads)(torch.arange(rows * H, dtype=torch.float32).view(rows, H)),
        torch.arange(rows * H, dtype=torch.float32).view(rows, H),
    )


def test_unsupported_ops_are_refused():
    entry = WeightConverter(
        source_patterns="a", target_patterns=["b", "c", "d"], operations=[Chunk(dim=0), Chunk(dim=1)]
    )
    with pytest.raises(ValueError, match="Unsupported conversion ops"):
        translate_converter(entry)
    # A single-source Concatenate is not the fan-in form.
    with pytest.raises(ValueError, match="Unsupported conversion ops"):
        translate_converter(WeightConverter(source_patterns="a", target_patterns="b", operations=[Concatenate(dim=0)]))


def test_multi_source_entry_translates_to_a_fan_in():
    """GLM-5 Next's ``q/k/v_conv1d → conv1d`` shape."""
    entry = WeightConverter(
        source_patterns=["attn.q_conv1d.weight", "attn.k_conv1d.weight", "attn.v_conv1d.weight"],
        target_patterns="attn.conv1d.weight",
        operations=[Concatenate(dim=0)],
    )
    assert translate_converter(entry) == (((),), 0)

    # transformers rejects a many-to-many entry at construction, so a second target never reaches the translator.
    bad = WeightConverter(
        source_patterns=["a", "b"], target_patterns="c", operations=[Concatenate(dim=0), Chunk(dim=1)]
    )
    with pytest.raises(ValueError, match="multi-source"):
        translate_converter(bad)


def test_chunk_then_permute_for_rope_lands_on_the_named_targets():
    """Step-3.5 vision's ``in_proj_weight`` shape, with the head count read off the sub-config."""
    entry = WeightConverter(
        source_patterns=r"attn\.in_proj_weight",
        target_patterns=[r"attn.q_proj.weight", r"attn.k_proj.weight", r"attn.v_proj.weight"],
        operations=[
            Chunk(dim=0),
            PermuteForRope(subconfig_key="vision_config", permute_layer_names=["q_proj", "k_proj"]),
        ],
    )
    config = SimpleNamespace(vision_config=SimpleNamespace(num_attention_heads=4))
    pipelines, concat_dim = translate_converter(entry, config=config)
    assert concat_dim is None
    assert pipelines == (
        (TakeChunk(0, 0, 3), PermuteRope(4)),
        (TakeChunk(0, 1, 3), PermuteRope(4)),
        (TakeChunk(0, 2, 3),),
    )
    with pytest.raises(ValueError, match="needs the model config"):
        translate_converter(entry)


def _mapping(keys, live, conversions, base_model_prefix=None, config=None):
    return lazy_key_mapping(build_key_plan(keys, live, conversions, base_model_prefix), config)


_FAN_IN = WeightConverter(
    source_patterns=["self_attn.q_conv1d.weight", "self_attn.k_conv1d.weight", "self_attn.v_conv1d.weight"],
    target_patterns="self_attn.conv1d.weight",
    operations=[Concatenate(dim=0)],
)


def test_a_fan_in_groups_by_target_in_declared_source_order():
    keys = [f"decoder.layers.{layer}.self_attn.{n}_conv1d.weight" for n in "vkq" for layer in (1, 0)]
    live = [f"model.layers.{layer}.self_attn.conv1d.weight" for layer in (0, 1)]
    rename = WeightRenaming(source_patterns=r"^decoder\.", target_patterns="model.")
    _, converted = _mapping(keys, live, [rename, _FAN_IN])
    assert converted == {
        f"decoder.layers.{layer}.self_attn.q_conv1d.weight": (
            (
                f"model.layers.{layer}.self_attn.conv1d.weight",
                (Concat(0, tuple(f"decoder.layers.{layer}.self_attn.{n}_conv1d.weight" for n in "kv")),),
            ),
        )
        for layer in (0, 1)
    }


def test_a_partial_or_doubled_fan_in_raises():
    with pytest.raises(ValueError, match="truncated tensor"):
        _mapping(
            ["l.self_attn.q_conv1d.weight", "l.self_attn.v_conv1d.weight"], ["l.self_attn.conv1d.weight"], [_FAN_IN]
        )
    doubled = WeightConverter(source_patterns=["a.*.x", "b.x"], target_patterns="c.x", operations=[Concatenate(dim=0)])
    with pytest.raises(ValueError, match="both match source"):
        _mapping(["a.1.x", "a.2.x", "b.x"], ["c.x"], [doubled])


def test_an_unsupported_converter_raises_only_when_it_claims_a_live_name():
    bad = WeightConverter(source_patterns=["a.x", "b.x"], target_patterns="c.x", operations=[Chunk(dim=0)])
    assert _mapping(["a.x", "b.x", "norm.weight"], ["norm.weight"], [bad]) == (
        {"a.x": "c.x", "b.x": "c.x", "norm.weight": "norm.weight"},
        {},
    )
    with pytest.raises(ValueError, match="multi-source"):
        _mapping(["a.x", "b.x"], ["c.x"], [bad])


def test_an_expert_bank_merge_keeps_the_renamed_per_expert_spelling():
    """The ``ExpertFuser`` stacks per-expert keys itself; a merge with an op it cannot reproduce is refused."""
    rename = WeightRenaming(source_patterns=r"model\.llm\.", target_patterns="model.")
    merge = WeightConverter(
        source_patterns=["mlp.experts.*.gate_proj.weight", "mlp.experts.*.up_proj.weight"],
        target_patterns="mlp.experts.gate_up_proj",
        operations=[MergeModulelist(dim=0), Concatenate(dim=1)],
    )
    keys = [f"model.llm.layers.0.mlp.experts.{e}.{name}_proj.weight" for e in (0, 1) for name in ("gate", "up")]
    disk_to_model, converted = _mapping(keys, ["model.layers.0.mlp.experts.gate_up_proj"], [rename, merge])
    assert disk_to_model == {key: key.replace("model.llm.", "model.") for key in keys}
    assert not converted
    merge.operations.append(Transpose(dim0=1, dim1=2))
    with pytest.raises(ValueError, match="ExpertFuser only reproduces"):
        _mapping(keys, ["model.layers.0.mlp.experts.gate_up_proj"], [rename, merge])


def test_planner_carries_ops_and_rejects_shard_axis_ops():
    fanout = {
        "disk.expert": (("model.experts.gate_up_proj", (Deinterleave(dim=1),)),),
        "disk.shared": (
            ("model.shared.gate_proj.weight", (TakeChunk(0, 0, 2),)),
            ("model.shared.up_proj.weight", (TakeChunk(0, 1, 2),)),
        ),
    }
    model_keys = {"model.experts.gate_up_proj", "model.shared.gate_proj.weight", "model.shared.up_proj.weight"}
    weight_map = {"disk.expert": "a.safetensors", "disk.shared": "a.safetensors"}

    plans = WeightPlanner(_EP()).build(weight_map, {}, model_keys, fanout=fanout)
    by_key = {p.model_key: p for p in plans}
    assert len(plans) == 3
    assert by_key["model.experts.gate_up_proj"].action == WeightAction.EXPERT_SHARD
    assert by_key["model.experts.gate_up_proj"].ops == (Deinterleave(dim=1),)
    assert by_key["model.shared.up_proj.weight"].ops == (TakeChunk(0, 1, 2),)

    bad = {"disk.expert": (("model.experts.gate_up_proj", (Deinterleave(dim=0),)),)}
    with pytest.raises(ValueError, match="expert shard axis"):
        WeightPlanner(_EP()).build(weight_map, {}, model_keys, fanout=bad)


class _EP:
    """A stub short of one field models a config that cannot exist."""

    expert_start_idx, expert_end_idx = 0, 2
    num_experts = 4


def test_planner_consumes_fan_in_siblings_and_checks_the_concat_axis():
    """Step-3.7's ``moe.gate_proj + moe.up_proj → experts.gate_up_proj`` is one expert-sharded plan reading two disk
    keys."""
    gate, up = "l.moe.gate_proj.weight", "l.moe.up_proj.weight"
    weight_map = {gate: "a.safetensors", up: "b.safetensors"}
    model_keys = {"l.mlp.experts.gate_up_proj"}
    fanout = {gate: (("l.mlp.experts.gate_up_proj", (Concat(1, (up,)),)),)}

    (plan,) = WeightPlanner(_EP()).build(weight_map, {}, model_keys, fanout=fanout)
    assert plan.action == WeightAction.EXPERT_SHARD and plan.ops == (Concat(1, (up,)),)
    assert plan.disk_keys == (gate, up)

    on_axis = {gate: (("l.mlp.experts.gate_up_proj", (Concat(0, (up,)),)),)}
    with pytest.raises(ValueError, match="expert shard axis"):
        WeightPlanner(_EP()).build(weight_map, {}, model_keys, fanout=on_axis)


def test_materializer_reads_every_fan_in_source_through_the_expert_slice(tmp_path):
    """This rank's slice of each source, and a sibling in a shard no plan names opens on demand."""
    gate, up = torch.randn(E, M, H), torch.randn(E, M, H)
    save_file({"l.moe.gate_proj.weight": gate}, str(tmp_path / "a.safetensors"))
    save_file({"l.moe.up_proj.weight": up}, str(tmp_path / "b.safetensors"))
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps(
            {"weight_map": {"l.moe.gate_proj.weight": "a.safetensors", "l.moe.up_proj.weight": "b.safetensors"}}
        )
    )
    plan = WeightPlan(
        WeightAction.EXPERT_SHARD,
        "a.safetensors",
        "l.moe.gate_proj.weight",
        "l.mlp.experts.gate_up_proj",
        shard_dim=0,
        shard_start=1,
        shard_end=3,
        ops=(Concat(1, ("l.moe.up_proj.weight",)),),
        shard_total=E,
    )
    loader = SafetensorsWeightLoader(str(tmp_path), resolve_checkpoint_weights(str(tmp_path)).weight_map)
    model = torch.nn.Module()
    model.add_module("l", torch.nn.Module())
    model.l.add_module("mlp", torch.nn.Module())
    model.l.mlp.add_module("experts", torch.nn.Module())
    model.l.mlp.experts.gate_up_proj = torch.nn.Parameter(torch.empty(E, 2 * M, H, device="meta"))
    loader.load_into_model(model, [plan])
    assert torch.equal(model.l.mlp.experts.gate_up_proj, torch.cat([gate[1:3], up[1:3]], dim=1))
    # A wider sibling must not be silently sliced.
    save_file({"l.moe.up_proj.weight": torch.randn(2 * E, M, H)}, str(tmp_path / "b.safetensors"))
    model.l.mlp.experts.gate_up_proj = torch.nn.Parameter(torch.empty(E, 2 * M, H, device="meta"))
    with pytest.raises(RuntimeError, match="config declares"):
        loader.load_into_model(model, [plan])


def _shell(model_type: str):
    """A sub-config no Auto mapping instantiates (``step3p5``) exists only inside its composite, so the composite's
    model_type drives the same conversions."""
    config = AutoConfig.for_model(model_type)
    with torch.device("meta"):
        try:
            return resolve_auto_model_class(config).from_config(config)
        except ValueError:
            pass
        try:
            return AutoModel.from_config(config)
        except ValueError:
            pytest.skip(f"{model_type} has no standalone model class — a composite-only text sub-config")


def _declaring_model_types() -> list[str]:
    """Read off the classes, so a family growing a declaration is covered."""
    return sorted(mt for mt, cls in ep_layer_class_by_model_type().items() if cls.FAMILY.hub_conversion_keys)


@cache
def _canonical_checkpoint_keys(model_type: str) -> tuple[str, ...]:
    """The state dict's keys, with a per-expert layout's fused banks spelled as the gathered save writes them."""
    state = _shell(model_type).state_dict()
    layout = ep_layer_class_by_model_type()[model_type].FAMILY.expert_layout
    if layout.kind is not ExpertLayoutKind.FUSED_SPLIT_PER_EXPERT:
        return tuple(sorted(state))
    gate, up, down = layout.per_expert_keys
    keys = []
    for key, tensor in state.items():
        root, _, leaf = key.rpartition(".")
        if not root.endswith("experts") or leaf not in ("gate_up_proj", "down_proj"):
            keys.append(key)
            continue
        names = (gate, up) if leaf == "gate_up_proj" else (down,)
        keys += [f"{root}.{e}.{name}.weight" for e in range(tensor.shape[0]) for name in names]
    return tuple(sorted(keys))


def test_a_declared_key_that_resolves_to_nothing_raises(monkeypatch):
    """Unconverted, the planner would skip every unmatched key and the model trains from its meta-device init."""
    monkeypatch.setattr(conversion_selection, "get_checkpoint_conversion_mapping", lambda key: None)
    with pytest.raises(ValueError, match="declares hub-conversion key"):
        family_key_mapping(_shell("inkling_mm_model"), [])


def test_a_non_declaring_family_is_untouched_by_an_absent_mapping(monkeypatch):
    monkeypatch.setattr(conversion_selection, "get_checkpoint_conversion_mapping", lambda key: None)
    keys = list(_canonical_checkpoint_keys("qwen3_moe"))
    assert family_key_mapping(_shell("qwen3_moe"), keys) == ({key: key for key in keys}, {})


def test_a_text_model_reads_its_composites_vendor_checkpoint(tmp_path):
    """transformers keys conversions under the composite alone, so the text model takes the entries and drops the
    text level."""
    build_tiny_family_checkpoint(TINY_MOE_VLM_FAMILIES["inkling_mm_model"], str(tmp_path))
    weight_map = resolve_checkpoint_weights(str(tmp_path)).weight_map
    with torch.device("meta"):
        shell = InklingForCausalLM(InklingTextConfig(**TINY_INKLING_CONFIG))
    model_keys = set(shell.state_dict())
    disk_to_model, fanout = family_key_mapping(shell, list(weight_map))
    plans = WeightPlanner(None).build(weight_map, disk_to_model, model_keys, fanout=fanout)
    assert {plan.model_key for plan in plans} == model_keys
    assert any(isinstance(op, Deinterleave) for plan in plans for op in plan.ops)


@pytest.mark.parametrize("model_type", _declaring_model_types())
def test_canonical_checkpoint_keys_resume_untouched(model_type):
    """Conversion sources are not vendor-anchored (deepseek_v4's norm rename also matches the final
    ``model.norm.weight``); identity rests on the planner's live-key retry."""
    layer_cls = ep_layer_class_by_model_type()[model_type]
    layout = layer_cls.FAMILY.expert_layout
    fused = layout.kind in (ExpertLayoutKind.FUSED, ExpertLayoutKind.FUSED_SPLIT_PER_EXPERT)
    assert fused and not layer_cls.FAMILY.export_key_renames, (
        f"{layer_cls.__name__} exports expert/replicated keys under names the live state dict does "
        f"not carry, so the state dict is no longer the checkpoint key space this test checks — "
        f"extend the oracle with the family's gathered-save layout."
    )
    model = _shell(model_type)
    keys = list(_canonical_checkpoint_keys(model_type))
    expert_suffix = "gate_up_proj" if layout.kind is ExpertLayoutKind.FUSED else f"{layout.per_expert_keys[0]}.weight"
    assert any(".experts." in key and key.endswith(expert_suffix) for key in keys), (
        "the oracle must reach the expert keys"
    )

    disk_to_model, fanout = family_key_mapping(model, keys)

    assert not fanout, (
        f"{model_type}: {len(fanout)} canonical checkpoint key(s) still convert on a resume. First few: "
        f"{[(key, [target for target, _ in targets]) for key, targets in list(fanout.items())[:3]]}"
    )
    perturbed = {key: target for key, target in disk_to_model.items() if target != key}
    assert not perturbed, (
        f"{model_type}: {len(perturbed)} canonical checkpoint key(s) are renamed on a resume, so the "
        f"tensor lands on a key the checkpoint writer never used. First few: "
        f"{list(perturbed.items())[:3]}"
    )


@pytest.mark.parametrize("recorded", ("absent", None, ()))
def test_eager_replay_does_not_invent_conversions_for_a_canonical_load(recorded):
    model = module_with_weight_keys(["model.norm.weight"])
    model.config = SimpleNamespace(model_type="qwen3")
    if recorded != "absent":
        model._weight_conversions = recorded
    assert family_key_mapping(model, ["model.norm.weight"], replay_loaded=True) == (
        {"model.norm.weight": "model.norm.weight"},
        {},
    )


def test_eager_replay_accepts_text_only_prefix_changes():
    model = module_with_weight_keys(["model.linear.weight"])
    model.base_model_prefix = "model"
    model.config = SimpleNamespace(model_type="qwen3_moe")
    model._weight_conversions = [PrefixChange(prefix_to_remove="language_model", model_prefix="model")]
    disk_key = "model.language_model.linear.weight"
    assert family_key_mapping(model, [disk_key], replay_loaded=True) == ({disk_key: "model.linear.weight"}, {})


def test_eager_replay_reproduces_a_conditional_group_rename_on_a_copy():
    """Planned on a copy, so the load's record keeps its guard unlatched."""
    model = module_with_weight_keys(["a.norm1.weight", "a.norm2.weight"])
    model.config = SimpleNamespace(model_type="qwen3")
    group = GroupWeightRename(source_patterns=[r"norm0", r"norm1"], target_patterns=["norm1", "norm2"])
    model._weight_conversions = [group]
    mapping, _ = family_key_mapping(model, ["a.norm1.weight", "a.norm0.weight"], replay_loaded=True)
    assert mapping == {"a.norm0.weight": "a.norm1.weight", "a.norm1.weight": "a.norm2.weight"}
    assert not group._active


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
