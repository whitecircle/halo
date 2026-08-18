#!/usr/bin/env python
"""CPU tests for the hub-conversion seam: op math, the key walker, and the planner integration.

The seam translates transformers' declarative conversion entries (renames + Interleave / Chunk /
PermuteForRope pipelines, multi-source Concatenate fan-ins, sub-model scoping) into per-disk-key
targets the lazy loaders apply at materialization. These tests drive it with hand-built entries;
each declaring family's real mapping is pinned bitwise against ``from_pretrained`` by
``tests/cpu/parallelism/test_lazy_load_converted_families.py`` and
``tests/gpu/parallelism/ep/test_lazy_load_inkling.py``.

    python tests/cpu/parallelism/test_hub_conversion.py
"""

import json
import sys
from functools import cache
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file
from transformers import AutoConfig, AutoModel
from transformers.core_model_loading import (
    Chunk,
    Concatenate,
    Interleave,
    PermuteForRope,
    WeightConverter,
    WeightRenaming,
)

import src.distributed.expert_parallel.hub_conversion as hub_conversion
from src.distributed.expert_parallel.expert_weights import ep_layer_class_by_model_type
from src.distributed.expert_parallel.hub_conversion import resolve_conversion_steps
from src.distributed.expert_parallel.lazy_loader import EPWeightPlanner, WeightAction, build_family_key_mapping
from src.models.loading.lazy_safetensors.conversion import (
    Concat,
    Deinterleave,
    PermuteRope,
    TakeChunk,
    convert_disk_keys,
    translate_converter,
    translate_renaming,
)
from src.models.loading.lazy_safetensors.weights import SafetensorsWeightLoader, WeightPlan, build_key_mapping
from src.models.loading.model_preparation import resolve_auto_model_class

E, M, H = 4, 6, 8


def _interleaved(gate: torch.Tensor, up: torch.Tensor, dim: int) -> torch.Tensor:
    """The vendor layout: rows of ``gate`` and ``up`` alternating along ``dim``."""
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
    """The op is a reimplementation of transformers' Interleave forward — pinned equal to it."""
    tensor = torch.randn(E, 2 * M, H)
    theirs = Interleave(dim=1).convert({"src": tensor}, ["src"], ["tgt"])["tgt"]
    assert torch.equal(Deinterleave(dim=1)(tensor), theirs)


def test_take_chunk_fan_out():
    tensor = torch.randn(2 * M, H)
    assert torch.equal(TakeChunk(0, 0, 2)(tensor), tensor[:M])
    assert torch.equal(TakeChunk(0, 1, 2)(tensor), tensor[M:])


def test_permute_rope_matches_transformers_op():
    """The op is a reimplementation of transformers' PermuteForRope forward — pinned equal to it on
    a 2-D weight and a 1-D bias, in both directions."""
    n_heads, rows = 4, 32
    config = SimpleNamespace(vision_config=SimpleNamespace(num_attention_heads=n_heads))
    for tensor in (torch.randn(rows, H), torch.randn(rows)):
        for inverse in (False, True):
            theirs = PermuteForRope(subconfig_key="vision_config", permute_layer_names=["q_proj"], inverse=inverse)
            expected = theirs.convert({"attn.q_proj.weight": tensor}, [], [], config=config)["attn.q_proj.weight"]
            assert torch.equal(PermuteRope(n_heads, inverse=inverse)(tensor), expected)
    # Anti-vacuity: the permutation is not the identity on a non-trivial tensor.
    assert not torch.equal(
        PermuteRope(n_heads)(torch.arange(rows * H, dtype=torch.float32).view(rows, H)),
        torch.arange(rows * H, dtype=torch.float32).view(rows, H),
    )


def _steps(entries, config=None, scope=""):
    """Translate hand-built entries through the resolver's own translators."""
    return tuple(
        translate_renaming(entry, scope)
        if isinstance(entry, WeightRenaming)
        else translate_converter(entry, config=config, scope=scope)
        for entry in entries
    )


_ENTRIES = [
    WeightRenaming(source_patterns=r"model\.llm\.layers", target_patterns="model.language_model.layers"),
    WeightConverter(
        source_patterns="mlp.experts.w13_weight",
        target_patterns=["mlp.experts.gate_up_proj"],
        operations=[Chunk(dim=1)],  # stands in for Interleave on images that lack it; ops are per-entry
    ),
    WeightConverter(
        source_patterns="mlp.shared_w13_weight",
        target_patterns=["mlp.shared.gate_proj.weight", "mlp.shared.up_proj.weight"],
        operations=[Chunk(dim=0)],
    ),
    WeightRenaming(source_patterns=r"attn\.wq_du", target_patterns="self_attn.q_proj"),
]


def test_walker_chains_renames_and_fans_out():
    entries = [e for e in _ENTRIES if not (isinstance(e, WeightConverter) and len(e.target_patterns) == 1)]
    steps = _steps(entries)
    keys = [
        "model.llm.layers.3.attn.wq_du.weight",
        "model.llm.layers.3.mlp.shared_w13_weight",
        "model.norm.weight",
    ]
    converted = convert_disk_keys(keys, steps)

    # Two renames compose on one key (namespace + attention spelling).
    assert converted["model.llm.layers.3.attn.wq_du.weight"] == (
        ("model.language_model.layers.3.self_attn.q_proj.weight", ()),
    )
    # Fan-out: one disk tensor, two targets, each with its chunk op.
    targets = dict(converted["model.llm.layers.3.mlp.shared_w13_weight"])
    assert set(targets) == {
        "model.language_model.layers.3.mlp.shared.gate_proj.weight",
        "model.language_model.layers.3.mlp.shared.up_proj.weight",
    }
    assert targets["model.language_model.layers.3.mlp.shared.gate_proj.weight"] == (TakeChunk(0, 0, 2),)
    assert targets["model.language_model.layers.3.mlp.shared.up_proj.weight"] == (TakeChunk(0, 1, 2),)
    # Untouched keys are absent — the walker is a no-op for canonical checkpoints.
    assert "model.norm.weight" not in converted


def test_walker_applies_renames_declared_after_the_converter():
    steps = _steps(
        [
            WeightConverter(
                source_patterns="mlp.w13.weight",
                target_patterns=["mlp.gate_proj.weight", "mlp.up_proj.weight"],
                operations=[Chunk(dim=0)],
            ),
            WeightRenaming(source_patterns=r"^decoder\.", target_patterns="model."),
        ]
    )
    converted = convert_disk_keys(["decoder.mlp.w13.weight"], steps)
    assert {k for k, _ in converted["decoder.mlp.w13.weight"]} == {
        "model.mlp.gate_proj.weight",
        "model.mlp.up_proj.weight",
    }


def test_unsupported_ops_are_refused():
    entry = WeightConverter(
        source_patterns="a", target_patterns=["b", "c", "d"], operations=[Chunk(dim=0), Chunk(dim=1)]
    )
    with pytest.raises(ValueError, match="Unsupported conversion ops"):
        translate_converter(entry)
    # A single-source Concatenate has nothing to concatenate — it is not the fan-in form.
    with pytest.raises(ValueError, match="Unsupported conversion ops"):
        translate_converter(WeightConverter(source_patterns="a", target_patterns="b", operations=[Concatenate(dim=0)]))


def test_multi_source_entry_translates_to_a_fan_in():
    """GLM-5 Next's ``q/k/v_conv1d → conv1d`` shape: N sources, one Concatenate, one target."""
    entry = WeightConverter(
        source_patterns=["attn.q_conv1d.weight", "attn.k_conv1d.weight", "attn.v_conv1d.weight"],
        target_patterns="attn.conv1d.weight",
        operations=[Concatenate(dim=0)],
    )
    convert = translate_converter(entry)
    assert [p.pattern for p in convert.patterns] == entry.source_patterns
    assert convert.concat_dim == 0
    assert convert.targets == (("attn.conv1d.weight", ()),)

    # Any other op pipeline on a multi-source entry is refused (transformers itself rejects a
    # many-to-many entry at construction, so a second target never reaches the translator).
    bad = WeightConverter(
        source_patterns=["a", "b"], target_patterns="c", operations=[Concatenate(dim=0), Chunk(dim=1)]
    )
    with pytest.raises(ValueError, match="multi-source"):
        translate_converter(bad)


def test_chunk_then_permute_for_rope_lands_on_the_named_targets():
    """Step-3.5 vision's ``in_proj_weight`` shape: chunk into q/k/v, permute q and k only, with the
    head count read off the config's sub-config — and refused without a config."""
    entry = WeightConverter(
        source_patterns=r"attn\.in_proj_weight",
        target_patterns=[r"attn.q_proj.weight", r"attn.k_proj.weight", r"attn.v_proj.weight"],
        operations=[
            Chunk(dim=0),
            PermuteForRope(subconfig_key="vision_config", permute_layer_names=["q_proj", "k_proj"]),
        ],
    )
    config = SimpleNamespace(vision_config=SimpleNamespace(num_attention_heads=4))
    targets = dict(translate_converter(entry, config=config).targets)
    assert targets["attn.q_proj.weight"] == (TakeChunk(0, 0, 3), PermuteRope(4))
    assert targets["attn.k_proj.weight"] == (TakeChunk(0, 1, 3), PermuteRope(4))
    assert targets["attn.v_proj.weight"] == (TakeChunk(0, 2, 3),)
    with pytest.raises(ValueError, match="needs the model config"):
        translate_converter(entry)


_FAN_IN = WeightConverter(
    source_patterns=["self_attn.q_conv1d.weight", "self_attn.k_conv1d.weight", "self_attn.v_conv1d.weight"],
    target_patterns="self_attn.conv1d.weight",
    operations=[Concatenate(dim=0)],
)


def test_walker_groups_a_fan_in_by_target_in_declared_source_order():
    """Members of one entry group by the target they produce (per layer); the first source's key
    carries the group as a leading Concat naming the others in DECLARED order, whatever the index
    order; the siblings are consumed (absent), and unrelated keys are untouched."""
    steps = _steps([_FAN_IN, WeightRenaming(source_patterns=r"^decoder\.", target_patterns="model.")])
    keys = [
        "decoder.layers.1.self_attn.v_conv1d.weight",
        "decoder.layers.0.self_attn.k_conv1d.weight",
        "decoder.layers.0.self_attn.q_conv1d.weight",
        "decoder.layers.1.self_attn.q_conv1d.weight",
        "decoder.layers.0.self_attn.v_conv1d.weight",
        "decoder.layers.1.self_attn.k_conv1d.weight",
        "decoder.layers.0.self_attn.o_proj.weight",
    ]
    converted = convert_disk_keys(keys, steps)
    for layer in (0, 1):
        q, k, v = (f"decoder.layers.{layer}.self_attn.{n}_conv1d.weight" for n in "qkv")
        assert converted[q] == ((f"model.layers.{layer}.self_attn.conv1d.weight", (Concat(0, (k, v)),)),)
        assert k not in converted and v not in converted
    assert converted["decoder.layers.0.self_attn.o_proj.weight"] == (("model.layers.0.self_attn.o_proj.weight", ()),)


def test_walker_refuses_a_partial_or_doubled_fan_in():
    steps = _steps([_FAN_IN])
    with pytest.raises(ValueError, match="truncated tensor"):
        convert_disk_keys(["l.self_attn.q_conv1d.weight", "l.self_attn.v_conv1d.weight"], steps)
    doubled = _steps(
        [WeightConverter(source_patterns=["a.*.x", "b.x"], target_patterns="c.x", operations=[Concatenate(dim=0)])]
    )
    with pytest.raises(ValueError, match="both match source"):
        convert_disk_keys(["a.1.x", "a.2.x", "b.x"], doubled)


def test_scoped_steps_see_only_their_sub_model():
    """transformers scopes a sub-model's entries under its module path (Step-3.7's
    ``step3p5_vision`` tower): the bare pattern ``\\.attn\\.`` must convert the tower's keys — with
    the base-model prefix or without it — and leave the text decoder's identical spelling alone."""
    vision = [
        WeightRenaming(source_patterns=r"\.attn\.", target_patterns=".self_attn."),
        WeightConverter(
            source_patterns=r"attn\.in_proj_weight",
            target_patterns=[r"attn.q_proj.weight", r"attn.v_proj.weight"],
            operations=[Chunk(dim=0)],
        ),
    ]
    steps = (
        *_steps([WeightRenaming(source_patterns=r"^vision_model\.", target_patterns="model.vision_model.")]),
        *_steps(vision, scope="vision_model"),
    )
    keys = [
        "vision_model.layers.0.attn.in_proj_weight",
        "model.vision_model.layers.0.attn.out.weight",
        "model.language_model.layers.0.attn.out.weight",
    ]
    converted = convert_disk_keys(keys, steps, base_prefix="model.")
    assert {k for k, _ in converted["vision_model.layers.0.attn.in_proj_weight"]} == {
        "model.vision_model.layers.0.self_attn.q_proj.weight",
        "model.vision_model.layers.0.self_attn.v_proj.weight",
    }
    assert converted["model.vision_model.layers.0.attn.out.weight"] == (
        ("model.vision_model.layers.0.self_attn.out.weight", ()),
    )
    assert "model.language_model.layers.0.attn.out.weight" not in converted


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

    plans = EPWeightPlanner(_EP()).build(weight_map, {}, model_keys, fanout=fanout)
    by_key = {p.model_key: p for p in plans}
    assert len(plans) == 3
    assert by_key["model.experts.gate_up_proj"].action == WeightAction.EXPERT_SHARD
    assert by_key["model.experts.gate_up_proj"].ops == (Deinterleave(dim=1),)
    assert by_key["model.shared.up_proj.weight"].ops == (TakeChunk(0, 1, 2),)

    bad = {"disk.expert": (("model.experts.gate_up_proj", (Deinterleave(dim=0),)),)}
    with pytest.raises(ValueError, match="expert shard axis"):
        EPWeightPlanner(_EP()).build(weight_map, {}, model_keys, fanout=bad)


class _EP:
    """Every field EPConfig declares for the planner, two of four experts on this rank. A stub short
    of one field models a config that cannot exist (they are set together at construction, filled
    together at finalization)."""

    expert_start_idx, expert_end_idx = 0, 2
    num_experts = 4


def test_planner_consumes_fan_in_siblings_and_checks_the_concat_axis():
    """Step-3.7's shape: ``moe.gate_proj + moe.up_proj → experts.gate_up_proj`` is ONE expert-sharded
    plan reading two disk keys; the sibling is neither planned nor unexpected, and a Concat on the
    expert axis is refused like any other shard-axis op."""
    gate, up = "l.moe.gate_proj.weight", "l.moe.up_proj.weight"
    weight_map = {gate: "a.safetensors", up: "b.safetensors"}
    model_keys = {"l.mlp.experts.gate_up_proj"}
    fanout = {gate: (("l.mlp.experts.gate_up_proj", (Concat(1, (up,)),)),)}

    (plan,) = EPWeightPlanner(_EP()).build(weight_map, {}, model_keys, fanout=fanout)
    assert plan.action == WeightAction.EXPERT_SHARD and plan.ops == (Concat(1, (up,)),)
    assert plan.disk_keys == (gate, up)

    on_axis = {gate: (("l.mlp.experts.gate_up_proj", (Concat(0, (up,)),)),)}
    with pytest.raises(ValueError, match="expert shard axis"):
        EPWeightPlanner(_EP()).build(weight_map, {}, model_keys, fanout=on_axis)


def test_materializer_reads_every_fan_in_source_through_the_expert_slice(tmp_path):
    """The fused expert tensor of a fan-in is this rank's slice of EACH source concatenated — never
    a whole source — and a sibling living in a shard the loader was not handed opens on demand."""
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
    loader = SafetensorsWeightLoader(str(tmp_path), ["a.safetensors"], device="cpu")
    loader._open()
    try:
        assert torch.equal(loader._materialize(plan), torch.cat([gate[1:3], up[1:3]], dim=1))
        # The expert-count guard covers the sibling too: a wider sibling must not be silently sliced.
        save_file({"l.moe.up_proj.weight": torch.randn(2 * E, M, H)}, str(tmp_path / "b.safetensors"))
        loader._close()
        loader._open()
        with pytest.raises(RuntimeError, match="config declares"):
            loader._materialize(plan)
    finally:
        loader._close()


def _shell(model_type: str):
    """The family's model on the meta device from its default config — zero bytes, no hub access —
    through the same ``resolve_auto_model_class`` the loaders use, so a composite family carries its
    vision/audio-tower sub-models too. A text sub-config with no generative class (``glm5_next_text``)
    builds the bare backbone ``AutoModel`` builds for embedding training; one that NO Auto mapping
    instantiates (``step3p5``, Step-3.7's text tower) exists only inside its composite, so no
    checkpoint is typed by it — the composite's own model_type drives the same layer class, and
    thereby the same steps, through this oracle."""
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


def test_resolution_is_declaration_gated():
    """Families that declare no conversion keys resolve to None — the pass cannot perturb them."""
    assert resolve_conversion_steps("qwen3_moe", _shell("qwen3_moe")) is None
    assert resolve_conversion_steps("", torch.nn.Module()) is None
    assert resolve_conversion_steps("glm4_moe_lite", _shell("glm4_moe_lite")) is None


def test_a_declared_key_that_resolves_to_nothing_raises(monkeypatch):
    """``None`` is the caller's "checkpoint is already canonical" answer, so a declaring family whose
    mapping is absent must not collapse into it — the loader would read the vendor namespace
    unconverted, the planner would skip every unmatched key, and the model would train from its
    meta-device init values with no error anywhere."""
    monkeypatch.setattr(hub_conversion, "get_checkpoint_conversion_mapping", lambda key: None)
    with pytest.raises(ValueError, match="declares hub-conversion key"):
        resolve_conversion_steps("inkling_mm_model", _shell("inkling_mm_model"))


def test_a_resolvable_key_still_returns_its_steps(monkeypatch):
    """Anti-vacuity: the raise is about an EMPTY resolution, not about declaring a key at all."""
    monkeypatch.setattr(hub_conversion, "get_checkpoint_conversion_mapping", lambda key: [_ENTRIES[0]])
    steps = resolve_conversion_steps("inkling_mm_model", _shell("inkling_mm_model"))
    # The entry must come back TRANSLATED, not merely counted: a resolution of the right arity that
    # dropped the rename would load the vendor namespace unconverted just the same.
    (step,) = steps
    assert isinstance(step, hub_conversion.Rename)
    assert step.pattern.pattern == _ENTRIES[0].source_patterns[0]
    assert step.replacement == _ENTRIES[0].target_patterns[0]


def test_a_non_declaring_family_is_untouched_by_an_absent_mapping(monkeypatch):
    """Anti-vacuity: the raise keys on the family's declaration, not on the mapping lookup — a family
    that declares nothing never looks one up."""
    monkeypatch.setattr(hub_conversion, "get_checkpoint_conversion_mapping", lambda key: None)
    assert resolve_conversion_steps("qwen3_moe", _shell("qwen3_moe")) is None


def test_a_sub_model_key_is_scoped_under_its_module_path(monkeypatch):
    """A declared key naming a SUB-model of the shell (Step-3.7's ``step3p5_vision`` tower) resolves
    its entries scoped to that module path while the root's stay unscoped — transformers' own
    scoping, which keeps the tower's bare ``\\.attn\\.`` rename off the text decoder's keys — and a
    PermuteForRope entry reads its head count off the shell's config."""
    entries = {
        "step3p7": [WeightRenaming(source_patterns=r"^vision_model\.", target_patterns="model.vision_model.")],
        "step3p5_vision": [
            WeightRenaming(source_patterns=r"\.attn\.", target_patterns=".self_attn."),
            WeightConverter(
                source_patterns=r"attn\.in_proj_weight",
                target_patterns=[r"attn.q_proj.weight", r"attn.k_proj.weight", r"attn.v_proj.weight"],
                operations=[
                    Chunk(dim=0),
                    PermuteForRope(subconfig_key="vision_config", permute_layer_names=["q_proj"]),
                ],
            ),
        ],
    }
    monkeypatch.setattr(hub_conversion, "get_checkpoint_conversion_mapping", entries.get)
    monkeypatch.setattr(ep_layer_class_by_model_type()["step3p7"], "_HUB_CONVERSION_KEYS", tuple(entries))
    shell = _shell("step3p7")

    root, attn, in_proj = resolve_conversion_steps("step3p7", shell)
    assert (root.scope, attn.scope, in_proj.scope) == ("", "vision_model", "vision_model")
    heads = shell.config.vision_config.num_attention_heads
    assert dict(in_proj.targets)["attn.q_proj.weight"] == (TakeChunk(0, 0, 3), PermuteRope(heads))


def _declaring_model_types() -> list[str]:
    """Every model_type whose EP layer class declares hub-conversion keys — read off the classes, so
    a family that grows a declaration is covered without touching this file."""
    return sorted(mt for mt, cls in ep_layer_class_by_model_type().items() if cls._HUB_CONVERSION_KEYS)


@cache
def _canonical_checkpoint_keys(model_type: str) -> tuple[str, ...]:
    """The key space a CANONICAL checkpoint of this family carries (see :func:`_shell`)."""
    return tuple(sorted(_shell(model_type).state_dict()))


@pytest.mark.parametrize("model_type", _declaring_model_types())
def test_canonical_checkpoint_keys_resume_untouched(model_type):
    """A canonical checkpoint of a declaring family must map every key to ITSELF, conversion-free.

    The steps are resolved and re-applied on every load of a declaring family, including an EP/PP
    RESUME from a checkpoint this toolkit itself wrote in the canonical namespace. transformers'
    conversion sources are not vendor-anchored (5.16's deepseek_v4 maps ``\\.norm\\.`` →
    ``.kv_norm.``, which also matches the canonical final ``model.norm.weight``), so the identity is
    enforced by the mapping's model-key fallback — the same validation transformers' own loader
    applies before accepting a rename — and this test drives the full mapping seam the loaders use.
    """
    layer_cls = ep_layer_class_by_model_type()[model_type]
    assert layer_cls._PER_EXPERT_UNFUSED_KEYS is None and not layer_cls._EXPORT_KEY_RENAMES, (
        f"{layer_cls.__name__} exports expert/replicated keys under names the live state dict does "
        f"not carry, so the state dict is no longer the checkpoint key space this test checks — "
        f"extend the oracle with the family's gathered-save layout."
    )
    model = _shell(model_type)
    steps = resolve_conversion_steps(model_type, model)
    assert steps, f"{model_type} declares {layer_cls._HUB_CONVERSION_KEYS} but resolved no steps"

    keys = list(_canonical_checkpoint_keys(model_type))
    disk_to_model, fanout = build_family_key_mapping(model, keys)

    assert not fanout, (
        f"{model_type}: {len(fanout)} canonical checkpoint key(s) still convert on a resume — the "
        f"model-key fallback did not catch them. First few: "
        f"{[(key, [target for target, _ in targets]) for key, targets in list(fanout.items())[:3]]}"
    )
    perturbed = {key: target for key, target in disk_to_model.items() if target != key}
    assert not perturbed, (
        f"{model_type}: {len(perturbed)} canonical checkpoint key(s) are renamed on a resume, so the "
        f"tensor lands on a key the checkpoint writer never used. First few: "
        f"{list(perturbed.items())[:3]}"
    )


class _WeightLeaf(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(1))


def _model_with_keys(paths: list[str]) -> torch.nn.Module:
    """A module tree whose state dict is exactly ``paths`` (each ending in ``.weight``)."""
    root = torch.nn.Module()
    for path in paths:
        parent = root
        parts = path.split(".")
        for name in parts[:-2]:
            child = parent._modules.get(name)
            if child is None:
                child = torch.nn.Module()
                parent.add_module(name, child)
            parent = child
        parent.add_module(parts[-2], _WeightLeaf())
    return root


def test_unanchored_source_falls_back_to_the_canonical_key():
    """Pin of the model-key fallback (transformers 5.16 semantics): a rename whose target misses the
    model's key space while the source key itself resolves is undone, and a genuine vendor key
    converting onto a real model key is NOT — the fallback must never mask a needed conversion."""
    model = _model_with_keys(["model.norm.weight", "model.layers.0.self_attn.kv_norm.weight"])
    steps = _steps(
        [
            WeightRenaming(source_patterns=r"\.attn\.", target_patterns=".self_attn."),
            WeightRenaming(source_patterns=r"\.norm\.", target_patterns=".kv_norm."),
        ]
    )
    disk_keys = [
        "model.norm.weight",  # canonical final norm — caught by the unanchored `\.norm\.`
        "model.layers.0.self_attn.kv_norm.weight",  # canonical, matches nothing
        "model.layers.0.attn.norm.weight",  # vendor spelling — must still convert
    ]
    disk_to_model, fanout = build_key_mapping(model, disk_keys, steps=steps)

    assert disk_to_model["model.norm.weight"] == "model.norm.weight"
    assert "model.norm.weight" not in fanout
    assert disk_to_model["model.layers.0.self_attn.kv_norm.weight"] == "model.layers.0.self_attn.kv_norm.weight"
    assert disk_to_model["model.layers.0.attn.norm.weight"] == "model.layers.0.self_attn.kv_norm.weight"
    assert fanout["model.layers.0.attn.norm.weight"] == (("model.layers.0.self_attn.kv_norm.weight", ()),)


@pytest.mark.parametrize("model_type", _declaring_model_types())
def test_the_canonical_key_oracle_reaches_the_walker(model_type):
    """Anti-vacuity: the test above passes because the patterns are vendor-anchored, not because the
    key list is empty or never reaches ``convert_disk_keys``."""
    drifted = _steps([WeightRenaming(source_patterns=r"experts\.gate_up_proj", target_patterns="experts.w13_weight")])
    assert convert_disk_keys(list(_canonical_checkpoint_keys(model_type)), drifted)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
