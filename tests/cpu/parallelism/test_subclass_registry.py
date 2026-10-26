#!/usr/bin/env python
"""CPU tests for the shared class-claim registry derivation.

``build_class_claim_map`` is the ONE subclass-tree walk behind the EP MoE-layer map, the CP
attention-wrapper map, the PP split-spec map, the head-transform spec map and the EP ``model_type``
roster. These tests pin its contract (own declaration only, deep tree, loud on a duplicate claim),
the noun each real registry names a clash with, that the EP side still reads through it rather than
a second copy, and that the EP and CP registries still route every wired family.

Run: ``python tests/cpu/parallelism/test_subclass_registry.py`` (or ``pytest -m cpu``).
"""

from __future__ import annotations

import pytest

from src.distributed.context_parallel.layers.registry import WRAPPER_CLASS_MAP, build_wrapper_class_map
from src.distributed.expert_parallel.expert_weights import ep_layer_class_by_model_type, ep_layer_classes
from src.distributed.expert_parallel.moe_layer.layer import EPMoELayerBase
from src.distributed.expert_parallel.patching import MOE_LAYER_MAP, build_moe_layer_map
from src.distributed.pipeline_parallel.split import build_pp_spec_map
from src.models.head_transform import build_head_transform_spec_map
from src.subclass_registry import build_class_claim_map, iter_subclasses


def test_walks_the_whole_subclass_tree_and_ignores_inherited_names():
    class Base:
        HF_MODULE_NAMES: tuple[str, ...] = ()

    class Intermediate(Base):
        HF_MODULE_NAMES = ("Mid",)

    class Leaf(Intermediate):  # inherits ("Mid",) but declares its own
        HF_MODULE_NAMES = ("Leaf",)

    class InheritsOnly(Intermediate):  # no own declaration → must NOT re-claim "Mid"
        pass

    mapping = build_class_claim_map(Base, "test")

    assert mapping == {"Mid": Intermediate, "Leaf": Leaf}
    assert InheritsOnly not in mapping.values()


def test_duplicate_claim_raises_with_the_kind_and_both_class_names():
    class Base:
        HF_MODULE_NAMES: tuple[str, ...] = ()

    class First(Base):
        HF_MODULE_NAMES = ("Shared",)

    class Second(Base):
        HF_MODULE_NAMES = ("Shared",)

    with pytest.raises(ValueError) as raised:
        build_class_claim_map(Base, "widget")

    walked = iter_subclasses(Base)
    assert set(walked) == {First, Second}
    assert str(raised.value) == f"widget 'Shared' claimed by both {walked[0].__name__} and {walked[1].__name__}"


def test_the_builder_reads_whatever_attribute_it_is_given():
    """The EP ``model_type`` roster is the second key kind: same walk, same own-declaration rule."""

    class Base:
        pass

    class Claims(Base):
        HF_MODEL_TYPES = ("alpha", "beta")

    class InheritsOnly(Claims):
        pass

    assert build_class_claim_map(Base, "model_type", attr="HF_MODEL_TYPES") == {"alpha": Claims, "beta": Claims}
    assert InheritsOnly not in build_class_claim_map(Base, "model_type", attr="HF_MODEL_TYPES").values()
    # A class declaring nothing under that attribute claims nothing (no crash, no empty-key entry).
    assert build_class_claim_map(Base, "HF MoE class name") == {}


def _model_type_roster() -> dict[str, type]:
    ep_layer_class_by_model_type.cache_clear()  # a cached map would never re-walk the tree
    return ep_layer_class_by_model_type()


@pytest.mark.parametrize(
    ("build", "noun"),
    [
        (build_moe_layer_map, "HF MoE class name"),
        (build_wrapper_class_map, "HF attention class name"),
        (build_pp_spec_map, "HF backbone class name"),
        (build_head_transform_spec_map, "HF causal-LM head class name"),
        (_model_type_roster, "model_type"),
    ],
    ids=["ep", "cp", "pp", "head-transform", "ep-model-type"],
)
def test_each_real_registry_names_its_clash(monkeypatch, build, noun):
    """A second class claiming a key already taken fails the real registry with the noun of what clashed."""
    key, owner = next(iter(build().items()))
    impostor = type("Impostor", (), {"HF_MODULE_NAMES": (key,), "HF_MODEL_TYPES": (key,)})
    monkeypatch.setattr("src.subclass_registry.iter_subclasses", lambda base: [*iter_subclasses(base), impostor])

    try:
        with pytest.raises(ValueError) as raised:
            build()
    finally:
        ep_layer_class_by_model_type.cache_clear()

    assert str(raised.value) == f"{noun} '{key}' claimed by both {owner.__name__} and Impostor"


def test_ep_layer_classes_is_the_shared_walk_not_a_second_copy():
    """``ep_layer_classes()`` is the EP-side NAME for the shared walk, not a re-implementation of it.

    Order included: the union helpers built on it (expert-weight roots, container attrs, fused keys)
    and the ``model_type`` roster must see exactly the tree the registry walks, or a family can be
    covered by one and missed by the other.
    """
    assert ep_layer_classes() == iter_subclasses(EPMoELayerBase)

    ep_layer_class_by_model_type.cache_clear()  # compare live trees, not a map cached before this test
    assert ep_layer_class_by_model_type() == build_class_claim_map(EPMoELayerBase, "model_type", attr="HF_MODEL_TYPES")


def test_ep_and_cp_registries_are_derived_and_cover_the_wired_families():
    assert build_moe_layer_map() == MOE_LAYER_MAP
    assert build_wrapper_class_map() == WRAPPER_CLASS_MAP

    # Every EP MoE family wired today must resolve; a dropped HF_MODULE_NAMES silently disables EP.
    expected_moe = {
        "GptOssMLP",
        "Cohere2MoeSparseMoeBlock",
        "Qwen3MoeSparseMoeBlock",
        "Qwen3_5MoeSparseMoeBlock",
        "BailingMoeV2SparseMoeBlock",
        "Glm4MoeLiteMoE",
        "Lfm2MoeSparseMoeBlock",
        "Mistral4MoE",
        "Gemma4TextExperts",
        "DeepseekV4SparseMoeBlock",
        "ZayaSparseMoeBlock",
    }
    assert expected_moe <= set(MOE_LAYER_MAP), f"EP families lost: {expected_moe - set(MOE_LAYER_MAP)}"
    assert all(issubclass(cls, EPMoELayerBase) for cls in MOE_LAYER_MAP.values())


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
