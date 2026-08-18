#!/usr/bin/env python
"""CPU tests for the shared class-claim registry derivation.

``build_class_claim_map`` is the ONE subclass-tree walk behind the EP MoE-layer map, the CP
attention-wrapper map (both via ``build_hf_module_name_map``) and the EP ``model_type`` roster.
These tests pin its contract (own declaration only, deep tree, loud on a duplicate claim), that
the EP side still reads through it rather than a second copy, and that both real registries still
route every wired family.

Run: ``python tests/cpu/parallelism/test_module_registry.py`` (or ``pytest -m cpu``).
"""

from __future__ import annotations

import pytest

from src.distributed.module_registry import build_class_claim_map, build_hf_module_name_map, iter_subclasses


def test_walks_the_whole_subclass_tree_and_ignores_inherited_names():
    class Base:
        HF_MODULE_NAMES: tuple[str, ...] = ()

    class Intermediate(Base):
        HF_MODULE_NAMES = ("Mid",)

    class Leaf(Intermediate):  # inherits ("Mid",) but declares its own
        HF_MODULE_NAMES = ("Leaf",)

    class InheritsOnly(Intermediate):  # no own declaration → must NOT re-claim "Mid"
        pass

    mapping = build_hf_module_name_map(Base, "test")

    assert mapping == {"Mid": Intermediate, "Leaf": Leaf}
    assert InheritsOnly not in mapping.values()


def test_duplicate_claim_raises_with_both_class_names():
    class Base:
        HF_MODULE_NAMES: tuple[str, ...] = ()

    class First(Base):
        HF_MODULE_NAMES = ("Shared",)

    class Second(Base):
        HF_MODULE_NAMES = ("Shared",)

    with pytest.raises(ValueError, match="claimed by both"):
        build_hf_module_name_map(Base, "test")

    # The message must name the kind and both classes so the clash is findable.
    try:
        build_hf_module_name_map(Base, "widget")
    except ValueError as exc:
        assert "widget" in str(exc)
        assert First.__name__ in str(exc) and Second.__name__ in str(exc)


def test_the_builder_reads_whatever_attribute_it_is_given():
    """The EP ``model_type`` roster is the second key kind: same walk, same own-declaration rule."""

    class Base:
        pass

    class Claims(Base):
        HF_MODEL_TYPES = ("alpha", "beta")

    class InheritsOnly(Claims):
        pass

    assert build_class_claim_map(Base, "HF_MODEL_TYPES", "model_type") == {"alpha": Claims, "beta": Claims}
    assert InheritsOnly not in build_class_claim_map(Base, "HF_MODEL_TYPES", "model_type").values()
    # A class declaring nothing under that attribute claims nothing (no crash, no empty-key entry).
    assert build_class_claim_map(Base, "HF_MODULE_NAMES", "HF MoE class name") == {}


def test_the_builder_names_the_kind_it_was_given_on_a_duplicate_claim():
    class Base:
        pass

    class First(Base):
        HF_MODEL_TYPES = ("shared_type",)

    class Second(Base):
        HF_MODEL_TYPES = ("shared_type",)

    with pytest.raises(ValueError, match="model_type 'shared_type' claimed by both"):
        build_class_claim_map(Base, "HF_MODEL_TYPES", "model_type")


def test_ep_layer_classes_is_the_shared_walk_not_a_second_copy():
    """``ep_layer_classes()`` is the EP-side NAME for the shared walk, not a re-implementation of it.

    Order included: the union helpers built on it (expert-weight roots, container attrs, fused keys)
    and the ``model_type`` roster must see exactly the tree the registry walks, or a family can be
    covered by one and missed by the other.
    """
    from src.distributed.expert_parallel.base_layer import EPMoELayerBase
    from src.distributed.expert_parallel.expert_weights import ep_layer_class_by_model_type, ep_layer_classes

    assert ep_layer_classes() == iter_subclasses(EPMoELayerBase)

    ep_layer_class_by_model_type.cache_clear()  # compare live trees, not a map cached before this test
    assert ep_layer_class_by_model_type() == build_class_claim_map(EPMoELayerBase, "HF_MODEL_TYPES", "model_type")


def test_ep_and_cp_registries_are_derived_and_cover_the_wired_families():
    from src.distributed.context_parallel.layers.registry import WRAPPER_CLASS_MAP, build_wrapper_class_map
    from src.distributed.expert_parallel.base_layer import EPMoELayerBase
    from src.distributed.expert_parallel.patching import MOE_LAYER_MAP, build_moe_layer_map

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
