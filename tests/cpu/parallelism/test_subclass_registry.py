#!/usr/bin/env python
"""CPU tests for the shared class-claim registry derivation.

``build_class_claim_map`` is the ONE subclass-tree walk behind the EP MoE-layer map, the CP
attention map and the head-transform spec map; the EP ``model_type`` roster walks the same tree off each class's
own ``FAMILY``. These tests pin its contract (own declaration only, deep tree, loud on a duplicate claim), the
noun each real registry names a clash with, and that the EP registry still routes every wired family.

Run: ``python tests/cpu/parallelism/test_subclass_registry.py`` (or ``pytest -m cpu``).
"""

from __future__ import annotations

import pytest

from src.distributed.expert_parallel.families.roster import MOE_LAYER_MAP, ep_layer_classes
from src.distributed.expert_parallel.moe_layer.layer import EPMoELayerBase
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
    """A second key kind: same walk, same own-declaration rule."""

    class Base:
        pass

    class Claims(Base):
        CLAIMED_KEYS = ("alpha", "beta")

    class InheritsOnly(Claims):
        pass

    assert build_class_claim_map(Base, "key", attr="CLAIMED_KEYS") == {"alpha": Claims, "beta": Claims}
    assert InheritsOnly not in build_class_claim_map(Base, "key", attr="CLAIMED_KEYS").values()
    # A class declaring nothing under that attribute claims nothing (no crash, no empty-key entry).
    assert build_class_claim_map(Base, "HF MoE class name") == {}


@pytest.mark.parametrize(
    ("build", "noun"),
    [
        (lambda: build_class_claim_map(EPMoELayerBase, "HF MoE class name"), "HF MoE class name"),
        (build_head_transform_spec_map, "HF causal-LM head class name"),
    ],
    ids=["ep", "head-transform"],
)
def test_each_real_registry_names_its_clash(monkeypatch, build, noun):
    """A second class claiming a key already taken fails the real registry with the noun of what clashed: an
    impostor family class re-declaring a taken spec included."""
    key, owner = next(iter(build().items()))
    impostor = type("Impostor", (), {"HF_MODULE_NAMES": (key,), "FAMILY": getattr(owner, "FAMILY", None)})

    def walk_with_impostor(base):
        return [*iter_subclasses(base), impostor]

    monkeypatch.setattr("src.subclass_registry.iter_subclasses", walk_with_impostor)

    with pytest.raises(ValueError) as raised:
        build()

    assert str(raised.value) == f"{noun} '{key}' claimed by both {owner.__name__} and Impostor"


def test_ep_layer_classes_is_the_shared_walk_not_a_second_copy():
    """``ep_layer_classes()`` is the EP-side NAME for the shared walk, order included, not a re-implementation of
    it: the config lookups built on it must see exactly the tree the registry walks."""
    assert ep_layer_classes() == iter_subclasses(EPMoELayerBase)


def test_ep_registry_is_derived_and_covers_the_wired_families():
    assert build_class_claim_map(EPMoELayerBase, "HF MoE class name") == MOE_LAYER_MAP

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
