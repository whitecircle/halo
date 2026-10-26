#!/usr/bin/env python
"""Each MoE family is one spec, held by exactly one EP layer class, and the spec index refuses a shared claim.

Run: ``python tests/cpu/models/test_moe_family_specs.py`` (or ``pytest -m cpu``).
"""

import pytest

from src.distributed.expert_parallel.families.roster import ep_layer_classes
from src.models.moe import family_specs
from src.models.moe.family_specs import (
    LAGUNA,
    MOE_FAMILY_SPECS,
    QWEN3,
    MoEFamilySpec,
    find_moe_family_spec,
    find_moe_family_spec_for_config,
    wraps_experts,
)


def _family_classes() -> list[type]:
    """Toolkit EP layer classes declaring their own spec; test doubles join the same subclass tree."""
    return [cls for cls in ep_layer_classes() if "FAMILY" in vars(cls) and cls.__module__.startswith("src.")]


def test_every_spec_is_held_by_exactly_one_family_class():
    held = [cls.FAMILY for cls in _family_classes()]
    assert len(held) == len(set(held)), "two EP layer classes declare one spec"
    assert set(held) == set(MOE_FAMILY_SPECS)


@pytest.mark.parametrize("cls", _family_classes(), ids=lambda cls: cls.__name__)
def test_the_live_balancing_slot_names_the_spec_slot(cls):
    """The live-tree slot a family adopts is its spec's first spelling and the rest are its hub renames; a drift
    would adopt one tensor and save another key."""
    slot = cls.FAMILY.balancing_slot
    if slot:
        assert set(slot.attrs) == {slot.attrs[0], cls.FAMILY.to_hub_key(slot.attrs[0])}


def test_a_model_type_two_specs_claim_is_refused():
    with pytest.raises(ValueError, match="'qwen3_moe' is claimed by two MoE family specs"):
        family_specs._index_by_model_type((QWEN3, MoEFamilySpec(model_types=("qwen3_moe",))))


def test_lookups_resolve_every_spelling_and_the_text_sub_config():
    assert all(find_moe_family_spec(mt) is spec for spec in MOE_FAMILY_SPECS for mt in spec.model_types)
    assert find_moe_family_spec("qwen3_next") is None
    assert find_moe_family_spec(None) is None

    class _Text:
        model_type = "qwen3_moe"
        num_experts = 8

    class _Composite:
        model_type = "some_multimodal_wrapper"

        def get_text_config(self):
            return _Text()

    assert find_moe_family_spec_for_config(_Composite()) is QWEN3
    assert wraps_experts(_Composite(), wrappers_requested=True)
    assert not wraps_experts(_Composite(), wrappers_requested=False)


def test_to_hub_key_renames_only_declared_spellings():
    assert LAGUNA.to_hub_key("gate.e_score_correction_bias") == "experts.e_score_correction_bias"
    assert QWEN3.to_hub_key("gate.weight") == "gate.weight"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
