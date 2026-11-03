#!/usr/bin/env python
"""Every EP family claims every ``model_type`` spelling its checkpoints ship under, or a trained checkpoint cannot be
merged or exported. A literal table, since deriving it from the registry would assert ``x == x``."""

from __future__ import annotations

import pytest

from halo.distributed.expert_parallel.families.roster import ep_layer_classes
from halo.models.moe.family_specs import find_moe_family_spec
from tests.common.rosters import ep_layer_class_by_model_type

# Multi-spelling entries are composite families: a text tower plus the wrapper config a multimodal checkpoint writes.
_EXPECTED_CLAIMS: dict[str, tuple[str, ...]] = {
    "EPBailingMoELayer": ("bailing_moe", "bailing_moe_linear", "bailing_hybrid"),
    "EPCohere2MoELayer": ("cohere2_moe", "cohere2_vision"),
    "EPDeepseekV4MoELayer": ("deepseek_v4",),
    "EPGemma4MoELayer": ("gemma4", "gemma4_text"),
    "EPGlm4MoELayer": ("glm4_moe_lite",),
    "EPGlm5NextMoELayer": ("glm5_next", "glm5_next_text"),
    "EPGptOssMoELayer": ("gpt_oss",),
    "EPInklingMoELayer": ("inkling_text", "inkling_mm_model"),
    "EPLagunaMoELayer": ("laguna",),
    "EPLfm2MoELayer": ("lfm2_moe", "lfm2_vl"),
    "EPMistral4MoELayer": ("mistral4", "mistral3"),
    "EPQwen3MoELayer": ("qwen3_moe",),
    "EPQwen3_5MoELayer": ("qwen3_5_moe", "qwen3_5_moe_text"),
    "EPQwen3NextMoELayer": ("qwen3_next",),
    "EPQwen4ExpMoELayer": ("qwen4_exp", "qwen4_exp_text"),
    "EPStep3p7MoELayer": ("step3p7", "step3p5"),
    "EPZayaMoELayer": ("zaya",),
}

# The flat registry the resolvers consult, derived from the table above.
_EXPECTED_MAPPING = {
    model_type: cls_name for cls_name, spellings in _EXPECTED_CLAIMS.items() for model_type in spellings
}

assert len(_EXPECTED_MAPPING) == sum(len(s) for s in _EXPECTED_CLAIMS.values()), "duplicate spelling in the table"


def _own_model_types(cls) -> tuple[str, ...]:
    """The class's own spec, never an inherited one."""
    family = vars(cls).get("FAMILY")
    return family.model_types if family else ()


def _declaring_classes() -> dict[str, type]:
    """Real families, not intermediate bases."""
    return {cls.__name__: cls for cls in ep_layer_classes() if vars(cls).get("HF_MODULE_NAMES", ())}


def test_the_roster_of_declaring_families_is_exactly_the_pinned_one():
    assert set(_declaring_classes()) == set(_EXPECTED_CLAIMS)


@pytest.mark.parametrize("cls_name", sorted(_EXPECTED_CLAIMS), ids=sorted(_EXPECTED_CLAIMS))
def test_each_family_claims_every_spelling_it_ships_under(cls_name):
    cls = _declaring_classes()[cls_name]
    assert _own_model_types(cls) == _EXPECTED_CLAIMS[cls_name], (
        f"{cls_name} now declares FAMILY.model_types={_own_model_types(cls)}; a dropped spelling makes "
        f"every checkpoint written under it unresolvable for merge/export"
    )


def test_the_flat_registry_matches_the_declarations():
    """A spelling that migrated to another class would still resolve, to the wrong layout."""
    assert {key: cls.__name__ for key, cls in ep_layer_class_by_model_type().items()} == _EXPECTED_MAPPING


_COMPOSITE = {name: spellings for name, spellings in _EXPECTED_CLAIMS.items() if len(spellings) > 1}
assert _COMPOSITE, "no composite family in the table — the both-towers check below would be vacuous"


@pytest.mark.parametrize("cls_name", sorted(_COMPOSITE), ids=sorted(_COMPOSITE))
def test_composite_families_resolve_from_every_tower(cls_name):
    resolved = {spelling: find_moe_family_spec(spelling) for spelling in _COMPOSITE[cls_name]}
    assert all(spec is _declaring_classes()[cls_name].FAMILY for spec in resolved.values()), (
        f"{cls_name}: {[s for s, spec in resolved.items() if spec is None]} resolve to no family spec"
    )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
