#!/usr/bin/env python
"""Every EP family must claim EVERY ``model_type`` spelling its checkpoints ship under.

``ep_layer_class_by_model_type`` is how an off-line consumer — the sharded-expert merge, the export
tooling, the save-time gate — resolves a ``config.json`` back to the class that produced its layout.
A family that declares only one of its spellings does not fail loudly: the composite/VLM wrapper
config simply resolves to ``None`` and the caller reports the family as unsupported, so a checkpoint
that trained fine cannot be merged or exported. That is the failure mode this pins, and it is exactly
the shape a new tower spelling arrives in (``lfm2_vl`` next to ``lfm2_moe``, ``gemma4`` next to
``gemma4_text``, ``mistral3`` — the VLM wrapper — next to ``mistral4``).

The claims are therefore pinned per class as a literal table. It is deliberately not derived from the
registry it checks (that would assert ``x == x``): adding a family, dropping a spelling, or moving a
spelling between classes all have to move this file, which is the point.

Run: ``python tests/cpu/parallelism/test_hf_model_types_completeness.py`` (or ``pytest -m cpu``).
"""

from __future__ import annotations

import pytest

from src.distributed.expert_parallel.expert_weights import (
    ep_layer_class_by_model_type,
    ep_layer_classes,
    resolve_ep_merge_layer_class,
)

# class name -> the ``model_type`` spellings the class declares ITSELF, in declaration order.
# Multi-spelling entries are the composite families: a text tower plus the wrapper config a
# multimodal (or renamed) checkpoint writes at the top level.
_EXPECTED_CLAIMS: dict[str, tuple[str, ...]] = {
    "EPBailingMoELayer": ("bailing_moe", "bailing_moe_linear", "bailing_hybrid"),
    "EPCohere2MoELayer": ("cohere2_moe", "cohere2_vision"),
    "EPDeepseekV4MoELayer": ("deepseek_v4",),
    "EPGemma4MoELayer": ("gemma4", "gemma4_text"),
    "EPGlm4MoELayer": ("glm4_moe_lite",),
    "EPGlm5NextMoELayer": ("glm5_next", "glm5_next_text"),
    "EPGptOssMoELayer": ("gpt_oss",),
    "EPInklingMoELayer": ("inkling_mm_model", "inkling_text"),
    "EPLagunaMoELayer": ("laguna",),
    "EPLfm2MoELayer": ("lfm2_moe", "lfm2_vl"),
    "EPMistral4MoELayer": ("mistral4", "mistral3"),
    "EPQwen3MoELayer": ("qwen3_moe",),
    "EPQwen3_5MoELayer": ("qwen3_5_moe", "qwen3_5_moe_text"),
    "EPStep3p7MoELayer": ("step3p7", "step3p5"),
    "EPZayaMoELayer": ("zaya",),
}

# The flat registry the resolvers actually consult, derived from the table above (one source).
_EXPECTED_MAPPING = {
    model_type: cls_name for cls_name, spellings in _EXPECTED_CLAIMS.items() for model_type in spellings
}

assert len(_EXPECTED_MAPPING) == sum(len(s) for s in _EXPECTED_CLAIMS.values()), "duplicate spelling in the table"


def _own_model_types(cls) -> tuple[str, ...]:
    """What the registry reads: the class's OWN declaration, never an inherited tuple."""
    return tuple(vars(cls).get("HF_MODEL_TYPES", ()))


def _declaring_classes() -> dict[str, type]:
    """Registered classes that claim an HF module — i.e. real families, not intermediate bases."""
    return {cls.__name__: cls for cls in ep_layer_classes() if vars(cls).get("HF_MODULE_NAMES", ())}


def test_the_roster_of_declaring_families_is_exactly_the_pinned_one():
    """A family added (or deleted) without its ``model_type`` spellings lands here first."""
    assert set(_declaring_classes()) == set(_EXPECTED_CLAIMS)


@pytest.mark.parametrize("cls_name", sorted(_EXPECTED_CLAIMS), ids=sorted(_EXPECTED_CLAIMS))
def test_each_family_claims_every_spelling_it_ships_under(cls_name):
    cls = _declaring_classes()[cls_name]
    assert _own_model_types(cls) == _EXPECTED_CLAIMS[cls_name], (
        f"{cls_name} now declares HF_MODEL_TYPES={_own_model_types(cls)}; a dropped spelling makes "
        f"every checkpoint written under it unresolvable for merge/export"
    )


def test_the_flat_registry_matches_the_declarations():
    """The union the resolvers read must contain exactly the pinned spellings and nothing more —
    a spelling that silently migrated to another class would still resolve, to the wrong layout."""
    assert {key: cls.__name__ for key, cls in ep_layer_class_by_model_type().items()} == _EXPECTED_MAPPING


_COMPOSITE = {name: spellings for name, spellings in _EXPECTED_CLAIMS.items() if len(spellings) > 1}
assert _COMPOSITE, "no composite family in the table — the both-towers check below would be vacuous"


@pytest.mark.parametrize("cls_name", sorted(_COMPOSITE), ids=sorted(_COMPOSITE))
def test_composite_families_resolve_from_every_tower(cls_name):
    """The user-facing consequence: each spelling must reach the SAME class through the public
    resolver, so a top-level wrapper config and its text tower export identically."""
    resolved = {spelling: resolve_ep_merge_layer_class(spelling) for spelling in _COMPOSITE[cls_name]}
    assert all(cls is not None for cls in resolved.values()), (
        f"{cls_name}: {[s for s, c in resolved.items() if c is None]} resolve to no EP layer class"
    )
    assert {cls.__name__ for cls in resolved.values()} == {cls_name}


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
