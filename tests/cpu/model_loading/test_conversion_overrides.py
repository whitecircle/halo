#!/usr/bin/env python
"""A family's corrections to transformers' conversion mapping: registered at import on every path that reads the
mapping, and refused the moment transformers' entries stop being what they were written against."""

import subprocess
import sys

import pytest
import torch
from transformers import NemotronHForCausalLM
from transformers.conversion_mapping import get_checkpoint_conversion_mapping
from transformers.core_model_loading import PrefixChange, WeightRenaming

from halo.model_loading.checkpoint_keys import build_key_plan
from halo.model_loading.lazy_safetensors.conversion_selection import family_key_mapping
from halo.models.conversion_overrides import ConversionOverride, register_conversion_overrides
from tests.common.conversion_mapping import restored_conversion_mapping
from tests.common.tiny_models import TINY_UNWRAPPED_COMPOSITE_FAMILIES, tiny_family_model

# transformers' own ``nemotron_h`` entries, as the pinned release spells them.
_NEMOTRON_H_UPSTREAM = (
    (("backbone.",), ("model.",)),
    (("mixer.experts.*.up_proj.weight",), ("mixer.experts.up_proj",)),
    (("mixer.experts.*.down_proj.weight",), ("mixer.experts.down_proj",)),
)


def _with_composite_level_dropped(entries: list) -> list:
    return [PrefixChange(prefix_to_remove="language_model"), *entries]


@pytest.fixture
def restored_mapping():
    with restored_conversion_mapping():
        yield


def test_an_override_is_what_the_key_plan_then_reads(restored_mapping):
    """A composite checkpoint's keys reach the text class, through the entries upstream ships and the added level."""
    register_conversion_overrides(
        [ConversionOverride("nemotron_h", _NEMOTRON_H_UPSTREAM, _with_composite_level_dropped)]
    )
    plan = build_key_plan(
        ["language_model.backbone.norm_f.weight"],
        ["model.norm_f.weight"],
        get_checkpoint_conversion_mapping("nemotron_h"),
        "model",
    )
    assert plan.renamed_by_key == {"language_model.backbone.norm_f.weight": "model.norm_f.weight"}


def test_without_the_override_the_plan_misses_the_tower():
    plan = build_key_plan(
        ["language_model.backbone.norm_f.weight"],
        ["model.norm_f.weight"],
        get_checkpoint_conversion_mapping("nemotron_h"),
        "model",
    )
    assert "model.norm_f.weight" not in plan.renamed_by_key.values()


def test_the_lazy_plan_reads_the_override(restored_mapping):
    """A lazy load plans through the same mapping, so a text class meets a composite checkpoint's keys there too."""
    text_config = tiny_family_model(TINY_UNWRAPPED_COMPOSITE_FAMILIES["nemotron_h_omni"]).config.text_config
    with torch.device("meta"):
        model = NemotronHForCausalLM(text_config)
    keys = [
        "language_model.backbone.norm_f.weight",
        "language_model.backbone.embeddings.weight",
        "language_model.lm_head.weight",
    ]
    renames, _ = family_key_mapping(model, keys)
    assert not set(renames.values()) & {"model.norm_f.weight", "lm_head.weight"}, "reached with no override"
    register_conversion_overrides(
        [ConversionOverride("nemotron_h", _NEMOTRON_H_UPSTREAM, _with_composite_level_dropped)]
    )
    renames, _ = family_key_mapping(model, keys)
    assert set(renames.values()) == {"model.norm_f.weight", "model.embeddings.weight", "lm_head.weight"}, renames


def test_a_key_transformers_ships_nothing_for_takes_new_entries(restored_mapping):
    rename = WeightRenaming(source_patterns=r"vision_projector\.", target_patterns="")
    register_conversion_overrides([ConversionOverride("nemotron_h_omni", (), lambda entries: [*entries, rename])])
    assert [entry.source_patterns for entry in get_checkpoint_conversion_mapping("nemotron_h_omni")] == [
        [r"vision_projector\."]
    ]


@pytest.mark.parametrize(
    ("key", "upstream"),
    [
        ("nemotron_h", _NEMOTRON_H_UPSTREAM[:-1]),
        ("nemotron_h", ()),
        ("nemotron_h_omni", _NEMOTRON_H_UPSTREAM),
    ],
    ids=["entries_moved", "transformers_began_shipping_entries", "transformers_dropped_its_entries"],
)
def test_entries_that_moved_refuse_the_override_and_register_nothing(restored_mapping, key, upstream):
    before = get_checkpoint_conversion_mapping(key)
    with pytest.raises(RuntimeError, match="re-derive"):
        register_conversion_overrides([ConversionOverride(key, upstream, _with_composite_level_dropped)])
    after = get_checkpoint_conversion_mapping(key)
    assert repr(after) == repr(before), "a refused override must leave transformers' entries in place"


@pytest.mark.parametrize(
    "module",
    [
        "halo.model_loading.checkpoint_keys",
        # The load funnel, which scripts reach through ``model_preparation`` with no key plan in their import graph.
        "halo.model_loading.checkpoint_coverage",
    ],
)
def test_every_path_reading_the_mapping_imports_the_overrides(module):
    """Fresh interpreter, as this one has already imported the leaf."""
    probe = f"import sys, {module}; sys.exit(0 if 'halo.models.conversion_overrides' in sys.modules else 3)"
    assert subprocess.run([sys.executable, "-c", probe], capture_output=True, check=False).returncode == 0


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
