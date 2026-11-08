#!/usr/bin/env python
"""Every class that declares an MTP tail is held to a head spec or a named refusal, read off transformers and the MoE
family roster, so a new family's tail cannot pass with no decision."""

import re
from types import SimpleNamespace

import pytest
import transformers
from transformers import CONFIG_MAPPING, AutoConfig
from transformers.models.auto.modeling_auto import (
    MODEL_FOR_CAUSAL_LM_MAPPING_NAMES,
    MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING_NAMES,
)

from halo.models.introspection.config_access import text_config
from halo.models.moe.family_specs import MOE_FAMILY_SPECS, NEMOTRON_H
from halo.models.mtp.spec import (
    MTP_HEAD_SPECS,
    MTP_LAYER_COUNT_FIELDS,
    MTP_REFUSALS,
    UNSURVEYED_TAIL,
    mtp_draft_layer_index,
    mtp_tail_depth,
    require_mtp_head_spec,
    stored_head_prefix,
    stored_tail_keys,
    tail_load_patterns,
)
from tests.common.nemotron_h import nemotron_h_lightning_config, nemotron_h_release_config
from tests.common.tiny_models import (
    TINY_DENSE_FAMILIES,
    TINY_MOE_FAMILIES,
    TINY_MOE_VLM_FAMILIES,
    tiny_family_model,
)

_MAPPED_CLASSES = {**MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING_NAMES, **MODEL_FOR_CAUSAL_LM_MAPPING_NAMES}
_CLAIMED = {model_type for spec in MTP_HEAD_SPECS for model_type in spec.model_types} | set(MTP_REFUSALS)
_TINY_ROSTER = {**TINY_MOE_FAMILIES, **TINY_MOE_VLM_FAMILIES, **TINY_DENSE_FAMILIES}


def _class_tail_patterns(class_name: str | None) -> tuple[str, ...]:
    model_class = getattr(transformers, class_name, None) if class_name else None
    return tail_load_patterns(getattr(model_class, "_keys_to_ignore_on_load_unexpected", None) or ())


def _unclaimed(declaring, claimed) -> list[str]:
    return sorted(model_type for model_type in declaring if model_type not in claimed)


def test_every_class_naming_a_tail_has_a_spec_or_a_named_refusal():
    declaring = {model_type for model_type, class_name in _MAPPED_CLASSES.items() if _class_tail_patterns(class_name)}
    assert len(declaring) >= 25, f"the sweep lost the tail-declaring classes: {sorted(declaring)}"
    assert _unclaimed(declaring, _CLAIMED) == []


def test_the_sweep_catches_a_class_with_no_decision():
    assert _unclaimed({"brand_new_family"}, _CLAIMED) == ["brand_new_family"]
    assert _unclaimed({"qwen3_5_moe"}, _CLAIMED) == []


@pytest.mark.parametrize("name", sorted(_TINY_ROSTER))
def test_a_tiny_family_resolves_to_its_spec_or_a_named_refusal(name):
    """A config carries the depth field where a class declares no pattern (Step-3.7, Bailing), so both are read."""
    config = tiny_family_model(_TINY_ROSTER[name]).config
    text = text_config(config)
    levels = {config.model_type, text.model_type}
    declares = bool(_class_tail_patterns(_MAPPED_CLASSES.get(config.model_type))) or any(
        hasattr(text, field) for field in MTP_LAYER_COUNT_FIELDS
    )
    if declares:
        assert levels & _CLAIMED, f"{name} declares a tail and has neither spec nor refusal"
    refused = levels & set(MTP_REFUSALS)
    if refused:
        with pytest.raises(ValueError, match=re.escape(MTP_REFUSALS[sorted(refused)[0]])):
            require_mtp_head_spec(config)
    elif levels & _CLAIMED:
        assert levels & set(require_mtp_head_spec(config).model_types)
    else:
        with pytest.raises(ValueError, match="no head to train"):
            require_mtp_head_spec(config)


@pytest.mark.parametrize("family", MOE_FAMILY_SPECS, ids=lambda family: family.model_types[0])
def test_every_moe_family_is_on_the_tiny_roster_the_sweep_reads(family):
    assert set(family.model_types) & set(_TINY_ROSTER), family.model_types


def test_the_families_a_design_decision_refuses_keep_their_named_reason():
    assert {"qwen4_exp", "glm5_next", "step3p7", "inkling_text", "deepseek_v4", "bailing_hybrid"} <= set(MTP_REFUSALS)
    assert not set(NEMOTRON_H.model_types) & set(MTP_REFUSALS)


@pytest.mark.parametrize("family", MOE_FAMILY_SPECS, ids=lambda family: family.model_types[0])
def test_no_family_on_the_roster_is_refused_as_outside_it(family):
    assert not [name for name in family.model_types if MTP_REFUSALS.get(name) == UNSURVEYED_TAIL], family.model_types


@pytest.mark.parametrize("release", [nemotron_h_lightning_config(), nemotron_h_release_config()["llm_config"]])
def test_a_nemotron_h_release_declares_the_chain_its_spec_builds(release):
    """Read off each release's own config, so a chain a later release changes refuses instead of building another."""
    readable = {key: value for key, value in release.items() if key != "time_step_limit"}
    config = CONFIG_MAPPING["nemotron_h"](**readable)
    spec = require_mtp_head_spec(config)
    assert tuple(config.mtp_layers_block_type) == spec.chain_kinds == ("full_attention", "moe")
    assert config.layer_types[mtp_draft_layer_index(config, "moe")] == "moe"
    assert config.layer_types[mtp_draft_layer_index(config)] == "full_attention"


@pytest.mark.parametrize("chain", [["moe", "attention"], ["attention"], ["attention", "moe", "moe"]])
def test_a_chain_no_release_ships_is_refused_by_name(chain):
    config = AutoConfig.for_model("nemotron_h", num_nextn_predict_layers=1, mtp_layers_block_type=chain)
    with pytest.raises(ValueError, match="mtp_layers_block_type declares"):
        require_mtp_head_spec(config)


@pytest.mark.parametrize(
    ("kinds", "missing"),
    [(["mamba", "moe"], "full_attention"), (["mamba", "attention"], "moe")],
    ids=["no_attention", "no_moe"],
)
def test_a_trunk_holding_no_layer_of_a_blocks_kind_is_refused_before_the_load(kinds, missing):
    """A depth cut can leave one out, and each block takes its class from such a layer."""
    config = AutoConfig.for_model("nemotron_h", num_nextn_predict_layers=1, layers_block_type=kinds)
    with pytest.raises(ValueError, match=f"trunk's last {missing} layer"):
        require_mtp_head_spec(config)


@pytest.mark.parametrize(
    ("level", "keys", "prefix"),
    [
        ("mtp", ["mtp.layers.0.norm.weight", "backbone.norm_f.weight"], "mtp."),
        ("mtp", ["language_model.mtp.layers.0.norm.weight"], "language_model.mtp."),
        ("language_model.mtp", ["mtp.layers.0.norm.weight"], "mtp."),
        ("language_model.mtp", ["language_model.mtp.layers.0.norm.weight"], "language_model.mtp."),
        ("mtp", ["backbone.norm_f.weight"], None),
    ],
)
def test_a_text_class_and_its_composite_read_the_tail_at_either_level(level, keys, prefix):
    assert stored_head_prefix(keys, AutoConfig.for_model("nemotron_h"), level) == prefix


def test_a_family_with_no_text_only_layout_reads_its_head_at_its_path_alone():
    config = AutoConfig.for_model("qwen3_next")
    assert stored_head_prefix(["mtp.fc.weight"], config, "mtp") == "mtp."
    assert stored_head_prefix(["language_model.mtp.fc.weight"], config, "mtp") is None


def test_the_composite_and_its_text_config_resolve_to_one_spec():
    composite = tiny_family_model(TINY_MOE_VLM_FAMILIES["qwen3_5_moe"]).config
    assert require_mtp_head_spec(composite) is require_mtp_head_spec(composite.text_config)


@pytest.mark.parametrize(
    ("fields", "depth"), [({}, 1), ({"mtp_num_hidden_layers": 1}, 1), ({"num_nextn_predict_layers": 2}, 2)]
)
def test_the_tail_depth_reads_the_declared_field_and_defaults_as_the_engines_do(fields, depth):
    config = AutoConfig.for_model("qwen3_next", **fields)
    assert mtp_tail_depth(config) == depth


def test_a_depth_no_release_ships_is_refused():
    with pytest.raises(ValueError, match="declares 2"):
        require_mtp_head_spec(AutoConfig.for_model("qwen3_next", mtp_num_hidden_layers=2))


def test_a_config_with_no_layer_running_the_draft_attention_is_refused_before_the_load():
    """The draft layer takes its class and call from such a layer, and a run has nothing to build it from."""
    config = AutoConfig.for_model("qwen3_next", num_hidden_layers=2, layer_types=["linear_attention"] * 2)
    with pytest.raises(ValueError, match=r"layer_types hold none: \['linear_attention'\]"):
        require_mtp_head_spec(config)


@pytest.mark.parametrize(
    ("layer_types", "index"),
    [
        (None, 5),
        (["linear_attention"] * 3 + ["full_attention"] + ["linear_attention"] * 2, 3),
        (["full_attention"] * 6, 5),
        # A list a config keeps past its trunk reads as the trunk's.
        (["full_attention"] + ["linear_attention"] * 5 + ["full_attention"], 0),
    ],
    ids=["no_kinds", "not_the_last_layer", "all_full", "past_the_trunk"],
)
def test_the_draft_layer_is_the_last_trunk_layer_running_its_attention(layer_types, index):
    text = SimpleNamespace(num_hidden_layers=6, layer_types=layer_types)
    assert mtp_draft_layer_index(text) == index


def test_a_class_pattern_that_names_no_tail_does_not_mark_keys():
    load_ignores = ("^mtp.*", "^model.visual.*")
    assert tail_load_patterns(load_ignores) == ("^mtp.*",)
    keys = ["mtp.fc.weight", "model.visual.blocks.0.weight", "lm_head.weight"]
    assert stored_tail_keys(keys, load_ignores) == ["mtp.fc.weight"]
    assert stored_tail_keys(
        ["model.layers.47.eh_proj.weight", "model.layers.46.mlp.weight"], (r"model\.layers\.47.*",)
    ) == ["model.layers.47.eh_proj.weight"]
    assert tail_load_patterns((r"layers\.\d+\.shared_head\.",)) == ()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
