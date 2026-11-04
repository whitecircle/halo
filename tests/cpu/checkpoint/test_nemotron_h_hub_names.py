#!/usr/bin/env python
"""Nemotron's checkpoint names: every key of the release's index reaches a live tensor or a named drop on both load
routes, and a save writes the names the release spells, through a load record or none (a lazy load keeps none)."""

import re

import pytest
import torch
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForImageTextToText,
    NemotronH_Omni_Reasoning_V3,
    NemotronH_Omni_Reasoning_V3_Config,
    NemotronHConfig,
    NemotronHForCausalLM,
)
from transformers.conversion_mapping import get_checkpoint_conversion_mapping, register_checkpoint_conversion_mapping
from transformers.core_model_loading import PrefixChange

from halo.checkpoint.hub_layout import export_conversions, revert_conversions
from halo.model_loading.checkpoint_keys import build_key_plan, resolve_model_conversions
from halo.model_loading.lazy_safetensors.conversion_selection import family_key_mapping
from halo.model_loading.parallel.expert_fuser import ExpertFuser
from halo.models.conversion_overrides import CONVERSION_OVERRIDES
from halo.models.param_roles import fp32_pinned_names
from tests.common.conversion_mapping import restored_conversion_mapping
from tests.common.models import (
    NEMOTRON_H_RELEASE_LAYOUT,
    NEMOTRON_H_RELEASE_TEXT_CONFIG,
    NEMOTRON_H_RELEASE_VISION_CONFIG,
)
from tests.common.nemotron_h import (
    HUB_VISION_NORM,
    NEMOTRON_H_COMPOSITE,
    VISION_NORM,
    write_tiny_hub_checkpoint,
)
from tests.common.nemotron_h_hub_keys import (
    NEMOTRON_H_HUB_INDEX_SHA256,
    NEMOTRON_H_HUB_KEY_COUNT,
    hub_keys_digest,
    nemotron_h_hub_keys,
)
from tests.common.tiny_models import tiny_family_model

_MTP_KEYS = 1040
_INPUT_CONDITIONER_KEYS = 2
# No class builds the multi-token head or the conditioner the processor applies; the reattach tool carries the head.
_COMPOSITE_DROPS = re.compile(r"^language_model\.mtp\.|\.input_conditioner\.")
# What the text class leaves of the composite's checkpoint, the level above its keys dropped first.
_TEXT_DROPS = re.compile(r"^mtp\.|^language_model\.mtp\.|^vision_model\.|^vision_projector\.|^mlp1\.")


def _release_composite() -> NemotronH_Omni_Reasoning_V3:
    config = NemotronH_Omni_Reasoning_V3_Config(
        text_config=NEMOTRON_H_RELEASE_TEXT_CONFIG,
        vision_config=NEMOTRON_H_RELEASE_VISION_CONFIG,
        vision_hidden_size=NEMOTRON_H_RELEASE_VISION_CONFIG["hidden_size"],
        projector_hidden_size=20480,
        image_token_id=18,
    )
    with torch.device("meta"):
        return NemotronH_Omni_Reasoning_V3(config)


def _release_text_class() -> NemotronHForCausalLM:
    with torch.device("meta"):
        return NemotronHForCausalLM(NemotronHConfig(**NEMOTRON_H_RELEASE_TEXT_CONFIG))


def _plan(model, keys):
    conversions = resolve_model_conversions(model, add_legacy=True)
    return build_key_plan(keys, model.state_dict(), conversions, model.base_model_prefix or None)


def _dropped_keys(plan, drops: re.Pattern) -> set[str]:
    """Every checkpoint key reaching no live tensor. Raises on one ``drops`` does not name, and on a named one a class
    builds after all."""
    live_names = set(plan.live)
    dropped = {key for key, target in plan.renamed_by_key.items() if target not in live_names}
    for group in plan.groups:
        if not live_names.intersection(group.targets):
            dropped.update(key for key, _ in group.parts)
    unnamed = sorted(key for key in dropped if not drops.search(key))
    if unnamed:
        raise AssertionError(f"{len(unnamed)} keys reach no live tensor and no named drop: {unnamed[:6]}")
    kept = sorted(key for key in plan.renamed_by_key.keys() - dropped if drops.search(key))
    if kept:
        raise AssertionError(f"named drops reach live tensors: {kept[:6]}")
    return dropped


def _supplied_names(plan) -> set[str]:
    return {name for name in plan.keys if name in plan.live}


def _named_rename_targets(live: set[str]) -> set[str]:
    return {name for name in live if "lambda1" not in name and "mixer.experts." not in name}


def _loaded_composite(directory) -> NemotronH_Omni_Reasoning_V3:
    return AutoModelForImageTextToText.from_pretrained(directory, dtype=torch.float32)


def _loaded_text_class(directory) -> NemotronHForCausalLM:
    config = AutoConfig.from_pretrained(directory).text_config
    return AutoModelForCausalLM.from_pretrained(directory, config=config, dtype=torch.float32)


def _exported_names(model) -> set[str]:
    return set(revert_conversions(model, dict(model.state_dict()), export_conversions(model)))


def test_the_fixture_is_the_release_index():
    keys = nemotron_h_hub_keys()
    assert len(keys) == len(set(keys)) == NEMOTRON_H_HUB_KEY_COUNT
    assert hub_keys_digest(keys) == NEMOTRON_H_HUB_INDEX_SHA256


def test_every_release_key_reaches_the_composite_or_a_named_drop():
    model = _release_composite()
    plan = _plan(model, nemotron_h_hub_keys())
    dropped = _dropped_keys(plan, _COMPOSITE_DROPS)
    assert len(dropped) == _MTP_KEYS + _INPUT_CONDITIONER_KEYS
    absent = set(model.state_dict()) - _supplied_names(plan)
    # Radio's layer scale is the one thing the release does not store: two per vision block.
    assert absent == {
        f"vision_model.encoder.layer.{block}.layer_scale{scale}.lambda1"
        for block in range(NEMOTRON_H_RELEASE_VISION_CONFIG["num_hidden_layers"])
        for scale in (1, 2)
    }


def test_every_release_key_reaches_the_text_class_or_a_named_drop():
    model = _release_text_class()
    plan = _plan(model, nemotron_h_hub_keys())
    _dropped_keys(plan, _TEXT_DROPS)
    assert _supplied_names(plan) == set(model.state_dict()), "a tensor of the text class is absent from the release"


def test_the_expert_banks_are_planned_per_layer_across_every_expert():
    """Both routes fuse the same 512 per-expert keys into the bank the class holds."""
    for model, prefix in ((_release_composite(), "language_model."), (_release_text_class(), "")):
        groups = {group.targets[0]: group for group in _plan(model, nemotron_h_hub_keys()).live_groups()}
        bank = groups[f"{prefix}model.layers.1.mixer.experts.up_proj"]
        assert len(bank.parts) == NEMOTRON_H_RELEASE_TEXT_CONFIG["n_routed_experts"]


def test_the_accounting_refuses_an_unnamed_key_and_a_kept_drop():
    plan = _plan(_release_composite(), [*nemotron_h_hub_keys(), "language_model.backbone.layers.0.mixer.unknown"])
    with pytest.raises(AssertionError, match="no named drop"):
        _dropped_keys(plan, _COMPOSITE_DROPS)
    kept_head = build_key_plan(
        ["language_model.mtp.layers.0.norm.weight"], ["language_model.mtp.layers.0.norm.weight"], [], None
    )
    with pytest.raises(AssertionError, match="named drops reach live tensors"):
        _dropped_keys(kept_head, _COMPOSITE_DROPS)


@pytest.mark.parametrize(
    ("model_factory", "prefix"),
    [(_release_composite, "language_model."), (_release_text_class, "")],
    ids=["composite", "text_class"],
)
def test_the_lazy_plan_takes_every_release_key(model_factory, prefix):
    """The lazy loader, which this family's size requires, reads the same mapping: every live name but the expert banks
    and the layer scale has its key, the banks keep the per-expert names its fuser reads, the head's reach nothing."""
    model = model_factory()
    keys = nemotron_h_hub_keys()
    renames, converted = family_key_mapping(model, keys)
    live = set(model.state_dict())
    fanned_out = {target for targets in converted.values() for target, _ in targets}
    assert _named_rename_targets(live) <= {*renames.values(), *fanned_out}
    assert not live.intersection(renames[key] for key in keys if ".mtp." in key)
    expert = "layers.1.mixer.experts.7.up_proj.weight"
    assert renames[f"language_model.backbone.{expert}"] == f"{prefix}model.{expert}"


@pytest.mark.parametrize("model_factory", [_release_composite, _release_text_class], ids=["composite", "text_class"])
def test_a_rank_plans_its_experts_and_the_pinned_bias_across_the_release(model_factory):
    """Rank 3 of ``ep8`` on the real index: one bank per projection per MoE layer, and the head's experts left out."""
    model = model_factory()
    keys = nemotron_h_hub_keys()
    live = set(model.state_dict())
    renames, _ = family_key_mapping(model, keys)
    start, end = 192, 256
    fuser = ExpertFuser(start, end)
    tasks = fuser.detect_tasks(dict.fromkeys(keys, "shard.safetensors"), renames, live)

    moe_layers = NEMOTRON_H_RELEASE_LAYOUT.count("E")
    assert len(tasks) == 2 * moe_layers
    assert {task.kind for task in tasks} == {"up", "down"}
    for task in tasks:
        fuser.validate_task(task)
        assert sorted(task.experts) == list(range(start, end))
        assert task.model_key in live
    pinned = fp32_pinned_names(model, include_buffers=True)
    assert len([name for name in pinned if name.endswith("gate.e_score_correction_bias")]) == moe_layers


def test_the_vision_norm_loads_from_the_release_spelling(tmp_path):
    write_tiny_hub_checkpoint(tmp_path, norm_weight=2.5, norm_bias=0.25)
    model = _loaded_composite(tmp_path)
    assert torch.all(getattr(model, VISION_NORM).weight == 2.5) and torch.all(getattr(model, VISION_NORM).bias == 0.25)


def test_without_the_rename_the_norm_silently_keeps_its_identity(tmp_path):
    """The control: upstream declares the norm absent, so the load reports nothing and the tower runs an identity."""
    write_tiny_hub_checkpoint(tmp_path, norm_weight=2.5, norm_bias=0.25)
    with restored_conversion_mapping():
        override = next(entry for entry in CONVERSION_OVERRIDES if entry.key == "NemotronH_Omni_Reasoning_V3")
        entries = [
            entry
            for entry in get_checkpoint_conversion_mapping(override.key)
            if (tuple(entry.source_patterns), tuple(entry.target_patterns)) in override.upstream
        ]
        register_checkpoint_conversion_mapping(override.key, entries, overwrite=True)
        model, info = AutoModelForImageTextToText.from_pretrained(
            tmp_path, dtype=torch.float32, output_loading_info=True
        )
    assert torch.all(getattr(model, VISION_NORM).weight == 1.0) and torch.all(getattr(model, VISION_NORM).bias == 0.0)
    assert not info["missing_keys"], "the load reports nothing, so no gate can catch it"


def test_a_loaded_composite_exports_the_release_spelling(tmp_path):
    tensors = write_tiny_hub_checkpoint(tmp_path)
    names = _exported_names(_loaded_composite(tmp_path))
    assert {f"{HUB_VISION_NORM}.weight", f"{HUB_VISION_NORM}.bias"} <= names and f"{VISION_NORM}.weight" not in names
    fused = {re.sub(r"experts\.\d+\.(up|down)_proj\.weight", r"experts.\1_proj", key) for key in tensors}
    assert names - fused == {name for name in names if "lambda1" in name}, sorted(names - fused)[:6]


def test_a_composite_with_no_load_record_writes_the_level_once():
    """A lazy load leaves no ``_weight_conversions``. A ``PrefixChange`` scoped under ``language_model`` would write
    it twice, and only a served checkpoint would notice."""
    torch.manual_seed(0)
    model = tiny_family_model(NEMOTRON_H_COMPOSITE)
    assert getattr(model, "_weight_conversions", None) is None
    names = _exported_names(model)
    assert "language_model.backbone.norm_f.weight" in names and "language_model.lm_head.weight" in names
    assert not [name for name in names if name.startswith("language_model.language_model.")]


def test_a_prefix_change_there_would_write_it_twice():
    """Holds the test above to the mechanism: the plain transformers spelling of the same drop doubles the level."""
    with restored_conversion_mapping():
        entries = get_checkpoint_conversion_mapping("nemotron_h")
        register_checkpoint_conversion_mapping(
            "nemotron_h", [PrefixChange(prefix_to_remove="language_model"), *entries[1:]], overwrite=True
        )
        torch.manual_seed(0)
        names = _exported_names(tiny_family_model(NEMOTRON_H_COMPOSITE))
    assert "language_model.language_model.backbone.norm_f.weight" in names


def test_the_text_class_reads_the_composite_checkpoint_and_exports_its_own_level(tmp_path):
    tensors = write_tiny_hub_checkpoint(tmp_path)
    model = _loaded_text_class(tmp_path)
    assert torch.equal(model.model.norm_f.weight, tensors["language_model.backbone.norm_f.weight"])
    assert torch.equal(model.lm_head.weight, tensors["language_model.lm_head.weight"])
    for name in _exported_names(model):
        assert name.startswith(("backbone.", "lm_head.")), name


def test_a_text_class_with_no_load_record_exports_its_own_level():
    torch.manual_seed(0)
    composite = tiny_family_model(NEMOTRON_H_COMPOSITE)
    names = _exported_names(NemotronHForCausalLM(composite.config.text_config))
    assert "backbone.norm_f.weight" in names and "lm_head.weight" in names
    assert not [name for name in names if name.startswith("language_model.")]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
