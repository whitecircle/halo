import copy
import json
import os

import pytest
import torch
from transformers import (
    NemotronH_Omni_Reasoning_V3_Config,
    Qwen3_5MoeConfig,
    Qwen3_5MoeForCausalLM,
    Qwen3_5MoeTextConfig,
    Qwen3Config,
)

from halo.model_loading.model_preparation import auto_load_model, resolve_text_only_config
from halo.models.introspection.config_access import (
    TIE_WORD_EMBEDDINGS_FLAG,
    built_level_path,
    narrow_to_text_config,
    text_config,
    tie_flag_configs,
    ties_word_embeddings,
    untie_word_embeddings,
)
from tests.common.models import (
    TINY_NEMOTRON_H_CONFIG,
    TINY_QWEN35_MOE_CONFIG,
    TINY_QWEN35_VISION_CONFIG,
    TINY_RADIO_VISION_CONFIG,
)
from tests.common.tiny_models import (
    TINY_DENSE_FAMILIES,
    TINY_MOE_FAMILIES,
    TINY_MOE_VLM_FAMILIES,
    TinyFamily,
    build_tiny_family_checkpoint,
)

_ROSTER: dict[str, TinyFamily] = {**TINY_MOE_FAMILIES, **TINY_MOE_VLM_FAMILIES, **TINY_DENSE_FAMILIES}


@pytest.fixture(scope="module")
def roster_configs():
    torch.manual_seed(0)
    return {name: family.build({}).config for name, family in _ROSTER.items()}


@pytest.mark.parametrize("name", sorted(_ROSTER))
def test_every_roster_config_ties_as_its_text_config_declares(roster_configs, name):
    config = roster_configs[name]
    assert ties_word_embeddings(config) is bool(getattr(text_config(config), TIE_WORD_EMBEDDINGS_FLAG, False))


def test_the_roster_covers_both_verdicts_and_composite_levels(roster_configs):
    verdicts = {ties_word_embeddings(config) for config in roster_configs.values()}
    assert verdicts == {True, False}
    assert any(len(tie_flag_configs(config)) > 1 for config in roster_configs.values())


def _composite(*, top: bool, text: bool) -> Qwen3_5MoeConfig:
    config = Qwen3_5MoeConfig(text_config=dict(TINY_QWEN35_MOE_CONFIG), vision_config=dict(TINY_QWEN35_VISION_CONFIG))
    config.tie_word_embeddings = top
    config.get_text_config().tie_word_embeddings = text
    return config


@pytest.mark.parametrize(("top", "text"), [(False, True), (True, False), (True, True)])
def test_a_composite_ties_when_any_level_does(top, text):
    """transformers resolves the tie per submodel, so either level's declaration ties the pair."""
    config = _composite(top=top, text=text)
    assert set(tie_flag_configs(config)) == {"", "text_config."}
    assert ties_word_embeddings(config) is True


def test_untie_clears_every_declaring_level():
    config = _composite(top=True, text=True)
    untie_word_embeddings(config)
    assert not config.tie_word_embeddings
    assert not config.get_text_config().tie_word_embeddings
    assert ties_word_embeddings(config) is False
    assert ties_word_embeddings(_composite(top=False, text=False)) is False


def test_a_built_config_names_the_level_its_class_took():
    composite = _composite(top=False, text=False)
    assert built_level_path(composite, composite) == ""
    assert built_level_path(composite, copy.deepcopy(text_config(composite))) == "text_config."
    assert built_level_path(composite, copy.deepcopy(composite.vision_config)) == "vision_config."
    assert built_level_path(text_config(composite), copy.deepcopy(text_config(composite))) == ""


def test_a_built_config_of_no_level_reads_as_the_root():
    """Its levels are then compared path by path, so a lost one still raises."""
    assert built_level_path(_composite(top=False, text=False), Qwen3Config()) == ""


@pytest.mark.parametrize(
    ("top", "text", "narrowed"),
    [(True, False, True), (False, True, True), (True, True, True), (False, False, False)],
)
def test_narrowing_to_the_text_level_keeps_a_tie_stated_at_either_level(top, text, narrowed):
    composite = _composite(top=top, text=text)
    level = narrow_to_text_config(composite)
    assert level is text_config(composite)
    assert getattr(level, TIE_WORD_EMBEDDINGS_FLAG) is narrowed
    assert composite.tie_word_embeddings is top, "the composite states what it stated"


def test_a_sibling_towers_flag_does_not_narrow_onto_the_text_level():
    composite = _composite(top=False, text=False)
    composite.vision_config.tie_word_embeddings = True
    assert narrow_to_text_config(composite).tie_word_embeddings is False


def test_narrowing_a_config_with_no_text_level_returns_it_untouched():
    config = Qwen3_5MoeTextConfig(**TINY_QWEN35_MOE_CONFIG)
    assert narrow_to_text_config(config) is config
    assert config.tie_word_embeddings is False


def _nemotron_composite(*, top: bool, text: bool) -> NemotronH_Omni_Reasoning_V3_Config:
    config = NemotronH_Omni_Reasoning_V3_Config(
        text_config=dict(TINY_NEMOTRON_H_CONFIG), vision_config=dict(TINY_RADIO_VISION_CONFIG)
    )
    config.tie_word_embeddings = top
    config.get_text_config().tie_word_embeddings = text
    return config


def test_the_family_declared_text_route_narrows_with_the_tie():
    """Nemotron-H reaches its text class through this route, before any loader sees the config."""
    assert resolve_text_only_config(_nemotron_composite(top=True, text=False)).tie_word_embeddings is True
    assert resolve_text_only_config(_nemotron_composite(top=False, text=False)).tie_word_embeddings is False


def test_a_composite_the_auto_factory_narrows_stays_whole_with_the_tie_on_its_text_level():
    composite = _composite(top=True, text=False)
    assert resolve_text_only_config(composite) is composite
    assert text_config(composite).tie_word_embeddings is True
    assert composite.tie_word_embeddings is True

    untied = _composite(top=False, text=False)
    assert resolve_text_only_config(untied) is untied
    assert text_config(untied).tie_word_embeddings is False


def _save_tied(model) -> None:
    model.config.tie_word_embeddings = True
    model.tie_weights()


def test_a_text_only_load_of_a_tied_composite_ties_the_head(tmp_path):
    """Saved with no head key and the tie on the root alone, as ``save_pretrained`` leaves it. The head is excused from
    the coverage gate as a task head, so a text class reading only its own flag serves random weights in its place."""
    path = str(tmp_path / "composite")
    build_tiny_family_checkpoint(TINY_MOE_VLM_FAMILIES["qwen3_5_moe"], path, edit=_save_tied)
    with open(os.path.join(path, "config.json")) as fh:
        raw = json.load(fh)
    assert raw[TIE_WORD_EMBEDDINGS_FLAG] is True, "the premise: the root declares the tie"
    assert raw["text_config"][TIE_WORD_EMBEDDINGS_FLAG] is False, "the premise: the text level does not"

    model = auto_load_model(path, text_only=True, dtype=torch.bfloat16, attn_implementation="sdpa")

    assert isinstance(model, Qwen3_5MoeForCausalLM)
    assert model.get_output_embeddings().weight is model.get_input_embeddings().weight


def test_a_sibling_towers_flag_leaves_a_text_only_load_untied(tmp_path):
    """A ``config.json`` dumped with its defaults can state a tie on its vision tower. Read as the text class's own, it
    would save an untied head under a tied config, which every later load and the pipeline gate read as a tie."""
    path = str(tmp_path / "composite")
    build_tiny_family_checkpoint(TINY_MOE_VLM_FAMILIES["qwen3_5_moe"], path)
    with open(os.path.join(path, "config.json")) as fh:
        raw = json.load(fh)
    raw["vision_config"][TIE_WORD_EMBEDDINGS_FLAG] = True
    with open(os.path.join(path, "config.json"), "w") as fh:
        json.dump(raw, fh)

    model = auto_load_model(path, text_only=True, dtype=torch.bfloat16, attn_implementation="sdpa")

    assert model.config.tie_word_embeddings is False
    assert model.get_output_embeddings().weight is not model.get_input_embeddings().weight


def test_a_config_without_the_flag_or_without_a_config_does_not_tie():
    class _RemoteCodeConfig:
        tie_word_embeddings = True

    assert ties_word_embeddings(_RemoteCodeConfig()) is True
    assert ties_word_embeddings(object()) is False
    assert ties_word_embeddings(None) is False


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
