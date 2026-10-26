"""The embedding-tie readers: a config ties when any level declaring the flag ties, which on every
roster family is the text sub-config's own verdict, and the untie clears every declaring level."""

import pytest
import torch
from transformers import Qwen3_5MoeConfig

from halo.models.config_access import (
    TIE_WORD_EMBEDDINGS_FLAG,
    text_config,
    tie_flag_configs,
    ties_word_embeddings,
    untie_word_embeddings,
)
from tests.common.models import TINY_QWEN35_MOE_CONFIG, TINY_QWEN35_VISION_CONFIG
from tests.common.tiny_models import TINY_DENSE_FAMILIES, TINY_MOE_FAMILIES, TINY_MOE_VLM_FAMILIES, TinyFamily

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
    """Anti-vacuity: the sweep above must see tied and untied families and a two-level config."""
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


def test_a_config_without_the_flag_or_without_a_config_does_not_tie():
    class _RemoteCodeConfig:
        tie_word_embeddings = True

    assert ties_word_embeddings(_RemoteCodeConfig()) is True
    assert ties_word_embeddings(object()) is False
    assert ties_word_embeddings(None) is False


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
