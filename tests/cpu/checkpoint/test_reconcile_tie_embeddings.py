"""CPU tests for ``reconcile_tie_word_embeddings`` (save-time tie-consistency check).

The checker flips ``config.tie_word_embeddings`` to False when the saved lm_head and embed_tokens
diverged, so ``from_pretrained`` keeps the trained lm_head instead of re-tying it away. Keys are
resolved by suffix so wrapped/VLM layouts (``model.language_model.embed_tokens.weight`` etc.) are
covered — an exact-key lookup silently skips them and the reload re-ties a trained head away. A composite
(VLM) config may declare the tie on its text sub-config alone, which transformers ties all the same, so every
declaring level is read and cleared.
"""

from types import SimpleNamespace

import pytest
import torch
from transformers import Qwen3_5Config

from halo.checkpoint.writing import reconcile_tie_word_embeddings
from halo.models.config_access import tie_flag_configs


def _model(tied=True):
    return SimpleNamespace(config=SimpleNamespace(tie_word_embeddings=tied))


def test_standard_layout_diverged_flips_flag():
    model = _model()
    state = {"lm_head.weight": torch.ones(4, 8), "model.embed_tokens.weight": torch.zeros(4, 8)}
    reconcile_tie_word_embeddings(model, state)
    assert model.config.tie_word_embeddings is False


def test_standard_layout_equal_keeps_tie():
    model = _model()
    w = torch.ones(4, 8)
    state = {"lm_head.weight": w, "model.embed_tokens.weight": w.clone()}
    reconcile_tie_word_embeddings(model, state)
    assert model.config.tie_word_embeddings is True


def test_wrapped_vlm_layout_diverged_flips_flag():
    """A non-standard prefix must still be found: an exact-key lookup misses it, silently keeping
    the tie and re-tying the trained lm_head away on reload."""
    model = _model()
    state = {
        "language_model.lm_head.weight": torch.ones(4, 8),
        "model.language_model.embed_tokens.weight": torch.zeros(4, 8),
    }
    reconcile_tie_word_embeddings(model, state)
    assert model.config.tie_word_embeddings is False


def test_shape_mismatch_counts_as_diverged():
    model = _model()
    state = {"lm_head.weight": torch.ones(6, 8), "model.embed_tokens.weight": torch.zeros(4, 8)}
    reconcile_tie_word_embeddings(model, state)
    assert model.config.tie_word_embeddings is False


def test_absent_lm_head_skips():
    # A still-tied checkpoint carries only one of the pair (named_parameters dedupes tied params).
    model = _model()
    reconcile_tie_word_embeddings(model, {"model.embed_tokens.weight": torch.zeros(4, 8)})
    assert model.config.tie_word_embeddings is True


def test_ambiguous_suffix_warns_and_skips():
    # Two embed_tokens matches (e.g. a vision tower): resolution is ambiguous — never guess.
    model = _model()
    state = {
        "lm_head.weight": torch.ones(4, 8),
        "model.embed_tokens.weight": torch.zeros(4, 8),
        "vision_tower.embed_tokens.weight": torch.zeros(4, 8),
    }
    reconcile_tie_word_embeddings(model, state)
    assert model.config.tie_word_embeddings is True


def test_suffix_requires_dot_boundary():
    # "custom_lm_head.weight" must not match the "lm_head.weight" suffix across a name boundary.
    model = _model()
    state = {"custom_lm_head.weight": torch.ones(4, 8), "model.embed_tokens.weight": torch.zeros(4, 8)}
    reconcile_tie_word_embeddings(model, state)
    assert model.config.tie_word_embeddings is True


def test_a_tie_declared_only_on_the_text_config_is_cleared_on_every_level():
    config = Qwen3_5Config(tie_word_embeddings=False, text_config={"tie_word_embeddings": True})
    model = SimpleNamespace(config=config)
    state = {
        "lm_head.weight": torch.ones(4, 8),
        "model.language_model.embed_tokens.weight": torch.zeros(4, 8),
    }
    reconcile_tie_word_embeddings(model, state)
    assert {path: level.tie_word_embeddings for path, level in tie_flag_configs(config).items()} == {
        "": False,
        "text_config.": False,
    }


def test_untied_config_untouched():
    model = _model(tied=False)
    state = {"lm_head.weight": torch.ones(4, 8), "model.embed_tokens.weight": torch.zeros(4, 8)}
    reconcile_tie_word_embeddings(model, state)
    assert model.config.tie_word_embeddings is False


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
