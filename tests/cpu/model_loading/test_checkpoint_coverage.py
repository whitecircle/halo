"""The initial load refuses a checkpoint that does not cover the live model. ``from_pretrained`` re-initializes missing
keys and reports through a logger silenced on non-logging ranks, so half a model trains on random weights."""

from __future__ import annotations

import os
import types

import pytest
import torch
import torch.nn as nn
from safetensors.torch import load_file, save_file
from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForSequenceClassification, Qwen3ForCausalLM

from halo.model_loading.checkpoint_coverage import (
    ALLOW_MISSING_CHECKPOINT_KEYS_ENV,
    from_pretrained_verified,
    unexpected_missing_keys,
    verify_checkpoint_coverage,
)
from halo.model_loading.model_preparation import auto_load_model
from halo.model_loading.remote_code import TIED_KEYS_WERE_LIST_ATTR, _fix_legacy_tied_weights_keys
from tests.common.repo import probe_findings


def _tiny_config(tie_word_embeddings: bool = False):
    return AutoConfig.for_model(
        "qwen3",
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=4,
        vocab_size=32,
        tie_word_embeddings=tie_word_embeddings,
    )


def _write_checkpoint(path, *, tie_word_embeddings: bool = False, drop: str | None = None) -> str:
    config = _tiny_config(tie_word_embeddings)
    model = AutoModelForCausalLM.from_config(config)
    # It drops the tied shadow safetensors refuses to write.
    model.save_pretrained(str(path), safe_serialization=True)
    weights = os.path.join(path, "model.safetensors")
    if drop is not None:
        state = {key: value for key, value in load_file(weights).items() if drop not in key}
        save_file(state, weights, metadata={"format": "pt"})
    return str(path)


def test_fixture_actually_drops_tensors(tmp_path):
    """Anti-vacuity: the truncated fixture really misses backbone tensors."""
    full = load_file(os.path.join(_write_checkpoint(tmp_path / "full"), "model.safetensors"))
    trunc = load_file(os.path.join(_write_checkpoint(tmp_path / "trunc", drop="layers.1."), "model.safetensors"))
    dropped = set(full) - set(trunc)
    assert dropped, "the truncated fixture dropped nothing"
    assert all(key.startswith("model.") for key in dropped), dropped


def test_stock_from_pretrained_still_does_not_raise(tmp_path):
    """The premise: transformers silently re-initializes."""
    path = _write_checkpoint(tmp_path / "trunc", drop="layers.1.")
    reloaded, info = AutoModelForCausalLM.from_pretrained(path, output_loading_info=True)
    assert info["missing_keys"], "transformers reported no missing keys for a truncated checkpoint"
    assert not any(value.is_meta for value in reloaded.state_dict().values())


def test_dropped_backbone_tensors_raise(tmp_path):
    path = _write_checkpoint(tmp_path / "trunc", drop="layers.1.")
    with pytest.raises(RuntimeError, match="absent from the checkpoint"):
        from_pretrained_verified(AutoModelForCausalLM, path)


def test_dropped_embedding_raises(tmp_path):
    """One backbone tensor is enough."""
    path = _write_checkpoint(tmp_path / "trunc", drop="embed_tokens")
    with pytest.raises(RuntimeError, match="embed_tokens"):
        from_pretrained_verified(AutoModelForCausalLM, path)


def test_complete_checkpoint_loads(tmp_path):
    path = _write_checkpoint(tmp_path / "full")
    model = from_pretrained_verified(AutoModelForCausalLM, path)
    assert isinstance(model, nn.Module)


@pytest.mark.parametrize("tie_word_embeddings", [False, True])
def test_sequence_classification_head_is_a_legitimate_absence(tmp_path, tie_word_embeddings):
    """The reward and classification entry points depend on this load succeeding."""
    path = _write_checkpoint(tmp_path / f"base{int(tie_word_embeddings)}", tie_word_embeddings=tie_word_embeddings)
    model = from_pretrained_verified(AutoModelForSequenceClassification, path, num_labels=2)
    assert model.score.weight.shape[0] == 2


def test_head_consuming_caller_refuses_a_base_checkpoint(tmp_path):
    """``halo/entrypoints/reward_scoring.py`` would otherwise score through a random ``score`` and report plausible
    numbers."""
    path = _write_checkpoint(tmp_path / "base")
    with pytest.raises(RuntimeError, match="score.weight"):
        from_pretrained_verified(AutoModelForSequenceClassification, path, excuse_task_head=False, num_labels=1)


def test_head_consuming_caller_still_loads_a_real_head(tmp_path):
    base = _write_checkpoint(tmp_path / "base")
    trained = str(tmp_path / "rm")
    AutoModelForSequenceClassification.from_pretrained(base, num_labels=1).save_pretrained(trained)

    model = from_pretrained_verified(AutoModelForSequenceClassification, trained, excuse_task_head=False)
    assert model.score.weight.shape[0] == 1


def test_tied_lm_head_is_a_legitimate_absence(tmp_path):
    """The tie supplies ``lm_head.weight``."""
    path = _write_checkpoint(tmp_path / "tied", tie_word_embeddings=True)
    assert "lm_head.weight" not in load_file(os.path.join(path, "model.safetensors"))
    model = from_pretrained_verified(AutoModelForCausalLM, path)
    assert model.lm_head.weight.data_ptr() == model.model.embed_tokens.weight.data_ptr()


def test_an_untied_lm_head_is_excused_like_any_other_head():
    """Self-announcing (loss pins at ``ln(vocab)``), and ``init_checkpoint_absent_modules`` draws this key for the
    lazy loaders; change only with that decision."""
    model = AutoModelForCausalLM.from_config(_tiny_config(tie_word_embeddings=False))
    assert unexpected_missing_keys(model, ["lm_head.weight"]) == []
    # The backbone next to it is not excused, so the rule does real work.
    assert unexpected_missing_keys(model, ["model.norm.weight"]) == ["model.norm.weight"]


def test_class_declared_ignores_are_honoured():
    """The escape hatch is the class's own declaration."""

    class _Head(nn.Module):
        base_model_prefix = "backbone"
        _keys_to_ignore_on_load_missing = None

        def __init__(self):
            super().__init__()
            self.backbone = nn.Linear(4, 4)
            self.extra = nn.Linear(4, 4)

    model = _Head()
    assert unexpected_missing_keys(model, ["backbone.weight"]) == ["backbone.weight"]

    type(model)._keys_to_ignore_on_load_missing = [r"backbone\.weight"]
    try:
        assert unexpected_missing_keys(model, ["backbone.weight"]) == []
    finally:
        type(model)._keys_to_ignore_on_load_missing = None


def test_a_bare_backbone_excuses_nothing():
    """``AutoModel`` is the backbone, so it has no outside to excuse."""

    class _Bare(nn.Module):
        base_model_prefix = "model"  # Declared, but there is no ``self.model``.
        _keys_to_ignore_on_load_missing = None

        def __init__(self):
            super().__init__()
            self.layer = nn.Linear(4, 4)

    assert unexpected_missing_keys(_Bare(), ["layer.weight"]) == ["layer.weight"]


def test_env_escape_hatch_downgrades_to_a_warning(monkeypatch):
    model = AutoModelForCausalLM.from_config(_tiny_config())
    missing = ["model.layers.1.mlp.down_proj.weight"]
    with pytest.raises(RuntimeError):
        verify_checkpoint_coverage(model, missing, source="unit")
    monkeypatch.setenv(ALLOW_MISSING_CHECKPOINT_KEYS_ENV, "1")
    verify_checkpoint_coverage(model, missing, source="unit")


def test_error_names_the_absent_tensors(tmp_path):
    """The logger that would carry the report is silenced on every non-logging rank."""
    path = _write_checkpoint(tmp_path / "trunc", drop="layers.1.mlp")
    with pytest.raises(RuntimeError) as excinfo:
        from_pretrained_verified(AutoModelForCausalLM, path)
    message = str(excinfo.value)
    assert "model.layers.1.mlp" in message, message
    assert ALLOW_MISSING_CHECKPOINT_KEYS_ENV in message, message


def test_gate_runs_on_the_reference_and_teacher_model_path(tmp_path):
    """A silently random reference shifts every logratio."""
    path = _write_checkpoint(tmp_path / "trunc", drop="layers.0.self_attn")
    with pytest.raises(RuntimeError, match="absent from the checkpoint"):
        auto_load_model(path, dtype=torch.float32)


def _legacy_list_tie_class() -> type:
    """Converted as a remote load converts it."""
    module = types.ModuleType("legacy_list_tie_remote")
    module.LegacyListTie = type(
        "LegacyListTie", (Qwen3ForCausalLM,), {"__module__": module.__name__, "_tied_weights_keys": ["lm_head.weight"]}
    )
    _fix_legacy_tied_weights_keys(module)
    return module.LegacyListTie


def test_a_tied_config_on_a_legacy_list_tie_is_refused(tmp_path):
    """The list-form tie converts to ``{}``, which ties nothing."""
    legacy = _legacy_list_tie_class()
    assert legacy._tied_weights_keys == {} and getattr(legacy, TIED_KEYS_WERE_LIST_ATTR)
    with pytest.raises(RuntimeError, match="declares tie_word_embeddings=True but shipped the legacy list-form"):
        from_pretrained_verified(legacy, _write_checkpoint(tmp_path / "tied", tie_word_embeddings=True))
    model = from_pretrained_verified(legacy, _write_checkpoint(tmp_path / "untied"))
    assert model.lm_head.weight.data_ptr() != model.model.embed_tokens.weight.data_ptr()


def test_importing_the_gate_makes_module_cache_copies_whole():
    script = """
import transformers.dynamic_module_utils as dynamic_module_utils
import halo.model_loading.checkpoint_coverage
print("FINDINGS:" + ("" if getattr(dynamic_module_utils.shutil, "_halo_whole_copy", False) else "copies not whole"))
"""
    assert not probe_findings(script, "FINDINGS:")


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
