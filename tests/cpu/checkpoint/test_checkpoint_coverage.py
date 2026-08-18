"""The initial model load must refuse a checkpoint that does not cover the live model.

``from_pretrained`` re-initializes every key it cannot find and reports it through the transformers
logger only — which ``setup_logging`` silences on every non-logging rank. A truncated or
wrong-architecture directory therefore trains half a model of random weights with no exception and no
output. :mod:`src.models.loading.checkpoint_coverage` turns that into a raise.

Both directions are pinned here, because a gate that over-rejects is as bad as one that never fires:

- a checkpoint missing backbone tensors must RAISE (``test_dropped_backbone_tensors_raise`` and the
  loader-level ``test_from_pretrained_verified_*``);
- a sequence-classification head absent from a base checkpoint, a tied ``lm_head``, and a class's own
  ``_keys_to_ignore_on_load_missing`` must still LOAD.

    python tests/cpu/checkpoint/test_checkpoint_coverage.py
"""

from __future__ import annotations

import os

import pytest
import torch
import torch.nn as nn
from safetensors.torch import load_file, save_file
from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForSequenceClassification

from src.models.loading.checkpoint_coverage import (
    ALLOW_MISSING_CHECKPOINT_KEYS_ENV,
    from_pretrained_verified,
    unexpected_missing_keys,
    verify_checkpoint_coverage,
)


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
    """A complete tiny causal-LM checkpoint, optionally missing every key containing ``drop``."""
    config = _tiny_config(tie_word_embeddings)
    model = AutoModelForCausalLM.from_config(config)
    # save_pretrained, not a raw state_dict dump: it drops the tied shadow safetensors refuses to write.
    model.save_pretrained(str(path), safe_serialization=True)
    weights = os.path.join(path, "model.safetensors")
    if drop is not None:
        state = {key: value for key, value in load_file(weights).items() if drop not in key}
        save_file(state, weights, metadata={"format": "pt"})
    return str(path)


def test_fixture_actually_drops_tensors(tmp_path):
    """Anti-vacuity: the truncated fixture must really be missing backbone tensors."""
    full = load_file(os.path.join(_write_checkpoint(tmp_path / "full"), "model.safetensors"))
    trunc = load_file(os.path.join(_write_checkpoint(tmp_path / "trunc", drop="layers.1."), "model.safetensors"))
    dropped = set(full) - set(trunc)
    assert dropped, "the truncated fixture dropped nothing"
    assert all(key.startswith("model.") for key in dropped), dropped


def test_stock_from_pretrained_still_does_not_raise(tmp_path):
    """The premise: transformers silently re-initializes. If this ever fails, the gate below became
    belt-and-braces rather than the only thing standing between a truncated directory and noise."""
    path = _write_checkpoint(tmp_path / "trunc", drop="layers.1.")
    reloaded, info = AutoModelForCausalLM.from_pretrained(path, output_loading_info=True)
    assert info["missing_keys"], "transformers reported no missing keys for a truncated checkpoint"
    assert not any(value.is_meta for value in reloaded.state_dict().values())


def test_dropped_backbone_tensors_raise(tmp_path):
    """A checkpoint missing a decoder layer must not load."""
    path = _write_checkpoint(tmp_path / "trunc", drop="layers.1.")
    with pytest.raises(RuntimeError, match="absent from the checkpoint"):
        from_pretrained_verified(AutoModelForCausalLM, path)


def test_dropped_embedding_raises(tmp_path):
    """The absence need not be a whole layer — one backbone tensor is enough."""
    path = _write_checkpoint(tmp_path / "trunc", drop="embed_tokens")
    with pytest.raises(RuntimeError, match="embed_tokens"):
        from_pretrained_verified(AutoModelForCausalLM, path)


def test_complete_checkpoint_loads(tmp_path):
    """Anti-over-rejection: a complete checkpoint is untouched by the gate."""
    path = _write_checkpoint(tmp_path / "full")
    model = from_pretrained_verified(AutoModelForCausalLM, path)
    assert isinstance(model, nn.Module)


@pytest.mark.parametrize("tie_word_embeddings", [False, True])
def test_sequence_classification_head_is_a_legitimate_absence(tmp_path, tie_word_embeddings):
    """A ``score`` head has no tensor on a causal-LM base checkpoint — the reward / classification
    entry points depend on that load succeeding."""
    path = _write_checkpoint(tmp_path / f"base{int(tie_word_embeddings)}", tie_word_embeddings=tie_word_embeddings)
    model = from_pretrained_verified(AutoModelForSequenceClassification, path, num_labels=2)
    assert model.score.weight.shape[0] == 2


def test_head_consuming_caller_refuses_a_base_checkpoint(tmp_path):
    """``excuse_task_head=False`` withdraws that excuse for a caller that only scores with the head.

    Reward-model inference (``scripts/inference/reward_model/_common.py``) would otherwise score every
    row through a randomly initialized ``score`` and report plausible numbers.
    """
    path = _write_checkpoint(tmp_path / "base")
    with pytest.raises(RuntimeError, match="score.weight"):
        from_pretrained_verified(AutoModelForSequenceClassification, path, excuse_task_head=False, num_labels=1)


def test_head_consuming_caller_still_loads_a_real_head(tmp_path):
    """Anti-over-rejection: a checkpoint that does carry the head loads under the same strict flag."""
    base = _write_checkpoint(tmp_path / "base")
    trained = str(tmp_path / "rm")
    AutoModelForSequenceClassification.from_pretrained(base, num_labels=1).save_pretrained(trained)

    model = from_pretrained_verified(AutoModelForSequenceClassification, trained, excuse_task_head=False)
    assert model.score.weight.shape[0] == 1


def test_tied_lm_head_is_a_legitimate_absence(tmp_path):
    """Under ``tie_word_embeddings`` the checkpoint carries no ``lm_head.weight``; the tie supplies it."""
    path = _write_checkpoint(tmp_path / "tied", tie_word_embeddings=True)
    assert "lm_head.weight" not in load_file(os.path.join(path, "model.safetensors"))
    model = from_pretrained_verified(AutoModelForCausalLM, path)
    assert model.lm_head.weight.data_ptr() == model.model.embed_tokens.weight.data_ptr()


def test_an_untied_lm_head_is_excused_like_any_other_head():
    """Documented limit of the outside-the-backbone rule: an untied ``lm_head`` is excused too.

    Randomly initialized output embeddings are self-announcing (loss pins at ``ln(vocab)`` from step
    one), and ``init_checkpoint_absent_modules`` deliberately draws this key for the lazy loaders —
    gating it here would contradict that. Change this only together with that decision."""
    model = AutoModelForCausalLM.from_config(_tiny_config(tie_word_embeddings=False))
    assert unexpected_missing_keys(model, ["lm_head.weight"]) == []
    # The backbone next to it is NOT excused, so the rule is doing real work.
    assert unexpected_missing_keys(model, ["model.norm.weight"]) == ["model.norm.weight"]


def test_class_declared_ignores_are_honoured():
    """The escape hatch is the class's own declaration, not a list maintained here."""

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
    """``AutoModel`` (the embedding path) IS the backbone, so it has no "outside" to excuse."""

    class _Bare(nn.Module):
        base_model_prefix = "model"  # declared, but there is no ``self.model``
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
    """The report must be in the exception text: the transformers logger that would otherwise carry
    it is silenced to ``error`` on every non-logging rank."""
    path = _write_checkpoint(tmp_path / "trunc", drop="layers.1.mlp")
    with pytest.raises(RuntimeError) as excinfo:
        from_pretrained_verified(AutoModelForCausalLM, path)
    message = str(excinfo.value)
    assert "model.layers.1.mlp" in message, message
    assert ALLOW_MISSING_CHECKPOINT_KEYS_ENV in message, message


def test_gate_runs_on_the_reference_and_teacher_model_path(tmp_path):
    """``auto_load_model`` is the bypass used for DPO/KTO reference and distillation teacher models —
    a silently random reference shifts every logratio."""
    from src.models.loading.model_preparation import auto_load_model

    path = _write_checkpoint(tmp_path / "trunc", drop="layers.0.self_attn")
    with pytest.raises(RuntimeError, match="absent from the checkpoint"):
        auto_load_model(path, dtype=torch.float32)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
