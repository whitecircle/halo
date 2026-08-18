#!/usr/bin/env python
"""``fix_rotary_inv_freq`` tests (src/models/patches/buffer_fixes.py) — CPU-only.

transformers 5 pops ``rope_theta``/``partial_rotary_factor`` off the config top level into
``config.rope_parameters``; a recompute that reads ``getattr(config, "rope_theta", 10000.0)``
silently rebuilds RoPE with the wrong base for every theta != 10000 model (Qwen3 1e6, LFM2 1e6,
GLM-4.7). This pins the default-rope recompute to the module's own rope_parameters-aware math.

Run: python tests/cpu/models/test_buffer_fixes.py
"""

import logging

import pytest
import torch
import torch.nn as nn
from transformers import Qwen3Config, Qwen3ForCausalLM
from transformers.models.qwen3.modeling_qwen3 import Qwen3RotaryEmbedding

from src.models.patches import buffer_fixes
from src.models.patches.buffer_fixes import finalize_loaded_model, fix_rotary_inv_freq


def _model_with_rotary(theta: float) -> tuple[nn.Module, Qwen3RotaryEmbedding, torch.Tensor]:
    cfg = Qwen3Config(rope_theta=theta, hidden_size=64, num_attention_heads=4, head_dim=16)
    rotary = Qwen3RotaryEmbedding(cfg)
    reference = rotary.inv_freq.clone()
    model = nn.Module()
    model.rotary_emb = rotary
    return model, rotary, reference


def test_recompute_preserves_nondefault_theta():
    model, rotary, reference = _model_with_rotary(theta=1_000_000.0)
    rotary.inv_freq.zero_()  # simulate the bf16/meta corruption the fixer repairs
    fix_rotary_inv_freq(model)
    assert torch.allclose(rotary.inv_freq, reference), (
        f"inv_freq rebuilt with wrong rope base: got {rotary.inv_freq[1].item():.6f}, "
        f"expected {reference[1].item():.6f} (theta=1e6)"
    )
    assert rotary.inv_freq.dtype == torch.float32


def test_recompute_default_theta():
    model, rotary, reference = _model_with_rotary(theta=10_000.0)
    rotary.inv_freq.zero_()
    fix_rotary_inv_freq(model)
    assert torch.allclose(rotary.inv_freq, reference)


def test_per_layer_type_recompute_matches_model_init():
    """Gemma4 registers ``head_dim`` per-layer (256 sliding / 512 global, transformers 5.16
    heterogeneity): the recompute must size each layer type's table off the layer-type-resolved
    config view, exactly as the module's own ``__init__`` does. A raw-config read raises
    ``AmbiguousGlobalPerLayerAttributeError`` on the sliding leg, and the proportional (global) leg
    sized off any single global field rebuilds a wrong-length table."""
    from transformers.models.gemma4 import Gemma4TextConfig
    from transformers.models.gemma4.modeling_gemma4 import Gemma4TextModel

    cfg = Gemma4TextConfig(
        num_hidden_layers=6,
        hidden_size=64,
        intermediate_size=64,
        moe_intermediate_size=32,
        num_attention_heads=4,
        num_key_value_heads=2,
        vocab_size=64,
    )
    model = Gemma4TextModel(cfg)
    rotary = model.rotary_emb
    reference = {name: buf.clone() for name, buf in rotary.named_buffers() if "inv_freq" in name}
    # Heterogeneity is live: the two layer types size different tables (head_dim 512 vs 256).
    assert reference["full_attention_inv_freq"].shape != reference["sliding_attention_inv_freq"].shape
    for name in reference:
        getattr(rotary, name).zero_()  # simulate the bf16/meta corruption the fixer repairs

    fix_rotary_inv_freq(model)

    rebuilt = dict(rotary.named_buffers())
    for name, expected in reference.items():
        assert rebuilt[name].shape == expected.shape, name
        assert torch.equal(rebuilt[name].float(), expected.float()), name


def test_recompute_preserves_declared_persistence():
    """A remote-code family may declare ``inv_freq`` PERSISTENT (shipped in its checkpoints);
    forcing ``persistent=False`` on the recompute silently drops the key from every save."""
    model, rotary, reference = _model_with_rotary(theta=10_000.0)
    rotary.register_buffer("inv_freq", rotary.inv_freq.clone(), persistent=True)
    rotary.inv_freq.zero_()
    fix_rotary_inv_freq(model)
    assert torch.allclose(rotary.inv_freq, reference)
    assert "rotary_emb.inv_freq" in model.state_dict(), "recompute silently dropped a persistent buffer from saves"


def _tiny_tied_lm() -> nn.Module:
    cfg = Qwen3Config(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=24,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        tie_word_embeddings=True,
    )
    torch.manual_seed(0)
    return Qwen3ForCausalLM(cfg)


def test_finalize_keeps_a_loaded_distinct_head():
    """transformers 5 honours a checkpoint that ships a DISTINCT ``lm_head.weight`` even under
    ``tie_word_embeddings: true``; an unconditional ``tie_weights()`` in the finalize seam silently
    overwrote it with the embedding on every non-lazy load path."""
    model = _tiny_tied_lm()
    distinct = torch.randn_like(model.get_input_embeddings().weight)
    model.set_output_embeddings(nn.Linear(16, 32, bias=False))
    with torch.no_grad():
        model.get_output_embeddings().weight.copy_(distinct)

    finalize_loaded_model(model)

    assert torch.equal(model.get_output_embeddings().weight, distinct), (
        "finalize re-tied over a materialized distinct head, discarding the loaded weights"
    )


def test_finalize_still_ties_a_meta_shadow_head():
    """The lazy loaders leave the tied shadow (``lm_head.weight``) on meta until tying — the case
    the finalize re-tie exists for. Skipping it here would strand the head on meta."""
    model = _tiny_tied_lm()
    head = model.get_output_embeddings()
    head.weight = nn.Parameter(torch.empty_like(head.weight, device="meta"))
    assert model.get_output_embeddings().weight.is_meta

    finalize_loaded_model(model)

    assert model.get_output_embeddings().weight is model.get_input_embeddings().weight, (
        "meta shadow head was not re-tied to the embedding"
    )


class _UnknownRotary(nn.Module):
    """A rotary layout the fixer does not recognize — no ``inv_freq``, no per-layer-type mapping."""


def _model_with_unknown_rotaries(count: int) -> nn.Module:
    model = nn.Module()
    for i in range(count):
        model.add_module(f"layer{i}", _UnknownRotary())
    return model


def test_per_module_repair_is_not_logged_at_info(caplog):
    """One INFO per rotary module per RANK, carrying ``inv_freq.min()/.max()``, costs twice: a
    92-layer model at 512 ranks buries the run's real diagnostics, and every message forces a device
    sync on the load path. The per-model summary is what survives."""
    caplog.set_level(logging.INFO, logger=buffer_fixes.__name__)
    model, rotary, _reference = _model_with_rotary(theta=1_000_000.0)
    rotary.inv_freq.zero_()

    fix_rotary_inv_freq(model)

    messages = [r.message for r in caplog.records]
    assert not [m for m in messages if "rotary_emb" in m], f"per-module line still at INFO: {messages}"
    assert [m for m in messages if "buffer(s) to float32" in m], "the per-model summary must remain"


def test_unrecognized_rotary_warns_once_per_class(caplog, monkeypatch):
    """The unhandled layout is a property of the FAMILY: one line says everything one-per-layer
    would, and this logger reaches every rank."""
    caplog.set_level(logging.WARNING, logger=buffer_fixes.__name__)
    # A fresh set per arm, restored afterwards: clearing the module's own would leave a later test
    # in this worker silently un-warned.
    monkeypatch.setattr(buffer_fixes, "_WARNED_ROTARY", set())

    fix_rotary_inv_freq(_model_with_unknown_rotaries(4))

    seen = [r for r in caplog.records if "_UnknownRotary" in r.message]
    assert len(seen) == 1, f"expected one line per rotary class, got {len(seen)}"

    monkeypatch.setattr(buffer_fixes, "_WARNED_ROTARY", set())
    caplog.clear()
    fix_rotary_inv_freq(_model_with_unknown_rotaries(1))
    assert [r for r in caplog.records if "_UnknownRotary" in r.message], "anti-vacuity: it must still warn once"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
