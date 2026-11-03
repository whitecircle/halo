#!/usr/bin/env python
"""``finalize_loaded_model`` buffer recompute (halo/model_loading/buffer_repair.py). transformers 5 moves
``rope_theta`` into ``rope_parameters``, so a ``getattr`` default rebuilds RoPE wrong."""

import logging

import pytest
import torch
import torch.nn as nn
from transformers import Qwen3Config, Qwen3ForCausalLM
from transformers.models.gemma3.modeling_gemma3 import Gemma3TextScaledWordEmbedding
from transformers.models.gemma4 import Gemma4TextConfig, Gemma4VisionConfig
from transformers.models.gemma4.modeling_gemma4 import (
    Gemma4TextModel,
    Gemma4TextScaledWordEmbedding,
    Gemma4VisionRotaryEmbedding,
)
from transformers.models.qwen3.modeling_qwen3 import Qwen3RotaryEmbedding

from halo.model_loading import buffer_repair
from halo.model_loading.finalization import finalize_loaded_model


def _model_with_rotary(theta: float) -> tuple[nn.Module, Qwen3RotaryEmbedding, torch.Tensor]:
    cfg = Qwen3Config(rope_theta=theta, hidden_size=64, num_attention_heads=4, head_dim=16)
    rotary = Qwen3RotaryEmbedding(cfg)
    reference = rotary.inv_freq.clone()
    model = nn.Module()
    model.rotary_emb = rotary
    return model, rotary, reference


def test_recompute_preserves_nondefault_theta():
    model, rotary, reference = _model_with_rotary(theta=1_000_000.0)
    rotary.inv_freq.zero_()  # Simulates the bf16/meta corruption the fixer repairs.
    finalize_loaded_model(model)
    assert torch.allclose(rotary.inv_freq, reference), (
        f"inv_freq rebuilt with wrong rope base: got {rotary.inv_freq[1].item():.6f}, "
        f"expected {reference[1].item():.6f} (theta=1e6)"
    )
    assert rotary.inv_freq.dtype == torch.float32


def test_recompute_default_theta():
    model, rotary, reference = _model_with_rotary(theta=10_000.0)
    rotary.inv_freq.zero_()
    finalize_loaded_model(model)
    assert torch.allclose(rotary.inv_freq, reference)


def test_per_layer_type_recompute_matches_model_init():
    """Gemma4 registers ``head_dim`` per layer, so a raw-config read raises
    ``AmbiguousGlobalPerLayerAttributeError`` and a single global field sizes a wrong table."""
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
    # The two layer types size different tables (head_dim 512 vs 256).
    assert reference["full_attention_inv_freq"].shape != reference["sliding_attention_inv_freq"].shape
    for name in reference:
        getattr(rotary, name).zero_()

    finalize_loaded_model(model)

    rebuilt = dict(rotary.named_buffers())
    for name, expected in reference.items():
        assert rebuilt[name].shape == expected.shape, name
        assert torch.equal(rebuilt[name].float(), expected.float()), name


def test_vision_rotary_recompute_matches_model_init():
    """Sized off ``head_dim // 2``, reached through the module's own ``compute_default_rope_parameters``."""
    rotary = Gemma4VisionRotaryEmbedding(Gemma4VisionConfig())
    reference = rotary.inv_freq.clone()
    model = nn.Module()
    model.rotary_emb = rotary
    rotary.inv_freq.zero_()
    rotary.original_inv_freq.zero_()

    finalize_loaded_model(model)

    assert torch.equal(rotary.inv_freq, reference)
    assert torch.equal(rotary.original_inv_freq, reference)


@pytest.mark.parametrize("embedding_cls", [Gemma4TextScaledWordEmbedding, Gemma3TextScaledWordEmbedding])
def test_scaled_embedding_scale_is_rebuilt_from_its_scalar(embedding_cls):
    """A zero ``embed_scale`` multiplies every input embedding by zero."""
    embedding = embedding_cls(16, 8, padding_idx=0, embed_scale=8**0.5)
    embedding.embed_scale.zero_()
    model = nn.Module()
    model.embed_tokens = embedding

    finalize_loaded_model(model)

    assert embedding.embed_scale.item() == pytest.approx(8**0.5)


def test_recompute_preserves_declared_persistence():
    """Forcing ``persistent=False`` silently drops a remote-code family's key from every save."""
    model, rotary, reference = _model_with_rotary(theta=10_000.0)
    rotary.register_buffer("inv_freq", rotary.inv_freq.clone(), persistent=True)
    rotary.inv_freq.zero_()
    finalize_loaded_model(model)
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
    """An unconditional ``tie_weights()`` would silently overwrite a checkpoint's distinct ``lm_head.weight``."""
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
    """The case the finalize re-tie exists for."""
    model = _tiny_tied_lm()
    head = model.get_output_embeddings()
    head.weight = nn.Parameter(torch.empty_like(head.weight, device="meta"))
    assert model.get_output_embeddings().weight.is_meta

    finalize_loaded_model(model)

    assert model.get_output_embeddings().weight is model.get_input_embeddings().weight, (
        "meta shadow head was not re-tied to the embedding"
    )


class _UnknownRotary(nn.Module):
    """No ``inv_freq``, no per-layer-type mapping."""


def _model_with_unknown_rotaries(count: int) -> nn.Module:
    model = nn.Module()
    for i in range(count):
        model.add_module(f"layer{i}", _UnknownRotary())
    return model


def test_per_module_repair_is_not_logged_at_info(caplog):
    """A 92-layer model at 512 ranks buries the real diagnostics, and each message forces a device sync on the load
    path."""
    caplog.set_level(logging.INFO, logger=buffer_repair.__name__)
    model, rotary, _reference = _model_with_rotary(theta=1_000_000.0)
    rotary.inv_freq.zero_()

    finalize_loaded_model(model)

    messages = [r.message for r in caplog.records]
    assert not [m for m in messages if "rotary_emb" in m], f"per-module line still at INFO: {messages}"
    assert [m for m in messages if "buffer(s) to float32" in m], "the per-model summary must remain"


def test_unrecognized_rotary_warns_once_per_class(caplog, monkeypatch):
    """A property of the family, and this logger reaches every rank."""
    caplog.set_level(logging.WARNING, logger=buffer_repair.__name__)
    # Clearing the module's own set would leave a later test silently un-warned.
    monkeypatch.setattr(buffer_repair, "_warned_unfixed", set())

    finalize_loaded_model(_model_with_unknown_rotaries(4))

    seen = [r for r in caplog.records if "_UnknownRotary" in r.message]
    assert len(seen) == 1, f"expected one line per rotary class, got {len(seen)}"

    monkeypatch.setattr(buffer_repair, "_warned_unfixed", set())
    caplog.clear()
    finalize_loaded_model(_model_with_unknown_rotaries(1))
    assert [r for r in caplog.records if "_UnknownRotary" in r.message], "anti-vacuity: it must still warn once"


class _ConfiglessSlopeAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.num_heads = 4
        self.layer_idx = 1
        self.register_buffer("slope", torch.empty(4), persistent=False)


class _ConfiglessLayerTypeRotary(nn.Module):
    rope_init_fns = {"full_attention": None}


class _ConfiglessSingleRotary(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("inv_freq", torch.empty(4), persistent=False)


@pytest.mark.parametrize(
    "module",
    [
        _ConfiglessSlopeAttention(),
        _ConfiglessLayerTypeRotary(),
        _ConfiglessSingleRotary(),
    ],
    ids=["alibi_slope", "per_layer_type_rope", "single_inv_freq_rope"],
)
def test_a_claimed_but_unfixable_buffer_warns(module, caplog, monkeypatch):
    """The one line naming the class is the only trace of the uninitialized values."""
    caplog.set_level(logging.WARNING, logger=buffer_repair.__name__)
    monkeypatch.setattr(buffer_repair, "_warned_unfixed", set())
    model = nn.Module()
    model.add_module("layer0", module)

    finalize_loaded_model(model)

    assert [r for r in caplog.records if type(module).__name__ in r.message and "Cannot recompute" in r.message]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
