#!/usr/bin/env python
"""``padding_free`` on an M-RoPE decoder is refused at collator selection, whatever backend the model resolved.

The flattened row of a decoder that rotates by M-RoPE (every Qwen3.5/3.6 decoder, text towers included) faults
the flash varlen kernels, and a dense backend takes no ``cu_seqlens`` at all, so the only correct answer is a
refusal before the first step. A family without M-RoPE keeps its padding-free collator, and ``packing`` on the
same decoders stays allowed.

Run: python tests/cpu/data/test_mrope_padding_free_refusal.py  (or pytest)
"""

from unittest import mock

import pytest
from transformers.models.auto.configuration_auto import CONFIG_MAPPING

from halo.data.collators import collator_rules
from halo.data.collators.factory import select_data_collator
from halo.models.attention.selection import VARLEN_ATTN_IMPLEMENTATIONS, effective_attn_implementation

# Decoders rotating by M-RoPE: the Qwen3.5/3.6 spellings (wrappers and text towers) and the Qwen-VL text models.
_MROPE_MODEL_TYPES = ("qwen3_5", "qwen3_5_text", "qwen3_5_moe", "qwen3_5_moe_text", "qwen3_vl", "qwen2_5_vl")
# Rotary families that keep padding_free, Qwen3-Next among them (the GatedDeltaNet family without M-RoPE).
_PLAIN_ROPE_MODEL_TYPES = ("qwen3_next", "qwen3", "llama", "glm4_moe_lite")
_DENSE_ATTN_IMPLEMENTATIONS = ("sdpa", "eager")


def _config(model_type: str, attn_impl: str):
    config = CONFIG_MAPPING[model_type]()
    config._attn_implementation = attn_impl
    assert effective_attn_implementation(config) == attn_impl
    return config


def _tokenizer():
    tok = mock.MagicMock()
    tok.pad_token_id, tok.eos_token_id, tok.padding_side = 0, 2, "right"
    return tok


def _select(config, **flags):
    # The GatedDeltaNet fast-path predicates need a CUDA torch; this file is about the M-RoPE rule alone.
    with mock.patch.object(collator_rules, "require_segment_aware_kernels"):
        return select_data_collator(
            tokenizer=_tokenizer(), train_on_completions_only=False, model_config=config, **flags
        )


@pytest.mark.parametrize("attn_impl", VARLEN_ATTN_IMPLEMENTATIONS)
@pytest.mark.parametrize("model_type", _MROPE_MODEL_TYPES)
def test_padding_free_on_an_mrope_decoder_is_refused_on_every_flash_backend(model_type, attn_impl):
    with pytest.raises(ValueError, match="M-RoPE"):
        _select(_config(model_type, attn_impl), padding_free=True)


@pytest.mark.parametrize("attn_impl", _DENSE_ATTN_IMPLEMENTATIONS)
@pytest.mark.parametrize("model_type", _MROPE_MODEL_TYPES)
def test_padding_free_on_an_mrope_decoder_stays_refused_on_a_dense_backend(model_type, attn_impl):
    """The refusal names the M-RoPE rotary ahead of the dense backend's own, whose remedy, a flash backend, would
    fault on the same row."""
    with pytest.raises(ValueError, match="rotates by M-RoPE"):
        _select(_config(model_type, attn_impl), padding_free=True)


@pytest.mark.parametrize("attn_impl", VARLEN_ATTN_IMPLEMENTATIONS)
@pytest.mark.parametrize("model_type", _PLAIN_ROPE_MODEL_TYPES)
def test_a_decoder_without_mrope_keeps_its_padding_free_collator(model_type, attn_impl):
    collator = _select(_config(model_type, attn_impl), padding_free=True)
    assert type(collator).__name__ == "DataCollatorWithFlattening"


@pytest.mark.parametrize("attn_impl", VARLEN_ATTN_IMPLEMENTATIONS)
@pytest.mark.parametrize("model_type", _MROPE_MODEL_TYPES)
def test_packing_on_an_mrope_decoder_is_not_refused(model_type, attn_impl):
    collator = _select(_config(model_type, attn_impl), packing=True)
    assert type(collator).__name__ == "DataCollatorWithPacking"


def test_the_rule_reads_only_padding_free_and_a_config():
    """No config (a script that passes none) and ``padding_free`` off both leave the rule silent."""
    collator_rules.reject_mrope_padding_free(None, padding_free=True)
    collator_rules.reject_mrope_padding_free(_config("qwen3_5_text", "flash_attention_4"), padding_free=False)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
