#!/usr/bin/env python
"""``padding_free`` on an M-RoPE decoder selects the flattening collator on every flash backend, as on a plain decoder,
and is refused on a dense backend, which takes no ``cu_seqlens``; ``packing`` stays allowed."""

from unittest import mock

import pytest
from transformers.models.auto.configuration_auto import CONFIG_MAPPING

from halo.data.collators import collator_rules
from halo.data.collators.factory import select_data_collator
from halo.models.attention.selection import VARLEN_ATTN_IMPLEMENTATIONS, effective_attn_implementation

# Decoders rotating by M-RoPE: the Qwen3.5/3.6 spellings (wrappers and text towers) and the Qwen-VL text models.
_MROPE_MODEL_TYPES = ("qwen3_5", "qwen3_5_text", "qwen3_5_moe", "qwen3_5_moe_text", "qwen3_vl", "qwen2_5_vl")
# Rotary families without M-RoPE, Qwen3-Next among them (the GatedDeltaNet family without it).
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
    # The GatedDeltaNet fast-path predicates need a CUDA torch; this file is about M-RoPE rows alone.
    with mock.patch.object(collator_rules, "require_segment_aware_kernels"):
        return select_data_collator(
            tokenizer=_tokenizer(), train_on_completions_only=False, model_config=config, **flags
        )


@pytest.mark.parametrize("attn_impl", VARLEN_ATTN_IMPLEMENTATIONS)
@pytest.mark.parametrize("model_type", _MROPE_MODEL_TYPES + _PLAIN_ROPE_MODEL_TYPES)
def test_padding_free_keeps_its_collator_on_every_flash_backend(model_type, attn_impl):
    collator = _select(_config(model_type, attn_impl), padding_free=True)
    assert type(collator).__name__ == "DataCollatorWithFlattening"


@pytest.mark.parametrize("attn_impl", _DENSE_ATTN_IMPLEMENTATIONS)
@pytest.mark.parametrize("model_type", _MROPE_MODEL_TYPES)
def test_padding_free_on_a_dense_backend_is_refused_for_the_missing_cu_seqlens(model_type, attn_impl):
    """Negative control: the flattened row's boundaries reach only a varlen kernel, M-RoPE or not."""
    with pytest.raises(ValueError, match="padding_free requires a varlen"):
        _select(_config(model_type, attn_impl), padding_free=True)


@pytest.mark.parametrize("attn_impl", VARLEN_ATTN_IMPLEMENTATIONS)
@pytest.mark.parametrize("model_type", _MROPE_MODEL_TYPES)
def test_packing_on_an_mrope_decoder_is_not_refused(model_type, attn_impl):
    collator = _select(_config(model_type, attn_impl), packing=True)
    assert type(collator).__name__ == "DataCollatorWithPacking"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
