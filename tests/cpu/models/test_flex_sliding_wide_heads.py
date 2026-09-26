"""Wide-head detection reads each layer's head width, including on configs whose ``head_dim`` varies by layer.

Gemma 4's config is heterogeneous: its sliding layers are 256 wide and its global layers 512, so reading
``config.head_dim`` raises ``AmbiguousGlobalPerLayerAttributeError`` (a ``RuntimeError``, which
``getattr(..., None)`` does not catch). The hub 26B-A4B checkpoint only reached the flex-sliding path
because the sliding-layer check ran first; the wide-head check has to stand on its own.
"""

import pytest
from transformers import Gemma4TextConfig, MistralConfig

from src.models.patches.flex_sliding_attention import FLASH_SDPA_MAX_HEAD_DIM, model_has_wide_heads


def _gemma4(head_dim: int, global_head_dim: int) -> Gemma4TextConfig:
    return Gemma4TextConfig(
        vocab_size=256,
        hidden_size=128,
        intermediate_size=128,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=head_dim,
        global_head_dim=global_head_dim,
        num_global_key_value_heads=1,
        sliding_window=32,
        layer_types=["sliding_attention", "sliding_attention", "full_attention", "full_attention"],
        enable_moe_block=False,
        hidden_size_per_layer_input=0,
        num_kv_shared_layers=0,
    )


def test_a_per_layer_head_dim_is_read_per_layer():
    config = _gemma4(head_dim=256, global_head_dim=2 * FLASH_SDPA_MAX_HEAD_DIM)
    with pytest.raises(RuntimeError):  # premise: the global attribute is ambiguous on this config
        _ = config.head_dim
    assert model_has_wide_heads(config) is True


def test_narrow_per_layer_heads_are_not_wide():
    config = _gemma4(head_dim=64, global_head_dim=FLASH_SDPA_MAX_HEAD_DIM)
    assert model_has_wide_heads(config) is False


def test_a_uniform_config_reads_its_single_head_dim():
    assert model_has_wide_heads(MistralConfig(head_dim=FLASH_SDPA_MAX_HEAD_DIM)) is False
    assert model_has_wide_heads(MistralConfig(head_dim=2 * FLASH_SDPA_MAX_HEAD_DIM)) is True


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
