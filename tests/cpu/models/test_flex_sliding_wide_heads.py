"""The flex-sliding backend reads head widths per layer, including on configs whose ``head_dim`` varies by
layer: the resolver's wide-head check, and the sliding-layer calls its kernel warm-up compiles for.

Gemma 4's config is heterogeneous: its sliding layers are 256 wide and its global layers 512, so reading
``config.head_dim`` raises ``AmbiguousGlobalPerLayerAttributeError`` (a ``RuntimeError``, which
``getattr(..., None)`` does not catch); the per-layer view carries each layer's width.
"""

import pytest
from transformers import Gemma4ForCausalLM, Gemma4TextConfig, MistralConfig
from transformers.models.gemma4.modeling_gemma4 import Gemma4TextAttention

from src.models.patches.attention import FLASH_MAX_HEAD_DIM, head_dim_exceeds_flash
from src.models.patches.flex_sliding_attention import sliding_attention_calls
from tests.common.models import TINY_GEMMA4_WIDE_HEAD_CONFIG


def _gemma4(head_dim: int, global_head_dim: int) -> Gemma4TextConfig:
    return Gemma4TextConfig(
        **{
            **TINY_GEMMA4_WIDE_HEAD_CONFIG,
            "num_hidden_layers": 4,
            "head_dim": head_dim,
            "global_head_dim": global_head_dim,
            "layer_types": ["sliding_attention", "sliding_attention", "full_attention", "full_attention"],
        }
    )


def test_a_per_layer_head_dim_is_read_per_layer():
    config = _gemma4(head_dim=256, global_head_dim=2 * FLASH_MAX_HEAD_DIM)
    with pytest.raises(RuntimeError):  # premise: the global attribute is ambiguous on this config
        _ = config.head_dim
    assert head_dim_exceeds_flash(config) is True


def test_narrow_per_layer_heads_are_not_wide():
    config = _gemma4(head_dim=64, global_head_dim=FLASH_MAX_HEAD_DIM)
    assert head_dim_exceeds_flash(config) is False


def test_a_uniform_config_reads_its_single_head_dim():
    assert head_dim_exceeds_flash(MistralConfig(head_dim=FLASH_MAX_HEAD_DIM)) is False
    assert head_dim_exceeds_flash(MistralConfig(head_dim=2 * FLASH_MAX_HEAD_DIM)) is True


def test_the_warmup_compiles_what_the_sliding_layers_call_with():
    """The warm-up must compile each sliding layer's own call: its head width, not the widest (global) one,
    and its ``scaling``, which Dynamo guards on (Gemma 4 attends at 1.0, not ``head_dim ** -0.5``). A graph
    for any other value leaves the real one to compile mid-run."""
    model = Gemma4ForCausalLM(_gemma4(head_dim=256, global_head_dim=2 * FLASH_MAX_HEAD_DIM))
    sliding = [m for m in model.modules() if isinstance(m, Gemma4TextAttention) and m.sliding_window]
    assert {m.scaling for m in sliding} == {1.0}  # premise: not the default head_dim ** -0.5
    assert sliding_attention_calls(model) == {(4, 2, 256, 32, 1.0)}


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
