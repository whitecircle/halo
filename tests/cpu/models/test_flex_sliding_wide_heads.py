"""The flex-sliding backend reads head widths per layer, in the resolver's wide-head check and the warm-up's calls.
A heterogeneous ``config.head_dim`` raises a ``RuntimeError``, which ``getattr(..., None)`` does not catch."""

import pytest
from transformers import Gemma4ForCausalLM, Gemma4TextConfig, MistralConfig
from transformers.models.gemma4.modeling_gemma4 import Gemma4TextAttention

from halo.models.attention.layout import AttentionCall
from halo.models.attention.sdpa_flex_sliding import flex_attention_calls
from halo.models.attention.selection import FLASH_MAX_HEAD_DIM, has_head_dim_beyond_flash
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
    with pytest.raises(RuntimeError):  # the global attribute is ambiguous on this config
        _ = config.head_dim
    assert has_head_dim_beyond_flash(config) is True


def test_narrow_per_layer_heads_are_not_wide():
    config = _gemma4(head_dim=64, global_head_dim=FLASH_MAX_HEAD_DIM)
    assert has_head_dim_beyond_flash(config) is False


def test_a_uniform_config_reads_its_single_head_dim():
    assert has_head_dim_beyond_flash(MistralConfig(head_dim=FLASH_MAX_HEAD_DIM)) is False
    assert has_head_dim_beyond_flash(MistralConfig(head_dim=2 * FLASH_MAX_HEAD_DIM)) is True


def test_the_warmup_compiles_what_the_sliding_layers_call_with():
    """Dynamo guards on ``scaling`` (Gemma 4 attends at 1.0), so a graph for any other value compiles mid-run.
    The 512-wide global layers take no FlexAttention call, so they add none."""
    model = Gemma4ForCausalLM(_gemma4(head_dim=256, global_head_dim=2 * FLASH_MAX_HEAD_DIM))
    sliding = [m for m in model.modules() if isinstance(m, Gemma4TextAttention) and m.sliding_window]
    assert {m.scaling for m in sliding} == {1.0}  # not the default head_dim ** -0.5
    assert flex_attention_calls(model) == {AttentionCall(4, 2, 256, 256, 32, 1.0, None)}


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
