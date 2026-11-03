"""The ``'default'`` RoPE init entry the compat shims restore. transformers 5 dropped it and moved ``rope_theta`` into
``rope_parameters``, while remote-code models written against v4 still look the entry up by name."""

import types

import pytest
import torch
from transformers import CONFIG_MAPPING
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS

from halo.model_loading.buffer_repair import default_rope_parameters
from halo.model_loading.remote_code import apply_remote_code_compat_shims

apply_remote_code_compat_shims()

V5_MODEL_TYPES = ("qwen3_moe", "gpt_oss", "llama", "qwen3")


def _closed_form(base: float, dim: int) -> torch.Tensor:
    return 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.int64).float() / dim))


def test_shim_registers_the_missing_entry():
    assert ROPE_INIT_FUNCTIONS.get("default") is default_rope_parameters


@pytest.mark.parametrize("model_type", V5_MODEL_TYPES)
def test_default_entry_runs_on_a_transformers_5_config(model_type):
    """transformers 5 configs define no flat ``config.rope_theta``."""
    config = CONFIG_MAPPING[model_type]()
    inv_freq, attention_scaling = ROPE_INIT_FUNCTIONS["default"](config, None)

    head_dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
    assert inv_freq.numel() == head_dim // 2
    assert attention_scaling == 1.0
    assert torch.isfinite(inv_freq).all()


@pytest.mark.parametrize("model_type", V5_MODEL_TYPES)
def test_base_comes_from_rope_parameters_not_a_hardcoded_default(model_type):
    """10000.0 for a theta-150000 model is silent drift."""
    config = CONFIG_MAPPING[model_type]()
    theta = config.rope_parameters["rope_theta"]
    head_dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads

    inv_freq, _ = default_rope_parameters(config)

    assert torch.allclose(inv_freq, _closed_form(theta, head_dim), rtol=0, atol=1e-12)
    if theta != 10000.0:
        assert not torch.allclose(inv_freq, _closed_form(10000.0, head_dim))


def test_flat_attribute_config_still_works():
    """The v4-style remote-code config the shim exists for."""
    config = types.SimpleNamespace(rope_theta=500000.0, hidden_size=64, num_attention_heads=4, head_dim=16)
    inv_freq, _ = default_rope_parameters(config)
    assert torch.allclose(inv_freq, _closed_form(500000.0, 16), rtol=0, atol=1e-12)


def test_partial_rotary_factor_shrinks_the_rotary_dim():
    config = types.SimpleNamespace(
        rope_parameters={"rope_theta": 10000.0, "partial_rotary_factor": 0.5},
        hidden_size=64,
        num_attention_heads=4,
        head_dim=16,
    )
    inv_freq, _ = default_rope_parameters(config)
    assert inv_freq.numel() == 8 // 2  # 16 * 0.5 = 8 rotary dims


def test_head_dim_derived_when_absent():
    config = types.SimpleNamespace(rope_parameters={"rope_theta": 10000.0}, hidden_size=64, num_attention_heads=4)
    inv_freq, _ = default_rope_parameters(config)
    assert inv_freq.numel() == (64 // 4) // 2


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
