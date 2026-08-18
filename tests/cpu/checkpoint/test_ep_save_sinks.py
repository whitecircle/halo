#!/usr/bin/env python
"""CPU tests for GptOss attention-sink reconstruction in the EP save path.

flash_attention_2 fine-tuning sets ``layer.self_attn.sinks = None``
(Blackwell FA2 rejects a non-None s_aux). A ``None`` parameter drops out of
``named_parameters()``, so an EP save reading that alone omits all 36 ``self_attn.sinks``
tensors — producing a checkpoint that ``GptOssForCausalLM`` reports as missing
weights (and that auto-inits to non-disabled values on load). ``neutralized_gpt_oss_sinks``
reconstructs the disabled (dtype-min) sinks so the saved checkpoint stays complete.
"""

import pytest
import torch
import torch.nn as nn

from src.models.patches.gpt_oss_sinks import neutralized_gpt_oss_sinks


class _Cfg:
    def __init__(self, model_type="gpt_oss", num_attention_heads=4):
        self.model_type = model_type
        self.num_attention_heads = num_attention_heads


class _Attn(nn.Module):
    def __init__(self, sinks_none=True):
        super().__init__()
        self.q_proj = nn.Linear(8, 8, bias=False).to(torch.bfloat16)
        self.register_parameter("sinks", nn.Parameter(torch.zeros(4, dtype=torch.bfloat16)))
        if sinks_none:
            self.sinks = None  # mimic _reset_gpt_oss_sinks FA2 path


class _Layer(nn.Module):
    def __init__(self, sinks_none=True):
        super().__init__()
        self.self_attn = _Attn(sinks_none=sinks_none)


class _Inner(nn.Module):
    def __init__(self, n_layers=2, sinks_none=True):
        super().__init__()
        self.layers = nn.ModuleList([_Layer(sinks_none) for _ in range(n_layers)])


class _Model(nn.Module):
    def __init__(self, model_type="gpt_oss", n_layers=2, heads=4, sinks_none=True):
        super().__init__()
        self.model = _Inner(n_layers, sinks_none)
        self.config = _Cfg(model_type, heads)


def test_reconstructs_sinks_when_none():
    """gpt_oss layers with sinks=None get one dtype-min sink tensor each."""
    m = _Model("gpt_oss", n_layers=3, heads=4, sinks_none=True)
    sinks = neutralized_gpt_oss_sinks(m)
    assert set(sinks) == {f"model.layers.{i}.self_attn.sinks" for i in range(3)}, sinks.keys()
    for t in sinks.values():
        assert tuple(t.shape) == (4,)
        assert t.dtype == torch.bfloat16
        assert bool((t == torch.finfo(torch.bfloat16).min).all())


def test_none_sinks_drop_from_named_parameters():
    """Confirms the root cause: a None param is absent from named_parameters()."""
    m = _Model("gpt_oss", n_layers=2, sinks_none=True)
    names = dict(m.named_parameters())
    assert "model.layers.0.self_attn.sinks" not in names


def test_skips_when_sinks_present():
    """When sinks are real tensors (eager/flex fill path), the normal param loop
    saves them, so the helper returns nothing."""
    m = _Model("gpt_oss", n_layers=2, sinks_none=False)
    assert neutralized_gpt_oss_sinks(m) == {}


def test_empty_for_non_gpt_oss():
    """Non-GptOss models never get synthetic sinks."""
    m = _Model("qwen3_moe", n_layers=2, sinks_none=True)
    assert neutralized_gpt_oss_sinks(m) == {}


def test_no_config_is_safe():
    """A model without a config returns empty, not an error."""
    m = nn.Linear(2, 2)
    assert neutralized_gpt_oss_sinks(m) == {}


def test_only_none_layers_reconstructed():
    """A mix of None and present sinks: only the dropped (None) layers are rebuilt;
    the present ones are left to the normal param loop (no duplicate save)."""
    m = _Model("gpt_oss", n_layers=3, heads=4, sinks_none=True)
    # Give layer 1 a real (present) sink — it must NOT appear in the result.
    m.model.layers[1].self_attn.sinks = nn.Parameter(torch.zeros(4, dtype=torch.bfloat16))
    sinks = neutralized_gpt_oss_sinks(m)
    assert set(sinks) == {
        "model.layers.0.self_attn.sinks",
        "model.layers.2.self_attn.sinks",
    }


def test_sink_width_follows_num_attention_heads():
    """The reconstructed sink length is num_attention_heads, not a fixed 4."""
    m = _Model("gpt_oss", n_layers=1, heads=7, sinks_none=True)
    (sink,) = neutralized_gpt_oss_sinks(m).values()
    assert tuple(sink.shape) == (7,)


def test_zero_heads_returns_empty():
    """A degenerate config with num_attention_heads <= 0 yields no sinks."""
    m = _Model("gpt_oss", n_layers=2, heads=0, sinks_none=True)
    assert neutralized_gpt_oss_sinks(m) == {}


def test_dtype_inferred_from_q_proj():
    """The neutralized sink takes its dtype from the layer's q_proj weight."""

    class _F32Attn(nn.Module):
        def __init__(self):
            super().__init__()
            self.q_proj = nn.Linear(8, 8, bias=False).to(torch.float32)
            self.sinks = None

    m = _Model("gpt_oss", n_layers=1, heads=4, sinks_none=True)
    m.model.layers[0].self_attn = _F32Attn()
    (sink,) = neutralized_gpt_oss_sinks(m).values()
    assert sink.dtype == torch.float32
    assert bool((sink == torch.finfo(torch.float32).min).all())


def test_nested_model_wrapper_unwrapped():
    """A wrapper that nests ``.model.model.layers`` (e.g. a CP/EP wrapper) is
    unwrapped to the decoder that owns ``.layers``."""

    class _Wrapper(nn.Module):
        def __init__(self, inner):
            super().__init__()
            self.model = inner
            self.config = _Cfg("gpt_oss", 4)

    base = _Model("gpt_oss", n_layers=2, heads=4, sinks_none=True)
    wrapped = _Wrapper(base)
    sinks = neutralized_gpt_oss_sinks(wrapped)
    assert set(sinks) == {f"model.layers.{i}.self_attn.sinks" for i in range(2)}


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
