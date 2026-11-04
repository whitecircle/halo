#!/usr/bin/env python
"""GptOss sink reconstruction in the EP save: FA2 sets ``sinks = None``, dropping them from ``named_parameters()``,
so the checkpoint would miss all 36, which auto-init to non-disabled values."""

import pytest
import torch
import torch.nn as nn

from halo.models.attention.gpt_oss_sinks import neutralized_gpt_oss_sinks


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
            self.sinks = None  # Mimics the _reset_gpt_oss_sinks FA2 path.


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
    m = _Model("gpt_oss", n_layers=3, heads=4, sinks_none=True)
    sinks = neutralized_gpt_oss_sinks(m)
    assert set(sinks) == {f"model.layers.{i}.self_attn.sinks" for i in range(3)}, sinks.keys()
    for t in sinks.values():
        assert tuple(t.shape) == (4,)
        assert t.dtype == torch.bfloat16
        assert bool((t == torch.finfo(torch.bfloat16).min).all())


def test_none_sinks_drop_from_named_parameters():
    m = _Model("gpt_oss", n_layers=2, sinks_none=True)
    names = dict(m.named_parameters())
    assert "model.layers.0.self_attn.sinks" not in names


def test_skips_when_sinks_present():
    """The normal param loop saves real sinks."""
    m = _Model("gpt_oss", n_layers=2, sinks_none=False)
    assert neutralized_gpt_oss_sinks(m) == {}


def test_empty_for_non_gpt_oss():
    m = _Model("qwen3_moe", n_layers=2, sinks_none=True)
    assert neutralized_gpt_oss_sinks(m) == {}


def test_no_config_is_safe():
    m = nn.Linear(2, 2)
    assert neutralized_gpt_oss_sinks(m) == {}


def test_only_none_layers_reconstructed():
    """Otherwise a present sink is saved twice."""
    m = _Model("gpt_oss", n_layers=3, heads=4, sinks_none=True)
    m.model.layers[1].self_attn.sinks = nn.Parameter(torch.zeros(4, dtype=torch.bfloat16))
    sinks = neutralized_gpt_oss_sinks(m)
    assert set(sinks) == {
        "model.layers.0.self_attn.sinks",
        "model.layers.2.self_attn.sinks",
    }


def test_sink_width_follows_num_attention_heads():
    """Not a fixed 4."""
    m = _Model("gpt_oss", n_layers=1, heads=7, sinks_none=True)
    (sink,) = neutralized_gpt_oss_sinks(m).values()
    assert tuple(sink.shape) == (7,)


def test_zero_heads_returns_empty():
    m = _Model("gpt_oss", n_layers=2, heads=0, sinks_none=True)
    assert neutralized_gpt_oss_sinks(m) == {}


def test_dtype_inferred_from_q_proj():
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
    """A CP/EP wrapper nesting ``.model.model.layers``."""

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
