#!/usr/bin/env python
"""Tests for ParameterStatsCallback. Run: python tests/cpu/callbacks/test_parameter_stats.py"""

import logging
import sys

import pytest
import torch

import src.callbacks.parameter_stats as mod
from src.callbacks.parameter_stats import (
    ParameterStatsCallback,
    _dtype_stats,
    _format_memory_size,
    _module_trainable_stats,
    _normalize_module_name,
    count_model_parameters,
)

# Mock helpers


class MockParam:
    """Lightweight mock parameter."""

    def __init__(self, shape, dtype=torch.bfloat16, requires_grad=True):
        self._tensor = torch.randn(*shape, dtype=dtype)
        self.requires_grad = requires_grad
        self.dtype = dtype

    def numel(self):
        return self._tensor.numel()

    def element_size(self):
        return self._tensor.element_size()


class MockModel:
    """Mock model with named_parameters and parameters."""

    def __init__(self, named_params=None):
        self._named_params = named_params or []

    def parameters(self):
        return iter([p for _, p in self._named_params])

    def named_parameters(self):
        return iter(self._named_params)


class MockPartialState:
    """Mock for accelerate.PartialState so we don't need accelerate init."""

    def __init__(self, is_main=True):
        self._is_main = is_main

    @property
    def is_main_process(self):
        return self._is_main


# Tests


def test_count_model_parameters():
    p_trainable = MockParam((100, 100), requires_grad=True)
    p_frozen = MockParam((50, 50), requires_grad=False)

    model = MockModel(
        named_params=[
            ("layer1.weight", p_trainable),
            ("layer2.weight", p_frozen),
        ]
    )
    total, trainable = count_model_parameters(model)
    assert total == 100 * 100 + 50 * 50, f"Expected total={100 * 100 + 50 * 50}, got {total}"
    assert trainable == 100 * 100, f"Expected trainable={100 * 100}, got {trainable}"


def test_normalize_module_name():
    result = _normalize_module_name("model.layers.12.mlp")
    assert result == "model.layers.X.mlp", f"Expected 'model.layers.X.mlp', got {result!r}"

    result = _normalize_module_name("model.layers.0.self_attn.q_proj")
    assert result == "model.layers.X.self_attn.q_proj", f"Got {result!r}"

    result = _normalize_module_name("")
    assert result == "<root>", f"Expected '<root>', got {result!r}"

    # Every numeric index collapses (nested experts: layer + expert index).
    result = _normalize_module_name("model.layers.3.mlp.experts.7.w1")
    assert result == "model.layers.X.mlp.experts.X.w1", f"Got {result!r}"

    # A name with no digits is unchanged.
    assert _normalize_module_name("model.embed_tokens.weight") == "model.embed_tokens.weight"


def test_compute_module_trainable_stats():
    p1 = MockParam((10, 10), requires_grad=True)
    p2 = MockParam((20, 20), requires_grad=False)
    p3 = MockParam((10, 10), requires_grad=True)

    model = MockModel(
        named_params=[
            ("model.layers.0.mlp.weight", p1),
            ("model.layers.0.mlp.bias", p2),
            ("model.layers.1.mlp.weight", p3),
        ]
    )
    stats = _module_trainable_stats(model)

    # "model.layers.0.mlp.weight" and "model.layers.1.mlp.weight" both normalize to
    # "model.layers.X.mlp.weight"
    key_weight = "model.layers.X.mlp.weight"
    key_bias = "model.layers.X.mlp.bias"

    assert key_weight in stats, f"Expected key {key_weight!r} in stats"
    assert key_bias in stats, f"Expected key {key_bias!r} in stats"

    total_w, trainable_w = stats[key_weight]
    assert total_w == 200, f"Expected total=200 for weight group, got {total_w}"
    assert trainable_w == 200, f"Expected trainable=200 for weight group, got {trainable_w}"

    total_b, trainable_b = stats[key_bias]
    assert total_b == 400, f"Expected total=400 for bias group, got {total_b}"
    assert trainable_b == 0, f"Expected trainable=0 for frozen bias, got {trainable_b}"


def test_compute_dtype_stats():
    p_bf16 = MockParam((100,), dtype=torch.bfloat16, requires_grad=True)
    p_fp32 = MockParam((50,), dtype=torch.float32, requires_grad=False)

    model = MockModel(
        named_params=[
            ("layer.bf16_weight", p_bf16),
            ("layer.fp32_weight", p_fp32),
        ]
    )
    stats = _dtype_stats(model)

    assert torch.bfloat16 in stats, "Expected bfloat16 in dtype stats"
    assert torch.float32 in stats, "Expected float32 in dtype stats"

    bf16_total, bf16_trainable, bf16_mem = stats[torch.bfloat16]
    assert bf16_total == 100, f"Expected bf16 total=100, got {bf16_total}"
    assert bf16_trainable == 100, f"Expected bf16 trainable=100, got {bf16_trainable}"
    # bf16 is 2 bytes per element
    assert bf16_mem == 100 * 2, f"Expected bf16 memory=200, got {bf16_mem}"

    fp32_total, fp32_trainable, fp32_mem = stats[torch.float32]
    assert fp32_total == 50, f"Expected fp32 total=50, got {fp32_total}"
    assert fp32_trainable == 0, f"Expected fp32 trainable=0, got {fp32_trainable}"
    # fp32 is 4 bytes per element
    assert fp32_mem == 50 * 4, f"Expected fp32 memory=200, got {fp32_mem}"


def test_format_memory_size():
    assert _format_memory_size(1024) == "1.00 KB", f"Got {_format_memory_size(1024)!r}"
    assert _format_memory_size(1073741824) == "1.00 GB", f"Got {_format_memory_size(1073741824)!r}"
    assert _format_memory_size(512) == "512.00 B", f"Got {_format_memory_size(512)!r}"
    assert _format_memory_size(1048576) == "1.00 MB", f"Got {_format_memory_size(1048576)!r}"
    # Boundary: 0 bytes stays in B.
    assert _format_memory_size(0) == "0.00 B", f"Got {_format_memory_size(0)!r}"
    # TB and the PB fall-through (loop exhausts the unit list at >= 1024 TB).
    assert _format_memory_size(1024**4) == "1.00 TB", f"Got {_format_memory_size(1024**4)!r}"
    assert _format_memory_size(1024**5) == "1.00 PB", f"Got {_format_memory_size(1024**5)!r}"


def test_callback_main_process_runs(monkeypatch, caplog):
    """On the main process, on_train_begin must EMIT the three tables — totals, deduplicated module
    groups, dtype breakdown — with this rank's real counts. The report is the callback's only
    product, so a body that walks the model and logs nothing is the failure to catch."""
    monkeypatch.setattr(mod, "PartialState", lambda: MockPartialState(is_main=True))

    model = MockModel(
        named_params=[
            ("model.layers.0.mlp.weight", MockParam((10, 10), requires_grad=True)),
            ("model.layers.1.mlp.weight", MockParam((10, 10), requires_grad=True)),
            ("model.embed_tokens.weight", MockParam((5, 5), dtype=torch.float32, requires_grad=False)),
        ]
    )

    with caplog.at_level(logging.INFO, logger=mod.__name__):
        ParameterStatsCallback().on_train_begin(object(), object(), object(), model=model)

    text = caplog.text
    assert "Total number of parameters (this rank): 225" in text, text
    assert "Number of trainable parameters (this rank): 200" in text, text
    assert "Percentage of trainable parameters: 88.89%" in text, text
    # The two layers share a normalized name, so they collapse into one module group row.
    assert "model.layers.X.mlp.weight" in text and "trainable: 200 / 200 (100.00%)" in text, text
    assert "torch.float32" in text and "trainable: 0 (0.00%)" in text, text
    assert "Total parameter memory: 500.00 B" in text, text


def test_callback_non_main_process_skips(monkeypatch):
    """Non-main ranks must return BEFORE touching the model (no logging duplication).

    We pass a model whose parameter walk would raise; reaching it means the
    early ``is_main_process`` guard failed.
    """
    monkeypatch.setattr(mod, "PartialState", lambda: MockPartialState(is_main=False))

    class ExplodingModel:
        def parameters(self):
            raise AssertionError("non-main rank must not inspect parameters")

        def named_parameters(self):
            raise AssertionError("non-main rank must not inspect parameters")

    cb = ParameterStatsCallback()
    # Should return cleanly without ever calling parameters().
    cb.on_train_begin(object(), object(), object(), model=ExplodingModel())


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
