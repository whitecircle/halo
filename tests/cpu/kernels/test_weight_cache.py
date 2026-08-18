"""CPU tests for :class:`src.kernels.lowp.quantization.WeightVersionCache` — the shared auto-evicting
weight cache behind the low-precision dequant cache (``quantization.cached_fake_quant``) and the
DeepGEMM quantized-operand cache.

Pins the contracts an ``id(weight)``-keyed dict violates: entries must evict when the weight is
garbage-collected (no unbounded growth / stale-id hits in multi-model processes), and distinct
tensors must never alias.

    python tests/cpu/kernels/test_weight_cache.py
"""

import gc
import sys

import pytest
import torch

from src.kernels.lowp import quantization
from src.kernels.lowp.quantization import WeightVersionCache


def test_hit_and_fingerprint_invalidation():
    cache = WeightVersionCache()
    w = torch.nn.Parameter(torch.randn(4, 4))
    cache.put(w, (w._version, "mxfp8"), "value")
    assert cache.get(w, (w._version, "mxfp8")) == "value"
    # Version bump or format change → miss.
    assert cache.get(w, (w._version + 1, "mxfp8")) is None
    assert cache.get(w, (w._version, "nvfp4")) is None


def test_entry_evicts_when_weight_is_collected():
    cache = WeightVersionCache()
    w = torch.nn.Parameter(torch.randn(4, 4))
    cache.put(w, (0, "mxfp8"), torch.zeros(4, 4))
    assert len(cache) == 1
    del w
    gc.collect()
    assert len(cache) == 0  # weakref finalizer dropped the entry (and its derived buffer)


def test_distinct_tensors_never_alias():
    cache = WeightVersionCache()
    a = torch.nn.Parameter(torch.randn(4, 4))
    b = torch.nn.Parameter(torch.randn(4, 4))
    cache.put(a, (0, "mxfp8"), "a-value")
    cache.put(b, (0, "mxfp8"), "b-value")
    assert cache.get(a, (0, "mxfp8")) == "a-value"
    assert cache.get(b, (0, "mxfp8")) == "b-value"
    # A different tensor object never reads another's entry, fingerprint match or not.
    c = torch.nn.Parameter(torch.randn(4, 4))
    assert cache.get(c, (0, "mxfp8")) is None


def test_cached_fake_quant_cache_evicts_end_to_end():
    baseline = len(quantization._WEIGHT_DEQUANT_CACHE)
    w = torch.nn.Parameter(torch.randn(8, 64) * 0.3)
    quantization.cached_fake_quant(w, "mxfp8", axis=-1)
    assert len(quantization._WEIGHT_DEQUANT_CACHE) == baseline + 1
    del w
    gc.collect()
    assert len(quantization._WEIGHT_DEQUANT_CACHE) == baseline


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
