""":class:`halo.kernels.lowp.quantization.WeightVersionCache`: entries evict on garbage collection and distinct
tensors never alias, which an ``id(weight)`` dict violates."""

import gc

import pytest
import torch

from halo.kernels.lowp import quantization
from halo.kernels.lowp.quantization import WeightVersionCache


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
    assert len(cache) == 0  # The weakref finalizer dropped the entry and its derived buffer.


def test_distinct_tensors_never_alias():
    cache = WeightVersionCache()
    a = torch.nn.Parameter(torch.randn(4, 4))
    b = torch.nn.Parameter(torch.randn(4, 4))
    cache.put(a, (0, "mxfp8"), "a-value")
    cache.put(b, (0, "mxfp8"), "b-value")
    assert cache.get(a, (0, "mxfp8")) == "a-value"
    assert cache.get(b, (0, "mxfp8")) == "b-value"
    c = torch.nn.Parameter(torch.randn(4, 4))
    assert cache.get(c, (0, "mxfp8")) is None


def test_cached_fake_quant_cache_evicts_end_to_end():
    baseline = len(quantization._weight_dequant_cache)
    w = torch.nn.Parameter(torch.randn(8, 64) * 0.3)
    quantization.cached_fake_quant(w, "mxfp8", axis=-1)
    assert len(quantization._weight_dequant_cache) == baseline + 1
    del w
    gc.collect()
    assert len(quantization._weight_dequant_cache) == baseline


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
