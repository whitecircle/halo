"""CPU tests for ``_hash_value`` (``src/diagnostics/debugging.py``).

The hash feeds ``assert_consistent``'s ``all_gather_object`` flow during TP/EP divergence triage —
exactly where bf16/fp8 tensors show up. A dtype that raises on one rank hangs the peers, so the
hash must be dtype-agnostic (no ``.numpy()`` on the element dtype) and containers must hash
element-wise regardless of the first element's type.

    python tests/cpu/diagnostics/test_dist_debug_hash.py
"""

import sys

import pytest
import torch

from src.diagnostics.debugging import _hash_value


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float8_e4m3fn, torch.float32, torch.int64, torch.bool])
def test_equal_tensors_hash_equal_and_different_differ(dtype):
    base = torch.arange(8, dtype=torch.float32).reshape(2, 4)
    a = base.to(dtype)
    b = base.clone().to(dtype)
    assert _hash_value(a) == _hash_value(b)

    changed = base.clone()
    changed[0, 0] = 5.0
    assert _hash_value(a) != _hash_value(changed.to(dtype))


def test_dtype_and_shape_distinguish():
    z = torch.zeros(4, dtype=torch.bfloat16)
    assert _hash_value(z) != _hash_value(torch.zeros(4, dtype=torch.float16))  # same bytes, other dtype
    assert _hash_value(z) != _hash_value(torch.zeros(2, 2, dtype=torch.bfloat16))  # same bytes, other shape


def test_zero_dim_and_empty_bf16():
    assert _hash_value(torch.tensor(1.5, dtype=torch.bfloat16)) == _hash_value(torch.tensor(1.5, dtype=torch.bfloat16))
    assert _hash_value(torch.empty(0, dtype=torch.bfloat16)) == _hash_value(torch.empty(0, dtype=torch.bfloat16))


def test_noncontiguous_matches_contiguous():
    t = torch.arange(16, dtype=torch.bfloat16).reshape(4, 4)
    assert _hash_value(t.t()) == _hash_value(t.t().contiguous())


def test_mixed_containers_hash_element_wise():
    t = torch.ones(3, dtype=torch.bfloat16)
    # Non-tensor first element must not divert the container into the pickle branch.
    assert _hash_value([5, t]) == _hash_value([5, t.clone()])
    assert _hash_value([5, t]) != _hash_value([5, t * 2])
    assert _hash_value([5, t]) != _hash_value([6, t])
    # Tensor-first mixed list (bf16) must not crash and must track the non-tensor tail.
    assert _hash_value([t, "x"]) == _hash_value([t.clone(), "x"])
    assert _hash_value([t, "x"]) != _hash_value([t, "y"])
    # Nesting is distinguished by the type/length prefix.
    assert _hash_value([t]) != _hash_value([[t]])


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
