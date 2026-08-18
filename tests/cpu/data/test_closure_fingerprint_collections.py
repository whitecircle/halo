#!/usr/bin/env python
"""Closure-fingerprint coverage for scalar collections + visibility of skipped cells.

``_get_closure_fingerprint`` keys the coordinated-map cache on the output-affecting values a map fn
captures. Skipping container-typed cells silently makes a map fn capturing e.g. a list of response
token ids reuse a stale cache when only that list changed. The contract:

* lists/tuples/sets of scalars are fingerprinted (sets order-independently),
* dicts stay deliberately un-fingerprinted (``none_example`` is pinned cache-irrelevant by
  ``test_cache_key_knobs``), and
* any cell the fingerprint cannot cover logs a ONE-TIME warning naming the function and cell type,
  so the omission is visible instead of silent.

Run: pytest tests/cpu/data/test_closure_fingerprint_collections.py
"""

import logging

import pytest

from src.data.pipeline.processing import (
    _UNFINGERPRINTABLE_WARNED,
    _get_closure_fingerprint,
)


def _capturing(value):
    """A closure over ``value`` with identical code for every call site."""

    def fn(row):
        return {"out": value}

    return fn


def test_captured_list_changes_fingerprint():
    """Two closures identical except for a captured list (e.g. response token ids) must fingerprint
    differently: hashed to the same key, the second run silently reuses the first's cache."""
    fp_a = _get_closure_fingerprint(_capturing([1, 2, 3]))
    fp_b = _get_closure_fingerprint(_capturing([1, 2, 4]))
    assert fp_a and fp_b, "list cells must contribute to the fingerprint, not be skipped"
    assert fp_a != fp_b, "captured-list change did not change the closure fingerprint (stale-cache reuse)"


def test_captured_tuple_and_set_change_fingerprint():
    assert _get_closure_fingerprint(_capturing((1, "a"))) != _get_closure_fingerprint(_capturing((1, "b")))
    assert _get_closure_fingerprint(_capturing({1, 2, 3})) != _get_closure_fingerprint(_capturing({1, 2, 9}))
    assert _get_closure_fingerprint(_capturing(frozenset({5, 7}))) != _get_closure_fingerprint(
        _capturing(frozenset({5, 8}))
    )


def test_set_fingerprint_is_order_independent():
    """Set iteration order must not fragment the cache: same elements → same fingerprint."""
    fp_a = _get_closure_fingerprint(_capturing({3, 1, 2}))
    fp_b = _get_closure_fingerprint(_capturing({2, 3, 1}))
    assert fp_a == fp_b


def test_identical_collections_same_fingerprint():
    assert _get_closure_fingerprint(_capturing([1, 2, 3])) == _get_closure_fingerprint(_capturing([1, 2, 3]))


def test_non_scalar_collection_not_fingerprinted():
    """A list of non-scalars (e.g. tensors/objects) stays outside the fingerprint — it falls
    through to the skip-with-warning path rather than producing an address-dependent repr."""
    fp_a = _get_closure_fingerprint(_capturing([object()]))
    fp_b = _get_closure_fingerprint(_capturing([object()]))
    assert fp_a == fp_b, "an object address leaked into the fingerprint, fragmenting the cache per run"
    assert fp_a == "", f"a non-scalar cell must be skipped outright, not hashed: {fp_a!r}"
    # …and skipping must stay distinguishable from covering: a fingerprint that went empty for
    # everything would satisfy the equality above while silently disabling cache invalidation.
    assert _get_closure_fingerprint(_capturing([1, 2, 3])) != ""


def test_unfingerprintable_cell_warns_once(caplog):
    """A cell type the fingerprint cannot cover must emit a one-time warning naming the function
    and the type — silence hides the gap; unbounded repetition would be noise."""

    class _Opaque:
        pass

    _UNFINGERPRINTABLE_WARNED.clear()
    with caplog.at_level(logging.WARNING, logger="src.data.pipeline.processing"):
        _get_closure_fingerprint(_capturing(_Opaque()))
    warnings = [r for r in caplog.records if "skips a value of type" in r.getMessage()]
    assert len(warnings) == 1
    assert "_Opaque" in warnings[0].getMessage()
    assert "_capturing" in warnings[0].getMessage()

    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="src.data.pipeline.processing"):
        _get_closure_fingerprint(_capturing(_Opaque()))
    assert not [r for r in caplog.records if "skips a value of type" in r.getMessage()], (
        "the skip warning must be one-time per (function, cell type)"
    )


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
