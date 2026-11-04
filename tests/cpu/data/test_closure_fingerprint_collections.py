#!/usr/bin/env python
"""``_closure_fingerprint`` covers scalar collections (sets order-independently), leaves dicts out (``none_example``
is cache-irrelevant), and warns once on any cell it cannot cover."""

import logging

import pytest

from halo.data.pipeline.map_cache_key import _closure_fingerprint, _unfingerprintable_warned


def _capturing(value):
    def fn(row):
        return {"out": value}

    return fn


def test_captured_list_changes_fingerprint():
    """Hashed to one key, the second run silently reuses the first's cache."""
    fp_a = _closure_fingerprint(_capturing([1, 2, 3]))
    fp_b = _closure_fingerprint(_capturing([1, 2, 4]))
    assert fp_a and fp_b, "list cells must contribute to the fingerprint, not be skipped"
    assert fp_a != fp_b, "captured-list change did not change the closure fingerprint (stale-cache reuse)"


def test_captured_tuple_and_set_change_fingerprint():
    assert _closure_fingerprint(_capturing((1, "a"))) != _closure_fingerprint(_capturing((1, "b")))
    assert _closure_fingerprint(_capturing({1, 2, 3})) != _closure_fingerprint(_capturing({1, 2, 9}))
    assert _closure_fingerprint(_capturing(frozenset({5, 7}))) != _closure_fingerprint(_capturing(frozenset({5, 8})))


def test_set_fingerprint_is_order_independent():
    fp_a = _closure_fingerprint(_capturing({3, 1, 2}))
    fp_b = _closure_fingerprint(_capturing({2, 3, 1}))
    assert fp_a == fp_b


def test_identical_collections_same_fingerprint():
    assert _closure_fingerprint(_capturing([1, 2, 3])) == _closure_fingerprint(_capturing([1, 2, 3]))


def test_non_scalar_collection_not_fingerprinted():
    """A list of non-scalars falls through to the skip-with-warning path and produces no address-dependent repr."""
    fp_a = _closure_fingerprint(_capturing([object()]))
    fp_b = _closure_fingerprint(_capturing([object()]))
    assert fp_a == fp_b, "an object address leaked into the fingerprint, fragmenting the cache per run"
    assert fp_a == "", f"a non-scalar cell must be skipped outright, not hashed: {fp_a!r}"
    # A fingerprint empty for everything would satisfy the equality above while disabling cache invalidation.
    assert _closure_fingerprint(_capturing([1, 2, 3])) != ""


def test_unfingerprintable_cell_warns_once(caplog):
    class _Opaque:
        pass

    _unfingerprintable_warned.clear()
    with caplog.at_level(logging.WARNING, logger="halo.data.pipeline.map_cache_key"):
        _closure_fingerprint(_capturing(_Opaque()))
    warnings = [r for r in caplog.records if "skips a value of type" in r.getMessage()]
    assert len(warnings) == 1
    assert "_Opaque" in warnings[0].getMessage()
    assert "_capturing" in warnings[0].getMessage()

    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="halo.data.pipeline.map_cache_key"):
        _closure_fingerprint(_capturing(_Opaque()))
    assert not [r for r in caplog.records if "skips a value of type" in r.getMessage()], (
        "the skip warning must be one-time per (function, cell type)"
    )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
