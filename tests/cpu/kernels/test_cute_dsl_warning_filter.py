#!/usr/bin/env python
"""The CuTe DSL deprecation filter: ``nvidia_cutlass_dsl`` warns per pointer construction, thousands of times per rank,
past Python's dedup. Scoped on both axes, or our own deprecations or real ``cutlass`` warnings go with it."""

from __future__ import annotations

import warnings

import pytest

from halo.kernels.jit_cache import silence_cute_dsl_deprecations


def _warner(module_name: str, category: type[Warning]):
    """``warnings`` matches the ``__name__`` in its caller's globals, so faking that one global reproduces
    ``cutlass.cute.core`` without a GPU."""
    namespace = {"__name__": module_name, "warnings": warnings, "category": category}
    exec("def emit():\n    warnings.warn('pointer API is deprecated', category)", namespace)
    return namespace["emit"]


def _caught(module_name: str, category: type[Warning]) -> list:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        silence_cute_dsl_deprecations()
        _warner(module_name, category)()
    return caught


def test_the_cute_dsl_deprecation_is_silenced():
    assert not _caught("cutlass.cute.core", DeprecationWarning), (
        "the CuTe DSL deprecation still surfaces — every FA4 compile floods the run log with it"
    )


def test_other_modules_keep_their_deprecations():
    assert _caught("halo.trainers.distributed_trainer", DeprecationWarning), (
        "the filter swallowed a deprecation from our own code; its module scope is too wide"
    )
    assert _caught("cutlass_helper.utils", DeprecationWarning), (
        "the filter matched a module merely prefixed 'cutlass'; it must match the package, not a prefix"
    )


def test_other_cutlass_warnings_still_surface():
    assert _caught("cutlass.cute.core", UserWarning), (
        "the filter muted a non-deprecation cutlass warning — an unsupported dtype or a silent kernel "
        "fallback would now be invisible"
    )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
