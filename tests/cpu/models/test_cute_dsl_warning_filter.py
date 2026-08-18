#!/usr/bin/env python
"""CPU tests for the CuTe DSL deprecation filter installed at training setup.

``nvidia_cutlass_dsl`` emits a ``DeprecationWarning`` per pointer construction while JIT-compiling the
FA4 kernels, from one line it reaches thousands of times per rank — Python's once-per-location dedup
never engages because each compile carries a fresh registry. The filter that mutes it is scoped two
ways, and both matter: widen the module scope and a real deprecation from our own code disappears;
widen the category and genuine ``cutlass`` warnings (an unsupported dtype, a fallback kernel) go with
it. These tests pin the scope on both axes.

Run: ``pytest -m cpu tests/cpu/models/test_cute_dsl_warning_filter.py``
"""

from __future__ import annotations

import warnings

import pytest

from src.models.patches.attention import silence_cute_dsl_deprecations


def _warner(module_name: str, category: type[Warning]):
    """A callable that warns as if it lived in ``module_name``.

    ``warnings`` attributes a warning to the ``__name__`` in its caller's globals, which is what the
    filter's ``module`` regex is matched against — so faking that one global reproduces exactly what
    the real ``cutlass.cute.core`` does, without a GPU or a kernel compile.
    """
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
    """The flood itself: a DeprecationWarning from the real module path must not reach the log."""
    assert not _caught("cutlass.cute.core", DeprecationWarning), (
        "the CuTe DSL deprecation still surfaces — every FA4 compile floods the run log with it"
    )


def test_other_modules_keep_their_deprecations():
    """Anti-over-reach on the module axis: only ``cutlass`` is muted, not every library."""
    assert _caught("src.trainers.mixins.base", DeprecationWarning), (
        "the filter swallowed a deprecation from our own code; its module scope is too wide"
    )
    assert _caught("cutlass_helper.utils", DeprecationWarning), (
        "the filter matched a module merely prefixed 'cutlass'; it must match the package, not a prefix"
    )


def test_other_cutlass_warnings_still_surface():
    """Anti-over-reach on the category axis: a real cutlass problem must still be visible."""
    assert _caught("cutlass.cute.core", UserWarning), (
        "the filter muted a non-deprecation cutlass warning — an unsupported dtype or a silent kernel "
        "fallback would now be invisible"
    )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
