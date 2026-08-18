#!/usr/bin/env python
"""Module contract of the shared fp8 → bf16 streaming converter.

``DequantRules`` is the whole per-release surface: a converter declares its scale spelling, layout
check, dequantization and passthrough sizing there and the shared driver applies them. That only
holds while the field list is written ONCE — a helper that takes the fields as loose keyword
arguments re-spells it, so every new rule has to be threaded through each signature and each call in
between, and a helper reached from outside can be handed a half-set of them.

Run: python tests/cpu/checkpoint/test_fp8_dequant_contract.py  (or pytest)
"""

import dataclasses
import inspect

import pytest

from src.checkpoint import fp8_dequant

_RULE_FIELDS = {field.name for field in dataclasses.fields(fp8_dequant.DequantRules)}


def _module_functions():
    return [
        fn for _, fn in inspect.getmembers(fp8_dequant, inspect.isfunction) if fn.__module__ == fp8_dequant.__name__
    ]


def test_the_rule_field_list_is_spelled_once():
    """No helper re-declares ``DequantRules``' fields as parameters — they travel on the object."""
    offenders = {
        fn.__qualname__: sorted(_RULE_FIELDS & set(inspect.signature(fn).parameters))
        for fn in _module_functions()
        if _RULE_FIELDS & set(inspect.signature(fn).parameters)
    }
    assert not offenders, (
        f"DequantRules fields {sorted(_RULE_FIELDS)} are re-spelled as parameters by {offenders}. "
        f"Pass the rules object instead, so a new rule is declared in one place."
    )


def test_only_the_release_rules_and_the_entry_point_are_public():
    """The staged internals are private: each takes a `DequantRules` mid-conversion and assumes the
    stage before it ran (a validated source, a not-yet-existing output dir), so an outside caller
    reaching one directly skips the preflight or the in-place refusal."""
    public = sorted(
        name
        for name, obj in vars(fp8_dequant).items()
        if not name.startswith("_") and getattr(obj, "__module__", None) == fp8_dequant.__name__
    )
    assert public == ["DequantRules", "run_dequant_conversion"], public


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
