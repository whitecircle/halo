#!/usr/bin/env python
"""``DequantRules`` is the whole per-release surface of the fp8 → bf16 converter, its field list written once."""

import dataclasses
import inspect

import pytest

from halo.checkpoint.tools import fp8_dequant

_RULE_FIELDS = {field.name for field in dataclasses.fields(fp8_dequant.DequantRules)}


def _module_functions():
    return [
        fn for _, fn in inspect.getmembers(fp8_dequant, inspect.isfunction) if fn.__module__ == fp8_dequant.__name__
    ]


def test_the_rule_field_list_is_spelled_once():
    """They travel on the object."""
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
    """Each internal assumes the stage before it ran, so an outside caller skips the preflight or the in-place
    refusal."""
    public = sorted(
        name
        for name, obj in vars(fp8_dequant).items()
        if not name.startswith("_") and getattr(obj, "__module__", None) == fp8_dequant.__name__
    )
    assert public == [
        "DequantRules",
        "block_fp8_rules",
        "dequantize_block_fp8",
        "run_dequant_conversion",
        "static_fp8_rules",
    ], public


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
