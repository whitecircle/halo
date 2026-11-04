#!/usr/bin/env python
"""``register_custom_optimizers`` creates genuine ``OptimizerNames`` members. A planted string breaks
``OptimizerNames.MUON``, CLI ``--optim muon``, and value-keyed lookups, since a member's hash follows its name."""

import tempfile

import pytest
from transformers import HfArgumentParser, TrainingArguments
from transformers.training_args import OptimizerNames

from halo.optimizers.registry import NAMED_OPTIMIZER_BUILDERS, optim_name, register_custom_optimizers

register_custom_optimizers()

CUSTOM_NAMES = tuple(NAMED_OPTIMIZER_BUILDERS)


def test_the_registry_holds_the_optimizers_the_configs_name():
    """A registry entry lost from ``NAMED_OPTIMIZER_BUILDERS`` would make the loops below pass vacuously."""
    assert set(CUSTOM_NAMES) == {"muon", "flash_adamw"}, f"custom-optimizer registry changed: {CUSTOM_NAMES}"


def test_call_returns_genuine_member():
    for name in CUSTOM_NAMES:
        member = OptimizerNames(name)
        assert isinstance(member, OptimizerNames), f"OptimizerNames({name!r}) returned {type(member).__name__}"
        assert member.value == name
        assert member.name == name.upper()


def test_attribute_access_and_iteration():
    for name in CUSTOM_NAMES:
        member = getattr(OptimizerNames, name.upper())
        assert member is OptimizerNames(name), f"attribute and call must resolve to the same {name} member"
        assert member in list(OptimizerNames), f"{name} member missing from enum iteration"
    member_names = [m.name for m in OptimizerNames]
    for name in CUSTOM_NAMES:
        assert member_names.count(name.upper()) == 1


def test_optim_name_routes_member_into_builders():
    for name in CUSTOM_NAMES:
        member = OptimizerNames(name)
        key = optim_name(member)
        assert key == name and type(key) is str
        assert key in NAMED_OPTIMIZER_BUILDERS
        builder, description = NAMED_OPTIMIZER_BUILDERS[key]
        assert callable(builder) and description
    assert optim_name(None) == ""
    assert optim_name("adamw_torch") == "adamw_torch"


def test_double_registration_is_noop():
    before_members = list(OptimizerNames)
    before_ids = {name: id(OptimizerNames(name)) for name in CUSTOM_NAMES}
    register_custom_optimizers()
    assert list(OptimizerNames) == before_members
    for name in CUSTOM_NAMES:
        assert id(OptimizerNames(name)) == before_ids[name], f"{name} member identity changed on re-registration"
        assert OptimizerNames._member_names_.count(name.upper()) == 1


def test_stock_members_unaffected():
    stock = OptimizerNames("adamw_torch")
    assert stock is OptimizerNames.ADAMW_TORCH
    assert isinstance(stock, OptimizerNames) and stock.value == "adamw_torch"
    assert [m.name for m in OptimizerNames].count("ADAMW_TORCH") == 1


def test_pure_cli_optim_parses():
    """Choices are built from member iteration."""
    for name in CUSTOM_NAMES:
        with tempfile.TemporaryDirectory() as tmp:
            parser = HfArgumentParser(TrainingArguments)
            (args,) = parser.parse_args_into_dataclasses(["--output_dir", tmp, "--optim", name])
            assert optim_name(args.optim) == name
            # An `or args.optim == name` fallback would accept the planted string.
            assert isinstance(args.optim, OptimizerNames)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
