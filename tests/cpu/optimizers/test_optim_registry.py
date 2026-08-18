#!/usr/bin/env python
"""``register_custom_optimizers`` must create GENUINE ``OptimizerNames`` members (CPU).

A planted plain-string attribute (the naive extension) makes ``OptimizerNames("muon")`` return a
``str`` instead of a member, leaves ``OptimizerNames.MUON`` raising, keeps the name out of the
argparse ``choices`` HfArgumentParser builds from member iteration (rejecting a pure-CLI
``--optim muon``), and — because an enum member's hash follows its NAME — silently misses every
value-keyed dict lookup unless ``optim_name`` normalizes first. Each of those seams is pinned here.

Run: pytest tests/cpu/optimizers/test_optim_registry.py  (or python <file>)
"""

import tempfile

import pytest
from transformers import HfArgumentParser, TrainingArguments
from transformers.training_args import OptimizerNames

from src.optimizers.registry import NAMED_OPTIMIZER_BUILDERS, optim_name, register_custom_optimizers

register_custom_optimizers()

CUSTOM_NAMES = tuple(NAMED_OPTIMIZER_BUILDERS)


def test_the_registry_holds_the_optimizers_the_configs_name():
    """Pins the parametrization source. Every test below loops over ``CUSTOM_NAMES`` with its
    assertions inside the loop, so a registry entry lost from ``NAMED_OPTIMIZER_BUILDERS`` would
    make them all pass vacuously while ``--optim muon`` stopped parsing for every Muon YAML."""
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
    """The user-facing seam: ``--optim muon`` on the CLI must pass HfArgumentParser's choices
    (built from member iteration) and TrainingArguments' own validation."""
    for name in CUSTOM_NAMES:
        with tempfile.TemporaryDirectory() as tmp:
            parser = HfArgumentParser(TrainingArguments)
            (args,) = parser.parse_args_into_dataclasses(["--output_dir", tmp, "--optim", name])
            assert optim_name(args.optim) == name
            # No `or args.optim == name` fallback: a planted plain string is the exact failure this
            # module exists to catch, and the disjunct would accept it.
            assert isinstance(args.optim, OptimizerNames)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
