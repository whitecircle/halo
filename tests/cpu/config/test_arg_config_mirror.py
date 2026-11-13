#!/usr/bin/env python
"""Parity between ``ParallelismArguments`` (the YAML/CLI surface) and ``ParallelismConfig``: a knob whose two
declarations disagree on the name and has no explicit entry in ``parallelism_config_from_args`` parses and does
nothing, and one the configuration reference never names cannot be set."""

import ast
import inspect
import re
import textwrap
from dataclasses import fields
from typing import get_args, get_type_hints

import pytest

from halo.configs.parallelism_arguments import ParallelismArguments, parallelism_config_from_args
from halo.configs.parser import _literal_choices
from halo.distributed.parallelism_config import ParallelismConfig
from halo.distributed.parallelism_rules import AXES
from halo.kernels.lowp.spec import LowpSpec
from halo.trainers.distributed_trainer import DistributedTrainerMixin
from tests.common.parallelism import make_parallelism_config
from tests.common.repo import REPO_ROOT

# Knobs the two dataclasses spell identically, read off the declarations since the builder is what is being checked.
_SAME_NAME_FIELDS = {f.name for f in fields(ParallelismArguments)} & {
    f.name for f in fields(ParallelismConfig) if f.init
}

_LOWP_ARG_FIELDS = {f"lowp_{f.name}": f.name for f in fields(LowpSpec)}

# Resolved from the live process group in ``__post_init__``, not from a YAML/CLI knob.
_CONFIG_FIELDS_NOT_FROM_ARGS = {"world_size", "gpus_per_node"}


class _LowpTrainer(DistributedTrainerMixin):
    """The mixin's gates with the ``lowp_*`` knobs admitted, as on the SFT path."""

    SUPPORTS_LOWP_PRECISION = True


def _defaults(cls) -> dict:
    return {f.name: f.default for f in fields(cls)}


def _builder_ast() -> ast.Module:
    return ast.parse(textwrap.dedent(inspect.getsource(parallelism_config_from_args)))


def _dist_args_names_read_by_builder() -> set[str]:
    """Every ``dist_args.<name>`` and ``getattr(dist_args, "<name>", ...)`` the builder evaluates."""
    tree = _builder_ast()
    names = {
        node.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id == "dist_args"
    }
    names |= {
        node.args[1].value
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and getattr(node.func, "id", None) == "getattr"
        and len(node.args) >= 2
        and isinstance(node.args[0], ast.Name)
        and node.args[0].id == "dist_args"
        and isinstance(node.args[1], ast.Constant)
    }
    # Plus the dynamic reads the AST cannot see: the ``LowpSpec`` field loop, the ``AXES`` size loop, and
    # the by-name forwarding, which reads every field the two dataclasses spell identically.
    return names | set(_LOWP_ARG_FIELDS) | {axis.flag for axis in AXES} | _SAME_NAME_FIELDS


def _config_kwargs_built_by_builder() -> set[str]:
    """Every ``ParallelismConfig`` keyword the builder assembles: dict literal keys, the
    ``kwargs.update(...)`` keywords and the ``AXES`` size fields."""
    tree = _builder_ast()
    keys: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Dict):
            keys |= {k.value for k in node.keys if isinstance(k, ast.Constant)}
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "update":
            keys |= {kw.arg for kw in node.keywords if kw.arg}
    return keys | {axis.size_field for axis in AXES} | _SAME_NAME_FIELDS


# The settings a knob is valid only beside: the native backend leaves dense projections to nothing, and an fp8 backward
# runs on it alone.
_LOWP_COMPANIONS = {
    "lowp_backend": {"lowp_apply_dense_mlp": False},
    "lowp_backward_precision": {"lowp_backend": "native", "lowp_apply_dense_mlp": False},
}


@pytest.mark.parametrize(("arg_field", "spec_field"), sorted(_LOWP_ARG_FIELDS.items()))
def test_every_lowp_knob_lands_on_the_built_spec(arg_field, spec_field):
    """A ``lowp_*`` knob set away from its default reaches ``ParallelismConfig.lowp`` on the SFT path."""
    requested = _non_default_value(arg_field)
    # Under an enabled mode, so a shape knob is not the bf16 no-op.
    companions = _LOWP_COMPANIONS.get(arg_field, {})
    dist_args = ParallelismArguments(**{"lowp_precision": "fp8", **companions, arg_field: requested})

    config = parallelism_config_from_args(dist_args, trainer_cls=_LowpTrainer)

    assert getattr(config.lowp, spec_field) == requested


def test_ep_scope_auto_default_resolves_a_hand_built_cross_domain_config():
    """A hand-built config for 16 GPUs of EP over two 8-GPU NVLink domains resolves to global scope. A ``"node"``
    default would reject it while the identical YAML run, which passes ep_scope='auto', is accepted."""
    config = make_parallelism_config(ep_size=16, world_size=16, gpus_per_node=8)
    assert config.ep_scope == "global"


def test_ep_scope_auto_default_stays_node_local_within_one_domain():
    config = make_parallelism_config(ep_size=8, world_size=8, gpus_per_node=8)
    assert config.ep_scope == "node"


def test_every_declared_argument_reaches_the_builder():
    """A knob whose two declarations disagree on the name needs an explicit line in
    ``parallelism_config_from_args``, or it parses, shows in ``--help`` and silently keeps the runtime default."""
    declared = {f.name for f in fields(ParallelismArguments)}
    unforwarded = declared - _dist_args_names_read_by_builder()
    assert unforwarded == set(), (
        f"{sorted(unforwarded)} are declared on ParallelismArguments but never read by "
        "parallelism_config_from_args: the knob parses and does nothing. Forward it, or move it off "
        "ParallelismArguments to the arguments its consumer reads."
    )


def _non_default_value(arg_field: str):
    """A valid non-default value, derived from the declared type so the sweep needs no table of values."""
    annotation = get_type_hints(ParallelismArguments)[arg_field]
    default = _defaults(ParallelismArguments)[arg_field]
    choices = _literal_choices(annotation)
    if choices is not None:
        return next(choice for choice in choices if choice != default)
    declared = [t for t in get_args(annotation) if t is not type(None)] or [annotation]
    if bool in declared:
        return not default
    if int in declared:
        return (default or 0) + 1
    raise AssertionError(f"no non-default value rule for ParallelismArguments.{arg_field}: {annotation}")


@pytest.mark.parametrize("arg_field", sorted(_SAME_NAME_FIELDS))
def test_every_same_name_knob_lands_on_the_built_config(arg_field):
    """A forward that goes missing leaves the runtime default standing, which is the value the user overrode."""
    requested = _non_default_value(arg_field)

    config = parallelism_config_from_args(
        ParallelismArguments(**{arg_field: requested}),
        trainer_cls=_LowpTrainer,
    )

    assert getattr(config, arg_field) == requested, (
        f"ParallelismArguments.{arg_field}={requested!r} never reached ParallelismConfig.{arg_field} "
        f"(got {getattr(config, arg_field)!r}): the knob parses and the runtime keeps its own default."
    )


class _WithFieldRemoved:
    """A ``ParallelismArguments`` proxy with one attribute absent: a ``getattr(..., <default>)`` read hands back the
    default, and for ``context_parallel_size`` the silent ``1`` lets the CP-rejection guard accept a config that
    requested CP."""

    def __init__(self, real, missing: str):
        self._real = real
        self._missing = missing

    def __getattr__(self, name: str):
        if name == self.__dict__["_missing"]:
            raise AttributeError(name)
        return getattr(self.__dict__["_real"], name)


@pytest.mark.parametrize(
    "missing", ["lowp_precision", "init_from_scratch", "context_parallel_size", "pipeline_parallel_size"]
)
def test_a_missing_declared_field_is_loud(missing):
    dist_args = _WithFieldRemoved(ParallelismArguments(), missing)
    with pytest.raises(AttributeError, match=missing):
        parallelism_config_from_args(dist_args, trainer_cls=DistributedTrainerMixin)


def test_every_settable_config_field_has_an_argument_surface():
    """A field the builder never sets has no YAML/CLI reach."""
    settable = {f.name for f in fields(ParallelismConfig) if f.init}
    unreachable = settable - _config_kwargs_built_by_builder() - _CONFIG_FIELDS_NOT_FROM_ARGS
    assert unreachable == set(), (
        f"{sorted(unreachable)} are settable ParallelismConfig fields that parallelism_config_from_args "
        "never passes: no YAML or CLI can reach them."
    )


def _documented_names() -> set[str]:
    """Every inline-code identifier in the configuration reference, with any flag dashes stripped."""
    text = (REPO_ROOT / "agent-docs/reference/configuration-reference.md").read_text(encoding="utf-8")
    return set(re.findall(r"`-{0,2}([a-z_][a-z0-9_]*)`", text))


_DOCUMENTED = _documented_names()

# init=False fields are derived in __post_init__, not knobs; the reference covers them as a group.
_SETTABLE_CONFIG_FIELDS = sorted(f.name for f in fields(ParallelismConfig) if f.init)


def test_the_documentation_sweep_sees_the_reference():
    assert len(_DOCUMENTED) > 100, f"the configuration reference parsed to {len(_DOCUMENTED)} identifiers"


@pytest.mark.parametrize("arg_field", sorted(f.name for f in fields(ParallelismArguments)))
def test_every_argument_is_documented(arg_field):
    """The YAML/CLI surface is only reachable through the reference; nothing else lists these names."""
    assert arg_field in _DOCUMENTED, (
        f"ParallelismArguments.{arg_field} appears nowhere in agent-docs/reference/configuration-reference.md: "
        "a user cannot set a knob whose name is not written down. Add its row, or delete the field."
    )


@pytest.mark.parametrize("config_field", _SETTABLE_CONFIG_FIELDS)
def test_every_settable_config_field_is_documented(config_field):
    """Same for the config-side spellings, which differ from the argument names for nine knobs."""
    assert config_field in _DOCUMENTED, (
        f"ParallelismConfig.{config_field} is settable but appears nowhere in "
        "agent-docs/reference/configuration-reference.md."
    )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
