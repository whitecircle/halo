"""``HaloArgumentParser``: YAML-plus-CLI argument parsing with the toolkit's config defaults."""

import argparse
import dataclasses
import os
import re
import sys
import types
from datetime import datetime
from pathlib import Path
from typing import Any, Literal, Union, get_args, get_origin, get_type_hints

from ruamel.yaml import YAML
from transformers import HfArgumentParser

# Name -> (value, guard): applies only to a field the config leaves unset, and only when the guard holds over the rest.
_TOOLKIT_DEFAULTS = {
    "use_liger_kernel": (True, None),
    # Yields to an explicitly-enabled fp16, which would otherwise form the fp16+bf16 pair TrainingArguments rejects.
    "bf16": (True, lambda values: not values.get("fp16")),
    # Upstream's True reads a device scalar back per micro-batch (stalling the launch queue, every rank waiting on the
    # slowest) and logs a running average in place of a NaN.
    "logging_nan_inf_filter": (False, None),
}

# (name, key) -> (value, guard): merged into the dict the config gives, never over a key; a JSON file path stays whole.
_TOOLKIT_KEY_DEFAULTS = {
    # A blocking copy of a pinned batch waits for queued kernels to drain; off where ``use_cpu`` unpins.
    ("accelerator_config", "non_blocking"): (
        True,
        lambda values: values.get("dataloader_pin_memory") is not False and not values.get("use_cpu"),
    ),
}

# A key that changed meaning under its old name, refused with the remedy instead of read as the new field.
_RETIRED_KEYS = {
    "num_rollout_workers": (
        "it was a per-rank actor count (default 64) and is now rollout_actor_budget, a per-node budget (default 256; "
        "cluster-wide with ray_address). Set rollout_actor_budget to the old value times the node's rollout ranks."
    ),
}

_NONE_STRINGS = ("None", "null", "none")

# Pre-seeded on every argparse destination, so a flag left off the command line stays recognizable.
_UNSET = object()

# Left to right: an existing `%%` escape, a `%(name)s` placeholder, or (captured) a bare percent to escape.
_PERCENT_TOKENS = re.compile(r"%%|%\(|(%)")

# `%<letter>` or the `%%` escape. Anything else (`100%-data`) must survive; a whole-string strftime would eat it, since
# glibc reads `%-d` as a no-padding day-of-month.
_STRFTIME_DIRECTIVE = re.compile(r"%(?:%|[a-zA-Z])")


def is_true_string(value: str) -> bool:
    """For a ``str | bool`` field whose CLI override arrives uncast (``--resume_from_checkpoint=true``)."""
    return value.strip().lower() in ("true", "1", "yes")


def is_null_string(value: str) -> bool:
    """For a non-Optional ``str`` field the parser leaves a null spelling on as literal text."""
    return value.strip().lower() in {spelling.lower() for spelling in _NONE_STRINGS}


def expand_strftime_directives(template: str) -> str:
    """From one shared timestamp; ``%%`` is a literal ``%`` and every other ``%`` survives."""
    now = datetime.now()
    return _STRFTIME_DIRECTIVE.sub(lambda m: "%" if m.group(0) == "%%" else now.strftime(m.group(0)), template)


class _PercentSafeHelpFormatter(argparse.ArgumentDefaultsHelpFormatter):
    """Renders a bare ``%`` literally: argparse formats help with ``help % params``, where one would break ``--help``
    for the whole script."""

    def _get_help_string(self, action: argparse.Action) -> str:
        help_string = super()._get_help_string(action) or ""
        return _PERCENT_TOKENS.sub(lambda match: "%%" if match.group(1) else match.group(0), help_string)


def _literal_choices(annotation) -> tuple | None:
    """Unwraps Optional/Union; ``None`` (no constraint) for a mixed union like ``float | Literal["auto"]``."""
    if get_origin(annotation) is Literal:
        return get_args(annotation)
    if get_origin(annotation) in (Union, types.UnionType):
        choices: list = []
        for member in get_args(annotation):
            if member is type(None):
                choices.append(None)
            elif get_origin(member) is Literal:
                choices.extend(get_args(member))
            else:
                return None
        return tuple(choices)
    return None


def _is_cli_uncastable(annotation) -> bool:
    """A dict, a list of containers, or a union with a container member, which its consumer would iterate char-wise or
    fail on far from the flag."""
    origin = get_origin(annotation)
    if annotation is dict or origin is dict:
        return True
    if origin is list:
        elem_args = get_args(annotation)
        return bool(elem_args) and (elem_args[0] in (list, dict) or get_origin(elem_args[0]) in (list, dict))
    if origin in (Union, types.UnionType):
        members = [a for a in get_args(annotation) if a is not type(None)]
        if len(members) == 1:
            return _is_cli_uncastable(members[0])
        return any(member in (list, dict) or get_origin(member) in (list, dict) for member in members)
    return False


def _is_bool_only(annotation) -> bool:
    """Admits ``bool`` but not ``str``, where a string is a spelling mistake: every non-empty string is truthy."""
    if annotation is bool:
        return True
    if get_origin(annotation) not in (Union, types.UnionType):
        return False
    members = get_args(annotation)
    return bool in members and str not in members


def _load_yaml(yaml_file: str) -> dict[str, Any]:
    """Read as YAML 1.2; an empty file is an empty config."""
    return YAML(typ="safe").load(Path(yaml_file)) or {}


def _reject_retired_keys(keys) -> None:
    retired = {key: _RETIRED_KEYS[key] for key in keys if key in _RETIRED_KEYS}
    if retired:
        raise ValueError("; ".join(f"'{key}' is retired: {remedy}" for key, remedy in retired.items()))


def _init_field_names(dataclass_type) -> set[str]:
    return {f.name for f in dataclasses.fields(dataclass_type) if f.init}


def _cast_to_field_type(name: str, raw: str, annotation, field_type) -> Any:
    """``field_type`` is the ``Field.type`` ``HfArgumentParser.__init__`` rewrote to the one member argparse would cast
    to (``float | str`` -> float); ``annotation`` is the class's own."""
    origin = get_origin(field_type)
    if origin in (Union, types.UnionType):
        members = [a for a in get_args(field_type) if a is not type(None)]
        if len(members) == 1:
            field_type = members[0]
            origin = get_origin(field_type)
    if field_type in (int, float):
        try:
            return field_type(raw)
        except ValueError:
            # ``float | str`` admits a sentinel its consumer resolves ("auto").
            if get_origin(annotation) not in (Union, types.UnionType) or str not in get_args(annotation):
                raise
            return raw
    if origin is list:
        elem_args = get_args(field_type)
        elem_cast = elem_args[0] if elem_args and elem_args[0] in (int, float, str) else str
        return [elem_cast(v) for v in raw.split(",")]
    if field_type is bool:
        lowered = raw.strip().lower()
        if is_true_string(lowered):
            return True
        if lowered in ("false", "0", "no"):
            return False
        raise ValueError(f"Cannot parse boolean CLI override --{name}={raw!r}")
    return raw


class HaloArgumentParser(HfArgumentParser):
    def __init__(self, *args, **kwargs):
        kwargs.setdefault("conflict_handler", "resolve")
        kwargs.setdefault("formatter_class", _PercentSafeHelpFormatter)
        super().__init__(*args, **kwargs)

    def load_values(self, argv: list[str] | None = None) -> dict[str, Any]:
        """The values ``argv`` sets, plus each toolkit default it leaves unset. ``argv[0]`` is the YAML config when it
        ends in .yaml/.yml and the rest its ``--key=value`` overrides, else ``argv`` is a flags-only command line."""
        argv = sys.argv[1:] if argv is None else argv
        if argv and argv[0].endswith((".yaml", ".yml")):
            values = self._merge_cli_overrides(_load_yaml(os.path.abspath(argv[0])), argv[1:])
        else:
            values = self._flag_values(argv)
        declared = self._declared_field_names()
        defaults = {
            name: default
            for name, (default, guard) in _TOOLKIT_DEFAULTS.items()
            if name in declared and name not in values and (guard is None or guard(values))
        }
        values = {**defaults, **values}
        for (name, key), (default, guard) in _TOOLKIT_KEY_DEFAULTS.items():
            given = values.get(name) or {}
            if name in declared and isinstance(given, dict) and key not in given and guard(values):
                values[name] = {**given, key: default}
        return values

    def parse_yaml_file(self, yaml_file: str, allow_extra_keys: bool = False):
        """Read as YAML 1.2 (upstream's 1.1 read makes ``packing: no`` a boolean); no spelling is migrated, so an
        unknown key raises naming the field."""
        return self.parse_dict(_load_yaml(yaml_file), allow_extra_keys=allow_extra_keys)

    def parse_dict(self, values: dict[str, Any], allow_extra_keys: bool = False):
        """Refuse a raw value its field's annotation rules out, then build: ``HfArgumentParser.parse_dict`` runs
        neither argparse ``choices`` nor a type cast. Annotations come from the class: ``HfArgumentParser.__init__``
        rewrites ``Field.type`` in place, where ``Optional[Literal[...]]`` loses its ``None``."""
        _reject_retired_keys(values)
        for dataclass_type in self.dataclass_types:
            annotations = get_type_hints(dataclass_type)
            for field in dataclasses.fields(dataclass_type):
                if not field.init or field.name not in values:
                    continue
                value = values[field.name]
                annotation = annotations[field.name]
                choices = _literal_choices(annotation)
                if choices is not None and value not in choices:
                    raise ValueError(
                        f"Invalid value {value!r} for field '{field.name}' of "
                        f"{dataclass_type.__name__}: allowed values are {list(choices)}"
                    )
                if isinstance(value, str) and _is_bool_only(annotation):
                    raise ValueError(
                        f"Field '{field.name}' of {dataclass_type.__name__} is boolean but got the "
                        f"string {value!r}. YAML 1.2 booleans are unquoted true/false — the 1.1 "
                        f"spellings yes/no/on/off parse as (truthy) strings."
                    )
        return super().parse_dict(values, allow_extra_keys=allow_extra_keys)

    def _merge_cli_overrides(self, values: dict[str, Any], other_args: list[str]) -> dict[str, Any]:
        overrides: dict[str, str] = {}
        for arg in other_args:
            if "=" not in arg:
                raise ValueError(f"CLI overrides must be in --key=value form, got {arg!r}")
            key, _, value = arg.partition("=")
            key = key.strip("-").replace("-", "_")
            if key in overrides:
                raise ValueError(f"Duplicate CLI override provided: {key!r}")
            overrides[key] = value
        _reject_retired_keys(overrides)
        unmatched = set(overrides) - self._declared_field_names()
        if unmatched:
            raise ValueError(
                f"Unknown CLI override(s) {sorted(unmatched)}: no field with these names exists on any "
                f"of the parsed config dataclasses."
            )
        return {**values, **{key: self._cast_cli_override(key, raw) for key, raw in overrides.items()}}

    def _declared_field_names(self) -> set[str]:
        return set().union(*(_init_field_names(dataclass_type) for dataclass_type in self.dataclass_types))

    def _cast_cli_override(self, name: str, raw: str) -> Any:
        """The value a YAML would carry; every dataclass declaring the field gets it, so they must declare one type."""
        declarers = [dt for dt in self.dataclass_types if name in _init_field_names(dt)]
        # HfArgumentParser.__init__ rewrites Field.type in place for argparse, so annotations come from the class.
        annotation, *others = (get_type_hints(dataclass_type)[name] for dataclass_type in declarers)
        if any(other != annotation for other in others):
            raise ValueError(
                f"CLI override --{name}: {[dt.__name__ for dt in declarers]} declare it with different types, "
                f"so one string cannot be cast for all of them. Set '{name}' in the YAML config instead."
            )
        declared = declarers[0].__dataclass_fields__[name]
        literal_choices = _literal_choices(annotation)
        if literal_choices is not None:
            # Match by string form so str and int literals cast alike. A literal "none" wins over None-clearing:
            # --moe_balancing=none selects the string, --field=None still clears an Optional.
            matches = [c for c in literal_choices if c is not None and str(c) == raw]
            if matches:
                return matches[0]
            if raw in _NONE_STRINGS and None in literal_choices:
                return None
            raise ValueError(
                f"Invalid value {raw!r} for CLI override --{name}: allowed values are {list(literal_choices)}"
            )
        if raw in _NONE_STRINGS and isinstance(declared.default, str) and is_null_string(declared.default):
            # --report_to=none must reach transformers as "none": None becomes [None] and fails the Trainer.
            return declared.default
        if raw in _NONE_STRINGS and type(None) in get_args(annotation) and not _is_bool_only(annotation):
            # As YAML's null, any optional field clears from the CLI. Optional bools are exempt: --bf16=none reads as a
            # mistyped boolean, and clearing it would flip the run's precision or re-arm an auto-default.
            return None
        if _is_cli_uncastable(annotation):
            # A raw string in a container field is iterated char-wise by its consumer.
            raise ValueError(
                f"CLI override --{name} is not supported: field type {annotation} cannot be cast from a "
                f"string. Set '{name}' in the YAML config instead."
            )
        return _cast_to_field_type(name, raw, annotation, declared.type)

    def _flag_values(self, args: list[str]) -> dict[str, Any]:
        """Each flag goes to every dataclass declaring it, where ``parse_args_into_dataclasses`` raises for a shared
        field's second declarer (``pad_token``). An unset flag is left out; an unknown one is a usage error."""
        seeded = argparse.Namespace(
            **{action.dest: _UNSET for action in self._actions if action.dest != argparse.SUPPRESS}
        )
        namespace = self.parse_args(args, namespace=seeded)
        return {key: value for key, value in vars(namespace).items() if value is not _UNSET}
