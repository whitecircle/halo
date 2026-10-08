"""``H4ArgumentParser`` — YAML-plus-CLI argument parsing with the toolkit's config defaults."""

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

from src.distributed.runtime import broadcast_from_rank0, init_distributed
from src.training.run_logging import install_log_tee

_yaml = YAML(typ="safe")

# Toolkit defaults differing from upstream; applied only when not explicitly set in YAML or CLI.
_TOOLKIT_DEFAULTS = {
    "use_liger_kernel": True,
    "bf16": True,
    # Upstream's True reads a device scalar back to the host on every micro-batch (stalling the
    # launch queue `gradient_accumulation_steps` times per step, every rank waiting on the slowest)
    # and logs a running average in place of a NaN instead of surfacing it.
    "logging_nan_inf_filter": False,
}

# A toolkit default applies only when its guard holds over the explicitly-set values: bf16 yields to
# an explicitly-enabled fp16, which would otherwise form the fp16+bf16 pair TrainingArguments rejects.
_TOOLKIT_DEFAULT_GUARDS = {
    "bf16": lambda values: not values.get("fp16"),
}

_TRUE_STRINGS = ("true", "1", "yes")
_FALSE_STRINGS = ("false", "0", "no")
_NONE_STRINGS = ("None", "null", "none")

_YAML_SUFFIXES = (".yaml", ".yml")

# Pre-seeded on every argparse destination, so a flag left off the command line stays recognizable.
_UNSET = object()

# Matched left to right: an existing `%%` escape, a `%(name)s` placeholder, or (captured) a bare
# percent that still needs escaping.
_PERCENT_TOKENS = re.compile(r"%%|%\(|(%)")

# A strftime directive to expand in output_dir: `%<letter>` or the `%%` escape. Anything else
# (`100%-data`, `50%_subset`) is prose and must survive byte-identical; a whole-string strftime
# would consume it, since glibc reads `%-d` as a no-padding day-of-month.
_STRFTIME_DIRECTIVE = re.compile(r"%(?:%|[a-zA-Z])")


def is_true_string(value: str) -> bool:
    """Whether a CLI/YAML string spells boolean true, in the vocabulary the bool casting below uses.

    Exported for the ``str | bool`` fields whose CLI override arrives uncast
    (``--resume_from_checkpoint=true``), so consumers read it the way ``--packing=true`` is read.
    """
    return value.strip().lower() in _TRUE_STRINGS


def is_null_string(value: str) -> bool:
    """Whether a CLI/YAML string spells null in the parser's vocabulary, compared case-insensitively.

    Exported for the non-Optional ``str`` fields, where the parser leaves such a spelling as literal
    text.
    """
    return value.strip().lower() in {spelling.lower() for spelling in _NONE_STRINGS}


class PercentSafeHelpFormatter(argparse.ArgumentDefaultsHelpFormatter):
    """Help formatter that renders a ``%`` in field help as a literal percent sign.

    argparse expands every help string with ``help % params``, so prose containing a bare percent
    breaks ``--help`` for the whole script. Already-escaped ``%%`` and argparse's own
    ``%(default)s`` expansion are left intact.
    """

    def _get_help_string(self, action: argparse.Action) -> str:
        help_string = super()._get_help_string(action) or ""
        return _PERCENT_TOKENS.sub(lambda match: "%%" if match.group(1) else match.group(0), help_string)


def _literal_choices(annotation) -> tuple | None:
    """Allowed values for a ``Literal[...]`` annotation, unwrapping Optional/Union.

    ``None`` (no constraint) unless every union member is a ``Literal`` or ``NoneType``: a mixed
    union (``float | Literal["auto"]``) admits values outside the literal set.
    """
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
    """Whether a CLI string value has no confident cast to ``annotation``.

    True for dicts, for lists of containers (comma-splitting ``list[dict]`` yields ``list[str]`` and
    a TypeError deep in its consumer), and for multi-member unions with a container member
    (``report_to: str | list[str]`` would be iterated char-wise). Scalar unions keep their cast.
    """
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
    """Whether ``annotation`` admits ``bool`` but not ``str`` (unwrapping Optional/Union).

    A string on such a field is a spelling mistake: every non-empty string is truthy, so it would
    invert an intended false. A union that also admits ``str`` is left alone.
    """
    if annotation is bool:
        return True
    if get_origin(annotation) not in (Union, types.UnionType):
        return False
    members = get_args(annotation)
    return bool in members and str not in members


def _argv_yaml_path() -> str | None:
    """Absolute path of the YAML config when ``sys.argv[1]`` is one, else ``None``: whether this was
    invoked as ``<script> config.yaml [--overrides]``."""
    if len(sys.argv) >= 2 and sys.argv[1].endswith(_YAML_SUFFIXES):
        return os.path.abspath(sys.argv[1])
    return None


def _load_yaml(yaml_file: str) -> dict[str, Any]:
    """The YAML config as a dict, read as YAML 1.2; an empty file is an empty config."""
    return _yaml.load(Path(yaml_file)) or {}


def _union_permits_str(annotation) -> bool:
    """Whether ``str`` is one of the union members (e.g. ``float | str``).

    Such a consumer already handles a sentinel like ``"auto"``, so a value failing the numeric cast
    is kept as the string rather than crashing the override.
    """
    if get_origin(annotation) in (Union, types.UnionType):
        return str in get_args(annotation)
    return False


def _init_field_names(dataclass_type) -> set[str]:
    """The fields ``dataclass_type`` takes in ``__init__``, the keys ``parse_dict`` hands it."""
    return {f.name for f in dataclasses.fields(dataclass_type) if f.init}


def _expand_strftime_directives(template: str) -> str:
    """Expand ``%<letter>`` strftime directives in ``template``, leaving every other ``%`` intact.

    Each directive is formatted on its own from one shared timestamp; an unrecognized letter comes
    back unchanged, and ``%%`` keeps its strftime meaning (a literal ``%``).
    """
    now = datetime.now()
    return _STRFTIME_DIRECTIVE.sub(lambda m: "%" if m.group(0) == "%%" else now.strftime(m.group(0)), template)


def _expand_output_dir(values: dict[str, Any]) -> dict[str, Any]:
    """``values`` with the strftime directives in ``output_dir`` expanded — rank 0's expansion, broadcast.

    A per-rank ``datetime.now()`` can straddle a second boundary, and the log tee installs from the
    built config: on a non-shared output filesystem each node's save rank would then tee ``run.log``
    into a directory other than the run's output_dir.
    """
    output_dir = values.get("output_dir")
    if not isinstance(output_dir, str):
        return values
    return {**values, "output_dir": broadcast_from_rank0(_expand_strftime_directives(output_dir))}


class H4ArgumentParser(HfArgumentParser):
    def __init__(self, *args, **kwargs):
        kwargs.setdefault("conflict_handler", "resolve")
        kwargs.setdefault("formatter_class", PercentSafeHelpFormatter)
        super().__init__(*args, **kwargs)

    def parse_yaml_file(self, yaml_file: str, allow_extra_keys: bool = False):
        """Override to load the YAML as 1.2 and route it through this parser's ``parse_dict``.

        Upstream's PyYAML read resolves ``packing: no`` to a boolean under YAML 1.1, rather than to
        the string :meth:`_validate_field_values` rejects. No spelling is migrated: an unrecognized
        key reaches the strict check and raises, naming the field.
        """
        return self.parse_dict(_load_yaml(yaml_file), allow_extra_keys=allow_extra_keys)

    def parse_dict(self, args: dict[str, Any], allow_extra_keys: bool = False):
        """Override to validate raw values, which the dict path otherwise accepts unchecked.

        ``HfArgumentParser.parse_dict`` constructs the dataclasses directly, so neither argparse
        ``choices`` nor type conversion runs: an invalid ``Literal`` would surface deep in training,
        and a YAML 1.1 boolean lands as a truthy string on a bool field.
        """
        self._validate_field_values(args)
        return super().parse_dict(args, allow_extra_keys=allow_extra_keys)

    def _validate_field_values(self, data: dict[str, Any]) -> None:
        """Raise when a raw config value cannot mean what its field's annotation declares.

        Validates the raw input (pre-``__post_init__``) against every target dataclass declaring the
        field, matching how ``parse_dict`` distributes keys. Annotations come from the class:
        ``HfArgumentParser.__init__`` rewrites ``Field.type`` in place for argparse, where
        ``Optional[Literal[...]]`` loses its ``None`` member.
        """
        for dataclass_type in self.dataclass_types:
            annotations = get_type_hints(dataclass_type)
            for field in dataclasses.fields(dataclass_type):
                if not field.init or field.name not in data:
                    continue
                value = data[field.name]
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

    def parse_yaml_and_args(self, yaml_arg: str, other_args: list[str] | None = None) -> tuple[Any, ...]:
        """Parse a YAML file with CLI args (e.g. ``['--arg=val']``) overriding its values.

        Each dataclass is built once from the merged values, so every ``__post_init__`` — TRL's,
        transformers' and the toolkit's — derives its state from, and validates, the final config.
        """
        return self.parse_dict(self._merge_cli_overrides(_load_yaml(os.path.abspath(yaml_arg)), other_args))

    def _merge_cli_overrides(self, values: dict[str, Any], other_args: list[str] | None) -> dict[str, Any]:
        """``values`` with each ``--key=value`` CLI arg, cast to its field's annotation, set over it."""
        overrides: dict[str, str] = {}
        for arg in other_args or []:
            if "=" not in arg:
                raise ValueError(f"CLI overrides must be in --key=value form, got {arg!r}")
            key, _, value = arg.partition("=")
            # Underscore form, matching argparse (which accepts both spellings) and the field names.
            key = key.strip("-").replace("-", "_")
            # Reject a repeated flag (--lr=1 --lr=2); the dict would otherwise keep only the last.
            if key in overrides:
                raise ValueError(f"Duplicate CLI override provided: {key!r}")
            overrides[key] = value
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
        """``raw`` cast to the type field ``name`` declares, the value a YAML would carry for it.

        The cast value reaches every dataclass declaring the field, so they must declare one type.
        """
        declarers = [dt for dt in self.dataclass_types if name in _init_field_names(dt)]
        # HfArgumentParser.__init__ rewrites Field.type in place for argparse, so annotations come from the class.
        annotation, *others = (get_type_hints(dataclass_type)[name] for dataclass_type in declarers)
        if any(other != annotation for other in others):
            raise ValueError(
                f"CLI override --{name}: {[dt.__name__ for dt in declarers]} declare it with different types, "
                f"so one string cannot be cast for all of them. Set '{name}' in the YAML config instead."
            )
        declared = declarers[0].__dataclass_fields__[name]
        # The rewritten Field.type is the one member argparse would cast to (``float | str`` -> float).
        base_type = declared.type
        origin = get_origin(base_type)
        if origin in (Union, types.UnionType):
            members = [a for a in get_args(base_type) if a is not type(None)]
            if len(members) == 1:
                base_type = members[0]
                origin = get_origin(base_type)

        literal_choices = _literal_choices(annotation)
        if literal_choices is not None:
            # Match by string form so str and int literals cast alike. A literal choice spelled "none"
            # takes precedence over None-clearing, so --moe_balancing=none selects the string, while
            # --field=None still clears an Optional.
            matches = [c for c in literal_choices if c is not None and str(c) == raw]
            if matches:
                return matches[0]
            if raw in _NONE_STRINGS and None in literal_choices:
                return None
            raise ValueError(
                f"Invalid value {raw!r} for CLI override --{name}: allowed values are {list(literal_choices)}"
            )
        if raw in _NONE_STRINGS and isinstance(declared.default, str) and is_null_string(declared.default):
            # The field spells "no value" as a string of its own, which its __post_init__ reads and None
            # is not: --report_to=none must reach transformers as "none" (no integrations), where None
            # becomes [None] and fails the Trainer.
            return declared.default
        if raw in _NONE_STRINGS and type(None) in get_args(annotation) and not _is_bool_only(annotation):
            # Symmetric with YAML's null: any optional field clears from the CLI, including unions with
            # no confident value cast. Optional bools are exempt: --bf16=none reads as a mistyped
            # boolean, and clearing it would flip the run's precision or re-arm an auto-default, so the
            # bool branch raises instead.
            return None
        if _is_cli_uncastable(annotation):
            # A raw string in a container field is iterated char-wise by its consumer.
            raise ValueError(
                f"CLI override --{name} is not supported: field type {annotation} cannot be cast from a "
                f"string. Set '{name}' in the YAML config instead."
            )
        if base_type in (int, float):
            try:
                return base_type(raw)
            except ValueError:
                # ``float | str`` admits a sentinel its consumer resolves ("auto").
                if not _union_permits_str(annotation):
                    raise
                return raw
        if origin is list:
            elem_args = get_args(base_type)
            elem_cast = elem_args[0] if elem_args and elem_args[0] in (int, float, str) else str
            return [elem_cast(v) for v in raw.split(",")]
        if base_type is bool:
            lowered = raw.strip().lower()
            if is_true_string(lowered):
                return True
            if lowered in _FALSE_STRINGS:
                return False
            raise ValueError(f"Cannot parse boolean CLI override --{name}={raw!r}")
        return raw

    def _flag_values(self, args: list[str] | None = None) -> dict[str, Any]:
        """The values of a flags-only command line, each flag given to every dataclass that declares it.

        Stands in for ``parse_args_into_dataclasses``, which deletes a key from the shared namespace once
        its first declaring dataclass consumes it, so a field two dataclasses declare (``pad_token``
        on the script args and TRL's config) raises for the second. Only the flags given are returned,
        as a YAML carries only its keys: an unset shared field keeps each declarer's own default. An
        unknown flag is argparse's usage error.
        """
        seeded = argparse.Namespace(
            **{action.dest: _UNSET for action in self._actions if action.dest != argparse.SUPPRESS}
        )
        namespace = self.parse_args(args, namespace=seeded)
        return {key: value for key, value in vars(namespace).items() if value is not _UNSET}

    def parse(self) -> Any:
        """Parse the script's YAML config and/or CLI overrides into its declared dataclasses.

        Returns them in declaration order, or the single dataclass when only one type was declared.
        The shape is per-script, so the return annotation can say no more than ``Any``.
        """
        # Before the dataclasses are built: constructing a TrainingArguments touches ``self.device``,
        # which makes accelerate initialize the default process group without ``device_id``, costing
        # a ``new_group`` per mesh dim instead of one ``ncclCommSplit`` and leaving every barrier to
        # infer its device.
        init_distributed()

        yaml_path = _argv_yaml_path()
        if yaml_path is None:
            values = self._flag_values()
        else:
            values = self._merge_cli_overrides(_load_yaml(yaml_path), sys.argv[2:])
        output = self.parse_dict(_expand_output_dir(self._with_toolkit_defaults(values)))

        for obj in output:
            if isinstance(getattr(obj, "output_dir", None), str):
                install_log_tee(obj.output_dir)
                break

        if len(output) == 1:
            output = output[0]
        return output

    def _with_toolkit_defaults(self, values: dict[str, Any]) -> dict[str, Any]:
        """``values`` plus each toolkit default it leaves unset, as if the YAML carried it."""
        declared = self._declared_field_names()
        defaults = {}
        for name, default in _TOOLKIT_DEFAULTS.items():
            guard = _TOOLKIT_DEFAULT_GUARDS.get(name)
            if name in declared and name not in values and (guard is None or guard(values)):
                defaults[name] = default
        return {**defaults, **values}
