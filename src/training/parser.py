"""``H4ArgumentParser`` — YAML-plus-CLI argument parsing with the toolkit's config defaults."""

import argparse
import dataclasses
import enum
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

# A toolkit default applies only when its guard holds: bf16 yields to an explicitly-enabled fp16,
# which would otherwise form the fp16+bf16 pair TrainingArguments.__post_init__ rejects.
_TOOLKIT_DEFAULT_GUARDS = {
    "bf16": lambda obj: not getattr(obj, "fp16", False),
}

# TrainingArguments.__post_init__ derives ``mixed_precision`` from these; mutating them post-parse must re-run it.
_PRECISION_FLAGS = ("fp16", "bf16")


_TRUE_STRINGS = ("true", "1", "yes")
_FALSE_STRINGS = ("false", "0", "no")
_NONE_STRINGS = ("None", "null", "none")

_YAML_SUFFIXES = (".yaml", ".yml")

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
    """Absolute path of the YAML config when ``sys.argv[1]`` is one, else ``None``.

    Answers "was this invoked as ``<script> config.yaml [--overrides]``"; the parse dispatch and the
    explicitly-set-field scan branch on the same answer.
    """
    if len(sys.argv) >= 2 and sys.argv[1].endswith(_YAML_SUFFIXES):
        return os.path.abspath(sys.argv[1])
    return None


def _union_permits_str(annotation) -> bool:
    """Whether ``str`` is one of the union members (e.g. ``float | str``).

    Such a consumer already handles a sentinel like ``"auto"``, so a value failing the numeric cast
    is kept as the string rather than crashing the override.
    """
    if get_origin(annotation) in (Union, types.UnionType):
        return str in get_args(annotation)
    return False


def _expand_strftime_directives(template: str) -> str:
    """Expand ``%<letter>`` strftime directives in ``template``, leaving every other ``%`` intact.

    Each directive is formatted on its own from one shared timestamp; an unrecognized letter comes
    back unchanged, and ``%%`` keeps its strftime meaning (a literal ``%``).
    """
    now = datetime.now()
    return _STRFTIME_DIRECTIVE.sub(lambda m: "%" if m.group(0) == "%%" else now.strftime(m.group(0)), template)


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
        return self.parse_dict(_yaml.load(Path(yaml_file)) or {}, allow_extra_keys=allow_extra_keys)

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

    def parse_yaml_and_args(self, yaml_arg: str, other_args: list[str] | None = None) -> list[Any]:
        """Parse a YAML file, then overwrite its values with CLI args (e.g. ['--arg=val'])."""
        arg_list = self.parse_yaml_file(os.path.abspath(yaml_arg))

        outputs = []
        parsed_other_args = {}
        for arg in other_args or []:
            if "=" not in arg:
                raise ValueError(f"CLI overrides must be in --key=value form, got {arg!r}")
            key, _, value = arg.partition("=")
            # Underscore form, matching argparse (which accepts both spellings) and the field names.
            key = key.strip("-").replace("-", "_")
            # Reject a repeated flag (--lr=1 --lr=2); the dict would otherwise keep only the last.
            if key in parsed_other_args:
                raise ValueError(f"Duplicate CLI override provided: {key!r}")
            parsed_other_args[key] = value
        used_args = set()

        # setattr, not re-instantiation: re-running __post_init__ trips validators on state it derived.
        for data_yaml in arg_list:
            overridden_fields: set[str] = set()
            keys = {f.name for f in dataclasses.fields(data_yaml) if f.init}
            # HfArgumentParser.__init__ rewrites Field.type in place, so annotations come from the class.
            annotations = get_type_hints(type(data_yaml)) if keys & set(parsed_other_args) else {}
            for arg, val in parsed_other_args.items():
                if arg not in keys:
                    continue

                base_type = data_yaml.__dataclass_fields__[arg].type
                origin = get_origin(base_type)
                if origin in (Union, types.UnionType):
                    members = [a for a in get_args(base_type) if a is not type(None)]
                    if len(members) == 1:
                        base_type = members[0]
                        origin = get_origin(base_type)

                cast_val: Any = val
                literal_choices = _literal_choices(annotations[arg])
                if literal_choices is not None:
                    # setattr bypasses argparse choices; match by string form so str and int literals
                    # cast alike. A literal choice spelled "none" takes precedence over
                    # None-clearing, so --moe_balancing=none selects the string, while --field=None
                    # still clears an Optional.
                    matches = [c for c in literal_choices if c is not None and str(c) == val]
                    if matches:
                        cast_val = matches[0]
                    elif val in _NONE_STRINGS and None in literal_choices:
                        cast_val = None
                    else:
                        raise ValueError(
                            f"Invalid value {val!r} for CLI override --{arg}: "
                            f"allowed values are {list(literal_choices)}"
                        )
                elif (
                    val in _NONE_STRINGS
                    and type(None) in get_args(annotations[arg])
                    and not _is_bool_only(annotations[arg])
                ):
                    # Symmetric with YAML's null: any optional field clears from the CLI, including
                    # unions with no confident value cast (--report_to=none on None | str |
                    # list[str], the usual way to silence logging on a smoke run). Optional bools are
                    # exempt: --bf16=none reads as a mistyped boolean, and clearing it would flip the
                    # run's precision or re-arm an auto-default, so the bool branch raises instead.
                    cast_val = None
                elif _is_cli_uncastable(annotations[arg]):
                    # setattr-ing the raw string would corrupt the field: __post_init__ conversions
                    # do not re-run, and a string in a list field is iterated char-wise.
                    raise ValueError(
                        f"CLI override --{arg} is not supported: field type {annotations[arg]} cannot "
                        f"be cast from a string. Set '{arg}' in the YAML config instead."
                    )
                elif isinstance(base_type, type) and issubclass(base_type, enum.Enum):
                    # setattr bypasses the enum coercion __post_init__ ran on the YAML value; a raw
                    # string equals no enum member, so a typo'd --save_strategy would disable
                    # checkpointing rather than fail. Let the enum's ValueError propagate.
                    cast_val = base_type(val)
                elif base_type in (int, float):
                    try:
                        cast_val = base_type(val)
                    except ValueError:
                        # ``float | str`` admits a sentinel its consumer resolves ("auto").
                        if not _union_permits_str(annotations[arg]):
                            raise
                        cast_val = val
                elif origin is list:
                    elem_args = get_args(base_type)
                    elem_cast = elem_args[0] if elem_args and elem_args[0] in (int, float, str) else str
                    cast_val = [elem_cast(v) for v in val.split(",")]
                elif base_type is bool:
                    lowered = val.strip().lower()
                    if is_true_string(lowered):
                        cast_val = True
                    elif lowered in _FALSE_STRINGS:
                        cast_val = False
                    else:
                        raise ValueError(f"Cannot parse boolean CLI override --{arg}={val!r}")

                setattr(data_yaml, arg, cast_val)
                overridden_fields.add(arg)
                used_args.add(arg)

            # Overrides bypass __init__; __post_override__ re-derives state __post_init__ computed from them.
            if overridden_fields and hasattr(data_yaml, "__post_override__"):
                data_yaml.__post_override__(overridden_fields)

            outputs.append(data_yaml)

        # Without re-deriving mixed_precision, the Accelerator autocasts with the pre-override dtype.
        if set(parsed_other_args) & set(_PRECISION_FLAGS):
            self._sync_mixed_precision(outputs)

        unmatched = set(parsed_other_args) - set(used_args)
        if unmatched:
            raise ValueError(
                f"Unknown CLI override(s) {sorted(unmatched)}: no field with these names exists on any "
                f"of the parsed config dataclasses."
            )

        return outputs

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

        explicitly_set = self._get_explicitly_set_fields()

        yaml_path = _argv_yaml_path()
        if yaml_path is None:
            output = self.parse_args_into_dataclasses()
        elif len(sys.argv) == 2:
            output = self.parse_yaml_file(yaml_path)
        else:
            output = self.parse_yaml_and_args(yaml_path, sys.argv[2:])

        self._apply_toolkit_defaults(output, explicitly_set)
        self._sync_mixed_precision(output)
        self._format_output_dir(output)

        for obj in output:
            if isinstance(getattr(obj, "output_dir", None), str):
                install_log_tee(obj.output_dir)
                break

        if len(output) == 1:
            output = output[0]
        return output

    @staticmethod
    def _get_explicitly_set_fields() -> set:
        """Determine which fields the user explicitly set via YAML file or CLI args."""
        explicitly_set = set()

        yaml_path = _argv_yaml_path()
        if yaml_path and os.path.isfile(yaml_path):
            raw = _yaml.load(Path(yaml_path)) or {}
            explicitly_set.update(raw.keys())

        # Without a YAML, sys.argv[1] is already a flag; skipping it lets a default overwrite `--bf16 false`.
        cli_args = sys.argv[2:] if yaml_path else sys.argv[1:]
        for arg in cli_args:
            # Underscore form: argparse accepts `--use-liger-kernel`, but the toolkit-default check
            # reads field names, and a dashed record would let the default overwrite the user's value.
            key = arg.split("=")[0].lstrip("-").replace("-", "_")
            explicitly_set.add(key)

        return explicitly_set

    @staticmethod
    def _apply_toolkit_defaults(parsed_objects: list, explicitly_set: set) -> None:
        """Apply toolkit-specific defaults to fields that weren't explicitly set by the user."""
        for field_name, default_value in _TOOLKIT_DEFAULTS.items():
            if field_name in explicitly_set:
                continue
            guard = _TOOLKIT_DEFAULT_GUARDS.get(field_name)
            for obj in parsed_objects:
                if hasattr(obj, field_name):
                    if guard is None or guard(obj):
                        setattr(obj, field_name, default_value)
                    break

    @staticmethod
    def _sync_mixed_precision(parsed_objects: list) -> None:
        """Re-derive ``mixed_precision`` after post-parse mutation of the precision flags.

        Mirrors ``TrainingArguments.__post_init__`` (env fallback, fp16/bf16 precedence, at-most-one
        enforcement): toolkit defaults and CLI overrides mutate the flags via setattr, and a stale
        ``mixed_precision`` makes the Accelerator autocast a dtype the model was not loaded with.
        """
        for obj in parsed_objects:
            if not all(hasattr(obj, name) for name in (*_PRECISION_FLAGS, "mixed_precision")):
                continue
            if obj.fp16 and obj.bf16:
                raise ValueError("At most one of fp16 and bf16 can be True, but not both")
            mixed_precision = os.environ.get("ACCELERATE_MIXED_PRECISION", "no")
            if obj.fp16:
                mixed_precision = "fp16"
            elif obj.bf16:
                mixed_precision = "bf16"
            obj.mixed_precision = mixed_precision

    @staticmethod
    def _format_output_dir(parsed_objects: list) -> None:
        """Expand strftime codes in ``output_dir`` — rank 0's expansion, broadcast.

        A per-rank ``datetime.now()`` can straddle a second boundary, and the log tee installs right
        after this with the local value: on a non-shared output filesystem each node's save rank
        would then tee ``run.log`` into a directory other than the run's output_dir.
        """
        for obj in parsed_objects:
            output_dir = getattr(obj, "output_dir", None)
            if isinstance(output_dir, str):
                obj.output_dir = broadcast_from_rank0(_expand_strftime_directives(output_dir))
