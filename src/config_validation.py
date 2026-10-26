"""Validation every config class shares: the bounds and type guards, the validation base guarded configs inherit, the
declared-default comparison, and the refusals of parsed knobs a script cannot honor.

A NaN passes every ordered comparison and a bool is an int a range check reads as 0 or 1, so the guards refuse
both before any bound applies."""

import math
from dataclasses import MISSING, fields
from typing import Any


def present(**values: Any) -> dict[str, Any]:
    """``values`` without the unset (``None``) ones, for guarding an optional knob."""
    return {key: value for key, value in values.items() if value is not None}


def _check_bounds(owner: str, values: dict[str, Any], *, integer: bool, gt, ge, lt, le, why: str) -> None:
    """Raise on the first of ``values`` the bounds exclude (or, ``integer``, that is no int) as
    ``"<owner>: <key> must be [an integer] <bounds><why>, got <value>"``; two bounds read as an interval."""
    if (gt is not None or ge is not None) and (lt is not None or le is not None):
        opening, low = ("(", gt) if gt is not None else ("[", ge)
        closing, high = (")", lt) if lt is not None else ("]", le)
        bounds = f"in {opening}{low}, {high}{closing}"
    else:
        bounds = next((f"{op} {b}" for op, b in ((">", gt), (">=", ge), ("<", lt), ("<=", le)) if b is not None), "")
    requirement = " ".join(part for part in ("an integer" if integer else "", bounds) if part)
    for key, value in values.items():
        if (
            (integer and (isinstance(value, bool) or not isinstance(value, int)))
            or (gt is not None and value <= gt)
            or (ge is not None and value < ge)
            or (lt is not None and value >= lt)
            or (le is not None and value > le)
        ):
            raise ValueError(f"{owner}: {key} must be {requirement}{why}, got {value!r}")


def is_finite_number(value: Any) -> bool:
    """Whether ``value`` is an int or float other than a bool, NaN or an infinity."""
    return not isinstance(value, bool) and isinstance(value, int | float) and math.isfinite(value)


def require_finite(owner: str, *, gt=None, ge=None, lt=None, le=None, why: str = "", **values: Any) -> None:
    """Refuse a value that is not a finite number, then one outside the given bounds."""
    for key, value in values.items():
        if not is_finite_number(value):
            raise ValueError(f"{owner}: {key} must be a finite number, got {value!r}")
    _check_bounds(owner, values, integer=False, gt=gt, ge=ge, lt=lt, le=le, why=why)


def require_int(owner: str, *, gt=None, ge=None, lt=None, le=None, why: str = "", **values: Any) -> None:
    """Refuse anything but an int inside the given bounds; a float would stand in for a count."""
    _check_bounds(owner, values, integer=True, gt=gt, ge=ge, lt=lt, le=le, why=why)


def require_bool(owner: str, **values: Any) -> None:
    """Refuse anything but a bool: YAML 1.2 reads ``no``/``off`` as strings, which a truthiness check takes as true."""
    for key, value in values.items():
        if not isinstance(value, bool):
            raise ValueError(f"{owner}: {key} must be true or false, got {value!r}")


def require_text(owner: str, **values: Any) -> None:
    """Refuse anything but a non-blank string."""
    for key, value in values.items():
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{owner}: {key} must be a non-blank string, got {value!r}")


class ValidatedConfig:
    """Base for configs whose fields carry guards: ``__post_init__`` runs the cooperative :meth:`_validate_fields`
    chain (each override opens with ``super()._validate_fields()``), then the next ``__post_init__`` in the MRO
    (``TrainingArguments``'). The parser builds each dataclass once from the merged YAML and CLI values, so the
    guards see CLI overrides too. A class that normalizes a field first overrides ``__post_init__`` and ends it
    with ``super().__post_init__()``."""

    def __post_init__(self) -> None:
        self._validate_fields()
        next_hook = getattr(super(), "__post_init__", None)
        if next_hook is not None:
            next_hook()

    def _validate_fields(self) -> None:
        """No-op terminator for the cooperative chain."""


def differs_from_default(args: Any, name: str) -> bool:
    """Whether ``args.<name>`` differs from its declared default, a ``default_factory``'s value included.

    A parsed dataclass does not record which fields its config set, so a changed value is the only sign
    of a request. A field without a default, or a name ``args`` does not declare, raises: there is no
    default to compare it against, and reading it as unchanged would hide a request.
    """
    declared = {candidate.name: candidate for candidate in fields(type(args))}.get(name)
    if declared is None or (declared.default is MISSING and declared.default_factory is MISSING):
        raise ValueError(f"{name!r} is not a defaulted field of {type(args).__name__}: no default to compare against")
    default = declared.default_factory() if declared.default is MISSING else declared.default
    return getattr(args, name) != default


def _reject_ignored_fields(context: str, set_fields: list[str]) -> None:
    if set_fields:
        raise ValueError(
            f"{context} does not support these config fields (they would be silently ignored): "
            f"{set_fields}. Remove them from the YAML."
        )


def reject_unsupported_args(context: str, **unsupported) -> None:
    """Raise on parsed knobs this script cannot honor.

    Shared config dataclasses expose fields no single script implements; ignoring one would train
    something other than what the config states. Only truthy values are reported, since an unset
    field is not a request.
    """
    _reject_ignored_fields(context, sorted(name for name, value in unsupported.items() if value))


def reject_non_default_args(context: str, args, *field_names: str) -> None:
    """Raise on parsed knobs this script cannot honor, judged against their declared default.

    :func:`reject_unsupported_args` reads a request off truthiness, which cannot express a knob whose
    own default is truthy (``num_eval_examples=50`` would reject every run). Every name must be a
    defaulted field of ``args``: a typo would leave the guard unable to fire, so it raises.
    """
    _reject_ignored_fields(context, sorted(name for name in field_names if differs_from_default(args, name)))
