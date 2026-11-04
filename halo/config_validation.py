"""Bounds and type guards every config class shares, the guarded-config base, and refusals of knobs a script cannot
honor.

A NaN passes every ordered comparison and a bool is an int, so the guards refuse both before any bound applies."""

import math
from dataclasses import MISSING, fields
from typing import Any


def present(**values: Any) -> dict[str, Any]:
    return {key: value for key, value in values.items() if value is not None}


def _check_bounds(owner: str, values: dict[str, Any], *, integer: bool, gt, ge, lt, le, why: str) -> None:
    """Raise on the first of ``values`` outside the bounds (or no int, with ``integer``); two bounds are a range."""
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
    return not isinstance(value, bool) and isinstance(value, int | float) and math.isfinite(value)


def require_finite(owner: str, *, gt=None, ge=None, lt=None, le=None, why: str = "", **values: Any) -> None:
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


def require_callable(owner: str, **values: Any) -> None:
    """Refuse anything but a callable, for a knob only Python can set: from YAML it arrives as a string."""
    for key, value in values.items():
        if not callable(value):
            raise ValueError(f"{owner}: {key} must be a callable passed from Python, got {value!r}")


def require_text(owner: str, **values: Any) -> None:
    """Refuse anything but a non-blank string."""
    for key, value in values.items():
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{owner}: {key} must be a non-blank string, got {value!r}")


class ValidatedConfig:
    """Base for guarded configs: ``__post_init__`` runs the cooperative :meth:`_validate_fields` chain, then the next
    ``__post_init__`` in the MRO. The parser builds each dataclass once from merged YAML and CLI values, so guards see
    CLI overrides. A class that normalizes a field first overrides ``__post_init__``, ending in ``super()``."""

    def __post_init__(self) -> None:
        self._validate_fields()
        next_hook = getattr(super(), "__post_init__", None)
        if next_hook is not None:
            next_hook()

    def _validate_fields(self) -> None:
        """No-op terminator for the cooperative chain."""


def differs_from_default(args: Any, name: str) -> bool:
    """Whether ``args.<name>`` differs from its declared default, the only sign of a request on a parsed dataclass.
    Raises on a field with no default, since reading it as unchanged would hide a request."""
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
    """Raise on truthy parsed knobs this script cannot honor: ignoring one would train other than the config states."""
    _reject_ignored_fields(context, sorted(name for name, value in unsupported.items() if value))


def reject_non_default_args(context: str, args, *field_names: str) -> None:
    """Raise on knobs off their declared default, for a truthy default :func:`reject_unsupported_args` cannot express
    (``num_eval_examples=50``). Every name must be a defaulted field of ``args``: a typo would disarm the guard."""
    _reject_ignored_fields(context, sorted(name for name in field_names if differs_from_default(args, name)))
