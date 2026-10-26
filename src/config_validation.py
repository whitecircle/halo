"""Validation every config class shares: the value guards, the range-validation base the guarded
script-argument and trainer-config classes inherit, the declared-default comparison, and the refusals of
parsed knobs a script cannot honor.

A NaN passes every ordered comparison, so a bare range check (``value <= 0``) admits it; the value guards
check finiteness first. A bool is an int subclass that a range check reads as 0 or 1, so every value
guard refuses it.
"""

import math
from dataclasses import MISSING, fields
from typing import Any


def present(**values: Any) -> dict[str, Any]:
    """``values`` without the unset (``None``) ones, for guarding an optional knob."""
    return {key: value for key, value in values.items() if value is not None}


def require_finite(owner: str, **values: Any) -> None:
    for key, value in values.items():
        if isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(value):
            raise ValueError(f"{owner}: {key} must be a finite number, got {value!r}")


def require_positive(owner: str, **values: Any) -> None:
    require_finite(owner, **values)
    for key, value in values.items():
        if value <= 0:
            raise ValueError(f"{owner}: {key} must be > 0, got {value!r}")


def require_int(owner: str, *, minimum: int | None = None, **values: Any) -> None:
    """Refuse anything but an int (at least ``minimum`` when given); a float would stand in for a count."""
    for key, value in values.items():
        if isinstance(value, bool) or not isinstance(value, int) or (minimum is not None and value < minimum):
            bound = "" if minimum is None else f" >= {minimum}"
            raise ValueError(f"{owner}: {key} must be an integer{bound}, got {value!r}")


def require_positive_int(owner: str, **values: Any) -> None:
    require_int(owner, minimum=1, **values)


class RangeValidatedConfig:
    """Base for configs whose ``__post_init__`` holds numeric/range guards.

    Subclasses put their guards in :meth:`_validate_ranges` and call it from ``__post_init__``, which
    sees every value, CLI overrides included: ``H4ArgumentParser`` builds each dataclass once from the
    merged YAML and CLI values.

    :meth:`_validate_ranges` is cooperative — an implementation opens with
    ``super()._validate_ranges()`` so a class mixing two guarded bases runs both instead of the MRO
    keeping only the first. This base terminates the chain, hence a no-op rather than a raise; a
    subclass that inherits the base and implements nothing is caught by
    ``tests/cpu/config/test_post_override_validation.py``, which compares each subclass's bound
    method against this one.
    """

    def _validate_ranges(self) -> None:
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
    """Shared raise for both rejection forms, so they report the same way."""
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
