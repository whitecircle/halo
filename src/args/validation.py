"""Range-validation base inherited by the guarded script-argument and trainer-config classes, and the
value guards their ranges (and the reward terms) are written with.

A NaN passes every ordered comparison, so a bare range check (``value <= 0``) admits it; these guards
check finiteness first. A bool is an int subclass that a range check reads as 0 or 1, so every guard
refuses it.
"""

import math
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
