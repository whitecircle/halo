"""Post-override validation base inherited by the guarded script-argument and trainer-config classes,
and the value guards their ranges (and the reward terms) are written with.

A NaN passes every ordered comparison, so a bare range check (``value <= 0``) admits it; these guards
check finiteness first.
"""

import math
from typing import Any


def require_finite(owner: str, **values: Any) -> None:
    for key, value in values.items():
        if isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(value):
            raise ValueError(f"{owner}: {key} must be a finite number, got {value!r}")


def require_positive(owner: str, **values: Any) -> None:
    require_finite(owner, **values)
    for key, value in values.items():
        if value <= 0:
            raise ValueError(f"{owner}: {key} must be > 0, got {value!r}")


def require_positive_int(owner: str, **values: Any) -> None:
    for key, value in values.items():
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{owner}: {key} must be an integer >= 1, got {value!r}")


class RangeValidatedConfig:
    """Base for configs whose ``__post_init__`` holds numeric/range guards.

    ``H4ArgumentParser`` applies ``--key=value`` overrides on top of a YAML with ``setattr``, so
    ``__post_init__`` never re-runs and the guards it holds are bypassed. Subclasses put their guards
    in :meth:`_validate_ranges` and call it from ``__post_init__``; this base re-runs it on the
    override path, so a CLI value is held to the same bounds as a YAML value.

    Guards are re-run whole rather than per overridden field: they are cheap, and a cross-field guard
    can be violated by an override of either side.

    :meth:`_validate_ranges` is cooperative — an implementation opens with
    ``super()._validate_ranges()`` so a class mixing two guarded bases runs both instead of the MRO
    keeping only the first. This base terminates the chain, hence a no-op rather than a raise; a
    subclass that inherits the base and implements nothing is caught by
    ``tests/cpu/config/test_post_override_validation.py``, which compares each subclass's bound
    method against this one.
    """

    def _validate_ranges(self) -> None:
        """No-op terminator for the cooperative chain."""

    def __post_override__(self, overridden_fields: set[str]) -> None:
        self._validate_ranges()
