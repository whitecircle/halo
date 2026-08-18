"""The post-override validation seam every guarded script-argument and trainer-config class inherits.

Its own module so neither package reaches the base through the other's field bundles.
"""


class RangeValidatedConfig:
    """Base for configs whose ``__post_init__`` holds numeric/range guards.

    ``H4ArgumentParser.parse_yaml_and_args`` applies ``--key=value`` overrides with ``setattr``, so
    ``__post_init__`` never re-runs and every guard it holds is bypassed. Subclasses put their guards
    in :meth:`_validate_ranges` and call it from ``__post_init__``; this base re-runs it on the
    override path, so a CLI value is held to exactly the bounds a YAML value is.

    Guards are re-run whole rather than per overridden field: they are cheap, and a cross-field guard
    can be violated by an override of either side.

    :meth:`_validate_ranges` is COOPERATIVE — an implementation opens with
    ``super()._validate_ranges()`` so a class mixing two guarded bases runs both, instead of the MRO
    silently keeping only the first. This base is the chain's terminator, hence a no-op rather than a
    raise; the "inherits the base and implements nothing" case is caught statically by
    ``tests/cpu/config/test_post_override_validation.py``, which compares each subclass's bound
    method against this one.
    """

    def _validate_ranges(self) -> None:
        """No-op terminator for the cooperative chain."""

    def __post_override__(self, overridden_fields: set[str]) -> None:
        self._validate_ranges()
