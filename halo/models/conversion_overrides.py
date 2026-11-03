"""Family corrections to transformers' checkpoint conversion mapping, registered once at import, so a load, a save's
revert and a merge tool all read the corrected mapping, whichever of them runs first and with no family patch applied."""

from collections.abc import Callable, Iterable
from dataclasses import dataclass

from transformers.conversion_mapping import get_checkpoint_conversion_mapping, register_checkpoint_conversion_mapping
from transformers.core_model_loading import WeightTransform


@dataclass(frozen=True)
class ConversionOverride:
    """``rewrite`` takes transformers' entries for ``key`` (a ``model_type`` or class name) and returns the ones to
    register. ``upstream`` spells those entries as ``(sources, targets)``, empty where transformers ships none."""

    key: str
    upstream: tuple[tuple[tuple[str, ...], tuple[str, ...]], ...]
    rewrite: Callable[[list[WeightTransform]], list[WeightTransform]]


CONVERSION_OVERRIDES: tuple[ConversionOverride, ...] = ()


def _spelled_entries(entries: list[WeightTransform] | None) -> tuple[tuple[tuple[str, ...], tuple[str, ...]], ...]:
    return tuple((tuple(entry.source_patterns), tuple(entry.target_patterns)) for entry in entries or ())


def register_conversion_overrides(overrides: Iterable[ConversionOverride]) -> None:
    """Raises when transformers' entries for a key are no longer what its rewrite was written against, since a rewrite
    of moved entries misplaces weights with no other symptom."""
    for override in overrides:
        entries = get_checkpoint_conversion_mapping(override.key)
        spelled = _spelled_entries(entries)
        if spelled != override.upstream:
            raise RuntimeError(
                f"transformers' conversion mapping for {override.key!r} is {spelled} in this release, but the "
                f"toolkit's override is written for {override.upstream}: re-derive it on this release's mapping."
            )
        register_checkpoint_conversion_mapping(override.key, override.rewrite(entries or []), overwrite=True)


register_conversion_overrides(CONVERSION_OVERRIDES)
