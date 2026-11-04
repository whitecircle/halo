"""Family corrections to transformers' checkpoint conversion mapping, registered once at import, so a load, a save's
revert and a merge tool all read the corrected mapping, whichever of them runs first and with no family patch applied.
A remote modeling file can replace them for the whole process; a native model reading them then is refused."""

from collections.abc import Callable, Iterable
from dataclasses import dataclass

from torch import nn
from transformers import PreTrainedModel
from transformers.conversion_mapping import get_checkpoint_conversion_mapping, register_checkpoint_conversion_mapping
from transformers.core_model_loading import WeightRenaming, WeightTransform

# The override keys a native model has read in this process, as a remote load can follow one and still strand it.
_natively_read_keys: set[str] = set()


@dataclass(frozen=True)
class ConversionOverride:
    """``rewrite`` takes transformers' entries for ``key`` (a ``model_type`` or class name) and returns the ones to
    register. ``upstream`` spells those entries as ``(sources, targets)``, empty where transformers ships none."""

    key: str
    upstream: tuple[tuple[tuple[str, ...], tuple[str, ...]], ...]
    rewrite: Callable[[list[WeightTransform]], list[WeightTransform]]


class _CompositeLevelDrop(WeightRenaming):
    """A text class reading its composite's checkpoint drops the ``language_model`` level, one way: its saves hold no
    such level, and inside the composite, scoped under that very level, this never matches. A ``PrefixChange`` would
    write the level twice on a composite save with no load record, which is every lazy load."""

    __slots__ = ()

    def __init__(self):
        super().__init__(source_patterns=r"^language_model\.(.+)$", target_patterns=r"\1")

    def reverse_transform(self) -> WeightRenaming:
        return WeightRenaming(source_patterns=r"^$", target_patterns="")


def _with_composite_level_dropped(entries: list[WeightTransform]) -> list[WeightTransform]:
    return [_CompositeLevelDrop(), *entries]


def _with_vision_norm_under_its_projector(entries: list[WeightTransform]) -> list[WeightTransform]:
    """The release stores the norm where its remote file keeps it; transformers' class declares it absent, so a native
    load would keep its identity init with no missing key."""
    return [*entries, WeightRenaming(r"^vision_projector\.vision_final_layernorm\.", r"vision_final_layernorm\.")]


CONVERSION_OVERRIDES: tuple[ConversionOverride, ...] = (
    ConversionOverride(
        "nemotron_h",
        (
            (("backbone.",), ("model.",)),
            (("mixer.experts.*.up_proj.weight",), ("mixer.experts.up_proj",)),
            (("mixer.experts.*.down_proj.weight",), ("mixer.experts.down_proj",)),
        ),
        _with_composite_level_dropped,
    ),
    ConversionOverride(
        "NemotronH_Omni_Reasoning_V3",
        (
            ((r"^mlp1\.0\.",), ("multi_modal_projector.layer_norm.",)),
            ((r"^mlp1\.1\.",), ("multi_modal_projector.linear_1.",)),
            ((r"^mlp1\.3\.",), ("multi_modal_projector.linear_2.",)),
            ((r"^sound_encoder\.encoder\.",), ("audio_tower.",)),
            ((r"^sound_projection\.norm\.",), ("embed_audio.layer_norm.",)),
            ((r"^sound_projection\.linear1\.",), ("embed_audio.linear_1.",)),
            ((r"^sound_projection\.linear2\.",), ("embed_audio.linear_2.",)),
        ),
        _with_vision_norm_under_its_projector,
    ),
)


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

_REGISTERED_SPELLINGS = {
    override.key: _spelled_entries(get_checkpoint_conversion_mapping(override.key))
    for override in CONVERSION_OVERRIDES
}


def verify_conversion_overrides_intact(model: nn.Module) -> None:
    """Raises when a remote modeling file replaced an override a native model of this process reads: transformers looks
    the entries up per process, so that model's weights would land under names it does not have."""
    _natively_read_keys.update(
        key
        for module in model.modules()
        if isinstance(module, PreTrainedModel) and not module.is_custom_code()
        for key in (type(module).__name__, module.config.model_type)
        if key in _REGISTERED_SPELLINGS
    )
    replaced = sorted(
        key
        for key in _natively_read_keys
        if _spelled_entries(get_checkpoint_conversion_mapping(key)) != _REGISTERED_SPELLINGS[key]
    )
    if replaced:
        raise RuntimeError(
            f"A remote modeling file replaced transformers' conversion mapping for {replaced} for this whole process, "
            f"and a model built from transformers' own classes reads those entries, so its weights would land under "
            f"names it does not have. Load this family one way per process: every model with trust_remote_code, or "
            f"none."
        )
