"""How a checkpoint's keys reach a model's live names, planned once on transformers' rename rules, from key names
alone, so it runs before any shard opens."""

from __future__ import annotations

import itertools
import re
from collections.abc import Iterable, Iterator, Mapping
from copy import deepcopy
from dataclasses import dataclass

import torch
from transformers.conversion_mapping import get_model_conversion_mapping
from transformers.core_model_loading import WeightConverter, WeightRenaming, dot_natural_key, rename_source_key

from halo.models.conversion_overrides import verify_conversion_overrides_intact
from halo.models.model_classes import transformers_model_class

# A conversion source addressing one tensor per module-list index (``…experts.*.w1.weight``).
_PER_INDEX_SOURCE = re.compile(r"\.\*\.")


@dataclass(frozen=True)
class ConversionGroup:
    """The checkpoint keys one converter claims for one first target, each with its matched pattern, in arrival order,
    and every name the converter produces."""

    converter: WeightConverter
    parts: tuple[tuple[str, str], ...]
    targets: tuple[str, ...]


@dataclass(frozen=True)
class CheckpointKeyPlan:
    """Where every checkpoint key lands: a key no converter claims under its rename target (``renamed_by_key``, natural
    order), the rest in converter ``groups``. ``live`` names what the model holds."""

    renamed_by_key: Mapping[str, str]
    groups: tuple[ConversionGroup, ...]
    live: Mapping[str, bool]
    base_model_prefix: str | None
    renamings: tuple[WeightRenaming, ...]

    @property
    def keys(self) -> frozenset[str]:
        """Every name the checkpoint supplies, live or not."""
        return frozenset(self.renamed_by_key.values()).union(*(group.targets for group in self.groups))

    @property
    def sources(self) -> frozenset[str]:
        """The checkpoint keys the live names read."""
        return frozenset(
            itertools.chain(
                self.live_renames().values(), (key for group in self.live_groups() for key, _ in group.parts)
            )
        )

    def live_renames(self) -> dict[str, str]:
        """Each live name a plain rename supplies → the first checkpoint key spelled to it."""
        first: dict[str, str] = {}
        for key, target in self.renamed_by_key.items():
            if target in self.live:
                first.setdefault(target, key)
        return first

    def live_groups(self) -> Iterator[ConversionGroup]:
        """The converter groups that produce at least one live name."""
        return (group for group in self.groups if any(target in self.live for target in group.targets))


def resolve_model_conversions(model: torch.nn.Module, *, add_legacy: bool) -> list:
    """transformers' conversion mapping for the family, empty without a ``config``. FSDP2's ``FSDP<Name>`` classes read
    as custom code there, so each module gets its transformers class back for the lookup."""
    if getattr(model, "config", None) is None:
        return []
    swapped = []
    try:
        for module in model.modules():
            original = transformers_model_class(module)
            if original is not None and type(module) is not original:
                swapped.append((module, type(module)))
                module.__class__ = original
        verify_conversion_overrides_intact(model)
        return get_model_conversion_mapping(model, add_legacy=add_legacy)
    finally:
        for module, wrapped_class in swapped:
            module.__class__ = wrapped_class


def is_expert_bank_merge(entry) -> bool:
    """Whether ``entry`` fuses one-tensor-per-expert keys into a bank: a converter whose every source carries a
    per-index wildcard. The expert code owns that layout at both ends, never a generic conversion replay."""
    return isinstance(entry, WeightConverter) and all(_PER_INDEX_SOURCE.search(p) for p in entry.source_patterns)


def build_key_plan(
    checkpoint_keys: Iterable[str],
    live_names: Iterable[str],
    conversions: Iterable[WeightRenaming | WeightConverter],
    base_model_prefix: str | None,
) -> CheckpointKeyPlan:
    """Plan ``checkpoint_keys`` onto ``live_names`` as ``from_pretrained`` does: renamings, at most one converter, then
    the base-model prefix. Keys walk in natural order, so a converter stacks experts in arrival order."""
    # A GroupWeightRename latches its guard and every transform its use: a recorded or snapshot mapping stays clean.
    conversions = deepcopy(list(conversions))
    # A mapping, not a set: ``rename_source_key`` probes its ``meta_state_dict`` with ``.get``.
    live = dict.fromkeys(live_names, True)
    renamings = tuple(c for c in conversions if isinstance(c, WeightRenaming))
    converters = [c for c in conversions if isinstance(c, WeightConverter)]
    by_source = {pattern: converter for converter in converters for pattern in converter.source_patterns}
    renamed_by_key: dict[str, str] = {}
    claims: dict[str, tuple[WeightConverter, list[tuple[str, str]]]] = {}
    for key in sorted(checkpoint_keys, key=dot_natural_key):
        target, pattern = rename_source_key(key, renamings, converters, base_model_prefix, live)
        if target not in live and key in live:
            target, pattern = rename_source_key(key, [], [], base_model_prefix, live)
        if pattern is None:
            renamed_by_key[key] = target
        else:
            claims.setdefault(target, (by_source[pattern], []))[1].append((key, pattern))
    groups = tuple(
        ConversionGroup(
            converter,
            tuple(parts),
            tuple(first.replace(converter.target_patterns[0], target, 1) for target in converter.target_patterns),
        )
        for first, (converter, parts) in claims.items()
    )
    return CheckpointKeyPlan(renamed_by_key, groups, live, base_model_prefix, renamings)
