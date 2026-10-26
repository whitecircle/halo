"""The hub's tensor names on save: the transformers load conversions a save reverts, and the view that reads a
saved layout back under the live module names.

transformers reverts its load conversions inside ``save_pretrained``; the gathered, TP, PP and EP writers bypass
that call, so they revert through the list resolved here.
"""

from __future__ import annotations

import itertools
import logging
from collections.abc import Callable, Iterable, Iterator
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass
from functools import partial

import torch
from transformers.conversion_mapping import get_model_conversion_mapping
from transformers.core_model_loading import (
    PrefixChange,
    WeightConverter,
    WeightRenaming,
    dot_natural_key,
    rename_source_key,
    revert_weight_conversion,
)

from src.model_loading.lazy_safetensors.conversion import is_expert_bank_merge, is_sub_model_prefix_change
from src.models.model_classes import transformers_model_class
from src.models.param_roles import persistent_buffers

logger = logging.getLogger(__name__)


def _has_child_at_prefix(model: torch.nn.Module, dotted_prefix: str) -> bool:
    """Whether ``model`` (under its ``base_model_prefix``) has a submodule at ``dotted_prefix``."""
    node = getattr(model, getattr(model, "base_model_prefix", ""), model)
    for part in dotted_prefix.split("."):
        node = getattr(node, part, None)
        if node is None:
            return False
    return True


def _family_conversions(model: torch.nn.Module, *, add_legacy: bool) -> list:
    """transformers' conversion mapping for ``model``'s family; empty without a ``config``, which keys the
    registry. FSDP2's ``FSDP<Name>`` classes read as custom code there, so each module gets its
    transformers class back for the lookup."""
    if getattr(model, "config", None) is None:
        return []
    swapped = []
    try:
        for module in model.modules():
            original = transformers_model_class(module)
            if original is not None and type(module) is not original:
                swapped.append((module, type(module)))
                module.__class__ = original
        return get_model_conversion_mapping(model, add_legacy=add_legacy)
    finally:
        for module, wrapped_class in swapped:
            module.__class__ = wrapped_class


def conversions_to_revert(model: torch.nn.Module) -> list:
    """The conversions a save-side revert, or a hub-namespace weight sync, inverts.

    What the load recorded, minus a ``PrefixChange`` the saved tree cannot carry; an empty record means
    the source was already in the module layout. Without a record (built from config, or lazily) the
    family's registry mapping minus its model-level ``PrefixChange``, as transformers does. A
    dequantizing load reverts the registry mapping instead, since a quantizer op's reverse respells keys
    into names no checkpoint has. A sub-model's own ``PrefixChange`` is its hub spelling and always stays.
    """
    conversions = getattr(model, "_weight_conversions", None)
    quantized = conversions is not None and any(
        hasattr(op, "hf_quantizer") for c in conversions for op in getattr(c, "operations", ())
    )
    if conversions is None or quantized:
        keep_prefix_change = quantized and any(isinstance(c, PrefixChange) for c in conversions)
        conversions = [
            c
            for c in _family_conversions(model, add_legacy=False)
            if keep_prefix_change or not isinstance(c, PrefixChange) or is_sub_model_prefix_change(c)
        ]
    # A text-only load of a multimodal checkpoint must not re-emit wrapper-prefixed keys engines cannot serve.
    return [
        c
        for c in conversions
        if not (
            isinstance(c, PrefixChange)
            and c.prefix_to_remove
            and not is_sub_model_prefix_change(c)
            and not _has_child_at_prefix(model, c.prefix_to_remove)
        )
    ]


@contextmanager
def save_pretrained_layout(model: torch.nn.Module) -> Iterator[None]:
    """Hold a ``save_pretrained`` of ``model`` to :func:`conversions_to_revert`'s list, the one the
    gathered writers revert through, by swapping it into the load's record for the block."""
    recorded = getattr(model, "_weight_conversions", None)
    model._weight_conversions = conversions_to_revert(model)
    try:
        yield
    finally:
        model._weight_conversions = recorded


def revert_conversions(model: torch.nn.Module, state_dict: dict, conversions: list) -> dict:
    """transformers' save-side revert of ``state_dict`` through ``conversions``, raising on failure."""
    if not conversions:
        return state_dict
    recorded = getattr(model, "_weight_conversions", None)
    model._weight_conversions = conversions
    try:
        return revert_weight_conversion(model, state_dict)
    finally:
        model._weight_conversions = recorded


def revert_to_load_layout(model: torch.nn.Module, state_dict: dict) -> dict:
    """Map a module-layout state dict back to the checkpoint layout the model was loaded from.

    Without it a wrapper-less MoE save emits fused expert keys that per-expert engine loaders refuse or
    drop. A failure warns instead of raising: the config may be on disk with the peers past their
    barrier, and the pre-revert dict is loadable after ``unfuse_moe_experts.py``.
    """
    try:
        return revert_conversions(model, state_dict, conversions_to_revert(model))
    except Exception as e:
        logger.warning(
            f"revert_weight_conversion failed ({e}); writing the module-layout state "
            f"dict instead — run scripts/after_training/unfuse_moe_experts.py before "
            f"serving this checkpoint on a per-expert engine."
        )
        return state_dict


def reversed_export_transforms(conversions: list) -> tuple[list[WeightRenaming], list[WeightConverter]]:
    """``conversions`` (:func:`export_conversions`)
    reversed in transformers' save-side order and split for a key-by-key respell
    (``rename_source_key(..., reverse=True)``): a rename maps one key on its own, a key a reverse converter
    claims needs that converter's other sources beside it."""
    renamings: list[WeightRenaming] = []
    converters: list[WeightConverter] = []
    for transform in (c.reverse_transform() for c in conversions[::-1]):
        (renamings if isinstance(transform, WeightRenaming) else converters).append(transform)
    return renamings, converters


@dataclass(frozen=True)
class CheckpointNamespace:
    """How a model's checkpoint keys reach its live names: the family's whole conversion mapping, as
    ``from_pretrained`` replays it, the base-model prefix a key may carry or lack, and the
    :func:`export_conversions` a save inverts.

    transformers resolves the mapping off a whole model's module tree, scoping each sub-model's entries
    by its path, so a module holding only part of a model (a pipeline stage, named stage-locally) takes
    this snapshot of the whole one and reads its checkpoint through :class:`ModuleLayoutView` as that
    model would.
    """

    conversions: tuple[WeightRenaming | WeightConverter, ...] = ()
    base_model_prefix: str | None = None
    export_conversions: tuple[WeightRenaming | WeightConverter, ...] = ()

    @classmethod
    def of(cls, model: torch.nn.Module) -> CheckpointNamespace:
        return cls(
            conversions=tuple(_family_conversions(model, add_legacy=True)),
            base_model_prefix=getattr(model, "base_model_prefix", None) or None,
            export_conversions=tuple(export_conversions(model)),
        )


class ModuleLayoutView:
    """A checkpoint's tensors under the live model's own names: the read side of :func:`revert_to_load_layout`.

    Reads the way ``from_pretrained`` does, through the family's whole conversion mapping (renames, then
    at most one converter per key; a key whose rename misses the live tree but which hits itself stays),
    so any layout a save wrote reads back. Planned from key names alone: :attr:`keys` (the names it
    supplies) and :attr:`sources` (the checkpoint keys the live names read) are known before any shard
    opens. ``live_names`` and ``namespace`` stand in for a module holding part of the model (a PP stage).
    """

    def __init__(
        self,
        model: torch.nn.Module,
        checkpoint_keys: Iterable[str],
        *,
        live_names: Iterable[str] | None = None,
        namespace: CheckpointNamespace | None = None,
    ):
        self._model = model
        if live_names is None:
            live_names = (name for name, _ in itertools.chain(model.named_parameters(), persistent_buffers(model)))
        # A mapping, not a set: ``rename_source_key`` probes its ``meta_state_dict`` with ``.get``.
        self._live = dict.fromkeys(live_names, True)
        if namespace is None:
            namespace = CheckpointNamespace.of(model)
        renamings = [c for c in namespace.conversions if isinstance(c, WeightRenaming)]
        converters = [c for c in namespace.conversions if isinstance(c, WeightConverter)]
        by_source = {pattern: converter for converter in converters for pattern in converter.source_patterns}
        prefix = namespace.base_model_prefix
        self._renamed: dict[str, str] = {}
        # First target → (the converter its group runs, its ``(checkpoint key, source pattern)`` in arrival order).
        self._converted: dict[str, tuple[WeightConverter, list[tuple[str, str]]]] = {}
        # Natural order, as from_pretrained collects: a converter stacks experts in arrival order.
        for key in sorted(checkpoint_keys, key=dot_natural_key):
            target, pattern = rename_source_key(key, renamings, converters, prefix, self._live)
            if target not in self._live and key in self._live:
                target, pattern = rename_source_key(key, [], [], prefix, self._live)
            if pattern is None:
                self._renamed.setdefault(target, key)
            else:
                self._converted.setdefault(target, (by_source[pattern], []))[1].append((key, pattern))
        self.keys = frozenset(self._renamed).union(
            *(self._targets(first, converter) for first, (converter, _) in self._converted.items())
        )
        self.sources = frozenset(
            itertools.chain(
                (key for target, key in self._renamed.items() if target in self._live),
                (key for _, _, parts in self._live_groups() for key, _ in parts),
            )
        )

    @staticmethod
    def _targets(first: str, converter: WeightConverter) -> tuple[str, ...]:
        """The live names a converter group produces, spelled off its first target as transformers does."""
        return tuple(first.replace(converter.target_patterns[0], target, 1) for target in converter.target_patterns)

    def _live_groups(self) -> Iterator[tuple[str, WeightConverter, list[tuple[str, str]]]]:
        """The converter groups that produce at least one live name."""
        for first, (converter, parts) in self._converted.items():
            if any(target in self._live for target in self._targets(first, converter)):
                yield first, converter, parts

    def items(self, read: Callable[[str], torch.Tensor]) -> Iterator[tuple[str, torch.Tensor]]:
        """``(live name, tensor)`` for every live tensor the checkpoint supplies, one converted group at a
        time; ``read`` serves one checkpoint tensor by key, called only when its group is reached."""
        for target, key in self._renamed.items():
            if target in self._live:
                yield target, read(key)
        config = getattr(self._model, "config", None)
        for first, converter, parts in self._live_groups():
            expected = self._targets(first, converter)
            group = deepcopy(converter)
            for key, pattern in parts:
                group.add_tensor(first, key, pattern, partial(read, key))
            for target, tensor in group.convert(first, model=self._model, config=config).items():
                if target not in expected:
                    raise RuntimeError(
                        f"The load conversion for {first!r} produced {target!r}, outside the names "
                        f"{list(expected)} the resume matched against the live model."
                    )
                if target in self._live:
                    yield target, tensor[0] if isinstance(tensor, list) else tensor


def export_conversions(model: torch.nn.Module) -> list:
    """The conversions an EP export inverts: :func:`conversions_to_revert` minus the expert-bank merges.

    Every other recorded rename and converter is reverted, on the non-expert params and the gathered layers
    alike, so the export writes the namespace the load read (a SigLIP tower's ``vision_model`` level, Inkling's
    ``model.llm.*``); the expert banks keep the layout their family's gather emits, which ``from_pretrained``
    and both engines read either way.
    """
    return [c for c in conversions_to_revert(model) if not is_expert_bank_merge(c)]
