"""The hub's tensor names on save: the transformers load conversions a save reverts, and the view that reads a
saved layout back under the live module names.

transformers loads several families through a conversion mapping (renames, expert-bank merges, sub-model prefix
strips) and reverts it inside ``save_pretrained``; the gathered, TP, PP and EP writers bypass that call, so they
revert through the list resolved here. Which per-expert merges an EP export leaves to its family's gather is
:mod:`src.distributed.expert_parallel.hub_conversion`'s.
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

from src.models.model_classes import transformers_model_class
from src.models.param_roles import persistent_buffers

logger = logging.getLogger(__name__)


def _has_child_at_prefix(model: torch.nn.Module, dotted_prefix: str) -> bool:
    """Whether ``model`` (under its ``base_model_prefix``, when present) has a submodule at
    ``dotted_prefix`` — the oracle for whether a ``PrefixChange`` revert re-adds a real prefix."""
    node = getattr(model, getattr(model, "base_model_prefix", ""), model)
    for part in dotted_prefix.split("."):
        node = getattr(node, part, None)
        if node is None:
            return False
    return True


@contextmanager
def _transformers_classes(model: torch.nn.Module) -> Iterator[None]:
    """Give every module of ``model`` back the transformers class FSDP2 swapped out, for the duration.

    ``fully_shard`` replaces a sharded module's class in place with an ``FSDP<Name>`` subclass defined
    in torch, which transformers' conversion registry neither finds by name nor accepts: it reads a
    class defined outside transformers as custom code, so a wrapped model resolves no conversions.
    """
    swapped = []
    try:
        for module in model.modules():
            original = transformers_model_class(module)
            if original is not None and type(module) is not original:
                swapped.append((module, type(module)))
                module.__class__ = original
        yield
    finally:
        for module, wrapped_class in swapped:
            module.__class__ = wrapped_class


def _family_conversions(model: torch.nn.Module, *, add_legacy: bool) -> list:
    """transformers' conversion mapping for ``model``'s family, resolved against its transformers
    classes (an FSDP2-wrapped model included). Empty for a module carrying no ``config``: the registry
    is keyed by the config's family."""
    if getattr(model, "config", None) is None:
        return []
    with _transformers_classes(model):
        return get_model_conversion_mapping(model, add_legacy=add_legacy)


def is_sub_model_prefix_change(conversion) -> bool:
    """Whether ``conversion`` strips the leading level of a sub-model's own keys: a ``PrefixChange``
    scoped to the sub-model, such as the ``vision_model`` level of a SigLIP tower (Command A+,
    LFM2-VL), which the hub index and both engines' loaders keep and the module tree drops."""
    return (
        isinstance(conversion, PrefixChange)
        and bool(conversion.scope_prefix)
        and bool(conversion.prefix_to_remove)
        and not conversion.model_prefix
    )


def _registry_conversions(model: torch.nn.Module, *, keep_prefix_change: bool) -> list:
    """The family's declared conversion mapping — what a model no load recorded conversions for reverts
    through. A sub-model's own ``PrefixChange`` (:func:`is_sub_model_prefix_change`) stays either way:
    it is that sub-model's hub spelling, whatever checkpoint the model was built from."""
    pristine = _family_conversions(model, add_legacy=False)
    return [
        c for c in pristine if keep_prefix_change or not isinstance(c, PrefixChange) or is_sub_model_prefix_change(c)
    ]


def _saved_tree_conversions(model: torch.nn.Module, conversions: list) -> list:
    """``conversions`` minus each model-level ``PrefixChange`` whose stripped prefix is not a child of
    ``model``.

    A text-only load of a multimodal checkpoint consumes
    ``PrefixChange(prefix_to_remove="language_model")``, and reverting that at save would re-emit
    wrapper-prefixed keys under a text-only config — transformers re-strips them on reload, engine
    loaders keyed on the architectures and ``reattach_vision_tower.py`` do not, and the artifact is
    serving-dead. A sub-model's own ``PrefixChange`` (:func:`is_sub_model_prefix_change`) always
    reverts: the saved tree still holds that sub-model.
    """
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


def _carries_quantizer_ops(conversion) -> bool:
    """Whether ``conversion`` holds an op a quantizer contributed: transformers' quantizer integrations
    build each one around the quantizer that made it (``hf_quantizer``)."""
    return any(hasattr(op, "hf_quantizer") for op in getattr(conversion, "operations", ()))


def conversions_to_revert(model: torch.nn.Module) -> list:
    """The conversions a save-side revert — or a hub-namespace weight sync — inverts.

    What the load recorded, minus a ``PrefixChange`` the saved tree cannot carry
    (:func:`_saved_tree_conversions`). An empty record is a load that converted nothing, its source
    already in the module layout (the fused Qwen3.5/3.6 hub, a fused EP save), so the save writes what
    the load read. A model with no record at all (built from its config, or by a lazy loader) takes
    the family's registry mapping minus its model-level ``PrefixChange``, as transformers' own revert
    does. One resolution, so the gathered save, ``save_pretrained``, the EP export and the sync cannot
    drift.

    A dequantizing load (fp8, MXFP4) records the quantizer's ops, whose reverse is a pass-through
    with no target split that respells keys into names no checkpoint has; it reverts the registry
    mapping instead, keeping a ``PrefixChange`` only where the load applied one.
    """
    load_conversions = getattr(model, "_weight_conversions", None)
    if load_conversions is None:
        return _registry_conversions(model, keep_prefix_change=False)
    if any(_carries_quantizer_ops(c) for c in load_conversions):
        prefix_applied = any(isinstance(c, PrefixChange) for c in load_conversions)
        load_conversions = _registry_conversions(model, keep_prefix_change=prefix_applied)
    return _saved_tree_conversions(model, load_conversions)


@contextmanager
def save_pretrained_layout(model: torch.nn.Module) -> Iterator[None]:
    """Hold a ``save_pretrained`` of ``model`` to the layout its own tree and config declare.

    ``save_pretrained`` reverts through the load's recorded conversions verbatim, so inside this
    block they are :func:`conversions_to_revert`'s — minus the ``PrefixChange`` the saved tree cannot
    carry — the list the gathered writers revert through, so both write one layout. The record is
    restored on exit.
    """
    recorded = getattr(model, "_weight_conversions", None)
    model._weight_conversions = conversions_to_revert(model)
    try:
        yield
    finally:
        model._weight_conversions = recorded


def revert_conversions(model: torch.nn.Module, state_dict: dict, conversions: list) -> dict:
    """transformers' save-side revert of ``state_dict`` through ``conversions``, raising on failure.

    ``revert_weight_conversion`` reads the list off the model, so it is swapped in for the call and
    the model's own record restored after it.
    """
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

    transformers loads several MoE families from a per-expert hub into a module-FUSED expert layout
    and reverts it inside ``save_pretrained``, which the gathered/TP writers bypass — so without this
    a wrapper-less MoE save emits fused keys that per-expert engine loaders hard-fail on (vLLM 0.26.0:
    GLM-4/LFM-2) or silently drop (Laguna). Identity where the load converted nothing: dense models,
    and a source already in the module layout.

    Reverts :func:`conversions_to_revert`'s list, restoring the model's own value afterwards so
    later saves see exactly what the load left behind. A failure warns instead of raising: the
    config may already be on disk with the peer ranks past their barrier, and the pre-revert dict is
    a loadable checkpoint that only needs ``unfuse_moe_experts.py`` before a per-expert engine.
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
    """``conversions`` (:func:`~src.distributed.expert_parallel.hub_conversion.gathered_export_conversions`)
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
    ``from_pretrained`` replays it, and the base-model prefix a key may carry or lack.

    transformers resolves the mapping off a whole model's module tree, scoping each sub-model's entries
    by its path, so a module holding only part of a model (a pipeline stage, named stage-locally) takes
    this snapshot of the whole one and reads its checkpoint through :class:`ModuleLayoutView` as that
    model would.
    """

    conversions: tuple[WeightRenaming | WeightConverter, ...] = ()
    base_model_prefix: str | None = None

    @classmethod
    def of(cls, model: torch.nn.Module) -> CheckpointNamespace:
        return cls(
            conversions=tuple(_family_conversions(model, add_legacy=True)),
            base_model_prefix=getattr(model, "base_model_prefix", None) or None,
        )


class ModuleLayoutView:
    """A checkpoint's tensors under the live model's own names: the read side of
    :func:`revert_to_load_layout`.

    A wrapper-less save writes the layout the model was loaded from (a per-expert hub's experts one
    tensor each) while the live module holds them fused, so a reader that matches raw checkpoint keys
    to live names keeps the experts it already had. This reads the checkpoint the way
    ``from_pretrained`` does, through the family's whole conversion mapping: renames, then at most one
    converter per key, and a key whose rename misses the live tree while the key itself hits is kept
    as is. So it reads whichever layout a save wrote — the hub's, the module's (an EP-gathered save, a
    load that converted nothing), a per-expert spelling the load itself never met.

    Planned from the checkpoint's key names alone, so :attr:`keys` (the names it supplies) and
    :attr:`sources` (the checkpoint keys the live names among them read) are known before any shard
    is opened.
    ``live_names`` and ``namespace`` stand in for the model's own when it holds only part of the model
    the checkpoint belongs to (a pipeline stage, named in the unsplit model's names).
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
        """``(live name, tensor)`` for every live tensor the checkpoint supplies, one group at a time.

        ``read`` serves one checkpoint tensor by key, called only when the walk reaches the group that
        needs it, so a streaming reader holds one converted group (a layer's experts) at a time.
        """
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
