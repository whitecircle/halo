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
from transformers.core_model_loading import PrefixChange, WeightConverter, WeightRenaming, revert_weight_conversion

from src.model_loading.checkpoint_keys import build_key_plan, is_expert_bank_merge, resolve_model_conversions
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


def _is_sub_model_prefix_change(conversion) -> bool:
    """Whether ``conversion`` strips the leading level of a sub-model's own keys: a ``PrefixChange`` scoped to
    the sub-model, such as the ``vision_model`` level of a SigLIP tower (Command A+, LFM2-VL), which the hub
    index and both engines' loaders keep and the module tree drops."""
    return (
        isinstance(conversion, PrefixChange)
        and bool(conversion.scope_prefix)
        and bool(conversion.prefix_to_remove)
        and not conversion.model_prefix
    )


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
            for c in resolve_model_conversions(model, add_legacy=False)
            if keep_prefix_change or not isinstance(c, PrefixChange) or _is_sub_model_prefix_change(c)
        ]
    # A text-only load of a multimodal checkpoint must not re-emit wrapper-prefixed keys engines cannot serve.
    return [
        c
        for c in conversions
        if not (
            isinstance(c, PrefixChange)
            and c.prefix_to_remove
            and not _is_sub_model_prefix_change(c)
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
            conversions=tuple(resolve_model_conversions(model, add_legacy=True)),
            base_model_prefix=getattr(model, "base_model_prefix", None) or None,
            export_conversions=tuple(export_conversions(model)),
        )


class ModuleLayoutView:
    """A checkpoint's tensors under the live model's own names (the read side of :func:`revert_to_load_layout`),
    planned by :func:`~src.model_loading.checkpoint_keys.build_key_plan` through the family's whole conversion
    mapping, so any layout a save wrote reads back. ``live_names`` and ``namespace`` stand in for a module
    holding part of the model (a PP stage).
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
        if namespace is None:
            namespace = CheckpointNamespace.of(model)
        self._plan = build_key_plan(checkpoint_keys, live_names, namespace.conversions, namespace.base_model_prefix)
        self.keys = self._plan.keys
        self.sources = self._plan.sources

    def items(self, read: Callable[[str], torch.Tensor]) -> Iterator[tuple[str, torch.Tensor]]:
        """``(live name, tensor)`` for every live tensor the checkpoint supplies, one converted group at a
        time; ``read`` serves one checkpoint tensor by key, called only when its group is reached."""
        for target, key in self._plan.live_renames().items():
            yield target, read(key)
        config = getattr(self._model, "config", None)
        for group in self._plan.live_groups():
            first = group.targets[0]
            converter = deepcopy(group.converter)
            for key, pattern in group.parts:
                converter.add_tensor(first, key, pattern, partial(read, key))
            for target, tensor in converter.convert(first, model=self._model, config=config).items():
                if target not in group.targets:
                    raise RuntimeError(
                        f"The load conversion for {first!r} produced {target!r}, outside the names "
                        f"{list(group.targets)} the resume matched against the live model."
                    )
                if target in self._plan.live:
                    yield target, tensor[0] if isinstance(tensor, list) else tensor


def export_conversions(model: torch.nn.Module) -> list:
    """The conversions an EP export inverts: :func:`conversions_to_revert` minus the expert-bank merges.

    Every other recorded rename and converter is reverted, on the non-expert params and the gathered layers
    alike, so the export writes the namespace the load read (a SigLIP tower's ``vision_model`` level, Inkling's
    ``model.llm.*``); the expert banks keep the layout their family's gather emits, which ``from_pretrained``
    and both engines read either way.
    """
    return [c for c in conversions_to_revert(model) if not is_expert_bank_merge(c)]
