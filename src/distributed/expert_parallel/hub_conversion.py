"""Resolve transformers' checkpoint-conversion entries for a declared MoE family.

Some hub checkpoints keep the vendor's original tensor namespace and layouts, bridged only by
transformers' conversion mapping inside ``from_pretrained`` (``transformers/conversion_mapping.py``:
Inkling's ``model.llm.*`` / ``wq_du`` / interleaved ``w13_weight``, GLM-5 Next's split
``q/k/v_conv1d``, Step-3.7's per-layer ``moe.gate_proj + moe.up_proj``). The lazy loaders read
safetensors directly, so a family whose checkpoint needs conversion declares the mapping key(s) on
its EP layer class (:attr:`~EPMoELayerBase._HUB_CONVERSION_KEYS`), and
:func:`resolve_conversion_steps` turns those entries into the ordered steps the generic walker
consumes, scoped as transformers scopes them: a key naming a sub-model of the shell (Step-3.7's
``step3p5_vision`` tower) applies only under that sub-model's path.

The op vocabulary, the translator and the key walker are family-agnostic and live in
``src/models/loading/lazy_safetensors/``; this module holds the registry seam — which family
declares which conversion key, and where in the module tree each key lives.

The save side runs the other way: :func:`gathered_export_conversions` is what an EP export (the
gathered save, the RL weight sync) inverts so it writes the namespace the load read.
"""

from __future__ import annotations

import re

import torch.nn as nn
from transformers import PreTrainedModel
from transformers.conversion_mapping import get_checkpoint_conversion_mapping, get_model_conversion_mapping
from transformers.core_model_loading import WeightConverter, WeightRenaming
from transformers.integrations.bitsandbytes import Bnb4bitDeserialize

from src.checkpoint.format import is_sub_model_prefix_change, revert_conversions_for
from src.distributed.expert_parallel.expert_weights import (
    ep_layer_class_by_model_type,
    experts_container_attrs,
)
from src.models.loading.dtype import is_packed_4bit_parameter
from src.models.loading.lazy_safetensors.conversion import Convert, Rename, translate_converter, translate_renaming

# A conversion source addressing one tensor per expert (``…experts.*.w1.weight``): the lazy loaders
# keep the per-expert on-disk form for ranged reads and fuse locally, so the walker skips these
# rather than rejecting the family's whole entry set.
_PER_EXPERT_SOURCE = re.compile(rf"\.(?:{'|'.join(re.escape(a) for a in experts_container_attrs())})\.\*\.")


def is_per_expert_merge(entry) -> bool:
    """Whether ``entry`` fuses one-tensor-per-expert checkpoint keys into an expert bank.

    That layout is owned at both ends by the expert code, never by a generic conversion replay: the
    lazy loaders' ``ExpertFuser`` reads it, each family's ``gather_expert_state_dict`` writes it.
    """
    return isinstance(entry, WeightConverter) and all(_PER_EXPERT_SOURCE.search(p) for p in entry.source_patterns)


def gathered_export_conversions(model: nn.Module) -> list:
    """The conversions an EP export inverts: :func:`~src.checkpoint.format.revert_conversions_for`
    minus the per-expert merges.

    Every other rename and converter the load recorded is reverted, on the non-expert params and the
    gathered layers alike, so the export writes the namespace the load read — a SigLIP tower's
    ``vision_model`` level, a vendor namespace such as Inkling's ``model.llm.*`` — as
    ``save_pretrained`` does. The expert banks keep the layout their family's gather emits (fused or
    per expert), which ``from_pretrained`` and both engines read either way. Empty, and the export the
    identity, for every load that converted nothing but the experts.
    """
    return [c for c in revert_conversions_for(model) if not is_per_expert_merge(c)]


def reversed_export_transforms(conversions: list) -> tuple[list[WeightRenaming], list[WeightConverter]]:
    """``conversions`` (:func:`gathered_export_conversions`) reversed in transformers' save-side order
    and split for a key-by-key respell (``rename_source_key(..., reverse=True)``): a rename maps one
    key on its own, a key a reverse converter claims needs that converter's other sources beside it."""
    renamings: list[WeightRenaming] = []
    converters: list[WeightConverter] = []
    for transform in (c.reverse_transform() for c in conversions[::-1]):
        (renamings if isinstance(transform, WeightRenaming) else converters).append(transform)
    return renamings, converters


def resolve_conversion_steps(model_type: str, model: nn.Module) -> tuple[Rename | Convert, ...] | None:
    """The ordered conversion steps for ``model_type``'s family on ``model``'s tree, or ``None``
    when there are none.

    Resolution goes through the EP layer class (``ep_layer_class_by_model_type``) so a text-only
    artifact of a composite family (Inkling's ``inkling_text``) still finds the composite entry its
    weights were written under. Each declared key is scoped by :func:`_conversion_scopes`; ``model``
    is the meta shell, whose config also supplies the head counts a ``PermuteForRope`` entry reads.
    The tree's own sub-model ``PrefixChange`` entries follow
    (:func:`~src.checkpoint.format.is_sub_model_prefix_change`): a SigLIP tower's hub keys carry a
    ``vision_model`` level the module drops, which every save writes back.

    A declared key resolving to no entries raises: ``None`` means "this checkpoint is already
    canonical", so returning it for a family that declares ``_HUB_CONVERSION_KEYS`` would load the
    vendor-namespace tensors unconverted, which the key planner skips as unmatched, leaving the model
    at its meta-device init values.
    """
    layer_cls = ep_layer_class_by_model_type().get(model_type)
    if layer_cls is None:
        return None
    scopes = _conversion_scopes(model)
    steps: list[Rename | Convert] = []
    for key in layer_cls._HUB_CONVERSION_KEYS:
        entries = get_checkpoint_conversion_mapping(key) or ()
        if not entries:
            raise ValueError(
                f"{layer_cls.__name__} declares hub-conversion key {key!r}, but this transformers "
                f"build resolves no entries for it — its checkpoints are in a vendor namespace that "
                f"the lazy loader would then read unconverted. Drop the declaration if the family's "
                f"checkpoint became canonical, or pin a transformers that ships the mapping."
            )
        scope = scopes.get(key, "")
        for entry in entries:
            step = _translate_entry(entry, model, scope, key=key)
            if step is not None:
                steps.append(step)
    for entry in get_model_conversion_mapping(model, add_legacy=False):
        if is_sub_model_prefix_change(entry):
            steps.append(_translate_entry(entry, model, entry.scope_prefix, key=model_type))
    return tuple(steps) or None


def resolve_loaded_conversion_steps(model: nn.Module) -> tuple[Rename | Convert, ...] | None:
    """Replay the conversions the eager load actually used, including nested-model renames.

    Canonical training checkpoints need no conversion; vendor namespaces use the model's recorded
    transforms rather than the lazy family's narrower supported subset.
    """
    steps = []
    for entry in getattr(model, "_weight_conversions", None) or ():
        # HF records the broad original Bnb4bitDeserialize template (target="weight"), not
        # its concrete packed targets. It also matches ordinary norms/embeddings, where the
        # exact op returns its single input unchanged. Packed Params4bit are never masters;
        # replay their *ordinary* neighbours as identity and retain all family transforms.
        if (
            isinstance(entry, WeightConverter)
            and entry.target_patterns == ["weight"]
            and "weight" in entry.source_patterns
            and len(entry.operations) == 1
            and type(entry.operations[0]) is Bnb4bitDeserialize
            and any(is_packed_4bit_parameter(param) for param in model.parameters())
        ):
            continue
        step = _translate_entry(
            entry,
            model,
            getattr(entry, "scope_prefix", "") or "",
            key=getattr(getattr(model, "config", None), "model_type", type(model).__name__),
            strict_renaming=True,
        )
        if step is not None:
            steps.append(step)
    return tuple(steps) or None


def _translate_entry(
    entry, model: nn.Module, scope: str, *, key: str, strict_renaming: bool = False
) -> Rename | Convert | None:
    if isinstance(entry, WeightRenaming):
        if strict_renaming and (
            len(entry.source_patterns) != 1
            or len(entry.target_patterns) != 1
            or type(entry).rename_source_key is not WeightRenaming.rename_source_key
        ):
            raise ValueError(
                f"Unsupported renaming entry {type(entry).__name__} for {key!r}: checkpoint replay requires "
                "one unconditional source/target pair."
            )
        return translate_renaming(entry, scope)
    if not isinstance(entry, WeightConverter):
        raise ValueError(f"Unsupported conversion entry type {type(entry).__name__} for {key!r}.")
    if is_per_expert_merge(entry):
        unknown = {type(op).__name__ for op in entry.operations} - {"MergeModulelist", "Concatenate"}
        if unknown:
            raise ValueError(
                f"per-expert conversion entry for {key!r}, {entry.source_patterns} carries "
                f"{sorted(unknown)} — the ExpertFuser only reproduces plain "
                "MergeModulelist/Concatenate merges, so skipping it would silently "
                "drop a real conversion"
            )
        return None  # expert merge, handled by the ExpertFuser (see _PER_EXPERT_SOURCE)
    try:
        return translate_converter(entry, config=model.config, scope=scope)
    except ValueError as error:
        raise ValueError(f"{error} for {key!r}") from error


def _conversion_scopes(model: nn.Module) -> dict[str, str]:
    """Mapping key (class name or ``model_type``) → the module path its entries are scoped under.

    Mirrors transformers' ``get_model_conversion_mapping`` walk: every ``PreTrainedModel`` in the tree
    claims its class name and ``model_type``, the first (outermost) claimant wins, and a sub-model's
    entries see only the keys under its path relative to ``base_model_prefix`` while the root's see
    every key. A declared key naming no module (a text-only artifact loaded under the composite's
    key) is absent here and resolves to the root scope.
    """
    base_prefix = getattr(model, "base_model_prefix", "")
    scopes: dict[str, str] = {}
    for module_name, module in model.named_modules():
        if not isinstance(module, PreTrainedModel):
            continue
        scope = module_name.removeprefix(base_prefix).removeprefix(".")
        for key in (type(module).__name__, getattr(module.config, "model_type", None)):
            if key:
                scopes.setdefault(key, scope)
    return scopes
