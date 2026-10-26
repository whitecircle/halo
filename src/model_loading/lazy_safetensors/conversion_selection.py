"""Which transformers checkpoint conversions a lazy load applies.

A family whose hub checkpoint keeps a vendor namespace (Inkling's ``model.llm.*``, GLM-5 Next's split
``q/k/v_conv1d``, Step-3.7's per-layer ``moe.*``) declares its conversion-mapping key(s); a resume replays
what the eager load recorded instead. Each entry is scoped as transformers scopes it and translated into
the closed op vocabulary of :mod:`.conversion`.
"""

from __future__ import annotations

import torch.nn as nn
from transformers import PreTrainedModel
from transformers.conversion_mapping import get_checkpoint_conversion_mapping, get_model_conversion_mapping
from transformers.core_model_loading import WeightConverter, WeightRenaming
from transformers.integrations.bitsandbytes import Bnb4bitDeserialize

from src.model_loading.dtype import is_packed_4bit_parameter
from src.model_loading.lazy_safetensors.conversion import (
    ConvertStep,
    RenameStep,
    is_expert_bank_merge,
    is_sub_model_prefix_change,
    translate_converter,
    translate_renaming,
)
from src.models.moe.family_specs import find_moe_family_spec


def _conversion_scopes(model: nn.Module) -> dict[str, str]:
    """Mapping key (class name or ``model_type``) → the module path its entries are scoped under.

    transformers' ``get_model_conversion_mapping`` walk: every ``PreTrainedModel`` in the tree claims its class
    name and ``model_type``, the outermost claimant wins, and a sub-model's entries see only the keys under its
    path relative to ``base_model_prefix``. A key naming no module resolves to the root scope.
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


def translate_conversion_entry(
    entry, model: nn.Module, scope: str, *, key: str, strict_renaming: bool = False
) -> RenameStep | ConvertStep | None:
    """One transformers conversion entry as a step under ``scope``, ``None`` for an expert-bank merge (the
    fuser's); ``strict_renaming`` refuses a rename that is not one unconditional source/target pair."""
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
    if is_expert_bank_merge(entry):
        unknown = {type(op).__name__ for op in entry.operations} - {"MergeModulelist", "Concatenate"}
        if unknown:
            raise ValueError(
                f"per-expert conversion entry for {key!r}, {entry.source_patterns} carries "
                f"{sorted(unknown)} — the ExpertFuser only reproduces plain "
                "MergeModulelist/Concatenate merges, so skipping it would silently "
                "drop a real conversion"
            )
        return None
    try:
        return translate_converter(entry, config=model.config, scope=scope)
    except ValueError as error:
        raise ValueError(f"{error} for {key!r}") from error


def resolve_conversion_steps(model: nn.Module, model_type: str) -> tuple[RenameStep | ConvertStep, ...] | None:
    """The ordered conversion steps ``model_type``'s family spec declares (``MoEFamilySpec.hub_conversion_keys``),
    scoped on ``model``'s tree (the meta shell, whose config supplies a ``PermuteForRope`` entry's head counts),
    then the tree's own sub-model ``PrefixChange`` entries; ``None`` when there are none.

    A declared key resolving to no entries raises: ``None`` would load the vendor-namespace tensors
    unconverted, which the planner skips as unmatched, leaving the model at its meta-device init.
    """
    spec = find_moe_family_spec(model_type)
    if spec is None:
        return None
    scopes = _conversion_scopes(model)
    steps: list[RenameStep | ConvertStep] = []
    for key in spec.hub_conversion_keys:
        entries = get_checkpoint_conversion_mapping(key) or ()
        if not entries:
            raise ValueError(
                f"model_type {model_type!r} declares hub-conversion key {key!r}, but this transformers "
                f"build resolves no entries for it — its checkpoints are in a vendor namespace that "
                f"the lazy loader would then read unconverted. Drop the declaration if the family's "
                f"checkpoint became canonical, or pin a transformers that ships the mapping."
            )
        scope = scopes.get(key, "")
        for entry in entries:
            step = translate_conversion_entry(entry, model, scope, key=key)
            if step is not None:
                steps.append(step)
    for entry in get_model_conversion_mapping(model, add_legacy=False):
        if is_sub_model_prefix_change(entry):
            steps.append(translate_conversion_entry(entry, model, entry.scope_prefix, key=model_type))
    return tuple(steps) or None


def replay_loaded_conversions(model: nn.Module) -> tuple[RenameStep | ConvertStep, ...] | None:
    """The conversions the eager load recorded, nested-model renames included, as steps; ``None`` for a
    canonical checkpoint."""
    steps = []
    for entry in getattr(model, "_weight_conversions", None) or ():
        # transformers records the broad Bnb4bitDeserialize template (target "weight"), which also matches
        # ordinary norms and embeddings as an identity; packed Params4bit are never masters, so it is skipped.
        if (
            isinstance(entry, WeightConverter)
            and entry.target_patterns == ["weight"]
            and "weight" in entry.source_patterns
            and len(entry.operations) == 1
            and type(entry.operations[0]) is Bnb4bitDeserialize
            and any(is_packed_4bit_parameter(param) for param in model.parameters())
        ):
            continue
        step = translate_conversion_entry(
            entry,
            model,
            getattr(entry, "scope_prefix", "") or "",
            key=getattr(getattr(model, "config", None), "model_type", type(model).__name__),
            strict_renaming=True,
        )
        if step is not None:
            steps.append(step)
    return tuple(steps) or None
