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
"""

from __future__ import annotations

import re

import torch.nn as nn
from transformers import PreTrainedModel
from transformers.conversion_mapping import get_checkpoint_conversion_mapping
from transformers.core_model_loading import WeightConverter, WeightRenaming

from src.distributed.expert_parallel.expert_weights import (
    ep_layer_class_by_model_type,
    experts_container_attrs,
)
from src.models.loading.lazy_safetensors.conversion import Convert, Rename, translate_converter, translate_renaming

# A conversion source addressing one tensor per expert (``…experts.*.w1.weight``): the lazy loaders
# keep the per-expert on-disk form for ranged reads and fuse locally, so the walker skips these
# rather than rejecting the family's whole entry set.
_PER_EXPERT_SOURCE = re.compile(rf"\.(?:{'|'.join(re.escape(a) for a in experts_container_attrs())})\.\*\.")


def resolve_conversion_steps(model_type: str, model: nn.Module) -> tuple[Rename | Convert, ...] | None:
    """The ordered conversion steps for ``model_type``'s family on ``model``'s tree, or ``None``
    when the family declares none.

    Resolution goes through the EP layer class (``ep_layer_class_by_model_type``) so a text-only
    artifact of a composite family (Inkling's ``inkling_text``) still finds the composite entry its
    weights were written under. Each declared key is scoped by :func:`_conversion_scopes`; ``model``
    is the meta shell, whose config also supplies the head counts a ``PermuteForRope`` entry reads.

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
            if isinstance(entry, WeightRenaming):
                steps.append(translate_renaming(entry, scope))
            elif isinstance(entry, WeightConverter):
                if all(_PER_EXPERT_SOURCE.search(p) for p in entry.source_patterns):
                    unknown = {type(op).__name__ for op in entry.operations} - {"MergeModulelist", "Concatenate"}
                    if unknown:
                        raise ValueError(
                            f"per-expert conversion entry for {entry.source_patterns} carries "
                            f"{sorted(unknown)} — the ExpertFuser only reproduces plain "
                            "MergeModulelist/Concatenate merges, so skipping it would silently "
                            "drop a real conversion"
                        )
                    continue  # expert merge, handled by the ExpertFuser (see _PER_EXPERT_SOURCE)
                steps.append(translate_converter(entry, config=model.config, scope=scope))
            else:
                raise ValueError(f"Unsupported conversion entry type {type(entry).__name__} for {key!r}.")
    return tuple(steps) or None


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
