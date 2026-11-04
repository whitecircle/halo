"""Which transformers checkpoint conversions a lazy load plans its keys with: the family's whole mapping, as
``from_pretrained`` applies it, or for an fp32-master reread the ones the eager load recorded.
"""

from __future__ import annotations

import torch.nn as nn
from transformers.conversion_mapping import get_checkpoint_conversion_mapping
from transformers.core_model_loading import PrefixChange, WeightConverter
from transformers.integrations.bitsandbytes import Bnb4bitDeserialize

from halo.model_loading.checkpoint_keys import build_key_plan, resolve_model_conversions
from halo.model_loading.dtype import is_packed_4bit_parameter
from halo.model_loading.lazy_safetensors.conversion import ConvertTargets, lazy_key_mapping
from halo.models.config_access import text_config
from halo.models.moe.family_specs import find_moe_family_spec


def _registry_conversions(model: nn.Module) -> list:
    """The whole conversion mapping, as ``from_pretrained`` applies it. Each ``hub_conversion_keys`` entry must
    resolve, or its vendor-namespace checkpoint reads unconverted, which the planner skips, leaving the model at its
    meta-device init."""
    model_type = getattr(text_config(model.config), "model_type", "")
    spec = find_moe_family_spec(model_type)
    declared = []
    for key in spec.hub_conversion_keys if spec else ():
        entries = get_checkpoint_conversion_mapping(key)
        if not entries:
            raise ValueError(
                f"model_type {model_type!r} declares hub-conversion key {key!r}, but this transformers "
                f"build resolves no entries for it — its checkpoints are in a vendor namespace that "
                f"the lazy loader would then read unconverted. Drop the declaration if the family's "
                f"checkpoint became canonical, or pin a transformers that ships the mapping."
            )
        declared += entries
    conversions = resolve_model_conversions(model, add_legacy=True)
    if declared and not resolve_model_conversions(model, add_legacy=False):
        # A text model reading its composite's checkpoint where transformers keys the mapping under the composite
        # alone (Inkling): the composite's entries, then its text level dropped as ``qwen3_5_text``'s mapping does.
        text_level = PrefixChange(prefix_to_remove="language_model", model_prefix=model.base_model_prefix)
        conversions = [*declared, text_level, *conversions]
    return conversions


def family_key_mapping(
    model: nn.Module, disk_keys: list[str], *, replay_loaded: bool = False
) -> tuple[dict[str, str], dict[str, ConvertTargets]]:
    """:func:`.conversion.lazy_key_mapping` planned through the family's whole mapping, or with ``replay_loaded`` the
    conversions an eager load recorded."""
    if replay_loaded:
        # transformers records the broad Bnb4bitDeserialize template (target "weight"), which also matches
        # ordinary norms and embeddings as an identity; packed Params4bit are never masters, so it is skipped.
        conversions = [
            entry
            for entry in getattr(model, "_weight_conversions", None) or ()
            if not (
                isinstance(entry, WeightConverter)
                and entry.target_patterns == ["weight"]
                and "weight" in entry.source_patterns
                and len(entry.operations) == 1
                and type(entry.operations[0]) is Bnb4bitDeserialize
                and any(is_packed_4bit_parameter(param) for param in model.parameters())
            )
        ]
    else:
        conversions = _registry_conversions(model)
    plan = build_key_plan(
        disk_keys, model.state_dict(), conversions, getattr(model, "base_model_prefix", None) or None
    )
    return lazy_key_mapping(plan, model.config)
