"""Which modules produce ``router_logits``, read off transformers' ``_can_record_outputs`` registry and the forward the
instance runs, so every consumer sees exactly what an ``output_router_logits`` capture hooks."""

from __future__ import annotations

import inspect
from dataclasses import dataclass

import torch

from halo.models.config_access import TOOLKIT_STAMP_PREFIX, get_config_field, set_config_field_run_scoped
from halo.models.unwrapping import base_transformers_model

ROUTER_LOGITS_KEY = "router_logits"
# Stamped where router logits must stay off (EP layers that never populate them, aux-loss paths that would crash).
# No export carries this toolkit stamp, and a loader drops one a checkpoint carries.
ROUTER_LOGITS_FORCED_OFF_ATTR = f"{TOOLKIT_STAMP_PREFIX}router_logits_forced_off"


@dataclass(frozen=True)
class DeclaredRouter:
    """``folds_discard_slot`` marks a router whose indices map skipped tokens onto a real expert id, so only its
    logits, which keep the trailing discard column, express the routing."""

    name: str
    module: torch.nn.Module
    logits_index: int
    folds_discard_slot: bool


def declares_router_logits(module: torch.nn.Module) -> bool:
    return (getattr(module, "_can_record_outputs", None) or {}).get(ROUTER_LOGITS_KEY) is not None


def has_discard_expert_slot(model) -> bool:
    """Zaya. The slot must never receive tokens, so the balancing update and the load metrics leave it out."""
    return any(getattr(type(module), "_has_discard_expert_slot", False) for module in model.modules())


def _router_logits_recorders(model) -> list[tuple]:
    """De-duplicated, as a composite declares it on the sub-model owning the routers and repeats it on the wrapper."""
    specs: dict[tuple, None] = {}
    for module in model.modules():
        if not declares_router_logits(module):
            continue
        entry = module._can_record_outputs[ROUTER_LOGITS_KEY]
        for spec in entry if isinstance(entry, (list, tuple)) else (entry,):
            target_class = getattr(spec, "target_class", spec if isinstance(spec, type) else None)
            class_name = getattr(spec, "class_name", spec if isinstance(spec, str) else None)
            # A bare-class spec names no index; its installer's default for every key but hidden_states is 1.
            index = getattr(spec, "index", 1)
            specs[(target_class, class_name, getattr(spec, "layer_name", None), index)] = None
    return list(specs)


def declared_routers(model) -> list[DeclaredRouter]:
    """Matching mirrors transformers' installer (target class, else a class-name suffix, refined by ``layer_name``)."""
    specs = _router_logits_recorders(model)
    routers: list[DeclaredRouter] = []
    for name, module in model.named_modules():
        for target_class, class_name, layer_name, index in specs:
            if target_class is not None:
                if not isinstance(module, target_class):
                    continue
            elif class_name is None or not name.endswith(class_name):
                continue
            if layer_name is not None and f".{layer_name.strip('.')}." not in f"{name}.":
                continue
            routers.append(DeclaredRouter(name, module, index, has_discard_expert_slot(module)))
            break
    return routers


def mark_router_logits_forced_off(config) -> None:
    """Run-scoped, as an exported stamp would suppress ``moe/*`` metrics in every later run loading the artifact."""
    set_config_field_run_scoped(config, ROUTER_LOGITS_FORCED_OFF_ATTR, True)


def router_logits_forced_off(config) -> bool:
    return bool(get_config_field(config, ROUTER_LOGITS_FORCED_OFF_ATTR, False))


def honors_router_logits_flag(model) -> bool:
    """HF resolves a config-backed forward flag only on an explicit parameter, so a forward reading it from ``kwargs``
    never consults the config. Probed on the instance, where a Liger fused loss can rebind a parameter-less forward."""
    forward = getattr(base_transformers_model(model), "forward", None)
    if forward is None:
        return False
    try:
        return "output_router_logits" in inspect.signature(forward).parameters
    except (TypeError, ValueError):  # C-implemented or otherwise un-introspectable forward
        return False
