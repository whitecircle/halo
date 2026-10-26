"""Registry-order golden: what each toolkit registry holds, in the order it holds it.

A registry fills as its classes are defined and its modules imported, and whatever walks it does so in that
order: the claim maps fill in it, the Liger appliers run in it, a roster reads in it. A change that moves
modules or reorders imports can change that order without changing any class. Classes and functions are
recorded by qualified name, not by module, so a move that keeps every name and its order leaves the file
unchanged; the Liger specs and the MoE family specs are recorded by value. The check renders in a fresh
interpreter, so the order is the one the generator's imports and the code produce, never a caller's.

Only the toolkit's own classes are recorded: a test double subclassing a registry base joins the live
subclass tree of whatever process imported it.
"""

from __future__ import annotations

import dataclasses
import json
from collections.abc import Iterable, Mapping
from pathlib import Path

from transformers import AutoModelForSequenceClassification

import halo.environments.registry as environment_registry
import halo.models.moe.balancing as moe_balancing
import halo.models.seq_cls_heads as seq_cls_heads
from halo.distributed.context_parallel.families import ULYSSES_ATTENTION_MAP
from halo.distributed.expert_parallel.families.roster import MOE_LAYER_MAP, ep_layer_classes
from halo.distributed.expert_parallel.moe_layer.layer import EPMoELayerBase
from halo.distributed.pipeline_parallel.families import PP_SPEC_MAP
from halo.model_loading.liger.family_specs import LIGER_FAMILY_SPECS
from halo.models.head_transform import _HEAD_TRANSFORM_SPEC_MAP
from halo.models.moe.family_specs import MOE_FAMILY_SPECS, moe_model_types
from halo.optimizers.registry import NAMED_OPTIMIZER_BUILDERS
from halo.weight_sync.registry import resolve_weight_sync_client, rollout_backends
from tests.common.goldens import regen_main
from tests.common.rosters import ep_layer_class_by_model_type

GOLDEN_PATH = Path(__file__).with_suffix(".json")
_PACKAGE = EPMoELayerBase.__module__.partition(".")[0]


def _is_toolkit(cls: type) -> bool:
    return cls.__module__.partition(".")[0] == _PACKAGE


def _classes(classes: Iterable[type]) -> list[str]:
    return [cls.__qualname__ for cls in classes if _is_toolkit(cls)]


def _claims(claims: Mapping[str, type]) -> dict[str, str]:
    return {key: cls.__qualname__ for key, cls in claims.items() if _is_toolkit(cls)}


def registry_order() -> dict[str, object]:
    """Every registry's entries in its own order; a set's sorted."""
    heads = AutoModelForSequenceClassification._model_mapping._extra_content
    return {
        "ep layer classes": _classes(ep_layer_classes()),
        "ep moe block claims": _claims(MOE_LAYER_MAP),
        "ep model_type claims": _claims(ep_layer_class_by_model_type()),
        "ep-wrapped model types": sorted(moe_model_types()),
        "source config schema model types": sorted(
            mt for spec in MOE_FAMILY_SPECS if spec.exports_source_config_schema for mt in spec.model_types
        ),
        "native balancing slots": {
            mt: dataclasses.asdict(spec.balancing_slot)
            for spec in MOE_FAMILY_SPECS
            if spec.balancing_slot
            for mt in spec.model_types
        },
        "balancing key suffixes": sorted(moe_balancing._BALANCING_KEY_SUFFIXES),
        "cp attention claims": _claims(ULYSSES_ATTENTION_MAP),
        "pp backbone claims": list(PP_SPEC_MAP),
        "head transform claims": _claims(_HEAD_TRANSFORM_SPEC_MAP),
        "weight-sync clients": _claims(
            {backend: resolve_weight_sync_client(backend) for backend in rollout_backends()}
        ),
        "liger family specs": [dataclasses.asdict(spec) for spec in LIGER_FAMILY_SPECS],
        "sequence-classification heads": {
            config.__qualname__: head.__qualname__
            for config, head in heads.items()
            if head.__module__ == seq_cls_heads.__name__
        },
        "optimizer builders": {name: builder.__qualname__ for name, (builder, _) in NAMED_OPTIMIZER_BUILDERS.items()},
        "environments": list(environment_registry._environment_registry),
    }


def render() -> str:
    return json.dumps(registry_order(), indent=1) + "\n"


if __name__ == "__main__":
    regen_main(GOLDEN_PATH, render)
