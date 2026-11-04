"""The EP family roster: one import per family layer, the HF MoE block each one replaces, and the layer lookups."""

from halo.distributed.expert_parallel.families.bailing import EPBailingMoELayer  # noqa: F401
from halo.distributed.expert_parallel.families.cohere2_moe import EPCohere2MoELayer  # noqa: F401
from halo.distributed.expert_parallel.families.deepseek_v4 import EPDeepseekV4MoELayer  # noqa: F401
from halo.distributed.expert_parallel.families.gemma4 import EPGemma4MoELayer  # noqa: F401
from halo.distributed.expert_parallel.families.glm4 import EPGlm4MoELayer  # noqa: F401
from halo.distributed.expert_parallel.families.glm5_next import EPGlm5NextMoELayer  # noqa: F401
from halo.distributed.expert_parallel.families.gpt_oss import EPGptOssMoELayer  # noqa: F401
from halo.distributed.expert_parallel.families.inkling import EPInklingMoELayer  # noqa: F401
from halo.distributed.expert_parallel.families.laguna import EPLagunaMoELayer  # noqa: F401
from halo.distributed.expert_parallel.families.lfm2 import EPLfm2MoELayer  # noqa: F401
from halo.distributed.expert_parallel.families.mistral4 import EPMistral4MoELayer  # noqa: F401
from halo.distributed.expert_parallel.families.nemotron_h import EPNemotronHMoELayer  # noqa: F401
from halo.distributed.expert_parallel.families.qwen3 import EPQwen3MoELayer  # noqa: F401
from halo.distributed.expert_parallel.families.qwen3_5 import EPQwen3_5MoELayer  # noqa: F401
from halo.distributed.expert_parallel.families.qwen3_next import EPQwen3NextMoELayer  # noqa: F401
from halo.distributed.expert_parallel.families.qwen4_exp import EPQwen4ExpMoELayer  # noqa: F401
from halo.distributed.expert_parallel.families.step3p7 import EPStep3p7MoELayer  # noqa: F401
from halo.distributed.expert_parallel.families.zaya import EPZayaMoELayer  # noqa: F401
from halo.distributed.expert_parallel.moe_layer.layer import EPMoELayerBase
from halo.models.config_access import config_model_types
from halo.models.moe.family_specs import find_moe_family_spec
from halo.subclass_registry import build_class_claim_map, iter_subclasses

# Walks the subclass tree the imports above populate; raises on an HF class name two families claim.
MOE_LAYER_MAP: dict[str, type[EPMoELayerBase]] = build_class_claim_map(EPMoELayerBase, "HF MoE class name")


def ep_layer_classes() -> list[type[EPMoELayerBase]]:
    return iter_subclasses(EPMoELayerBase)


def ep_layer_classes_for_config(config) -> list[type[EPMoELayerBase]]:
    """The EP layer classes of the families ``config``'s ``model_type`` spellings claim, for gates that run
    without a live layer."""
    specs = {spec for model_type in config_model_types(config) if (spec := find_moe_family_spec(model_type))}
    return [cls for cls in ep_layer_classes() if vars(cls).get("FAMILY") in specs]
