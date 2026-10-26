"""Every MoE family's facts, declared once as a frozen :class:`MoEFamilySpec`.

The EP layer of a family points at its spec (``FAMILY``); offline tools, config writers, weight sync and loaders
read the same spec by ``config.model_type`` without importing the EP package.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from functools import lru_cache
from types import MappingProxyType

from src.models.config_access import config_sources
from src.models.moe.config_fields import has_routed_experts
from src.models.moe.expert_layout import ExpertLayout, ExpertLayoutKind
from src.models.moe.expert_lora_spec import EXPERT_LORA_SUFFIXES

GLU_PROJECTIONS: tuple[str, ...] = ("gate_up_proj", "gate_proj", "up_proj", "down_proj")


@dataclass(frozen=True)
class NativeBalancingSlot:
    """A family's checkpoint-exported balancing slot.

    ``attrs`` are the dotted paths the slot answers to: the live module's and, where they differ, the hub's.
    ``config_flag`` is the field an engine reads to decide whether to apply the tensor at all (LFM-2's
    ``use_expert_bias``), ``None`` where the slot is unconditional.
    """

    attrs: tuple[str, ...]
    config_flag: str | None = None


@dataclass(frozen=True, eq=False)
class MoEFamilySpec:
    """One MoE family's facts. Identity-compared: each family has exactly one spec constant."""

    # Every ``config.model_type`` its checkpoints ship under: the family's own first, then the composite wrappers
    # and sibling spellings that carry it.
    model_types: tuple[str, ...]
    balancing_slot: NativeBalancingSlot | None = None
    # ``{flat key: (layer_type, per-layer field)}`` the pinned vLLM reads, where transformers re-serializes only
    # the folded ``per_layer_config``.
    served_flat_config_keys: Mapping[str, tuple[str, str]] = MappingProxyType({})
    # No pinned engine reads transformers' serialization of this config, so exports carry the source repo's
    # ``config.json`` and ``auto_map`` modules.
    exports_source_config_schema: bool = False
    # ``(live module, hub)`` spellings transformers renames only inside ``from_pretrained``.
    export_key_renames: tuple[tuple[str, str], ...] = ()
    expert_layout: ExpertLayout = ExpertLayout()
    # Every expert-weight attribute the EP layer may hold across its layouts (fused, separate, ETP).
    expert_weight_roots: tuple[str, ...] = GLU_PROJECTIONS
    # Attribute spellings the HF block may hold its expert container and shared expert under, tried in order.
    experts_container_attrs: tuple[str, ...] = ("experts",)
    shared_expert_attrs: tuple[str, ...] = ()
    # Reduction axis of the fused expert tensors as the family writes them: -1 = ``F.linear`` ``[E, N, K]``.
    fused_contraction_axis: int = -1
    # transformers' conversion-mapping keys for a vendor-namespace hub checkpoint.
    hub_conversion_keys: tuple[str, ...] = ()
    supports_lazy_loading: bool = True
    supports_fp32_non_ep_params: bool = True
    supports_routing_replay: bool = True
    supports_gradient_checkpointing: bool = True
    # Why no online weight sync into the family's served model is validated; ``None`` where it is.
    weight_sync_refusal: str | None = None

    def to_hub_key(self, key: str) -> str:
        """``key`` with the first ``export_key_renames`` pair it contains applied once."""
        for module_spelling, hub_spelling in self.export_key_renames:
            if module_spelling in key:
                return key.replace(module_spelling, hub_spelling, 1)
        return key

    def hub_to_module_renames(self) -> tuple[tuple[str, str], ...]:
        """``(hub, live module)`` pairs: the load-side inverse of :meth:`to_hub_key`."""
        return tuple((hub, module) for module, hub in self.export_key_renames)


_E_SCORE_CORRECTION_BIAS = NativeBalancingSlot(("gate.e_score_correction_bias",))
_GATE_UP_DOWN_PROJ = ("gate_proj", "up_proj", "down_proj")

# Ling-mini-2.0, Ring-mini-linear-2.0 and Ling-3.0 share one MoE block.
BAILING = MoEFamilySpec(
    model_types=("bailing_moe", "bailing_moe_linear", "bailing_hybrid"),
    expert_layout=ExpertLayout(ExpertLayoutKind.SEPARATE_GLU, _GATE_UP_DOWN_PROJ),
    balancing_slot=NativeBalancingSlot(("gate.expert_bias",)),
    shared_expert_attrs=("shared_experts",),
)
COHERE2_MOE = MoEFamilySpec(
    model_types=("cohere2_moe", "cohere2_vision"),
    expert_layout=ExpertLayout(per_expert_keys=_GATE_UP_DOWN_PROJ),
    shared_expert_attrs=("shared_experts",),
    # No lazy load of the Command A+ checkpoint has been checked against from_pretrained.
    supports_lazy_loading=False,
    weight_sync_refusal=(
        "no end-to-end weight sync has been validated for the Cohere2 MoE family on either pinned "
        "rollout engine (agent-docs/models/cohere2-moe.md)"
    ),
)
DEEPSEEK_V4 = MoEFamilySpec(
    model_types=("deepseek_v4",),
    expert_layout=ExpertLayout(per_expert_keys=("w1", "w3", "w2")),
    balancing_slot=_E_SCORE_CORRECTION_BIAS,
    shared_expert_attrs=("shared_experts",),
    hub_conversion_keys=("deepseek_v4",),
)
GEMMA4 = MoEFamilySpec(
    model_types=("gemma4", "gemma4_text"),
    expert_layout=ExpertLayout(key_prefix=""),
    # The full-attention layers' geometry, which older vLLM servers read off these flat keys; the pinned
    # server's transformers folds them back into ``per_layer_config`` at parse.
    served_flat_config_keys=MappingProxyType(
        {
            "global_head_dim": ("full_attention", "head_dim"),
            "num_global_key_value_heads": ("full_attention", "num_key_value_heads"),
        }
    ),
    # Documented to fail at the first dispatch (agent-docs/models/gemma4.md); refused at load instead.
    supports_fp32_non_ep_params=False,
    # The sibling router passes only pre-weighted (indices, weights), with nothing to re-derive.
    supports_routing_replay=False,
)
# The remote-code revisions spell the expert container ``routed_experts`` and the shared expert in the singular.
GLM4 = MoEFamilySpec(
    model_types=("glm4_moe_lite",),
    expert_layout=ExpertLayout(ExpertLayoutKind.FUSED_SPLIT_PER_EXPERT, _GATE_UP_DOWN_PROJ),
    balancing_slot=_E_SCORE_CORRECTION_BIAS,
    experts_container_attrs=("experts", "routed_experts"),
    shared_expert_attrs=("shared_experts", "shared_expert"),
)
GLM5_NEXT = MoEFamilySpec(
    model_types=("glm5_next", "glm5_next_text"),
    expert_layout=ExpertLayout(per_expert_keys=_GATE_UP_DOWN_PROJ),
    balancing_slot=_E_SCORE_CORRECTION_BIAS,
    shared_expert_attrs=("shared_experts",),
    hub_conversion_keys=("glm5_next",),
    weight_sync_refusal=(
        "both pinned rollout engines register glm5_next, but their expert loaders take only per-expert keys "
        "where the gather emits the fused pair, and no online update into either has been validated"
    ),
)
GPT_OSS = MoEFamilySpec(
    model_types=("gpt_oss",),
    expert_layout=ExpertLayout(ExpertLayoutKind.INTERLEAVED_GLU),
    # The router's own logit-space bias, which vLLM and SGLang both load and route with.
    balancing_slot=NativeBalancingSlot(("router.bias",)),
    # Its biases, then the de-interleaved pair the grouped-mm and ETP layouts store; a root missing here drops an
    # expert weight from grad-sync registration.
    expert_weight_roots=(
        *GLU_PROJECTIONS,
        "gate_up_proj_bias",
        "gate_proj_gmm",
        "up_proj_gmm",
        "gate_proj_gmm_bias",
        "up_proj_gmm_bias",
        "gate_proj_bias",
        "up_proj_bias",
        "down_proj_bias",
    ),
    # Matmul convention [E, H, 2M] / [E, M, H]: nothing is transposed on the way out.
    fused_contraction_axis=1,
)
INKLING = MoEFamilySpec(
    # The composite config spells it ``inkling_mm_model``; no bare ``inkling`` spelling exists.
    model_types=("inkling_text", "inkling_mm_model"),
    balancing_slot=_E_SCORE_CORRECTION_BIAS,
    shared_expert_attrs=("shared_experts",),
    hub_conversion_keys=("inkling_mm_model",),
    weight_sync_refusal=(
        "no end-to-end weight sync into this family's served model has been validated on either pinned rollout engine"
    ),
)
# GLM-4 MoE Lite's layout under a hub spelling of its own, pinned to transformers' ``laguna`` WeightRenaming
# entries by tests/cpu/parallelism/test_laguna_export_key_renames.py.
LAGUNA = MoEFamilySpec(
    model_types=("laguna",),
    expert_layout=ExpertLayout(ExpertLayoutKind.FUSED_SPLIT_PER_EXPERT, _GATE_UP_DOWN_PROJ),
    balancing_slot=NativeBalancingSlot(("gate.e_score_correction_bias", "experts.e_score_correction_bias")),
    export_key_renames=(
        ("gate.e_score_correction_bias", "experts.e_score_correction_bias"),
        ("shared_experts.", "shared_expert."),
    ),
    experts_container_attrs=("experts", "routed_experts"),
    shared_expert_attrs=("shared_experts", "shared_expert"),
)
LFM2 = MoEFamilySpec(
    model_types=("lfm2_moe", "lfm2_vl"),
    expert_layout=ExpertLayout(ExpertLayoutKind.FUSED_SPLIT_PER_EXPERT, ("w1", "w3", "w2")),
    balancing_slot=NativeBalancingSlot(("expert_bias",), config_flag="use_expert_bias"),
    shared_expert_attrs=("shared_experts",),
)
MISTRAL4 = MoEFamilySpec(
    # The published Mistral-Small-4 checkpoints are composite: ``mistral3`` wrapping a ``mistral4`` text tower.
    model_types=("mistral4", "mistral3"),
    shared_expert_attrs=("shared_experts",),
)
QWEN3 = MoEFamilySpec(
    model_types=("qwen3_moe",), expert_layout=ExpertLayout(ExpertLayoutKind.SEPARATE_GLU, _GATE_UP_DOWN_PROJ)
)
# Qwen3.6 ships under these same types.
QWEN3_5 = MoEFamilySpec(
    model_types=("qwen3_5_moe", "qwen3_5_moe_text"),
    expert_layout=ExpertLayout(per_expert_keys=_GATE_UP_DOWN_PROJ),
    shared_expert_attrs=("shared_expert",),
)
STEP3P7 = MoEFamilySpec(
    model_types=("step3p7", "step3p5"),
    balancing_slot=_E_SCORE_CORRECTION_BIAS,
    # The pinned server reads the family only through the release's own config.json and auto_map modules.
    exports_source_config_schema=True,
    shared_expert_attrs=("shared_experts",),
    hub_conversion_keys=("step3p7", "step3p5_vision"),
)
ZAYA = MoEFamilySpec(
    model_types=("zaya",),
    # The gate's cross-layer EDA state leaves replay undefined and re-wraps under per-layer recompute.
    supports_routing_replay=False,
    supports_gradient_checkpointing=False,
)

MOE_FAMILY_SPECS: tuple[MoEFamilySpec, ...] = (
    BAILING,
    COHERE2_MOE,
    DEEPSEEK_V4,
    GEMMA4,
    GLM4,
    GLM5_NEXT,
    GPT_OSS,
    INKLING,
    LAGUNA,
    LFM2,
    MISTRAL4,
    QWEN3,
    QWEN3_5,
    STEP3P7,
    ZAYA,
)


def _index_by_model_type(specs: tuple[MoEFamilySpec, ...]) -> dict[str, MoEFamilySpec]:
    by_model_type: dict[str, MoEFamilySpec] = {}
    for spec in specs:
        for model_type in spec.model_types:
            if by_model_type.setdefault(model_type, spec) is not spec:
                raise ValueError(f"model_type {model_type!r} is claimed by two MoE family specs")
    return by_model_type


_SPECS_BY_MODEL_TYPE = _index_by_model_type(MOE_FAMILY_SPECS)


def find_moe_family_spec(model_type: str | None) -> MoEFamilySpec | None:
    """The spec claiming ``model_type``, or ``None``."""
    return _SPECS_BY_MODEL_TYPE.get(model_type or "")


def moe_model_types() -> frozenset[str]:
    """Every ``config.model_type`` some family spec claims."""
    return frozenset(_SPECS_BY_MODEL_TYPE)


def find_moe_family_spec_for_config(config) -> MoEFamilySpec | None:
    """The spec claiming any level of ``config``: the wrapper's ``model_type`` first, then its text sub-config's,
    where a multimodal checkpoint declares the MoE family."""
    for source in config_sources(config):
        if spec := find_moe_family_spec(getattr(source, "model_type", None)):
            return spec
    return None


def wraps_experts(config, *, wrappers_requested: bool) -> bool:
    """Whether the run replaces this model's expert FFN with an EP layer.

    ``wrappers_requested`` is the run's intent (``ParallelismConfig.needs_ep_wrappers``: EP distribution or
    grouped GEMM); only a family with a spec has a layer to put in place (Qwen3-Next keeps the stock loop).
    """
    return wrappers_requested and has_routed_experts(config) and find_moe_family_spec_for_config(config) is not None


@lru_cache(maxsize=1)
def expert_weight_roots() -> frozenset[str]:
    """Every expert-weight attribute root a family declares."""
    return frozenset(root for spec in MOE_FAMILY_SPECS for root in spec.expert_weight_roots)


@lru_cache(maxsize=1)
def shared_expert_attrs() -> frozenset[str]:
    """Every shared-expert attribute a family declares: replicated, and free to name its raw parameters like the
    routed ones (Inkling's ``shared_experts.gate_proj``), so an expert-weight grouping leaves it out."""
    return frozenset(attr for spec in MOE_FAMILY_SPECS for attr in spec.shared_expert_attrs)


@lru_cache(maxsize=1)
def experts_container_attrs() -> tuple[str, ...]:
    """Every expert-container attribute a family declares, longest first so a matcher trying them in order never
    shadows a longer spelling with one it contains."""
    attrs = {attr for spec in MOE_FAMILY_SPECS for attr in spec.experts_container_attrs}
    return tuple(sorted(attrs, key=lambda attr: (-len(attr), attr)))


@lru_cache(maxsize=1)
def per_expert_layouts() -> tuple[tuple[str, str, str], ...]:
    """Every ``(gate, up, down)`` per-expert hub spelling a family declares, deduplicated: what a fused-to-per-expert
    conversion may emit and what the lazy loader may fuse back."""
    return tuple(dict.fromkeys(keys for spec in MOE_FAMILY_SPECS if (keys := spec.expert_layout.per_expert_keys)))


@lru_cache(maxsize=1)
def per_expert_hub_model_types() -> tuple[str, ...]:
    """The ``model_type`` spellings, sorted, whose family declares a per-expert hub spelling: what
    ``unfuse_moe_experts.py`` may convert."""
    return tuple(
        sorted(mt for spec in MOE_FAMILY_SPECS if spec.expert_layout.per_expert_keys for mt in spec.model_types)
    )


@lru_cache(maxsize=1)
def per_expert_fusion_map() -> dict[str, tuple[str, int]]:
    """``{"<name>.weight": (fusion_group, position)}`` over every per-expert hub spelling, halves by position.

    Keyed by projection name across the roster, so two families spelling one name at different positions raise:
    the lazy load would otherwise fuse one family's gate/up halves swapped, at identical shapes.
    """
    fusion: dict[str, tuple[str, int]] = {}
    claimed_by: dict[str, str] = {}
    for spec in MOE_FAMILY_SPECS:
        if spec.expert_layout.per_expert_keys is None:
            continue
        gate, up, down = spec.expert_layout.per_expert_keys
        for name, slot in ((gate, ("gate_up", 0)), (up, ("gate_up", 1)), (down, ("down", 0))):
            key = f"{name}.weight"
            if fusion.setdefault(key, slot) != slot:
                raise ValueError(
                    f"per-expert projection '{key}' is claimed as {fusion[key]} by {claimed_by[key]} and as {slot} "
                    f"by {spec.model_types[0]}; one of the two families' experts would be fused with its gate/up "
                    f"halves swapped."
                )
            claimed_by.setdefault(key, spec.model_types[0])
    return fusion


def is_expert_weight_attr(param_name: str) -> bool:
    """Whether a live parameter name belongs to an expert weight, by its leading segment: a native expert adapter
    counts as its weight, since the EP gather folds it in and an engine cannot load it raw."""
    root = param_name.split(".")[0]
    for suffix in EXPERT_LORA_SUFFIXES:
        root = root.removesuffix(suffix)
    return root in expert_weight_roots()
