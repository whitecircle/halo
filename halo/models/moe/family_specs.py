"""Every MoE family's facts, declared once as a frozen :class:`MoEFamilySpec` and read by ``config.model_type``."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

from halo.models.config_access import config_sources, text_config
from halo.models.moe.config_fields import has_routed_experts
from halo.models.moe.expert_layout import SEPARATE_GLU_PROJECTIONS, ExpertLayout, ExpertLayoutKind

GLU_PROJECTIONS: tuple[str, ...] = ("gate_up_proj", "gate_proj", "up_proj", "down_proj")


@dataclass(frozen=True)
class NativeBalancingSlot:
    """``attrs`` are the dotted paths the slot answers to, live and hub. ``config_flag`` is the field an engine reads
    to decide whether to apply the tensor at all (LFM-2's ``use_expert_bias``)."""

    attrs: tuple[str, ...]
    config_flag: str | None = None


@dataclass(frozen=True)
class CompositeTextLayout:
    """Where a composite checkpoint stores what its text-only export keeps at the root: the tower's keys under
    ``composite_tower``, the head's under ``composite_head``. The defaults are transformers' multimodal wrapper layout."""

    tower: str = "model."
    composite_tower: str = "model.language_model."
    head: str = "lm_head."
    composite_head: str = "lm_head."

    def to_composite_key(self, key: str) -> str:
        if key.startswith(self.tower):
            return self.composite_tower + key[len(self.tower) :]
        return self.composite_head + key[len(self.head) :] if key.startswith(self.head) else key


@dataclass(frozen=True)
class EngineLoadedBuffers:
    """Hub tensors the trainer holds as buffers and an engine loads beside the parameters of one module, so a weight
    push must carry them. ``config_flag`` is the field that creates them on the engine, whose loader refuses the name
    without it."""

    attrs: tuple[str, ...]
    config_flag: str


@dataclass(frozen=True, eq=False)
class MoEFamilySpec:
    """Identity-compared, as each family has exactly one spec constant."""

    # The family's own first, then the composite wrappers and sibling spellings that carry it.
    model_types: tuple[str, ...]
    balancing_slot: NativeBalancingSlot | None = None
    # No pinned engine reads transformers' serialization of this config, so exports carry the source repo's
    # ``config.json`` and ``auto_map`` modules.
    exports_source_config_schema: bool = False
    # ``(live module, hub)`` spellings transformers renames only inside ``from_pretrained``.
    export_key_renames: tuple[tuple[str, str], ...] = ()
    expert_layout: ExpertLayout = ExpertLayout()
    # Every expert-weight attribute the EP layer may hold across its layouts (fused, separate, ETP).
    expert_weight_roots: tuple[str, ...] = GLU_PROJECTIONS
    # Tried in order.
    experts_container_attrs: tuple[str, ...] = ("experts",)
    shared_expert_attrs: tuple[str, ...] = ()
    # Reduction axis of the fused expert tensors as the family writes them: -1 = ``F.linear`` ``[E, N, K]``.
    fused_contraction_axis: int = -1
    # transformers' conversion-mapping keys for a vendor-namespace hub checkpoint.
    hub_conversion_keys: tuple[str, ...] = ()
    # A composite transformers' CausalLM mapping does not name: ``text_only_model`` builds its text sub-config's class.
    text_only_layout: CompositeTextLayout | None = None
    supports_lazy_loading: bool = True
    supports_fp32_non_ep_params: bool = True
    supports_routing_replay: bool = True
    supports_gradient_checkpointing: bool = True
    # The special token that ends a rendered tool call, declared where it is no eos of the family's own
    # ``generation_config``, so a served turn runs past the call unless a request stops on it.
    tool_call_terminator: str | None = None
    # Why no online weight sync into the family's served model is validated.
    weight_sync_refusal: str | None = None
    # vLLM's reload rebuilds a module that received part of its tensors from uninitialized storage
    # (agent-docs/infrastructure/rollout-servers.md#partly-loaded-layers).
    engine_loaded_buffers: EngineLoadedBuffers | None = None
    # Why TRL's own vision collator and loss (DPO, KTO) do not train the family on images.
    refuses_trl_image_rows: str | None = None
    # Why the composite of the family's own remote modeling file does not train, where transformers' class does.
    remote_composite_refusal: str | None = None
    # Why the family runs eager attention only.
    eager_only_attention: str | None = None
    # Its forward passes no ``position_ids`` into mask construction, so a packed row on a dense backend runs as
    # one causal sequence.
    dense_packing_leaks: bool = False
    # Its attention reads a token's row index, not only relative distance, so a left-padded row scores as another row.
    attention_reads_row_positions: bool = False
    # The sequence mixer its forward reaches with no segment marker, so it carries every document of a
    # multi-document row into the next on every backend, named for the refusal.
    unmarked_sequence_mixer: str | None = None
    # The buffer its routers index by token id, one frozen expert assignment per row, which gives a new token no row.
    token_indexed_table: str | None = None

    def to_hub_key(self, key: str) -> str:
        for module_spelling, hub_spelling in self.export_key_renames:
            if module_spelling in key:
                return key.replace(module_spelling, hub_spelling, 1)
        return key


_E_SCORE_CORRECTION_BIAS = NativeBalancingSlot(("gate.e_score_correction_bias",))

# Ling-mini-2.0, Ring-mini-linear-2.0 and Ling-3.0 share one MoE block.
BAILING = MoEFamilySpec(
    model_types=("bailing_moe", "bailing_moe_linear", "bailing_hybrid"),
    dense_packing_leaks=True,
    expert_layout=ExpertLayout(ExpertLayoutKind.SEPARATE_GLU, SEPARATE_GLU_PROJECTIONS),
    balancing_slot=NativeBalancingSlot(("gate.expert_bias",)),
    shared_expert_attrs=("shared_experts",),
    engine_loaded_buffers=EngineLoadedBuffers(("gate.expert_bias",), config_flag="moe_router_enable_expert_bias"),
)
COHERE2_MOE = MoEFamilySpec(
    model_types=("cohere2_moe", "cohere2_vision"),
    expert_layout=ExpertLayout(per_expert_keys=SEPARATE_GLU_PROJECTIONS),
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
    token_indexed_table="gate.tid2eid",
    attention_reads_row_positions=True,
    eager_only_attention=(
        "its head_dim=512 exceeds every flash kernel, SDPA and flex cannot carry its sink column, and its "
        "compressor concatenates KV entries after the mask is built"
    ),
)
GEMMA4 = MoEFamilySpec(
    model_types=("gemma4", "gemma4_text"),
    expert_layout=ExpertLayout(key_prefix=""),
    # Documented to fail at the first dispatch (agent-docs/models/gemma4.md); refused at load instead.
    supports_fp32_non_ep_params=False,
    # The sibling router passes only pre-weighted (indices, weights), with nothing to re-derive.
    supports_routing_replay=False,
)
# The remote-code revisions spell the expert container ``routed_experts`` and the shared expert in the singular.
GLM4 = MoEFamilySpec(
    model_types=("glm4_moe_lite",),
    expert_layout=ExpertLayout(ExpertLayoutKind.FUSED_SPLIT_PER_EXPERT, SEPARATE_GLU_PROJECTIONS),
    balancing_slot=_E_SCORE_CORRECTION_BIAS,
    experts_container_attrs=("experts", "routed_experts"),
    shared_expert_attrs=("shared_experts", "shared_expert"),
)
GLM5_NEXT = MoEFamilySpec(
    model_types=("glm5_next", "glm5_next_text"),
    dense_packing_leaks=True,
    expert_layout=ExpertLayout(per_expert_keys=SEPARATE_GLU_PROJECTIONS),
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
    tool_call_terminator="<|call|>",
    dense_packing_leaks=True,
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
    # InklingShortConvolution reads seq_idx, but its call sites forward no kwargs.
    unmarked_sequence_mixer="depthwise causal short convolutions (on the projected K/V and around each sublayer)",
    weight_sync_refusal=(
        "no end-to-end weight sync into this family's served model has been validated on either pinned rollout engine"
    ),
)
# GLM-4 MoE Lite's layout under a hub spelling of its own, pinned to transformers' ``laguna`` WeightRenaming entries.
LAGUNA = MoEFamilySpec(
    model_types=("laguna",),
    expert_layout=ExpertLayout(ExpertLayoutKind.FUSED_SPLIT_PER_EXPERT, SEPARATE_GLU_PROJECTIONS),
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
    attention_reads_row_positions=True,
)
# Gateless experts on a latent, stored per expert; no ``per_expert_keys``, as there is no gate half to name. Both engines
# parse the release's own config schema (``llm_config``, ``vit_hidden_size``), which transformers absorbs at load.
NEMOTRON_H = MoEFamilySpec(
    model_types=("nemotron_h", "nemotron_h_omni"),
    expert_layout=ExpertLayout(ExpertLayoutKind.SEPARATE_MLP),
    balancing_slot=_E_SCORE_CORRECTION_BIAS,
    exports_source_config_schema=True,
    shared_expert_attrs=("shared_experts",),
    text_only_layout=CompositeTextLayout(
        tower="backbone.", composite_tower="language_model.backbone.", composite_head="language_model.lm_head."
    ),
    weight_sync_refusal=(
        "both pinned rollout engines serve the release, and no online update into either has been validated end to "
        "end (agent-docs/models/nemotron-h.md#rl-weight-sync says what lifts this)"
    ),
    refuses_trl_image_rows=(
        "TRL's vision path computes no reference ahead of training, so a full fine-tune holds the unsharded frozen "
        "reference on every GPU beside the policy, and neither method has been validated on this family's images"
    ),
    remote_composite_refusal=(
        "its forward hands the text tower no segment marker, so packed documents would cross the Mamba layers, reads "
        "no image_grid_hw, and trains Radio's layer scale, which no engine reads"
    ),
)
QWEN3 = MoEFamilySpec(
    model_types=("qwen3_moe",), expert_layout=ExpertLayout(ExpertLayoutKind.SEPARATE_GLU, SEPARATE_GLU_PROJECTIONS)
)
# Qwen3.6 ships under these same types.
QWEN3_5 = MoEFamilySpec(
    model_types=("qwen3_5_moe", "qwen3_5_moe_text"),
    expert_layout=ExpertLayout(per_expert_keys=SEPARATE_GLU_PROJECTIONS),
    shared_expert_attrs=("shared_expert",),
)
# Qwen3.5-MoE's block with a router that renormalizes only under ``norm_topk_prob``; the hub stores the experts
# one tensor per expert, which transformers fuses on load.
QWEN3_NEXT = MoEFamilySpec(
    model_types=("qwen3_next",),
    expert_layout=ExpertLayout(ExpertLayoutKind.FUSED_SPLIT_PER_EXPERT, SEPARATE_GLU_PROJECTIONS),
    shared_expert_attrs=("shared_expert",),
    weight_sync_refusal=(
        "both pinned rollout engines register Qwen3NextForCausalLM and load the per-expert keys the gather emits, "
        "but no online update into either has been validated end to end"
    ),
)
# Qwen3.8-Flash-Next: Qwen3-Next's router over Qwen3.5-MoE's block, its experts fused on the hub too.
QWEN4_EXP = MoEFamilySpec(
    model_types=("qwen4_exp", "qwen4_exp_text"),
    expert_layout=ExpertLayout(per_expert_keys=SEPARATE_GLU_PROJECTIONS),
    shared_expert_attrs=("shared_expert",),
    weight_sync_refusal=(
        "no online update into either pinned rollout engine has been validated end to end; on SGLang, each "
        "hyper-connection GatedResidual also caches a padded copy of its mix-up weight on SM100-family GPUs that an "
        "update leaves stale (agent-docs/models/qwen4_exp.md#rl-weight-sync)"
    ),
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
    NEMOTRON_H,
    QWEN3,
    QWEN3_5,
    QWEN3_NEXT,
    QWEN4_EXP,
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
    return _SPECS_BY_MODEL_TYPE.get(model_type or "")


def moe_model_types() -> frozenset[str]:
    return frozenset(_SPECS_BY_MODEL_TYPE)


def find_moe_family_spec_for_config(config) -> MoEFamilySpec | None:
    """The wrapper's ``model_type`` first, then its text sub-config's, where a multimodal checkpoint declares it."""
    for source in config_sources(config):
        if spec := find_moe_family_spec(getattr(source, "model_type", None)):
            return spec
    return None


def attention_reads_row_positions(config) -> bool:
    """Read off the text tower's own type, since the composite ``mistral3`` also wraps a dense ``mistral`` tower."""
    spec = find_moe_family_spec(getattr(text_config(config), "model_type", None)) if config is not None else None
    return spec is not None and spec.attention_reads_row_positions


def wraps_experts(config, *, ep_layers_requested: bool) -> bool:
    """Only a family with a spec has an EP layer to put in place (Mixtral keeps the stock experts)."""
    return ep_layers_requested and has_routed_experts(config) and find_moe_family_spec_for_config(config) is not None


@lru_cache(maxsize=1)
def expert_weight_roots() -> frozenset[str]:
    return frozenset(root for spec in MOE_FAMILY_SPECS for root in spec.expert_weight_roots)


@lru_cache(maxsize=1)
def shared_expert_attrs() -> frozenset[str]:
    """Replicated, and free to name raw parameters like the routed ones (Inkling's ``shared_experts.gate_proj``), so an
    expert-weight grouping leaves it out."""
    return frozenset(attr for spec in MOE_FAMILY_SPECS for attr in spec.shared_expert_attrs)


@lru_cache(maxsize=1)
def experts_container_attrs() -> tuple[str, ...]:
    """Longest first, so a matcher trying them in order never shadows a longer spelling with one it contains."""
    attrs = {attr for spec in MOE_FAMILY_SPECS for attr in spec.experts_container_attrs}
    return tuple(sorted(attrs, key=lambda attr: (-len(attr), attr)))


@lru_cache(maxsize=1)
def per_expert_layouts() -> tuple[tuple[str, str, str], ...]:
    """What a fused-to-per-expert conversion may emit and what the lazy loader may fuse back."""
    return tuple(dict.fromkeys(keys for spec in MOE_FAMILY_SPECS if (keys := spec.expert_layout.per_expert_keys)))


@lru_cache(maxsize=1)
def per_expert_hub_model_types() -> tuple[str, ...]:
    """What ``unfuse_moe_experts.py`` may convert."""
    return tuple(
        sorted(mt for spec in MOE_FAMILY_SPECS if spec.expert_layout.per_expert_keys for mt in spec.model_types)
    )


@lru_cache(maxsize=1)
def per_expert_fusion_map() -> dict[str, tuple[str, int]]:
    """``{"<name>.weight": (fusion_group, position)}``. Two families spelling one name at different positions raise, as
    the lazy load would fuse one family's gate/up halves swapped, at identical shapes."""
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
