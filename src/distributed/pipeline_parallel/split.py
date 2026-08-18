"""Model → pipeline-stage splitting: the per-family contract registry and the layer partition.

A pipeline stage is the model's REAL backbone with its decoder-layer list sliced and the parts the
stage does not own neutralized (see :mod:`.stage`), so every family's own rotary embeddings, mask
construction, ``layer_types``, sliding-window and MoE handling are inherited rather than
reimplemented. That makes the splitter model-agnostic by default; only families whose forward
threads state a contiguous cut would sever declare themselves here.

Registration mirrors the EP/CP wrapper registries: a family self-registers by subclassing
:class:`PPModelSpec` with its own ``HF_MODULE_NAMES``, discovered via
:func:`~src.distributed.module_registry.build_hf_module_name_map`. Absence of a spec means the
generic path applies — new families are supported by default and only genuine exceptions cost code.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from types import MethodType

import torch.nn as nn

from src.distributed.module_registry import build_hf_module_name_map
from src.distributed.pipeline_parallel.stage_adapters import deepseek_v4_stream_forward, glm5_next_stream_forward
from src.models.attention_geometry import resolve_head_dim
from src.models.loading.config_levels import get_config_field, text_config
from src.models.moe_balancing import (
    config_has_experts,
    resolve_balancing_mode,
    resolve_router_topk,
)
from src.models.structure import DECODER_LAYER_LIST_ATTRS, backbone_with_layers, decoder_layers

logger = logging.getLogger(__name__)

# The config field every family spells its input/output embedding tie with.
TIE_WORD_EMBEDDINGS_FLAG = "tie_word_embeddings"

# Config fields a family may use to declare multi-token-prediction tail layers. No class owns this —
# it is a config spelling, and the families disagree (DeepSeek/Bailing ``num_nextn_predict_layers``,
# Qwen3.5/3.6 ``mtp_num_hidden_layers``, Inkling ``num_mtp_layers``) — so it is data, checked in
# full by _reject_live_mtp_layers.
MTP_LAYER_COUNT_FIELDS: tuple[str, ...] = ("num_nextn_predict_layers", "mtp_num_hidden_layers", "num_mtp_layers")


class PPModelSpec:
    """Per-family pipeline-split contract, keyed by the HF backbone class name.

    Subclasses declare ``HF_MODULE_NAMES`` (the backbone classes they claim) and override only what
    is genuinely distinct. A family with no spec gets the generic contiguous split.
    """

    # HF backbone class names this spec claims (the module that owns ``.layers``).
    HF_MODULE_NAMES: tuple[str, ...] = ()

    # False rejects the family at config time; UNSUPPORTED_REASON must name the mechanism.
    SUPPORTS_PP: bool = True
    UNSUPPORTED_REASON: str = ""

    # Where the splitter locates the two modules it neutralizes, probed in order (first hit wins).
    # The norm is applied AFTER the last decoder layer, so non-last stages neutralize it, else the
    # hidden state is normalized mid-network. The decoder-layer list is deliberately not here:
    # ``backbone_with_layers`` owns that spelling repo-wide (``DECODER_LAYER_LIST_ATTRS``).
    FINAL_NORM_ATTRS: tuple[str, ...] = ("norm", "final_layernorm", "ln_f", "embedding_norm")
    HEAD_ATTRS: tuple[str, ...] = ("lm_head", "score", "classifier")

    # Further post-layers modules the backbone applies before/inside the final norm (the
    # hyper-connection collapse ``hc_head``). Exact attribute names, not probes: each one declared
    # here must exist, is neutralized to Identity on non-last stages, and is last-stage-owned state
    # for the stage-aware loader's droppable-prefix filter.
    TAIL_MODULE_ATTRS: tuple[str, ...] = ()

    # True only where a sliced stage provably selects every per-layer input by the LAYER'S OWN
    # attributes (its ``layer_idx``/``block_type``, set at construction from the global index) rather
    # than by list position — either because the upstream forward already does, or because this
    # spec's ``adapt_stage_backbone`` installs a mid-chain forward that does. Exempts the family from
    # ``reject_layer_type_rebase`` and from period-snapping the automatic partition.
    LAYER_TYPES_REBASE_SAFE: bool = False

    # The mid-chain backbone forward a non-first stage runs, for a family whose boundary activation
    # is not an ``inputs_embeds``-shaped tensor (:mod:`.stage_adapters`). ``None`` = the upstream
    # forward, driven through ``inputs_embeds`` by the generic contract.
    STREAM_FORWARD: Callable | None = None

    @classmethod
    def adapt_stage_backbone(cls, backbone: nn.Module, *, is_first: bool, is_last: bool) -> None:
        """Family hook run once per stage build, after slicing and neutralization.

        Binds :attr:`STREAM_FORWARD` onto the backbone INSTANCE for non-first stages — instance
        binding, not a class patch, so stage 0 and any unsplit model in the same process keep the
        real forward; ``nn.Module.__call__`` resolves ``self.forward`` through the instance, so
        FSDP2/GC hooks and the stage wrapper see the bound method transparently. A family declaring
        no ``STREAM_FORWARD`` keeps its own forward on every stage.
        """
        if cls.STREAM_FORWARD is not None and not is_first:
            backbone.forward = MethodType(cls.STREAM_FORWARD, backbone)

    @classmethod
    def validate_partition(cls, model: nn.Module, partition: list[tuple[int, int]]) -> None:
        """Raise if a stage boundary severs family-specific cross-layer state.

        Runs on every rank with the FULL partition (like ``reject_layer_type_rebase``), so an
        invalid split fails the whole world together instead of stranding peers at the first
        collective. Default: no constraint.
        """

    def __init_subclass__(cls, **kwargs):
        """Reject a subclass that disables PP without saying why."""
        super().__init_subclass__(**kwargs)
        if not cls.SUPPORTS_PP and not vars(cls).get("UNSUPPORTED_REASON"):
            raise TypeError(f"{cls.__name__} sets SUPPORTS_PP=False but declares no UNSUPPORTED_REASON")


class ZayaPPSpec(PPModelSpec):
    """Zaya carries TWO streams across every layer boundary; one of them accumulates."""

    HF_MODULE_NAMES = ("ZayaModel",)
    SUPPORTS_PP = False
    UNSUPPORTED_REASON = (
        "Zaya's forward carries two tensors across every decoder-layer boundary: the fp32 residual "
        "stream (cast to float32 before the loop, de-cast only at the final norm) and the EDA "
        "router state (prev_router_hidden_states), which each layer's router adds to and forwards, "
        "so it accumulates down the stack and cannot be recomputed on a later stage. A stage cut "
        "would need a two-tensor, mixed-dtype P2P boundary the single-activation pipeline contract "
        "does not carry; the loop also selects layer_types and rotary embeddings by list position."
    )


class DeepSeekV4PPSpec(PPModelSpec):
    """DeepSeek-V4: hyper-connection stream boundary + hash-router layers pinned to stage 0.

    The inter-layer activation is the ``hc_mult``-widened ``[B, S, hc_mult, D]`` stream (widened
    from ``inputs_embeds`` before layer 0, collapsed by the learned ``hc_head`` after the last
    layer), so non-first stages run the mirrored mid-chain forward and non-last stages neutralize
    ``hc_head`` alongside the norm. ``input_ids`` is consumed only by the ``hash_moe`` (tid2eid)
    routers, which the partition gate confines to stage 0 — the one stage that has the ids.
    Nothing in the loop indexes by list position (masks are loop-invariant, per-layer state hangs
    off each module's own ``layer_idx``), so any split offset is exact.
    """

    HF_MODULE_NAMES = ("DeepseekV4Model",)
    TAIL_MODULE_ATTRS = ("hc_head",)
    LAYER_TYPES_REBASE_SAFE = True
    STREAM_FORWARD = staticmethod(deepseek_v4_stream_forward)

    @classmethod
    def validate_partition(cls, model: nn.Module, partition: list[tuple[int, int]]) -> None:
        """Every ``hash_moe`` layer must land on stage 0.

        The hash router routes on token identity (``tid2eid[input_ids]``); a non-first stage has no
        ``input_ids`` and would crash on the ``None`` dereference mid-schedule, stranding its peers.
        """
        mlp_layer_types = getattr(text_config(getattr(model, "config", None)), "mlp_layer_types", None) or ()
        hash_layers = [i for i, kind in enumerate(mlp_layer_types) if kind == "hash_moe"]
        stage0_hi = partition[0][1]
        stranded = [i for i in hash_layers if i >= stage0_hi]
        if stranded:
            raise ValueError(
                f"DeepSeek-V4 hash_moe layer(s) {stranded} fall outside pipeline stage 0 (layers "
                f"[0, {stage0_hi})). Their tid2eid routers consume input_ids, which only stage 0 "
                f"receives. Use pipeline_split to give stage 0 at least {max(hash_layers) + 1} layers."
            )


class Gemma4PPSpec(PPModelSpec):
    """Gemma4 indexes per-layer inputs by enumerate position and shares KV across distant layers."""

    HF_MODULE_NAMES = ("Gemma4TextModel",)
    SUPPORTS_PP = False
    UNSUPPORTED_REASON = (
        "Gemma4 slices its per-layer embeddings as per_layer_inputs[:, :, i, :] where i is the "
        "ENUMERATE position in the layer list, which slicing the list silently re-bases, and its "
        "trailing num_kv_shared_layers consume an earlier layer's K/V through a forward-threaded "
        "dict that a stage boundary would break."
    )


class Glm5NextPPSpec(PPModelSpec):
    """GLM-5 (``glm5_next``): hyper-connection stream boundary + stage-local DSA indexer chains.

    Same widened-stream contract as DeepSeek-V4 (``[B, S, hc_mult, D]`` boundary, parameterless
    ``hc_head`` mean-collapse neutralized off non-last stages, mirrored mid-chain forward). The DSA
    indexer's top-k indices flow only between a "full" indexer layer and the "shared" layers
    immediately after it (any KDA layer resets the chain to None), so the partition gate keeps each
    chain within one stage. ``input_ids`` reaches the layer kwargs upstream but nothing in the
    decoder consumes it, and the mid-chain forward keys each mask by the layer's own ``block_type``,
    so split offsets are position-independent.

    The family ships only the composite ``Glm5NextForConditionalGeneration`` (no text-only CausalLM
    sibling in transformers); a text-only run of it reaches this spec through the trainer's
    multimodal admission, which stashes the vision tower and projector for the save.
    """

    HF_MODULE_NAMES = ("Glm5NextTextModel",)
    TAIL_MODULE_ATTRS = ("hc_head",)
    LAYER_TYPES_REBASE_SAFE = True
    STREAM_FORWARD = staticmethod(glm5_next_stream_forward)

    @classmethod
    def validate_partition(cls, model: nn.Module, partition: list[tuple[int, int]]) -> None:
        """No stage may begin on a "shared" DSA layer.

        A shared layer reuses the previous full indexer layer's top-k indices; the model loop hands
        them layer-to-layer, so a stage starting on one would raise mid-schedule ("Shared DSA layers
        require top-k indices") with its peers already committed to the step.
        """
        indexer_types = getattr(text_config(getattr(model, "config", None)), "indexer_types", None) or ()
        severed = [lo for lo, _ in partition[1:] if lo < len(indexer_types) and indexer_types[lo] == "shared"]
        if severed:
            raise ValueError(
                f"GLM-5 pipeline stage boundary at layer(s) {severed} severs a DSA indexer chain: "
                f"config.indexer_types marks those layers 'shared', so they reuse the previous full "
                f"indexer layer's top-k indices from across the cut. Choose pipeline_split boundaries "
                f"that keep each full+shared indexer run on one stage."
            )


def build_pp_spec_map() -> dict[str, type[PPModelSpec]]:
    """HF backbone class name → :class:`PPModelSpec`, derived from every registered spec."""
    return build_hf_module_name_map(PPModelSpec, "backbone")


PP_SPEC_MAP = build_pp_spec_map()


def resolve_pp_spec(backbone: nn.Module) -> type[PPModelSpec]:
    """The spec claiming ``backbone``'s class, or the permissive generic base."""
    return PP_SPEC_MAP.get(type(backbone).__name__, PPModelSpec)


def _embedding_module(model: nn.Module, getter: str) -> nn.Module | None:
    """``model.get_{input,output}_embeddings()`` when the model implements it, else ``None``."""
    fn = getattr(model, getter, None)
    return fn() if callable(fn) else None


def tie_flag_configs(config, prefix: str = "") -> dict[str, object]:
    """Dotted sub-config path → the config object there that declares the embedding tie.

    A composite (VLM) config declares it on its text sub-config as well as, or instead of, the top
    level, and transformers resolves the tie across every submodel — so the top level alone is not
    the whole declaration. Keyed by path rather than position so the values can be put back on a
    DIFFERENT config tree (``from_pretrained`` hands the model a deep copy).
    """
    holders = {prefix: config} if hasattr(config, TIE_WORD_EMBEDDINGS_FLAG) else {}
    for name in getattr(config, "sub_configs", None) or ():
        sub = getattr(config, name, None)
        if sub is not None:
            holders.update(tie_flag_configs(sub, f"{prefix}{name}."))
    return holders


def _reject_tied_embeddings(model: nn.Module) -> None:
    """Raise if the input embedding and the output head are one weight.

    Stage 0 owns ``embed_tokens`` and the last stage ``lm_head``; splitting gives each rank only its
    own path's gradient, so the two diverge from step 1, and ``reconcile_tie_word_embeddings`` never
    fires because it needs both keys in ONE rank's state dict. The declared flag alone is
    disqualifying too: it is saved into every stage's config, so ``from_pretrained`` re-ties the head
    and discards the trained one.
    """
    declared = any(
        bool(getattr(holder, TIE_WORD_EMBEDDINGS_FLAG, False))
        for holder in tie_flag_configs(getattr(model, "config", None)).values()
    )
    input_weight = getattr(_embedding_module(model, "get_input_embeddings"), "weight", None)
    output_weight = getattr(_embedding_module(model, "get_output_embeddings"), "weight", None)

    # Shared storage cannot be the only signal: the stage-aware loader builds its meta shell with the
    # tie suppressed, so a composite/VLM declaring it on a sub-config would leave both False here.
    # The flag counts only where an output embedding exists, since a score head inherits it and ties nothing.
    tied_storage = input_weight is not None and input_weight is output_weight
    if not (tied_storage or (declared and output_weight is not None)):
        return
    raise ValueError(
        f"{type(model).__name__} ties its output head to its input embedding "
        f"(tie_word_embeddings), which pipeline parallelism does not support: the embedding lands on "
        f"stage 0 and the head on the last stage, so each copy would receive only its own path's "
        f"gradient and the two diverge from step 1 — silently, because "
        f"reconcile_tie_word_embeddings only fires when both keys are in ONE rank's state dict. "
        f"Supporting it requires an all-reduce of the tied weight's gradient across the pipeline "
        f"chain plus a cross-stage reconcile at save. Use a checkpoint with untied embeddings, or "
        f"train this model without PP."
    )


def _reject_unapplied_logit_scale(model: nn.Module) -> None:
    """Raise if the model's forward scales its lm_head output (Cohere ``logit_scale``).

    The stage head path computes ``head(hidden)`` directly, so a scale the ForCausalLM forward
    applies after the matmul would silently vanish — every PP logit, loss and gradient off by that
    factor. Unit scale is a no-op and passes.
    """
    text = text_config(getattr(model, "config", None))
    scale = getattr(text, "logit_scale", None)
    if scale is None or scale == 1.0:
        return
    raise ValueError(
        f"{type(model).__name__} declares logit_scale={scale}, which its forward applies to the "
        f"lm_head output after the matmul; the pipeline stage head computes the matmul alone, so "
        f"every PP logit and loss would silently be off by that factor. Train this model without PP."
    )


def _reject_aux_loss_balancing(model: nn.Module, moe_balancing: str) -> None:
    """Raise if this run's MoE load balancing rides the HF aux-loss path a stage severs.

    A stage calls the BACKBONE's forward and applies the head itself, so ``*ForCausalLM.forward`` —
    where ``loss += router_aux_loss_coef * load_balancing_loss_func(...)`` lives — never runs and the
    balancing term is dropped with no error, collapsing the routing thousands of steps later.
    Unaffected: bias-update balancing, a zero aux-loss coefficient, and non-causal-LM heads.
    """
    config = getattr(model, "config", None)
    if config is None or not config_has_experts(config):
        return
    if _embedding_module(model, "get_output_embeddings") is None:
        return
    if resolve_balancing_mode(moe_balancing, model, is_moe=True) != "aux_loss":
        return
    coef = get_config_field(config, "router_aux_loss_coef", 0) or 0
    if coef <= 0:
        return
    raise ValueError(
        f"{type(model).__name__} balances its experts through the HuggingFace router aux loss "
        f"(moe_balancing={moe_balancing!r} resolves to 'aux_loss', router_aux_loss_coef={coef}), "
        f"which pipeline parallelism drops: a stage runs the backbone and applies the head itself, "
        f"so the *ForCausalLM.forward that adds router_aux_loss_coef * load_balancing_loss_func(...) "
        f"never executes. Nothing raises — the experts just collapse over training. Supporting it "
        f"requires each stage to request its own router logits and add its share of the aux loss to "
        f"the pipeline loss. Use moe_balancing=bias_update on a family whose EP wrappers support the "
        f"aux-loss-free bias update, moe_balancing=none to accept unbalanced experts, or train this "
        f"model without PP."
    )


def _reject_live_mtp_layers(model: nn.Module) -> None:
    """Raise if the model INSTANTIATED multi-token-prediction tail layers.

    MTP layers re-embed ``input_ids`` (rolled) inside the backbone forward — a second
    input_ids-dependent stream that a contiguous hidden-state cut cannot carry, and on the
    ``inputs_embeds`` path a non-first stage would crash on the ``input_ids=None`` dereference
    (Bailing's ``roll_tensor``).

    Liveness is judged by the BUILT tree, not the declared field: the live-MTP construction
    (remote-code Bailing v3) appends its MTP layers to the decoder list, so a list longer than
    ``num_hidden_layers`` is the signature. The native families (Step3p7, Qwen3.5/3.6, Inkling,
    DeepSeek-V4, GLM-5) carry the field as checkpoint metadata and drop the MTP weights at load
    (``_keys_to_ignore_on_load_unexpected``) — their hub configs declare 1–3 tail layers that are
    never built, which a declaration-only gate would falsely refuse. Every declared spelling still
    counts, so a family that starts building its declared head is caught the release it does.
    """
    text = text_config(getattr(model, "config", None))
    declared = {field: n for field in MTP_LAYER_COUNT_FIELDS if (n := getattr(text, field, 0) or 0) > 0}
    if not declared:
        return
    backbone = backbone_with_layers(model)
    num_hidden = getattr(text, "num_hidden_layers", None)
    # A pre-sliced stage (the stage-aware loader) holds fewer layers than num_hidden_layers and
    # trivially passes — its unsplit shell was already judged before slicing.
    if backbone is not None and num_hidden is not None and len(decoder_layers(backbone)) <= num_hidden:
        return
    detail = ", ".join(f"{field}={n}" for field, n in sorted(declared.items()))
    raise ValueError(
        f"{type(model).__name__} declares {detail} and its decoder list holds the extra layers: "
        f"multi-token-prediction tail layers re-embed input_ids inside the backbone forward, a "
        f"second cross-stage input stream pipeline parallelism cannot carry. Train with the MTP "
        f"head disabled (set every field above to 0 so the tail layers are never built) or "
        f"without PP."
    )


def validate_model_structure_supports_pp(model: nn.Module) -> None:
    """Raise if this model's STRUCTURE cannot be split into pipeline stages.

    Everything here reads classes, config fields and weight identity — never a tensor's values — so
    it holds on a meta-device shell and the stage-aware loader can run it before materializing a
    single weight. The run-knob gate (MoE balancing) lives in :func:`validate_model_supports_pp`,
    which the trainer calls once it knows the run's setting.

    Raises:
        ValueError: the family declares ``SUPPORTS_PP = False``, has no discoverable layer list, ties
            its embedding to its head, or declares multi-token-prediction tail layers.
    """
    backbone = backbone_with_layers(model)
    if backbone is None:
        raise ValueError(
            f"Pipeline parallelism needs a transformer backbone with a decoder-layer list, but none "
            f"was found on {type(model).__name__}. Supported layouts are exactly those "
            f"backbone_with_layers understands (src/models/structure.py)."
        )
    spec = resolve_pp_spec(backbone)
    if not spec.SUPPORTS_PP:
        raise ValueError(f"{type(backbone).__name__} does not support pipeline parallelism. {spec.UNSUPPORTED_REASON}")
    _reject_tied_embeddings(model)
    _reject_unapplied_logit_scale(model)
    _reject_live_mtp_layers(model)


def validate_model_supports_pp(model: nn.Module, moe_balancing: str = "auto") -> None:
    """Raise if this model cannot be split into pipeline stages under this run's settings.

    Takes the run's resolved ``moe_balancing`` so the aux-loss gate judges the strategy the run will
    actually apply — a stage severs the aux-loss path — rather than a hypothetical one.
    """
    validate_model_structure_supports_pp(model)
    _reject_aux_loss_balancing(model, moe_balancing)


def resolve_module_attr(module: nn.Module, candidates: tuple[str, ...], what: str) -> str:
    """Name of the first attribute in ``candidates`` that ``module`` actually carries.

    Single probe used for the layer list, the final norm and the task head, so the "which attribute
    holds it" question is answered one way everywhere.

    Raises:
        ValueError: none is present. Always fatal — every caller would otherwise silently skip a
            step (an un-neutralized norm, an unsliced layer list) and produce wrong numbers.
    """
    for name in candidates:
        if getattr(module, name, None) is not None:
            return name
    raise ValueError(
        f"{type(module).__name__} exposes no {what} under any of {candidates}. Pipeline parallelism "
        f"needs it; declare the attribute on this family's PPModelSpec if the layout differs."
    )


def decoder_layers_attr(backbone: nn.Module) -> str:
    """Name of the attribute holding this backbone's decoder-layer list."""
    return resolve_module_attr(backbone, DECODER_LAYER_LIST_ATTRS, "decoder-layer list")


def compute_layer_partition(
    num_layers: int,
    pp_size: int,
    *,
    head_layer_equivalents: float = 0.0,
    boundary_period: int = 1,
    pp_split: list[int] | None = None,
) -> list[tuple[int, int]]:
    """Split ``num_layers`` into ``pp_size`` contiguous ``[lo, hi)`` ranges.

    ``pp_split`` (per-stage decoder-layer counts) overrides everything — the manual escape hatch
    for a measured imbalance. Otherwise, ``head_layer_equivalents`` > 0 rebalances the LAST stage:
    it carries the ``lm_head`` matmul + CE (the dominant non-layer cost — the embedding is a
    lookup), so its layer budget shrinks by that many layer-equivalents, in whole multiples of
    ``boundary_period`` (heterogeneous ``config.layer_types`` select the attention mask by list
    position, so stage boundaries must land on the pattern period). Any leftover goes to the
    MIDDLE stages first. With no head weighting and an exact division every stage gets the same
    count.

    Raises:
        ValueError: fewer layers than stages (a stage would hold no decoder layer), or a
            ``pp_split`` whose length/sum/minimum does not fit the model.
    """
    if num_layers < pp_size:
        raise ValueError(
            f"Cannot split {num_layers} decoder layers across {pp_size} pipeline stages: every stage "
            f"needs at least one layer. Reduce pipeline_parallel_size to at most {num_layers}."
        )
    if pp_split is not None:
        if len(pp_split) != pp_size:
            raise ValueError(f"pp_split has {len(pp_split)} entries but pipeline_parallel_size={pp_size}.")
        if min(pp_split) < 1:
            raise ValueError(f"pp_split entries must be >= 1 decoder layer, got {pp_split}.")
        if sum(pp_split) != num_layers:
            raise ValueError(f"pp_split sums to {sum(pp_split)} but the model has {num_layers} decoder layers.")
        sizes = list(pp_split)
    else:
        sizes = _balanced_sizes(num_layers, pp_size, head_layer_equivalents, max(1, boundary_period))

    partition, lo = [], 0
    for size in sizes:
        partition.append((lo, lo + size))
        lo += size
    return partition


def _balanced_sizes(num_layers: int, pp_size: int, head_layer_equivalents: float, period: int) -> list[int]:
    """Per-stage layer counts; the last stage's budget shrinks by the head's layer-equivalent cost.

    With ``period > 1`` every boundary is a whole number of periods and the ragged tail
    (``num_layers % period``) stays on the LAST stage — a mid-period boundary would be refused by
    ``reject_layer_type_rebase``, so an un-snapped "uniform" split of a ragged list is not a
    fallback, it is a dead end.
    """
    uniform = [num_layers // pp_size] * pp_size
    # Middle-out remainder: stage 0 carries the embedding and the last the head, so the ends stay light.
    order = sorted(range(pp_size), key=lambda i: (-min(i, pp_size - 1 - i), i))
    for i in order[: num_layers % pp_size]:
        uniform[i] += 1

    if pp_size == 1 or (head_layer_equivalents <= 0.0 and period <= 1):
        return uniform
    units_total, tail = divmod(num_layers, period)
    if units_total < pp_size:
        return uniform  # period-sized units cannot give every stage a layer; keep the uniform split

    # In period-sized units: the last stage's budget is the per-stage share of (layers + head) minus
    # the head and the ragged tail it already carries.
    head_units = head_layer_equivalents / period
    tail_units = tail / period
    target = (units_total + tail_units + head_units) / pp_size
    last = round(target - head_units - tail_units)
    # The clamp leaves at least one unit per front stage: rest >= pp_size - 1 by construction. A
    # last stage that carries the tail may hold zero whole units — the tail layers are its share.
    last = max(0 if tail else 1, min(units_total - (pp_size - 1), last))
    rest = units_total - last
    front = [rest // (pp_size - 1)] * (pp_size - 1)
    front_order = sorted(range(pp_size - 1), key=lambda i: (-min(i, pp_size - 2 - i), i))
    for i in front_order[: rest % (pp_size - 1)]:
        front[i] += 1
    return [units * period for units in front] + [last * period + tail]


def head_cost_layer_equivalents(config) -> float:
    """The ``lm_head`` matmul's cost in decoder-layer equivalents, from parameter counts.

    FLOPs/token of a matmul are proportional to its parameters; the attention quadratic term and
    the (free) embedding lookup are ignored, which only softens the correction. A MoE layer counts
    its per-token ACTIVATED expert parameters. Returns 0.0 (uniform split) when the config lacks
    the needed fields.
    """
    text = text_config(config)
    hidden = getattr(text, "hidden_size", None)
    vocab = getattr(text, "vocab_size", None)
    # Attention and FFN widths read through the per-layer-aware seam: on a per-layer-heterogeneous
    # family (step3p7's 64 full / 96 sliding heads) the bare attribute RAISES transformers'
    # AmbiguousGlobalPerLayerAttributeError — a RuntimeError the getattr default does not swallow —
    # and this is a cost ceiling, so ``max`` is the right reduce.
    heads = get_config_field(config, "num_attention_heads", per_layer_reduce=max)
    if not hidden or not vocab or not heads:
        return 0.0
    head_dim = resolve_head_dim(config)
    kv_heads = get_config_field(config, "num_key_value_heads", per_layer_reduce=max) or heads
    attn = 2 * hidden * heads * head_dim + 2 * hidden * kv_heads * head_dim
    intermediate = get_config_field(config, "moe_intermediate_size", per_layer_reduce=max) or get_config_field(
        config, "intermediate_size", per_layer_reduce=max
    )
    if not intermediate:
        return 0.0
    # A MoE layer's per-token cost scales with the router width, read through the one registry that
    # owns its spellings — a family whose config uses a spelling this cost model missed would be
    # costed as dense and skew the split. Gated on the model actually being MoE: the ``top_k``
    # spelling doubles as the legacy generation sampling parameter, which a dense config may carry.
    ffn_mult = (resolve_router_topk(config) or 1) if config_has_experts(config) else 1
    mlp = ffn_mult * 3 * hidden * intermediate
    return (hidden * vocab) / (attn + mlp)


def layer_types_shift_invariant(layer_types, shift: int) -> bool:
    """Whether re-basing a decoder-layer list by ``shift`` leaves every layer's ``layer_types`` entry
    unchanged.

    The one criterion behind both PP layer-type rules: :func:`layer_types_period` returns the smallest
    shift satisfying it, so an automatic partition can snap to whole periods, and
    :func:`~src.distributed.pipeline_parallel.stage.reject_layer_type_rebase` refuses a stage offset
    that does not. Spelled twice, a partition could snap to a boundary the gate then rejects.
    """
    return not shift or not layer_types or layer_types[shift:] == layer_types[:-shift]


def layer_types_period(config) -> int:
    """Smallest shift that leaves ``layer_types`` unchanged (1 when uniform or absent).

    Stage boundaries must land on whole periods — several families select the attention mask by
    list position. Requiring the pattern to tile ``len(layer_types)`` exactly would miss the real
    ragged lists (step3p7's 45 layers = 11 whole ``full,s,s,s`` periods plus one trailing layer) and
    leave their default partition un-snapped, i.e. loudly refused.
    """
    text = text_config(config)
    layer_types = getattr(text, "layer_types", None)
    if not layer_types or len(set(layer_types)) == 1:
        return 1
    n = len(layer_types)
    return next((period for period in range(1, n) if layer_types_shift_invariant(layer_types, period)), n)


def resolve_layer_partition(
    model: nn.Module, pp_size: int, pp_split: list[int] | None = None
) -> list[tuple[int, int]]:
    """The partition every PP consumer must share: manual ``pp_split``, else head-weighted default."""
    backbone = backbone_with_layers(model)
    if backbone is None:
        raise ValueError(
            f"{type(model).__name__} has no backbone with a decoder-layer list — it cannot be split "
            f"into pipeline stages."
        )
    num_layers = len(decoder_layers(backbone))
    config = getattr(model, "config", None)
    # A rebase-safe family selects per-layer inputs by the layer's own attributes, so its boundaries
    # need no period snapping — which would otherwise disable the head-weighted rebalance entirely
    # for aperiodic layer_types (GLM-5's 45-layer L,L,L,D pattern has no divisor period).
    period = 1 if resolve_pp_spec(backbone).LAYER_TYPES_REBASE_SAFE else layer_types_period(config)
    return compute_layer_partition(
        num_layers,
        pp_size,
        head_layer_equivalents=head_cost_layer_equivalents(config),
        boundary_period=period,
        pp_split=pp_split,
    )
