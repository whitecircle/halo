"""Apply a resolved ``moe_balancing`` mode to a live model: the config writes, the bias-state enable,
and the ``bias_update`` export contract.

The mode itself is resolved by :func:`~src.models.moe_balancing.resolve_balancing_mode`. Both entry
points are collective — every rank must call them. Which families have a checkpoint-exported
balancing slot is read off the EP layer-class registry rather than a roster list here.
"""

import logging
from typing import get_args

import torch.distributed as dist

from src.distributed.runtime import get_global_world_size, rank_consensus, reject_across_ranks
from src.models.loading.config_levels import (
    get_config_field,
    set_config_field,
    set_config_field_run_scoped,
)
from src.models.moe_balancing import (
    BIAS_UPDATE_MODES,
    NATIVE_BALANCING_BIAS_ADOPTED_ATTR,
    BalancingMode,
    accepts_bias_balancing,
    ep_severs_aux_loss,
    has_balancing_routers,
    honors_output_router_logits_config,
    is_transient_balancing_router,
    iter_balancing_routers,
    mark_router_logits_forced_off,
    native_balancing_hub_model_types,
)

logger = logging.getLogger(__name__)

# Resolved-mode precedence for :func:`agree_balancing_mode`, most informed first: a stage that found
# a bias-capable router has evidence a stage holding no MoE layer does not.
_BALANCING_MODE_PRECEDENCE: tuple[BalancingMode, ...] = ("bias_update", "bias_update_transient", "aux_loss", "none")
assert set(_BALANCING_MODE_PRECEDENCE) == set(get_args(BalancingMode)) - {"auto"}, "a balancing mode has no precedence"


def agree_balancing_mode(mode: BalancingMode) -> BalancingMode:
    """One balancing mode for the whole job. Collective — every rank must call it.

    ``resolve_balancing_mode`` introspects the live module tree, and under pipeline parallelism a
    stage without MoE layers resolves ``auto`` with no evidence: a split result sends stages down
    different branches of :func:`apply_balancing_strategy`, with different collectives and a raise on
    some ranks only. Identity outside distributed, or for an explicit mode.
    """
    if get_global_world_size() <= 1:
        return mode
    modes: list[BalancingMode | None] = [None] * dist.get_world_size()
    dist.all_gather_object(modes, mode)
    seen = {m for m in modes if m is not None}
    agreed = next((candidate for candidate in _BALANCING_MODE_PRECEDENCE if candidate in seen), mode)
    if agreed != mode:
        logger.info(
            f"moe_balancing resolved to {mode!r} on this pipeline stage and {agreed!r} elsewhere; "
            f"using {agreed!r} for the whole chain (a stage holding no MoE layer cannot decide it)."
        )
    return agreed


def _enable_router_bias_balancing(model) -> tuple[int, list[str]]:
    """Attach bias-update balancing state to every EP MoE layer exposing an ``enable_bias_balancing``
    hook; runs after EP patching, before FSDP wraps.

    Returns ``(enabled_count, hook_carrying_layer_class_names)`` so the caller can raise when nothing
    accepted the bias.
    """
    enabled = 0
    hook_layers: set[str] = set()
    for module in model.modules():
        hook = getattr(module, "enable_bias_balancing", None)
        if not callable(hook):
            continue
        hook_layers.add(type(module).__name__)
        if hook():  # True only for families that apply the bias
            enabled += 1
    if enabled:
        logger.info(f"Enabled bias-update balancing on {enabled} EP MoE layers.")
    return enabled, sorted(hook_layers)


def _enforce_bias_export_contract(model, mode: BalancingMode, is_moe: bool) -> None:
    """Reject a bias-update mode whose spelling misstates where the trained bias lands.

    ``bias_update`` requires the bias to reach every exported checkpoint, so a router left on the
    transient side-buffer raises here; ``bias_update_transient`` is the explicit opt-in for that, and
    raises in turn where every balancing router adopts a native slot.

    Collective — every rank must call it: both checks concern the whole model while ``transient`` is
    read off this rank's stage, so a PP stage holding no MoE layer would decide the opposite way.
    """
    transient = sorted({type(m).__name__ for m in iter_balancing_routers(model) if is_transient_balancing_router(m)})
    if mode == "bias_update":
        native_types = native_balancing_hub_model_types()
        reason = (
            (
                f"moe_balancing=bias_update would train a routing bias that NO EXPORT CARRIES on "
                f"{', '.join(transient)}: this architecture has no checkpoint slot for a selection "
                f"bias, so every exported checkpoint would silently serve WITHOUT the bias the model "
                f"trained with — vLLM/SGLang route on the raw gate scores and near-tied top-k picks "
                f"flip vs training. Set moe_balancing=bias_update_transient to accept trainer-only "
                f"balancing deliberately, or moe_balancing=none. bias_update stays available on model "
                f"types with an exportable slot ({', '.join(native_types)}), plus any router shipping "
                f"a native balancing_biases buffer (Zaya)."
            )
            if transient
            else None
        )
        reject_across_ranks(reason, "moe_balancing=bias_update export contract", ValueError)
        return
    if mode == "bias_update_transient":
        # Any rank's transient router makes the whole model's bias transient. Taken before the
        # `is_moe` test, never on the right of it: rank_consensus is a world all-reduce, so a rank
        # short-circuiting past it would leave every peer inside the collective.
        any_transient = rank_consensus(bool(transient))[1]
        if is_moe and not any_transient:
            raise ValueError(
                "moe_balancing=bias_update_transient, but every balancing router on this model "
                "carries its bias in checkpoint-exported state — nothing here is transient, and the "
                "exported bias serves exactly as trained. Use moe_balancing=bias_update."
            )
        if transient:
            logger.warning(
                f"moe_balancing=bias_update_transient — the routing bias on {', '.join(transient)} "
                f"steers TRAINING-TIME routing only: it is a plain attribute no checkpoint, export "
                f"or weight-sync carries (resume rides the router_balancing_biases.pt sidecar). "
                f"Every exported checkpoint serves without it — near-tied top-k picks flip between "
                f"trainer and server, and serving-time expert load reverts to the raw gate's."
            )


def _export_native_balancing_config_flags(model) -> None:
    """Mirror a materialized native slot's config gate (``_NATIVE_BALANCING_CONFIG_FLAG``, LFM-2's
    ``use_expert_bias``) into ``model.config``, so the export does not tell serving engines to skip
    the tensor the bias update trained."""
    cfg = getattr(model, "config", None)
    if cfg is None:
        return
    flags = {
        flag
        for module in model.modules()
        if getattr(module, NATIVE_BALANCING_BIAS_ADOPTED_ATTR, False)
        and (flag := getattr(type(module), "_NATIVE_BALANCING_CONFIG_FLAG", None))
    }
    for flag in sorted(flags):
        if not get_config_field(cfg, flag, False):
            logger.info(
                f"moe_balancing bias update — setting config.{flag}=True so serving engines load "
                f"and apply the materialized native balancing tensor."
            )
            set_config_field(cfg, flag, True, only_declared=False)


def apply_balancing_strategy(
    model, mode: BalancingMode, policy_gradient_loss: bool = False, is_moe: bool = False
) -> None:
    """Apply the side effects required by the resolved balancing mode.

    bias_update / bias_update_transient: zero ``router_aux_loss_coef``, force
                 ``output_router_logits=False``, create the bias state on supporting routers (raising
                 when nothing would accept it), then check the export contract via
                 :func:`_enforce_bias_export_contract`.
    aux_loss:    force ``output_router_logits=True``; leave it off when the model has no usable aux-loss
                 term or its EP wrappers sever the aux-loss path, where enabling it would crash.
    none:        leave both untouched.

    Under ``policy_gradient_loss`` (GRPO) the loss never adds the router aux loss, so ``aux_loss`` is
    inert. Every leave-off decision stamps the config via :func:`mark_router_logits_forced_off`, so
    ``MoEMetricsCallback`` respects it instead of re-enabling router logits at train begin.
    """
    cfg = getattr(model, "config", None)
    if cfg is None or mode == "none":
        return

    if mode == "aux_loss" and policy_gradient_loss:
        mark_router_logits_forced_off(cfg)
        logger.warning(
            "moe_balancing=aux_loss has NO EFFECT under a policy-gradient (GRPO) trainer: the loss "
            "is computed from per-token log-probs and never adds the model's router aux loss, so "
            "expert load is left unbalanced and no moe/* metrics are produced. Use "
            "moe_balancing=bias_update (aux-loss-free DeepSeek-V3 bias update — works under EP and "
            "emits moe/* load metrics via RouterBiasBalancingCallback) for real balancing, or "
            "moe_balancing=none to silence this."
        )
        return

    if mode in BIAS_UPDATE_MODES:
        mark_router_logits_forced_off(cfg)
        coef = get_config_field(cfg, "router_aux_loss_coef", 0)
        if coef not in (0, 0.0, None):
            logger.info(
                f"moe_balancing={mode} — overriding router_aux_loss_coef ({coef!r}) to 0 to avoid double-balancing."
            )
            # float: HF strict-dataclass configs (e.g. GptOss) reject int. Run-scoped: the zero
            # configures this run only, and the export restores the hub coefficient.
            set_config_field_run_scoped(cfg, "router_aux_loss_coef", 0.0)
        # The EP bias path bypasses the hooked router module: aux loss would IndexError an empty tuple.
        if get_config_field(cfg, "output_router_logits", False):
            logger.info(
                f"moe_balancing={mode} — setting output_router_logits=False "
                "(EP bias path bypasses router-logit recording)."
            )
            set_config_field_run_scoped(cfg, "output_router_logits", False)
        # The export contract runs after this enable: adoption can fall back to the transient
        # side-buffer at runtime (a DTensor-wrapped slot, an upstream rename), so it reads the
        # post-enable state, not the class predicate.
        enabled, hook_layers = _enable_router_bias_balancing(model)
        # Engine-side serving gap declared by the layer class (Laguna under vLLM 0.26.0): the export
        # carries the trained bias but that engine's loader drops the key. Warned, not rejected:
        # training and the export are correct.
        dropped_by = {
            engine
            for module in model.modules()
            if (engine := getattr(type(module), "_SERVED_BALANCING_BIAS_DROPPED_BY", None))
        }
        if enabled and dropped_by:
            logger.warning(
                f"moe_balancing={mode}: this family's exported routing bias is DROPPED by "
                f"{sorted(dropped_by)} at serve time (the engine loader skips the key), so a copy "
                f"served there routes on the PRETRAINED bias. A transformers reload routes as "
                f"trained. See agent-docs/models/laguna.md."
            )
        # World-wide, not stage-local: under PP a stage holding no MoE layer accepts nothing by
        # construction, and its raise would leave the stages that did accept the bias in their next
        # collective.
        if not rank_consensus(enabled > 0 or has_balancing_routers(model))[1]:
            model_type = getattr(cfg, "model_type", "unknown")
            detail = (
                f"its EP MoE layers {hook_layers} do not support bias-update balancing "
                "(_supports_bias_balancing=False — routing happens inside the HF gate or outside the "
                "EP wrapper)"
                if hook_layers
                else (
                    "it has no EP MoE wrappers or native balancing_biases routers"
                    if is_moe
                    else "it is a dense model — there are no experts to balance"
                )
            )
            raise ValueError(
                f"moe_balancing={mode} would silently balance NOTHING on this model "
                f"(model_type={model_type!r}): {detail}, and no router ships a native "
                f"balancing_biases buffer (Zaya). Prefer moe_balancing: auto, which resolves the one "
                f"strategy this model supports — naming a mode by family is unreliable here, because "
                f"whether aux_loss works depends on the loaded CLASS, not the family: a wrapper whose "
                f"forward never declares output_router_logits (the multimodal Qwen3.5/3.6 checkpoints) "
                f"rejects aux_loss, while its text-only sibling accepts it. moe_balancing: none trains "
                f"unbalanced, which for a large expert count is a real cost, not a neutral default."
            )
        _enforce_bias_export_contract(model, mode, is_moe=is_moe)
        _export_native_balancing_config_flags(model)
        return

    if mode == "aux_loss":
        if ep_severs_aux_loss(model):
            # Enabling output_router_logits would index the never-populated tuple: a crash, not balancing.
            mark_router_logits_forced_off(cfg)
            logger.warning(
                "moe_balancing=aux_loss has NO EFFECT for this model under EP: its EP MoE wrappers "
                "re-derive routing internally and never record router logits, so the HF aux-loss "
                "path is severed (enabling output_router_logits would crash on the empty "
                "router_logits tuple). Leaving output_router_logits off. Use "
                "moe_balancing=bias_update (or auto) for real balancing."
            )
            return
        coef = get_config_field(cfg, "router_aux_loss_coef")
        if not coef or coef <= 0:
            # Without a usable aux-loss term, TRL would crash on the never-populated ``outputs.aux_loss``.
            mark_router_logits_forced_off(cfg)
            logger.warning(
                "moe_balancing=aux_loss but model.config.router_aux_loss_coef is "
                f"{coef!r}; this model has no usable aux-loss term (e.g. aux-loss-free routers "
                "like GLM-4 MoE Lite's noaux_tc). Leaving output_router_logits off to avoid a TRL "
                "'outputs.aux_loss' crash. Use moe_balancing=bias_update for real balancing. "
                "Setting router_aux_loss_coef in model_init_kwargs only works on a config that "
                "already declares the field — where it is absent the override is rejected at load."
            )
            return
        if not honors_output_router_logits_config(model):
            # bias_update is a remedy only where something would accept the bias; without an
            # acceptor it raises in turn, so name the knobs that create one instead.
            if accepts_bias_balancing(model) or has_balancing_routers(model):
                remedy = (
                    "Use moe_balancing=bias_update (aux-loss-free, works under EP and PP), or "
                    "moe_balancing=none to train unbalanced deliberately."
                )
            else:
                remedy = (
                    "moe_balancing=bias_update is no way out on this model AS LOADED — nothing here "
                    "carries the bias (no EP MoE wrapper, no native balancing_biases router), so it "
                    "raises in turn. The wrappers are what balancing needs first: launch under "
                    "torchrun with use_grouped_gemm: true, or with expert parallelism "
                    "(--expert_parallel_size>1); moe_balancing=auto then resolves to bias_update by "
                    "itself. Otherwise set moe_balancing=none to train unbalanced deliberately — a "
                    "real cost at this expert count, not a neutral default."
                )
            raise ValueError(
                f"moe_balancing=aux_loss cannot balance {type(model).__name__}: its forward does not "
                f"take output_router_logits, so it never consults the config flag this mode sets "
                f"(HF's config fallback lives on that parameter). The flag would still switch router-"
                f"logit RECORDING on — a [tokens, num_experts] plane per MoE layer, every forward — "
                f"while router_aux_loss_coef={get_config_field(cfg, 'router_aux_loss_coef')!r} never "
                f"reaches the loss. Multimodal wrappers are the case that bites: "
                f"Qwen3_5MoeForConditionalGeneration reads the flag from kwargs only, while its "
                f"text-only Qwen3_5MoeForCausalLM sibling honours the config. {remedy}"
            )
        if not get_config_field(cfg, "output_router_logits", False):
            logger.info(
                "moe_balancing=aux_loss — setting output_router_logits=True "
                "(required for HF MoE forwards to add aux loss to the total loss). Run-scoped: "
                "exported, the flag would make every plain-transformers forward of the artifact "
                "materialize the router-logit plane and inflate its eval loss."
            )
            set_config_field_run_scoped(cfg, "output_router_logits", True)
