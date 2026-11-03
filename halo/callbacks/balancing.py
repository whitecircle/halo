"""Run-level MoE balancing: resolve and agree the run's ``moe_balancing`` mode, apply it, and step the bias."""

import logging

import torch
import torch.distributed as dist

from halo.callbacks.moe_metrics import MoELoadMetricsBase, compute_moe_load_metrics
from halo.distributed.parallelism_config import ParallelismConfig
from halo.models.config_access import (
    config_model_types,
    get_config_field,
    set_config_field,
    set_config_field_run_scoped,
)
from halo.models.moe.balancing import (
    BALANCING_MODE_PRECEDENCE,
    BIAS_UPDATE_MODES,
    NATIVE_BALANCING_BIAS_ADOPTED_ATTR,
    BalancingMode,
    accepts_bias_balancing,
    has_balancing_routers,
    is_transient_balancing_router,
    iter_balancing_routers,
    resolve_balancing_mode,
    severs_aux_loss,
)
from halo.models.moe.config_fields import resolve_moe_routing
from halo.models.moe.family_specs import MOE_FAMILY_SPECS
from halo.models.moe.router_aux_gradient import install_router_aux_gradient
from halo.models.moe.router_logits import honors_router_logits_flag, mark_router_logits_forced_off
from halo.runtime.consensus import rank_consensus, reject_across_ranks
from halo.runtime.ranks import get_global_world_size, is_global_main_process
from halo.weight_sync.registry import resolve_weight_sync_client, rollout_backends

logger = logging.getLogger(__name__)

# Of the uniform scale 1/num_experts: a sign step above the warn fraction oscillates; the warning quotes the other.
_TRANSIENT_STEP_WARN_FRACTION_OF_UNIFORM = 0.25
_TRANSIENT_STEP_SUGGESTED_FRACTION_OF_UNIFORM = 0.1
# Far below the routed slots' bias range, so the discard slot never wins a top-k comparison it would not win unbiased.
_DISCARD_SLOT_BIAS = -1.0


def _enforce_bias_export_contract(model, mode: BalancingMode, is_moe: bool) -> None:
    """COLLECTIVE, since ``transient`` is read off this rank's stage only. Reject ``bias_update`` on a router left on
    the transient side-buffer, and ``bias_update_transient`` where every router adopts a native slot."""
    transient = sorted({type(m).__name__ for m in iter_balancing_routers(model) if is_transient_balancing_router(m)})
    if mode == "bias_update":
        native_types = sorted(mt for spec in MOE_FAMILY_SPECS if spec.balancing_slot for mt in spec.model_types)
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
        # Before `is_moe`: rank_consensus is a world all-reduce, and a rank short-circuiting past it strands its peers.
        any_transient = rank_consensus(bool(transient)).any_ok
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


def _nothing_balanced_detail(hook_layers: list[str], is_moe: bool) -> str:
    if hook_layers:
        return (
            f"its EP MoE layers {hook_layers} do not support bias-update balancing "
            "(SUPPORTS_BIAS_BALANCING=False — routing happens inside the HF gate or outside the "
            "EP layer)"
        )
    if is_moe:
        return "it has no EP layers or native balancing_biases routers"
    return "it is a dense model — there are no experts to balance"


def _apply_bias_update(model, cfg, mode: BalancingMode, is_moe: bool) -> None:
    """Zero the aux coefficient and router logits, enable the bias on every accepting EP layer (after EP patching,
    before FSDP wraps), then check that something balances and where the bias exports."""
    mark_router_logits_forced_off(cfg)
    coef = get_config_field(cfg, "router_aux_loss_coef", 0)
    if coef not in (0, 0.0, None):
        logger.info(
            f"moe_balancing={mode} — overriding router_aux_loss_coef ({coef!r}) to 0 to avoid double-balancing."
        )
        # float: HF strict-dataclass configs (GptOss) reject int; the export restores the hub value.
        set_config_field_run_scoped(cfg, "router_aux_loss_coef", 0.0)
    # The EP bias path bypasses the hooked router module: aux loss would IndexError an empty tuple.
    if get_config_field(cfg, "output_router_logits", False):
        logger.info(
            f"moe_balancing={mode} — setting output_router_logits=False "
            "(EP bias path bypasses router-logit recording)."
        )
        set_config_field_run_scoped(cfg, "output_router_logits", False)
    enabled = 0
    hook_layers: set[str] = set()
    for module in model.modules():
        hook = getattr(module, "enable_bias_balancing", None)
        if callable(hook):
            hook_layers.add(type(module).__name__)
            enabled += bool(hook())
    if enabled:
        logger.info(f"Enabled bias-update balancing on {enabled} EP MoE layers.")
    # An engine's loader can drop the family's exported key (vLLM's Laguna); training and export are correct, so warn.
    model_types = set(config_model_types(cfg))
    dropped_by = sorted(
        client.ENGINE_RELEASE
        for client in map(resolve_weight_sync_client, rollout_backends())
        if client.DROPPED_BALANCING_BIAS_MODEL_TYPES & model_types
    )
    if enabled and dropped_by:
        logger.warning(
            f"moe_balancing={mode}: this family's exported routing bias is DROPPED by "
            f"{dropped_by} at serve time (the engine loader skips the key), so a copy "
            f"served there never routes on the TRAINED bias. A transformers reload routes as "
            f"trained. See agent-docs/models/laguna.md."
        )
    # World-wide: a PP stage holding no MoE layer accepts nothing by construction.
    if not rank_consensus(enabled > 0 or has_balancing_routers(model)).any_ok:
        model_type = getattr(cfg, "model_type", "unknown")
        detail = _nothing_balanced_detail(sorted(hook_layers), is_moe)
        raise ValueError(
            f"moe_balancing={mode} would silently balance NOTHING on this model "
            f"(model_type={model_type!r}): {detail}, and no router ships a native "
            f"balancing_biases buffer (Zaya). Prefer moe_balancing: auto, which resolves the one "
            f"strategy this model supports — naming a mode by family is unreliable here, because "
            f"whether aux_loss works depends on the loaded CLASS, not the family: a wrapper whose "
            f"forward never declares output_router_logits rejects aux_loss, while a sibling class "
            f"that declares it accepts it. moe_balancing: none trains unbalanced, which for a large "
            f"expert count is a real cost, not a neutral default."
        )
    # After the enable: adoption can fall back to the side-buffer at runtime (a DTensor slot, an upstream rename).
    _enforce_bias_export_contract(model, mode, is_moe=is_moe)
    # A materialized config-gated slot (LFM-2 use_expert_bias) must flip its flag, or engines skip the trained tensor.
    flags = {
        flag
        for module in model.modules()
        if vars(module).get(NATIVE_BALANCING_BIAS_ADOPTED_ATTR, False)
        and (flag := type(module).FAMILY.balancing_slot.config_flag)
    }
    for flag in sorted(flags):
        if not get_config_field(cfg, flag, False):
            logger.info(
                f"moe_balancing bias update — setting config.{flag}=True so serving engines load "
                f"and apply the materialized native balancing tensor."
            )
            set_config_field(cfg, flag, True, only_declared=False)


def _apply_aux_loss(model, cfg, policy_gradient_loss: bool) -> None:
    """Turn router logits on and route the aux gradient through reentrant checkpointing, unless the aux term is inert
    (policy-gradient loss), severed by EP layers or unusable; each leave-off stamps the config so
    ``MoEMetricsCallback`` does not re-enable it."""
    if policy_gradient_loss:
        mark_router_logits_forced_off(cfg)
        logger.warning(
            "moe_balancing=aux_loss has NO EFFECT under a policy-gradient (GRPO) trainer: the loss "
            "is computed from per-token log-probs and never adds the model's router aux loss, so "
            "expert load is left unbalanced and no moe/* metrics are produced. Use "
            "moe_balancing=bias_update (aux-loss-free DeepSeek-V3 bias update — works under EP and "
            "emits moe/* load metrics via BalancingBiasCallback) for real balancing, or "
            "moe_balancing=none to silence this."
        )
        return
    if severs_aux_loss(model):
        # Enabling output_router_logits would index the never-populated tuple and crash.
        mark_router_logits_forced_off(cfg)
        logger.warning(
            "moe_balancing=aux_loss has NO EFFECT for this model under EP: its EP layers "
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
    if not honors_router_logits_flag(model):
        # bias_update is a remedy only where something accepts the bias; else it raises too, so name what creates one.
        if accepts_bias_balancing(model) or has_balancing_routers(model):
            remedy = (
                "Use moe_balancing=bias_update (aux-loss-free, works under EP and PP), or "
                "moe_balancing=none to train unbalanced deliberately."
            )
        else:
            remedy = (
                "moe_balancing=bias_update is no way out on this model AS LOADED — nothing here "
                "carries the bias (no EP layer, no native balancing_biases router), so it "
                "raises in turn. The EP layers are what balancing needs first: launch under "
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
            f"reaches the loss. {remedy}"
        )
    if not get_config_field(cfg, "output_router_logits", False):
        logger.info(
            "moe_balancing=aux_loss — setting output_router_logits=True "
            "(required for HF MoE forwards to add aux loss to the total loss). Run-scoped: "
            "exported, the flag would make every plain-transformers forward of the artifact "
            "materialize the router-logit plane and inflate its eval loss."
        )
        set_config_field_run_scoped(cfg, "output_router_logits", True)
    # MoE checkpointing is reentrant (outside PP); its no_grad pass leaves a layer's aux term with no router gradient.
    if not install_router_aux_gradient(model):
        logger.warning(
            f"moe_balancing=aux_loss: {type(model).__name__} declares no transformers "
            f"router_logits capture, so under gradient checkpointing a checkpointed layer's "
            f"router aux loss reaches the logged loss but adds no router gradient."
        )


def agree_balancing_mode(mode: BalancingMode) -> BalancingMode:
    """COLLECTIVE. One mode for the job, by :data:`BALANCING_MODE_PRECEDENCE`: under PP a stage without MoE layers
    resolves ``auto`` with no evidence, and a split verdict sends stages down different branches."""
    if get_global_world_size() <= 1:
        return mode
    modes: list[BalancingMode | None] = [None] * dist.get_world_size()
    dist.all_gather_object(modes, mode)
    seen = {m for m in modes if m is not None}
    agreed = next((candidate for candidate in BALANCING_MODE_PRECEDENCE if candidate in seen), mode)
    if agreed != mode:
        logger.info(
            f"moe_balancing resolved to {mode!r} on this pipeline stage and {agreed!r} elsewhere; "
            f"using {agreed!r} for the whole chain (a stage holding no MoE layer cannot decide it)."
        )
    return agreed


def apply_balancing_mode(
    model, mode: BalancingMode, *, policy_gradient_loss: bool = False, is_moe: bool = False, pp_size: int = 1
) -> None:
    """COLLECTIVE. Apply a resolved mode's side effects; ``policy_gradient_loss`` (GRPO) never adds the aux loss, so
    ``aux_loss`` is inert there."""
    if mode == "bias_update_transient" and pp_size > 1:
        raise ValueError(
            f"moe_balancing=bias_update_transient is untested under pipeline parallelism (pp_size={pp_size}): "
            "the transient side-buffer is a plain attribute the stage split and the PP resume path never "
            "carry. Use moe_balancing=bias_update on a family with an exported slot, or moe_balancing=none."
        )
    cfg = getattr(model, "config", None)
    if cfg is None or mode == "none":
        return
    if mode in BIAS_UPDATE_MODES:
        _apply_bias_update(model, cfg, mode, is_moe)
    elif mode == "aux_loss":
        _apply_aux_loss(model, cfg, policy_gradient_loss)


def configure_moe_balancing(
    args,
    model,
    parallelism_config: ParallelismConfig,
    *,
    policy_gradient_loss: bool = False,
    syncs_to_external_generator: bool = False,
) -> BalancingMode:
    """COLLECTIVE. Resolve, agree and apply ``args.moe_balancing``. ``syncs_to_external_generator`` downgrades both
    bias modes to ``none``: the sync carries no bias that moves, so a trained bias would split routing."""
    routing = resolve_moe_routing(model)
    mode = agree_balancing_mode(resolve_balancing_mode(args.moe_balancing, model, routing.routes_between_experts))
    if syncs_to_external_generator and mode in BIAS_UPDATE_MODES:
        logger.warning(
            f"moe_balancing={mode} is incompatible with on-policy weight-sync RL: the rollout-engine weight "
            "sync carries no routing bias that moves (an adopted native slot is a buffer), so "
            "trainer↔generator routing diverges. Downgrading to moe_balancing=none — "
            "with aux_loss also inert under a policy-gradient loss, THIS RUN HAS NO ROUTER BALANCING "
            "AT ALL and router_balancing_rate is unreachable. Families that only balance via "
            "bias_update (Zaya, DeepSeek-V4) train with unbalanced experts here."
        )
        mode = "none"
    apply_balancing_mode(
        model,
        mode,
        policy_gradient_loss=policy_gradient_loss,
        is_moe=routing.routes_between_experts,
        pp_size=parallelism_config.pp_size,
    )
    return mode


def _expert_load_counter(router) -> torch.Tensor:
    """Lazily created as fp32 zeros of the bias's width and device."""
    counter = router.expert_load_counter
    if counter is None:
        biases = router.balancing_biases
        counter = torch.zeros(biases.numel(), dtype=torch.float32, device=biases.device)
        router.expert_load_counter = counter
    return counter


class BalancingBiasCallback(MoELoadMetricsBase):
    """DeepSeek-V3 bias update after every optimizer step: ``b_i <- b_i + update_rate * sign(mean - count_i)``. The
    bias adds to detached top-k scores, so gradients flow through the unbiased gate. ``expert_load_counter`` is a plain
    attribute, unseen by ``state_dict()`` and the PP lazy loader. The ``moe/*`` summary is a periodic host sync."""

    def __init__(self, update_rate: float = 1e-3, exclude_last_slot: bool = False, log_every_n_steps: int = 1):
        if update_rate <= 0:
            raise ValueError(f"update_rate must be > 0, got {update_rate}")
        super().__init__(exclude_last_slot=exclude_last_slot, log_every_n_steps=log_every_n_steps)
        self.update_rate = float(update_rate)

    def on_train_begin(self, args, state, control, **kwargs):
        model = kwargs.get("model")
        if model is None:
            return
        routers = list(iter_balancing_routers(model))
        if not routers:
            if is_global_main_process():
                logger.warning(
                    "BalancingBiasCallback enabled but no router with a "
                    "balancing_biases buffer was found in the model — callback is a no-op."
                )
            return
        for router in routers:
            # Arms recording on routers whose bias exists regardless of balancing (Zaya's native buffer).
            router.balancing_active = True
        # Transient only: a native slot sits in the family's own score space, where GPT-OSS logits want a larger rate.
        transient_widths = {r.balancing_biases.numel() for r in routers if is_transient_balancing_router(r)}
        if transient_widths:
            num_experts = max(transient_widths) - (1 if self.exclude_last_slot else 0)
            step_fraction_of_uniform = self.update_rate * num_experts
            if step_fraction_of_uniform > _TRANSIENT_STEP_WARN_FRACTION_OF_UNIFORM and is_global_main_process():
                suggested = _TRANSIENT_STEP_SUGGESTED_FRACTION_OF_UNIFORM / num_experts
                logger.warning(
                    f"router_balancing_rate={self.update_rate} is {step_fraction_of_uniform:.2f}x "
                    f"the uniform routing probability at {num_experts} experts — the transient bias "
                    f"step overshoots and load will oscillate. Scale it down in proportion "
                    f"(~{suggested:.1e} keeps one step at "
                    f"{_TRANSIENT_STEP_SUGGESTED_FRACTION_OF_UNIFORM:.0%} of uniform); watch moe/load_cv."
                )
        if is_global_main_process():
            logger.info(
                f"BalancingBiasCallback active: {len(routers)} routers, "
                f"update_rate={self.update_rate}, exclude_last_slot={self.exclude_last_slot}"
            )

    @torch.no_grad()
    def on_step_end(self, args, state, control, **kwargs):
        model = kwargs.get("model")
        if model is None:
            return
        routers = list(iter_balancing_routers(model))
        if not routers:
            return

        device = routers[0].balancing_biases.device
        widths = {r.balancing_biases.numel() for r in routers}
        if len(widths) > 1:
            raise ValueError(
                f"BalancingBiasCallback requires one expert-slot width across routers, got "
                f"{sorted(widths)}. The per-slot bias update and its all-reduce assume a single width."
            )
        counters = torch.stack([_expert_load_counter(r).to(device) for r in routers], dim=0)
        if get_global_world_size() > 1:
            dist.all_reduce(counters, op=dist.ReduceOp.SUM, group=self.reduce_group)

        # sign(0-0)=0 makes the update a no-op, so an all-zero counter is reported.
        if self._first_log and counters.sum() == 0 and is_global_main_process():
            logger.warning(
                "BalancingBiasCallback: all-reduced expert-load counters are entirely ZERO at "
                "the first bias update — no routed tokens were recorded, so bias updates are no-ops. "
                "The forward path is not reaching the balancing-enabled routers' load recording."
            )

        # The summary ends in a host sync only on_log consumes, so it runs on logged steps only (a rank-uniform gate).
        if self._first_log or state.global_step % self.log_every_n_steps == 0:
            self._pending = compute_moe_load_metrics(counters, exclude_last_slot=self.exclude_last_slot)

        real_counts = counters[:, :-1] if self.exclude_last_slot else counters
        mean_real = real_counts.mean(dim=-1, keepdim=True)
        update = self.update_rate * torch.sign(mean_real - real_counts)
        # strict: a router list that changed under the callback would leave the tail on a frozen bias.
        for router, upd in zip(routers, update, strict=True):
            if self.exclude_last_slot:
                router.balancing_biases[:-1].add_(upd)
                router.balancing_biases[-1] = _DISCARD_SLOT_BIAS
            else:
                router.balancing_biases.add_(upd)
            router.expert_load_counter.zero_()

        if self._first_log:
            if is_global_main_process():
                logger.info(
                    f"BalancingBiasCallback: first update applied to {len(routers)} routers "
                    f"(reduce-group per-slot totals, router 0 — under TP/ETP partners recount shared "
                    f"tokens, a uniform scale the sign update is invariant to): {counters[0].tolist()} | "
                    f"metrics: {self._pending}"
                )
            self._first_log = False
