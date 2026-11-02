"""Liger Kernel dispatch: resolve the applier per ``model_type`` (the toolkit spec for it or its text sub-config's,
then Liger's registry in the same order) and apply it with the per-family defaults and the parallelism safety rules.

Cross-entropy is always the toolkit's scoped patch and an upstream applier's ``layer_norm`` stays off: upstream installs
both process-wide (``torch.nn.functional.cross_entropy``, and before the model exists ``torch.nn.LayerNorm``)."""

from __future__ import annotations

import inspect
import logging
from collections.abc import Callable

import torch
from liger_kernel.transformers.auto_model import MODEL_TYPE_TO_APPLY_LIGER_FN
from transformers import AutoConfig

from halo.kernels.jit_cache import pin_liger_impls
from halo.model_loading.liger.applier import LigerApplier, build_liger_appliers
from halo.model_loading.liger.cross_entropy import patch_loss_utils_cross_entropy
from halo.model_loading.liger.family_specs import LIGER_FAMILY_SPECS
from halo.models.attention.prefix_lm import has_prefix_lm_attention
from halo.models.config_access import set_config_field_run_scoped, text_config
from halo.models.moe.config_fields import has_routed_experts
from halo.models.moe.family_specs import wraps_experts
from halo.runtime.ranks import is_global_main_process

logger = logging.getLogger(__name__)

# The effective Liger config the classes were patched with, stamped run-scoped on ``model.config`` for the
# post-load TRL flag reconciliation to read back.
LIGER_APPLIED_CONFIG_ATTR = "_halo_liger_applied_config"
# Under CP, whether the run resolves ``fused_linear_cross_entropy`` on for the family: the Ulysses wrapper, which
# scores the loss outside the model's forward, then runs the fused kernel over its chunk itself.
LIGER_CP_FUSED_LOSS_ATTR = "_halo_liger_cp_fused_loss"

# Flags whose applier-declared ``False`` default states that Liger's kernel cannot serve the family
# (a partial, mrope or YARN rotary; GptOss's clamped experts) — honored over the shared default-on.
_APPLIER_DEFAULT_OFF_FLAGS = ("rope", "swiglu", "geglu")

# Every family the toolkit patches. A family registered into Liger's own dict instead would resolve on the
# upstream branch and miss the rules this module derives from this registry.
_TOOLKIT_LIGER_APPLIERS = build_liger_appliers(LIGER_FAMILY_SPECS)

# Model types whose fused GLU survives an EP layer: a toolkit spec names the dense and shared-expert
# MLPs, which every EP layer adopts unchanged; a spec naming none has at most upstream's routed-expert swap.
_TOOLKIT_GLU_SURVIVES_EP = frozenset(
    model_type for spec in LIGER_FAMILY_SPECS if spec.glu_mlp for model_type in spec.model_types
)

# Per-family loss defaults, read off each toolkit applier's own signature so the two stay in sync.
_PER_MODEL_DEFAULTS = {
    model_type: {
        name: param.default
        for name, param in inspect.signature(applier).parameters.items()
        if name in ("cross_entropy", "fused_linear_cross_entropy") and param.default is not inspect.Parameter.empty
    }
    for model_type, applier in _TOOLKIT_LIGER_APPLIERS.items()
}

# Applier lookup order: a toolkit applier wins over Liger's own entry, for either spelling of the family.
_APPLIER_REGISTRIES = (_TOOLKIT_LIGER_APPLIERS, MODEL_TYPE_TO_APPLY_LIGER_FN)

# The shared Liger defaults; per-family and user overrides layer on top of a fresh copy.
_LIGER_DEFAULTS = {
    "rope": True,
    "cross_entropy": True,
    "fused_linear_cross_entropy": False,
    "rms_norm": True,
    "swiglu": True,
    "geglu": True,
}


def _liger_model_types(model_config) -> tuple[str | None, str | None]:
    """``(model_type, text-config fallback)``: the two spellings an applier may be registered under, since a
    multimodal wrapper carries its decoder classes on its text sub-config."""
    model_type = getattr(model_config, "model_type", None)
    text_type = getattr(text_config(model_config), "model_type", None)
    return model_type, (text_type if text_type and text_type != model_type else None)


def _resolve_applier(model_type: str | None, fallback_type: str | None) -> tuple[str | None, Callable | None]:
    """``(model_type, applier)`` Liger resolves to: the toolkit's applier for either spelling, else upstream's.

    The toolkit owns a family through its multimodal wrappers too: an upstream wrapper applier (Gemma 4's)
    runs its own text-tower roles, which would replace the ones the tower's spec takes over.
    """
    for registry in _APPLIER_REGISTRIES:
        for candidate in (model_type, fallback_type):
            if candidate in registry:
                return candidate, registry[candidate]
    return None, None


def _toolkit_applier(model_type: str | None, applier) -> LigerApplier | None:
    """``applier`` as the toolkit's :class:`LigerApplier`, or ``None`` where liger-kernel's own resolved.

    Which flags a toolkit applier hands upstream is read off its spec, so a toolkit entry that is not a
    ``LigerApplier`` is refused rather than read as withholding nothing.
    """
    if model_type not in _TOOLKIT_LIGER_APPLIERS:
        return None
    if not isinstance(applier, LigerApplier):
        raise TypeError(
            f"The toolkit Liger applier for {model_type} is a {type(applier).__name__}, not a LigerApplier, "
            f"so which flags it hands upstream is unknown. Register the family through a LigerFamilySpec."
        )
    return applier


def liger_ep_disables_fused_glu(needs_ep_layers: bool, model_config) -> bool:
    """The ``has_ep_wrapped_experts`` argument both Liger application sites pass.

    True only when an EP layer replaces every module the resolved applier's GLU patch swaps: upstream's
    MoE appliers swap the routed-experts class the EP layer replaces wholesale, while a toolkit spec names
    the dense and shared-expert MLPs every EP layer adopts unchanged.
    """
    if not wraps_experts(model_config, ep_layers_requested=needs_ep_layers):
        return False
    return not any(candidate in _TOOLKIT_GLU_SURVIVES_EP for candidate in _liger_model_types(model_config))


def liger_routed_expert_overrides(needs_ep_layers: bool, model_config) -> dict[str, str]:
    """``{"swiglu": reason}`` where upstream Liger's ``swiglu`` could replace routed experts Halo does not wrap.

    Upstream's MoE appliers swap routed experts for ``LigerExperts`` under ``swiglu``, whose Blackwell
    input-gradient fix is not validated here. The rule keys on the config having experts, and the flag goes
    whole: any dense, shared-expert or vision SwiGLU upstream fuses under it runs eager too.
    """
    if not has_routed_experts(model_config) or wraps_experts(model_config, ep_layers_requested=needs_ep_layers):
        return {}
    model_type, applier = _resolve_applier(*_liger_model_types(model_config))
    if applier is None:
        return {}
    toolkit = _toolkit_applier(model_type, applier)
    # liger-kernel's own applier takes every flag it declares; a toolkit one only those it delegates.
    hands_swiglu_upstream = (
        "swiglu" in inspect.signature(applier).parameters if toolkit is None else toolkit.hands_upstream("swiglu")
    )
    if not hands_swiglu_upstream:
        return {}
    # An applier declaring `swiglu=False` has no SwiGLU patch for the family (GptOss's clamped experts).
    if inspect.signature(applier).parameters["swiglu"].default is False:
        return {}
    return {
        "swiglu": (
            f"{model_type} has routed experts Halo does not wrap here and upstream liger-kernel owns its "
            f"swiglu, a flag under which upstream's MoE appliers can replace routed experts with LigerExperts "
            f"(its Blackwell input-gradient fix is not validated here); the whole flag goes, so any dense, "
            f"shared-expert or vision SwiGLU upstream fuses under it runs eager too"
        )
    }


def liger_prefix_lm_overrides(model_config) -> dict[str, str]:
    """``{"fused_linear_cross_entropy": reason}`` on a prefix-LM model (:func:`has_prefix_lm_attention`): upstream's
    fused forward replaces the wrapper's, so the decoder never builds the prefix-LM mask from ``token_type_ids``."""
    if not has_prefix_lm_attention(model_config):
        return {}
    return {
        "fused_linear_cross_entropy": (
            f"{model_config.model_type} is a prefix LM, and upstream liger-kernel's fused forward replaces its own, "
            f"so the decoder would never build the prefix-LM mask from token_type_ids"
        )
    }


def liger_parallelism_overrides(
    *,
    has_ep_wrapped_experts: bool = False,
    tp_size: int = 1,
    cp_size: int = 1,
    pp_size: int = 1,
) -> dict[str, str]:
    """Liger kernels a parallelism axis makes incorrect or inert, mapped to the reason."""
    overrides: dict[str, str] = {}
    if has_ep_wrapped_experts:
        # EP layers replace `.experts` with a DeepEP-aware path; Liger swaps a single FFN module.
        reason = "EP layers replace the expert FFN Liger would swap"
        overrides["swiglu"] = overrides["geglu"] = reason
    if tp_size > 1:
        reason = f"TP (tp_size={tp_size}) shards lm_head, making the fused softmax a partial-vocab slice"
        overrides["cross_entropy"] = overrides["fused_linear_cross_entropy"] = reason
    external_loss = {"CP": cp_size, "PP": pp_size}
    engaged = [f"{name} (size {size})" for name, size in external_loss.items() if size > 1]
    if engaged:
        # CP's wrapper and PP's last stage compute the loss from logits and pass no labels, so Liger's
        # `skip_logits` gate never fires; CE goes too, since the scoped CE patch reaches only the
        # model's own loss.
        reason = f"{' and '.join(engaged)} computes the loss outside the model's forward, so the kernel never fires"
        overrides["cross_entropy"] = overrides["fused_linear_cross_entropy"] = reason
    return overrides


def _force_off(user_config: dict, forced_off: dict[str, str], *, yields_to_request: tuple[str, ...]) -> dict:
    """``user_config`` with every ``forced_off`` kernel off, bar an explicit request for one that yields."""
    result = dict(user_config)
    for key, reason in forced_off.items():
        if key in yields_to_request and key in user_config:
            continue
        if result.get(key) and is_global_main_process():
            logger.warning(f"Liger {key} was explicitly enabled but {reason}; forcing it off.")
        result[key] = False
    return result


def sanitize_liger_config(
    user_config: dict,
    *,
    needs_ep_layers: bool,
    model_config,
    tp_size: int = 1,
    cp_size: int = 1,
    pp_size: int = 1,
) -> dict:
    """``user_config`` with every kernel the safety rules force off turned off.

    Both Liger applications (model load, and the trainer's before HF Trainer re-applies) call this, since a
    rule present at one site only is undone by whichever runs second. The parallelism rules yield to an
    explicit ``swiglu``/``geglu`` request, which they make inert rather than wrong; the routed-experts
    and prefix-LM rules do not. It also pins the Liger implementations (:func:`pin_liger_impls`), since a
    model passed to the trainer reaches only the second site.
    """
    pin_liger_impls()
    forced_off = liger_parallelism_overrides(
        has_ep_wrapped_experts=liger_ep_disables_fused_glu(needs_ep_layers, model_config),
        tp_size=tp_size,
        cp_size=cp_size,
        pp_size=pp_size,
    )
    config = _force_off(user_config, forced_off, yields_to_request=("swiglu", "geglu"))
    if forced_off and is_global_main_process():
        logger.info(f"Liger disabled by parallelism: {', '.join(sorted(forced_off))}")
    routed_expert_overrides = liger_routed_expert_overrides(needs_ep_layers, model_config)
    config = _force_off(config, routed_expert_overrides, yields_to_request=())
    if is_global_main_process():
        for key, reason in routed_expert_overrides.items():
            logger.info(f"Liger {key} off: {reason}")
    return _force_off(config, liger_prefix_lm_overrides(model_config), yields_to_request=())


def trl_reapplication_config(model_config, applied: dict) -> dict:
    """The config HF Trainer re-applies Liger with at ``train()``: ``applied`` with the toolkit's roles off.

    HF's re-application runs the applier liger-kernel registers for the model's own ``model_type`` alone, its
    defaults filling every omitted flag. Where the toolkit owns the family, every flag that applier declares and
    the toolkit applier does not hand upstream goes off, as do the roles a delegating spec took over
    (``upstream_off``); otherwise upstream's instance patch binds its variant over the family's, or, for a family
    the toolkit patches alone that upstream also registers (DeepSeek-V4), its norm swap and fused head.
    """
    toolkit = _toolkit_applier(*_resolve_applier(*_liger_model_types(model_config)))
    if toolkit is None:
        return applied
    reapplier = MODEL_TYPE_TO_APPLY_LIGER_FN.get(getattr(model_config, "model_type", None))
    reapplied_flags = set(inspect.signature(reapplier).parameters) - {"model"} if reapplier is not None else set()
    owned = {flag for flag in reapplied_flags if not toolkit.hands_upstream(flag)} | set(toolkit.spec.upstream_off)
    return {**applied, **dict.fromkeys(sorted(owned), False)}


def warn_if_flce_unreachable(model_config: AutoConfig, trainer_name: str) -> None:
    """Warn when FLCE was applied to a model whose trainer's loss can never engage it.

    A trainer computing its loss outside the model's forward passes no ``labels``, so the fused path never
    engages. The load-time patch site does not know the trainer and re-applying would not unpatch the
    forward, so this warns instead.
    """
    applied = getattr(model_config, LIGER_APPLIED_CONFIG_ATTR, None) or {}
    if applied.get("fused_linear_cross_entropy") and is_global_main_process():
        logger.warning(
            f"Liger fused_linear_cross_entropy is applied but unreachable on {trainer_name}: its "
            f"loss runs outside the model's forward and never passes labels, so the fused path "
            f"cannot engage and the full logits plane still materializes. For the memory saving "
            f"set use_chunked_grpo_logprobs instead."
        )


def _layered_liger_config(model_type: str, user_overrides: dict) -> dict:
    """The shared Liger defaults under the family's own and the user's overrides."""
    return {**_LIGER_DEFAULTS, **_PER_MODEL_DEFAULTS.get(model_type, {}), **user_overrides}


def _resolves_fused_loss(model_type: str | None, fallback_type: str | None, user_overrides: dict) -> bool:
    """Whether ``fused_linear_cross_entropy`` resolves on for the family: requested or its default, and offered by
    the applier Liger resolves for it."""
    resolved_type, apply_fn = _resolve_applier(model_type, fallback_type)
    if apply_fn is None or "fused_linear_cross_entropy" not in inspect.signature(apply_fn).parameters:
        return False
    return bool(_layered_liger_config(resolved_type, user_overrides)["fused_linear_cross_entropy"])


def _build_liger_config(
    apply_fn: Callable,
    model_type: str,
    user_overrides: dict,
    requested_kernels: frozenset[str],
    wrapper_head: bool,
) -> dict:
    """The flags ``apply_fn`` is called with: the toolkit and per-model defaults under the user's overrides, the
    fused loss off under a multimodal wrapper's own head, CE yielding to a requested FLCE, the applier's own
    default-off flags kept off, and every key the applier does not accept dropped."""
    signature_params = inspect.signature(apply_fn).parameters
    valid_params = set(signature_params.keys())

    config = _layered_liger_config(model_type, user_overrides)

    flce_requested = bool(user_overrides.get("fused_linear_cross_entropy"))
    if wrapper_head and "fused_linear_cross_entropy" in valid_params:
        if flce_requested and is_global_main_process():
            logger.warning(
                f"Liger fused_linear_cross_entropy requested for model_type={model_type}, but the "
                f"checkpoint loads as its multimodal wrapper, whose own head runs: the fused loss patches "
                f"the text tower's causal-LM class and would never engage. Keeping cross_entropy instead."
            )
        config["fused_linear_cross_entropy"] = False
        flce_requested = False

    # CE and FLCE are mutually exclusive (FLCE fuses the lm_head projection into the loss) and every
    # applier asserts on the pair. Only CE is a toolkit default, so the default yields to explicit FLCE.
    if flce_requested and {"cross_entropy", "fused_linear_cross_entropy"} <= valid_params:
        if user_overrides.get("cross_entropy"):
            raise ValueError(
                f"liger_kernel_config for {model_type} sets both cross_entropy: true and "
                f"fused_linear_cross_entropy: true, which no Liger applier accepts (FLCE already fuses "
                f"the lm_head matmul into the cross-entropy). Keep fused_linear_cross_entropy: true and "
                f"set cross_entropy: false."
            )
        if "cross_entropy" not in user_overrides:
            if is_global_main_process():
                logger.info(
                    f"Liger cross_entropy defaulted off for {model_type}: fused_linear_cross_entropy was "
                    f"requested and the two are mutually exclusive."
                )
            config["cross_entropy"] = False

    # An applier declaring one of these ``False`` states Liger's kernel cannot serve the family; read
    # off the applier so a family added upstream is covered.
    for name in _APPLIER_DEFAULT_OFF_FLAGS:
        parameter = signature_params.get(name)
        if parameter is not None and parameter.default is False and name not in user_overrides:
            config[name] = False

    # A key this applier does not accept is dropped by the filter below, so a requested one is named.
    unsatisfiable = sorted(requested_kernels - valid_params)
    if unsatisfiable and is_global_main_process():
        logger.warning(
            f"Liger {unsatisfiable} requested for model_type={model_type} but its applier does not "
            f"offer {'them' if len(unsatisfiable) > 1 else 'it'} — the request is dropped and those "
            f"paths run eager. Coverage per family: agent-docs/optimization/liger-kernels.md."
        )

    return {k: v for k, v in config.items() if k in valid_params}


def _apply_liger_for_standard_models(
    model_type: str,
    user_overrides: dict,
    fallback_model_type: str | None = None,
    requested_kernels: frozenset[str] = frozenset(),
) -> dict | None:
    """Apply the applier :func:`_resolve_applier` picks; returns the applied (signature-filtered) config, or
    ``None`` when none ran.

    A family no applier covers warns, since ``use_liger_kernel`` defaults on, and raises when the config
    asked for a specific kernel nothing can deliver.
    """
    resolved_type, apply_fn = _resolve_applier(model_type, fallback_model_type)
    # Resolved through the text tower of a multimodal wrapper: the checkpoint loads as the wrapper
    # class, whose own head runs. The applier's fused loss patches the tower's `*ForCausalLM`, which
    # that model never instantiates, so it would report as applied while the logits plane materializes.
    wrapper_head = apply_fn is not None and resolved_type != model_type
    if wrapper_head:
        suffix = " (toolkit applier)" if resolved_type in _TOOLKIT_LIGER_APPLIERS else ""
        if is_global_main_process():
            logger.info(
                f"Liger Kernel not available for model_type={model_type}; "
                f"falling back to text sub-config model_type={resolved_type}{suffix}"
            )
        model_type = resolved_type
    if apply_fn is None:
        if requested_kernels:
            raise ValueError(
                f"liger_kernel_config requests {sorted(requested_kernels)} for model_type={model_type}, "
                f"but no Liger applier covers this family — neither LIGER_FAMILY_SPECS nor liger_kernel's "
                f"own registry. Remove the keys and set use_liger_kernel: false, or add a LigerFamilySpec "
                f"for the family (agent-docs/models/adding-a-model.md)."
            )
        if is_global_main_process():
            logger.warning(
                f"No Liger applier for model_type={model_type}: this run trains UNFUSED — RMSNorm, the "
                f"GLU MLPs, cross-entropy and RoPE all run eager, costing throughput and the logits-plane "
                f"memory a fused loss would save. Add a LigerFamilySpec to LIGER_FAMILY_SPECS, or set "
                f"use_liger_kernel: false to state the choice."
            )
        return None

    filtered_config = _build_liger_config(apply_fn, model_type, user_overrides, requested_kernels, wrapper_head)
    upstream = model_type not in _TOOLKIT_LIGER_APPLIERS

    # Called before the model exists, an upstream applier swaps LayerNorm by rebinding `torch.nn.LayerNorm` (its
    # `modeling_<family>.nn` is `torch.nn`) under every model the process builds afterwards.
    if upstream and "layer_norm" in inspect.signature(apply_fn).parameters:
        if filtered_config.get("layer_norm") and is_global_main_process():
            logger.warning(
                f"Liger layer_norm was requested for {model_type}, but its upstream applier installs it before the "
                f"model is built by rebinding torch.nn.LayerNorm for the whole process; forcing it off."
            )
        filtered_config["layer_norm"] = False

    # The scoped CE patch serves an upstream applier's CE; a toolkit applier withholds the flag itself.
    call_config = dict(filtered_config)
    if upstream and call_config.get("cross_entropy"):
        call_config["cross_entropy"] = False
        patch_loss_utils_cross_entropy()

    if is_global_main_process():
        logger.info(f"Applying Liger Kernel for {model_type}: {filtered_config}")
    apply_fn(**call_config)
    if torch.nn.LayerNorm is not torch.nn.modules.normalization.LayerNorm:
        raise RuntimeError(
            f"The Liger applier for {model_type} rebound torch.nn.LayerNorm to {torch.nn.LayerNorm.__name__} for "
            f"the whole process, so every LayerNorm built afterwards (a teacher, a reference, a vision tower) would "
            f"run that class. Set the flag that installs it to false in liger_kernel_config, or set "
            f"use_liger_kernel: false."
        )
    if is_global_main_process():
        logger.info(f"✓ Liger Kernel applied for {model_type}")
    return filtered_config


def apply_liger_kernel(
    model_config: AutoConfig,
    liger_kernel_config: dict | None = None,
    needs_ep_layers: bool = False,
    tp_size: int = 1,
    cp_size: int = 1,
    pp_size: int = 1,
) -> dict | None:
    """Apply Liger Kernel before model loading, under the per-family defaults and the safety rules of
    :func:`sanitize_liger_config`.

    Returns the effective applied config, or ``None`` when nothing was applied, and records it run-scoped
    on ``model_config`` under :data:`LIGER_APPLIED_CONFIG_ATTR`; under CP, also whether the wrapper runs the
    fused loss, under :data:`LIGER_CP_FUSED_LOSS_ATTR`.
    """
    model_type, fallback_type = _liger_model_types(model_config)
    patch_config = liger_kernel_config or {}
    if cp_size > 1:
        cp_fused_loss = _resolves_fused_loss(model_type, fallback_type, patch_config)
        set_config_field_run_scoped(model_config, LIGER_CP_FUSED_LOSS_ATTR, cp_fused_loss)
        # Taken out of the patch request rather than forced off under a warning: the CP override still keeps the
        # model's own head unpatched, and the wrapper serves the request.
        patch_config = {key: value for key, value in patch_config.items() if key != "fused_linear_cross_entropy"}
        if cp_fused_loss and is_global_main_process():
            logger.info(
                f"Liger fused_linear_cross_entropy under Context Parallelism ({model_type}): the model's own head "
                f"stays unpatched, and the CP wrapper runs the fused kernel on a labelled training forward (SFT); "
                f"any other forward keeps the full logits."
            )

    user_overrides = sanitize_liger_config(
        patch_config,
        needs_ep_layers=needs_ep_layers,
        model_config=model_config,
        tp_size=tp_size,
        cp_size=cp_size,
        pp_size=pp_size,
    )

    # The raw dict, not the sanitized one: the fold writes `swiglu: False` under EP, which is the
    # toolkit's decision and must not read as a user request on a family nothing covers.
    requested = frozenset(key for key, value in (liger_kernel_config or {}).items() if value)
    applied_config = _apply_liger_for_standard_models(model_type, user_overrides, fallback_type, requested)
    set_config_field_run_scoped(model_config, LIGER_APPLIED_CONFIG_ATTR, applied_config)
    return applied_config


def finalize_liger_after_direct_load(training_config, original_use_liger: bool, model) -> None:
    """Reconcile TRL's ``use_liger_kernel`` flag after a direct model load already applied Liger.

    Reads the effective config :func:`apply_liger_kernel` applied. Under ``fused_linear_cross_entropy``
    the model returns no logits, so the flag must stay on or TRL slices ``None`` in its metric path, and
    pinning the applied config makes TRL's re-application an identical re-patch. Otherwise the flag is
    cleared, so TRL does not re-apply Liger with its own defaults over the patched modules.
    """
    if not original_use_liger:
        return
    applied = getattr(model.config, LIGER_APPLIED_CONFIG_ATTR, None)
    if applied and applied.get("fused_linear_cross_entropy", False):
        training_config.use_liger_kernel = True
        training_config.liger_kernel_config = trl_reapplication_config(model.config, applied)
    else:
        training_config.use_liger_kernel = False
