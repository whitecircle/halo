"""Liger Kernel dispatch: resolves the applier per model_type and applies it with the right defaults.

Resolution order: toolkit applier (built from :mod:`~src.kernels.liger.families`) → Liger registry →
``fallback_model_type`` (text-config for multimodal wrappers). A delegating spec resolves on the toolkit
branch, so every rule here applies to it. Parallelism safety filters
(:func:`liger_parallelism_overrides`) then force kernels off; a family no applier covers warns, and an
explicit per-kernel request for it raises.
"""

from __future__ import annotations

import inspect

from accelerate.logging import get_logger
from liger_kernel.transformers.auto_model import MODEL_TYPE_TO_APPLY_LIGER_FN
from transformers import AutoConfig

from src.kernels.liger.builder import build_liger_appliers
from src.kernels.liger.families import LIGER_FAMILY_SPECS
from src.models.loading.config_levels import set_config_field_run_scoped
from src.models.moe_balancing import ep_wraps_experts

logger = get_logger(__name__)

# The effective Liger config this module patched the classes with, stamped on model.config for
# the post-load TRL flag finalization to read back. Run-scoped, so it never reaches an exported
# ``config.json``.
LIGER_APPLIED_CONFIG_ATTR = "_halo_liger_applied_config"


# liger-kernel 0.8.0's Qwen3.5 applier imports ``liger_cross_entropy`` from a module where the symbol
# does not live, so its CE branch raises ImportError. Keyed by qualified name to cover every alias.
_LIGER_CROSS_ENTROPY_BROKEN_APPLIERS = frozenset(
    {"liger_kernel.transformers.monkey_patch.apply_liger_kernel_to_qwen3_5"}
)


# Kernels a parallelism override makes inert rather than incorrect, so an explicit request is honored.
# A kernel that would compute the wrong loss (CE/FLCE under TP/CP/PP) is forced off even when requested.
_YIELDS_TO_EXPLICIT_REQUEST = ("swiglu", "geglu")


# Every family the toolkit patches, built from its declarative spec (:mod:`src.kernels.liger.families`)
# by the role-driven builder: the families upstream Liger has no entry for, plus those whose spec
# delegates to upstream's applier and adds a role on top. A family registered into Liger's own dict
# instead would resolve on the upstream branch and miss the rules this module derives from this registry.
_TOOLKIT_LIGER_APPLIERS = build_liger_appliers(LIGER_FAMILY_SPECS)

# Model types whose fused GLU survives an EP wrapper: a toolkit spec names the dense and shared-expert
# MLPs, which every wrapper adopts unchanged. A delegating spec that names none has only upstream's
# routed-expert swap, which the wrapper replaces, so it is forced off as for any upstream family.
_TOOLKIT_GLU_SURVIVES_EP = frozenset(
    model_type
    for spec in LIGER_FAMILY_SPECS
    if spec.glu_mlp or not spec.delegates_to_upstream
    for model_type in spec.model_types
)


# Per-family overrides of the shared Liger defaults, read off each toolkit applier's own signature so
# the two stay in sync. Built from the real appliers at import, so a substituted applier still gets them.
_PER_MODEL_DEFAULTS = {
    model_type: {
        name: param.default
        for name, param in inspect.signature(applier).parameters.items()
        if name in ("cross_entropy", "fused_linear_cross_entropy") and param.default is not inspect.Parameter.empty
    }
    for model_type, applier in _TOOLKIT_LIGER_APPLIERS.items()
}

# Applier lookup order: a toolkit applier wins over Liger's own entry for the same family.
_APPLIER_REGISTRIES = (_TOOLKIT_LIGER_APPLIERS, MODEL_TYPE_TO_APPLY_LIGER_FN)

# The shared Liger defaults; per-family overrides and user overrides layer on top. Shared and never
# mutated: callers build a fresh dict from it (see :func:`_apply_liger_for_standard_models`).
_LIGER_DEFAULTS = {
    "rope": True,
    "cross_entropy": True,
    "fused_linear_cross_entropy": False,
    "rms_norm": True,
    "swiglu": True,
    "geglu": True,
}


def resolve_liger_applier(model_type: str):
    """The Liger applier claiming ``model_type``, toolkit first, or ``None``."""
    return next((registry[model_type] for registry in _APPLIER_REGISTRIES if model_type in registry), None)


def _applier_identities(apply_fn) -> set[str]:
    """Qualified names of every applier the call will run; a toolkit delegate also runs upstream's.

    A defect belongs to the function that actually runs, so a spec delegating to it inherits both the
    defect and the default that works around it.
    """
    chain = (apply_fn, getattr(apply_fn, "upstream", None))
    # A callable object has no __qualname__ of its own; fall back to its class's.
    return {f"{fn.__module__}.{getattr(fn, '__qualname__', type(fn).__qualname__)}" for fn in chain if fn is not None}


def _liger_model_types(model_config) -> tuple[str | None, str | None]:
    """``(model_type, text-config fallback)``: the two spellings an applier may be registered under.

    Multimodal wrappers carry the decoder classes on their text sub-config, so Liger resolves them
    through it; both the resolution and the EP predicate below read the same pair.
    """
    model_type = getattr(model_config, "model_type", None)
    text_type = getattr(getattr(model_config, "text_config", None), "model_type", None)
    return model_type, (text_type if text_type and text_type != model_type else None)


def liger_ep_disables_fused_glu(needs_ep_wrappers: bool, model_config) -> bool:
    """The ``has_ep_wrapped_experts`` argument both Liger application sites pass.

    True only when an EP wrapper replaces every module the resolved applier's GLU patch swaps: upstream's
    MoE appliers set the routed-experts class, which ``patch_moe_model_for_ep`` replaces wholesale
    (leaving the patch inert under EP), while a toolkit spec names the dense and shared-expert MLPs every
    wrapper adopts unchanged. Derived from the specs rather than a model_type list.
    """
    if not ep_wraps_experts(needs_ep_wrappers, model_config):
        return False
    return not any(candidate in _TOOLKIT_GLU_SURVIVES_EP for candidate in _liger_model_types(model_config))


def liger_parallelism_overrides(
    *,
    has_ep_wrapped_experts: bool = False,
    tp_size: int = 1,
    cp_size: int = 1,
    pp_size: int = 1,
) -> dict[str, str]:
    """Liger kernels a parallelism axis makes incorrect or inert, mapped to the reason.

    Liger is applied twice: once at model load (:func:`apply_liger_kernel`) and once by the trainer
    mixin, which re-sanitizes ``liger_kernel_config`` before TRL can re-apply it. A filter present at one
    site but not the other is undone by whichever runs second, so both call this rather than restating
    the rules.
    """
    overrides: dict[str, str] = {}
    if has_ep_wrapped_experts:
        # EP wrappers replace `.experts` with a DeepEP-aware path; Liger swaps a single FFN module.
        reason = "EP wrappers replace the expert FFN Liger would swap"
        overrides["swiglu"] = overrides["geglu"] = reason
    if tp_size > 1:
        # A ColwiseParallel-sharded lm_head makes the fused CE softmax a partial-vocab slice.
        reason = f"TP (tp_size={tp_size}) shards lm_head, making the fused softmax a partial-vocab slice"
        overrides["cross_entropy"] = overrides["fused_linear_cross_entropy"] = reason
    external_loss = {"CP": cp_size, "PP": pp_size}
    engaged = [f"{name} (size {size})" for name, size in external_loss.items() if size > 1]
    if engaged:
        # CP's wrapper and PP's last stage compute the loss from logits and pass no labels, so Liger's
        # `skip_logits` gate never fires; CE goes too, since upstream rebinds F.cross_entropy
        # process-wide.
        reason = f"{' and '.join(engaged)} computes the loss outside the model's forward, so the kernel never fires"
        overrides["cross_entropy"] = overrides["fused_linear_cross_entropy"] = reason
    return overrides


def apply_liger_parallelism_overrides(user_config: dict, forced_off: dict[str, str]) -> dict:
    """Fold :func:`liger_parallelism_overrides` into a user config, honoring the explicit-request exemption.

    Both application sites fold the shared rule table the same way here, so one cannot undo the other.
    """
    result = dict(user_config)
    for key, reason in forced_off.items():
        if key in _YIELDS_TO_EXPLICIT_REQUEST and key in user_config:
            continue
        if result.get(key):
            logger.warning(f"Liger {key} was explicitly enabled but {reason}; forcing it off.")
        result[key] = False
    return result


def warn_if_flce_unreachable(model_config: AutoConfig, trainer_name: str) -> None:
    """Warn when FLCE was applied to a model whose trainer's loss can never engage it.

    A trainer computing its loss outside the model's forward (the GRPO family) passes no ``labels``, so
    Liger's ``skip_logits`` gate never engages and the full logits plane materializes unfused. It cannot
    be forced off here: the load-time patch site does not know the trainer, and re-applying would not
    unpatch ``lce_forward``, so this site warns instead.
    """
    applied = getattr(model_config, LIGER_APPLIED_CONFIG_ATTR, None) or {}
    if applied.get("fused_linear_cross_entropy"):
        logger.warning(
            f"Liger fused_linear_cross_entropy is applied but unreachable on {trainer_name}: its "
            f"loss runs outside the model's forward and never passes labels, so the fused path "
            f"cannot engage and the full logits plane still materializes. For the memory saving "
            f"set use_chunked_grpo_logprobs instead."
        )


def apply_liger_kernel(
    model_config: AutoConfig,
    liger_kernel_config: dict | None = None,
    needs_ep_wrappers: bool = False,
    tp_size: int = 1,
    cp_size: int = 1,
    pp_size: int = 1,
) -> dict | None:
    """Apply Liger Kernel optimizations before model loading.

    Defaults ``rope/cross_entropy/rms_norm/swiglu=True, fused_linear_cross_entropy=False`` (overridable
    via ``liger_kernel_config``; a requested FLCE makes the defaulted ``cross_entropy`` yield, since the
    two are mutually exclusive). Auto-disables ``swiglu``/``geglu`` where the family has an EP wrapper
    class, and CE/FLCE under TP (sharded lm_head) and under CP/PP (the loss is computed outside the
    model's forward).

    Returns the effective applied config (per-model defaults + safety filters, not the raw user dict),
    or ``None`` when nothing was applied; also records it on ``model_config`` under
    :data:`LIGER_APPLIED_CONFIG_ATTR` (run-scoped, so it never reaches an exported ``config.json``) so
    post-load consumers (e.g. TRL flag finalization) see what the model was actually patched with.
    """
    # Multimodal wrappers: Liger registers only the inner text path, hence the text_config fallback.
    model_type, fallback_type = _liger_model_types(model_config)

    user_overrides = liger_kernel_config or {}
    forced_off = liger_parallelism_overrides(
        # A MoE family with no registered EP wrapper (qwen3_next) keeps Liger's swiglu as its only
        # fused expert path, as does one whose applier never patched the routed experts.
        has_ep_wrapped_experts=liger_ep_disables_fused_glu(needs_ep_wrappers, model_config),
        tp_size=tp_size,
        cp_size=cp_size,
        pp_size=pp_size,
    )
    user_overrides = apply_liger_parallelism_overrides(user_overrides, forced_off)
    if forced_off:
        logger.info(f"Liger disabled by parallelism: {', '.join(sorted(forced_off))}")

    # The raw dict, not the parallelism-folded one: the fold writes `swiglu: False` under EP, which
    # is the toolkit's decision and must not read as a user request on a family nothing covers.
    requested = frozenset(key for key, value in (liger_kernel_config or {}).items() if value)
    applied_config = _apply_liger_for_standard_models(model_type, user_overrides, fallback_type, requested)
    set_config_field_run_scoped(model_config, LIGER_APPLIED_CONFIG_ATTR, applied_config)
    return applied_config


def _apply_liger_for_standard_models(
    model_type: str,
    user_overrides: dict,
    fallback_model_type: str | None = None,
    requested_kernels: frozenset[str] = frozenset(),
) -> dict | None:
    """Apply Liger Kernel: toolkit appliers → Liger registry → fallback text-config model_type.

    Returns the applied (signature-filtered) config dict, or ``None`` when no applier ran. A family no
    applier covers warns, since ``use_liger_kernel`` defaults on and a run would otherwise train unfused
    without notice, and raises when the config asked for a specific kernel nothing can deliver.
    """
    apply_fn = resolve_liger_applier(model_type)
    if apply_fn is None and fallback_model_type:
        apply_fn = resolve_liger_applier(fallback_model_type)
        if apply_fn is not None:
            suffix = " (toolkit applier)" if fallback_model_type in _TOOLKIT_LIGER_APPLIERS else ""
            logger.info(
                f"Liger Kernel not available for model_type={model_type}; "
                f"falling back to text sub-config model_type={fallback_model_type}{suffix}"
            )
            model_type = fallback_model_type
    if apply_fn is None:
        if requested_kernels:
            raise ValueError(
                f"liger_kernel_config requests {sorted(requested_kernels)} for model_type={model_type}, "
                f"but no Liger applier covers this family — neither the toolkit registry "
                f"(src/kernels/liger/families.py) nor liger_kernel's own. Remove the keys and set "
                f"use_liger_kernel: false, or add a LigerFamilySpec for the family "
                f"(agent-docs/models/adding-a-model.md)."
            )
        logger.warning(
            f"No Liger applier for model_type={model_type}: this run trains UNFUSED — RMSNorm, the "
            f"GLU MLPs, cross-entropy and RoPE all run eager, costing throughput and the logits-plane "
            f"memory a fused loss would save. Add a LigerFamilySpec (src/kernels/liger/families.py), "
            f"or set use_liger_kernel: false to state the choice."
        )
        return None

    signature_params = inspect.signature(apply_fn).parameters
    valid_params = set(signature_params.keys())

    # FLCE-only appliers exist solely for the fused loss path; the generic FLCE=False would no-op them.
    # Layered into a fresh dict rather than written into _LIGER_DEFAULTS, which is shared: the first
    # FLCE-only family would otherwise flip the default for every later call in the process.
    flce_only = "fused_linear_cross_entropy" in valid_params and "cross_entropy" not in valid_params
    config = {
        **_LIGER_DEFAULTS,
        **({"fused_linear_cross_entropy": True} if flce_only else {}),
        **_PER_MODEL_DEFAULTS.get(model_type, {}),
        **user_overrides,
    }

    # CE and FLCE are mutually exclusive (FLCE fuses the lm_head projection into the loss) and every
    # applier asserts on the pair. Only CE is a toolkit default, so the default yields to explicit FLCE.
    if (
        user_overrides.get("fused_linear_cross_entropy")
        and {"cross_entropy", "fused_linear_cross_entropy"} <= valid_params
    ):
        if user_overrides.get("cross_entropy"):
            raise ValueError(
                f"liger_kernel_config for {model_type} sets both cross_entropy: true and "
                f"fused_linear_cross_entropy: true, which no Liger applier accepts (FLCE already fuses "
                f"the lm_head matmul into the cross-entropy). Keep fused_linear_cross_entropy: true and "
                f"set cross_entropy: false."
            )
        if "cross_entropy" not in user_overrides:
            logger.info(
                f"Liger cross_entropy defaulted off for {model_type}: fused_linear_cross_entropy was "
                f"requested and the two are mutually exclusive."
            )
            config["cross_entropy"] = False

    # An applier declaring ``rope=False`` states Liger's generic rotary cannot serve this family
    # (mrope, partial rotary, YARN); read it off the applier so a family added upstream is covered.
    rope_param = signature_params.get("rope")
    if rope_param is not None and rope_param.default is False and "rope" not in user_overrides:
        config["rope"] = False
    broken_ce = _LIGER_CROSS_ENTROPY_BROKEN_APPLIERS & _applier_identities(apply_fn)
    if broken_ce and "cross_entropy" not in user_overrides:
        logger.info(f"Liger cross_entropy disabled for {model_type}: {sorted(broken_ce)[0]} cannot apply it.")
        config["cross_entropy"] = False

    # A key this applier does not accept is dropped by the filter below. Dropping a requested kernel
    # without notice is the same failure the no-applier branch raises over: a config requesting
    # `fused_linear_cross_entropy` on a family whose head cannot be fused (GLM-5 Next, Step-3.7,
    # Inkling) would report as applied and then materialize the full logits plane anyway.
    unsatisfiable = sorted(requested_kernels - valid_params)
    if unsatisfiable:
        logger.warning(
            f"Liger {unsatisfiable} requested for model_type={model_type} but its applier does not "
            f"offer {'them' if len(unsatisfiable) > 1 else 'it'} — the request is dropped and those "
            f"paths run eager. Coverage per family: agent-docs/optimization/liger-kernels.md."
        )

    filtered_config = {k: v for k, v in config.items() if k in valid_params}

    logger.info(f"Applying Liger Kernel for {model_type}: {filtered_config}")
    apply_fn(**filtered_config)
    logger.info(f"✓ Liger Kernel applied for {model_type}")
    return filtered_config


def apply_liger_kernel_for_direct_loading(
    model_name_or_path: str,
    training_config,
    trust_remote_code: bool = True,
) -> None:
    """Apply Liger Kernel before direct ``from_pretrained`` loading (non-distributed path).

    Applies toolkit defaults and disables TRL's re-application (prevents double-patching).
    """
    if not getattr(training_config, "use_liger_kernel", False):
        return

    config = AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=trust_remote_code)
    apply_liger_kernel(config, liger_kernel_config=getattr(training_config, "liger_kernel_config", None))

    training_config.use_liger_kernel = False  # prevent TRL re-applying
