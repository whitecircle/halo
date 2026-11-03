"""Attention-implementation selection: FA4/FA3, then FA2, SDPA, flex, eager, by GPU capability, the model class's
support flags and what each kernel carries (live sinks, logit softcapping, FA4's head_dim-256 and varlen limits)."""

from __future__ import annotations

import importlib
import inspect
import logging
import os

import torch
from transformers import AutoModelForCausalLM, AutoModelForImageTextToText
from transformers.utils import is_flash_attn_2_available

from halo.hardware import is_blackwell_gpu, is_hopper_gpu
from halo.kernels.jit_cache import ensure_fa4_kernel_cache_env
from halo.models.attention.gpt_oss_sinks import (
    RESOLVED_SINKS_RESET_ATTR,
    SINK_CARRYING_ATTN_HINT,
    SinksPolicy,
    holds_unneutralized_sinks,
    install_sdpa_for_reset_sinks,
    uses_gpt_oss_sinks,
)
from halo.models.attention.layout import (
    declares_attention_heads,
    has_latent_attention,
    has_sliding_window_layers,
    resolve_head_dim,
    resolve_qk_v_head_dims,
)
from halo.models.attention.prefix_lm import has_prefix_lm_attention
from halo.models.config_access import get_config_field, set_config_field_run_scoped, text_config
from halo.models.model_classes import mapped_model_class
from halo.models.moe.family_specs import find_moe_family_spec_for_config
from halo.runtime.ranks import is_global_main_process

logger = logging.getLogger(__name__)

# The modules each flash backend's kernels import from, in transformers' dispatch order: FA3 runs
# through flash_attn_interface first, so a coexisting flash_attn_3 build never answers for it.
_FLASH_KERNEL_MODULES = {
    "flash_attention_2": ("flash_attn",),
    "flash_attention_3": ("flash_attn_interface", "flash_attn_3"),
    "flash_attention_4": ("flash_attn.cute",),
}

# Varlen backends consume cu_seqlens from packed position_ids; dense backends isolate packed documents only where the
# model plumbs position_ids into its mask.
VARLEN_ATTN_IMPLEMENTATIONS = tuple(_FLASH_KERNEL_MODULES)

# The names transformers forwards a learnable sink under: FA3's ``s_aux``, FA4's ``learnable_sink``.
FLASH_SINK_PARAM_NAMES = ("s_aux", "learnable_sink")

# The widest head FA2 takes; FA4's SM100 kernel overflows tensor memory past it.
FLASH_MAX_HEAD_DIM = 256
# The query/key head width flash-attn-4 trains on its dedicated SM100/SM110 2-CTA backward kernel.
FA4_DEDICATED_HEAD_DIM = 256
# transformers' switch for its flash kernels' ``deterministic`` argument; ``full_determinism`` sets it.
FLASH_ATTENTION_DETERMINISTIC_ENV = "FLASH_ATTENTION_DETERMINISTIC"


def is_flash_attn_implementation(attn_impl: str | None) -> bool:
    return attn_impl in VARLEN_ATTN_IMPLEMENTATIONS


def flash_kernel(attn_impl: str, *, varlen: bool = True):
    """``ImportError`` when no installed build has it. Varlen by default, as transformers reads the capabilities of
    both calls off ``_flash_varlen_fn`` alone."""
    if attn_impl == "flash_attention_4":
        ensure_fa4_kernel_cache_env()
    name = "flash_attn_varlen_func" if varlen else "flash_attn_func"
    for module_name in _FLASH_KERNEL_MODULES[attn_impl]:
        try:
            kernel = getattr(importlib.import_module(module_name), name, None)
        except ImportError:
            continue
        if kernel is not None:
            return kernel
    raise ImportError(f"no installed {attn_impl} build provides {name}")


def _flash_build_imports(attn_impl: str) -> bool:
    try:
        flash_kernel(attn_impl)
    except ImportError:
        return False
    return True


def preferred_flash_impl(*, allow_fa4: bool = True) -> str:
    """By GPU architecture alone; whether its build imports is each caller's own probe."""
    if allow_fa4 and is_blackwell_gpu():
        return "flash_attention_4"
    if is_hopper_gpu():
        return "flash_attention_3"
    return "flash_attention_2"


def _detect_attention_impl() -> str:
    """A broken (not absent) build warns, since nothing else would report the slower kernel."""
    if not torch.cuda.is_available():
        return "flash_attention_2"  # CPU host; the choice is re-validated against the model downstream
    preferred = preferred_flash_impl()
    detected, build_note = preferred, ""
    if preferred != "flash_attention_2":
        build = _FLASH_KERNEL_MODULES[preferred][-1]
        try:
            flash_kernel(preferred)
            build_note = f" ({build} installed)"
        except ImportError:
            detected, build_note = "flash_attention_2", f" ({build} not installed)"
        except Exception as e:
            if is_global_main_process():
                logger.warning(
                    f"{build} IS present but unusable ({type(e).__name__}: {e}); falling back to "
                    f"FlashAttention-2, which is correct but slower on attention. Repair the {build} install "
                    f"to regain {preferred}."
                )
            return "flash_attention_2"
    if is_global_main_process():
        logger.info(f"{detected} is this GPU's flash backend{build_note}; model-level checks may move a model off it.")
    return detected


def effective_attn_implementation(model_config) -> str | None:
    """Read off the text sub-config first, as composite wrappers record it there and only sometimes mirror it up."""
    if model_config is None:
        return None
    return getattr(text_config(model_config), "_attn_implementation", None) or getattr(
        model_config, "_attn_implementation", None
    )


def has_head_dim_beyond_flash(model_config) -> bool:
    """Gemma 4's 512-wide global heads; ``False`` on an attention-free config (Mamba)."""
    return declares_attention_heads(model_config) and resolve_head_dim(model_config) > FLASH_MAX_HEAD_DIM


def has_attention_logit_softcap(model_config) -> bool:
    """Gemma 2's softcap, which transformers' SDPA path drops without a word."""
    return bool(get_config_field(model_config, "attn_logit_softcapping", per_layer_reduce=max))


def uses_fa4_dedicated_head_dim(model_config) -> bool:
    """MLA's nope + RoPE split counted (GLM-4.7-Flash's 192 + 64). The value width is not read, as transformers' flash
    path pads values up to the query/key width."""
    return declares_attention_heads(model_config) and resolve_qk_v_head_dims(model_config)[0] == FA4_DEDICATED_HEAD_DIM


def fa4_dedicated_kernel_refusal(model_config) -> str | None:
    """It asserts on Gemma 2's softcap and Gemma 3's sliding windows; latent attention stays on FA2 as a precaution."""
    if not uses_fa4_dedicated_head_dim(model_config):
        return None
    if has_attention_logit_softcap(model_config):
        return "it takes no attention logit softcapping"
    if has_sliding_window_layers(model_config):
        return "it takes no sliding-window attention"
    if has_latent_attention(model_config):
        return "multi-head latent attention at this width stays on FA2 as a precaution"
    return None


def fa4_varlen_refusal(model_config) -> str | None:
    """Gemma 2's softcap is unvalidated on the varlen kernels, which packed, padding-free and padded batches reach."""
    return "attention logit softcapping" if has_attention_logit_softcap(model_config) else None


def flash_attention_2_in_place_of_fa4(reason: str) -> str:
    """Raises without an FA2 build, as the rest either drops a softcap (SDPA) or cannot compile under EP (flex)."""
    if not is_flash_attn_2_available():
        raise ValueError(
            f"{reason}, and no FlashAttention-2 build imports to run it instead. Install flash-attn, or set "
            f"attn_implementation to flex_attention or eager (sdpa too for a model without a softcap)."
        )
    if is_global_main_process():
        logger.warning(f"{reason}; falling back to attn_implementation='flash_attention_2'.")
    return "flash_attention_2"


def flash_attention_deterministic() -> bool:
    """By transformers' own read of the variable: only ``"1"`` turns it on."""
    return os.environ.get(FLASH_ATTENTION_DETERMINISTIC_ENV, "0") == "1"


def enable_flash_attention_deterministic() -> None:
    os.environ[FLASH_ATTENTION_DETERMINISTIC_ENV] = "1"


def supports_fa4_backward(model_config) -> bool:
    """The dedicated kernel's deterministic mode is unvalidated, so a run asking for it takes FA4 off that kernel."""
    if fa4_dedicated_kernel_refusal(model_config) is not None:
        return False
    return not (flash_attention_deterministic() and uses_fa4_dedicated_head_dim(model_config))


def is_fa4_demoted_by_determinism(model_config) -> bool:
    """Whether the deterministic backward alone takes FA4 off; a dedicated-kernel refusal moves it to FA2 instead."""
    return fa4_dedicated_kernel_refusal(model_config) is None and not supports_fa4_backward(model_config)


def _attn_impl_handles_sinks(attn_impl: str) -> bool:
    """A flash kernel's verdict is read off its signature, so a build that adds sink support needs no code change."""
    if attn_impl in ("eager", "flex_attention"):
        return True
    if not is_flash_attn_implementation(attn_impl):
        return False
    try:
        params = inspect.signature(flash_kernel(attn_impl)).parameters
    except ImportError:
        return False
    return any(name in params for name in FLASH_SINK_PARAM_NAMES)


def declared_model_class(model_config) -> type | None:
    """The class declaring this config's ``_supports_*`` flags. None for a remote-code config."""
    return mapped_model_class(model_config, AutoModelForCausalLM, AutoModelForImageTextToText)


def declares_attention_backends(model_config) -> bool:
    """False for a remote-code config; a flash label its class refuses (Bailing's v4-era code) raises at the build."""
    return declared_model_class(model_config) is not None


def declares_flash_attention(model_config) -> bool:
    """No flash kernel reads a prefix LM's mask, so its trained tokens would attend to those they are scored on."""
    if has_prefix_lm_attention(model_config):
        return False
    return getattr(declared_model_class(model_config), "_supports_flash_attn", True)


def validate_attn_implementation(
    model_config, attn_impl: str, sinks_policy: SinksPolicy | None = None, *, dtype: torch.dtype | None = None
) -> str:
    """``attn_impl`` or the first fallback the class's flags and the sink matrix accept. A neutralized ``sinks_policy``
    allows SDPA on a sinks model (the dropped column is 0), a live one never; ``None`` re-validates against the verdict
    recorded on this config. A float32 ``dtype`` keeps the fallback off FA3 and FA2, which are half-precision only."""
    has_sinks = uses_gpt_oss_sinks(model_config)
    sinks_reset = None if sinks_policy is None else not sinks_policy.live
    if attn_impl == "sdpa" and has_sinks and sinks_reset:
        install_sdpa_for_reset_sinks(model_config)

    # Recorded per config instance, so nested re-validations see it while a fresh config cannot inherit
    # an earlier model's sdpa approval; run-scoped, so no exported config.json carries it.
    if sinks_reset is not None:
        set_config_field_run_scoped(model_config, RESOLVED_SINKS_RESET_ATTR, sinks_reset)
    recorded_reset = getattr(model_config, RESOLVED_SINKS_RESET_ATTR, None)

    # An unmapped config keeps sdpa allowed (the model build raises if it must) and the sinks escape hatch closed.
    resolved_class = declared_model_class(model_config)
    supports_sdpa = getattr(resolved_class, "_supports_sdpa", True)
    supports_flash = declares_flash_attention(model_config)

    # sdpa drops sinks and the softcap, so it is valid only without either, or once the sinks were reset.
    softcapped = has_attention_logit_softcap(model_config)
    sdpa_ok = supports_sdpa and not softcapped and (not has_sinks or recorded_reset is True)

    live_sinks = has_sinks and recorded_reset is False
    if live_sinks and is_flash_attn_implementation(attn_impl) and not _flash_build_imports(attn_impl):
        raise ValueError(
            f"attn_implementation='{attn_impl}' is not installed in this environment (no build provides its "
            f"kernels), so it cannot carry the learnable attention sinks that reset_sinks=false keeps live. Use "
            f"a sink-carrying implementation this image has ({SINK_CARRYING_ATTN_HINT} on Hopper), or set "
            f"reset_sinks: true for off-policy fine-tuning."
        )
    if live_sinks and not _attn_impl_handles_sinks(attn_impl):
        raise ValueError(
            f"attn_implementation='{attn_impl}' silently drops the learnable attention sinks that "
            f"reset_sinks=false keeps live — every training logprob would shift by nats vs the served "
            f"policy. Use a sink-carrying implementation ({SINK_CARRYING_ATTN_HINT} on Hopper — the stock "
            f"Hopper flash_attention_3 build lacks s_aux, so FA3 works only with a build exposing it), or set "
            f"reset_sinks: true for off-policy fine-tuning."
        )

    # Fallback chain: sinks and softcap models prefer flash over flex, whose compile path is unusable under EP; on
    # Hopper the arch's own FA3 comes first.
    candidates = [attn_impl]
    if has_sinks or softcapped:
        if live_sinks:
            candidates.append("flash_attention_4")
        if dtype != torch.float32 and not has_head_dim_beyond_flash(model_config):
            if is_hopper_gpu() and _flash_build_imports("flash_attention_3"):
                candidates.append("flash_attention_3")
            if is_flash_attn_2_available():
                candidates.append("flash_attention_2")
        candidates.append("flex_attention")
    if sdpa_ok:
        candidates.append("sdpa")
    candidates.append("eager")

    for candidate in candidates:
        if candidate == "sdpa" and not sdpa_ok:
            continue
        # transformers raises on the same flag only at model build, after the checkpoint is fetched.
        if is_flash_attn_implementation(candidate) and not supports_flash:
            continue
        if live_sinks and not _attn_impl_handles_sinks(candidate):
            continue
        try:
            model_config._attn_implementation = candidate
        except ValueError:
            continue  # a config that validates the assignment refused this candidate
        if candidate != attn_impl and is_global_main_process():
            reason = (
                " (a prefix-LM decoder: flash kernels never read its token_type_ids mask)"
                if is_flash_attn_implementation(attn_impl) and has_prefix_lm_attention(model_config)
                else ""
            )
            logger.warning(
                f"Model {type(model_config).__name__} does not support "
                f"attn_implementation='{attn_impl}'{reason}, using '{candidate}'"
            )
        return candidate
    raise ValueError(
        f"{type(model_config).__name__} accepts none of the attention implementations tried "
        f"({', '.join(candidates)}); set attn_implementation to one this model's config accepts."
    )


def revalidate_attn_kwarg(model_kwargs: dict, model_config) -> None:
    """So a family whose ``_supports_*`` flags refuse the policy's choice fails before the load."""
    attn_impl = model_kwargs.pop("attn_implementation", None)
    if attn_impl is not None:
        model_kwargs["attn_implementation"] = validate_attn_implementation(model_config, attn_impl)


def verify_live_sinks_carried(model) -> None:
    """Read off the built model, so it holds for a backend no validator chose. Neutralized sinks (``None``,
    ``dtype.min``) carry no softmax mass and pass."""
    config = getattr(model, "config", None)
    if config is None or not uses_gpt_oss_sinks(config):
        return
    attn_impl = effective_attn_implementation(config)
    if _attn_impl_handles_sinks(attn_impl) or not holds_unneutralized_sinks(model):
        return
    raise ValueError(
        f"This GptOss model runs attn_implementation={attn_impl!r}, which drops its attention sinks, while the "
        f"sinks are live: every logprob would shift by nats against the served model. Use a sink-carrying "
        f"implementation ({SINK_CARRYING_ATTN_HINT}), or neutralize the sinks first (reset_sinks: true for a "
        f"training run)."
    )


def resolve_attn_implementation(
    model_config,
    attn_implementation: str | None,
    dtype: torch.dtype,
    *,
    sinks_policy: SinksPolicy,
    fa4_fallback: str = "sdpa",
) -> str:
    """Auto-detected when None, narrowed by the per-family kernel limits, then validated. ``fa4_fallback`` is taken in
    place of FA4 where :func:`supports_fa4_backward` refuses it. FA4's varlen limit and the kernel taken in place of a
    deterministic FA4 on multi-document rows depend on the rows, so the scripts' requests apply them."""
    attn_implementation = attn_implementation or _detect_attention_impl()

    if dtype == torch.float32 and is_flash_attn_implementation(attn_implementation):
        # flex carries the sinks and the softcap SDPA drops.
        needs_flex = uses_gpt_oss_sinks(model_config) or has_attention_logit_softcap(model_config)
        fp32_fallback = "flex_attention" if needs_flex else "sdpa"
        if is_global_main_process():
            logger.warning(
                f"dtype=float32 (fp32 training) is incompatible with {attn_implementation}; "
                f"falling back to attn_implementation='{fp32_fallback}' (FlashAttention is half-precision only)."
            )
        attn_implementation = fp32_fallback

    # FA2 takes head_dim 256 with a softcap and a window, where SDPA would drop the softcap, so the dedicated kernel's
    # refusal runs ahead of the deterministic-backward fallback, which would hand such a model SDPA.
    runs_fa4 = attn_implementation == "flash_attention_4" and declares_flash_attention(model_config)
    if runs_fa4 and (refused := fa4_dedicated_kernel_refusal(model_config)):
        attn_implementation = flash_attention_2_in_place_of_fa4(
            f"FlashAttention-4 trains head_dim-{FA4_DEDICATED_HEAD_DIM} attention on a dedicated backward kernel: "
            f"{refused}"
        )
    elif runs_fa4 and is_fa4_demoted_by_determinism(model_config):
        if is_global_main_process():
            logger.warning(
                "FlashAttention-4's head_dim-256 kernel is not validated with a deterministic backward, which this "
                "run asks for (FLASH_ATTENTION_DETERMINISTIC=1, full_determinism); falling back to "
                f"attn_implementation='{fa4_fallback}'."
            )
        attn_implementation = fa4_fallback

    # Force eager before from_pretrained rejects the auto-detected FA default.
    spec = find_moe_family_spec_for_config(model_config)
    if attn_implementation != "eager" and spec is not None and spec.eager_only_attention:
        if is_global_main_process():
            logger.warning(
                f"{spec.model_types[0]} supports only eager attention ({spec.eager_only_attention}); overriding "
                f"attn_implementation='{attn_implementation}' to 'eager'."
            )
        attn_implementation = "eager"

    if is_flash_attn_implementation(attn_implementation) and has_head_dim_beyond_flash(model_config):
        if is_global_main_process():
            logger.warning(
                f"head_dim={resolve_head_dim(model_config)} attention is unsupported by {attn_implementation} "
                f"(FA2 caps at {FLASH_MAX_HEAD_DIM}; FA4 overflows tensor memory); falling back to "
                f"attn_implementation='sdpa'."
            )
        attn_implementation = "sdpa"

    return validate_attn_implementation(model_config, attn_implementation, sinks_policy, dtype=dtype)
