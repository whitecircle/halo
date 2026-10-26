"""Attention-implementation selection: the flash vocabulary, the backend resolver and its validator.

Picks ``attn_implementation`` (FA4/FA3, then FA2, SDPA, flex, eager) by GPU capability and the model
class's support flags, including which implementations may run against GptOss's live sinks.
"""

from __future__ import annotations

import importlib
import inspect
import logging

import torch
from transformers import AutoModelForCausalLM, AutoModelForImageTextToText
from transformers.utils import is_flash_attn_2_available

from halo.hardware import is_blackwell_gpu, is_hopper_gpu
from halo.kernels.jit_cache import ensure_fa4_kernel_cache_env
from halo.models.attention.gpt_oss_sinks import (
    RESOLVED_SINKS_RESET_ATTR,
    SinksPolicy,
    install_local_flash_for_sinks,
    install_sdpa_for_reset_sinks,
    uses_gpt_oss_sinks,
)
from halo.models.attention.layout import GDN_MODEL_TYPE_PREFIXES, declares_attention_heads, resolve_head_dim
from halo.models.config_access import model_type_matches, set_config_field_run_scoped, text_config
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

# Varlen backends consume cu_seqlens from packed position_ids. Dense backends isolate packed
# documents only where the model plumbs position_ids into its mask — see agent-docs/data/collators.md.
VARLEN_ATTN_IMPLEMENTATIONS = tuple(_FLASH_KERNEL_MODULES)

# The two names transformers forwards a learnable sink under: FA3's ``s_aux`` and its
# ``_flash_api_alternative_names`` entry, FA4's ``learnable_sink``.
FLASH_SINK_PARAM_NAMES = ("s_aux", "learnable_sink")

# The widest head FA2 takes; FA4's SM100 kernel overflows tensor memory past it.
FLASH_MAX_HEAD_DIM = 256


def is_flash_attn_implementation(attn_impl: str | None) -> bool:
    """Whether ``attn_impl`` names a FlashAttention backend (the varlen kernels)."""
    return attn_impl in VARLEN_ATTN_IMPLEMENTATIONS


def flash_kernel(attn_impl: str, *, varlen: bool = True):
    """The ``flash_attn_varlen_func`` (or, with ``varlen=False``, ``flash_attn_func``) transformers
    dispatches for ``attn_impl``; ``ImportError`` when no installed build has it. Varlen by default:
    transformers reads the capabilities of both calls off ``_flash_varlen_fn`` alone.
    """
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


def preferred_flash_impl(*, allow_fa4: bool = True) -> str:
    """The flash backend this GPU's architecture runs: FA4 on Blackwell (unless ``allow_fa4`` is False), FA3
    on Hopper, else FA2. Whether its build imports is each caller's own probe."""
    if allow_fa4 and is_blackwell_gpu():
        return "flash_attention_4"
    if is_hopper_gpu():
        return "flash_attention_3"
    return "flash_attention_2"


def _detect_attention_impl() -> str:
    """:func:`preferred_flash_impl` when its build imports, else FA2. A broken (not absent) build warns
    before degrading, since nothing else would report the slower kernel."""
    if not torch.cuda.is_available():
        return "flash_attention_2"  # CPU host; the choice is re-validated against the model downstream
    preferred = preferred_flash_impl()
    if preferred != "flash_attention_2":
        build = _FLASH_KERNEL_MODULES[preferred][-1]
        try:
            flash_kernel(preferred)
            if is_global_main_process():
                logger.info(f"Using {preferred} ({build} installed)")
            return preferred
        except ImportError:
            if is_global_main_process():
                logger.info(f"{build} not installed, using FlashAttention-2")
        except Exception as e:
            if is_global_main_process():
                logger.warning(
                    f"{build} IS present but unusable ({type(e).__name__}: {e}); falling back to "
                    f"FlashAttention-2, which is correct but slower on attention. Repair the {build} install "
                    f"to regain {preferred}."
                )
    if is_global_main_process():
        logger.info("Using FlashAttention-2")
    return "flash_attention_2"


def effective_attn_implementation(model_config) -> str | None:
    """The attention backend this config's decoder runs, or ``None`` when none is recorded. Read off the
    text sub-config first: composite wrappers record it there and only sometimes mirror it upward."""
    if model_config is None:
        return None
    return getattr(text_config(model_config), "_attn_implementation", None) or getattr(
        model_config, "_attn_implementation", None
    )


def has_head_dim_beyond_flash(model_config) -> bool:
    """Whether the widest attention head is past :data:`FLASH_MAX_HEAD_DIM` (Gemma 4's 512-wide global
    heads), where no flash kernel runs; ``False`` on an attention-free config (Mamba)."""
    return declares_attention_heads(model_config) and resolve_head_dim(model_config) > FLASH_MAX_HEAD_DIM


def is_fa4_backward_nan_prone(model_config) -> bool:
    """Whether the FA4 backward emits NaN gradients for this model (256-wide qk heads, partially rotated): a MoE
    family's spec says, and every GatedDeltaNet family without one (dense Qwen3.5, Qwen3-Next) is.

    Keyed on the family, not the numbers: Qwen3-Next declares ``head_dim: 256`` with a partial-rotary
    factor, while MLA GLM-4 MoE Lite reaches 256 as ``qk_nope_head_dim + qk_rope_head_dim``.
    """
    spec = find_moe_family_spec_for_config(model_config)
    if spec is not None:
        return spec.fa4_backward_nan_prone
    return model_type_matches(model_config, *GDN_MODEL_TYPE_PREFIXES)


def _attn_impl_handles_sinks(attn_impl: str) -> bool:
    """Whether an implementation carries the learnable sink column into the softmax. A flash kernel's
    verdict is read off its signature, so a flash-attn build that adds sink support needs no code change."""
    if attn_impl in ("eager", "flex_attention"):
        return True
    if not is_flash_attn_implementation(attn_impl):
        return False
    try:
        params = inspect.signature(flash_kernel(attn_impl)).parameters
    except ImportError:
        return False
    return any(name in params for name in FLASH_SINK_PARAM_NAMES)


def validate_attn_implementation(model_config, attn_impl: str, sinks_policy: SinksPolicy | None = None) -> str:
    """Return ``attn_impl`` or the first fallback the model class's ``_supports_*`` flags and the sink
    matrix accept. A neutralized ``sinks_policy`` allows SDPA on a sinks model (the dropped column is 0),
    a live one never; ``None`` re-validates against the verdict already recorded on this config.
    """
    has_sinks = uses_gpt_oss_sinks(model_config)
    sinks_reset = None if sinks_policy is None else not sinks_policy.live
    if attn_impl == "sdpa" and has_sinks and sinks_reset:
        install_sdpa_for_reset_sinks(model_config)

    # Recorded per config instance, so nested re-validations see it while a fresh config cannot inherit
    # an earlier model's sdpa approval; run-scoped, so no exported config.json carries it.
    if sinks_reset is not None:
        set_config_field_run_scoped(model_config, RESOLVED_SINKS_RESET_ATTR, sinks_reset)
    recorded_reset = getattr(model_config, RESOLVED_SINKS_RESET_ATTR, None)

    # A composite VLM with no CausalLM sibling declares its flags on the image-text class. An unmapped
    # config keeps sdpa allowed (the model build raises if it must) and the sinks escape hatch closed.
    resolved_class = mapped_model_class(model_config, AutoModelForCausalLM, AutoModelForImageTextToText)
    supports_sdpa = getattr(resolved_class, "_supports_sdpa", True)
    supports_flash = getattr(resolved_class, "_supports_flash_attn", True)

    # sdpa drops sinks, so it is valid only without sinks, or when they were reset (here or earlier).
    sdpa_ok = supports_sdpa and (not has_sinks or recorded_reset is True)

    live_sinks = has_sinks and recorded_reset is False
    if live_sinks and not _attn_impl_handles_sinks(attn_impl):
        raise ValueError(
            f"attn_implementation='{attn_impl}' silently drops the learnable attention sinks that "
            f"reset_sinks=false keeps live — every training logprob would shift by nats vs the served "
            f"policy. Use a sink-carrying implementation (flash_attention_4 on Blackwell; flex_attention "
            f"or eager on Hopper — the stock Hopper flash_attention_3 build lacks s_aux, so FA3 works "
            f"only with a build exposing it), or set reset_sinks: true for off-policy fine-tuning."
        )

    # Fallback chain: sinks models prefer FA2 over flex, whose compile path is unusable under EP.
    candidates = [attn_impl]
    if has_sinks:
        if live_sinks:
            candidates.append("flash_attention_4")
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
            logger.warning(
                f"Model {type(model_config).__name__} does not support "
                f"attn_implementation='{attn_impl}', using '{candidate}'"
            )
        if is_flash_attn_implementation(candidate):
            install_local_flash_for_sinks(model_config)
        return candidate
    raise ValueError(
        f"{type(model_config).__name__} accepts none of the attention implementations tried "
        f"({', '.join(candidates)}); set attn_implementation to one this model's config accepts."
    )


def revalidate_attn_kwarg(model_kwargs: dict, model_config) -> None:
    """Re-validate a ``from_pretrained`` kwargs' ``attn_implementation`` against ``model_config`` in
    place, so a family whose ``_supports_*`` flags refuse the policy's choice fails before the load."""
    attn_impl = model_kwargs.pop("attn_implementation", None)
    if attn_impl is not None:
        model_kwargs["attn_implementation"] = validate_attn_implementation(model_config, attn_impl)


def resolve_attn_implementation(
    model_config,
    attn_implementation: str | None,
    dtype: torch.dtype,
    *,
    sinks_policy: SinksPolicy,
    fa4_nan_fallback: str = "sdpa",
) -> str:
    """The attention backend for ``model_config``: auto-detected when ``attn_implementation`` is None,
    narrowed by the per-family kernel limits (fp32, FA4 backward NaN, DeepSeek-V4, heads past the flash
    cap), then validated. Parallelism-specific overrides stay with the caller that knows the topology,
    among them ``fa4_nan_fallback``, the label an FA4-backward-NaN family takes in place of FA4.
    """
    attn_implementation = attn_implementation or _detect_attention_impl()

    if dtype == torch.float32 and is_flash_attn_implementation(attn_implementation):
        fp32_fallback = "flex_attention" if uses_gpt_oss_sinks(model_config) else "sdpa"
        if is_global_main_process():
            logger.warning(
                f"dtype=float32 (fp32 training) is incompatible with {attn_implementation}; "
                f"falling back to attn_implementation='{fp32_fallback}' (FlashAttention is half-precision only)."
            )
        attn_implementation = fp32_fallback

    if attn_implementation == "flash_attention_4" and is_fa4_backward_nan_prone(model_config):
        if is_global_main_process():
            logger.warning(
                "FlashAttention-4 produces NaN gradients for this model's head_dim-256 partial-rotary "
                f"attention; falling back to attn_implementation='{fa4_nan_fallback}'."
            )
        attn_implementation = fa4_nan_fallback

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

    return validate_attn_implementation(model_config, attn_implementation, sinks_policy)
