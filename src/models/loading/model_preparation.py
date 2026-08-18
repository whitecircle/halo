"""Model construction shared by every entry point: the widest Auto* class for a config and the
verified load through it, the per-family attention patches, and the post-load generation-config /
Liger fixups.

The rules it builds on live beside it (config levels, dtype, tokenizer setup); the callers that
place a model on a parallelism live in :mod:`src.distributed.loading`, above it.
"""

from accelerate.logging import get_logger
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForImageTextToText,
    PreTrainedModel,
    PreTrainedTokenizer,
)
from transformers.models.auto.modeling_auto import (
    MODEL_FOR_CAUSAL_LM_MAPPING_NAMES,
    MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING_NAMES,
)

import src.models.seq_cls_heads  # noqa: F401  registers the heads before any Auto* class resolves
from src.kernels.liger.orchestrator import LIGER_APPLIED_CONFIG_ATTR
from src.models.loading.checkpoint_coverage import from_pretrained_verified
from src.models.patches.attention import (
    model_is_gemma4,
    model_is_mistral4,
    model_is_zaya,
    patch_mistral4_flash_packed_position_ids,
    patch_sdpa_for_gemma4_long_seq,
    patch_transformers_flash_varlen_int_seqlen,
)
from src.models.patches.gpt_oss_sinks import SinksPolicy, apply_sinks_policy
from src.models.patches.remote_code_compat import apply_remote_code_compat_shims
from src.models.patches.zaya import apply_zaya_patches

logger = get_logger(__name__)

# Neutral sampling values, the thresholds GenerationConfig.validate() warns against. Stated
# explicitly because the stock GenerationConfig() instance carries None for all three fields, so
# comparing against the instance would read a neutral temperature=1.0 as a sampling request.
_NEUTRAL_SAMPLING_VALUES = {"temperature": 1.0, "top_p": 1.0, "top_k": 50}
_SAMPLING_FIELDS = tuple(_NEUTRAL_SAMPLING_VALUES)


# Auto-map keys under which a hub repo names its multimodal (vision-bearing) class. Read to spot a
# CausalLM auto_map entry that is really that class re-declared for pre-VLM loaders.
_VISION_AUTO_MAP_KEYS = ("AutoModelForImageTextToText", "AutoModelForVision2Seq")


def _auto_map_class_name(auto_map: dict, key: str) -> str | None:
    """Bare class name of an ``auto_map`` entry (``"module.Class"``, or hub's two-slot list form)."""
    target = auto_map.get(key)
    if isinstance(target, (tuple, list)):
        target = next((entry for entry in target if entry), None)
    if not target:
        return None
    return str(target).rsplit(".", 1)[-1]


def resolve_auto_model_class(model_config, *, text_only: bool = False):
    """Pick the widest Auto* class: ``AutoModelForImageTextToText`` for multimodal (else the vision tower
    is dropped), ``AutoModelForCausalLM`` for text-only.

    ``text_only=True`` forces ``AutoModelForCausalLM`` for a multimodal config, i.e. the deliberate
    text-only load of a VLM checkpoint: the CausalLM sibling drops the vision tower (and any MTP
    tail) and, unlike the wrapper's forward, honors ``config.output_router_logits``, which is what
    ``moe_balancing: aux_loss`` needs. ``mistral4`` resolves to ``Mistral4ForCausalLM`` even via the
    image-text-to-text mapping (there, ``mistral4`` is the text decoder; ``mistral3`` is the VLM).

    Raises:
        ValueError: ``text_only=True`` for a family with no text-only CausalLM sibling: no
            ``AutoModelForCausalLM`` entry at all (glm5_next ships only the conditional-generation
            wrapper), or an ``auto_map`` entry pointing that key at the multimodal class itself
            (step3p7). Accepting it would load the remote-code VLM as the text-only model.
    """
    if text_only:
        model_type = getattr(model_config, "model_type", None)
        has_native_sibling = model_type in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES
        auto_map = getattr(model_config, "auto_map", None) or {}
        remote_target = _auto_map_class_name(auto_map, "AutoModelForCausalLM")
        vision_targets = {name for key in _VISION_AUTO_MAP_KEYS if (name := _auto_map_class_name(auto_map, key))}
        masquerading = remote_target is not None and (
            remote_target.endswith("ForConditionalGeneration") or remote_target in vision_targets
        )
        has_remote_sibling = remote_target is not None and not masquerading
        if not (has_native_sibling or has_remote_sibling):
            detail = (
                f"its auto_map points AutoModelForCausalLM at {remote_target!r} — the multimodal "
                f"conditional-generation class itself, not a text-only sibling"
                if masquerading
                else "transformers maps no text-only *ForCausalLM class for this family"
            )
            raise ValueError(
                f"text_only_model=True cannot load model_type={model_type!r}: {detail}, so there is "
                f"no text-only sibling to load the checkpoint into. Load it as its native "
                f"multimodal class (text_only_model: false)."
            )
        return AutoModelForCausalLM
    # Bare mapping membership rather than the value-inspecting
    # :func:`~src.models.modality.config_declares_multimodality`: the two disagree on ``mistral4``,
    # whose ITT entry is its text decoder and which has no ``AutoModelForCausalLM`` entry at all, so
    # unifying them would leave that family with no loadable class here, and would send it down the
    # VLM branch (a processor it does not ship) at every reader of that predicate.
    if getattr(model_config, "model_type", None) in MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING_NAMES:
        return AutoModelForImageTextToText
    return AutoModelForCausalLM


def auto_load_model(model_name_or_path: str, *, trust_remote_code: bool = False, model_class=None, **kwargs):
    """Load a model from a path, picking the widest Auto* class for its config unless one is pinned.

    The unsharded load core: the compat shims, the config fetch, the class resolution and the
    coverage-verified ``from_pretrained``. Every caller holding only a path string runs it;
    distributed paths use ``load_distributed_model()``.

    ``model_class`` pins the class and skips the config fetch, for a caller that resolved its own
    (a task head).
    """
    # Before the config fetch, which is already enough to import a remote modeling file: every
    # standalone tool loads through here, so the shims cannot depend on each caller applying them
    # first (a CLI is the process's first remote-code load by definition). Idempotent.
    apply_remote_code_compat_shims()
    if model_class is None:
        # revision must reach the config fetch too, else a pinned checkpoint pairs with hub-main's config.
        config = AutoConfig.from_pretrained(
            model_name_or_path, trust_remote_code=trust_remote_code, revision=kwargs.get("revision")
        )
        model_class = resolve_auto_model_class(config)
    return from_pretrained_verified(model_class, model_name_or_path, trust_remote_code=trust_remote_code, **kwargs)


def apply_family_attention_patches(model_config, attn_implementation: str) -> None:
    """The family/backend-keyed attention patches every loaded model needs, trainable or frozen.

    Shared by both loaders: a frozen teacher or reference paired with a different-family policy
    would otherwise miss its family's patch (the policy's load keys on the policy's config) and, for
    example, score packed rows with cross-document flash attention while the student is isolated.
    """
    if attn_implementation == "sdpa" and model_is_gemma4(model_config):
        patch_sdpa_for_gemma4_long_seq()
    # Else Mistral4's attention swallows position_ids before the flash interface, so the varlen
    # packed path never engages and packed documents attend across each other.
    if attn_implementation.startswith("flash_attention") and model_is_mistral4(model_config):
        patch_mistral4_flash_packed_position_ids()
    # Load recording for the native balancing buffer, the GC refusal, and (on flash) the same
    # position_ids plumbing as Mistral4, since Zaya's model forward never passes the tensor down.
    if model_is_zaya(model_config):
        apply_zaya_patches(attn_implementation)
    # Else FA4's varlen backward JIT-recompiles every step on transformers' 0-dim max_seqlen tensor.
    if attn_implementation == "flash_attention_4":
        patch_transformers_flash_varlen_int_seqlen()


def sanitize_generation_config(model: PreTrainedModel) -> None:
    """Ensure ``model.generation_config`` is valid for ``save_pretrained``.

    Some models ship sampling params with ``do_sample=False``, which transformers rejects on save;
    set ``do_sample=True`` to make the flags consistent.
    """
    generation_config = getattr(model, "generation_config", None)
    if generation_config is None:
        return

    # None means explicitly unset rather than a sampling request; a neutral value is not one either.
    sampling_params_set = any(
        (value := getattr(generation_config, field, None)) is not None and value != _NEUTRAL_SAMPLING_VALUES[field]
        for field in _SAMPLING_FIELDS
    )
    if sampling_params_set and not getattr(generation_config, "do_sample", False):
        generation_config.do_sample = True
        logger.info("Sanitized generation_config: set do_sample=True (model ships sampling params without do_sample)")


def finalize_run_model(
    model, model_config, *, sinks_policy: SinksPolicy, attn_implementation: str, sinks_pretrained: bool = True
) -> None:
    """The sharding-agnostic finalization every loaded model needs, in a fixed order.

    Shared by the policy dispatcher and the frozen auxiliary loader, so a reference or teacher is
    finalized exactly like the policy whose logprobs it is differenced against.
    ``sinks_pretrained=False`` (a from-scratch build) skips only the neutralized policy: there are
    no pretrained sinks to neutralize, and the live policies still apply.
    """
    sanitize_generation_config(model)
    if sinks_pretrained or sinks_policy is not SinksPolicy.NEUTRALIZED:
        apply_sinks_policy(model, model_config, policy=sinks_policy, attn_implementation=attn_implementation)


def finalize_liger_after_direct_load(training_config, original_use_liger: bool, model) -> None:
    """Reconcile TRL's ``use_liger_kernel`` flag after a direct model load already applied Liger.

    Reads the effective config the orchestrator applied (per-model defaults and EP/TP force-offs
    included), not the user's raw ``liger_kernel_config``. Under ``fused_linear_cross_entropy`` the
    model returns no logits, so the flag must stay on or TRL slices ``None`` in its metric path
    (FLCE-only families: DeepSeek-V4, GLM-4 MoE Lite, Zaya); pinning the applied config makes TRL's
    re-application an identical re-patch. Otherwise the flag is cleared, so TRL does not re-apply
    Liger with its own defaults over the patched modules.
    """
    if not original_use_liger:
        return
    applied = getattr(model.config, LIGER_APPLIED_CONFIG_ATTR, None)
    if applied and applied.get("fused_linear_cross_entropy", False):
        training_config.use_liger_kernel = True
        training_config.liger_kernel_config = dict(applied)
    else:
        training_config.use_liger_kernel = False


def log_model_info(model: PreTrainedModel, tokenizer: PreTrainedTokenizer) -> None:
    """Log model and tokenizer information on the main process."""
    logger.info(f"Tokenizer: {tokenizer}")
    logger.info(f"Model config: {model.config}")
