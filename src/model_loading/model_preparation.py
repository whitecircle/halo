"""Sharding-agnostic model construction every entry point shares: the widest Auto* class for a config and
the verified load through it, the per-family patches and the implementation a model is built with, and
the run policies every loaded model takes."""

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
from src.checkpoint.model_card import HUB_TAGS
from src.model_loading.checkpoint_coverage import from_pretrained_verified
from src.model_loading.remote_code import apply_remote_code_compat_shims
from src.models.attention.backend_setup import patch_sdpa_for_wide_heads, patch_transformers_flash_varlen_int_seqlen
from src.models.attention.gpt_oss_sinks import SinksPolicy, apply_sinks_policy
from src.models.attention.packed_position_ids import patch_mistral4_flash_packed_position_ids
from src.models.attention.sdpa_flex_sliding import resolve_flex_sliding_attn_implementation
from src.models.attention.selection import has_head_dim_beyond_flash
from src.models.config_access import model_type_matches
from src.models.zaya import apply_zaya_patches

logger = get_logger(__name__)


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
        # A CausalLM entry naming the multimodal class itself is that class re-declared for pre-VLM loaders.
        vision_targets = {
            name
            for key in ("AutoModelForImageTextToText", "AutoModelForVision2Seq")
            if (name := _auto_map_class_name(auto_map, key))
        }
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
    # Bare mapping membership, not ``config_declares_multimodality``: ``mistral4``'s image-text-to-text
    # entry is its text decoder and it has no ``AutoModelForCausalLM`` entry at all.
    if getattr(model_config, "model_type", None) in MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING_NAMES:
        return AutoModelForImageTextToText
    return AutoModelForCausalLM


def concrete_model_class(model_class, model_config):
    """The class an Auto class dispatches ``model_config`` to, which carries the class-level declarations
    (``_tp_plan``, the tie, the load-ignored keys). Returns the Auto class itself for a remote-code
    config outside the mapping."""
    mapping = getattr(model_class, "_model_mapping", None)
    if mapping is None:
        return model_class
    try:
        return mapping[type(model_config)]
    except KeyError:
        return model_class


def auto_load_model(
    model_name_or_path: str, *, trust_remote_code: bool = False, model_class=None, text_only: bool = False, **kwargs
):
    """Load a model from a path, picking the widest Auto* class for its config unless one is pinned.

    The unsharded load core: the compat shims, the config fetch, the class resolution and the
    coverage-verified ``from_pretrained``. ``model_class`` pins the class and skips the config fetch;
    ``text_only`` is :func:`resolve_auto_model_class`'s.
    """
    # Before the config fetch, which can already import a remote modeling file.
    apply_remote_code_compat_shims()
    if model_class is None:
        # revision must reach the config fetch too, else a pinned checkpoint pairs with hub-main's config.
        config = AutoConfig.from_pretrained(
            model_name_or_path, trust_remote_code=trust_remote_code, revision=kwargs.get("revision")
        )
        model_class = resolve_auto_model_class(config, text_only=text_only)
    return from_pretrained_verified(model_class, model_name_or_path, trust_remote_code=trust_remote_code, **kwargs)


def apply_family_patches(model_config, attn_implementation: str) -> str:
    """Apply the family/backend-keyed patches every loaded model needs, trainable or frozen, and return
    the implementation to build it with.

    Keyed on each model's own config, so a frozen teacher or reference of another family gets its own
    patches. The patches key on the resolved backend, the build on its variant
    (:func:`resolve_flex_sliding_attn_implementation`), which still hands SDPA some calls, so a
    wide-head model keeps the SDPA pin under it.
    """
    if attn_implementation == "sdpa" and has_head_dim_beyond_flash(model_config):
        patch_sdpa_for_wide_heads()
    # Else Mistral4's attention swallows position_ids before the flash interface, so the varlen
    # packed path never engages and packed documents attend across each other.
    if attn_implementation.startswith("flash_attention") and model_type_matches(model_config, "mistral4"):
        patch_mistral4_flash_packed_position_ids()
    # Load recording for the native balancing buffer, the GC refusal, and (on flash) the same
    # position_ids plumbing as Mistral4, since Zaya's model forward never passes the tensor down.
    if model_type_matches(model_config, "zaya"):
        apply_zaya_patches(attn_implementation)
    # Else FA4's varlen backward JIT-recompiles every step on transformers' 0-dim max_seqlen tensor.
    if attn_implementation == "flash_attention_4":
        patch_transformers_flash_varlen_int_seqlen()
    return resolve_flex_sliding_attn_implementation(model_config, attn_implementation)


def sanitize_generation_config(model: PreTrainedModel) -> None:
    """Ensure ``model.generation_config`` is valid for ``save_pretrained``.

    Some models ship sampling params with ``do_sample=False``, which transformers rejects on save;
    set ``do_sample=True`` to make the flags consistent.
    """
    generation_config = getattr(model, "generation_config", None)
    if generation_config is None:
        return

    # The neutral values GenerationConfig.validate() tests against, stated because the stock instance
    # carries None for all three; neither None nor a neutral value is a sampling request.
    sampling_params_set = any(
        (value := getattr(generation_config, field, None)) is not None and value != neutral
        for field, neutral in {"temperature": 1.0, "top_p": 1.0, "top_k": 50}.items()
    )
    if sampling_params_set and not getattr(generation_config, "do_sample", False):
        generation_config.do_sample = True
        logger.info("Sanitized generation_config: set do_sample=True (model ships sampling params without do_sample)")


def apply_run_model_policies(
    model, model_config, *, sinks_policy: SinksPolicy, attn_implementation: str, sinks_pretrained: bool = True
) -> None:
    """The sharding-agnostic run policies every loaded model takes, policy and frozen reference alike.

    ``sinks_pretrained=False`` (a from-scratch build) skips only the neutralized sinks policy: there are
    no pretrained sinks to neutralize. The Hub tags go on ``model_tags`` for the card writers that read
    nothing else (PEFT's adapter save, ``push_to_hub``).
    """
    model.add_model_tags(list(HUB_TAGS))
    sanitize_generation_config(model)
    if sinks_pretrained or sinks_policy is not SinksPolicy.NEUTRALIZED:
        apply_sinks_policy(model, model_config, policy=sinks_policy, attn_implementation=attn_implementation)


def log_model_info(model: PreTrainedModel, tokenizer: PreTrainedTokenizer) -> None:
    """Log model and tokenizer information on the main process."""
    logger.info(f"Tokenizer: {tokenizer}")
    logger.info(f"Model config: {model.config}")
