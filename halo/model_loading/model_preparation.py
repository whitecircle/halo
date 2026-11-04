"""Sharding-agnostic model construction: the widest Auto* class and its verified load, per-family patches, and the run
policies every loaded model takes."""

import logging

from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForImageTextToText,
    PretrainedConfig,
    PreTrainedModel,
)
from transformers.models.auto.modeling_auto import (
    MODEL_FOR_CAUSAL_LM_MAPPING_NAMES,
    MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING_NAMES,
)

import halo.models.seq_cls_heads  # noqa: F401  registers the heads before any Auto* class resolves
from halo.checkpoint.model_card import HUB_TAGS
from halo.model_loading.checkpoint_coverage import from_pretrained_verified, reject_adapter_directory
from halo.model_loading.remote_code import apply_remote_code_compat_shims
from halo.models.attention.backend_setup import (
    install_fa4_varlen_softcap_refusal,
    patch_sdpa_for_wide_heads,
    patch_transformers_flash_unpad_memo,
    patch_transformers_flash_varlen_int_seqlen,
    patch_transformers_packed_sequence_probe,
)
from halo.models.attention.gpt_oss_sinks import SinksPolicy, apply_sinks_policy
from halo.models.attention.prefix_lm import apply_prefix_lm_causal_decoder
from halo.models.attention.sdpa_flex_sliding import resolve_flex_sliding_attn_implementation
from halo.models.attention.selection import (
    fa4_varlen_refusal,
    has_head_dim_beyond_flash,
    is_flash_attn_implementation,
    verify_live_sinks_carried,
)
from halo.models.config_access import drop_toolkit_stamps, model_type_matches, text_config
from halo.models.mistral4 import patch_mistral4_flash_packed_position_ids
from halo.models.model_classes import auto_map_references, mapped_model_class
from halo.models.moe.family_specs import find_moe_family_spec
from halo.models.nemotron_h import apply_nemotron_h_patches
from halo.models.qwen4_exp.forwards import apply_qwen4_exp_patches
from halo.models.zaya import apply_zaya_patches
from halo.runtime.ranks import is_global_main_process

logger = logging.getLogger(__name__)


def _auto_map_class_name(auto_map: dict, key: str) -> str | None:
    """Bare class name of an ``auto_map`` entry's first reference."""
    entry = auto_map.get(key)
    references = auto_map_references(entry) if entry else ()
    return references[0].rsplit(".", 1)[-1] if references else None


def resolve_text_only_config(model_config):
    """The config a ``text_only`` build takes: a family declaring ``text_only_layout`` reaches its CausalLM only through
    its text sub-config's ``model_type``; any other stays composite, which the Auto factory narrows itself."""
    spec = find_moe_family_spec(getattr(model_config, "model_type", None))
    return text_config(model_config) if spec is not None and spec.text_only_layout is not None else model_config


def resolve_auto_model_class(model_config, *, text_only: bool = False):
    """The widest Auto* class: ``AutoModelForImageTextToText`` for multimodal (else the vision tower drops), else
    ``AutoModelForCausalLM``. ``text_only=True`` forces CausalLM, raising for a family with no text-only sibling
    (glm5_next, gemma4, step3p7), which would load the VLM as the text-only model."""
    if text_only:
        model_config = resolve_text_only_config(model_config)
        model_type = getattr(model_config, "model_type", None)
        auto_map = getattr(model_config, "auto_map", None) or {}
        remote_vision = {
            name
            for key in ("AutoModelForImageTextToText", "AutoModelForVision2Seq")
            if (name := _auto_map_class_name(auto_map, key))
        }
        # (source, class name, the multimodal class names it must not be) per CausalLM entry.
        entries = [
            ("its auto_map", _auto_map_class_name(auto_map, "AutoModelForCausalLM"), remote_vision),
            (
                "transformers' CausalLM mapping",
                MODEL_FOR_CAUSAL_LM_MAPPING_NAMES.get(model_type),
                {MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING_NAMES.get(model_type)},
            ),
        ]
        declared = [(source, target, vision) for source, target, vision in entries if target is not None]
        # A CausalLM entry naming the multimodal class itself is that class re-declared for pre-VLM loaders.
        masquerades = [
            (source, target)
            for source, target, vision in declared
            if target.endswith("ForConditionalGeneration") or target in vision
        ]
        if len(masquerades) == len(declared):
            detail = (
                f"{masquerades[0][0]} points AutoModelForCausalLM at {masquerades[0][1]!r} — the multimodal "
                f"conditional-generation class itself, not a text-only sibling"
                if masquerades
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


def resolve_build_config(
    model_name_or_path: str, *, revision: str | None, trust_remote_code: bool
) -> PretrainedConfig:
    """The config a build takes, after the caller's :func:`apply_family_patches`, without toolkit stamps. A build
    reading ``config.json`` itself loses the patches' run-scoped stamps (Qwen3.8-Flash-Next's n-gram table turns
    trainable)."""
    # Before the config fetch, which can already import a remote modeling file.
    apply_remote_code_compat_shims()
    config = AutoConfig.from_pretrained(model_name_or_path, revision=revision, trust_remote_code=trust_remote_code)
    drop_toolkit_stamps(config)
    return config


def declared_load_ignores(config) -> tuple[str, ...]:
    """What the class ``config`` loads through drops at load (``_keys_to_ignore_on_load_unexpected``); none for a
    remote-code class no mapping names, which builds what it stores."""
    auto_class = resolve_auto_model_class(config)
    model_class = mapped_model_class(config, auto_class) or auto_class
    return tuple(getattr(model_class, "_keys_to_ignore_on_load_unexpected", None) or ())


def auto_load_model(
    model_name_or_path: str, *, trust_remote_code: bool = False, model_class=None, text_only: bool = False, **kwargs
):
    """Load a model, picking the widest Auto* class for its config unless ``model_class`` pins one; a ``config`` in
    ``kwargs`` resolves it, else the checkpoint's is fetched."""
    # Before the config fetch, which can already import a remote modeling file.
    apply_remote_code_compat_shims()
    if model_class is None:
        config = kwargs.get("config")
        if config is None:
            reject_adapter_directory(model_name_or_path)
            # revision must reach the config fetch too, else a pinned checkpoint pairs with hub-main's config.
            config = resolve_build_config(
                model_name_or_path, revision=kwargs.get("revision"), trust_remote_code=trust_remote_code
            )
        model_class = resolve_auto_model_class(config, text_only=text_only)
        if text_only and (sibling := resolve_text_only_config(config)) is not config:
            kwargs["config"] = sibling
    return from_pretrained_verified(model_class, model_name_or_path, trust_remote_code=trust_remote_code, **kwargs)


def apply_family_patches(model_config, attn_implementation: str | None) -> str | None:
    """Apply the family/backend-keyed patches every loaded model needs, trainable or frozen, and return the
    implementation to build with. ``None`` applies only the patches no backend keys. Flex-sliding still hands SDPA some
    calls, so a wide-head model keeps the SDPA pin under it."""
    if attn_implementation == "sdpa" and has_head_dim_beyond_flash(model_config):
        patch_sdpa_for_wide_heads()
    # Else each forward's mask construction reads whether position_ids hold packed documents, for a mask flash ignores.
    if is_flash_attn_implementation(attn_implementation):
        patch_transformers_packed_sequence_probe()
    # Else Mistral4's attention swallows position_ids before the flash interface, so the varlen
    # packed path never engages and packed documents attend across each other.
    if is_flash_attn_implementation(attn_implementation) and model_type_matches(model_config, "mistral4"):
        patch_mistral4_flash_packed_position_ids()
    # Else every attention layer of a padded batch re-derives the unpad set, a nonzero and a host read each, and
    # again in the checkpoint recompute.
    if is_flash_attn_implementation(attn_implementation):
        patch_transformers_flash_unpad_memo()
    # Load recording for the native balancing buffer, the GC refusal, and the packed-document plumbing
    # its model forward lacks (it never passes position_ids down to attention or the CCA convs).
    if model_type_matches(model_config, "zaya"):
        apply_zaya_patches()
    # Else packed documents cross the Mamba scan, autocast runs its fp32 body in bf16, the trainer clamps a time step
    # no engine clamps, and Radio's layer scale trains a value no engine reads.
    if model_type_matches(model_config, "nemotron_h"):
        apply_nemotron_h_patches()
    # Else the 95 GiB n-gram table is a trainable parameter every wrap, cast, optimizer and save would carry, every
    # sparse layer reads a [T, T] mask, and packed documents leak through the per-layer embedding.
    if model_type_matches(model_config, "qwen4_exp"):
        apply_qwen4_exp_patches(model_config)
    # Else transformers' 0-dim max_seqlen tensor runs FA4's head_dim-256 varlen kernels on a variant the warm-up does
    # not compile, and a softcapped batch that unpads reaches its varlen kernels.
    if attn_implementation == "flash_attention_4":
        patch_transformers_flash_varlen_int_seqlen()
        if fa4_varlen_refusal(model_config):
            install_fa4_varlen_softcap_refusal()
    return resolve_flex_sliding_attn_implementation(model_config, attn_implementation)


def sanitize_generation_config(model: PreTrainedModel) -> None:
    """Set ``do_sample=True`` where a model ships sampling params without it, which transformers rejects on save."""
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
        if is_global_main_process():
            logger.info(
                "Sanitized generation_config: set do_sample=True (model ships sampling params without do_sample)"
            )


def apply_run_model_policies(
    model, model_config, *, sinks_policy: SinksPolicy, attn_implementation: str, sinks_pretrained: bool = True
) -> None:
    """The run policies every loaded model takes: Hub tags, generation config, prefix-LM causal default, sinks policy.
    ``sinks_pretrained=False`` skips only a neutralized policy, with no pretrained sinks to neutralize. A sink-dropping
    backend over live sinks raises."""
    model.add_model_tags(list(HUB_TAGS))
    sanitize_generation_config(model)
    apply_prefix_lm_causal_decoder(model, model_config)
    if sinks_pretrained or sinks_policy is not SinksPolicy.NEUTRALIZED:
        apply_sinks_policy(model, model_config, policy=sinks_policy, attn_implementation=attn_implementation)
    verify_live_sinks_carried(model)
