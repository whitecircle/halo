"""Sharding-agnostic model construction every entry point shares: the widest Auto* class for a config and
the verified load through it, the per-family patches and the implementation a model is built with, and
the run policies every loaded model takes."""

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
from halo.model_loading.checkpoint_coverage import from_pretrained_verified
from halo.model_loading.remote_code import apply_remote_code_compat_shims
from halo.models.attention.backend_setup import (
    install_fa4_varlen_softcap_refusal,
    patch_sdpa_for_wide_heads,
    patch_transformers_flash_unpad_memo,
    patch_transformers_flash_varlen_int_seqlen,
    patch_transformers_packed_sequence_probe,
)
from halo.models.attention.gpt_oss_sinks import SinksPolicy, apply_sinks_policy
from halo.models.attention.packed_position_ids import patch_mistral4_flash_packed_position_ids
from halo.models.attention.sdpa_flex_sliding import resolve_flex_sliding_attn_implementation
from halo.models.attention.selection import (
    fa4_varlen_refusal,
    has_head_dim_beyond_flash,
    is_flash_attn_implementation,
)
from halo.models.config_access import drop_toolkit_stamps, model_type_matches
from halo.models.qwen4_exp.forwards import apply_qwen4_exp_patches
from halo.models.zaya import apply_zaya_patches
from halo.runtime.ranks import is_global_main_process

logger = logging.getLogger(__name__)


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
    tail). ``mistral4`` resolves to ``Mistral4ForCausalLM`` even via the
    image-text-to-text mapping (there, ``mistral4`` is the text decoder; ``mistral3`` is the VLM).

    Raises:
        ValueError: ``text_only=True`` for a family with no text-only CausalLM sibling: no
            ``AutoModelForCausalLM`` entry at all (glm5_next ships only the conditional-generation
            wrapper), or an entry naming the multimodal class itself, native (gemma4) or in the
            ``auto_map`` (step3p7). Accepting it would load the VLM as the text-only model.
    """
    if text_only:
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
    """The config a build of ``model_name_or_path`` takes as ``config``, once the caller has applied
    :func:`apply_family_patches` to it: read at ``revision`` after the remote-code shims, without the toolkit stamps a
    checkpoint's ``config.json`` can carry.

    A build that reads ``config.json`` itself loses the run-scoped stamps the patches set, so Qwen3.8-Flash-Next's
    n-gram table builds as a trainable ``nn.Embedding`` instead of a host table.
    """
    # Before the config fetch, which can already import a remote modeling file.
    apply_remote_code_compat_shims()
    config = AutoConfig.from_pretrained(model_name_or_path, revision=revision, trust_remote_code=trust_remote_code)
    drop_toolkit_stamps(config)
    return config


def auto_load_model(
    model_name_or_path: str, *, trust_remote_code: bool = False, model_class=None, text_only: bool = False, **kwargs
):
    """Load a model from a path, picking the widest Auto* class for its config unless one is pinned.

    The unsharded load core: the compat shims, the config fetch, the class resolution and the
    coverage-verified ``from_pretrained``. ``model_class`` pins the class; otherwise a ``config`` in ``kwargs``
    (the one the build takes) resolves it, or the checkpoint's is fetched. ``text_only`` is
    :func:`resolve_auto_model_class`'s.
    """
    # Before the config fetch, which can already import a remote modeling file.
    apply_remote_code_compat_shims()
    if model_class is None:
        config = kwargs.get("config")
        if config is None:
            # revision must reach the config fetch too, else a pinned checkpoint pairs with hub-main's config.
            config = AutoConfig.from_pretrained(
                model_name_or_path, trust_remote_code=trust_remote_code, revision=kwargs.get("revision")
            )
        model_class = resolve_auto_model_class(config, text_only=text_only)
    return from_pretrained_verified(model_class, model_name_or_path, trust_remote_code=trust_remote_code, **kwargs)


def apply_family_patches(model_config, attn_implementation: str | None) -> str | None:
    """Apply the family/backend-keyed patches every loaded model needs, trainable or frozen, and return
    the implementation to build it with.

    Keyed on each model's own config, so a frozen teacher or reference of another family gets its own
    patches. The patches key on the resolved backend, the build on its variant
    (:func:`resolve_flex_sliding_attn_implementation`), which still hands SDPA some calls, so a
    wide-head model keeps the SDPA pin under it. ``attn_implementation`` None (a build that picks its own
    backend) applies only the patches no backend keys and returns None.
    """
    if attn_implementation == "sdpa" and has_head_dim_beyond_flash(model_config):
        patch_sdpa_for_wide_heads()
    # Else every forward's mask construction reads back whether position_ids hold packed documents, for a packed
    # mask flash never reads.
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
        if is_global_main_process():
            logger.info(
                "Sanitized generation_config: set do_sample=True (model ships sampling params without do_sample)"
            )


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
