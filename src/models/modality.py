"""Is this checkpoint multimodal? The verdict every loader and data path shares, read off a config in
hand or a name. The collective read of a checkpoint's config is
:func:`~src.model_loading.parallel.model_source.probe_checkpoint`; whether a RUN takes the VLM path
(checkpoint AND image data) is :func:`~src.data.vlm.is_vlm_run`.
"""

from transformers.models.auto.configuration_auto import CONFIG_MAPPING
from transformers.models.auto.modeling_auto import MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING_NAMES

_VLM_NAME_HINTS = (
    "-vl",  # Qwen-VL, Qwen2.5-VL, Qwen3-VL
    "vl-",  # InternVL-*, etc.
    "vl2",  # InternVL2
    "llava",  # LLaVA family
    "vision",  # Llama Vision, etc.
    "pixtral",  # Mistral Pixtral
    "molmo",  # Molmo VLM
    "idefics",  # IDEFICS
    "paligemma",  # PaliGemma
    "cogvlm",  # CogVLM
    "minicpm-v",  # MiniCPM-V
    "qwen3.5-",  # Qwen3.5 series (natively multimodal, no "-VL" suffix)
    "qwen3.6-",  # Qwen3.6 series (natively multimodal, no "-VL" suffix)
)


def config_declares_multimodality(config) -> bool:
    """The two config-level multimodality signals, read off a config CLASS or a config INSTANCE.

    First an ``AutoModelForImageTextToText`` entry that is not a plain ``*ForCausalLM``: the exception
    covers text backbones registered there as an upstream quirk (mistral4), which have no processor to
    route to. Then a declared vision tower in either spelling — the class's ``sub_configs`` entry, or a
    live ``vision_config`` that a remote-code config built in ``__init__`` and neither the mapping nor
    ``sub_configs`` sees. Both are read here so the runtime probe and the class-level rosters cannot
    classify a family differently.
    """
    itt_class = MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING_NAMES.get(getattr(config, "model_type", None), "")
    if itt_class and not itt_class.endswith("ForCausalLM"):
        return True
    return "vision_config" in (getattr(config, "sub_configs", None) or {}) or (
        getattr(config, "vision_config", None) is not None
    )


def is_vlm_name(model_name_or_path: str) -> bool:
    """The name-substring heuristic, for a checkpoint whose config cannot decide."""
    return any(hint in model_name_or_path.lower() for hint in _VLM_NAME_HINTS)


def is_vlm_model(model_name_or_path: str, config) -> bool:
    """Detect a multimodal (vision-language) model from its config (:func:`config_declares_multimodality`).

    A config decides BOTH ways, but only when transformers KNOWS its ``model_type``: for a registered
    architecture the silence is a real text-only verdict, and it must veto the hints, which match
    mid-word ("re**vision**-8472618"). For an unregistered one (remote code) the same silence means
    nothing, so :func:`is_vlm_name` decides.
    """
    if config is None:
        raise TypeError(
            f"is_vlm_model({model_name_or_path!r}) needs the checkpoint's config; read it with "
            f"probe_checkpoint, which every rank enters together."
        )
    if config_declares_multimodality(config):
        return True
    if getattr(config, "model_type", None) in CONFIG_MAPPING:
        return False
    return is_vlm_name(model_name_or_path)
