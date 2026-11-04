"""Is this checkpoint multimodal? The verdict every loader and data path shares, off a config in hand or a name."""

from transformers.models.auto.configuration_auto import CONFIG_MAPPING
from transformers.models.auto.modeling_auto import MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING_NAMES

_VLM_NAME_HINTS = (
    "-vl",  # Qwen-VL, Qwen2.5-VL, Qwen3-VL
    "vl-",  # InternVL-*, etc.
    "vl2",  # InternVL2
    "llava",
    "vision",  # Llama Vision, etc.
    "pixtral",
    "molmo",
    "idefics",
    "paligemma",
    "cogvlm",
    "minicpm-v",
    "qwen3.5-",  # natively multimodal, no "-VL" suffix
    "qwen3.6-",  # natively multimodal, no "-VL" suffix
)


def config_declares_multimodality(config) -> bool:
    """Off a config class or instance: an ``AutoModelForImageTextToText`` entry that is not a plain ``*ForCausalLM``
    (text backbones registered there as an upstream quirk, mistral4), or a declared vision tower, in the class's
    ``sub_configs`` or a live ``vision_config`` a remote-code config built in ``__init__``."""
    itt_class = MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING_NAMES.get(getattr(config, "model_type", None), "")
    if itt_class and not itt_class.endswith("ForCausalLM"):
        return True
    return "vision_config" in (getattr(config, "sub_configs", None) or {}) or (
        getattr(config, "vision_config", None) is not None
    )


def is_vlm_name(model_name_or_path: str) -> bool:
    return any(hint in model_name_or_path.lower() for hint in _VLM_NAME_HINTS)


def is_vlm_model(model_name_or_path: str, config) -> bool:
    """A config decides both ways only when transformers knows its ``model_type``: for a registered architecture
    silence vetoes the name hints, which match mid-word (``revision-8472618``); for remote code the name decides."""
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
