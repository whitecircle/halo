"""Is this checkpoint multimodal? — the config-level verdict every loader and data path shares.

The model's own classification, kept out of the data package so the loaders reach it without
importing the dataset machinery. Whether a RUN takes the VLM path (checkpoint AND image data) is
:func:`~src.data.vlm.is_vlm_run`.
"""

import logging

from transformers import AutoConfig
from transformers.models.auto.configuration_auto import CONFIG_MAPPING
from transformers.models.auto.modeling_auto import MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING_NAMES

from src.distributed.filesystem import hub_metadata_main_first
from src.log import warn_once

logger = logging.getLogger(__name__)

# Name-substring fallback for the offline / no-config case; :func:`is_vlm_model`'s config check wins.
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

# Warn once per model path, not once per call: is_vlm_model runs from five call sites per rank.
_NAME_HEURISTIC_WARNED: set[str] = set()


def _probe_checkpoint_config(model_name_or_path: str, revision: str | None):
    """The checkpoint's config for the modality verdict, ``None`` when the hub read itself fails.

    Only the hub read is caught: this runs inside a store coordination phase, and degrading a failure
    of THAT to the name heuristic lets the timed-out rank pick a different ``Auto*`` class than its
    peers — a hang, not a wrong answer. An unreadable config fails for every rank alike, so the
    heuristic verdict stays uniform.
    """
    try:
        return AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=True, revision=revision)
    except Exception as exc:  # unreachable/missing config → fall back to the name heuristic
        # Loud because the fallback can DISAGREE with the config answer, and this runs before the
        # cache is warm: a rank that reaches the hub and one that does not would pick different
        # Auto* classes for the same run. Divergence must be visible in the log.
        warn_once(
            logger,
            _NAME_HEURISTIC_WARNED,
            model_name_or_path,
            f"Could not load a config for '{model_name_or_path}' ({type(exc).__name__}: {exc}); "
            "falling back to the model-name heuristic to decide text-vs-multimodal. If ranks "
            "disagree here they will build different model classes — warm the cache first.",
        )
        return None


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


def is_vlm_model(model_name_or_path: str, config=None, revision: str | None = None) -> bool:
    """Detect a multimodal (vision-language) model.

    The config is authoritative (:func:`config_declares_multimodality`); the name-substring heuristic
    is the fallback when it cannot be loaded. Pass ``config`` to skip the load; ``revision`` pins the
    fetch, so a revision-pinned checkpoint is not routed by hub ``main``'s config.

    A config decides BOTH ways, but only when transformers KNOWS its ``model_type``: for a registered
    architecture the silence is a real text-only verdict, and it must veto the hints, which match
    mid-word ("re**vision**-8472618"). For an unregistered one (remote code) the same silence means
    nothing, so the heuristic still runs and a false positive from it fails loud.
    """
    if config is None:
        # Main-rank-first: this is the run's FIRST hub contact, ahead of the weight download's own
        # coordination, and under trust_remote_code every rank would also race to populate
        # transformers' unlocked dynamic-module cache. The fetch owns its own failure handling —
        # a coordination error must propagate, not resolve to a per-rank heuristic.
        config = hub_metadata_main_first("vlm_probe", lambda: _probe_checkpoint_config(model_name_or_path, revision))
    if config is not None:
        if config_declares_multimodality(config):
            return True
        if getattr(config, "model_type", None) in CONFIG_MAPPING:
            return False  # a registered architecture declaring neither → genuinely text-only
    return any(hint in model_name_or_path.lower() for hint in _VLM_NAME_HINTS)
