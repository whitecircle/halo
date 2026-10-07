"""Is this checkpoint multimodal? — the config-level verdict every loader and data path shares.

The model's own classification, kept out of the data package so the loaders reach it without
importing the dataset machinery. Whether a RUN takes the VLM path (checkpoint AND image data) is
:func:`~src.data.vlm.is_vlm_run`.
"""

import logging
import traceback
from typing import Any

from transformers import AutoConfig
from transformers.models.auto.configuration_auto import CONFIG_MAPPING
from transformers.models.auto.modeling_auto import MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING_NAMES

from src.distributed.filesystem import hub_metadata_main_first
from src.distributed.runtime import rank_consensus
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

# Warn once per model path, not once per call: the probe runs from several call sites per rank.
_NAME_HEURISTIC_WARNED: set[str] = set()

# transformers' gate refusing a config whose class is remote code the caller did not trust.
_REMOTE_CODE_GATE = "resolve_trust_remote_code"


def _probe_checkpoint_config(model_name_or_path: str, revision: str | None, trust_remote_code: bool):
    """The checkpoint's config for the modality verdict, ``None`` when the hub read itself fails.

    Only the hub read is caught: this runs inside a store coordination phase, and falling back to
    the name heuristic on a coordination failure would let the timed-out rank pick a different
    ``Auto*`` class than its peers. A remote-code config under ``trust_remote_code: false`` is
    not a hub failure: the model load refuses the same config, after the datasets are in, so it is
    refused here, on every rank alike.
    """
    try:
        return AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=trust_remote_code, revision=revision)
    except Exception as exc:  # unreachable or missing config: fall back to the name heuristic below
        if any(frame.name == _REMOTE_CODE_GATE for frame in traceback.extract_tb(exc.__traceback__)):
            raise ValueError(
                f"'{model_name_or_path}' defines its architecture in remote code, which this run does not "
                f"trust (trust_remote_code: false), so the model load would refuse it. Set "
                f"trust_remote_code: true to run that code."
            ) from exc
        # Warned because the fallback can disagree with the config answer, and this runs before the
        # cache is warm: a rank that reaches the hub and one that does not would pick different
        # Auto* classes for the same run. Divergence must be visible in the log.
        warn_once(
            logger,
            _NAME_HEURISTIC_WARNED,
            model_name_or_path,
            f"Could not load a config for '{model_name_or_path}' ({type(exc).__name__}: {exc}); "
            "falling back to the model-name heuristic to decide text-vs-multimodal. A verdict that "
            "differs across ranks fails the probe on every rank.",
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


def is_vlm_model(
    model_name_or_path: str, config=None, revision: str | None = None, *, trust_remote_code: bool = False
) -> bool:
    """Detect a multimodal (vision-language) model.

    The config is authoritative (:func:`config_declares_multimodality`); the name-substring heuristic
    is the fallback when it cannot be loaded. Pass ``config`` to skip the load; ``revision`` pins the
    fetch, so a revision-pinned checkpoint is not routed by hub ``main``'s config, and
    ``trust_remote_code`` is the run's own setting, so the probe executes no code the load would not.

    A config decides BOTH ways, but only when transformers KNOWS its ``model_type``: for a registered
    architecture the silence is a real text-only verdict, and it must veto the hints, which match
    mid-word ("re**vision**-8472618"). For an unregistered one (remote code) the same silence means
    nothing, so the heuristic still runs and a false positive from it fails loud.
    """
    if config is None:
        return probe_checkpoint(model_name_or_path, revision, trust_remote_code=trust_remote_code)[1]
    return _config_or_name_says_vlm(model_name_or_path, config)


def probe_checkpoint(
    model_name_or_path: str, revision: str | None = None, *, trust_remote_code: bool = False
) -> tuple[Any, bool]:
    """The checkpoint's config and its :func:`is_vlm_model` verdict, from one hub read.

    COLLECTIVE once the process group exists — every rank calls it, as it enters the ``vlm_probe``
    phase. The config is ``None`` when the hub read fails, and the verdict then falls back to the
    name heuristic. An entry script that gates on the config before the model load takes it from
    here rather than paying a second coordinated read.

    The verdict is agreed across ranks and a split raises on every rank: ranks that disagree build
    different model classes and enter different store phases, which surfaces only as a coordination
    timeout. A node that cannot read the source (a local directory on one mount, a Hub cache only some
    nodes see) is the usual cause, and the source agreement that names it runs later, in the load.
    """
    # Main-rank-first: this is the run's FIRST hub contact, ahead of the weight download's own
    # coordination. The fetch handles its own failures; a coordination error propagates rather
    # than resolving to a per-rank heuristic.
    config = hub_metadata_main_first(
        "vlm_probe", lambda: _probe_checkpoint_config(model_name_or_path, revision, trust_remote_code)
    )
    verdict = _config_or_name_says_vlm(model_name_or_path, config)
    all_vlm, any_vlm = rank_consensus(verdict)
    if any_vlm and not all_vlm:
        source = "read its config" if config is not None else "could not read its config and fell back to its name"
        raise RuntimeError(
            f"'{model_name_or_path}' probes as multimodal on some ranks and text-only on others (this "
            f"rank {source}: {'multimodal' if verdict else 'text-only'}), so the ranks would build "
            f"different model classes. Make the same checkpoint readable on every node — a local "
            f"directory on a mount every node sees, or a Hub snapshot in every node's cache — and pin "
            f"model_revision so per-node caches cannot hold different commits."
        )
    return config, verdict


def _config_or_name_says_vlm(model_name_or_path: str, config) -> bool:
    if config is not None:
        if config_declares_multimodality(config):
            return True
        if getattr(config, "model_type", None) in CONFIG_MAPPING:
            return False  # a registered architecture declaring neither → genuinely text-only
    return any(hint in model_name_or_path.lower() for hint in _VLM_NAME_HINTS)
