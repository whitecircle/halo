"""Shared setup for Vision-Language Model (VLM) training.

- :func:`load_model_for_training` — modality-aware entry point; auto-detects VLM vs text and dispatches.
- :func:`load_vlm_model_and_processor` — processor/tokenizer load, then delegates the model load to
  ``load_distributed_model``.
- :func:`load_vlm_processor` — the processor load on its own, for the scripts that build their model
  through a task-head loader of their own (reward modeling) and only need the processing class.
- :func:`require_multimodal_sequence_classification_head` — pre-load gate for the pooled-head
  scripts on a multimodal checkpoint.

Dataset-side preparation lives in :mod:`src.data.pipeline.vlm_dataset`.
"""

from accelerate.logging import get_logger
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoProcessor, AutoTokenizer
from transformers.models.auto.configuration_auto import CONFIG_MAPPING
from transformers.models.auto.modeling_auto import MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING_NAMES
from trl import ModelConfig

from src.distributed.filesystem import hub_metadata_main_first
from src.distributed.loading.model_loading import load_distributed_model
from src.distributed.parallelism_config import ParallelismConfig
from src.distributed.runtime import is_global_main_process
from src.models.loading.dtype import resolve_quantization_config, resolve_training_dtype
from src.models.loading.model_preparation import finalize_liger_after_direct_load
from src.models.modality import config_declares_multimodality, is_vlm_model

logger = get_logger(__name__, log_level="INFO")


def load_model_consuming_init_kwargs(
    model_config: ModelConfig,
    training_config,
    parallelism_config: ParallelismConfig,
    *,
    weights_source: str | None,
    attn_default: str | None,
    trust_remote_code: bool,
    revision: str | None,
    model_config_overrides: dict | None = None,
    **loader_kwargs,
):
    """``load_distributed_model`` with ``model_init_kwargs`` consumed and Liger re-finalized.

    The training config's ``model_init_kwargs`` are forwarded to the loader (``model_config_overrides``,
    a script's own overrides such as ``num_labels``, merged over them) and cleared off the config so
    TRL doesn't re-apply them, and the Liger flags are captured before the load so
    ``finalize_liger_after_direct_load`` can replay them on the already-built model. Returns
    ``(model, tokenizer)``.
    """
    # Non-TRL configs (DistillationConfig, the plain HF classification/embedding configs) have no
    # model_init_kwargs — nothing to consume, nothing to clear.
    overrides = dict(getattr(training_config, "model_init_kwargs", None) or {})
    if hasattr(training_config, "model_init_kwargs"):
        training_config.model_init_kwargs = None
    overrides.update(model_config_overrides or {})
    original_use_liger = training_config.use_liger_kernel
    original_liger_config = training_config.liger_kernel_config
    model, tokenizer = load_distributed_model(
        model_name_or_path=weights_source or model_config.model_name_or_path,
        parallelism_config=parallelism_config,
        dtype=resolve_training_dtype(training_config),
        trust_remote_code=trust_remote_code,
        attn_implementation=model_config.attn_implementation or attn_default,
        model_config_overrides=overrides,
        use_liger_kernel=original_use_liger,
        liger_kernel_config=original_liger_config,
        quantization_config=resolve_quantization_config(model_config, training_config),
        revision=revision,
        **loader_kwargs,
    )
    finalize_liger_after_direct_load(training_config, original_use_liger, model)
    return model, tokenizer


def load_vlm_processor(model_config: ModelConfig):
    """Load the checkpoint's ``AutoProcessor``, pinned to the model revision.

    The one home for the call: an unpinned processor silently loads hub ``main``, whose image
    geometry (patch size, pixel budget) can differ from the commit the run trains. Being the
    trainer's ``processing_class`` is also what puts ``processor_config.json`` next to the
    weights in every exported checkpoint — without it the artifact trains but cannot be served.

    Main-rank-first, like every other pre-download hub read: the processor pulls several small
    files and, under ``trust_remote_code``, writes transformers' unlocked dynamic-module cache.
    """
    return hub_metadata_main_first(
        "vlm_processor",
        lambda: AutoProcessor.from_pretrained(
            model_config.model_name_or_path,
            trust_remote_code=True,
            revision=getattr(model_config, "model_revision", None),
        ),
    )


def multimodal_sequence_classification_model_types() -> list[str]:
    """Model types that have BOTH a registered sequence-classification head and a vision tower.

    Read off the live registry — the transformers mapping plus the ``_extra_content`` the toolkit's
    own heads register into it (:mod:`src.models.seq_cls_heads`) — so a family added upstream or by
    the toolkit appears here without anyone maintaining a list. The modality verdict comes from the
    same :func:`~src.models.modality.config_declares_multimodality` the run-time probe uses,
    read off the config CLASS (instantiating every registered config to ask would be slow and
    fragile), so this roster cannot name a family that probe would call text-only, or omit one it
    would call multimodal.
    """
    # ``_extra_content`` is the same storage src/models/seq_cls_heads.py writes the toolkit heads into.
    config_classes = list(AutoModelForSequenceClassification._model_mapping._extra_content)
    config_classes += [
        CONFIG_MAPPING[name] for name in MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING_NAMES if name in CONFIG_MAPPING
    ]
    return sorted({cls.model_type for cls in config_classes if config_declares_multimodality(cls)})


def require_multimodal_sequence_classification_head(model_config: ModelConfig) -> None:
    """Gate a pooled-head script (reward modeling) on a multimodal checkpoint.

    ``AutoModelForSequenceClassification`` refuses a family it has no head for with "Unrecognized
    configuration class", naming neither the modality nor any alternative — and it does so only
    after the distributed init, on every rank. Raising here instead names the families that do work,
    from the same registry the resolution reads.
    """
    config = hub_metadata_main_first(
        "seq_cls_head_probe",
        lambda: AutoConfig.from_pretrained(
            model_config.model_name_or_path,
            trust_remote_code=model_config.trust_remote_code,
            revision=getattr(model_config, "model_revision", None),
        ),
    )
    # Exactly the containment test ``from_config`` resolves through, so this cannot disagree with it.
    if not is_vlm_model(model_config.model_name_or_path, config=config) or (
        type(config) in AutoModelForSequenceClassification._model_mapping
    ):
        return
    raise ValueError(
        f"'{model_config.model_name_or_path}' is a multimodal checkpoint "
        f"(model_type={getattr(config, 'model_type', None)!r}) with no sequence-classification "
        f"head, so it cannot carry a reward/classification score head. Families that can: "
        f"{multimodal_sequence_classification_model_types()}."
    )


def load_vlm_model_and_processor(
    model_config: ModelConfig,
    training_config,
    parallelism_config: ParallelismConfig,
    weights_source: str | None = None,
    attn_default: str | None = None,
    *,
    reset_sinks: bool = True,
    train_sinks: bool = False,
):
    """Load a VLM model + processor + tokenizer for distributed training. Returns ``(model, processor, tokenizer)``.

    Delegating the model load to ``load_distributed_model`` (not a bare ``from_pretrained``) is what gives
    MoE VLMs full EP/TP/CP wrapping + attention fallbacks + QLoRA + Liger. ``model_init_kwargs`` is consumed
    so TRL doesn't re-apply it. ``weights_source`` overrides where the weights load from (EP/CP resume
    checkpoint); the processor/tokenizer still load from ``model_config.model_name_or_path``. The sinks
    flags reach the loader exactly as on the text branch, so a contradictory pair is refused here too.

    The caller has already resolved the modality verdict; this function must not re-probe it:
    :func:`~src.models.modality.is_vlm_model` enters the ``vlm_probe`` store phase, and only
    multimodal runs reach here — a phase entered from a branch some ranks do not take leaves the rest
    waiting on a key nobody writes.
    """
    # Same revision as the model weights — an unpinned processor/tokenizer silently loads hub main.
    revision = getattr(model_config, "model_revision", None)

    processor = load_vlm_processor(model_config)
    tokenizer = (
        processor.tokenizer
        if hasattr(processor, "tokenizer")
        else hub_metadata_main_first(
            "vlm_tokenizer",
            lambda: AutoTokenizer.from_pretrained(
                model_config.model_name_or_path, trust_remote_code=True, revision=revision
            ),
        )
    )

    model, _ = load_model_consuming_init_kwargs(
        model_config,
        training_config,
        parallelism_config,
        weights_source=weights_source,
        attn_default=attn_default,
        trust_remote_code=True,
        revision=revision,
        reset_sinks=reset_sinks,
        train_sinks=train_sinks,
    )

    if is_global_main_process():
        logger.info(f"Loaded VLM model: {type(model).__name__}")

    return model, processor, tokenizer


def load_model_for_training(
    model_config: ModelConfig,
    training_config,
    parallelism_config: ParallelismConfig,
    *,
    attn_default: str | None = None,
    reset_sinks: bool = True,
    train_sinks: bool = False,
    init_from_scratch: bool = False,
    weights_source: str | None = None,
    text_only_model: bool = False,
):
    """Modality-aware model load — the single entry point for every training script.

    Returns ``(model, processing_class, tokenizer, is_vlm)``: VLM → processor as processing_class; text →
    tokenizer. Both apply QLoRA + parallelism-aware Liger, leaving parallelism wrapping to the trainer.
    ``attn_default`` is the fallback attn impl when the config sets none. ``init_from_scratch`` (text only)
    loads fresh weights. ``weights_source`` overrides where the weights load from (EP/CP resume checkpoint).
    ``text_only_model`` skips the VLM branch entirely — the multimodal checkpoint loads through its
    text-only CausalLM sibling with the text processing path. The flag is a rank-uniform YAML key,
    so every rank takes the same branch and any ``vlm_probe`` store phase a caller enters for its
    own reasons (sft.py's pre-load run-label probe) stays uniform too.
    """
    revision = getattr(model_config, "model_revision", None)
    if not text_only_model and is_vlm_model(model_config.model_name_or_path, revision=revision):
        if init_from_scratch:
            raise ValueError("init_from_scratch is not supported for VLM models.")
        model, processor, tokenizer = load_vlm_model_and_processor(
            model_config,
            training_config,
            parallelism_config,
            weights_source=weights_source,
            attn_default=attn_default,
            reset_sinks=reset_sinks,
            train_sinks=train_sinks,
        )
        return model, processor, tokenizer, True

    model, tokenizer = load_model_consuming_init_kwargs(
        model_config,
        training_config,
        parallelism_config,
        weights_source=weights_source,
        attn_default=attn_default,
        trust_remote_code=model_config.trust_remote_code,
        revision=revision,
        reset_sinks=reset_sinks,
        train_sinks=train_sinks,
        init_from_scratch=init_from_scratch,
        text_only_model=text_only_model,
    )
    return model, tokenizer, tokenizer, False
