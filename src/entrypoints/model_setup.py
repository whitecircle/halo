"""The run's policy model and processing class, for text and multimodal checkpoints, and the window,
padding and attention defaults every training script applies around that load."""

import os
from typing import Any, NamedTuple

from accelerate.logging import get_logger
from huggingface_hub import hf_hub_download, is_offline_mode
from huggingface_hub.errors import LocalEntryNotFoundError, RemoteEntryNotFoundError
from transformers import (
    AutoModelForSequenceClassification,
    AutoProcessor,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
    PreTrainedTokenizerBase,
)
from transformers.utils import IMAGE_PROCESSOR_NAME, PROCESSOR_NAME

from src.configs.script_arguments import ModelArguments
from src.distributed.tensor_parallel.hf_plan import is_input_embedding_tp_sharded
from src.entrypoints.script import ScriptContext
from src.model_loading.dtype import resolve_quantization_config, resolve_training_dtype
from src.model_loading.liger.dispatch import finalize_liger_after_direct_load
from src.model_loading.parallel.loader import load_distributed_model
from src.model_loading.parallel.model_source import fetch_checkpoint_config, probe_checkpoint
from src.model_loading.tokenizer_setup import adopt_tokenizer_chat_template, configure_run_tokenizer
from src.models.context_window import is_bounded_length, resolve_length_to_context
from src.models.modality import is_vlm_model
from src.models.seq_cls_heads import multimodal_sequence_classification_model_types
from src.runtime.consensus import rank_consensus
from src.runtime.filesystem import hub_metadata_main_first
from src.runtime.ranks import is_global_main_process

logger = get_logger(__name__, log_level="INFO")

# The files ``AutoProcessor`` reads a checkpoint's image processor from; a checkpoint shipping neither
# has no native processor to build.
_PROCESSOR_CONFIG_FILES = (PROCESSOR_NAME, IMAGE_PROCESSOR_NAME)


class PolicyLoad(NamedTuple):
    """What :func:`load_policy_model` loaded; ``is_vlm`` is the checkpoint's modality, not the run's."""

    model: PreTrainedModel
    processing_class: Any
    tokenizer: PreTrainedTokenizer
    is_vlm: bool


def default_padded_attn_implementation(
    model_config, *, sinks_reset: bool, context_parallel: bool = False
) -> str | None:
    """Attention implementation for a script that forwards right-padded (non-varlen) batches.

    SDPA when the YAML pins none: the auto-detected FA4 runs padded shapes through its slow varlen
    path. ``sinks_reset=False`` (on-policy gpt-oss, pretrained sinks live) drops the default, since only
    a sink-carrying impl is accepted there; pass the run's own ``reset_sinks``. ``context_parallel``
    drops it too: Ulysses calls FlashAttention itself, so the loader's hardware-aware pick names it.
    """
    if model_config.attn_implementation:
        return model_config.attn_implementation
    if context_parallel:
        return None
    return "sdpa" if sinks_reset else None


def reject_multimodal_model_without_seq_cls_head(model_config: ModelArguments) -> None:
    """Refuse a multimodal checkpoint with no sequence-classification head, before the model load.

    ``AutoModelForSequenceClassification`` refuses such a family with "Unrecognized configuration
    class", naming neither the modality nor an alternative, and only after the distributed init on every
    rank; this names the families that work, from the registry the resolution reads.
    """
    config = fetch_checkpoint_config(model_config, tag="seq_cls_head_probe")
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


def load_model_and_tokenizer(
    context: ScriptContext,
    training_config,
    model_config: ModelArguments,
    args,
    *,
    attn_default: str | None = None,
    model_class=None,
    model_config_overrides: dict | None = None,
    init_from_scratch: bool = False,
) -> tuple[PreTrainedModel, PreTrainedTokenizer]:
    """The run's policy and its tokenizer through ``load_distributed_model``, for a script that pins its own
    head or takes no VLM, and for both branches of :func:`load_policy_model`.

    ``model_class`` is the ``Auto*`` class of a non-default head (``AutoModelForSequenceClassification``,
    ``AutoModel``); ``model_config_overrides`` (``num_labels``) merge over the config's ``model_init_kwargs``,
    which are cleared so TRL does not re-apply them. The Liger flags are captured before the load and replayed
    on the built model. A Path-B resume reads the weights from the checkpoint at its own precision and keeps
    the run's base as ``_name_or_path``, the ``base_model`` TRL's model card names; where the weights were read
    stays on the loader's ``LOADED_WEIGHTS_FROM_ATTR`` stamp.
    """
    overrides = {**(getattr(training_config, "model_init_kwargs", None) or {}), **(model_config_overrides or {})}
    # Non-TRL configs (DistillationConfig, the plain HF classification/embedding ones) carry none.
    if hasattr(training_config, "model_init_kwargs"):
        training_config.model_init_kwargs = None
    use_liger_kernel = training_config.use_liger_kernel
    model, tokenizer = load_distributed_model(
        model_name_or_path=context.model_source,
        parallelism_config=context.parallelism_config,
        dtype=resolve_training_dtype(training_config),
        trust_remote_code=model_config.trust_remote_code,
        attn_implementation=model_config.attn_implementation or attn_default,
        model_config_overrides=overrides,
        use_liger_kernel=use_liger_kernel,
        liger_kernel_config=training_config.liger_kernel_config,
        model_class=model_class,
        quantization_config=resolve_quantization_config(model_config, training_config),
        reset_sinks=args.reset_sinks,
        train_sinks=args.train_sinks,
        init_from_scratch=init_from_scratch,
        revision=model_config.model_revision,
        text_only_model=args.text_only_model,
        preserve_checkpoint_precision=context.policy_from_checkpoint,
    )
    finalize_liger_after_direct_load(training_config, use_liger_kernel, model)
    if context.policy_from_checkpoint:
        model.config._name_or_path = model_config.model_name_or_path
    return model, tokenizer


def _ships_processor_config(model_path: str, revision: str | None) -> bool:
    """Whether the checkpoint carries a processor config, on a confirmed answer only.

    A local directory answers by its files, the Hub by a 404 for this revision; offline, the cache
    stands for the checkpoint. A Hub it cannot reach raises rather than reading as absent, which
    would drop the processor from the run's exports.
    """
    if os.path.isdir(model_path):
        return any(os.path.isfile(os.path.join(model_path, name)) for name in _PROCESSOR_CONFIG_FILES)
    for name in _PROCESSOR_CONFIG_FILES:
        try:
            hf_hub_download(model_path, name, revision=revision)
            return True
        except RemoteEntryNotFoundError:
            continue
        except LocalEntryNotFoundError:
            if not is_offline_mode():
                raise
    return False


def load_vlm_processor(model_config: ModelArguments, *, required: bool = True):
    """The checkpoint's ``AutoProcessor`` at the model revision, loaded main-rank-first.

    Hub ``main`` may carry other image geometry than the trained commit, and passing the processor as
    the trainer's ``processing_class`` is what exports the ``processor_config.json`` serving engines
    build from. ``required=False`` (a run without image data) returns ``None`` for a checkpoint that
    ships no processor config (Step-3.7 Flash), a verdict agreed across ranks.
    """
    model_path = model_config.model_name_or_path
    revision = model_config.model_revision
    if not required:
        shipped = hub_metadata_main_first("processor_probe", lambda: _ships_processor_config(model_path, revision))
        # A shipped config on any rank wins: ranks split between processor and tokenizer would enter
        # different hub phases and save different processing classes.
        if not rank_consensus(shipped).any_ok:
            return None
    return hub_metadata_main_first(
        "vlm_processor",
        lambda: AutoProcessor.from_pretrained(
            model_path, trust_remote_code=model_config.trust_remote_code, revision=revision
        ),
    )


def _load_vlm_model_and_processor(
    context: ScriptContext,
    training_config,
    model_config: ModelArguments,
    args,
    *,
    vlm_run: bool,
    attn_default: str | None,
) -> tuple[PreTrainedModel, Any, PreTrainedTokenizer]:
    """A multimodal policy with ``(model, processing_class, tokenizer)``: the processing class is the
    checkpoint's processor, required when ``vlm_run``, else the tokenizer for a checkpoint shipping none.

    The processor and tokenizer load from ``model_name_or_path`` even when the weights come from a
    resume checkpoint. The modality is not re-probed here: the probe enters the ``vlm_probe`` store
    phase, and one entered on a branch some ranks skip leaves the rest waiting on a key nobody writes.
    """
    processor = load_vlm_processor(model_config, required=vlm_run)
    tokenizer = (
        processor.tokenizer
        if hasattr(processor, "tokenizer")
        else hub_metadata_main_first(
            "vlm_tokenizer",
            lambda: AutoTokenizer.from_pretrained(
                model_config.model_name_or_path,
                trust_remote_code=model_config.trust_remote_code,
                revision=model_config.model_revision,
            ),
        )
    )
    model, _ = load_model_and_tokenizer(context, training_config, model_config, args, attn_default=attn_default)
    if is_global_main_process():
        logger.info(f"Loaded VLM model: {type(model).__name__}")
    return model, processor if processor is not None else tokenizer, tokenizer


def load_policy_model(
    context: ScriptContext,
    training_config,
    model_config: ModelArguments,
    args,
    *,
    vlm_run: bool,
    attn_default: str | None = None,
    init_from_scratch: bool = False,
) -> PolicyLoad:
    """The run's policy in the class its checkpoint declares, with the processing class its data takes.

    A multimodal checkpoint loads with its processor when ``vlm_run`` (the run feeds images), a text one
    with its tokenizer; ``attn_default`` is the backend when the config pins none. ``args.text_only_model``
    loads a multimodal checkpoint through its text-only CausalLM sibling; it is a rank-uniform YAML key,
    so every rank takes the same branch.
    """
    if (
        not args.text_only_model
        and probe_checkpoint(
            model_config.model_name_or_path,
            revision=model_config.model_revision,
            trust_remote_code=model_config.trust_remote_code,
        )[1]
    ):
        if init_from_scratch:
            raise ValueError("init_from_scratch is not supported for VLM models.")
        model, processing_class, tokenizer = _load_vlm_model_and_processor(
            context, training_config, model_config, args, vlm_run=vlm_run, attn_default=attn_default
        )
        return PolicyLoad(model, processing_class, tokenizer, is_vlm=True)
    model, tokenizer = load_model_and_tokenizer(
        context, training_config, model_config, args, attn_default=attn_default, init_from_scratch=init_from_scratch
    )
    return PolicyLoad(model, tokenizer, tokenizer, is_vlm=False)


def apply_tokenizer_window(args, model: PreTrainedModel, tokenizer: PreTrainedTokenizer, window: int | None):
    """Pin ``window`` (``None``: unbounded) on the tokenizer and apply the run's token setup; returns the
    tokenizer resolved through ``args.tokenizer_backend``, which callers use in place of the one passed."""
    return configure_run_tokenizer(args, model, tokenizer, window, embeddings_sharded=is_input_embedding_tp_sharded)


def apply_max_length(
    training_config, args, model: PreTrainedModel, tokenizer: PreTrainedTokenizer
) -> PreTrainedTokenizer:
    """Fix the run's sequence cap on ``training_config`` and pin it on the tokenizer
    (:func:`apply_tokenizer_window`).

    A null or non-positive ``max_length`` means the model's own limit: it resolves to the context window
    and is written back, so collators, length filters and packers all read one number.
    """
    training_config.max_length = resolve_length_to_context(training_config.max_length, model, tokenizer)
    return apply_tokenizer_window(args, model, tokenizer, training_config.max_length)


def apply_prompt_completion_window(
    args,
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    *,
    max_prompt_length: int | None,
    max_completion_length: int | None,
    completion_budget_required: bool = False,
) -> tuple[PreTrainedTokenizer, int | None]:
    """Pin the GRPO family's window, the prompt plus the completion budget; returns ``(tokenizer, window)``.

    Pinned only when *both* halves are bounded (:func:`is_bounded_length`): HF resolves every
    ``truncation=True, max_length=None`` call against ``model_max_length``, so a partial sum would cap
    the unbounded half. ``completion_budget_required`` marks the on-policy callers, whose completion
    half is a generation budget with no unbounded setting.
    """
    if completion_budget_required and not is_bounded_length(max_completion_length):
        raise ValueError(
            f"max_completion_length={max_completion_length!r} is the online generation budget (TRL's "
            f"max_new_tokens / the vLLM SamplingParams cap), so it must be a positive int — there is "
            f"no unbounded setting for a policy that has to stop generating. Set it in the config."
        )
    window = None
    if is_bounded_length(max_prompt_length) and is_bounded_length(max_completion_length):
        window = max_prompt_length + max_completion_length
    return apply_tokenizer_window(args, model, tokenizer, window), window


def install_resolved_tokenizer(processing_class, tokenizer: PreTrainedTokenizer):
    """The trainer's ``processing_class`` carrying the resolved tokenizer and its chat template.

    The window pins may hand back another object than the one loaded (the ``tokenizer_backend``
    proxy), and a processor still holds the raw inner tokenizer and its own template. Read off the
    loaded object, not the checkpoint's modality: a multimodal checkpoint shipping no processor loads
    a tokenizer for a run without image data.
    """
    if isinstance(processing_class, PreTrainedTokenizerBase):
        return tokenizer
    processing_class.tokenizer = tokenizer
    adopt_tokenizer_chat_template(processing_class)
    return processing_class


def apply_text_path_padding_side(tokenizer: PreTrainedTokenizer, vlm_run: bool) -> None:
    """Pad a text-path run on the right, whatever side a multimodal checkpoint's processor keeps.

    Gemma 4 and GLM-4.7-Flash keep left padding, which the packing collator rejects and the padding
    collators read as an attended leading document. A VLM run keeps the processor's side, since its
    collators pad through the processor.
    """
    if not vlm_run:
        tokenizer.padding_side = "right"


def log_model_info(model: PreTrainedModel, tokenizer: PreTrainedTokenizer) -> None:
    """Log the tokenizer and the model config on the main process."""
    logger.info(f"Tokenizer: {tokenizer}")
    logger.info(f"Model config: {model.config}")
