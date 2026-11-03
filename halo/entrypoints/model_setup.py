"""The run's policy model and processing class, the window, padding and attention defaults around that load, and a
preference run's policy with its reference."""

import logging
import os
from collections.abc import Sequence
from typing import Any, NamedTuple

import torch
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
from transformers.utils import is_flash_attn_2_available

from halo.checkpoint.format import PROCESSOR_FILES
from halo.configs.script_arguments import ModelArguments
from halo.data.collators.collator_rules import (
    evaluation_batch_size,
    reject_fa4_varlen_training,
    rows_reach_varlen_kernel,
)
from halo.distributed.tensor_parallel.hf_plan import is_input_embedding_tp_sharded
from halo.entrypoints.script import ScriptContext
from halo.model_loading.dtype import resolve_quantization_config, resolve_training_dtype
from halo.model_loading.liger.dispatch import finalize_liger_after_direct_load
from halo.model_loading.parallel.frozen_models import load_reference_model_for_preference
from halo.model_loading.parallel.loader import load_distributed_model, resolve_policy_attn_implementation
from halo.model_loading.parallel.model_source import fetch_checkpoint_config, probe_checkpoint
from halo.model_loading.parallel.peft_setup import configure_adapter_training
from halo.model_loading.tokenizer_setup import adopt_tokenizer_chat_template, configure_run_tokenizer
from halo.models.attention.gpt_oss_sinks import SinksPolicy
from halo.models.attention.layout import has_gated_delta_net_layers, has_mrope_rotary
from halo.models.attention.selection import (
    FA4_DEDICATED_HEAD_DIM,
    FLASH_ATTENTION_DETERMINISTIC_ENV,
    declares_attention_backends,
    declares_flash_attention,
    fa4_varlen_refusal,
    flash_attention_2_in_place_of_fa4,
    flash_attention_deterministic,
    is_fa4_demoted_by_determinism,
    preferred_flash_impl,
    uses_fa4_dedicated_head_dim,
)
from halo.models.context_window import is_bounded_length, resolve_length_to_context
from halo.models.modality import is_vlm_model
from halo.models.seq_cls_heads import multimodal_sequence_classification_model_types
from halo.runtime.consensus import rank_consensus
from halo.runtime.filesystem import hub_metadata_main_first
from halo.runtime.ranks import is_global_main_process

logger = logging.getLogger(__name__)


class PolicyLoad(NamedTuple):
    """What :func:`load_policy_model` loaded; ``is_vlm`` is the checkpoint's modality, not the run's."""

    model: PreTrainedModel
    processing_class: Any
    tokenizer: PreTrainedTokenizer
    is_vlm: bool


class RowAttentionPlan(NamedTuple):
    """The attention request the policy loads with when the YAML pins none, and whether training or evaluation rows
    reach FlashAttention's varlen call."""

    attn_default: str | None
    varlen_rows: bool


def default_padded_attn_implementation(
    model_config,
    *,
    sinks_reset: bool,
    context_parallel: bool = False,
    checkpoint_config=None,
    companion_configs: Sequence = (),
) -> str | None:
    """Attention for a script that forwards right-padded batches, when the YAML pins none.

    On Blackwell ``None``, the loader's pick: FA4 where it applies (it outruns SDPA on these batches), else the
    resolver's FA2 where FA4 does not train the attention (head_dim-256 Gemma 2/3, multi-head latent attention).
    ``sdpa`` on every other GPU, and on Blackwell where a loaded checkpoint rules that out: a remote-code class the
    resolver cannot read (Bailing's refuses every flash label), or an attention FA4's varlen kernels are not validated
    with (:func:`fa4_varlen_refusal`, Gemma 2's softcap), since a padded batch unpads to them. Where that attention is
    the policy's own and an FA2 build imports, ``flash_attention_2`` replaces ``sdpa``.

    ``checkpoint_config`` omitted is fetched (COLLECTIVE); ``companion_configs`` are further models on the same request
    (a distillation teacher). ``sinks_reset=False`` (on-policy gpt-oss, only sink-carrying impls pass) and
    ``context_parallel`` (Ulysses calls FlashAttention itself) drop the default."""
    if model_config.attn_implementation:
        return model_config.attn_implementation
    if context_parallel or not sinks_reset:
        return None
    if preferred_flash_impl() != "flash_attention_4":
        return "sdpa"
    if checkpoint_config is None:
        checkpoint_config = fetch_checkpoint_config(model_config, tag="padded_attn_default")
    loaded = (checkpoint_config, *companion_configs)
    if not all(declares_attention_backends(config) for config in loaded):
        return "sdpa"
    if all(fa4_varlen_refusal(config) is None for config in loaded):
        return None
    refused = fa4_varlen_refusal(checkpoint_config)
    if refused is None or not is_flash_attn_2_available():
        return "sdpa"
    return flash_attention_2_in_place_of_fa4(
        f"FlashAttention-4's varlen kernels are not validated with {refused}, and this run's padded batches unpad "
        f"to them"
    )


def default_training_rows_attn_implementation(
    model_config: ModelArguments,
    *,
    packing: bool,
    padding_free: bool,
    per_device_train_batch_size: int,
    dtype: torch.dtype,
    checkpoint_config=None,
    per_device_eval_batch_size: int = 0,
) -> str | None:
    """Attention request for a script that knows its row shape (SFT, self-distillation, embedding):
    ``flash_attention_2`` where the load would resolve FA4 on rows reaching its varlen call
    (:func:`rows_reach_varlen_kernel`) and a kernel limit applies, else the pin (``None``: the loader's pick).

    - FA4's varlen kernels are not validated with the checkpoint's attention (:func:`fa4_varlen_refusal`, Gemma 2's
      softcap), FA4 unpinned; raises where no FA2 build imports. A pinned FA4 is refused instead. One unpadded row per
      batch runs FA4's dense kernel, which takes the softcap.
    - Multi-document rows (``packing``, ``padding_free``) on a GatedDeltaNet decoder
      (:func:`has_gated_delta_net_layers`) whose heads FA4 trains on its dedicated head_dim-256 backward, FA4 unpinned:
      FA4's gain on packed rows is below the 5% a default change needs.
    - Multi-document rows under ``full_determinism``, whose backward that kernel is not validated with
      (:func:`is_fa4_demoted_by_determinism`), pinned or not: the loader would demote FA4 to SDPA, which runs a packed
      row as one dense ``[L, L]`` mask with K/V repeated per query head.

    Neither multi-document rule applies to ``padding_free`` on an M-RoPE decoder (:func:`has_mrope_rotary`), whose
    varlen path faults FA2, nor without an FA2 build. ``checkpoint_config`` omitted is fetched (COLLECTIVE), past the
    checks that need no config only."""
    pinned = model_config.attn_implementation or None
    several_documents = packing or padding_free
    deterministic_fa4_pin = pinned == "flash_attention_4" and several_documents and flash_attention_deterministic()
    if pinned is not None and not deterministic_fa4_pin:
        return pinned
    varlen_rows = rows_reach_varlen_kernel(
        packing=packing,
        padding_free=padding_free,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
    )
    gpu_picks_fa4 = preferred_flash_impl() == "flash_attention_4"
    if not varlen_rows or dtype == torch.float32 or (pinned is None and not gpu_picks_fa4):
        return pinned
    if checkpoint_config is None:
        checkpoint_config = fetch_checkpoint_config(model_config, tag="training_rows_attn_default")
    refused = fa4_varlen_refusal(checkpoint_config)
    if pinned is None and refused is not None:
        return flash_attention_2_in_place_of_fa4(
            f"FlashAttention-4's varlen kernels are not validated with {refused}, and this run forwards rows that "
            f"reach them (packed, padding-free or padded, in training or evaluation)"
        )
    if (
        not several_documents
        or not is_flash_attn_2_available()
        or (padding_free and has_mrope_rotary(checkpoint_config))
    ):
        return pinned
    if (
        pinned is None
        and uses_fa4_dedicated_head_dim(checkpoint_config)
        and has_gated_delta_net_layers(checkpoint_config)
    ):
        # A default, so info; the two kernel limits around it warn.
        if is_global_main_process():
            logger.info(
                f"Multi-document rows on a GatedDeltaNet decoder whose attention FlashAttention-4 trains on its "
                f"head_dim-{FA4_DEDICATED_HEAD_DIM} kernel, where it gains under 5% over FlashAttention-2: requesting "
                "attn_implementation='flash_attention_2'. Pin attn_implementation: flash_attention_4 to keep "
                "FlashAttention-4 outside full_determinism."
            )
        return "flash_attention_2"
    if declares_flash_attention(checkpoint_config) and is_fa4_demoted_by_determinism(checkpoint_config):
        return flash_attention_2_in_place_of_fa4(
            f"FlashAttention-4's head_dim-{FA4_DEDICATED_HEAD_DIM} kernel is not validated with a deterministic "
            "backward, which this run asks for (FLASH_ATTENTION_DETERMINISTIC=1, full_determinism), and its rows hold "
            "several documents, which SDPA would run as one dense mask"
        )
    return pinned


def reject_fa4_varlen_request(
    model_config: ModelArguments,
    attn_request: str | None,
    *,
    varlen_rows: bool,
    dtype: torch.dtype,
    sinks_policy: SinksPolicy,
    checkpoint_config=None,
    init_from_scratch: bool = False,
) -> None:
    """Refuse, before any weight loads, a ``flash_attention_4`` request the policy load keeps on rows reaching varlen
    kernels that refuse the checkpoint's attention. Only a pinned request can be FA4 there, since both defaults name
    another backend. ``checkpoint_config`` omitted is fetched (COLLECTIVE) where the request could be refused."""
    if attn_request != "flash_attention_4" or not varlen_rows:
        return
    if checkpoint_config is None:
        checkpoint_config = fetch_checkpoint_config(model_config, tag="fa4_varlen_request")
    if fa4_varlen_refusal(checkpoint_config) is None:
        return
    resolved = resolve_policy_attn_implementation(
        checkpoint_config, attn_request, dtype, sinks_policy=sinks_policy, init_from_scratch=init_from_scratch
    )
    reject_fa4_varlen_training(checkpoint_config, varlen_rows=varlen_rows, attn_impl=resolved)


def plan_row_attention(
    model_config: ModelArguments,
    training_args,
    *,
    sinks_policy: SinksPolicy,
    packing: bool = False,
    padding_free: bool = False,
    checkpoint_config=None,
    init_from_scratch: bool = False,
) -> RowAttentionPlan:
    """For a script shaping its own rows: the request :func:`default_training_rows_attn_implementation` makes and
    whether rows reach the varlen call, evaluation batches included, refusing a pinned FA4 there before any weight
    loads. COLLECTIVE where ``checkpoint_config`` is omitted and a check needs it. Raises on a ``full_determinism``
    config whose deterministic flash switch is off; :func:`~halo.entrypoints.script.parse_script_args` sets it."""
    if training_args.full_determinism and not flash_attention_deterministic():
        raise RuntimeError(
            f"full_determinism is set but {FLASH_ATTENTION_DETERMINISTIC_ENV} is not, so this row plan would pick the "
            "attention of a non-deterministic run for flash kernels the trainer then runs deterministic. "
            "parse_script_args sets it as it parses the config; set it before planning a config built another way."
        )
    rows = {
        "packing": packing,
        "padding_free": padding_free,
        "per_device_train_batch_size": training_args.per_device_train_batch_size,
        "per_device_eval_batch_size": evaluation_batch_size(training_args),
    }
    dtype = resolve_training_dtype(training_args)
    plan = RowAttentionPlan(
        attn_default=default_training_rows_attn_implementation(
            model_config, **rows, dtype=dtype, checkpoint_config=checkpoint_config
        ),
        varlen_rows=rows_reach_varlen_kernel(**rows),
    )
    reject_fa4_varlen_request(
        model_config,
        plan.attn_default,
        varlen_rows=plan.varlen_rows,
        dtype=dtype,
        sinks_policy=sinks_policy,
        checkpoint_config=checkpoint_config,
        init_from_scratch=init_from_scratch,
    )
    return plan


def reject_multimodal_model_without_seq_cls_head(model_config: ModelArguments) -> None:
    """Refuse a multimodal checkpoint with no sequence-classification head before the load:
    ``AutoModelForSequenceClassification`` names neither the modality nor an alternative, and only after the
    distributed init on every rank."""
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
    """The run's policy and tokenizer through ``load_distributed_model``, for a script that pins its own head or takes
    no VLM, and for both branches of :func:`load_policy_model`.

    ``model_class`` is the ``Auto*`` class of a non-default head; ``model_config_overrides`` merge over
    ``model_init_kwargs``, which are cleared so TRL does not re-apply them. ``attn_default`` keeps the YAML's pin
    unless a kernel limit moves it. Liger flags are captured before the load and replayed on the built model. A Path-B
    resume reads weights from the checkpoint at its own precision and keeps the run's base as ``_name_or_path``, the
    ``base_model`` TRL's model card names; where the weights were read stays on ``LOADED_WEIGHTS_FROM_ATTR``."""
    overrides = {**(getattr(training_config, "model_init_kwargs", None) or {}), **(model_config_overrides or {})}
    # Non-TRL configs (TeacherDistillationConfig, the plain HF classification/embedding ones) carry none.
    if hasattr(training_config, "model_init_kwargs"):
        training_config.model_init_kwargs = None
    use_liger_kernel = training_config.use_liger_kernel
    model, tokenizer = load_distributed_model(
        model_name_or_path=context.model_source,
        parallelism_config=context.parallelism_config,
        dtype=resolve_training_dtype(training_config),
        trust_remote_code=model_config.trust_remote_code,
        attn_implementation=attn_default or model_config.attn_implementation,
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


def _ships_processor_config(model_path: str, revision: str | None, *, local_files_only: bool) -> bool:
    """Whether the checkpoint carries a processor config, on a confirmed answer only. A local directory answers by its
    files, the Hub by a 404; offline, or under ``local_files_only``, the cache stands in. An unreachable Hub raises,
    since reading as absent would drop the processor from the run's exports."""
    if os.path.isdir(model_path):
        return any(os.path.isfile(os.path.join(model_path, name)) for name in PROCESSOR_FILES)
    for name in PROCESSOR_FILES:
        try:
            hf_hub_download(model_path, name, revision=revision, local_files_only=local_files_only)
            return True
        except RemoteEntryNotFoundError:
            continue
        except LocalEntryNotFoundError:
            if not (local_files_only or is_offline_mode()):
                raise
    return False


def load_vlm_processor(model_config: ModelArguments, *, required: bool = True):
    """The ``AutoProcessor`` at the model revision, main-rank-first: Hub ``main`` may carry other image geometry than
    the trained commit, and passing it as ``processing_class`` exports the ``processor_config.json`` engines build
    from. ``required=False`` returns ``None`` for a checkpoint shipping none (Step-3.7 Flash), agreed across ranks."""
    model_path = model_config.model_name_or_path
    revision = model_config.model_revision
    if not required:
        shipped = hub_metadata_main_first(
            "processor_probe",
            lambda local_files_only: _ships_processor_config(model_path, revision, local_files_only=local_files_only),
        )
        # A shipped config on any rank wins: ranks split between processor and tokenizer would enter
        # different hub phases and save different processing classes.
        if not rank_consensus(shipped).any_ok:
            return None
    return hub_metadata_main_first(
        "vlm_processor",
        lambda local_files_only: AutoProcessor.from_pretrained(
            model_path,
            trust_remote_code=model_config.trust_remote_code,
            revision=revision,
            local_files_only=local_files_only,
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
    """A multimodal policy with ``(model, processing_class, tokenizer)``, both from ``model_name_or_path`` even on a
    resume. The processor is required when ``vlm_run``. The modality is not re-probed: the ``vlm_probe`` store phase
    entered on a branch some ranks skip leaves the rest waiting on a key nobody writes."""
    processor = load_vlm_processor(model_config, required=vlm_run)
    tokenizer = (
        processor.tokenizer
        if hasattr(processor, "tokenizer")
        else hub_metadata_main_first(
            "vlm_tokenizer",
            lambda local_files_only: AutoTokenizer.from_pretrained(
                model_config.model_name_or_path,
                trust_remote_code=model_config.trust_remote_code,
                revision=model_config.model_revision,
                local_files_only=local_files_only,
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
    """The policy in the class its checkpoint declares, with its processor when ``vlm_run``, else its tokenizer.
    ``args.text_only_model`` loads a multimodal checkpoint through its CausalLM sibling; a rank-uniform YAML key, so
    every rank takes the same branch."""
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
    """Pin ``window`` (``None``: unbounded) and apply the token setup; returns the tokenizer resolved through
    ``args.tokenizer_backend``, which callers use in place of the one passed."""
    return configure_run_tokenizer(args, model, tokenizer, window, embeddings_sharded=is_input_embedding_tp_sharded)


def apply_max_length(
    training_config, args, model: PreTrainedModel, tokenizer: PreTrainedTokenizer
) -> PreTrainedTokenizer:
    """Fix the run's sequence cap and pin it on the tokenizer. A null or non-positive ``max_length`` resolves to the
    context window and is written back, so collators, length filters and packers read one number."""
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
    """Pin the GRPO family's window; returns ``(tokenizer, window)``. Pinned only when *both* halves are bounded: HF
    resolves ``truncation=True, max_length=None`` against ``model_max_length``, so a partial sum would cap the
    unbounded half. ``completion_budget_required`` marks on-policy callers, whose completion half is always bounded."""
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
    """The ``processing_class`` carrying the resolved tokenizer and its chat template: the window pins may hand back
    another object (the ``tokenizer_backend`` proxy), and a processor still holds the raw inner tokenizer and its own
    template. Read off the loaded object, since a multimodal checkpoint shipping no processor loads a tokenizer."""
    if isinstance(processing_class, PreTrainedTokenizerBase):
        return tokenizer
    processing_class.tokenizer = tokenizer
    adopt_tokenizer_chat_template(processing_class)
    return processing_class


def apply_text_path_padding_side(tokenizer: PreTrainedTokenizer, vlm_run: bool) -> None:
    """Pad a text-path run on the right: Gemma 4 and GLM-4.7-Flash keep left padding, which the packing collator
    rejects and the padding collators read as an attended leading document. A VLM run keeps the processor's side, as
    its collators pad through it."""
    if not vlm_run:
        tokenizer.padding_side = "right"


def log_model_info(model: PreTrainedModel, tokenizer: PreTrainedTokenizer) -> None:
    if is_global_main_process():
        logger.info(f"Tokenizer: {tokenizer}")
        logger.info(f"Model config: {model.config}")


class PreferenceModels(NamedTuple):
    """A preference run's policy, its frozen reference (``None`` under PEFT, whose adapter-free base is the
    reference, or a precomputed one), the adapter config, and the windowed tokenizer and processing class."""

    model: PreTrainedModel
    ref_model: PreTrainedModel | None
    peft_config: Any
    tokenizer: PreTrainedTokenizer
    processing_class: Any


def load_preference_models(
    context: ScriptContext, config, model_config: ModelArguments, args, *, vlm_run: bool, method: str
) -> PreferenceModels:
    """Load a DPO/KTO run's policy and reference under one attention binding, set up its adapter, pin the window and
    install the tokenizer. A logratio whose halves came from different attention kernels is biased."""
    attn_default = default_padded_attn_implementation(model_config, sinks_reset=args.reset_sinks)
    policy = load_policy_model(context, config, model_config, args, vlm_run=vlm_run, attn_default=attn_default)
    peft_config = configure_adapter_training(args, policy.model, model_config, "CAUSAL_LM")
    ref_model = load_reference_model_for_preference(
        args,
        model_config,
        config,
        context.parallelism_config,
        policy.tokenizer,
        is_vlm=policy.is_vlm,
        method=method,
        reset_sinks=args.reset_sinks,
        attn_default=attn_default,
    )
    tokenizer = apply_max_length(config, args, policy.model, policy.tokenizer)
    processing_class = install_resolved_tokenizer(policy.processing_class, tokenizer)
    log_model_info(policy.model, tokenizer)
    return PreferenceModels(policy.model, ref_model, peft_config, tokenizer, processing_class)
