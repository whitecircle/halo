"""Shared scaffold for the training entry scripts (``scripts/training/**``).

The backbone (``init_training_script`` → ``load_script_datasets`` → ``load_script_model`` →
``apply_distributed_trainer_config`` → ``run_trainer``) plus the helpers that must behave identically
at every call site: window pins, tokenizer/attention resolution, callback assembly, and the
``reject_*`` guards. Parsing tuples, collators, and trainer construction stay in the scripts.
"""

from collections.abc import Callable, Sequence
from dataclasses import MISSING, fields
from typing import Any, NamedTuple

import torch
from accelerate import PartialState
from accelerate.logging import get_logger
from transformers import PreTrainedModel, PreTrainedTokenizer

from src.callbacks.generate_examples import GenerateExamplesCallback
from src.callbacks.parameter_stats import ParameterStatsCallback
from src.callbacks.wiring import build_perf_callbacks, reorder_integration_callbacks_last
from src.data.pipeline.preferences import prepare_generative_dataset, prepare_preference_datasets
from src.data.pipeline.processing import log_dataset_examples, resolve_map_num_proc
from src.data.sources.loading import is_presharded_dataset_load, load_datasets, reject_image_columns
from src.distributed.expert_parallel.dispatcher import verify_rank_uniform_env
from src.distributed.filesystem import verify_output_filesystem_sharing
from src.distributed.loading.peft_setup import split_expert_lora_targets
from src.distributed.loading.vlm_setup import load_model_consuming_init_kwargs
from src.distributed.parallelism_config import ParallelismConfig
from src.distributed.runtime import get_local_rank, init_distributed, is_global_main_process
from src.distributed.tensor_parallel.state_dict import input_embeddings_tp_sharded
from src.models.loading.tokenizer_setup import (
    is_bounded_length,
    resolve_length_to_context,
    setup_model_and_tokenizer,
)
from src.training.environment import (
    prepare_distributed_resume,
    setup_training_environment,
)
from src.training.parallelism_args import parallelism_config_from_args

logger = get_logger(__name__, log_level="INFO")


class ScriptRuntime(NamedTuple):
    """Everything :func:`init_training_script` resolves for a training entry script."""

    parallelism_config: ParallelismConfig
    mode_suffix: str
    local_rank: int
    resume_checkpoint: str | None
    model_source: str


def init_training_script(
    args,
    training_config,
    model_config,
    dist_args,
    *,
    script_prefix: str,
    supports_cp: bool = True,
    supports_pp: bool = True,
    sync_tokens: Sequence[str] = (),
    split_expert_lora: bool = True,
    **parallelism_kwargs,
) -> ScriptRuntime:
    """Run the common pre-model phase of a training entry script.

    Token-field sync → ``init_distributed`` → ``PartialState`` → CUDA device pinning →
    :func:`parallelism_config_from_args` (+ expert-LoRA split) → ``setup_training_environment``
    (run name ``{script_prefix}-{mode_suffix}``) → checkpoint-resume resolution. Every entry script
    depends on that order.

    Args:
        args: parsed script-arguments dataclass.
        training_config: parsed TRL/HF training config.
        model_config: parsed TRL ``ModelConfig``.
        dist_args: parsed ``DistributedArguments``.
        script_prefix: run-name prefix (e.g. ``"sft"``); the parallelism mode suffix is appended.
        supports_cp: forwarded to :func:`parallelism_config_from_args` (``False`` rejects CP).
        supports_pp: forwarded to :func:`parallelism_config_from_args` (``False`` rejects PP at
            config time, before the model — or a teacher/reference/vLLM probe — loads).
        sync_tokens: token field names to mirror between ``args`` and ``training_config``
            (configs that re-declare ``eos_token``/``pad_token`` under the resolve-conflict parser).
        split_expert_lora: peel MoE expert targets out of ``model_config.lora_target_modules`` into
            native EP grouped-LoRA (must run before the model load; no-op without expert targets).
        **parallelism_kwargs: extra :func:`parallelism_config_from_args` knobs
            (SFT passes ``allow_low_precision`` / ``supports_init_from_scratch``).
    """
    for field_name in sync_tokens:
        sync_token_field(args, training_config, field_name)

    # Every launcher that declares a world (torchrun/accelerate export RANK, a bare srun is caught by
    # its SLURM world size) gets the toolkit's watchdog timeout and eager device bind.
    init_distributed()
    PartialState()
    local_rank = get_local_rank()
    if torch.cuda.is_available():  # CUDA-less interpreters (config-parse checks, CPU boxes) stay usable
        torch.cuda.set_device(local_rank)

    # Before the weight load (a check placed after it would be the first collective a straggler
    # misses, and the watchdog would then time out here) and after the device bind (the gather
    # runs over NCCL).
    verify_rank_uniform_env()

    # Peeled before the config is built: __post_init__ validates expert_lora (PP rejects it) and
    # create_ep_config caches its EPConfig, so a later assignment would not reach the layers.
    parallelism_config = parallelism_config_from_args(
        dist_args,
        training_config=training_config,
        supports_cp=supports_cp,
        supports_pp=supports_pp,
        expert_lora=split_expert_lora_targets(model_config) if split_expert_lora else None,
        **parallelism_kwargs,
    )

    # The parser routes a YAML key only to the dataclass declaring it: these two are declared on
    # DistributedArguments but read off the training config, so without the forward they are unsettable.
    for field_name in ("save_max_shard_size", "overwrite_output_dir"):
        setattr(training_config, field_name, getattr(dist_args, field_name))

    mode_suffix = parallelism_config.mode_string or "standard"
    setup_training_environment(args, training_config, f"{script_prefix}-{mode_suffix}")
    # output_dir is final and still unwritten: the last point at which a mis-declared
    # DIST_*_SHARED_FILESYSTEM surfaces as a startup error rather than as a resume that restarts
    # every non-zero node at global_step=0.
    verify_output_filesystem_sharing(training_config.output_dir)

    resume_checkpoint, model_source = prepare_distributed_resume(training_config, model_config, parallelism_config)

    return ScriptRuntime(parallelism_config, mode_suffix, local_rank, resume_checkpoint, model_source)


def sync_token_field(args, training_config, field_name: str) -> None:
    """Mirror a token field across ``args`` and ``training_config``; the config wins when both set."""
    value = getattr(training_config, field_name, None)
    if value is None:
        value = getattr(args, field_name, None)
    if value is not None:
        setattr(args, field_name, value)
        setattr(training_config, field_name, value)


def reject_images_under_text_only_model(args, datasets, *, text_only_model: bool) -> None:
    """Reject image data on a run that loaded a multimodal checkpoint through its text-only class.

    Under ``text_only_model`` the loaded config is the text sub-config, so ``is_vlm_run`` cannot
    route to a VLM data path: an image column would be pruned and the run would train on the rows'
    text alone. Called once the dataset is in hand, before the modality dispatch.
    """
    if not text_only_model:
        return
    images_field = getattr(args, "images_field", None)
    if images_field:
        raise ValueError(
            f"images_field={images_field!r} is set but text_only_model=True loads the text-only "
            f"CausalLM class, which has no vision path. Drop text_only_model to train the "
            f"multimodal wrapper, or drop images_field for a text-only run."
        )
    reject_image_columns(datasets, "text_only_model=True (text-only CausalLM load)")


def load_script_datasets(
    args,
    parallelism_config: ParallelismConfig,
    *,
    loader: Callable[..., Any] = load_datasets,
    **loader_kwargs,
) -> tuple[Any, bool]:
    """Load the script's dataset(s) with DP-shard awareness.

    Returns ``(loader_result, dataset_presharded)``. ``loader`` is any of the
    ``load_datasets``-shaped loaders (``load_datasets_auto`` returns
    ``(DatasetDict, is_preprocessed)`` — that tuple passes through unchanged); ``loader_kwargs``
    forward extras such as ``conversation_field`` or ``seed``.
    """
    data_parallel_rank = parallelism_config.get_data_parallel_rank()
    data_parallel_size = parallelism_config.data_parallel_size
    dataset_presharded = is_presharded_dataset_load(args.dataset, data_parallel_size)
    # Threaded for every script so a declared tools column is validated at load; a typo'd knob
    # otherwise renders the whole run without tools. The scripts that cannot render it (KTO,
    # environmental GRPO) reject the knob through reject_unsupported_args before their load.
    loader_kwargs.setdefault("tools_field", getattr(args, "tools_field", None))
    ds = loader(
        args.dataset,
        args.test_size,
        args.dataset_ratio,
        data_parallel_rank=data_parallel_rank,
        data_parallel_size=data_parallel_size,
        **loader_kwargs,
    )
    return ds, dataset_presharded


def log_script_dataset_examples(datasets: dict, tokenizer: PreTrainedTokenizer, args, training_config) -> None:
    """``log_dataset_examples`` with the three arguments every training entry script passes alike.

    The destination and the decoded-sample switch are properties of the run, not of the script.
    Absent splits pass through as ``None`` and are skipped, so callers need not assemble the dict
    conditionally.
    """
    log_dataset_examples(
        datasets,
        tokenizer=tokenizer,
        output_dir=training_config.output_dir,
        write_decoded_samples=args.log_decoded_samples,
    )


def distributed_trainer_kwargs(args, dist_args, parallelism_config: ParallelismConfig, *, dataset_presharded: bool):
    """The constructor kwargs every ``DistributedTrainerMixin`` trainer takes.

    A call site passing only three of the four gathers a save the config asked to shard, or
    re-shards a dataset that is already per-DP-rank.
    """
    return {
        "parallelism_config": parallelism_config,
        "moe_balancing": args.moe_balancing,
        "save_sharded_ep": dist_args.save_sharded_ep,
        "dataset_presharded": dataset_presharded,
    }


def disable_trl_dataset_prep(sft_config) -> None:
    """Turn TRL's own dataset preparation off on a config whose data the script already prepared.

    ``skip_prepare_dataset`` alone is not enough: TRL reads ``packing`` and ``padding_free`` again at
    collator-selection time, and either one left on replaces the collator the script built with
    TRL's own — over rows TRL never tokenized.
    """
    sft_config.dataset_kwargs = {"skip_prepare_dataset": True}
    sft_config.padding_free = False
    sft_config.packing = False


def prepare_script_preference_data(
    args,
    training_config,
    ds,
    tokenizer: PreTrainedTokenizer,
    *,
    is_vlm_data: bool,
    generation_max_prompt_length: int | None,
):
    """Tokenize a preference dataset, build the generation-eval split, and log the examples.

    The VLM arm passes the rows through unchanged: both trainers chat-template and tokenize vision
    preference rows themselves, so a prep here would double-render them. ``generation_max_prompt_length``
    is the one per-script difference (DPO takes its own knob, SMPO reuses ``max_prompt_length``).

    Returns ``(train_dataset, eval_dataset, generate_callback)``; the callback is ``None`` on the VLM
    arm and whenever generation-eval is off.
    """
    generate_dataset = None
    if is_vlm_data:
        train_dataset, eval_dataset = ds["train"], ds.get("test")
    else:
        num_proc = resolve_map_num_proc(training_config.dataset_num_proc)
        train_dataset, eval_dataset = prepare_preference_datasets(
            ds["train"],
            ds["test"],
            tokenizer,
            num_proc=num_proc,
            tools_field=args.tools_field,
        )
        # Only when the callback will consume it — the prep tokenizes the whole test split.
        if args.generate_eval_examples:
            generate_dataset = prepare_generative_dataset(
                ds["test"],
                tokenizer,
                generation_max_prompt_length,
                num_proc=num_proc,
                tools_field=args.tools_field,
            )

    log_script_dataset_examples(
        {"train": train_dataset, "test": eval_dataset, "generate": generate_dataset},
        tokenizer,
        args,
        training_config,
    )
    generate_callback = GenerateExamplesCallback.from_config(args, training_config, generate_dataset, tokenizer)
    return train_dataset, eval_dataset, generate_callback


def apply_max_length(
    training_config, args, model: PreTrainedModel, tokenizer: PreTrainedTokenizer
) -> PreTrainedTokenizer:
    """Fix the run's sequence cap on ``training_config`` and pin it on the tokenizer.

    ``max_length`` null / non-positive means "use the model's own limit": it resolves to the context
    window and is written back, so collators, length filters and packers all read one resolved number.
    Returns the tokenizer resolved through ``args.tokenizer_backend``, which callers must use in
    place of the one passed in. The GRPO family pins its two-knob budget through
    :func:`apply_prompt_completion_window` instead.
    """
    training_config.max_length = resolve_length_to_context(training_config.max_length, model, tokenizer)
    return setup_model_and_tokenizer(
        args, model, tokenizer, training_config.max_length, embeddings_sharded=input_embeddings_tp_sharded
    )


def install_resolved_tokenizer(processing_class, tokenizer: PreTrainedTokenizer, is_vlm: bool):
    """Return the trainer's ``processing_class`` carrying the resolved tokenizer.

    ``apply_max_length`` may hand back a different object than the one loaded (the
    ``tokenizer_backend`` proxy), and a VLM processor still holds the raw inner tokenizer.
    """
    if is_vlm:
        processing_class.tokenizer = tokenizer
        return processing_class
    return tokenizer


def enforce_text_path_padding_side(tokenizer: PreTrainedTokenizer, vlm_run: bool) -> None:
    """Force right padding on a text-path run whose tokenizer came off a VLM processor.

    A multimodal checkpoint loads through ``load_vlm_model_and_processor``, whose tokenizer keeps the
    checkpoint's side (left for Gemma 4 and GLM-4.7-Flash); feeding that to the text collators makes
    the packing collator reject it and the padding ones read leading pads as an attended document.
    A VLM run keeps the processor's side, since its collators pad through the processor.
    """
    if not vlm_run:
        tokenizer.padding_side = "right"


def apply_prompt_completion_window(
    args,
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    *,
    max_prompt_length: int | None,
    max_completion_length: int | None,
    completion_budget_required: bool = False,
) -> tuple[PreTrainedTokenizer, int | None]:
    """Pin the tokenizer's window for the GRPO family, whose budget is two knobs rather than one.

    The window is the sum of the prompt and completion budgets, pinned only when *both* halves are
    bounded (:func:`is_bounded_length`, so ``0`` means unset): HF resolves every ``truncation=True,
    max_length=None`` call against ``model_max_length``, so a partial sum would turn the unbounded
    half into an unintended cap. ``completion_budget_required`` marks the on-policy callers, where
    the completion half is a *generation* budget with no unbounded setting.

    Returns ``(tokenizer, window)``: the tokenizer resolved through ``args.tokenizer_backend``,
    which callers must use in place of the one passed in, and the window, ``None`` when unbounded.
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
    tokenizer = setup_model_and_tokenizer(
        args, model, tokenizer, window, embeddings_sharded=input_embeddings_tp_sharded
    )
    return tokenizer, window


def padded_workload_attn_implementation(model_config, *, sinks_reset: bool) -> str | None:
    """Attention implementation for padded (non-varlen) workloads: reward modeling, the GRPO family,
    and every other script that forwards right-padded batches.

    Defaults to SDPA when the YAML pins none, because the auto-detected FA4 runs padded shapes
    through its slow varlen path. ``sinks_reset=False`` (on-policy gpt-oss, pretrained sinks live)
    drops the default: only a sink-carrying implementation is accepted there, so requesting SDPA
    would reject the run. Pass the run's own ``reset_sinks`` rather than a hardcoded value.
    """
    if model_config.attn_implementation:
        return model_config.attn_implementation
    return "sdpa" if sinks_reset else None


def reject_unsupported_args(context: str, **unsupported) -> None:
    """Raise on parsed knobs this script cannot honor.

    Shared config dataclasses expose fields no single script implements; ignoring one would train
    something other than what the config states. Only truthy values are reported, since an unset
    field is not a request.
    """
    _reject_ignored_fields(context, sorted(name for name, value in unsupported.items() if value))


def reject_non_default_args(context: str, args, *field_names: str) -> None:
    """Raise on parsed knobs this script cannot honor, judged against their declared default.

    :func:`reject_unsupported_args` reads a request off truthiness, which cannot express a knob whose
    own default is truthy (``num_eval_examples=50`` would reject every run). Every name must be a
    defaulted field of ``args``: a typo would leave the guard unable to fire, so it raises here.
    """
    defaults: dict[str, Any] = {}
    for declared in fields(type(args)):
        if declared.default is not MISSING:
            defaults[declared.name] = declared.default
        elif declared.default_factory is not MISSING:
            defaults[declared.name] = declared.default_factory()
    undeclared = sorted(name for name in field_names if name not in defaults)
    if undeclared:
        raise ValueError(f"{undeclared} are not defaulted fields of {type(args).__name__} — the guard cannot fire")
    _reject_ignored_fields(context, sorted(name for name in field_names if getattr(args, name) != defaults[name]))


def _reject_ignored_fields(context: str, set_fields: list[str]) -> None:
    """Shared raise for both rejection forms, so they report the same way."""
    if set_fields:
        raise ValueError(
            f"{context} does not support these config fields (they would be silently ignored): "
            f"{set_fields}. Remove them from the YAML."
        )


def load_script_model(
    runtime: ScriptRuntime,
    training_config,
    model_config,
    dist_args,
    *,
    model_class=None,
    model_config_overrides: dict | None = None,
    attn_implementation: str | None = None,
) -> tuple[PreTrainedModel, PreTrainedTokenizer]:
    """Load the policy model + tokenizer for a training entry script.

    The text-path twin of :func:`~src.distributed.loading.vlm_setup.load_model_for_training`, for the
    scripts that pin their own head or take no VLM at all: the same
    :func:`~src.distributed.loading.vlm_setup.load_model_consuming_init_kwargs` seam under the run's
    ``ParallelismConfig`` and its sinks / text-only flags.

    Args:
        model_class: explicit ``Auto*`` class for scripts whose head is not the config's default
            (``AutoModelForSequenceClassification`` for reward/classification, ``AutoModel`` for
            embeddings); ``None`` auto-resolves from the model config.
        model_config_overrides: per-script config overrides (e.g. ``num_labels``), merged over the
            consumed ``model_init_kwargs``.
        attn_implementation: the request when ``model_config.attn_implementation`` is unset — pass
            :func:`padded_workload_attn_implementation` for scripts that forward right-padded batches.
    """
    return load_model_consuming_init_kwargs(
        model_config,
        training_config,
        runtime.parallelism_config,
        weights_source=runtime.model_source,
        attn_default=attn_implementation,
        trust_remote_code=model_config.trust_remote_code,
        revision=model_config.model_revision,
        model_config_overrides=model_config_overrides,
        model_class=model_class,
        reset_sinks=dist_args.reset_sinks,
        text_only_model=dist_args.text_only_model,
        train_sinks=dist_args.train_sinks,
    )


def build_training_callbacks(
    args,
    training_config,
    model,
    parallelism_config: ParallelismConfig,
    *,
    generate_callback=None,
    **perf_kwargs,
) -> list:
    """Assemble the standard per-script callback set: ``ParameterStatsCallback``, the optional
    generation-eval callback, then the perf callbacks (``build_perf_callbacks`` extras such as
    ``policy_gradient_loss`` pass through ``perf_kwargs``)."""
    callbacks: list = [ParameterStatsCallback]
    if generate_callback is not None:
        callbacks.append(generate_callback)
    callbacks.extend(build_perf_callbacks(args, training_config, model, parallelism_config, **perf_kwargs))
    return callbacks


def apply_distributed_trainer_config(training_config, parallelism_config: ParallelismConfig) -> None:
    """Trainer-config knobs every distributed trainer requires: disable HF's own FSDP wiring (the
    mixin handles sharding) and let DDP tolerate the EP layers' externally-synced parameters."""
    training_config.fsdp = ""
    if parallelism_config.is_ep_mode:
        training_config.ddp_find_unused_parameters = True


def run_trainer(
    trainer,
    runtime: ScriptRuntime,
    *,
    method_name: str,
    extra_start_log: Sequence[str] | None = None,
) -> None:
    """Run the common post-construction phase: integration-callback reordering, the canonical
    start log (mode + EP/CP/TP/ETP/DP sizes + ``extra_start_log`` lines), resume log, training,
    and EP cleanup."""
    # Integrations add themselves at the head of the list and would consume `logs` before the
    # toolkit's own callbacks run their on_log.
    reorder_integration_callbacks_last(trainer)

    parallelism_config = runtime.parallelism_config
    if is_global_main_process():
        logger.info(f"Starting {method_name} training (mode: {runtime.mode_suffix})...")
        if parallelism_config.is_ep_mode:
            logger.info(f"  EP size: {parallelism_config.ep_size}")
        if parallelism_config.is_cp_mode:
            logger.info(f"  CP size: {parallelism_config.cp_size}")
        if parallelism_config.is_tp_mode:
            logger.info(f"  TP size: {parallelism_config.tp_size}")
        if parallelism_config.is_expert_tp_mode:
            logger.info(f"  ETP size: {parallelism_config.expert_tp_size}")
        logger.info(f"  DP size: {parallelism_config.data_parallel_size}")
        for line in extra_start_log or ():
            logger.info(f"  {line}")

    if runtime.resume_checkpoint and is_global_main_process():
        logger.info(f"Resuming from checkpoint: {runtime.resume_checkpoint}")
    trainer.train(resume_from_checkpoint=runtime.resume_checkpoint)

    trainer.cleanup_ep()

    if is_global_main_process():
        logger.info(f"{method_name} training completed successfully!")
