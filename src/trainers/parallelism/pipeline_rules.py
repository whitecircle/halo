"""Pipeline-parallel trainer refusals: pure functions the PP trainer integration and the PP-enabled
trainers call at construction, before the model is split into this rank's stage, and on the trainer's
final collator after it."""

from __future__ import annotations

from src.data.collators.packing import DataCollatorWithFlattening, DataCollatorWithPacking
from src.data.vlm_routing import VLM_IMAGE_COLUMNS, dataset_image_evidence
from src.distributed.parallelism_config import ParallelismConfig
from src.distributed.pipeline_parallel.stage import reject_reentrant_checkpointing
from src.model_loading.tokenizer_setup import resolve_tokenizer
from src.optimizers.registry import NAMED_OPTIMIZER_BUILDERS, optim_name
from src.runtime.consensus import reject_across_ranks
from src.runtime.filesystem import is_output_shared_filesystem
from src.runtime.ranks import get_num_nodes


def reject_unsupported_pp_training_args(training_args, *, save_sharded_ep: bool, evaluates: bool) -> None:
    """Reject the training arguments a pipeline cannot honor; ``evaluates`` says whether any evaluation runs."""
    if not getattr(training_args, "max_length", None):
        raise ValueError(
            "Pipeline parallelism requires a fixed sequence length (the P2P buffer shapes "
            "freeze on the first step). Set max_length in the training config."
        )
    if save_sharded_ep:
        raise ValueError(
            "save_sharded_ep is not supported under pipeline parallelism: the per-rank shard format "
            "keys tensors with no pipeline-stage layer offset, so stages would collide. The PP save "
            "already writes one complete-tensor shard per stage under the unsplit model's names."
        )
    if getattr(training_args, "push_to_hub", False) and not is_output_shared_filesystem() and get_num_nodes() > 1:
        raise ValueError(
            "push_to_hub is not supported under pipeline parallelism with a per-node output "
            "filesystem (DIST_OUTPUT_SHARED_FILESYSTEM=0) across nodes: global rank 0 uploads, and its "
            "node lacks the other nodes' stage shards. Write checkpoints to a shared output "
            "filesystem, or gather every node's checkpoint directory into one with "
            "scripts/after_training/merge_pp_stages.py and upload that."
        )
    if training_args.gradient_checkpointing:
        # The stage's gradient_checkpointing_enable refuses it too; here it fails before the split.
        reject_reentrant_checkpointing(training_args.gradient_checkpointing_kwargs)
    if evaluates and training_args.per_device_eval_batch_size != training_args.per_device_train_batch_size:
        raise ValueError(
            f"per_device_eval_batch_size ({training_args.per_device_eval_batch_size}) must "
            f"equal per_device_train_batch_size ({training_args.per_device_train_batch_size}) "
            f"under pipeline parallelism: the P2P buffer shapes are frozen to the training batch."
        )
    if training_args.torch_compile:
        raise ValueError(
            "torch_compile is not supported under pipeline parallelism: the schedule captures the "
            "stage module at setup, so a compiled wrapper installed afterwards never runs. Disable "
            "torch_compile."
        )
    # TRL's flag, which SFTConfig and RewardConfig carry too; plain TrainingArguments do not.
    if getattr(training_args, "activation_offloading", False):
        raise ValueError(
            "activation_offloading is not supported under pipeline parallelism: TRL applies it "
            "by wrapping training_step, which the PP mixin's schedule-driven step bypasses — it "
            "would silently never engage."
        )


def reject_pp_peft(peft_config, explicit_param_trainer: str | None = None) -> None:
    """Reject PEFT/LoRA under PP, whose adapter checkpoint path is not stage-aware.

    ``explicit_param_trainer`` names a trainer whose ``peft_config`` is an explicit ctor parameter the
    mixin's kwargs-based rejection cannot see.
    """
    if peft_config is None:
        return
    message = (
        "PEFT/LoRA is not supported under pipeline parallelism: the adapter checkpoint path is "
        "not stage-aware — save_peft_adapters writes adapter_model.safetensors from the save rank "
        "under the stage's own re-based layer indices (stage 1's layers 0..k claim stage 0's "
        "names, the other stages' adapters are never written), the PP resume path never reaches "
        "the adapter restore, and layers_to_transform / target_modules='all-linear' would resolve "
        "against a stage's re-based layers and its bare task head. Train the adapter without "
        "pipeline parallelism, or fully fine-tune under PP."
    )
    if explicit_param_trainer:
        message += (
            f" ({explicit_param_trainer} takes peft_config as an explicit parameter, so the "
            f"mixin's kwargs-based PP rejection cannot see it.)"
        )
    raise ValueError(message)


def require_pp_stage_model(kwargs: dict):
    """The ``model`` keyword the split turns into this rank's stage, refusing a missing or config-less one."""
    model = kwargs.get("model")
    if model is None or not hasattr(model, "config"):
        raise ValueError(
            "Pipeline parallelism requires the model to be passed as the `model` keyword (a "
            "loaded PreTrainedModel), so the trainer can split it into this rank's stage."
        )
    return model


def reject_pp_vlm(train_dataset, eval_dataset, collator) -> None:
    """COLLECTIVE: refuse a multimodal wrapper's run that feeds it images, naming the evidence.

    The split keeps the text backbone and the task head, so the vision tower and projector are in no
    stage. Joined across ranks: the dataset probe is rank-local (a pre-sharded dataset can carry images
    on one rank only), and a lone raise would leave the peers at the split's collectives.
    """
    consumed = set(getattr(collator, "required_dataset_columns", ()) or ()) & set(VLM_IMAGE_COLUMNS)
    evidence = (
        dataset_image_evidence(train_dataset)
        or dataset_image_evidence(eval_dataset)
        or (f"the data collator consuming {sorted(consumed)}" if consumed else None)
    )
    reason = (
        (
            f"Vision-language training is not supported under pipeline parallelism: the stage split "
            f"keeps only the text backbone and the task head, and {evidence} means this run feeds "
            f"images the model would never see. Train the VLM without pipeline parallelism, or drop "
            f"the image data for a text-only run of its text tower."
        )
        if evidence
        else None
    )
    reject_across_ranks(reason, "Pipeline-parallel multimodal gate", exc_type=ValueError)


def reject_positional_model_or_args(kwargs: dict) -> None:
    """Reject positional ``model``/``args`` under PP: the pipeline seam rewrites both by key."""
    if "model" not in kwargs or "args" not in kwargs:
        raise ValueError(
            "Pipeline parallelism requires `model` and `args` passed as keywords: the mixin's "
            "pipeline seam splits kwargs['model'] into this rank's stage and reads "
            "kwargs['args'] for the microbatch conversion."
        )


def reject_pp_ref_model(ref_model, column_hint: str) -> None:
    """Reject a live reference model under PP. ``column_hint`` names the trainer's precomputed
    reference column(s) to ship instead."""
    if ref_model is None:
        return
    raise ValueError(
        f"An explicit ref_model is not supported under pipeline parallelism: TRL keeps the "
        f"full reference resident beside this rank's stage and forwards it inside "
        f"_compute_loss and the precompute sweep — full-model, PP-unaware paths (and a full "
        f"model next to one stage defeats PP's memory point). Ship {column_hint} with "
        f"precompute_ref_log_probs=True instead."
    )


def reject_missing_precomputed_columns(dataset, columns: tuple[str, ...], role: str = "train") -> None:
    """Reject a ``role`` dataset that lacks the precomputed reference column(s) under PP."""
    present = getattr(dataset, "column_names", None) or []
    if all(column in present for column in columns):
        return
    subject = f"the '{columns[0]}' column" if len(columns) == 1 else f"the {columns} columns"
    noun = "column" if len(columns) == 1 else "columns"
    raise ValueError(
        f"precompute_ref_log_probs=True under pipeline parallelism requires {subject} to ALREADY "
        f"be in the {role} dataset: TRL's in-init precompute sweep forwards self.model over the "
        f"whole dataset, and under PP self.model is this rank's bare stage (its forward takes "
        f"hidden_states, before the pipeline runner even exists). Precompute the {noun} in a "
        f"non-PP run first."
    )


def reject_missing_precomputed_reference(
    trainer: str, training_args, train_dataset, eval_dataset, columns: tuple[str, ...]
) -> None:
    """Reject a run off PP's precompute-only reference contract: the flag set and ``columns`` present in
    the train and every eval split, since TRL's in-init sweep runs over those too. ``trainer`` names the
    wrapper in the message."""
    if not training_args.precompute_ref_log_probs:
        subject = f"the {columns[0]} column" if len(columns) == 1 else f"the {'/'.join(columns)} columns"
        raise ValueError(
            f"{trainer} under pipeline parallelism is precompute-only: without "
            f"precompute_ref_log_probs=True TRL instantiates a live reference model from the "
            f"config path and forwards it every step — a full-model path no pipeline rank can "
            f"run. Set precompute_ref_log_probs=True and put {subject} in the dataset."
        )
    reject_missing_precomputed_columns(train_dataset, columns)
    if eval_dataset is not None:
        for dataset in eval_dataset.values() if isinstance(eval_dataset, dict) else (eval_dataset,):
            if dataset is not None:
                reject_missing_precomputed_columns(dataset, columns, role="eval")


def reject_pp_compute_metrics(compute_metrics, mechanism: str) -> None:
    """Reject ``compute_metrics`` under PP for a trainer with no ``PipelineLossAdapter.predictions_fn``
    reduction to reproduce (its non-PP ``prediction_step`` hands metrics the raw logits plane, or
    nothing); ``mechanism`` is that trainer's reason."""
    if compute_metrics is None:
        return
    raise ValueError(f"compute_metrics is not supported under pipeline parallelism: {mechanism}")


def reject_pp_named_optimizer(training_args, parallelism_config) -> None:
    """Reject a toolkit-named optimizer (``NAMED_OPTIMIZER_BUILDERS``) under PP, where none is validated."""
    optim = optim_name(getattr(training_args, "optim", None))
    if parallelism_config.is_pp_mode and optim in NAMED_OPTIMIZER_BUILDERS:
        raise ValueError(
            f"optim={optim!r} is not supported under pipeline parallelism: the stage-local optimizer "
            f"build and the per-stage optimizer shards a resume restores are validated only with the "
            f"AdamW family. Use an AdamW optimizer under pipeline parallelism, or run {optim} without it."
        )


def reject_uncovered_pp_ep_layers(config: ParallelismConfig, ep_layers) -> None:
    """COLLECTIVE: refuse a PP+EP run where this rank's stage carries no EP layer, joined across ranks
    since the verdict is rank-local (a hybrid MoE with leading dense layers can give one stage none)."""
    reason = None
    if config.has_ep_group and not ep_layers:
        reason = (
            f"PP stage {config.pp_rank}/{config.pp_size} carries no EP-patched modules: the model was "
            "loaded without EP patching (load_distributed_model must run before the trainer), or the "
            "partition gave this stage only dense layers, whose collectives would deadlock against its MoE "
            "peers. Pick a pipeline_parallel_size or pipeline_split that puts a MoE layer on every stage."
        )
    reject_across_ranks(reason, "PP stage EP coverage")


def reject_pp_collator(collator, *, per_device_train_batch_size: int, n_microbatches: int) -> None:
    """Reject a collator whose batches cannot take the pipeline's fixed shape, frozen after the first step."""
    if collator is None:
        raise ValueError("Pipeline parallelism requires a data collator; the trainer has none.")
    if isinstance(collator, DataCollatorWithFlattening):
        # Its cu_seq_lens side outputs cannot be split per microbatch either, and its one row
        # cannot be microbatched at all.
        raise ValueError(
            f"{type(collator).__name__} (padding-free) is incompatible with pipeline parallelism: "
            f"its flattened width is the summed document length, which varies every step, and "
            f"the pipeline's P2P buffers are frozen after the first. Use packing instead "
            f"(`packing: true`), whose packed rows are a fixed shape."
        )
    if isinstance(collator, DataCollatorWithPacking) and per_device_train_batch_size != n_microbatches:
        raise ValueError(
            f"Packing under pipeline parallelism needs one packed row per microbatch, but "
            f"per_device_train_batch_size={per_device_train_batch_size} and the "
            f"microbatch count is {n_microbatches}: a microbatch holding >1 packed row "
            f"leaks attention across documents under FlashAttention. Set "
            f"per_device_train_batch_size equal to the microbatch count."
        )


def require_pad_token_id(processing_class, why: str) -> int:
    """The tokenizer's pad id; ``why`` opens the refusal for a processing class without one."""
    pad_id = getattr(resolve_tokenizer(processing_class), "pad_token_id", None)
    if pad_id is None:
        raise ValueError(
            f"{why} and needs the tokenizer's pad_token_id; the processing_class passed to the trainer has none."
        )
    return pad_id


def require_pooling_pad_id(config, tokenizer_pad_id: int | None) -> int:
    """The pad id sequence-classification pooling keys on, or a raise when it is unset or disagrees with the
    tokenizer's, which pads every PP batch: no filler would count as padding, and every row would pool a pad."""
    pad_id = config.get_text_config().pad_token_id
    if pad_id is None:
        raise ValueError(
            "Sequence-classification pooling under pipeline parallelism picks each row's rightmost "
            "token != config.pad_token_id, but the model config has pad_token_id=None (transformers "
            "itself cannot pool batches > 1 in this state). The setup seam records the tokenizer's "
            "pad id for you, so reaching here means the tokenizer settled on none either: give it "
            "one with --pad_token (a token already in the vocabulary) or --added_special_tokens (to "
            "add the token first)."
        )
    if tokenizer_pad_id is not None and tokenizer_pad_id != pad_id:
        raise ValueError(
            f"config.pad_token_id ({pad_id}) disagrees with the tokenizer's pad_token_id "
            f"({tokenizer_pad_id}). Pipeline parallelism pads every batch to max_length with the "
            f"tokenizer's id while pooling keys on the config's, so every row would pool at the last "
            f"filler token instead of its last real one — a finite loss that learns nothing. Set "
            f"model.config.pad_token_id = tokenizer.pad_token_id (or pass --pad_token)."
        )
    return pad_id
