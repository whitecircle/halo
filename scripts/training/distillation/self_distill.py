#!/usr/bin/env python
"""SDPG self-distillation (text or VLM) with distributed parallelism.

Privileged-context self-distillation on top of SFT (:class:`DistributedSelfDistillationTrainer`):
one model is both the student (prompt only) and a privileged teacher (prompt + a hint revealing
the gold answer), and the teacher's full-vocabulary distribution supervises the student on the
shared response tokens (``L = L_sft + beta(k)·L_OPD + alpha·L_ref``). One script serves both text
and vision-language models: the run's modality (``resolve_vlm_run``, read off the data and the
checkpoint) selects the text or VLM self-distillation collator. This is the offline SDPG
approximation (scores a fixed dataset; the faithful on-policy SDPG runs via
``online_grpo/rlvr.py --use_sdpg``). For off-policy teacher→student distillation use
``teacher_distill.py``.

Dataset: raw SFT conversations (``conversation_field``) plus a privileged answer field (VLM adds an
``images`` column). The collator tokenizes the student and teacher (hinted) branches at collation
time, so a pre-tokenized dataset is not usable here.

Usage:
    torchrun --nproc_per_node=8 scripts/training/distillation/self_distill.py \\
        examples/distillation/qwen3_5/self-distill-qwen3.5-9b.yaml

That student is dense, so it takes no --expert_parallel_size; the MoE configs pin it themselves.
"""

from functools import partial

from accelerate.logging import get_logger
from transformers import AutoTokenizer, PreTrainedModel
from trl import ModelConfig, SFTConfig

from src.args.distributed_args import DistributedArguments
from src.args.mixins import format_field_names
from src.args.self_distill_args import SelfDistillationArguments
from src.data.collators.self_distill import (
    SelfDistillTextCollator,
    audit_self_distill_row,
    confidence_normalizer,
    privileged_hint,
    teacher_history,
)
from src.data.collators.vlm import SelfDistillVLMDataCollator
from src.data.pipeline.processing import coordinated_map, require_render_column, resolve_map_num_proc
from src.data.pipeline.row_processors import text_render_kwargs
from src.data.pipeline.vlm_dataset import prepare_vlm_dataset
from src.distributed.filesystem import hub_metadata_main_first
from src.distributed.loading.frozen_models import load_frozen_reference_model
from src.distributed.loading.peft_setup import setup_peft_model
from src.distributed.loading.vlm_setup import load_model_for_training
from src.distributed.runtime import barrier, is_global_main_process
from src.models.loading.model_preparation import log_model_info
from src.trainers.distillation.losses import require_same_token_ids
from src.trainers.distillation.self_distillation import DistributedSelfDistillationTrainer
from src.training.environment import run_training
from src.training.parser import H4ArgumentParser
from src.training.script_runner import (
    apply_distributed_trainer_config,
    apply_max_length,
    build_training_callbacks,
    disable_trl_dataset_prep,
    distributed_trainer_kwargs,
    enforce_text_path_padding_side,
    init_training_script,
    install_resolved_tokenizer,
    load_script_datasets,
    log_script_dataset_examples,
    reject_non_default_args,
    reject_trl_dataset_prep_args,
    resolve_vlm_run,
    run_trainer,
)

logger = get_logger(__name__, log_level="INFO")


def _require_privileged_columns(ds, args) -> None:
    """Raise when a column the collators read is absent: the confidence column, and while the OPD
    term carries weight, each column filling a slot the hint template names.

    A missing hint column would leave every row without a hint, and the OPD term would distil toward
    the unhinted model. The on-policy arm (:class:`DistributedSDPGTrainer`) gates its answer the same
    way; this is the offline half of it. Rows with a blank named slot get no hint, which is logged
    once here.
    """
    if args.confidence_field:
        require_render_column(ds, str(args.dataset), "confidence_field", args.confidence_field)
    if args.sdpg_beta_base == 0.0:
        return
    fields = SelfDistillationArguments.HINT_SLOT_FIELDS
    named = format_field_names(args.sdpg_hint_template)
    columns = {slot: getattr(args, field) for slot, field in fields.items() if slot in named and getattr(args, field)}
    for slot, column in columns.items():
        require_render_column(ds, str(args.dataset), fields[slot], column)
    train = ds["train"]
    values = [train[column] for column in columns.values()]
    hintless = sum(
        privileged_hint(args.sdpg_hint_template, **dict(zip(columns, row, strict=True))) is None
        for row in zip(*values, strict=True)
    )
    if hintless and is_global_main_process():
        logger.warning(
            f"{hintless} of {len(train)} train rows leave a slot of sdpg_hint_template blank ({sorted(columns.values())}), "
            f"so their privileged teacher runs on the plain prompt (no hint) and OPD distils them toward the "
            f"unhinted model. Check the dataset's hint columns."
        )


def _confidence_normalizer(train_dataset, eval_dataset, args) -> float | None:
    """The train split's mean ``conf ** confidence_power``, or ``None`` without a confidence column.

    Every rank reads the whole split (self-distillation refuses a presharded load with confidence
    weighting), so every rank computes the same divisor.
    """
    if args.confidence_field is None:
        return None
    return confidence_normalizer(train_dataset, eval_dataset, args.confidence_field, args.confidence_power)


def _audit_rows(collator, train_dataset, eval_dataset, num_proc) -> None:
    """Run every row through the collator's contract (text: length and alignment; VLM: alignment) now,
    where a raise is world-uniform: at collate time only the rank drawing the bad row raises, and its
    peers hang in the step's collectives until the watchdog.

    Every input column is removed from the map's output: the function still reads them, but a map
    writes each column it leaves in place to its cache file, so the audit would store a full copy of
    the split, image bytes included, on every node of a non-shared filesystem.
    """
    for split, dataset in (("train", train_dataset), ("eval", eval_dataset)):
        if dataset is not None:
            coordinated_map(
                dataset,
                audit_self_distill_row,
                desc=f"Auditing self-distill {split} rows",
                num_proc=num_proc,
                remove_columns=dataset.column_names,
                fn_kwargs={"collator": collator},
                cache_key_extras=collator.cache_signature(),
            )


def _build_text_dataset_and_collator(ds, args, tokenizer, max_length, model_config, num_proc, hint_template):
    train_dataset = ds["train"]
    eval_dataset = ds.get("test")
    if args.conversation_field not in train_dataset.column_names:
        raise ValueError(
            f"self_distillation requires a RAW conversation dataset with a '{args.conversation_field}' "
            f"field plus the privileged answer field ('{args.sdpg_answer_field}'); "
            f"got columns {train_dataset.column_names}."
        )
    collator = SelfDistillTextCollator(
        tokenizer=tokenizer,
        max_length=max_length,
        hint_template=hint_template,
        answer_field=args.sdpg_answer_field,
        solution_field=args.privileged_solution_field,
        confidence_field=args.confidence_field,
        confidence_power=args.confidence_power,
        confidence_normalizer=_confidence_normalizer(train_dataset, eval_dataset, args),
        response_prompt_template=args.assistant_message_template if args.train_on_completions_only else None,
        train_on_completions_only=args.train_on_completions_only,
        model_config=model_config,
        **text_render_kwargs(args),
    )
    _audit_rows(collator, train_dataset, eval_dataset, num_proc)
    return train_dataset, eval_dataset, collator


def _build_vlm_dataset_and_collator(ds, args, processor, tokenizer, max_length, num_proc, model_config, hint_template):
    """VLM path: map raw conversations to history/images (keeping the privileged fields) + collator.

    The over-length filter measures the hinted teacher branch, the longer of the two: a row whose
    hint alone pushes it over would otherwise raise inside one rank's collator and hang its peers.
    """
    privileged = tuple(f for f in (args.sdpg_answer_field, args.privileged_solution_field, args.confidence_field) if f)
    hint_kwargs = {"answer_field": args.sdpg_answer_field, "solution_field": args.privileged_solution_field}
    measured_history = (
        None if hint_template is None else partial(teacher_history, hint_template=hint_template, **hint_kwargs)
    )
    ds = prepare_vlm_dataset(
        ds,
        args,
        processor,
        tokenizer,
        max_length,
        num_proc,
        keep=privileged,
        desc="Processing VLM self-distillation dataset",
        measured_history=measured_history,
    )
    collator = SelfDistillVLMDataCollator(
        processor,
        tokenizer,
        max_length,
        response_prompt_template=args.assistant_message_template if args.train_on_completions_only else None,
        train_on_completions_only=args.train_on_completions_only,
        hint_template=hint_template,
        confidence_field=args.confidence_field,
        confidence_power=args.confidence_power,
        confidence_normalizer=_confidence_normalizer(ds["train"], ds.get("test"), args),
        model_config=model_config,
        **hint_kwargs,
    )
    _audit_rows(collator, ds["train"], ds.get("test"), num_proc)
    return ds["train"], ds.get("test"), collator


def _reference_path(args, model_config) -> str:
    """The ``L_ref`` anchor's weights: ``reference_model_name_or_path``, else the policy's init weights."""
    return args.reference_model_name_or_path or model_config.model_name_or_path


def _require_reference_token_ids(args, model_config) -> None:
    """A weighted anchor from another repo must tokenize as the policy's repo does.

    Reads only the two tokenizers' files, so main runs it before the dataset prep and the policy load;
    the run's added tokens are then grown into both alike.
    """
    ref_path = _reference_path(args, model_config)
    if args.reference_kl_coef <= 0 or ref_path == model_config.model_name_or_path:
        return

    def fetch():
        trust = model_config.trust_remote_code
        return (
            AutoTokenizer.from_pretrained(
                model_config.model_name_or_path, revision=model_config.model_revision, trust_remote_code=trust
            ),
            AutoTokenizer.from_pretrained(ref_path, trust_remote_code=trust),
        )

    require_same_token_ids(*hub_metadata_main_first("reference_tokenizers", fetch))


def _load_sdpg_reference(
    *, args, model_config, sft_config, dist_args, tokenizer, is_vlm: bool
) -> PreTrainedModel | None:
    """Load the frozen reference for the ``L_ref`` KL anchor, or ``None`` when the anchor is off.

    It goes through the shared frozen-reference loader for the reason the KL term exists: ``L_ref``
    is a divergence between this model's distribution and the policy's, so a reference on a
    different attention backend, carrying live sinks the policy reset, or missing the special tokens
    the run added moves the anchor rather than the model. The token-id check ran before any load.
    """
    if args.reference_kl_coef <= 0:
        return None
    ref_path = _reference_path(args, model_config)
    reference_model = load_frozen_reference_model(
        args,
        model_config,
        sft_config,
        tokenizer,
        ref_path,
        is_vlm=is_vlm,
        reset_sinks=dist_args.reset_sinks,
        attn_default=None,
        # The policy's pin names a commit in its own repo; a separate reference repo takes its main.
        revision=model_config.model_revision if ref_path == model_config.model_name_or_path else None,
    )
    if is_global_main_process():
        logger.info(f"Loaded frozen reference model from '{ref_path}' (alpha={args.reference_kl_coef})")
    return reference_model


def main():
    parser = H4ArgumentParser((SelfDistillationArguments, SFTConfig, ModelConfig, DistributedArguments))
    args, sft_config, model_config, distributed_args = parser.parse()

    # Inherited from SFTScriptArguments but unreachable: GenerateExamplesCallback needs a tokenized
    # "generate" split and self-distillation keeps the dataset raw. num_eval_examples rides along on
    # the default-comparing form, since its own default (50) is truthy.
    reject_non_default_args("Self-distillation", args, "generate_eval_examples", "num_eval_examples")

    # Both only feed the reference term, which is not built at all below this coefficient.
    if args.reference_kl_coef <= 0:
        reject_non_default_args(
            "Self-distillation with the reference anchor off (reference_kl_coef <= 0)",
            args,
            "reference_model_name_or_path",
            "reference_kl_loss",
        )
    # Both only shape the confidence weights, which no row carries without a confidence column.
    if args.confidence_field is None:
        reject_non_default_args(
            "Self-distillation without a confidence_field", args, "confidence_power", "confidence_weight_opd"
        )
    # The trainer builds its own token-mean cross-entropy, so TRL's loss_type reaches no loss: only the
    # two spellings of that same nll may stand.
    if sft_config.loss_type not in ("nll", "chunked_nll"):
        raise ValueError(
            f"loss_type={sft_config.loss_type!r} is not implemented for self-distillation: the trainer "
            f"computes its own cross-entropy (nll), so the requested loss would be silently ignored. "
            f"Remove loss_type from the YAML."
        )

    # Reject inherited options the SelfDistill collators do not implement; ignoring them would train
    # something other than the config states.
    if args.train_on_last_assistant_only:
        raise ValueError(
            "train_on_last_assistant_only is not implemented for self-distillation (the SelfDistill "
            "collators mask all assistant turns when train_on_completions_only is set). Remove it "
            "from the YAML."
        )
    if sft_config.packing or sft_config.padding_free:
        raise ValueError(
            "packing / padding_free are not supported for self-distillation: the SelfDistill "
            "collators tokenize the student and teacher branches at collation time and always "
            "right-pad, so both are forced off below. Remove them from the YAML."
        )
    # eval_packing has nothing to narrow with packing refused above.
    reject_trl_dataset_prep_args("Self-distillation", sft_config, "eval_packing")

    # The SelfDistill collator tokenizes the raw branches at collation time, so the raw conversation
    # and privileged columns have to survive HF's remove_unused_columns=True, which would strip them.
    sft_config.remove_unused_columns = False

    runtime = init_training_script(
        args,
        sft_config,
        model_config,
        distributed_args,
        script_prefix="self-distill",
        trainer_cls=DistributedSelfDistillationTrainer,
    )
    parallelism_config = runtime.parallelism_config
    _require_reference_token_ids(args, model_config)

    ds, dataset_presharded = load_script_datasets(args, parallelism_config, conversation_field=args.conversation_field)
    if dataset_presharded and args.confidence_field:
        raise ValueError(
            "confidence_field divides every weight by the train split's mean confidence, and a presharded "
            "dataset gives each data-parallel rank a different shard, so each would divide by its own. "
            "Load the raw dataset unsharded, or drop confidence_field."
        )
    _require_privileged_columns(ds, args)
    # The data path follows the run, not the checkpoint class: a natively-multimodal student
    # distilled on text-only rows is a text run (see is_vlm_run). Decided before the model load,
    # which requires the checkpoint's processor for an image run.
    is_vlm = resolve_vlm_run(args, model_config, ds, text_only_model=distributed_args.text_only_model)

    model, processing_class, tokenizer, is_vlm_checkpoint = load_model_for_training(
        model_config,
        sft_config,
        parallelism_config,
        vlm_run=is_vlm,
        reset_sinks=distributed_args.reset_sinks,
        train_sinks=distributed_args.train_sinks,
        weights_source=runtime.model_source,
        preserve_checkpoint_precision=runtime.policy_from_checkpoint,
        text_only_model=distributed_args.text_only_model,
    )
    tokenizer = apply_max_length(sft_config, args, model, tokenizer)
    processing_class = install_resolved_tokenizer(processing_class, tokenizer)
    enforce_text_path_padding_side(tokenizer, is_vlm)
    peft_config = setup_peft_model(args, model, model_config, "CAUSAL_LM")
    log_model_info(model, tokenizer)

    # No teacher branch is built while the OPD term is off.
    hint_template = args.sdpg_hint_template if args.sdpg_beta_base > 0 else None
    num_proc = resolve_map_num_proc(sft_config.dataset_num_proc)
    if is_vlm:
        train_dataset, eval_dataset, collator = _build_vlm_dataset_and_collator(
            ds, args, processing_class, tokenizer, sft_config.max_length, num_proc, model.config, hint_template
        )
    else:
        train_dataset, eval_dataset, collator = _build_text_dataset_and_collator(
            ds, args, tokenizer, sft_config.max_length, model.config, num_proc, hint_template
        )

    log_script_dataset_examples({"train": train_dataset, "test": eval_dataset}, tokenizer, args, sft_config)

    reference_model = _load_sdpg_reference(
        args=args,
        model_config=model_config,
        sft_config=sft_config,
        dist_args=distributed_args,
        tokenizer=tokenizer,
        is_vlm=is_vlm_checkpoint,
    )

    disable_trl_dataset_prep(sft_config)
    apply_distributed_trainer_config(sft_config, parallelism_config)

    callbacks = build_training_callbacks(args, sft_config, model, parallelism_config)

    barrier()

    trainer = DistributedSelfDistillationTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=processing_class,
        peft_config=peft_config,
        data_collator=collator,
        callbacks=callbacks,
        **distributed_trainer_kwargs(
            args, distributed_args, parallelism_config, dataset_presharded=dataset_presharded
        ),
        **args.build_sdpg_kwargs(),
        reference_model=reference_model,
        reference_kl_coef=args.reference_kl_coef,
        reference_kl_loss=args.reference_kl_loss,
        confidence_weight_opd=args.confidence_weight_opd,
        opd_exclude_eos=args.opd_exclude_eos,
    )
    run_trainer(
        trainer,
        runtime,
        method_name="Self-distillation",
        extra_start_log=[f"modality: {'vlm' if is_vlm else 'text'}"],
    )


if __name__ == "__main__":
    run_training(main)()
