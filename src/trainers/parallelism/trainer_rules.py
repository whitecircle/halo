"""Trainer construction refusals for the parallelism modes.

Each helper raises the mechanism-naming error its trainers share, parameterized by the per-trainer parts;
gates unique to one trainer stay in its ``_validate_pp_mode`` / ``_reject_pp_explicit_options`` hook.
"""

from __future__ import annotations

from src.distributed.context_parallel.config import CPConfigError
from src.optimizers.registry import NAMED_OPTIMIZER_BUILDERS, optim_name


def require_model_and_args_kwargs(kwargs: dict) -> None:
    """Reject positional ``model``/``args`` under PP: the pipeline seam rewrites both by key."""
    if "model" not in kwargs or "args" not in kwargs:
        raise ValueError(
            "Pipeline parallelism requires `model` and `args` passed as keywords: the mixin's "
            "pipeline seam splits kwargs['model'] into this rank's stage and reads "
            "kwargs['args'] for the microbatch conversion."
        )


def reject_pp_compute_metrics(compute_metrics, mechanism: str) -> None:
    """Reject ``compute_metrics`` under PP for a trainer whose prediction convention a pipeline
    cannot serve, naming why.

    The pipeline serves metrics by reducing the last stage's output through
    ``PipelineLossAdapter.predictions_fn`` before broadcasting it down the chain. A trainer whose non-PP
    ``prediction_step`` hands metrics the raw logits plane, or nothing at all, has no reduction to
    reproduce. ``mechanism`` is that trainer's specific reason.
    """
    if compute_metrics is None:
        return
    raise ValueError(f"compute_metrics is not supported under pipeline parallelism: {mechanism}")


def reject_pp_peft(peft_config, explicit_param_trainer: str | None = None) -> None:
    """Reject PEFT/LoRA under PP (``explicit_param_trainer`` names a trainer whose ``peft_config``
    is an explicit ctor parameter the mixin's kwargs-based rejection cannot see).

    Injection itself would work on a stage; the checkpoint path does not. The adapter save runs
    through ``save_peft_adapters`` ahead of the PP saver, so it writes stage-local layer indices from
    the save rank alone, the PP resume path never reaches the adapter restore, and
    ``layers_to_transform`` / ``all-linear`` would select re-based layers and the stage head.
    """
    if peft_config is None:
        return
    message = (
        "PEFT/LoRA is not supported under pipeline parallelism: the adapter checkpoint path is "
        "not stage-aware — PeftAdapterSaver writes adapter_model.safetensors from the save rank "
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


def require_precomputed_columns(dataset, columns: tuple[str, ...], role: str = "train") -> None:
    """Require the precomputed reference column(s) to already be in the ``role`` dataset under PP.

    Applies to the eval dataset too: the loss-only PP eval consumes the same columns, and TRL's
    in-init sweep otherwise runs over the eval dataset as well.
    """
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


def require_precomputed_reference(
    trainer: str, training_args, train_dataset, eval_dataset, columns: tuple[str, ...]
) -> None:
    """Require PP's precompute-only reference contract: the flag set and the column(s) present.

    ``trainer`` names the wrapper in the message; ``columns`` are its reference log-prob columns.
    The eval dataset is checked too (a dict of splits included): TRL's in-init sweep runs over it as
    well, and under PP no rank holds a full model to run it.
    """
    if not training_args.precompute_ref_log_probs:
        subject = f"the {columns[0]} column" if len(columns) == 1 else f"the {'/'.join(columns)} columns"
        raise ValueError(
            f"{trainer} under pipeline parallelism is precompute-only: without "
            f"precompute_ref_log_probs=True TRL instantiates a live reference model from the "
            f"config path and forwards it every step — a full-model path no pipeline rank can "
            f"run. Set precompute_ref_log_probs=True and put {subject} in the dataset."
        )
    require_precomputed_columns(train_dataset, columns)
    if eval_dataset is not None:
        for dataset in eval_dataset.values() if isinstance(eval_dataset, dict) else (eval_dataset,):
            if dataset is not None:
                require_precomputed_columns(dataset, columns, role="eval")


def reject_pp_activation_offloading(training_args) -> None:
    """Reject TRL's activation_offloading under PP: its training_step wrap would never engage."""
    if getattr(training_args, "activation_offloading", False):
        raise ValueError(
            "activation_offloading is not supported under pipeline parallelism: TRL applies it "
            "by wrapping training_step, which the PP mixin's schedule-driven step bypasses — it "
            "would silently never engage."
        )


def reject_pp_named_optimizer(training_args, parallelism_config) -> None:
    """Reject a toolkit-named optimizer (``NAMED_OPTIMIZER_BUILDERS``) under PP, where none is validated."""
    optim = optim_name(getattr(training_args, "optim", None))
    if parallelism_config.is_pp_mode and optim in NAMED_OPTIMIZER_BUILDERS:
        raise ValueError(
            f"optim={optim!r} is not supported under pipeline parallelism: the stage-local optimizer "
            f"build and the per-stage optimizer shards a resume restores are validated only with the "
            f"AdamW family. Use an AdamW optimizer under pipeline parallelism, or run {optim} without it."
        )


def reject_cp_trainer_args(
    training_args,
    compute_metrics=None,
    preprocess_logits_for_metrics=None,
) -> None:
    """Reject trainer settings whose loss/metric path needs full-sequence logits.

    A CP rank's forward returns logits for its own chunk only, and
    :class:`~src.distributed.context_parallel.ulysses_model.UlyssesModel` computes the
    boundary-aware loss itself. Anything that re-derives the loss or a metric from
    ``(logits, labels)`` outside the wrapper pairs a chunk of logits with the full-length labels.

    Raises:
        CPConfigError: If a rejected setting is active.
    """
    if getattr(training_args, "label_smoothing_factor", 0.0):
        raise CPConfigError(
            "label_smoothing_factor > 0 is not supported with Context Parallelism: the Trainer pops "
            "`labels` before the model call and smooths the loss itself, so the CP wrapper returns "
            "no loss and the smoother compares FULL-length labels against this rank's "
            "sequence-chunk logits. Set label_smoothing_factor=0, or train without CP."
        )

    if getattr(training_args, "loss_type", None) == "dft":
        raise CPConfigError(
            "loss_type='dft' is not supported with Context Parallelism: TRL implements it either by "
            "installing a compute_loss_func (the Trainer then pops `labels`, leaving the CP wrapper "
            "with no loss and the func with full-length labels) or, under Liger, by a "
            "`use_token_scaling` forward flag the CP wrapper's own loss ignores. Use "
            "loss_type='nll', or train without CP."
        )

    # Not gated on eval_strategy: a trainer built with eval_strategy='no' can still reach the
    # misaligned metric path through a direct evaluate()/predict() call.
    if compute_metrics is not None or preprocess_logits_for_metrics is not None:
        raise CPConfigError(
            "Evaluation under Context Parallelism is loss-only: compute_metrics / "
            "preprocess_logits_for_metrics receive this rank's sequence-chunk logits gathered "
            "against the full-length labels, so the two are misaligned. Drop them (eval_loss still "
            "reports), or evaluate from a saved checkpoint without CP."
        )
