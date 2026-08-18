"""The construction-time PP rejection vocabulary shared by trainer adapters.

Each helper raises the exact, mechanism-naming error its trainers share; only the genuinely
per-trainer parts — column names, trainer name, the mechanism clause — are parameters. Gates unique
to one trainer (DPO's loss-type subset, KTO's ``apo_zero_unpaired`` restriction, SMPO's clip/label
gates, offline GRPO's length caps) stay in that trainer's ``_validate_pp_mode`` hook, or — where the
options are explicit ctor parameters the mixin cannot see — its ``_reject_pp_explicit_options``.
"""

from __future__ import annotations


def require_model_and_args_kwargs(kwargs: dict) -> None:
    """Reject positional ``model``/``args`` under PP — the pipeline seam rewrites both by key."""
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
    ``PPLossAdapter.predictions_fn`` before broadcasting it down the chain. A trainer whose non-PP
    ``prediction_step`` hands metrics the raw logits plane — or hands them nothing at all — has no
    reduction to reproduce, and silently substituting a different tensor would make the PP metric
    incomparable with the same run without PP. ``mechanism`` is that trainer's specific reason.
    """
    if compute_metrics is None:
        return
    raise ValueError(f"compute_metrics is not supported under pipeline parallelism: {mechanism}")


def reject_pp_peft(peft_config, explicit_param_trainer: str | None = None) -> None:
    """Reject PEFT/LoRA under PP (``explicit_param_trainer`` names a trainer whose ``peft_config``
    is an explicit ctor parameter the mixin's kwargs-based rejection cannot see).

    Injection itself would work on a stage (target modules match by name suffix, and the stage's
    naming map round-trips adapter paths); what does not is everything after it, silently: the
    adapter save runs through ``PeftAdapterSaver`` ahead of the PP saver, so it writes
    stage-LOCAL layer indices from the save rank alone (stage 1 claims stage 0's layers, stages
    1..N-1 are dropped), the PP resume path never reaches the adapter restore, and
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
    """Reject a live reference model under PP; ``column_hint`` names the trainer's precomputed
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
        f"hidden_states, before the pipeline runtime even exists). Precompute the {noun} in a "
        f"non-PP run first."
    )


def require_precomputed_reference(
    trainer: str, training_args, train_dataset, eval_dataset, columns: tuple[str, ...]
) -> None:
    """Require PP's precompute-only reference contract: the flag ON and the column(s) already there.

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
    """Reject TRL's activation_offloading under PP — its training_step wrap would never engage."""
    if getattr(training_args, "activation_offloading", False):
        raise ValueError(
            "activation_offloading is not supported under pipeline parallelism: TRL applies it "
            "by wrapping training_step, which the PP mixin's schedule-driven step bypasses — it "
            "would silently never engage."
        )
