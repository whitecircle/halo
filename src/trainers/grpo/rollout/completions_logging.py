"""Decoupled completion/trajectory artifact logging for GRPO trainers.

TRL's ``GRPOTrainer.log`` couples the console table, the parquet record, and the wandb table under
one ``log_completions`` flag. :func:`emit_completion_artifacts` splits them (parquet and backend table
gated by ``save``, console table by ``console``) so a run can keep the record without console output.
Built from TRL's ``_logs`` (detokenized text, rewards, advantages, extras, images).
"""

from __future__ import annotations

import logging
import os

import pandas as pd
import wandb
from transformers.utils import is_rich_available
from trl.trainer.utils import print_prompt_completions_sample

from src.distributed.runtime import fs_aware_save_rank

logger = logging.getLogger(__name__)

# Cap per-cell text in the backend table (parquet keeps the full text); untruncated cells stall the log call.
_TABLE_CELL_CHARS = 8000


def log_with_decoupled_completions(trainer, logs, start_time, super_log, *, save_completions: bool) -> None:
    """Run the base ``Trainer.log`` with TRL's coupled completions block off, then emit the decoupled
    parquet / console artifacts.

    ``super_log`` is the caller's bound ``super().log`` (kept in the caller so ``super()`` resolves at the
    right MRO position). ``log_completions`` is forced off around the delegated call.
    """
    trl_log_completions = trainer.log_completions
    trainer.log_completions = False
    try:
        super_log(logs, start_time)
    finally:
        trainer.log_completions = trl_log_completions
    emit_completion_artifacts(trainer, console=trl_log_completions, save=save_completions)


def emit_completion_artifacts(trainer, *, console: bool, save: bool) -> None:
    """Write the completions parquet + backend table and/or print the console sample table.

    Writer rank only; reads ``trainer._logs``. ``console`` prints the per-sample table; ``save`` writes
    the parquet under ``<output_dir>/completions/`` and logs a ``completions`` table to each backend.

    The writer is elected with ``fs_aware_save_rank`` like every other output artifact: with a shared
    output filesystem that is global rank 0, otherwise each node's local rank 0 writes its own copy.
    """
    if not fs_aware_save_rank():
        return
    logs = trainer._logs
    prompts = list(logs["prompt"])
    if not prompts:
        return

    if console and is_rich_available():
        print_prompt_completions_sample(
            prompts,
            list(logs["completion"]),
            {k: list(v) for k, v in logs["rewards"].items()},
            list(logs["advantages"]),
            trainer.state.global_step,
            trainer.num_completions_to_print,
            extra={k: list(v) for k, v in logs["extra"].items()},
        )

    if not save:
        return

    df_base = pd.DataFrame(
        {
            "step": [trainer.state.global_step] * len(prompts),
            "prompt": prompts,
            "completion": list(logs["completion"]),
            **{k: list(v) for k, v in logs["rewards"].items()},
            **{k: list(v) for k, v in logs["extra"].items()},
            "advantage": list(logs["advantages"]),
        }
    )

    # Eval logs arrive at an unchanged global_step; the suffix keeps them off the train parquet.
    mode_suffix = "" if trainer.model.training else "_eval"
    completions_dir = os.path.join(trainer.args.output_dir, "completions")
    try:
        os.makedirs(completions_dir, exist_ok=True)
        df_base.to_parquet(
            os.path.join(completions_dir, f"completions_{trainer.state.global_step:05d}{mode_suffix}.parquet")
        )
    except Exception as e:  # main-process-only; a raise here desyncs this rank into a hang
        logger.warning(f"completions parquet write failed (step {trainer.state.global_step}): {e}")

    if "wandb" not in (trainer.args.report_to or []) or wandb.run is None:
        return

    df = df_base.copy()
    for col in ("prompt", "completion"):
        df[col] = df[col].map(lambda s: s if len(s) <= _TABLE_CELL_CHARS else s[:_TABLE_CELL_CHARS] + " …[truncated]")

    # Images (VLM-GRPO) go only to the wandb table, matching TRL; the parquet stays text-only.
    images_raw = list(logs.get("images") or [])
    if images_raw:
        images = [[wandb.Image(img) for img in image_list] if image_list else [] for image_list in images_raw]
        df = pd.concat([df, pd.Series(images, name="image")], axis=1, copy=False)
    if trainer.log_unique_prompts:
        df = df.drop_duplicates(subset=["prompt"])
    wandb.log({"completions": wandb.Table(dataframe=df)})
