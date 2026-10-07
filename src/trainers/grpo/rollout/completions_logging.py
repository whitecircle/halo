"""Decoupled completion/trajectory artifact logging for GRPO trainers.

TRL's ``GRPOTrainer.log`` couples the console table, the parquet record, and the wandb table
under one ``log_completions`` flag. :func:`emit_completion_artifacts` splits them — parquet + wandb
table gated by ``save``, console table by ``console`` — so a run can keep the record without console
spam, and :class:`DecoupledCompletionsLogMixin` routes the on-policy trainers' ``log`` through it.
Built from TRL's ``_logs`` (detokenized text, rewards, advantages, extras, images).
"""

from __future__ import annotations

import os
from collections import defaultdict, deque
from collections.abc import Mapping
from functools import partial

import pandas as pd
import wandb
from transformers.utils import is_rich_available
from trl.trainer.utils import print_prompt_completions_sample

from src.distributed.runtime import DeferredRankFailure, fs_aware_save_rank, is_global_main_process

# Cap per-cell text in the wandb table (parquet keeps the full text); untruncated cells stall the log call.
_TABLE_CELL_CHARS = 8000


class DecoupledCompletionsLogMixin:
    """``log`` with TRL's coupled completions block off, then the decoupled artifacts
    (:func:`emit_completion_artifacts`): the record under ``save_completions``, the console table under
    TRL's ``log_completions``. Mixed in ahead of TRL's ``GRPOTrainer``; ``_save_completions`` is set by
    the on-policy init spine."""

    _save_completions: bool = True

    def log(self, logs: dict[str, float], start_time: float | None = None) -> None:
        trl_log_completions = self.log_completions
        self.log_completions = False
        try:
            super().log(logs, start_time)
        finally:
            self.log_completions = trl_log_completions
        emit_completion_artifacts(self, console=trl_log_completions, save=self._save_completions)


def unbounded_completion_logs() -> dict:
    """TRL's ``_logs`` layout without its cap of one generation batch per slot.

    TRL sizes each deque to ``generation_batch_size`` because it refills them every generation and
    prints only the last batch. The async trainer fills them per rollout round, and an eval round is
    the whole eval set, which the cap would cut to its tail. :func:`emit_completion_artifacts` empties
    the slots after every log instead, so each holds exactly the rows since the last one.
    """
    return {
        "images": deque(),
        "prompt": deque(),
        "completion": deque(),
        "rewards": defaultdict(deque),
        "advantages": deque(),
        "extra": defaultdict(deque),
    }


def _clear_completion_logs(logs: Mapping) -> None:
    for slot in logs.values():
        for buffer in slot.values() if isinstance(slot, Mapping) else (slot,):
            buffer.clear()


def emit_completion_artifacts(trainer, *, console: bool, save: bool, mode: str | None = None) -> None:
    """Write the completions parquet + wandb table and/or print the console sample table, then
    empty ``trainer._logs`` on every rank. COLLECTIVE — every rank calls it.

    Writer rank only; reads ``trainer._logs``. ``console`` prints the per-sample table on the global
    main process alone, where a per-node writer would print it once per node; ``save`` writes the
    parquet under ``<output_dir>/completions/`` and logs a ``completions`` table to wandb.
    ``mode`` names the rows' mode when it is not the model's current one (train rows written as an
    eval round begins); by default the model's mode picks the file.

    Elected with ``fs_aware_save_rank`` like every other output artifact: on a non-shared output
    filesystem global rank 0 is the only writer, so nodes 1..N would lose their completions entirely.
    A writer's failure (a full disk, a wandb outage) is raised on every rank, since raised on the
    writer alone it would leave the peers in the next collective; with neither artifact asked for (a
    config verdict every rank shares) nothing is written and no verdict is taken. The buffers are
    emptied on writers and non-writers alike: they are unbounded on the async trainer, and a rank that
    never wrote would otherwise hold the run's whole record.
    """
    if not (console or save):
        _clear_completion_logs(trainer._logs)
        return
    mode = mode or ("train" if trainer.model.training else "eval")
    guard = DeferredRankFailure(f"Completions record at step {trainer.state.global_step}")
    try:
        if fs_aware_save_rank():
            guard.run(partial(_emit_completion_artifacts, trainer, console=console, save=save, mode=mode))
    finally:
        _clear_completion_logs(trainer._logs)
    guard.reject()


def _emit_completion_artifacts(trainer, *, console: bool, save: bool, mode: str) -> None:
    logs = trainer._logs
    prompts = list(logs["prompt"])
    if not prompts:
        return

    if console and is_rich_available() and is_global_main_process():
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
    mode_suffix = "" if mode == "train" else "_eval"
    completions_dir = os.path.join(trainer.args.output_dir, "completions")
    os.makedirs(completions_dir, exist_ok=True)
    df_base.to_parquet(
        os.path.join(completions_dir, f"completions_{trainer.state.global_step:05d}{mode_suffix}.parquet")
    )

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
