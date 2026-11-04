"""Completion artifacts of the GRPO trainers. TRL's ``GRPOTrainer.log`` couples the console table, parquet record and
wandb table under one flag; here ``save`` gates the record and ``console`` the table."""

from __future__ import annotations

import math
import os
from collections import defaultdict, deque
from collections.abc import Mapping, Sequence
from functools import partial
from typing import Any, ClassVar

import pandas as pd
import wandb
from transformers.utils import is_rich_available
from trl.trainer.utils import print_prompt_completions_sample

from halo.runtime.collectives import gather_on_rank0
from halo.runtime.consensus import DeferredRankFailure
from halo.runtime.filesystem import is_fs_save_rank
from halo.runtime.ranks import get_global_world_size, is_global_main_process, rank_tag

# Cap per-cell text in the wandb table (parquet keeps the full text); untruncated cells stall the log call.
_TABLE_CELL_CHARS = 8000


class DecoupledCompletionsLogMixin:
    """TRL's coupled completions block off, then the decoupled artifacts. Mixed in ahead of TRL's ``GRPOTrainer``; the
    on-policy init spine sets ``_save_completions``."""

    _save_completions: bool = True
    # Whether ``_logs`` holds this rank's own rows, each rank writing its own record part and sending a sample to
    # the tables, rather than the world's (TRL's all-gather), written whole by the filesystem save ranks.
    RANK_LOCAL_COMPLETION_LOGS: ClassVar[bool] = False

    def log(self, logs: dict[str, float], start_time: float | None = None) -> None:
        trl_log_completions = self.log_completions
        self.log_completions = False
        try:
            super().log(logs, start_time)
        finally:
            self.log_completions = trl_log_completions
        emit_completion_artifacts(self, console=trl_log_completions, save=self._save_completions)


def unbounded_completion_logs() -> dict:
    """No cap of one generation batch per slot: an eval round is the whole eval set, which the cap would cut to its
    tail. :func:`emit_completion_artifacts` empties the slots after every log instead."""
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


def _logged_rows(logs: Mapping, rows: Sequence[int] | None = None) -> dict[str, Any]:
    def take(buffer) -> list:
        values = list(buffer)
        return values if rows is None else [values[i] for i in rows]

    images = list(logs.get("images") or ())
    return {
        "prompt": take(logs["prompt"]),
        "completion": take(logs["completion"]),
        "rewards": {k: take(v) for k, v in logs["rewards"].items()},
        "advantages": take(logs["advantages"]),
        "extra": {k: take(v) for k, v in logs["extra"].items()},
        "images": take(images) if images else [],
    }


def _merged_rows(parts: Sequence[Mapping[str, Any] | None]) -> dict[str, Any]:
    """Parts are in :func:`_logged_rows`'s layout, None from a rank whose step failed."""
    parts = [part for part in parts if part is not None and part["prompt"]]

    def cat(slot: str) -> list:
        return [value for part in parts for value in part[slot]]

    def cat_columns(slot: str) -> dict[str, list]:
        names = dict.fromkeys(name for part in parts for name in part[slot])
        return {name: [value for part in parts for value in part[slot][name]] for name in names}

    return {
        "prompt": cat("prompt"),
        "completion": cat("completion"),
        "rewards": cat_columns("rewards"),
        "advantages": cat("advantages"),
        "extra": cat_columns("extra"),
        "images": cat("images"),
    }


def _cell(text: str) -> str:
    return text if len(text) <= _TABLE_CELL_CHARS else text[:_TABLE_CELL_CHARS] + " …[truncated]"


def _record_frame(rows: Mapping[str, Any], step: int) -> pd.DataFrame:
    """Images go to the wandb table alone, matching TRL."""
    return pd.DataFrame(
        {
            "step": [step] * len(rows["prompt"]),
            "prompt": rows["prompt"],
            "completion": rows["completion"],
            **rows["rewards"],
            **rows["extra"],
            "advantage": rows["advantages"],
        }
    )


def _write_record(trainer, rows: Mapping[str, Any], mode: str) -> None:
    step = trainer.state.global_step
    # Eval logs arrive at an unchanged global_step; the suffix keeps them off the train record.
    name = f"completions_{step:05d}{'' if mode == 'train' else '_eval'}"
    completions_dir = os.path.join(trainer.args.output_dir, "completions")
    # One directory per step for the parts: a flat one would hold world × steps files.
    path = (
        os.path.join(completions_dir, name, f"{rank_tag()}.parquet")
        if trainer.RANK_LOCAL_COMPLETION_LOGS
        else os.path.join(completions_dir, f"{name}.parquet")
    )
    os.makedirs(os.path.dirname(path), exist_ok=True)
    _record_frame(rows, step).to_parquet(path)


def _wants_wandb_table(trainer, save: bool) -> bool:
    """The config's verdict, alike on every rank; only global rank 0 holds a wandb run."""
    return save and "wandb" in (trainer.args.report_to or [])


def _show_tables(trainer, rows: Mapping[str, Any], *, console: bool, wandb_table: bool) -> None:
    if not rows["prompt"] or not is_global_main_process():
        return
    if console and is_rich_available():
        print_prompt_completions_sample(
            rows["prompt"],
            rows["completion"],
            rows["rewards"],
            rows["advantages"],
            trainer.state.global_step,
            trainer.num_completions_to_print,
            extra=rows["extra"],
        )
    if not wandb_table or wandb.run is None:
        return
    df = _record_frame(rows, trainer.state.global_step)
    for col in ("prompt", "completion"):
        df[col] = df[col].map(_cell)
    if rows["images"]:
        images = [[wandb.Image(img) for img in image_list] if image_list else [] for image_list in rows["images"]]
        df = pd.concat([df, pd.Series(images, name="image")], axis=1, copy=False)
    if trainer.log_unique_prompts:
        df = df.drop_duplicates(subset=["prompt"])
    wandb.log({"completions": wandb.Table(dataframe=df)})


def _table_sample(trainer) -> dict[str, Any]:
    """``ceil(num_completions_to_print / world size)`` rows spread evenly over this rank's, the text cut to cell width,
    so what reaches global rank 0 stays a few cells per rank."""
    logs = trainer._logs
    count = len(logs["prompt"])
    wanted = trainer.num_completions_to_print
    quota = count if wanted is None else min(count, math.ceil(wanted / get_global_world_size()))
    sample = _logged_rows(logs, [k * count // quota for k in range(quota)] if quota else [])
    sample["prompt"] = [_cell(text) for text in sample["prompt"]]
    sample["completion"] = [_cell(text) for text in sample["completion"]]
    return sample


def _emit_world_rows(trainer, *, console: bool, save: bool, mode: str) -> None:
    rows = _logged_rows(trainer._logs)
    if not rows["prompt"]:
        return
    if save:
        _write_record(trainer, rows, mode)
    _show_tables(trainer, rows, console=console, wandb_table=_wants_wandb_table(trainer, save))


def _emit_rank_rows(trainer, guard: DeferredRankFailure, *, console: bool, save: bool, mode: str) -> None:
    """COLLECTIVE when a table is wanted (a config verdict every rank shares)."""
    if save and trainer._logs["prompt"]:
        guard.run(lambda: _write_record(trainer, _logged_rows(trainer._logs), mode))
    wandb_table = _wants_wandb_table(trainer, save)
    if not (console or wandb_table):
        return
    # A rank whose step failed sends None, so the gather still completes; its failure is joined after.
    parts = gather_on_rank0(guard.run(partial(_table_sample, trainer)))
    if parts is not None:
        guard.run(lambda: _show_tables(trainer, _merged_rows(parts), console=console, wandb_table=wandb_table))


def emit_completion_artifacts(trainer, *, console: bool, save: bool, mode: str | None = None) -> None:
    """COLLECTIVE. Write the requested artifacts, then empty the logs on every rank, since they are unbounded on the
    async trainer. ``mode`` names the rows' mode when it is not the model's current one. A failure on any rank raises on
    every rank, so no peer waits."""
    if not (console or save):
        _clear_completion_logs(trainer._logs)
        return
    mode = mode or ("train" if trainer.model.training else "eval")
    guard = DeferredRankFailure(f"Completions record at step {trainer.state.global_step}")
    try:
        if trainer.RANK_LOCAL_COMPLETION_LOGS:
            _emit_rank_rows(trainer, guard, console=console, save=save, mode=mode)
        elif is_fs_save_rank():
            guard.run(partial(_emit_world_rows, trainer, console=console, save=save, mode=mode))
    finally:
        _clear_completion_logs(trainer._logs)
    guard.reject()
