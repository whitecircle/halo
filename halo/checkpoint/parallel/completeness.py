"""Completion markers a save publishes last and a resume takes a directory by, and clearing a rewritten step dir."""

from __future__ import annotations

import json
import logging
import os
from collections.abc import Iterator
from contextlib import contextmanager, suppress
from functools import partial

from halo.checkpoint.atomic import publish_staged_file
from halo.checkpoint.format import (
    CHECKPOINT_COMPLETION_MARKERS,
    RESUME_ADAPTER_DIR,
    RESUME_ADAPTER_MARKER_FILE,
    TRAINER_STATE_FILE,
    UNCOMMITTED_TRAINER_STATE_FILE,
    is_per_rank_resume_file,
)
from halo.runtime.consensus import DeferredRankFailure, barrier_on_exit, rank_consensus
from halo.runtime.filesystem import is_fs_save_rank

logger = logging.getLogger(__name__)


def complete_on_every_rank(checkpoint: str) -> bool:
    """COLLECTIVE. On a non-shared FS each node reads its own copy, so the verdict is a world one."""
    return rank_consensus(os.path.isfile(os.path.join(checkpoint, TRAINER_STATE_FILE))).all_ok


def _earlier_save_files(checkpoint_dir: str) -> list[str]:
    """What a save rewriting ``checkpoint_dir`` must remove first. Markers go by name, so a stale listing cannot keep
    one that vouches for this save; only the per-rank names, which follow the earlier world size, come from the
    listing."""
    try:
        listed = os.listdir(checkpoint_dir)
    except FileNotFoundError:
        return []
    return [*CHECKPOINT_COMPLETION_MARKERS, *filter(is_per_rank_resume_file, listed)]


def prepare_checkpoint_rewrite(checkpoint_dir: str) -> None:
    """COLLECTIVE. Unlink an earlier save's leftovers on each filesystem's save rank ahead of every rank's write.
    Markers go so a torn save is refused on resume; per-rank resume files go because their writers truncate an existing
    name, which a shard merge may have hard-linked into another directory whose copy would change too."""
    with barrier_on_exit(f"Clearing the earlier save's files from {checkpoint_dir}"):
        if is_fs_save_rank():
            for name in _earlier_save_files(checkpoint_dir):
                with suppress(FileNotFoundError):
                    os.remove(os.path.join(checkpoint_dir, name))


@contextmanager
def trainer_state_withheld(state) -> Iterator[None]:
    """Have the base save write ``state`` as :data:`UNCOMMITTED_TRAINER_STATE_FILE`, since it writes it before the
    sidecars and shards; :func:`commit_trainer_state` publishes it last."""
    write = state.save_to_json

    def withheld(json_path: str) -> None:
        directory, name = os.path.split(json_path)
        write(os.path.join(directory, UNCOMMITTED_TRAINER_STATE_FILE) if name == TRAINER_STATE_FILE else json_path)

    state.save_to_json = withheld
    try:
        yield
    finally:
        del state.save_to_json


def _publish_withheld_trainer_state(checkpoint_dir: str) -> None:
    staged = os.path.join(checkpoint_dir, UNCOMMITTED_TRAINER_STATE_FILE)
    if not os.path.isfile(staged):
        raise FileNotFoundError(
            f"{staged} is missing: the base save wrote no trainer state on this save rank. Its writers "
            f"(args.should_save, set by save_on_each_node) must be the output filesystem's save ranks "
            f"(DIST_OUTPUT_SHARED_FILESYSTEM)."
        )
    publish_staged_file(staged, os.path.join(checkpoint_dir, TRAINER_STATE_FILE))


def commit_trainer_state(checkpoint_dir: str) -> None:
    """COLLECTIVE. Publish the withheld trainer state last, once every other file is on disk and before rotation. A
    save rank holding none raises everywhere, since its checkpoint would never resume and rotation could remove the
    last that does."""
    guard = DeferredRankFailure(f"Committing {TRAINER_STATE_FILE} in {checkpoint_dir}")
    if is_fs_save_rank():
        guard.run(partial(_publish_withheld_trainer_state, checkpoint_dir))
    guard.reject()


def mark_resume_adapter_complete(checkpoint_dir: str, *, is_save_rank: bool) -> None:
    """COLLECTIVE. Mark the checkpoint as resuming from :data:`RESUME_ADAPTER_DIR`, after each save rank's write."""
    with barrier_on_exit(f"Writing {RESUME_ADAPTER_MARKER_FILE} to {checkpoint_dir}"):
        if is_save_rank:
            with open(os.path.join(checkpoint_dir, RESUME_ADAPTER_MARKER_FILE), "w") as fh:
                json.dump({"adapter_dir": RESUME_ADAPTER_DIR}, fh, indent=2)
            logger.info(f"Saved the resume adapter of merged checkpoint {checkpoint_dir}")
