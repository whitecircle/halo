"""The TRL/HF TrainingArguments fields the toolkit writes or reads before or around the TRL/HF constructor."""

import logging

from halo.runtime.filesystem import is_output_shared_filesystem
from halo.runtime.ranks import get_num_nodes, is_global_main_process

logger = logging.getLogger(__name__)


def align_save_on_each_node(training_args) -> None:
    """Make ``save_on_each_node`` name the writers the toolkit's saves use; the base save and rotation key on
    ``should_save``. Per-node storage needs it on, or no checkpoint resumes on nodes 1..N; shared storage needs it off,
    or every node's local rank 0 writes and rotates the same files at once."""
    per_node = not is_output_shared_filesystem()
    requested = bool(getattr(training_args, "save_on_each_node", False))
    if requested == per_node or (requested and get_num_nodes() <= 1):
        return
    training_args.save_on_each_node = per_node
    if not is_global_main_process():
        return
    if per_node:
        logger.info(
            "Non-shared output filesystem → forcing save_on_each_node=True so trainer_state.json / "
            "scheduler.pt / rng_state are written on every node (otherwise no checkpoint resumes on "
            "the non-zero nodes)."
        )
    else:
        logger.warning(
            f"save_on_each_node=True on a shared output filesystem across {get_num_nodes()} nodes → "
            "forcing it off: every node's local rank 0 would write and rotate the same checkpoint "
            "files concurrently. Set DIST_OUTPUT_SHARED_FILESYSTEM=0 if each node writes its own "
            "output directory."
        )


def disable_trl_liger(training_args, reason: str | None = None) -> bool:
    """Force TRL's ``use_liger_kernel`` off, leaving the kernels ``load_distributed_model`` applied; True when it was
    on."""
    if training_args is None or not getattr(training_args, "use_liger_kernel", False):
        return False
    if reason and is_global_main_process():
        logger.warning(reason)
    training_args.use_liger_kernel = False
    return True


def keep_all_dataset_columns(training_args, reason: str | None = None) -> None:
    """Force ``remove_unused_columns`` off for a collator or rollout reading columns the forward signature lacks."""
    if training_args is None or not training_args.remove_unused_columns:
        return
    if reason and is_global_main_process():
        logger.warning(reason)
    training_args.remove_unused_columns = False


def uses_evaluation(training_args) -> bool:
    return training_args.eval_strategy not in ("no", None) or bool(training_args.eval_on_start)
