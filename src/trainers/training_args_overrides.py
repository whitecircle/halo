"""The TRL/HF TrainingArguments fields the toolkit owns, written before the TRL/HF constructor reads them."""

from accelerate.logging import get_logger

from src.models.config_access import config_sources
from src.runtime.filesystem import is_output_shared_filesystem
from src.runtime.ranks import get_num_nodes, is_global_main_process

logger = get_logger(__name__, log_level="info")


def align_save_on_each_node(training_args) -> None:
    """Make HF's ``save_on_each_node`` name the writers the toolkit's own saves use.

    The base save writes ``trainer_state.json``, ``scheduler.pt`` and ``optimizer.pt`` and rotates old
    checkpoints on ``args.should_save``, while every toolkit writer follows the output filesystem. On
    per-node storage the flag must be on, or no checkpoint resumes on nodes 1..N; on storage shared across
    nodes it must be off, or every node's local rank 0 writes and rotates the same files at once.
    """
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
    """Force TRL's ``use_liger_kernel`` off, leaving the model-level Liger kernels ``load_distributed_model``
    applied; True when the flag was on. ``reason`` is logged as a warning when given."""
    if training_args is None or not getattr(training_args, "use_liger_kernel", False):
        return False
    if reason:
        logger.warning(reason)
    training_args.use_liger_kernel = False
    return True


def keep_all_dataset_columns(training_args, reason: str | None = None) -> None:
    """Force ``remove_unused_columns`` off for a trainer whose collator or rollout reads columns the model's
    forward signature does not name; ``reason`` is logged as a warning when the flag was on."""
    if training_args is None or not training_args.remove_unused_columns:
        return
    if reason:
        logger.warning(reason)
    training_args.remove_unused_columns = False


def uses_evaluation(training_args) -> bool:
    """Whether the HF loop evaluates at all: an eval strategy, or ``eval_on_start`` under ``"no"``."""
    return training_args.eval_strategy not in ("no", None) or bool(training_args.eval_on_start)


def resolve_active_router_aux_loss_coef(model) -> float:
    """The router aux-loss coefficient ``model`` would add to its loss, else ``0``, over every config level: a
    multimodal MoE declares the routing fields on the text config, and either level may carry
    ``output_router_logits``."""
    config = getattr(model, "config", None)
    if config is None:
        return 0.0
    for cfg in config_sources(config):
        if not getattr(cfg, "output_router_logits", False):
            continue
        coef = getattr(cfg, "router_aux_loss_coef", 0) or 0
        if coef > 0:
            return float(coef)
    return 0.0
