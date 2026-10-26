"""Resume sidecars beside the model weights: the LR scheduler and the router balancing biases."""

from __future__ import annotations

import logging
import os
from functools import partial

import torch

from src.checkpoint.format import ROUTER_BALANCING_BIASES_FILE, SCHEDULER_STATE_FILE
from src.checkpoint.parallel.reading import consensus_read
from src.models.moe.balancing import BALANCING_BIASES_ATTR
from src.runtime.collectives import barrier_on_exit, combine_across_ranks
from src.runtime.consensus import rank_consensus
from src.runtime.filesystem import is_fs_save_rank
from src.runtime.ranks import get_global_rank, is_global_main_process

logger = logging.getLogger(__name__)


def save_lr_scheduler(lr_scheduler, checkpoint_dir: str) -> None:
    """COLLECTIVE. Write ``scheduler.pt`` from the FS-aware save rank(s), for a ``save_only_model`` save the base
    Trainer drops it from: the LR schedule must continue from the resumed step. No-op without a scheduler."""
    if lr_scheduler is None:
        return
    with barrier_on_exit():
        if is_fs_save_rank():
            os.makedirs(checkpoint_dir, exist_ok=True)
            torch.save(lr_scheduler.state_dict(), os.path.join(checkpoint_dir, SCHEDULER_STATE_FILE))
            if is_global_main_process():
                logger.info(f"✓ Persisted LR scheduler to {checkpoint_dir} (EP/CP resume under save_only_model)")


def restore_lr_scheduler(lr_scheduler, optimizer, checkpoint: str) -> bool:
    """COLLECTIVE. Restore ``lr_scheduler`` from ``scheduler.pt`` and push its LR into ``optimizer``; True when
    restored. Runs on every resume path, the optimizer state's warm restarts included."""

    def restore(path: str) -> None:
        """Applied inside the joined read: a state dict the scheduler refuses fails the world like a torn file."""
        if lr_scheduler is None:
            return  # nothing to restore into; a present file is then not an error
        lr_scheduler.load_state_dict(torch.load(path, map_location="cpu", weights_only=False))

    # Absent everywhere is a legitimate warm restart; a subset or a torn copy raises.
    _state, path = consensus_read(
        os.path.join(checkpoint, SCHEDULER_STATE_FILE),
        restore,
        what=SCHEDULER_STATE_FILE,
        checkpoint=checkpoint,
        remedy=f" Or delete {SCHEDULER_STATE_FILE} to explicitly accept an LR-schedule warm restart from step 0.",
    )
    if path is None or lr_scheduler is None:
        return False
    # load_state_dict updates the scheduler alone and HF steps the optimizer first, so the first resumed
    # step would run at the LR _initial_step() left in param_groups (~0 under a warmup).
    last_lr = getattr(lr_scheduler, "_last_lr", None)
    if optimizer is not None and last_lr:
        if len(last_lr) != len(optimizer.param_groups):
            logger.warning(
                f"Restored LR schedule tracks {len(last_lr)} param group(s) but the optimizer has "
                f"{len(optimizer.param_groups)} — leaving the live learning rates untouched; the first "
                f"resumed step runs at the schedule's step-0 LR."
            )
        else:
            for group, lr in zip(optimizer.param_groups, last_lr, strict=True):
                group["lr"] = lr
    if is_global_main_process():
        logger.info(f"✓ Restored LR scheduler state from {checkpoint}")
    return True


def save_balancing_biases(model, output_dir: str, *, is_pp_mode: bool) -> None:
    """COLLECTIVE. Write the router balancing biases, replica-identical plain attributes the model checkpoint
    lacks, from the FS-aware save rank(s); no-op without a balancing router.

    Under PP the stages' biases are gathered to rank 0 under unsplit-model names and the merge broadcast
    back (an all-gather would materialize one dict per rank on every rank).
    """
    biases = {
        name: getattr(module, BALANCING_BIASES_ATTR).detach().cpu()
        for name, module in model.named_modules()
        if getattr(module, BALANCING_BIASES_ATTR, None) is not None
    }
    if is_pp_mode:
        biases = {model.global_parameter_name(name): bias for name, bias in biases.items()}
        # FS-aware writer: a rank-0-only write leaves nodes 1..N without the file on a non-shared
        # FS, so the merge has to reach every node's save rank.
        merged = combine_across_ranks(
            biases,
            lambda gathered: {name: bias for stage in gathered for name, bias in stage.items()},
            what="the router balancing biases",
        )
        with barrier_on_exit():
            if merged and is_fs_save_rank():
                torch.save(merged, os.path.join(output_dir, ROUTER_BALANCING_BIASES_FILE))
        return
    with barrier_on_exit():
        if biases and is_fs_save_rank():
            torch.save(biases, os.path.join(output_dir, ROUTER_BALANCING_BIASES_FILE))


def restore_balancing_biases(model, checkpoint: str, *, is_pp_mode: bool) -> None:
    """COLLECTIVE. Copy the biases :func:`save_balancing_biases` wrote back into the live routers.

    A saved name the model lacks is skipped; a live router the file lacks, a file matching no live router
    and an absent file under balancing routers each warn. A saved shape other than the live one raises on
    every rank, since ``copy_`` would broadcast it.
    """
    saved, path = consensus_read(
        os.path.join(checkpoint, ROUTER_BALANCING_BIASES_FILE),
        partial(torch.load, map_location="cpu", weights_only=True),
        what=ROUTER_BALANCING_BIASES_FILE,
        checkpoint=checkpoint,
    )
    if path is None:
        if any(getattr(module, BALANCING_BIASES_ATTR, None) is not None for module in model.modules()):
            # Every rank logs: under PP only the stages holding routers can see this.
            logger.warning(
                f"[rank {get_global_rank()}] No {ROUTER_BALANCING_BIASES_FILE} in {checkpoint} but the "
                f"model carries router balancing biases — they stay zero-init and balancing restarts "
                f"from scratch (checkpoint predates the balancing-bias sidecar?).",
            )
        return  # no balancing biases anywhere
    restored = 0
    missing: list[str] = []
    mismatched: list[str] = []
    for name, module in model.named_modules():
        bias = getattr(module, BALANCING_BIASES_ATTR, None)
        if bias is None:
            continue
        # Mirrors the save-side remap: without it stage 0's biases match every stage's locals.
        key = model.global_parameter_name(name) if is_pp_mode else name
        if key not in saved:
            missing.append(key)
            continue
        value = saved[key]
        # copy_ broadcasts, so a checkpoint from a different expert count would fill the live
        # bias with a single saved entry instead of failing, affecting every routed token.
        if tuple(value.shape) != tuple(bias.shape):
            mismatched.append(f"{key}: saved {tuple(value.shape)} vs live {tuple(bias.shape)}")
            continue
        bias.copy_(value.to(bias.device, bias.dtype))
        restored += 1
    # Consensus'd, not raised where it was found: under PP the drift is stage-local, and a
    # rank-local raise here would leave the peers blocked in the next collective.
    if not rank_consensus(not mismatched).all_ok:
        raise RuntimeError(
            f"{ROUTER_BALANCING_BIASES_FILE} at {checkpoint} holds router balancing biases whose "
            f"shape does not match the live routers — a different expert count or router layout. "
            f"Mismatches on this rank: {mismatched[:5] or ['(reported by another rank)']}"
        )
    # A collective, entered by every rank: under PP a stage may legitimately hold no router, so
    # only a world-wide zero means the sidecar matched nothing.
    any_restored = rank_consensus(restored > 0).any_ok
    if saved and not any_restored and not missing:
        # Every rank logs: the balancing route is resolved per model, and under PP only
        # the router-holding stages could see it at all.
        logger.warning(
            f"[rank {get_global_rank()}] {path} carries {len(saved)} trained router balancing bias(es) but no live "
            "router in the world accepts one, so they are DROPPED. This run's balancing mode resolved to none — "
            "weight-sync RL downgrades both bias modes, since the sync ships parameters only. Families whose bias "
            "exports into a native checkpoint slot keep routing as trained (frozen); side-buffer families "
            "re-balance from zero, and this run's own saves emit no sidecar to resume from.",
        )
    if missing:
        # Every rank logs: under PP the miss can be stage-local, invisible to rank 0.
        logger.warning(
            f"[rank {get_global_rank()}] {len(missing)} router(s) absent from {path} keep zero-init "
            f"balancing biases (checkpoint from a different router layout?): "
            f"{missing[:5]}{' ...' if len(missing) > 5 else ''}",
        )
    if restored and is_global_main_process():
        logger.info(f"✓ Restored router balancing biases for {restored} routers from {path}")
