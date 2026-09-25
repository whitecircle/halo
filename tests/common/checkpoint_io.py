"""Readers for the artifacts the toolkit's checkpoint writers produce, and the probes round-trip tests
compare them by.

Key-set assertions go through the production reader (``load_full_state_dict``): it accepts both
layouts the writers pick between by size (index+shards vs a bare ``model.safetensors``) and refuses a
per-rank EP/TP index, which a test-local reader would accept as a whole checkpoint.

A save→reload or save→resume test proves the weights survived by value: the same fixed batch's forward
loss before the save and after the reload (:func:`fixed_batch_loss`). A resume test also reads the
optimizer and scheduler state the checkpoint restored, snapshotted before the first resumed step
(:class:`ResumeCapture`).
"""

import os

import torch
from transformers import TrainerCallback

from src.checkpoint.format import SAFETENSORS_INDEX_FILE, load_full_state_dict
from tests.common.utils import local_optimizer_state


def written_keys(output_dir: str) -> set[str]:
    """Every tensor key present in the gathered checkpoint at ``output_dir`` (see module docstring)."""
    state = load_full_state_dict(output_dir)
    assert state is not None, f"no checkpoint at {output_dir}"
    return set(state)


def weight_files(output_dir: str, *, include_index: bool = False) -> list[str]:
    """Sorted ``model*.safetensors`` basenames a save left in ``output_dir``.

    ``include_index`` makes the index policy explicit, so the same helper answers the same question
    for every caller.
    """
    names = sorted(
        name for name in os.listdir(output_dir) if name.startswith("model") and name.endswith(".safetensors")
    )
    if include_index:
        index = os.path.join(output_dir, SAFETENSORS_INDEX_FILE)
        if os.path.isfile(index):
            names.append(SAFETENSORS_INDEX_FILE)
    return names


def fixed_batch_loss(model, input_ids: torch.Tensor, labels: torch.Tensor) -> float:
    """``model``'s causal-LM loss on a fixed batch: eval mode, no grad, no KV cache, no optimizer step.

    The loss reflects only the current weights. The train/eval mode is restored, so a probe taken
    between training steps leaves the run as it was.
    """
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            return model(input_ids=input_ids, labels=labels, use_cache=False).loss.item()
    finally:
        model.train(was_training)


def optimizer_moments_stats(optimizer) -> tuple[bool, bool, bool]:
    """``(materialized, any_nonzero, all_finite)`` over this rank's Adam second moments (``exp_avg_sq``).

    ``materialized`` is whether any parameter carries one at all, so a reinitialized optimizer reads
    ``(False, False, True)``. A DTensor moment is read through ``to_local()``, this rank's shard, so the
    scan runs no collective and each rank can call it alone.
    """
    materialized = any_nonzero = False
    all_finite = True
    for state in optimizer.state.values():
        sq = state.get("exp_avg_sq")
        if sq is None:
            continue
        materialized = True
        local = (sq.to_local() if hasattr(sq, "to_local") else sq).detach()
        if (local != 0).any().item():
            any_nonzero = True
        if not torch.isfinite(local).all().item():
            all_finite = False
    return materialized, any_nonzero, all_finite


class ResumeCapture(TrainerCallback):
    """Snapshot of the state a resume restored, taken at ``on_train_begin``.

    That hook fires after the checkpoint's model, optimizer and scheduler state are loaded and before
    the first resumed step, so the snapshot is what the checkpoint carried rather than a stepped state.
    ``capture`` stays ``None`` if the hook never fired, and otherwise holds:

    * ``l_post``: :func:`fixed_batch_loss` on ``ids``/``labels``;
    * ``moments_materialized`` / ``moments_nonzero`` / ``moments_finite``: :func:`optimizer_moments_stats`;
    * ``sched_last_epoch``: the LR scheduler's ``last_epoch``, ``None`` without a scheduler;
    * ``optimizer_state``, only with ``optimizer_state=True``: this rank's
      :func:`~tests.common.utils.local_optimizer_state`, for a bit-exact compare. Opt-in because it
      copies the rank's whole optimizer state to host memory.
    """

    def __init__(self, trainer, ids: torch.Tensor, labels: torch.Tensor, *, optimizer_state: bool = False):
        self.trainer = trainer
        self.ids = ids
        self.labels = labels
        self.optimizer_state = optimizer_state
        self.capture: dict | None = None

    def on_train_begin(self, args, state, control, **kwargs):
        trainer = self.trainer
        materialized, nonzero, finite = optimizer_moments_stats(trainer.optimizer)
        scheduler = trainer.lr_scheduler
        self.capture = {
            "l_post": fixed_batch_loss(trainer.model, self.ids, self.labels),
            "moments_materialized": materialized,
            "moments_nonzero": nonzero,
            "moments_finite": finite,
            "sched_last_epoch": int(scheduler.last_epoch) if scheduler is not None else None,
        }
        if self.optimizer_state:
            self.capture["optimizer_state"] = local_optimizer_state(trainer.model, trainer.optimizer)
        return control
