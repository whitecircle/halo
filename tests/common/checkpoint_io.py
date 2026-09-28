"""Readers for the artifacts the toolkit's checkpoint writers produce, and the probes round-trip tests
compare them by.

Key-set assertions go through the production reader (``load_full_state_dict``): it accepts both
layouts the writers pick between by size (index+shards vs a bare ``model.safetensors``) and refuses a
per-rank EP/TP index, which a test-local reader would accept as a whole checkpoint. A stock
``from_pretrained`` load of a checkpoint is clean when :func:`loading_problems` reports nothing.

A two-phase resume compares the state at the save with the state the resume restored
(:class:`RestorePointSnapshot`).

A save→reload or save→resume test proves the weights survived by value: the same fixed batch's forward
loss before the save and after the reload (:func:`fixed_batch_loss`). A resume test also reads the
optimizer and scheduler state the checkpoint restored, snapshotted before the first resumed step
(:class:`ResumeCapture`) and graded by :func:`resume_continuity_checks`. The file probes (:func:`model_save_checks`, :func:`resume_checkpoint_checks`)
name what a save must have left on disk before any of that is read.
"""

import math
import os

import torch
from transformers import TrainerCallback
from transformers.trainer import OPTIMIZER_NAME, SCHEDULER_NAME, TRAINER_STATE_NAME
from transformers.utils import CONFIG_NAME

from src.checkpoint.format import SAFETENSORS_INDEX_FILE, has_whole_model_weight_file, load_full_state_dict
from src.distributed.fsdp import reshard_fsdp2_modules
from tests.common.peft_helpers import snapshot_adapters, unwrap
from tests.common.utils import local_optimizer_state, log, optimizer_state_matches

# The key lists ``from_pretrained``'s loading info reports; a clean load leaves every one empty.
LOADING_INFO_KINDS = ("missing_keys", "unexpected_keys", "mismatched_keys")
# Truncation bound for a :func:`fixed_text_batch` sequence; the short probe texts stay under it.
FIXED_TEXT_BATCH_MAX_TOKENS = 64
# The per-rank optimizer shards and the fingerprint meta a sharded save writes, spelled independently of
# the writer so a rename fails here instead of agreeing with itself.
OPTIMIZER_SHARD_FILE = "optimizer_shard_{rank:05d}.pt"
OPTIMIZER_META_FILE = "optimizer_meta.pt"
# The fixed-batch probe of the TP resume suites, scored before the save and after the resume.
TP_RESUME_PROBE_TEXT = (
    "User: What is 17 plus 25?\nAssistant: The answer is 42. "
    "The TP gather and re-shard must survive a checkpoint save and resume intact."
)


def written_keys(output_dir: str) -> set[str]:
    """Every tensor key present in the gathered checkpoint at ``output_dir`` (see module docstring)."""
    state = load_full_state_dict(output_dir)
    assert state is not None, f"no checkpoint at {output_dir}"
    return set(state)


def loading_problems(info: dict) -> dict[str, list]:
    """The non-empty key lists of a ``from_pretrained(..., output_loading_info=True)`` load: missing,
    unexpected and mismatched. Indexed rather than ``get``, so a kind transformers stops reporting
    raises instead of reading as clean."""
    return {kind: info[kind] for kind in LOADING_INFO_KINDS if info[kind]}


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


def model_save_checks(save_dir: str, rank: int) -> dict[str, bool]:
    """Named checks that a ``save_model`` directory holds a loadable whole model: ``config.json`` and the
    weights.

    The full-model counterpart of :func:`tests.common.peft_helpers.adapter_save_checks`. Only rank 0
    reads the disk; the other ranks return no checks.
    """
    if rank != 0:
        return {}
    exists = os.path.isdir(save_dir)
    checks = {
        "save_dir_exists": exists,
        "has_config": exists and os.path.isfile(os.path.join(save_dir, CONFIG_NAME)),
        "has_model_weights": exists and has_whole_model_weight_file(save_dir),
    }
    log(f"  Save at {save_dir}: {sorted(os.listdir(save_dir)) if exists else 'missing'}")
    for name, ok in checks.items():
        log(f"  {name}: {'PASS' if ok else 'FAIL'}")
    return checks


def resume_checkpoint_checks(checkpoint_dir: str, world_size: int) -> dict[str, bool]:
    """Named checks that a mid-training checkpoint holds every file a full resume reads back.

    Every run writes the trainer state, the LR scheduler and whole-model weights. A sharded run
    (``world_size > 1``) adds one optimizer shard and one RNG state per rank plus the fingerprint meta,
    whose ``num_ranks`` must be the world that wrote it; a single process writes HF's ``optimizer.pt``
    and one ``rng_state.pth``. Reads the disk only, so rank 0 calls it and shares the verdict
    (``ctx.broadcast_checks``).
    """
    files = set(os.listdir(checkpoint_dir)) if os.path.isdir(checkpoint_dir) else set()
    if world_size > 1:
        per_run = [OPTIMIZER_META_FILE]
        per_run += [OPTIMIZER_SHARD_FILE.format(rank=rank) for rank in range(world_size)]
        per_run += [f"rng_state_{rank}.pth" for rank in range(world_size)]
    else:
        per_run = [OPTIMIZER_NAME, "rng_state.pth"]
    checks = {name: name in files for name in (TRAINER_STATE_NAME, SCHEDULER_NAME, *per_run)}
    checks["model_weights"] = has_whole_model_weight_file(checkpoint_dir)
    if checks.get(OPTIMIZER_META_FILE):
        meta = torch.load(os.path.join(checkpoint_dir, OPTIMIZER_META_FILE), map_location="cpu", weights_only=False)
        checks["optimizer_meta_num_ranks"] = meta.get("num_ranks") == world_size
    missing = [name for name, ok in checks.items() if not ok]
    log(f"  Checkpoint {checkpoint_dir}: {sorted(files)}" + (f"; FAILED: {missing}" if missing else ""))
    return checks


def fixed_text_batch(tokenizer, device, text: str) -> tuple[torch.Tensor, torch.Tensor]:
    """``text`` as a one-sequence ``(input_ids, labels)`` batch for :func:`fixed_batch_loss`, every token
    scored.

    Tokenized identically on every call, so the loss before a save and after the reload or resume
    compares the same tokens.
    """
    encoded = tokenizer(text, return_tensors="pt", truncation=True, max_length=FIXED_TEXT_BATCH_MAX_TOKENS)
    input_ids = encoded["input_ids"].to(device)
    return input_ids, input_ids.clone()


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


class RestorePointSnapshot(TrainerCallback):
    """Trainer/optimizer/scheduler state at one lifecycle point.

    ``"save"`` fires when the checkpoint is written, ``"train_begin"`` after the resume restore and
    before the first resumed step, so the two snapshots describe the same step and a warm-restarted
    optimizer cannot hide behind the steps that follow.

    The first occurrence wins: a run that stops on ``max_steps`` writes a final checkpoint too, and a
    snapshot overwritten there would describe a step the resume never restores.

    ``capture_optimizer`` is off for a full fine-tune of a large policy: ``local_optimizer_state``
    offloads the whole state to host RAM, which is 6 B/param of AdamWBF16 moments. ``expert_lora``
    is the adapter-gather flag (``None`` = do not capture adapters); the capture has to happen here
    because a resumed run takes a step of its own before the body can look. A subclass adds its own
    entries through :meth:`extra`.
    """

    def __init__(self, event: str, trainer, *, capture_optimizer: bool, expert_lora: bool | None = None):
        self.event = event
        self.trainer = trainer
        self.capture_optimizer = capture_optimizer
        self.expert_lora = expert_lora
        self.captured: dict | None = None

    def extra(self) -> dict:
        """Extra entries for the snapshot, taken at the same point. Empty here."""
        return {}

    def _capture(self, state) -> None:
        if self.captured is not None:
            return
        # Every reader below goes by parameter identity, and a hook can fire while the FSDP2 modules
        # still hold the transient unsharded params an eval-only forward left registered.
        reshard_fsdp2_modules(unwrap(self.trainer.model))
        self.captured = {
            "global_step": state.global_step,
            "sched_last_epoch": self.trainer.lr_scheduler.last_epoch,
            "optimizer": local_optimizer_state(self.trainer.model, self.trainer.optimizer)
            if self.capture_optimizer
            else None,
            "adapters": snapshot_adapters(unwrap(self.trainer.model), expert_lora=self.expert_lora)
            if self.expert_lora is not None
            else None,
            **self.extra(),
        }

    def on_save(self, args, state, control, **kwargs):
        if self.event == "save":
            self._capture(state)

    def on_train_begin(self, args, state, control, **kwargs):
        if self.event == "train_begin":
            self._capture(state)


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


def resume_continuity_checks(
    capture: dict | None,
    l_pre: float,
    *,
    save_step: int,
    loss_tol: float,
    optimizer_state: dict | None = None,
) -> dict[str, bool]:
    """Named checks on what a :class:`ResumeCapture` read at the first resumed step.

    ``resume_capture_fired``; ``resume_weights_restored`` (the fixed-batch loss within ``loss_tol`` of
    ``l_pre``, the same batch's loss before the save); ``resume_optimizer_moments_restored`` (Adam's
    second moments present, not all zero, finite); ``resume_scheduler_restored`` (the LR scheduler at
    ``save_step``, not back at warmup); and, given the pre-save ``optimizer_state`` (the capture must
    have been built with ``optimizer_state=True``), ``resume_optimizer_state_bit_exact``.
    """
    checks = {"resume_capture_fired": capture is not None}
    if capture is None:
        log("  ERROR: the resume capture never fired (on_train_begin missed)")
        return checks
    l_post = capture["l_post"]
    delta = abs(l_post - l_pre)
    checks["resume_weights_restored"] = math.isfinite(l_post) and delta < loss_tol
    checks["resume_optimizer_moments_restored"] = (
        capture["moments_materialized"] and capture["moments_nonzero"] and capture["moments_finite"]
    )
    checks["resume_scheduler_restored"] = capture["sched_last_epoch"] == save_step
    log(f"  L_pre={l_pre:.6f}  L_post={l_post:.6f}  |delta|={delta:.6f} (tol {loss_tol})")
    log(
        f"  exp_avg_sq materialized={capture['moments_materialized']} nonzero={capture['moments_nonzero']} "
        f"finite={capture['moments_finite']}; scheduler last_epoch={capture['sched_last_epoch']} "
        f"(expected {save_step})"
    )
    if optimizer_state is not None:
        bit_exact, why = optimizer_state_matches(optimizer_state, capture["optimizer_state"])
        checks["resume_optimizer_state_bit_exact"] = bit_exact
        if not bit_exact:
            log(f"  ERROR: optimizer state not bit-exact after resume: {why}")
    return checks
