"""Readers for checkpoint artifacts and the probes save, reload and resume tests compare them by."""

import hashlib
import json
import math
import os
from collections.abc import Callable
from pathlib import Path

import torch
import torch.nn as nn
from safetensors import safe_open
from transformers import TrainerCallback
from transformers.trainer import OPTIMIZER_NAME, SCHEDULER_NAME, TRAINER_STATE_NAME
from transformers.utils import CONFIG_NAME

from halo.checkpoint.format import (
    DEFAULT_MAX_SHARD_SIZE,
    RESUME_ADAPTER_DIR,
    RESUME_ADAPTER_MARKER_FILE,
    SAFETENSORS_INDEX_FILE,
)
from halo.checkpoint.parallel.context import CheckpointContext, CheckpointLoadContext
from halo.checkpoint.reading import has_whole_model_weight_file, load_full_state_dict
from halo.distributed.fsdp import reshard_fsdp2_modules
from halo.runtime.filesystem import is_fs_save_rank
from tests.common.adapters import snapshot_adapters, unwrap
from tests.common.memory import cleanup_memory
from tests.common.optimizer_state import local_optimizer_state, optimizer_state_matches
from tests.common.parallel_shape import parallel_shape_checks
from tests.common.parallelism import make_parallelism_config
from tests.common.rank_log import log
from tests.common.tolerances import TOL
from tests.common.training_checks import training_run_checks

# A clean ``from_pretrained`` load leaves every one empty.
LOADING_INFO_KINDS = ("missing_keys", "unexpected_keys", "mismatched_keys")
# Truncation bound and padded length of a :func:`fixed_text_batch` sequence.
FIXED_TEXT_BATCH_MAX_TOKENS = 64
# Spelled independently of the writer, so a rename fails here instead of agreeing with itself.
OPTIMIZER_SHARD_FILE = "optimizer_shard_{rank:05d}.pt"
OPTIMIZER_META_FILE = "optimizer_meta.pt"
SHARD_ROUND_TRIP_LR = 0.05
RESUME_PROBE_TEXT = (
    "User: What is 17 plus 25?\nAssistant: The answer is 42. "
    "Weights must survive a checkpoint save and resume intact across parallelism modes."
)


def safetensors_state_dict(checkpoint_dir: str | Path) -> dict[str, torch.Tensor]:
    """Every tensor across a checkpoint's ``*.safetensors`` parts, keyed as written; ``from_pretrained`` would re-spell
    and re-cast them."""
    tensors: dict[str, torch.Tensor] = {}
    for part in sorted(Path(checkpoint_dir).glob("*.safetensors")):
        with safe_open(str(part), framework="pt") as reader:
            for key in reader.keys():  # noqa: SIM118 - safe_open has .keys() but is not a mapping
                tensors[key] = reader.get_tensor(key)
    return tensors


def file_digests(directory: str | Path) -> dict[str, str]:
    """sha256 of every file in ``directory``, by name: a byte-level fingerprint of a save, index and config
    included."""
    digests = {}
    for path in sorted(Path(directory).iterdir()):
        with open(path, "rb") as handle:
            digests[path.name] = hashlib.file_digest(handle, "sha256").hexdigest()
    return digests


def written_keys(output_dir: str) -> set[str]:
    """Every tensor key in the gathered checkpoint, via the production reader, which refuses a per-rank EP/TP index."""
    state = load_full_state_dict(output_dir)
    assert state is not None, f"no checkpoint at {output_dir}"
    return set(state)


def mark_resume_adapter(checkpoint_dir: str) -> None:
    """Mark ``checkpoint_dir`` as a merged save whose resume adapter is complete, as the save's last write does."""
    with open(os.path.join(checkpoint_dir, RESUME_ADAPTER_MARKER_FILE), "w") as fh:
        json.dump({"adapter_dir": RESUME_ADAPTER_DIR}, fh)


def loading_problems(info: dict) -> dict[str, list]:
    """The non-empty key lists of a loading info. Indexed rather than ``get``, so a kind transformers stops reporting
    raises."""
    return {kind: info[kind] for kind in LOADING_INFO_KINDS if info[kind]}


def weight_files(output_dir: str, *, include_index: bool = False) -> list[str]:
    """Sorted ``model*.safetensors`` basenames a save left in ``output_dir``."""
    names = sorted(
        name for name in os.listdir(output_dir) if name.startswith("model") and name.endswith(".safetensors")
    )
    if include_index:
        index = os.path.join(output_dir, SAFETENSORS_INDEX_FILE)
        if os.path.isfile(index):
            names.append(SAFETENSORS_INDEX_FILE)
    return names


def model_save_checks(save_dir: str, rank: int) -> dict[str, bool]:
    """Named checks that a ``save_model`` directory holds a loadable whole model. Only rank 0 reads the disk; others
    return no checks."""
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


def save_schedule(save_every: int | None) -> dict:
    return {"save_strategy": "steps", "save_steps": save_every} if save_every else {"save_strategy": "no"}


def resume_checkpoint_checks(checkpoint_dir: str, world_size: int) -> dict[str, bool]:
    """Named checks that a mid-training checkpoint holds every file a full resume reads back. A sharded run adds
    per-rank optimizer shards and RNG plus the meta, whose ``num_ranks`` must match. Disk only: rank 0 calls it and
    shares the verdict."""
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


def fixed_text_batch(tokenizer, device, text: str, *, pad: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
    """``text`` as a one-sequence ``(input_ids, labels)`` batch, every token scored; ``pad`` pads to an even length CP
    can split, unscored."""
    encoded = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=FIXED_TEXT_BATCH_MAX_TOKENS,
        padding="max_length" if pad else False,
    )
    input_ids = encoded["input_ids"].to(device)
    labels = input_ids.clone()
    if pad:
        labels[input_ids == tokenizer.pad_token_id] = -100
    return input_ids, labels


def fixed_batch_loss(model, input_ids: torch.Tensor, labels: torch.Tensor) -> float:
    """``model``'s causal-LM loss on a fixed batch: eval mode, no grad, no KV cache, no optimizer step. Refreshes
    FSDP2's params from the shards first and restores sharded registration after, so no unstepped copy is left for the
    next reader."""
    reshard_fsdp2_modules(model)
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            return model(input_ids=input_ids, labels=labels, use_cache=False).loss.item()
    finally:
        reshard_fsdp2_modules(model)
        model.train(was_training)


def optimizer_moments_stats(optimizer) -> tuple[bool, bool, bool]:
    """``(materialized, any_nonzero, all_finite)`` over this rank's Adam ``exp_avg_sq``; a fresh optimizer reads
    ``(False, False, True)``. No collective."""
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


def shard_round_trip_model(seed: int) -> nn.Module:
    """Identical on every rank for one seed; only the data, and so the optimizer moments, diverge."""
    torch.manual_seed(seed)
    return nn.Sequential(nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 4))


def shard_round_trip_optimizer(model: nn.Module) -> torch.optim.Optimizer:
    return torch.optim.AdamW(model.parameters(), lr=SHARD_ROUND_TRIP_LR, betas=(0.9, 0.95))


def step_on_seeded_data(model: nn.Module, optimizer: torch.optim.Optimizer, seed: int, steps: int) -> None:
    """Real steps on seeded random data: non-trivial moments that differ between seeds."""
    generator = torch.Generator().manual_seed(seed)
    for _ in range(steps):
        inputs = torch.randn(4, 8, generator=generator)
        target = torch.randn(4, 4, generator=generator)
        optimizer.zero_grad()
        ((model(inputs) - target) ** 2).mean().backward()
        optimizer.step()


def _base_trainer_fallback(*_args, **_kwargs) -> None:
    raise AssertionError("the base-Trainer optimizer path ran; the sharded per-rank path was expected")


def load_context(
    model: nn.Module,
    parallelism_config=None,
    *,
    tp_size: int = 1,
    optimizer=None,
    lr_scheduler=None,
    has_ep_layers: bool = False,
    fsdp_wrapped: bool = False,
    tp_rank: int = 0,
    super_load_optimizer_and_scheduler=_base_trainer_fallback,
    allow_optimizer_warm_restart: bool = False,
    base_owns_whole_weight_load: bool = False,
) -> CheckpointLoadContext:
    """A resume context over a real ``ParallelismConfig``, by default one ``tp_size``-way TP node. The base-Trainer
    optimizer load raises unless given."""
    return CheckpointLoadContext(
        model=model,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        parallelism_config=parallelism_config
        or make_parallelism_config(world_size=tp_size, gpus_per_node=tp_size, tp_size=tp_size),
        has_ep_layers=has_ep_layers,
        fsdp_wrapped=fsdp_wrapped,
        tp_rank=tp_rank,
        super_load_optimizer_and_scheduler=super_load_optimizer_and_scheduler,
        allow_optimizer_warm_restart=allow_optimizer_warm_restart,
        base_owns_whole_weight_load=base_owns_whole_weight_load,
    )


def shard_store_context(
    model: nn.Module, optimizer, parallelism_config, *, lr_scheduler=None, allow_optimizer_warm_restart=False
) -> CheckpointLoadContext:
    """A load context sending a CPU model through the per-rank shard store; the base-Trainer fallback raises, so a test
    cannot pass unsharded."""
    return load_context(
        model,
        parallelism_config,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        fsdp_wrapped=True,
        allow_optimizer_warm_restart=allow_optimizer_warm_restart,
    )


def save_context(model: nn.Module, parallelism_config=None, **fields) -> CheckpointContext:
    """A saver's context for ``model`` over a real ``ParallelismConfig``, one process unless given."""
    defaults = {
        "model": model,
        "parallelism_config": parallelism_config or make_parallelism_config(world_size=1, gpus_per_node=1),
        "has_ep_layers": False,
        "toolkit_fsdp2": False,
        "accelerate_manages_fsdp": False,
        "is_save_rank": is_fs_save_rank(),
        "max_shard_size": DEFAULT_MAX_SHARD_SIZE,
        "save_sharded_ep": False,
        "has_expert_lora": False,
        "cp_wrapper": None,
        "tokenizer": None,
    }
    return CheckpointContext(**(defaults | fields))


class RestorePointSnapshot(TrainerCallback):
    """Trainer, optimizer and scheduler state at ``"save"`` or ``"train_begin"``, so both snapshots describe the same
    step. The first occurrence wins: a run stopping on ``max_steps`` writes a final checkpoint the resume never
    restores. ``capture_optimizer`` is off for a large full fine-tune, since it offloads 6 B/param of AdamWBF16 moments
    to host RAM."""

    def __init__(self, event: str, trainer, *, capture_optimizer: bool, expert_lora: bool | None = None):
        self.event = event
        self.trainer = trainer
        self.capture_optimizer = capture_optimizer
        self.expert_lora = expert_lora
        self.captured: dict | None = None

    def extra(self) -> dict:
        """Extra entries for the snapshot, taken at the same point."""
        return {}

    def _capture(self, state) -> None:
        if self.captured is not None:
            return
        # A hook can fire while FSDP2 modules hold the transient unsharded params an eval-only forward left registered.
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
    """Snapshot at ``on_train_begin``, after the checkpoint's state loads and before the first resumed step;
    ``capture`` is None if it never fired. ``optimizer_state=True`` adds a bit-exact local copy, opt-in because it
    moves the rank's whole optimizer state to host memory."""

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
    """Named checks on what a :class:`ResumeCapture` read at the first resumed step. The bit-exact check needs a
    pre-save ``optimizer_state`` and a capture built with ``optimizer_state=True``."""
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


def run_two_phase_resume(
    ctx,
    build_trainer: Callable[..., object],
    *,
    output_dir: str,
    probe: tuple[torch.Tensor, torch.Tensor],
    save_at_step: int = 3,
    total_steps: int = 6,
    optimizer_state: bool = False,
) -> dict[str, bool]:
    """COLLECTIVE. Train to ``save_at_step``, resume a fresh trainer to ``total_steps``, and return both phases'
    checks, phase 2's prefixed ``resume_``. ``build_trainer(checkpoint, **schedule)`` builds a phase's trainer;
    ``checkpoint`` is None in phase 1."""
    checkpoint = os.path.join(output_dir, f"checkpoint-{save_at_step}")
    log(f"\n  Phase 1: train {save_at_step} steps, checkpoint at the last")
    trainer = build_trainer(
        None,
        output_dir=output_dir,
        max_steps=save_at_step,
        save_strategy="steps",
        save_steps=save_at_step,
        save_total_limit=1,
    )
    shape_checks = parallel_shape_checks(trainer.model, trainer.parallelism_config)
    checks = training_run_checks(trainer.train(), trainer, save_at_step)
    l_pre = fixed_batch_loss(trainer.model, *probe)
    # Nothing steps the optimizer after the save at the last step, so this is the state its shards hold.
    saved_optimizer = local_optimizer_state(trainer.model, trainer.optimizer) if optimizer_state else None
    ctx.barrier()
    checks |= ctx.broadcast_checks(resume_checkpoint_checks(checkpoint, ctx.world_size) if ctx.rank == 0 else {})
    trainer.cleanup_ep()
    del trainer
    cleanup_memory()
    ctx.barrier()
    # Gated on the rank-uniform checks alone so every rank takes the same branch; the shape checks are each rank's own.
    if not all(checks.values()):
        log("  Phase 1 FAILED, skipping the resume")
        return shape_checks | checks

    log(f"\n  Phase 2: resume from {checkpoint} to step {total_steps}")
    trainer = build_trainer(checkpoint, output_dir=output_dir, max_steps=total_steps, save_strategy="no")
    capture = ResumeCapture(trainer, *probe, optimizer_state=optimizer_state)
    trainer.add_callback(capture)
    resumed = parallel_shape_checks(trainer.model, trainer.parallelism_config)
    resumed |= training_run_checks(trainer.train(resume_from_checkpoint=checkpoint), trainer, total_steps)
    checks |= {f"resume_{name}": ok for name, ok in resumed.items()}
    checks |= resume_continuity_checks(
        capture.capture,
        l_pre,
        save_step=save_at_step,
        loss_tol=TOL.resume_fixed_batch_loss_abs,
        optimizer_state=saved_optimizer,
    )
    trainer.cleanup_ep()
    del trainer, capture
    cleanup_memory()
    ctx.barrier()
    return shape_checks | checks
