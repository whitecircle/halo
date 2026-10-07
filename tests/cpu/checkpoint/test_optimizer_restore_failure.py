"""A per-rank optimizer restore that fails on ONE rank must fail the resume on EVERY rank.

The shards passed every topology gate, so they are this run's state: a rank that cannot read its
shard (truncated, an I/O error) or apply it (a CUDA OOM inside ``set_optimizer_state_dict``) must not
turn into a fresh optimizer while the weights, step and LR schedule resume. Two ranks on a real gloo
group, with only rank 1 failing, pin:

1. **Uniform raise.** Rank 0, whose own restore succeeded, raises too, and both messages name the
   failing rank and its error. Rank 0 cannot learn the failure from a single-process stub. A rank
   whose apply failed has dropped what the attempt left behind (moments, zero gradients, ``lr=0``)
   by then, which is what lets a rank that ran out of memory enter the verdict's gather.
2. **The opt-in.** ``allow_optimizer_warm_restart`` keeps the warm restart instead: no rank raises,
   every rank drops its optimizer state (rank 0's restored moments included), the LR schedule still
   resumes, and rank 0 warns with the failing rank and error.
3. **No optimizer state saved** (``save_only_model``) is not a failure: the resume proceeds with a
   fresh optimizer and a warning, without the opt-in.
4. **An interrupted save is.** Shards whose meta never landed, and the base save's rank-0
   ``optimizer.pt`` with no shards beside it, are the two halves a kill mid-save leaves: every rank
   refuses them, naming the opt-in, rather than warm-restarting over them as if they were complete.

A single process takes both outcomes too, and the trainer hands the opt-in to the store it builds.

    python tests/cpu/checkpoint/test_optimizer_restore_failure.py
"""

import datetime
import json
import logging
import os
import re
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.nn as nn

import src.distributed.checkpoint.optimizer as optimizer_mod
from src.distributed.checkpoint.optimizer import OptimizerShardStore
from src.distributed.runtime import fs_aware_makedirs
from src.trainers.mixins.checkpointing import CheckpointingMixin
from tests.common.checkpoint_io import (
    OPTIMIZER_SHARD_FILE,
    SHARD_ROUND_TRIP_LR,
    shard_round_trip_model,
    shard_round_trip_optimizer,
    shard_store_context,
    step_on_seeded_data,
)
from tests.common.gloo import run_gloo_ranks
from tests.common.parallelism import make_parallelism_config

WORLD_SIZE = 2
FAILING_RANK = 1
# Spelled literally: importing the writer's constants would make a rename invisible here.
META_FILE = "optimizer_meta.pt"
BASE_OPTIMIZER_FILE = "optimizer.pt"
SEED = 20261003
TRAIN_STEPS = 3
OOM_MESSAGE = "CUDA out of memory. Tried to allocate 20.00 MiB"
# A rank left in a collective its peer never entered fails at this bound instead of stalling the suite.
PG_TIMEOUT = datetime.timedelta(seconds=60)


def _schedule(optimizer: torch.optim.Optimizer) -> torch.optim.lr_scheduler.LRScheduler:
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 0.5**step)


def _fresh():
    model = shard_round_trip_model(SEED)
    optimizer = shard_round_trip_optimizer(model)
    return model, optimizer, _schedule(optimizer)


def _store(model, optimizer, scheduler, rank: int, world_size: int, allow: bool = False) -> OptimizerShardStore:
    config = make_parallelism_config(world_size=world_size, gpus_per_node=world_size, rank=rank)
    return OptimizerShardStore(
        shard_store_context(model, optimizer, config, lr_scheduler=scheduler, allow_optimizer_warm_restart=allow)
    )


def _save(checkpoint: str, rank: int, world_size: int) -> torch.optim.lr_scheduler.LRScheduler:
    """Save this rank's own moments; returns the schedule, ``TRAIN_STEPS`` in, for the caller to keep."""
    model, optimizer, scheduler = _fresh()
    step_on_seeded_data(model, optimizer, SEED + 1 + rank, TRAIN_STEPS)
    for _ in range(TRAIN_STEPS):
        scheduler.step()
    _store(model, optimizer, scheduler, rank, world_size).save(checkpoint)
    return scheduler


def _truncate_shard(checkpoint: str, rank: int) -> None:
    with open(os.path.join(checkpoint, OPTIMIZER_SHARD_FILE.format(rank=rank)), "wb") as fh:
        fh.write(b"truncated mid-save")


def _fail_like_the_init_step(model: nn.Module, optimizer: torch.optim.Optimizer) -> None:
    """What torch's ``_init_optim_state`` leaves behind when its zero-LR step runs out of memory."""
    for group in optimizer.param_groups:
        group["lr"] = 0.0
    for param in model.parameters():
        param.grad = torch.zeros_like(param)
    raise torch.OutOfMemoryError(OOM_MESSAGE)


class _WarningRecorder(logging.Handler):
    def __init__(self):
        super().__init__(level=logging.WARNING)
        self.messages: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.messages.append(record.getMessage())


def _worker(rank: int, root: str, scenario: str) -> None:
    """Save a checkpoint (unless the scenario saved none), then resume it into a fresh optimizer."""
    outcome = {"raised": None, "applied_locally": False}
    warnings = _WarningRecorder()
    logging.getLogger(optimizer_mod.__name__).addHandler(warnings)
    try:
        checkpoint = os.path.join(root, f"checkpoint-{TRAIN_STEPS}")
        fs_aware_makedirs(checkpoint)
        if scenario not in ("no_optimizer_state", "base_optimizer_only"):
            scheduler = _save(checkpoint, rank, WORLD_SIZE)
            if rank == 0:
                torch.save(scheduler.state_dict(), os.path.join(checkpoint, "scheduler.pt"))
            if scenario == "unreadable_shard" and rank == FAILING_RANK:
                _truncate_shard(checkpoint, rank)
        dist.barrier()
        if rank == 0 and scenario.startswith("torn_meta"):
            os.remove(os.path.join(checkpoint, META_FILE))
        if rank == 0 and scenario == "base_optimizer_only":
            model, optimizer, _ = _fresh()
            step_on_seeded_data(model, optimizer, SEED, TRAIN_STEPS)
            torch.save(optimizer.state_dict(), os.path.join(checkpoint, BASE_OPTIMIZER_FILE))
        dist.barrier()

        real_apply = optimizer_mod.set_optimizer_state_dict

        def apply(model, optimizer, **kwargs):
            real_apply(model, optimizer, **kwargs)
            outcome["applied_locally"] = True
            if rank == FAILING_RANK and scenario.startswith("apply_failure"):
                _fail_like_the_init_step(model, optimizer)

        optimizer_mod.set_optimizer_state_dict = apply

        model, optimizer, scheduler = _fresh()
        store = _store(model, optimizer, scheduler, rank, WORLD_SIZE, allow=scenario.endswith("_opt_in"))
        try:
            store.load(checkpoint)
        except Exception as e:
            outcome["raised"] = f"{type(e).__name__}: {e}"
        outcome["state_entries"] = len(optimizer.state)
        outcome["grads_left"] = sum(param.grad is not None for param in model.parameters())
        outcome["lr"] = optimizer.param_groups[0]["lr"]
        outcome["scheduler_epoch"] = scheduler.last_epoch
    except Exception as e:  # the setup failing must still leave a verdict file for the assertions
        outcome["raised"] = f"setup {type(e).__name__}: {e}"
    outcome["warnings"] = warnings.messages
    with open(os.path.join(root, f"result_{rank}.json"), "w") as fh:
        json.dump(outcome, fh)


def _run(tmp_path, scenario: str) -> list[dict]:
    run_gloo_ranks(_worker, WORLD_SIZE, str(tmp_path), scenario, pg_timeout=PG_TIMEOUT)
    return [json.loads((tmp_path / f"result_{rank}.json").read_text()) for rank in range(WORLD_SIZE)]


def _assert_every_rank_raised(results: list[dict], what: str, error: str) -> None:
    for rank, result in enumerate(results):
        raised = result["raised"]
        assert raised is not None, f"rank {rank} resumed with a fresh optimizer over a restorable checkpoint"
        assert raised.startswith("RuntimeError"), f"rank {rank}: {raised}"
        assert "timed out" not in raised.lower(), f"rank {rank} only saw a collective timeout: {raised}"
        assert f"{what} " in raised, f"rank {rank} did not name the failed step: {raised}"
        assert f"1 of {WORLD_SIZE} rank(s) [{FAILING_RANK}]" in raised, f"rank {rank} did not name the failing rank"
        assert error in raised, f"rank {rank} did not carry the error: {raised}"
        assert "allow_optimizer_warm_restart" in raised, f"rank {rank} did not name the opt-in: {raised}"


def test_an_apply_failure_on_one_rank_raises_on_every_rank(tmp_path):
    results = _run(tmp_path, "apply_failure")

    assert results[0]["applied_locally"], "premise: rank 0's own restore must succeed, so only the peer failed"
    _assert_every_rank_raised(results, "Restoring the per-rank optimizer state", f"OutOfMemoryError: {OOM_MESSAGE}")
    failing = results[FAILING_RANK]
    assert failing["state_entries"] == 0, "the failing rank kept the failed attempt's moments through the gather"
    assert failing["grads_left"] == 0, "the failing rank kept the init step's zero gradients through the gather"
    assert failing["lr"] == SHARD_ROUND_TRIP_LR, f"the failing rank was left at the init step's lr={failing['lr']}"


def test_an_unreadable_shard_on_one_rank_raises_on_every_rank(tmp_path):
    results = _run(tmp_path, "unreadable_shard")

    _assert_every_rank_raised(
        results, "Reading the per-rank optimizer shards", OPTIMIZER_SHARD_FILE.format(rank=FAILING_RANK)
    )
    assert not any(result["applied_locally"] for result in results), "a rank applied its shard past a failed read"


def test_the_opt_in_warm_restarts_every_rank_with_a_warning(tmp_path):
    results = _run(tmp_path, "apply_failure_opt_in")

    for rank, result in enumerate(results):
        assert result["raised"] is None, f"rank {rank} raised under the opt-in: {result['raised']}"
        assert result["applied_locally"], f"premise: rank {rank} must have applied its shard before the verdict"
        assert result["state_entries"] == 0, (
            f"rank {rank} kept {result['state_entries']} restored optimizer entries while a peer reinitialized"
        )
        assert result["scheduler_epoch"] == TRAIN_STEPS, f"rank {rank} did not resume the LR schedule"
        assert result["lr"] == SHARD_ROUND_TRIP_LR * 0.5**TRAIN_STEPS, f"rank {rank} steps at lr={result['lr']}"
        assert result["grads_left"] == 0, f"rank {rank} kept the init step's zero gradients"
    warning = "\n".join(results[0]["warnings"])
    assert f"rank(s) [{FAILING_RANK}]" in warning, warning
    assert f"OutOfMemoryError: {OOM_MESSAGE}" in warning, warning
    assert "Warm restart (allow_optimizer_warm_restart)" in warning, warning


def test_a_checkpoint_without_optimizer_state_resumes_without_the_opt_in(tmp_path):
    results = _run(tmp_path, "no_optimizer_state")

    for rank, result in enumerate(results):
        assert result["raised"] is None, f"rank {rank} refused a checkpoint saved without optimizer state"
        assert not result["applied_locally"], f"rank {rank} restored a shard that was never written"
        assert result["state_entries"] == 0, f"rank {rank}"
    assert "No optimizer shards" in "\n".join(results[0]["warnings"]), results[0]["warnings"]


def _assert_every_rank_refused_the_torn_save(results: list[dict], names: str) -> None:
    for rank, result in enumerate(results):
        raised = result["raised"]
        assert raised is not None, f"rank {rank} resumed over an interrupted save"
        assert raised.startswith("RuntimeError"), f"rank {rank}: {raised}"
        assert names in raised, f"rank {rank} did not name the torn half: {raised}"
        assert "allow_optimizer_warm_restart" in raised, f"rank {rank} did not name the opt-in: {raised}"
        assert not result["applied_locally"], f"rank {rank} applied a shard of a torn set"
        assert result["state_entries"] == 0, f"rank {rank}"


def test_shards_whose_meta_never_landed_are_refused_on_every_rank(tmp_path):
    _assert_every_rank_refused_the_torn_save(_run(tmp_path, "torn_meta"), f"{META_FILE} is missing")


def test_the_base_optimizer_without_its_shards_is_refused_on_every_rank(tmp_path):
    """Killed after the base save's ``trainer_state.json`` but before the per-rank shards: what is
    left is rank 0's replicated view, which the mixin deletes only once its shards are on disk."""
    _assert_every_rank_refused_the_torn_save(_run(tmp_path, "base_optimizer_only"), BASE_OPTIMIZER_FILE)


def test_the_opt_in_warm_restarts_over_a_torn_shard_set(tmp_path):
    results = _run(tmp_path, "torn_meta_opt_in")

    for rank, result in enumerate(results):
        assert result["raised"] is None, f"rank {rank} raised under the opt-in: {result['raised']}"
        assert not result["applied_locally"], f"rank {rank} applied a shard of a torn set"
        assert result["scheduler_epoch"] == TRAIN_STEPS, f"rank {rank} did not resume the LR schedule"
    assert "Warm restart (allow_optimizer_warm_restart)" in "\n".join(results[0]["warnings"])


@pytest.mark.parametrize("failure", ["unreadable_shard", "apply_failure"])
@pytest.mark.parametrize("allow", [False, True], ids=["default", "opted-in"])
def test_a_single_process_restore_failure(tmp_path, monkeypatch, failure, allow):
    """The verdict's non-distributed branch, for both failures and both outcomes.

    No ``scheduler.pt`` here, so nothing re-applies a learning rate after the warm restart: the run's
    own must survive the failed apply, which torch's init step left at 0.
    """
    _save(str(tmp_path), rank=0, world_size=1)
    if failure == "unreadable_shard":
        _truncate_shard(str(tmp_path), rank=0)
    else:
        monkeypatch.setattr(
            optimizer_mod,
            "set_optimizer_state_dict",
            lambda model, optimizer, **kwargs: _fail_like_the_init_step(model, optimizer),
        )
    model, optimizer, scheduler = _fresh()
    store = _store(model, optimizer, scheduler, rank=0, world_size=1, allow=allow)

    if not allow:
        with pytest.raises(RuntimeError, match=r"failed on 1 of 1 rank\(s\) \[0\]"):
            store.load(str(tmp_path))
        return
    store.load(str(tmp_path))
    assert not optimizer.state, "the warm restart kept optimizer state"
    assert all(param.grad is None for param in model.parameters()), "the warm restart kept the init step's gradients"
    assert optimizer.param_groups[0]["lr"] == SHARD_ROUND_TRIP_LR, "the warm restart left the init step's lr=0"


def _rebuild_one_group_too_many(optimizer: torch.optim.Optimizer) -> None:
    """What a shard saved under another param-group layout leaves after ``set_optimizer_state_dict``."""
    optimizer.param_groups.append({**optimizer.param_groups[0], "params": []})


def test_a_drifted_group_count_never_masks_the_restore_error(tmp_path, monkeypatch):
    """The group-settings cleanup runs while the apply's own error unwinds: a group count it cannot
    pair up must not replace that error, which is the only diagnosis every rank receives."""
    _save(str(tmp_path), rank=0, world_size=1)

    def drift_then_fail(model, optimizer, **kwargs):
        _rebuild_one_group_too_many(optimizer)
        _fail_like_the_init_step(model, optimizer)

    monkeypatch.setattr(optimizer_mod, "set_optimizer_state_dict", drift_then_fail)
    model, optimizer, scheduler = _fresh()
    store = _store(model, optimizer, scheduler, rank=0, world_size=1)

    with pytest.raises(RuntimeError, match=f"OutOfMemoryError: {re.escape(OOM_MESSAGE)}"):
        store.load(str(tmp_path))
    assert optimizer.param_groups[0]["lr"] == SHARD_ROUND_TRIP_LR, "the run's group settings were not restored"


def test_a_restore_that_rebuilt_another_group_count_fails_loud(tmp_path, monkeypatch):
    """An unmatched group would step on the checkpoint's hyperparameters, so a clean apply that
    drifted the group count still fails the resume."""
    _save(str(tmp_path), rank=0, world_size=1)
    real_apply = optimizer_mod.set_optimizer_state_dict

    def apply_then_drift(model, optimizer, **kwargs):
        real_apply(model, optimizer, **kwargs)
        _rebuild_one_group_too_many(optimizer)

    monkeypatch.setattr(optimizer_mod, "set_optimizer_state_dict", apply_then_drift)
    model, optimizer, scheduler = _fresh()
    store = _store(model, optimizer, scheduler, rank=0, world_size=1)

    with pytest.raises(RuntimeError, match=r"holds 2 param group\(s\) but this run built 1"):
        store.load(str(tmp_path))


class _TrainerBase:
    """The two base-Trainer methods the load context binds through ``super()``."""

    def _load_from_checkpoint(self, resume_from_checkpoint, model=None) -> None:
        raise AssertionError("not reached")

    def _load_optimizer_and_scheduler(self, checkpoint) -> None:
        raise AssertionError("not reached")


class _CheckpointingHost(CheckpointingMixin, _TrainerBase):
    def __init__(self, args):
        self.args = args
        self.model = nn.Linear(2, 2)
        self.optimizer = None
        self.lr_scheduler = None
        self.parallelism_config = make_parallelism_config(world_size=1, gpus_per_node=1)
        self._has_ep_layers = False
        self._fsdp_wrapped = True

    def _get_tp_rank(self) -> int:
        return 0


@pytest.mark.parametrize(
    ("args", "expected"),
    [
        (SimpleNamespace(allow_optimizer_warm_restart=True), True),
        (SimpleNamespace(allow_optimizer_warm_restart=False), False),
        (SimpleNamespace(), False),
    ],
    ids=["opted-in", "default", "config-built-outside-the-entry-scripts"],
)
def test_the_trainer_hands_the_opt_in_to_the_optimizer_store(args, expected):
    """The entry scripts forward the knob onto the training config; the store reads it only off the
    context the trainer builds, so a context that drops it makes the opt-in a silent no-op."""
    assert _CheckpointingHost(args)._optimizer_store().ctx.allow_optimizer_warm_restart is expected


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
