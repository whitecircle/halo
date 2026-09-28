#!/usr/bin/env python
"""The checkpoint probes in ``tests.common.checkpoint_io`` fail on the save they exist to catch.

The SFT, SMPO and offline GRPO resume suites and the save smokes gate on these probes and run only on
the GPU tiers, so a probe that stopped reading a file (or a capture field) would go unnoticed. Each is
driven here with a directory or a capture that holds everything, then with one piece taken away.

Run: python tests/cpu/conventions/test_checkpoint_probes.py
"""

from pathlib import Path

import pytest
import torch
from transformers.trainer import OPTIMIZER_NAME, SCHEDULER_NAME, TRAINER_STATE_NAME
from transformers.utils import CONFIG_NAME

from src.checkpoint.format import SAFETENSORS_WEIGHTS_FILE
from tests.common.checkpoint_io import (
    OPTIMIZER_META_FILE,
    OPTIMIZER_SHARD_FILE,
    model_save_checks,
    resume_checkpoint_checks,
    resume_continuity_checks,
)

WORLD_SIZE = 2
SAVE_STEP = 3
L_PRE = 1.5
LOSS_TOL = 1e-2
MOMENT = torch.ones(2)
# The optimizer state a save wrote, as the resume capture must read it back.
SAVED_STATE = {"state": {"w": {"exp_avg_sq": MOMENT, "step": SAVE_STEP}}}


def _sharded_checkpoint(root: Path, *, num_ranks: int = WORLD_SIZE, skip: str | None = None) -> Path:
    """A mid-training checkpoint as a ``WORLD_SIZE``-rank sharded save writes it, minus ``skip``."""
    names = [TRAINER_STATE_NAME, SCHEDULER_NAME, SAFETENSORS_WEIGHTS_FILE]
    names += [OPTIMIZER_SHARD_FILE.format(rank=rank) for rank in range(WORLD_SIZE)]
    names += [f"rng_state_{rank}.pth" for rank in range(WORLD_SIZE)]
    root.mkdir(parents=True, exist_ok=True)
    for name in names:
        if name != skip:
            (root / name).write_bytes(b"")
    if skip != OPTIMIZER_META_FILE:
        torch.save({"num_ranks": num_ranks}, root / OPTIMIZER_META_FILE)
    return root


def _failed(checks: dict[str, bool]) -> list[str]:
    return [name for name, ok in checks.items() if not ok]


def test_a_complete_sharded_checkpoint_passes(tmp_path):
    checks = resume_checkpoint_checks(str(_sharded_checkpoint(tmp_path / "ckpt")), WORLD_SIZE)
    assert not _failed(checks) and "optimizer_meta_num_ranks" in checks


@pytest.mark.parametrize(
    "missing",
    [TRAINER_STATE_NAME, SCHEDULER_NAME, OPTIMIZER_SHARD_FILE.format(rank=1), "rng_state_1.pth", OPTIMIZER_META_FILE],
)
def test_each_missing_file_fails_by_name(tmp_path, missing):
    checks = resume_checkpoint_checks(str(_sharded_checkpoint(tmp_path / "ckpt", skip=missing)), WORLD_SIZE)
    assert _failed(checks) == [missing]


def test_missing_weights_fail(tmp_path):
    checks = resume_checkpoint_checks(
        str(_sharded_checkpoint(tmp_path / "ckpt", skip=SAFETENSORS_WEIGHTS_FILE)), WORLD_SIZE
    )
    assert _failed(checks) == ["model_weights"]


def test_a_meta_written_by_another_world_fails(tmp_path):
    checks = resume_checkpoint_checks(str(_sharded_checkpoint(tmp_path / "ckpt", num_ranks=4)), WORLD_SIZE)
    assert _failed(checks) == ["optimizer_meta_num_ranks"]


def test_a_single_process_checkpoint_needs_the_hf_optimizer_file(tmp_path):
    root = tmp_path / "ckpt"
    root.mkdir()
    for name in (TRAINER_STATE_NAME, SCHEDULER_NAME, SAFETENSORS_WEIGHTS_FILE, "rng_state.pth"):
        (root / name).write_bytes(b"")
    assert _failed(resume_checkpoint_checks(str(root), 1)) == [OPTIMIZER_NAME]


def test_an_absent_checkpoint_fails_every_check(tmp_path):
    checks = resume_checkpoint_checks(str(tmp_path / "never_written"), WORLD_SIZE)
    assert checks and not any(checks.values())


@pytest.mark.parametrize(
    ("missing", "failed"),
    [(None, []), (CONFIG_NAME, ["has_config"]), (SAFETENSORS_WEIGHTS_FILE, ["has_model_weights"])],
)
def test_a_model_save_needs_its_config_and_weights(tmp_path, missing, failed):
    for name in (CONFIG_NAME, SAFETENSORS_WEIGHTS_FILE):
        if name != missing:
            (tmp_path / name).write_bytes(b"")
    assert _failed(model_save_checks(str(tmp_path), rank=0)) == failed


def test_a_model_save_is_read_on_rank_zero_only(tmp_path):
    assert model_save_checks(str(tmp_path / "never_written"), rank=1) == {}


def _capture(**overrides) -> dict:
    return {
        "l_post": L_PRE,
        "moments_materialized": True,
        "moments_nonzero": True,
        "moments_finite": True,
        "sched_last_epoch": SAVE_STEP,
        "optimizer_state": {"state": {"w": {"exp_avg_sq": MOMENT.clone(), "step": SAVE_STEP}}},
        **overrides,
    }


def _continuity(capture, **kwargs) -> dict[str, bool]:
    return resume_continuity_checks(capture, L_PRE, save_step=SAVE_STEP, loss_tol=LOSS_TOL, **kwargs)


def test_a_faithful_resume_passes_every_continuity_check():
    checks = _continuity(_capture(), optimizer_state=SAVED_STATE)
    assert not _failed(checks) and "resume_optimizer_state_bit_exact" in checks


@pytest.mark.parametrize(
    ("overrides", "failed"),
    [
        ({"l_post": L_PRE + 2 * LOSS_TOL}, ["resume_weights_restored"]),
        ({"l_post": float("nan")}, ["resume_weights_restored"]),
        ({"moments_nonzero": False}, ["resume_optimizer_moments_restored"]),
        ({"moments_materialized": False}, ["resume_optimizer_moments_restored"]),
        ({"sched_last_epoch": 0}, ["resume_scheduler_restored"]),
        (
            {"optimizer_state": {"state": {"w": {"exp_avg_sq": MOMENT * 2, "step": SAVE_STEP}}}},
            ["resume_optimizer_state_bit_exact"],
        ),
    ],
    ids=["weights-moved", "weights-nan", "moments-zero", "moments-absent", "scheduler-reset", "moments-changed"],
)
def test_each_unrestored_piece_fails_by_name(overrides, failed):
    assert _failed(_continuity(_capture(**overrides), optimizer_state=SAVED_STATE)) == failed


def test_a_capture_that_never_fired_fails():
    assert _continuity(None) == {"resume_capture_fired": False}


def test_the_bit_exact_check_is_opt_in():
    assert "resume_optimizer_state_bit_exact" not in _continuity(_capture())


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
