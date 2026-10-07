"""Tests for checkpoint-resume detection and the output_dir overwrite guard
(src/training/environment.py), single-process (no torch.distributed init).

Run: python -m pytest tests/cpu/config/test_training_environment.py
"""

import json
from types import SimpleNamespace

import pytest
from accelerate import PartialState

from src.checkpoint.format import INCOMPLETE_CHECKPOINTS_DIR_NAME, REFERENCE_CACHE_DIR_NAME

# The probe's own spelling: the guard's exemption and the sentinel on disk must be the same name.
from src.distributed.filesystem import OUTPUT_FS_PROBE_PREFIX, RUN_LOG_DIR_NAME
from src.training.environment import _validate_output_dir, detect_resume_checkpoint

# The module logs through accelerate's logger, which requires an initialized state (CPU here).
PartialState(cpu=True)


def _config(output_dir, resume=None, overwrite=False):
    return SimpleNamespace(
        output_dir=str(output_dir),
        resume_from_checkpoint=resume,
        overwrite_output_dir=overwrite,
    )


def _write_checkpoint(output_dir, step=5):
    ckpt = output_dir / f"checkpoint-{step}"
    ckpt.mkdir(parents=True)
    (ckpt / "trainer_state.json").write_text(json.dumps({"global_step": step}))
    return ckpt


# Explicit resume path


def test_explicit_missing_resume_path_raises(tmp_path):
    """An explicit-but-nonexistent path must fail loud — degrading to a from-scratch run silently
    discards the user's resume intent (and the output_dir guard was skipped for it)."""
    config = _config(tmp_path / "out", resume=str(tmp_path / "does-not-exist" / "checkpoint-100"))
    with pytest.raises(ValueError, match="does not exist"):
        detect_resume_checkpoint(config)


def test_explicit_existing_resume_path_returned(tmp_path):
    ckpt = _write_checkpoint(tmp_path / "out")
    config = _config(tmp_path / "out", resume=str(ckpt))
    assert detect_resume_checkpoint(config) == str(ckpt)


# Auto-detect (resume_from_checkpoint: true)


def test_auto_resume_finds_last_checkpoint(tmp_path):
    out = tmp_path / "out"
    _write_checkpoint(out, step=5)
    last = _write_checkpoint(out, step=10)
    config = _config(out, resume=True)
    assert detect_resume_checkpoint(config) == str(last)


def test_auto_resume_passes_over_a_newer_checkpoint_whose_save_never_completed(tmp_path):
    """The mixin publishes trainer_state.json last, so a step directory without it is a save that
    stopped partway; resuming it would load a mix of the new and the absent files. It leaves the
    step-directory namespace with its data intact, so rotation cannot take it for the newest."""
    out = tmp_path / "out"
    complete = _write_checkpoint(out, step=5)
    stopped = out / "checkpoint-10"
    stopped.mkdir()
    (stopped / "model.safetensors").write_text("weights of a save that never finished")
    (stopped / ".trainer_state.json.uncommitted").write_text(json.dumps({"global_step": 10}))
    config = _config(out, resume=True)
    assert detect_resume_checkpoint(config) == str(complete)
    assert not stopped.exists(), "the incomplete step directory stayed where rotation counts it"
    kept = out / INCOMPLETE_CHECKPOINTS_DIR_NAME / "checkpoint-10" / "model.safetensors"
    assert kept.read_text() == "weights of a save that never finished"


def test_an_explicit_resume_also_sets_aside_an_incomplete_newer_checkpoint(tmp_path):
    out = tmp_path / "out"
    complete = _write_checkpoint(out, step=5)
    (out / "checkpoint-10").mkdir()
    assert detect_resume_checkpoint(_config(out, resume=str(complete))) == str(complete)
    assert {path.name for path in out.iterdir()} == {"checkpoint-5", INCOMPLETE_CHECKPOINTS_DIR_NAME}


def test_a_second_set_aside_of_the_same_step_keeps_both(tmp_path):
    """A step torn twice (two resumes, each stopped mid-save of the same step) keeps both copies."""
    out = tmp_path / "out"
    _write_checkpoint(out, step=5)
    for attempt in ("first", "second"):
        (out / "checkpoint-10").mkdir()
        (out / "checkpoint-10" / "attempt").write_text(attempt)
        detect_resume_checkpoint(_config(out, resume=True))
    holding = out / INCOMPLETE_CHECKPOINTS_DIR_NAME
    assert sorted((path / "attempt").read_text() for path in holding.iterdir()) == ["first", "second"]


def test_auto_resume_with_no_complete_checkpoint_raises(tmp_path):
    """Starting fresh over an output_dir whose only checkpoint is incomplete would discard it."""
    out = tmp_path / "out"
    (out / "checkpoint-10").mkdir(parents=True)
    config = _config(out, resume=True)
    with pytest.raises(RuntimeError, match="incomplete"):
        detect_resume_checkpoint(config)
    assert (out / "checkpoint-10").is_dir(), "a refused resume moved the checkpoint it refused"


def test_auto_resume_orders_steps_numerically(tmp_path):
    out = tmp_path / "out"
    _write_checkpoint(out, step=9)
    last = _write_checkpoint(out, step=10)
    (out / "checkpoint-11-copy").mkdir()
    assert detect_resume_checkpoint(_config(out, resume=True)) == str(last)


def test_auto_resume_missing_output_dir_starts_fresh(tmp_path):
    config = _config(tmp_path / "never-created", resume=True)
    assert detect_resume_checkpoint(config) is None


def test_auto_resume_nothing_found_revalidates_output_dir(tmp_path):
    """resume=True skipped the output_dir guard in setup; when auto-detection finds no checkpoint
    the run is effectively fresh, so a non-empty output_dir must raise, not be overwritten."""
    out = tmp_path / "out"
    out.mkdir()
    (out / "model.safetensors").write_text("weights from a previous run")
    config = _config(out, resume=True)
    with pytest.raises(ValueError, match="already exists and is not empty"):
        detect_resume_checkpoint(config)


def test_auto_resume_true_string_also_revalidates(tmp_path):
    """The YAML string forms 'True'/'true' take the auto-detect path, guard included."""
    out = tmp_path / "out"
    out.mkdir()
    (out / "stale.bin").write_text("x")
    config = _config(out, resume="true")
    with pytest.raises(ValueError, match="already exists and is not empty"):
        detect_resume_checkpoint(config)


def test_auto_resume_nothing_found_internal_dirs_ok(tmp_path):
    """The run's own log/ directory does not count as previous-run content."""
    out = tmp_path / "out"
    (out / RUN_LOG_DIR_NAME).mkdir(parents=True)
    config = _config(out, resume=True)
    assert detect_resume_checkpoint(config) is None


def test_auto_resume_nothing_found_stale_fs_probe_marker_ok(tmp_path):
    """A job killed inside the output-filesystem probe leaves its ``.halo_fs_probe_<ns>`` sentinel
    behind (rank 0's removal never ran). The marker is the toolkit's own inert dropping — blocking
    the next launch on it bricks the output_dir, and its ns suffix means only a PREFIX match fires."""
    out = tmp_path / "out"
    out.mkdir()
    (out / f"{OUTPUT_FS_PROBE_PREFIX}123456789").write_text("halo")
    config = _config(out, resume=True)
    assert detect_resume_checkpoint(config) is None


def test_stale_fs_probe_marker_does_not_exempt_other_leftovers(tmp_path):
    """Anti-over-exemption: the prefix exempts that one sentinel, not hidden files at large — a
    previous run's leftovers must still refuse the directory even beside a stale marker."""
    out = tmp_path / "out"
    out.mkdir()
    (out / f"{OUTPUT_FS_PROBE_PREFIX}123456789").write_text("halo")
    (out / ".nfs00000001").write_text("a previous run's deleted-but-open file")
    config = _config(out, resume=True)
    with pytest.raises(ValueError, match="already exists and is not empty"):
        detect_resume_checkpoint(config)


@pytest.mark.parametrize("nfs_remnant", [False, True])
def test_reference_scratch_does_not_block_a_fresh_launch(tmp_path, nfs_remnant):
    out = tmp_path / "out"
    scratch = out / REFERENCE_CACHE_DIR_NAME
    scratch.mkdir(parents=True)
    if nfs_remnant:
        mapped = scratch / "run-id"
        mapped.mkdir()
        (mapped / ".nfs-live-mapping").write_bytes(b"open reference scores")
    _validate_output_dir(str(out))
    assert detect_resume_checkpoint(_config(out, resume=True)) is None


@pytest.mark.parametrize("leftover", ["checkpoint", "model.safetensors", "unrelated", ".nfs-outside-cache"])
def test_reference_scratch_does_not_exempt_previous_run_contents(tmp_path, leftover):
    out = tmp_path / "out"
    (out / REFERENCE_CACHE_DIR_NAME).mkdir(parents=True)
    if leftover == "checkpoint":
        _write_checkpoint(out)
    elif leftover == "unrelated":
        (out / leftover).mkdir()
    else:
        (out / leftover).write_bytes(b"previous run")
    with pytest.raises(ValueError, match="already exists and is not empty"):
        _validate_output_dir(str(out))


def test_auto_resume_nothing_found_overwrite_flag_skips_validation(tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    (out / "model.safetensors").write_text("weights")
    config = _config(out, resume=True, overwrite=True)
    assert detect_resume_checkpoint(config) is None


# No resume requested


def test_no_resume_returns_none(tmp_path):
    assert detect_resume_checkpoint(_config(tmp_path / "out")) is None
    assert detect_resume_checkpoint(_config(tmp_path / "out", resume=False)) is None


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
