"""Tests for checkpoint-resume detection and the output_dir overwrite guard
(src/training/environment.py), single-process (no torch.distributed init).

Run: python -m pytest tests/cpu/config/test_training_environment.py
"""

import json
import os
from types import SimpleNamespace

import pytest
from accelerate import PartialState

# The probe's own spelling: the guard's exemption and the sentinel on disk must be the same name.
from src.distributed.filesystem import OUTPUT_FS_PROBE_PREFIX, RUN_LOG_DIR_NAME
from src.training.environment import detect_resume_checkpoint

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
    os.environ.setdefault("RANK", "0")
    raise SystemExit(pytest.main([__file__, "-q"]))
