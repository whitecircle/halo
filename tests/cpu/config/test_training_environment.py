"""Tests for checkpoint-resume detection, the output_dir overwrite guard and the pre-load run setup
(halo/entrypoints/run_setup.py), single-process (no torch.distributed init).

Run: python -m pytest tests/cpu/config/test_training_environment.py
"""

import json
import logging
from types import SimpleNamespace
from unittest import mock

import pytest
from transformers import Qwen3NextConfig, TrainerState, TrainingArguments

from halo.checkpoint.format import INCOMPLETE_CHECKPOINTS_DIR_NAME, REFERENCE_CACHE_DIR_NAME
from halo.entrypoints import run_setup
from halo.entrypoints.run_setup import _validate_output_dir, configure_training_run, resume_checkpoint_from_config
from halo.entrypoints.script import run_trainer
from halo.log import RUN_LOG_DIR_NAME
from halo.models.attention.selection import supports_fa4_backward

# The probe's own spelling: the guard's exemption and the sentinel on disk must be the same name.
from halo.runtime.filesystem import FS_PROBE_PREFIX


def _config(output_dir, resume=None):
    return SimpleNamespace(output_dir=str(output_dir), resume_from_checkpoint=resume)


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
        resume_checkpoint_from_config(config, overwrite_output_dir=False)


def test_explicit_existing_resume_path_returned(tmp_path):
    ckpt = _write_checkpoint(tmp_path / "out")
    config = _config(tmp_path / "out", resume=str(ckpt))
    assert resume_checkpoint_from_config(config, overwrite_output_dir=False) == str(ckpt)


# Auto-detect (resume_from_checkpoint: true)


def test_auto_resume_finds_last_checkpoint(tmp_path):
    out = tmp_path / "out"
    _write_checkpoint(out, step=5)
    last = _write_checkpoint(out, step=10)
    config = _config(out, resume=True)
    assert resume_checkpoint_from_config(config, overwrite_output_dir=False) == str(last)


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
    assert resume_checkpoint_from_config(config, overwrite_output_dir=False) == str(complete)
    assert not stopped.exists(), "the incomplete step directory stayed where rotation counts it"
    kept = out / INCOMPLETE_CHECKPOINTS_DIR_NAME / "checkpoint-10" / "model.safetensors"
    assert kept.read_text() == "weights of a save that never finished"


def test_an_explicit_resume_also_sets_aside_an_incomplete_newer_checkpoint(tmp_path):
    out = tmp_path / "out"
    complete = _write_checkpoint(out, step=5)
    (out / "checkpoint-10").mkdir()
    assert resume_checkpoint_from_config(_config(out, resume=str(complete)), overwrite_output_dir=False) == str(
        complete
    )
    assert {path.name for path in out.iterdir()} == {"checkpoint-5", INCOMPLETE_CHECKPOINTS_DIR_NAME}


def test_a_second_set_aside_of_the_same_step_keeps_both(tmp_path):
    """A step torn twice (two resumes, each stopped mid-save of the same step) keeps both copies."""
    out = tmp_path / "out"
    _write_checkpoint(out, step=5)
    for attempt in ("first", "second"):
        (out / "checkpoint-10").mkdir()
        (out / "checkpoint-10" / "attempt").write_text(attempt)
        resume_checkpoint_from_config(_config(out, resume=True), overwrite_output_dir=False)
    holding = out / INCOMPLETE_CHECKPOINTS_DIR_NAME
    assert sorted((path / "attempt").read_text() for path in holding.iterdir()) == ["first", "second"]


def test_auto_resume_with_no_complete_checkpoint_raises(tmp_path):
    """Starting fresh over an output_dir whose only checkpoint is incomplete would discard it."""
    out = tmp_path / "out"
    (out / "checkpoint-10").mkdir(parents=True)
    config = _config(out, resume=True)
    with pytest.raises(RuntimeError, match="incomplete"):
        resume_checkpoint_from_config(config, overwrite_output_dir=False)
    assert (out / "checkpoint-10").is_dir(), "a refused resume moved the checkpoint it refused"


def test_auto_resume_orders_steps_numerically(tmp_path):
    out = tmp_path / "out"
    _write_checkpoint(out, step=9)
    last = _write_checkpoint(out, step=10)
    (out / "checkpoint-11-copy").mkdir()
    assert resume_checkpoint_from_config(_config(out, resume=True), overwrite_output_dir=False) == str(last)


def test_auto_resume_missing_output_dir_starts_fresh(tmp_path):
    config = _config(tmp_path / "never-created", resume=True)
    assert resume_checkpoint_from_config(config, overwrite_output_dir=False) is None


def test_auto_resume_nothing_found_revalidates_output_dir(tmp_path):
    """resume=True skipped the output_dir guard in setup; when auto-detection finds no checkpoint
    the run is effectively fresh, so a non-empty output_dir must raise, not be overwritten."""
    out = tmp_path / "out"
    out.mkdir()
    (out / "model.safetensors").write_text("weights from a previous run")
    config = _config(out, resume=True)
    with pytest.raises(ValueError, match="already exists and is not empty"):
        resume_checkpoint_from_config(config, overwrite_output_dir=False)


def test_auto_resume_true_string_also_revalidates(tmp_path):
    """The YAML string forms 'True'/'true' take the auto-detect path, guard included."""
    out = tmp_path / "out"
    out.mkdir()
    (out / "stale.bin").write_text("x")
    config = _config(out, resume="true")
    with pytest.raises(ValueError, match="already exists and is not empty"):
        resume_checkpoint_from_config(config, overwrite_output_dir=False)


def test_auto_resume_nothing_found_internal_dirs_ok(tmp_path):
    """The run's own log/ directory does not count as previous-run content."""
    out = tmp_path / "out"
    (out / RUN_LOG_DIR_NAME).mkdir(parents=True)
    config = _config(out, resume=True)
    assert resume_checkpoint_from_config(config, overwrite_output_dir=False) is None


def test_auto_resume_nothing_found_stale_fs_probe_marker_ok(tmp_path):
    """A job killed inside the output-filesystem probe leaves its ``.halo_fs_probe_<ns>`` sentinel
    behind (rank 0's removal never ran). The marker is the toolkit's own inert dropping — blocking
    the next launch on it bricks the output_dir, and its ns suffix means only a PREFIX match fires."""
    out = tmp_path / "out"
    out.mkdir()
    (out / f"{FS_PROBE_PREFIX}123456789").write_text("halo")
    config = _config(out, resume=True)
    assert resume_checkpoint_from_config(config, overwrite_output_dir=False) is None


def test_stale_fs_probe_marker_does_not_exempt_other_leftovers(tmp_path):
    """Anti-over-exemption: the prefix exempts that one sentinel, not hidden files at large — a
    previous run's leftovers must still refuse the directory even beside a stale marker."""
    out = tmp_path / "out"
    out.mkdir()
    (out / f"{FS_PROBE_PREFIX}123456789").write_text("halo")
    (out / ".nfs00000001").write_text("a previous run's deleted-but-open file")
    config = _config(out, resume=True)
    with pytest.raises(ValueError, match="already exists and is not empty"):
        resume_checkpoint_from_config(config, overwrite_output_dir=False)


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
    assert resume_checkpoint_from_config(_config(out, resume=True), overwrite_output_dir=False) is None


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
    config = _config(out, resume=True)
    assert resume_checkpoint_from_config(config, overwrite_output_dir=True) is None


# No resume requested


def test_no_resume_returns_none(tmp_path):
    assert resume_checkpoint_from_config(_config(tmp_path / "out"), overwrite_output_dir=False) is None
    assert resume_checkpoint_from_config(_config(tmp_path / "out", resume=False), overwrite_output_dir=False) is None


def test_a_resumed_run_logs_its_checkpoint_once(tmp_path, caplog):
    """The resolution names the checkpoint it picked; the runner that trains from it adds no second line."""
    out = tmp_path / "out"
    ckpt = _write_checkpoint(out, step=5)
    trainer = SimpleNamespace(
        callback_handler=SimpleNamespace(callbacks=[]),
        state=TrainerState(max_steps=5),
        train=lambda resume_from_checkpoint=None: None,
        cleanup_ep=lambda: None,
    )
    with caplog.at_level(logging.INFO):
        context = SimpleNamespace(
            parallelism_config=SimpleNamespace(
                has_ep_group=False, is_cp_mode=False, is_tp_mode=False, is_expert_tp_mode=False, data_parallel_size=1
            ),
            mode_suffix="ddp",
            resume_checkpoint=resume_checkpoint_from_config(_config(out, resume=True), overwrite_output_dir=False),
        )
        run_trainer(trainer, context, method_name="SFT")
    lines = [record.getMessage() for record in caplog.records if "Resuming from checkpoint" in record.getMessage()]
    assert lines == [f"Resuming from checkpoint: {ckpt}"]


# Run setup before the model load


@pytest.mark.parametrize("full_determinism", [True, False])
def test_full_determinism_reaches_the_attention_choice_before_the_load(tmp_path, monkeypatch, full_determinism):
    """HF's Trainer turns ``FLASH_ATTENTION_DETERMINISTIC`` on only in its ``__init__``, after the model load picked
    the attention backend; set later, a head_dim-256 family keeps FA4 and its first backward asserts."""
    # Recorded, so the variables the setup writes are restored after the test.
    for name in ("FLASH_ATTENTION_DETERMINISTIC", "HF_DATASETS_CACHE"):
        monkeypatch.setenv(name, "0")
    monkeypatch.setattr(run_setup.datasets.config, "HF_DATASETS_CACHE", run_setup.datasets.config.HF_DATASETS_CACHE)
    config = TrainingArguments(output_dir=str(tmp_path), full_determinism=full_determinism, use_cpu=True, report_to=[])
    with (
        mock.patch.object(run_setup, "ensure_fa4_kernel_cache_env"),
        mock.patch.object(run_setup, "anchor_jit_cache_dir"),
        mock.patch.object(run_setup, "configure_float32_matmul_precision"),
        mock.patch.object(run_setup, "ensure_cache_dir", return_value=str(tmp_path / "datasets")),
        mock.patch.object(run_setup.transformers.logging, "set_verbosity_info"),
        mock.patch.object(run_setup.transformers.logging, "set_verbosity_error"),
        mock.patch.object(run_setup, "_configure_tracking_env_vars"),
        mock.patch.object(run_setup, "set_seed"),
        mock.patch.object(run_setup, "barrier"),
    ):
        configure_training_run(SimpleNamespace(), config, "sft-test", overwrite_output_dir=True)
    assert supports_fa4_backward(Qwen3NextConfig()) is not full_determinism


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
