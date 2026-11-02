"""Tests for checkpoint-resume detection, the output_dir overwrite guard, the pre-load run setup
(halo/entrypoints/run_setup.py) and the attention switch the parse sets, single-process (no torch.distributed init).

Run: python -m pytest tests/cpu/config/test_training_environment.py
"""

import json
import logging
from dataclasses import fields
from types import SimpleNamespace

import pytest
import torch
import yaml
from transformers import Qwen3NextConfig, TrainerState, TrainingArguments

from halo.checkpoint.format import INCOMPLETE_CHECKPOINTS_DIR_NAME, REFERENCE_CACHE_DIR_NAME, RUN_PLAN_FILE
from halo.cli import training_methods
from halo.entrypoints import script
from halo.entrypoints.run_setup import _validate_output_dir, resume_checkpoint_from_config
from halo.entrypoints.script import run_trainer
from halo.log import RUN_LOG_DIR_NAME
from halo.models.attention.selection import supports_fa4_backward

# The probe's own spelling: the guard's exemption and the sentinel on disk must be the same name.
from halo.runtime.filesystem import FS_PROBE_PREFIX
from tests.common.parallelism import make_parallelism_config
from tests.common.repo import REPO_ROOT
from tests.cpu.config.test_examples_parse import TRAINING_ROOT, parser_for

_TRAINING_SCRIPTS = sorted(set(training_methods(REPO_ROOT).values()))


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


def test_a_checkpointing_plan_record_does_not_block_the_relaunch_it_is_for(tmp_path):
    """A run that crashed before its first checkpoint leaves only its log and its checkpointing-plan record, which
    the relaunch must read to checkpoint more; anything else beside it still refuses the directory."""
    out = tmp_path / "out"
    (out / RUN_LOG_DIR_NAME).mkdir(parents=True)
    (out / RUN_PLAN_FILE).write_text("{}")
    assert resume_checkpoint_from_config(_config(out, resume=True), overwrite_output_dir=False) is None
    (out / "model.safetensors").write_bytes(b"previous run")
    with pytest.raises(ValueError, match="already exists and is not empty"):
        _validate_output_dir(str(out))


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
        model=torch.nn.Module(),
        train=lambda resume_from_checkpoint=None: None,
        cleanup_ep=lambda: None,
    )
    with caplog.at_level(logging.INFO):
        context = SimpleNamespace(
            parallelism_config=SimpleNamespace(
                has_ep_group=False,
                is_cp_mode=False,
                is_tp_mode=False,
                is_expert_tp_mode=False,
                is_pp_mode=False,
                data_parallel_size=1,
            ),
            mode_suffix="ddp",
            resume_checkpoint=resume_checkpoint_from_config(_config(out, resume=True), overwrite_output_dir=False),
        )
        run_trainer(trainer, context, method_name="SFT")
    lines = [record.getMessage() for record in caplog.records if "Resuming from checkpoint" in record.getMessage()]
    assert lines == [f"Resuming from checkpoint: {ckpt}"]


def _banner_trainer(model: torch.nn.Module) -> SimpleNamespace:
    return SimpleNamespace(
        callback_handler=SimpleNamespace(callbacks=[]),
        state=TrainerState(max_steps=0),
        model=model,
        train=lambda resume_from_checkpoint=None: None,
        cleanup_ep=lambda: None,
    )


def _banner_context(parallelism_config) -> SimpleNamespace:
    return SimpleNamespace(parallelism_config=parallelism_config, mode_suffix="test", resume_checkpoint=None)


@pytest.mark.parametrize(
    ("gpus_per_node", "axes", "lines"),
    [
        (4, {"ep_size": 2}, ["EP size: 2", "DP size: 4"]),
        (4, {"cp_size": 2}, ["CP size: 2", "DP size: 2"]),
        (4, {"tp_size": 2}, ["TP size: 2", "DP size: 2"]),
        (4, {"expert_tp_size": 2}, ["EP size: 1", "ETP size: 2", "DP size: 2"]),
        (2, {"pp_size": 2}, ["PP size: 2", "DP size: 2"]),
    ],
)
def test_the_start_banner_names_each_axis_the_run_uses(caplog, gpus_per_node, axes, lines):
    """The banner is where a user confirms the run has the shape they asked for, so every axis the run uses is
    named beside the DP size it leaves (PP stages sit on NVLink-domain boundaries, hence two 2-GPU nodes)."""
    parallelism_config = make_parallelism_config(world_size=4, gpus_per_node=gpus_per_node, **axes)
    with caplog.at_level(logging.INFO):
        run_trainer(_banner_trainer(torch.nn.Module()), _banner_context(parallelism_config), method_name="SFT")
    assert [record.getMessage().strip() for record in caplog.records if " size: " in record.getMessage()] == lines


@pytest.mark.parametrize(
    ("recorded", "lines"),
    [
        ("flash_attention_2", ["Attention: flash_attention_2"]),
        (None, []),
        ("no config", []),
    ],
)
def test_the_start_banner_names_the_attention_backend_the_policy_was_built_with(caplog, recorded, lines):
    """The detection line names only the GPU's default, which the model-level checks may move a model off, so the
    banner names the backend the policy's config records; a model that records none, or has no config, adds no
    line."""
    model = torch.nn.Module()
    if recorded != "no config":
        model.config = Qwen3NextConfig()
        model.config._attn_implementation = recorded
    with caplog.at_level(logging.INFO):
        run_trainer(
            _banner_trainer(model),
            _banner_context(make_parallelism_config(world_size=1, gpus_per_node=1)),
            method_name="SFT",
        )
    assert [record.getMessage().strip() for record in caplog.records if "Attention:" in record.getMessage()] == lines


# Run setup before the model load


def _required_values(dataclass_types) -> dict:
    """The fields a script's configs refuse to construct without: the distillation teacher."""
    declared = {field.name for config in dataclass_types for field in fields(config)}
    return {"teacher_model": "org/teacher"} if "teacher_model" in declared else {}


@pytest.mark.parametrize("full_determinism", [True, False])
@pytest.mark.parametrize("entry_script", _TRAINING_SCRIPTS, ids=lambda path: path.stem)
def test_full_determinism_reaches_the_attention_choice_as_the_config_is_parsed(
    tmp_path, monkeypatch, entry_script, full_determinism
):
    """HF's Trainer turns ``FLASH_ATTENTION_DETERMINISTIC`` on only in its ``__init__``, after the model load picked
    the attention backend, and a script may plan its rows before its run setup: every training script's own parse
    turns the switch on, or a head_dim-256 family keeps FA4 in the plan while the load demotes it."""
    # Recorded, so the variable the parse writes is restored after the test.
    monkeypatch.setenv("FLASH_ATTENTION_DETERMINISTIC", "0")
    dataclass_types = parser_for(str(entry_script.relative_to(TRAINING_ROOT))).dataclass_types
    values = {
        "output_dir": str(tmp_path / "out"),
        "use_cpu": True,
        "bf16": False,
        "full_determinism": full_determinism,
        **_required_values(dataclass_types),
    }
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(values))
    monkeypatch.setattr(script, "init_distributed", lambda: None)
    monkeypatch.setattr(script, "install_log_tee", lambda output_dir: None)
    monkeypatch.setattr("sys.argv", ["train.py", str(path)])
    parsed = script.parse_script_args(*dataclass_types)
    (training_config,) = [config for config in parsed if isinstance(config, TrainingArguments)]
    assert training_config.full_determinism is full_determinism
    assert supports_fa4_backward(Qwen3NextConfig()) is not full_determinism


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
