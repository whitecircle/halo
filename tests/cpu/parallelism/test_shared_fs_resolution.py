#!/usr/bin/env python
"""Input/output shared-filesystem resolution.

``DIST_INPUT_SHARED_FILESYSTEM`` (read side: downloads, dataset processing, HF caches) and
``DIST_OUTPUT_SHARED_FILESYSTEM`` (write side: checkpoints, run.log, dumped artifacts) each fall
back to the ``DIST_SHARED_FILESYSTEM`` umbrella, so single-flag configs keep their behavior while
a multi-node run over a flaky NFS export can combine per-node input caches (avoids cross-node
stale-file-handle reads) with single-writer shared-mount checkpoints.

    python tests/cpu/parallelism/test_shared_fs_resolution.py
"""

import logging
import os

import pytest
from datasets import Dataset

from src.data.pipeline import processing
from src.distributed import runtime
from src.training import run_logging as run_logger

FS_VARS = ("DIST_SHARED_FILESYSTEM", "DIST_INPUT_SHARED_FILESYSTEM", "DIST_OUTPUT_SHARED_FILESYSTEM")
RANK_VARS = ("RANK", "SLURM_PROCID", "LOCAL_RANK", "SLURM_LOCALID")


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for var in FS_VARS + RANK_VARS:
        monkeypatch.delenv(var, raising=False)


def test_default_is_shared_everywhere():
    assert runtime.is_shared_filesystem()
    assert runtime.is_input_shared_filesystem()
    assert runtime.is_output_shared_filesystem()


def test_umbrella_propagates_to_both_sides(monkeypatch):
    monkeypatch.setenv("DIST_SHARED_FILESYSTEM", "0")
    assert not runtime.is_shared_filesystem()
    assert not runtime.is_input_shared_filesystem()
    assert not runtime.is_output_shared_filesystem()


def test_per_node_input_shared_output_combo(monkeypatch):
    monkeypatch.setenv("DIST_INPUT_SHARED_FILESYSTEM", "0")
    assert not runtime.is_input_shared_filesystem()
    assert runtime.is_output_shared_filesystem()
    assert runtime.is_shared_filesystem()


def test_specific_overrides_umbrella(monkeypatch):
    monkeypatch.setenv("DIST_SHARED_FILESYSTEM", "0")
    monkeypatch.setenv("DIST_OUTPUT_SHARED_FILESYSTEM", "on")
    assert runtime.is_output_shared_filesystem()
    assert not runtime.is_input_shared_filesystem()


def test_set_falsey_specific_var_does_not_fall_back(monkeypatch):
    # env_flag only falls back when the var is ABSENT; a set "0" must win over a truthy umbrella.
    monkeypatch.setenv("DIST_SHARED_FILESYSTEM", "1")
    monkeypatch.setenv("DIST_OUTPUT_SHARED_FILESYSTEM", "0")
    assert not runtime.is_output_shared_filesystem()
    assert runtime.is_input_shared_filesystem()


@pytest.mark.parametrize(
    ("env", "expected"),
    [
        ({}, {"DIST_SHARED_FILESYSTEM": "shared", "DIST_INPUT_SHARED_FILESYSTEM": "shared"}),
        (
            {"DIST_SHARED_FILESYSTEM": "0", "DIST_OUTPUT_SHARED_FILESYSTEM": "1"},
            {
                "DIST_SHARED_FILESYSTEM": "per-node",
                "DIST_INPUT_SHARED_FILESYSTEM": "per-node",
                "DIST_OUTPUT_SHARED_FILESYSTEM": "shared",
            },
        ),
    ],
)
def test_resolved_shared_filesystem_scope_is_logged(monkeypatch, caplog, env, expected):
    """The resolved scope must reach the log, per var.

    It decides whether the dataset cache is filled once or once per node, and a wrong setting is
    otherwise silent — the run stays correct while every rank re-maps the corpus, so nothing but the
    wall clock says so. Reading it back from the env is not the same check: the side vars fall back to
    the umbrella and rank 0's values override a disagreeing rank's, so only the agreed dict is the
    scope the run actually used.
    """
    monkeypatch.setattr(runtime, "_SHARED_FILESYSTEM_CONSENSUS", None)
    for var, value in env.items():
        monkeypatch.setenv(var, value)

    with caplog.at_level(logging.INFO, logger=runtime.logger.name):
        runtime.resolve_shared_filesystem_consensus()

    lines = [
        record.getMessage()
        for record in caplog.records
        if record.levelno == logging.INFO and "DIST_SHARED_FILESYSTEM" in record.getMessage()
    ]
    assert len(lines) == 1, f"expected exactly one resolved-scope line, got {lines}"
    for var, scope in expected.items():
        assert f"'{var}': '{scope}'" in lines[0], lines[0]


def test_load_rank_follows_input_flag(monkeypatch):
    monkeypatch.setattr(runtime, "is_global_main_process", lambda: True)
    monkeypatch.setattr(runtime, "is_local_main_process", lambda: False)
    assert runtime.fs_aware_load_rank()  # shared input → global main
    monkeypatch.setenv("DIST_INPUT_SHARED_FILESYSTEM", "0")
    assert not runtime.fs_aware_load_rank()  # per-node input → local main
    # The output flag must not leak into the load rank.
    monkeypatch.setenv("DIST_OUTPUT_SHARED_FILESYSTEM", "1")
    assert not runtime.fs_aware_load_rank()


def test_save_rank_follows_output_flag(monkeypatch):
    monkeypatch.setattr(runtime, "is_global_main_process", lambda: True)
    monkeypatch.setattr(runtime, "is_local_main_process", lambda: False)
    assert runtime.fs_aware_save_rank()  # shared output → global main
    monkeypatch.setenv("DIST_OUTPUT_SHARED_FILESYSTEM", "0")
    assert not runtime.fs_aware_save_rank()  # per-node output → local main
    monkeypatch.setenv("DIST_INPUT_SHARED_FILESYSTEM", "1")
    assert not runtime.fs_aware_save_rank()


def test_log_tee_elects_its_writer_with_fs_aware_save_rank(monkeypatch, tmp_path):
    """``run.log`` is an output artifact, so the tee must elect its writer through
    ``fs_aware_save_rank`` — the same predicate, and therefore the same output flag, the checkpoint
    writers use (pinned above). A tee carrying its own rank rule would drift from that flag and let
    two nodes both believe they are the sole writer, interleaving one ``run.log``.

    Only the non-writer path is exercised: the writer path ``dup2``s fds 1/2 into a ``tee``
    subprocess, which would hijack the test runner's own output.
    """
    consulted = []

    def _not_the_writer():
        consulted.append(True)
        return False

    monkeypatch.setattr(run_logger, "fs_aware_save_rank", _not_the_writer)
    monkeypatch.setattr(run_logger, "_TEE_LOG_PATH", None)

    run_logger.install_log_tee(str(tmp_path))

    assert consulted, "install_log_tee did not consult fs_aware_save_rank"
    assert not (tmp_path / "log").exists(), "a non-writer rank created the run.log directory"


def test_coordinated_dataset_map_elects_on_the_input_side(monkeypatch):
    """The dataset map fills a cache the OTHER ranks then read, so it is read-side work and must
    elect with ``fs_aware_load_rank``.

    Reverting it to ``fs_aware_save_rank`` is invisible whenever the two agree — i.e. every
    single-flag config, which is what the GPU coordination test exercises. It only surfaces on
    ``DIST_INPUT_SHARED_FILESYSTEM=0`` with a shared output, exactly the config the split exists for,
    where every node re-maps the whole corpus or reads a peer's half-written cache.
    """
    consulted: list[str] = []
    monkeypatch.setattr(processing, "fs_aware_load_rank", lambda: consulted.append("load") or True)
    monkeypatch.setattr(processing, "fs_aware_save_rank", lambda: consulted.append("save") or True)

    processing.coordinated_map(Dataset.from_list([{"x": 1}, {"x": 2}]), lambda row: row, desc="probe")

    assert "load" in consulted, "the coordinated map did not elect its writer with fs_aware_load_rank"
    assert "save" not in consulted, "the coordinated map elected on the OUTPUT side — read-side work"


def test_fs_aware_makedirs_honours_the_writer_rank_side(monkeypatch, tmp_path):
    """``fs_aware_makedirs`` defaults to the output side, and the read-side caller (the
    ``HF_DATASETS_CACHE`` dir) passes ``fs_aware_load_rank``. A parameter that were ignored would put
    every read-side cache dir back on the output flag with no visible symptom until the two diverge.
    """
    # Drive the real chain rather than patching the predicates: the default argument binds the
    # function OBJECT at import, so a patched name would never reach it.
    monkeypatch.setattr(runtime, "is_global_main_process", lambda: True)
    monkeypatch.setattr(runtime, "is_local_main_process", lambda: False)
    monkeypatch.setenv("DIST_OUTPUT_SHARED_FILESYSTEM", "0")  # save rank → local main → False
    monkeypatch.setenv("DIST_INPUT_SHARED_FILESYSTEM", "1")  # load rank → global main → True
    assert not runtime.fs_aware_save_rank() and runtime.fs_aware_load_rank()

    default_dir = tmp_path / "output_side"
    runtime.fs_aware_makedirs(str(default_dir))
    assert not default_dir.exists(), "the default writer rank is not fs_aware_save_rank"

    read_dir = tmp_path / "input_side"
    runtime.fs_aware_makedirs(str(read_dir), writer_rank=runtime.fs_aware_load_rank)
    assert read_dir.exists(), "writer_rank was ignored — the read-side caller followed the output flag"


def test_resetting_the_consensus_lets_the_flags_read_the_environment_again(monkeypatch):
    """``reset_shared_filesystem_consensus`` is the seam that makes the memo re-readable.

    The flags are agreed ONCE per process and then answered from the memo, so without a public
    reset a later caller keeps getting the scope some earlier call agreed — silently, since every
    getter still answers. A no-op reset would leave this asserting the stale value.
    """
    monkeypatch.setattr(runtime, "_SHARED_FILESYSTEM_CONSENSUS", dict.fromkeys(FS_VARS, False))
    assert not runtime.is_shared_filesystem(), "the memo is not being consulted at all"

    runtime.reset_shared_filesystem_consensus()
    monkeypatch.setenv("DIST_SHARED_FILESYSTEM", "1")
    assert runtime.is_shared_filesystem(), "the memo survived the reset — the env is still ignored"


def test_a_second_log_tee_for_another_output_dir_warns_instead_of_silently_dropping_it(monkeypatch, caplog):
    """The fd redirect is process-global and one-shot, so a second output_dir cannot get one.

    Silently returning would leave that run's console output in the FIRST run's run.log with no
    trace; the guard must name both paths.
    """
    monkeypatch.setattr(run_logger, "_TEE_LOG_PATH", os.path.abspath("/first/log/run.log"))
    monkeypatch.setattr(run_logger, "fs_aware_save_rank", lambda: True)

    with caplog.at_level(logging.WARNING, logger=run_logger.__name__):
        run_logger.install_log_tee("/second")

    assert "/first/log/run.log" in caplog.text and "/second" in caplog.text, (
        "a second install_log_tee for a different output_dir was dropped without a warning"
    )
    assert os.path.abspath("/first/log/run.log") == run_logger._TEE_LOG_PATH, "the live tee target was overwritten"


def test_a_second_log_tee_for_the_same_output_dir_is_a_silent_no_op(monkeypatch, caplog):
    """Re-entering the same run's setup is ordinary (the parser can run twice) — no warning there."""
    monkeypatch.setattr(run_logger, "_TEE_LOG_PATH", os.path.abspath("/same/log/run.log"))
    monkeypatch.setattr(run_logger, "fs_aware_save_rank", lambda: True)

    with caplog.at_level(logging.WARNING, logger=run_logger.__name__):
        run_logger.install_log_tee("/same")

    assert not caplog.text, f"an idempotent re-install warned: {caplog.text}"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
