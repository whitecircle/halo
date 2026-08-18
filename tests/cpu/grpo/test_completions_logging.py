#!/usr/bin/env python
"""Unit tests for :func:`src.trainers.grpo.rollout.completions_logging.emit_completion_artifacts`.

Locks the decoupling contract used by the online + env GRPO trainers: the durable parquet record is
gated by ``save`` (CommonScriptArguments.save_completions) and the rich console table by ``console``
(TRL's log_completions) — independently. Recoupling them, e.g. skipping the parquet unless the
console prints, is the failure this forbids.

Also pins how the record REACHES that writer (``rollout_metrics._populate_completion_logs``): the
full trajectory render is gathered to the writer rank, never all-gathered to all of them, and every
rank enters the same four gathers whether or not it accumulates the result. WHEN it is called
relative to the env-GRPO trust-region breaker is pinned where the breaker lives
(``test_grpo_update_breaker_and_group_effort.py``).

Run::

    python tests/cpu/grpo/test_completions_logging.py
"""

from __future__ import annotations

import os
import types

import pandas as pd
import pytest
import torch

import src.trainers.grpo.rollout.completions_logging as cl
import src.trainers.grpo.rollout.rollout_metrics as rm
from src.distributed import runtime
from src.trainers.grpo.rollout.completions_logging import emit_completion_artifacts


class _Accel:
    def __init__(self, is_main=True):
        self.is_main_process = is_main


class _State:
    def __init__(self, step=7):
        self.global_step = step


class _Args:
    def __init__(self, output_dir, report_to=None):
        self.output_dir = output_dir
        self.report_to = report_to or []


class _Model:
    def __init__(self, training=True):
        self.training = training


class _Trainer:
    """Minimal stand-in exposing exactly what emit_completion_artifacts reads."""

    def __init__(self, output_dir, is_main=True, step=7, report_to=None):
        self.accelerator = _Accel(is_main)
        self.state = _State(step)
        self.args = _Args(output_dir, report_to)
        self.model = _Model()
        self.num_completions_to_print = 2
        self.log_unique_prompts = False
        self._logs = {
            "prompt": ["p1", "p2"],
            "completion": ["c1", "c2"],
            "rewards": {"environment_reward": [1.0, 0.5]},
            "advantages": [0.4, -0.4],
            "extra": {},
            "images": [],
        }


def _parquet_path(output_dir, step=7):
    return os.path.join(output_dir, "completions", f"completions_{step:05d}.parquet")


def test_save_writes_parquet_without_console(tmp_path, monkeypatch) -> None:
    """save=True, console=False: parquet is written, console table is NOT printed."""
    printed = []
    monkeypatch.setattr(cl, "print_prompt_completions_sample", lambda *a, **k: printed.append(a))

    trainer = _Trainer(str(tmp_path))
    emit_completion_artifacts(trainer, console=False, save=True)

    path = _parquet_path(str(tmp_path))
    assert os.path.exists(path), "parquet must be written when save=True regardless of console"
    assert printed == [], "console table must NOT print when console=False"

    df = pd.read_parquet(path)
    assert list(df["prompt"]) == ["p1", "p2"]
    assert list(df["completion"]) == ["c1", "c2"]
    assert list(df["advantage"]) == [0.4, -0.4]
    assert list(df["environment_reward"]) == [1.0, 0.5]
    assert list(df["step"]) == [7, 7]


def test_console_only_without_save(tmp_path, monkeypatch) -> None:
    """console=True, save=False: console table prints, NO parquet written."""
    printed = []
    monkeypatch.setattr(cl, "print_prompt_completions_sample", lambda *a, **k: printed.append(a))
    monkeypatch.setattr(cl, "is_rich_available", lambda: True)

    trainer = _Trainer(str(tmp_path))
    emit_completion_artifacts(trainer, console=True, save=False)

    assert not os.path.exists(_parquet_path(str(tmp_path))), "no parquet when save=False"
    assert len(printed) == 1, "console table must print exactly once when console=True"


def test_non_writer_rank_writes_nothing(tmp_path, monkeypatch) -> None:
    """Only the elected writer persists artifacts (else N ranks race the same file).

    The election is ``fs_aware_save_rank`` — the same predicate every other output artifact uses —
    not ``accelerator.is_main_process``: on a non-shared output filesystem each node must write its
    own completions, and gating on the accelerator flag would lose nodes 1..N entirely.

    Driven through the real flag chain (``is_global_main_process`` / ``is_local_main_process`` +
    the env flags) rather than by patching ``fs_aware_save_rank`` itself, so the test pins WHICH
    predicate is consulted, not merely that some boolean gates the write.
    """
    printed = []
    monkeypatch.setattr(cl, "print_prompt_completions_sample", lambda *a, **k: printed.append(a))
    monkeypatch.setattr(runtime, "is_global_main_process", lambda: False)
    monkeypatch.setattr(runtime, "is_local_main_process", lambda: True)
    monkeypatch.delenv("DIST_SHARED_FILESYSTEM", raising=False)
    monkeypatch.delenv("DIST_OUTPUT_SHARED_FILESYSTEM", raising=False)
    monkeypatch.setattr(runtime, "_SHARED_FILESYSTEM_CONSENSUS", None)

    trainer = _Trainer(str(tmp_path))
    emit_completion_artifacts(trainer, console=True, save=True)

    assert not os.path.exists(_parquet_path(str(tmp_path)))
    assert printed == []


def test_every_node_writes_when_the_output_filesystem_is_not_shared(tmp_path, monkeypatch) -> None:
    """Anti-vacuity for the test above, and the reason the predicate changed: with a per-node output
    filesystem this same non-global-main rank IS its node's writer — gating on global rank 0 would
    silently drop that node's completions."""
    monkeypatch.setattr(cl, "print_prompt_completions_sample", lambda *a, **k: None)
    monkeypatch.setattr(runtime, "is_global_main_process", lambda: False)
    monkeypatch.setattr(runtime, "is_local_main_process", lambda: True)
    monkeypatch.setenv("DIST_OUTPUT_SHARED_FILESYSTEM", "0")
    monkeypatch.setattr(runtime, "_SHARED_FILESYSTEM_CONSENSUS", None)

    trainer = _Trainer(str(tmp_path))
    emit_completion_artifacts(trainer, console=False, save=True)

    assert os.path.exists(_parquet_path(str(tmp_path))), "a per-node output FS lost this node's completions"


def test_eval_mode_writes_suffixed_parquet(tmp_path, monkeypatch) -> None:
    """model.training=False: the parquet gets the ``_eval`` suffix and never overwrites the train file."""
    monkeypatch.setattr(cl, "print_prompt_completions_sample", lambda *a, **k: None)

    trainer = _Trainer(str(tmp_path))
    emit_completion_artifacts(trainer, console=False, save=True)
    trainer.model.training = False
    trainer._logs["completion"] = ["e1", "e2"]
    emit_completion_artifacts(trainer, console=False, save=True)

    train_df = pd.read_parquet(_parquet_path(str(tmp_path)))
    assert list(train_df["completion"]) == ["c1", "c2"], "eval write must not overwrite the train parquet"
    eval_path = os.path.join(str(tmp_path), "completions", "completions_00007_eval.parquet")
    assert os.path.exists(eval_path), "eval-mode write must land in the _eval-suffixed parquet"
    assert list(pd.read_parquet(eval_path)["completion"]) == ["e1", "e2"]


def test_empty_logs_is_noop(tmp_path) -> None:
    """Nothing generated yet (empty buffer): no parquet, no crash."""
    trainer = _Trainer(str(tmp_path))
    trainer._logs["prompt"] = []
    emit_completion_artifacts(trainer, console=True, save=True)
    assert not os.path.exists(_parquet_path(str(tmp_path)))


class _MetricsHost(rm.RolloutMetricsMixin):
    """Minimal stand-in exposing exactly what ``_populate_completion_logs`` reads."""

    def __init__(self, save=True, console=False):
        self._save_completions = save
        self.log_completions = console
        self._logs = {"prompt": [], "completion": [], "rewards": {"environment_reward": []}, "advantages": []}


def _rollouts(*prompts):
    return [types.SimpleNamespace(prompt=p, trajectory=None) for p in prompts]


def _fake_world(monkeypatch, *, world, rank, shared_fs):
    """Simulate a ``world``-rank job on rank ``rank``; returns the recorded collective counts.

    Every other rank contributes ``peer<i>``, so the writer's gathered content can be asserted to be
    the whole world's record in rank order — the property the artifact depends on.
    """
    calls = {"all_gather": 0, "gather_dst": 0}

    def _fake_gather_object(values, chunks=None, dst=None, **kwargs):
        calls["gather_dst"] += 1
        assert dst == 0, "a destination gather must target the shared-FS writer (global rank 0)"
        if chunks is not None:
            for i in range(world):
                chunks[i] = list(values) if i == rank else [f"peer{i}"]

    def _fake_all_gather(values):
        calls["all_gather"] += 1
        return [item for i in range(world) for item in (list(values) if i == rank else [f"peer{i}"])]

    # A stand-in bound to the module, not a patch of the real torch.distributed: the process must
    # not start believing it is in a 4-rank job.
    monkeypatch.setattr(
        rm,
        "dist",
        types.SimpleNamespace(
            is_available=lambda: True,
            is_initialized=lambda: True,
            get_world_size=lambda group=None: world,
            gather_object=_fake_gather_object,
        ),
    )
    monkeypatch.setattr(rm, "gather_object", _fake_all_gather)
    # The multi-rank gate reads the REAL torch.distributed, which this process never initializes.
    monkeypatch.setattr(rm, "is_multi_rank_run", lambda: world > 1)
    # Driven through the REAL writer election, so the test pins that the gather lands where the
    # parquet is written rather than merely that some boolean picks a rank.
    monkeypatch.setattr(runtime, "is_global_main_process", lambda: rank == 0)
    monkeypatch.setattr(runtime, "is_local_main_process", lambda: rank == 0)
    monkeypatch.setattr(rm, "is_output_shared_filesystem", lambda: shared_fs)
    monkeypatch.setattr(runtime, "is_output_shared_filesystem", lambda: shared_fs)
    return calls


def test_shared_fs_completion_logs_reach_the_writer_only(monkeypatch) -> None:
    """On a shared output FS the full trajectory render goes to the ONE writer, not to every rank.

    ``accelerate``'s ``gather_object`` is an all-gather: every rank receives every rank's payload and
    (on NCCL) stages the pickle through its own CUDA device, so the transient grows with world size
    while the only consumer is the writer rank's parquet. The writer's record must be unchanged.
    """
    calls = _fake_world(monkeypatch, world=4, rank=0, shared_fs=True)
    host = _MetricsHost()

    host._populate_completion_logs(_rollouts("p0"), torch.tensor([1.0]), torch.tensor([0.25]))

    assert calls["all_gather"] == 0, "the full-text payload must not be all-gathered to every rank"
    assert calls["gather_dst"] == 4, "all four gathers must still run on every rank (lock-step)"
    assert host._logs["prompt"] == ["p0", "peer1", "peer2", "peer3"], "the writer must hold the world record"
    assert host._logs["completion"] == ["(empty trajectory)", "peer1", "peer2", "peer3"]
    assert host._logs["rewards"]["environment_reward"] == [1.0, "peer1", "peer2", "peer3"]
    assert host._logs["advantages"] == [0.25, "peer1", "peer2", "peer3"]


def test_shared_fs_non_writer_still_runs_every_gather(monkeypatch) -> None:
    """A non-writer rank contributes to all four gathers and accumulates nothing.

    Skipping a gather here desyncs the world; accumulating restores the cost the split removes.
    """
    calls = _fake_world(monkeypatch, world=4, rank=2, shared_fs=True)
    host = _MetricsHost()

    host._populate_completion_logs(_rollouts("p2"), torch.tensor([1.0]), torch.tensor([0.25]))

    assert calls["gather_dst"] == 4, "a non-writer must enter every gather its peers enter"
    assert host._logs["prompt"] == [] and host._logs["completion"] == []


def test_non_shared_fs_keeps_the_all_gather_every_node_writer_needs(monkeypatch) -> None:
    """With a per-node output FS every node's local rank 0 writes the WORLD record, so the
    all-gather is what feeds them — narrowing to rank 0 would empty nodes 1..N's parquet."""
    calls = _fake_world(monkeypatch, world=4, rank=2, shared_fs=False)
    host = _MetricsHost()

    host._populate_completion_logs(_rollouts("p2"), torch.tensor([1.0]), torch.tensor([0.25]))

    assert calls["gather_dst"] == 0 and calls["all_gather"] == 4
    assert host._logs["prompt"] == ["peer0", "peer1", "p2", "peer3"]


def test_completion_logs_skipped_entirely_when_no_artifact_is_wanted(monkeypatch) -> None:
    """Neither parquet nor console: no gather at all (the payload is the heaviest object here)."""
    calls = _fake_world(monkeypatch, world=4, rank=0, shared_fs=True)
    host = _MetricsHost(save=False, console=False)

    host._populate_completion_logs(_rollouts("p0"), torch.tensor([1.0]), torch.tensor([0.25]))

    assert calls == {"all_gather": 0, "gather_dst": 0}


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
