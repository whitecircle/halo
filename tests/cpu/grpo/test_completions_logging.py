#!/usr/bin/env python
"""Unit tests for :func:`halo.trainers.grpo.rollout.completions_logging.emit_completion_artifacts`.

Locks the decoupling contract used by the online + env GRPO trainers: the durable parquet record is
gated by ``save`` (CommonScriptArguments.save_completions) and the rich console table by ``console``
(TRL's log_completions) — independently. Recoupling them, e.g. skipping the parquet unless the
console prints, is the failure this forbids. The record follows the per-node writer election, the
console table global rank 0 alone.

Also pins the environmental trainer's record (``rollout_metrics._populate_completion_logs``): each
rank keeps only its own rows (a TP/ETP follower none) and writes them as its part, ``rank<NN>.parquet``
in the step's directory, gathering nothing for it; only the tables' bounded sample reaches global rank 0
(on a real gloo group). Each row carries its own episode's reward components and scored-term rationale.
WHEN it is called relative to the env-GRPO trust-region breaker is pinned where the breaker lives
(``test_grpo_update_breaker_and_group_effort.py``).

Run::

    python tests/cpu/grpo/test_completions_logging.py
"""

from __future__ import annotations

import datetime
import json
import os
import types
from collections import defaultdict

import pandas as pd
import pytest
import torch

import halo.trainers.grpo.environmental.rollout_metrics as rm
import halo.trainers.grpo.rollout.completions_logging as cl
from halo.environments.episode_keys import REWARD_COMPONENTS_KEY, REWARD_DETAILS_KEY
from halo.environments.messages import Message, Trajectory
from halo.rewards.samples import CUT_CALL_NOTE
from halo.runtime import consensus, filesystem, ranks
from halo.trainers.distributed_trainer import DistributedTrainerMixin
from halo.trainers.grpo.environmental.trainer import DistributedAsyncEnvironmentalGRPOTrainer
from halo.trainers.grpo.on_policy import OnPolicyGRPOInitMixin
from halo.trainers.grpo.online import DistributedGRPOTrainer
from halo.trainers.grpo.rollout.completions_logging import emit_completion_artifacts
from halo.trainers.parallelism.tp_leader import TPLeaderBroadcastMixin
from tests.common.distributed import reset_shared_filesystem_consensus
from tests.common.gloo import run_gloo_ranks


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


def _fresh_logs() -> dict:
    return {
        "prompt": ["p1", "p2"],
        "completion": ["c1", "c2"],
        "rewards": {"environment_reward": [1.0, 0.5]},
        "advantages": [0.4, -0.4],
        "extra": {},
        "images": [],
    }


class _Trainer:
    """Minimal stand-in exposing exactly what emit_completion_artifacts reads: the online trainer's shape, whose
    logs hold the world's rows."""

    RANK_LOCAL_COMPLETION_LOGS = False

    def __init__(self, output_dir, is_main=True, step=7, report_to=None):
        self.accelerator = _Accel(is_main)
        self.state = _State(step)
        self.args = _Args(output_dir, report_to)
        self.model = _Model()
        self.num_completions_to_print = 2
        self.log_unique_prompts = False
        self._logs = _fresh_logs()


def _parquet_path(output_dir, step=7, part=""):
    return os.path.join(output_dir, "completions", f"completions_{step:05d}{part}.parquet")


def _part_path(output_dir, step=9, rank=0, mode=""):
    """A rank's part of a step's record under per-rank completion logs."""
    return os.path.join(output_dir, "completions", f"completions_{step:05d}{mode}", f"rank{rank:02d}.parquet")


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

    The election is ``is_fs_save_rank`` — the same predicate every other output artifact uses —
    not ``accelerator.is_main_process``: on a non-shared output filesystem each node must write its
    own completions, and gating on the accelerator flag would lose nodes 1..N entirely.

    Driven through the real flag chain (``is_global_main_process`` / ``is_local_main_process`` +
    the env flags) rather than by patching ``is_fs_save_rank`` itself, so the test pins WHICH
    predicate is consulted, not merely that some boolean gates the write.
    """
    printed = []
    monkeypatch.setattr(cl, "print_prompt_completions_sample", lambda *a, **k: printed.append(a))
    monkeypatch.setattr(filesystem, "is_global_main_process", lambda: False)
    monkeypatch.setattr(filesystem, "is_local_main_process", lambda: True)
    monkeypatch.delenv("DIST_SHARED_FILESYSTEM", raising=False)
    monkeypatch.delenv("DIST_OUTPUT_SHARED_FILESYSTEM", raising=False)
    reset_shared_filesystem_consensus()

    trainer = _Trainer(str(tmp_path))
    emit_completion_artifacts(trainer, console=True, save=True)

    assert not os.path.exists(_parquet_path(str(tmp_path)))
    assert printed == []


def test_every_node_writes_when_the_output_filesystem_is_not_shared(tmp_path, monkeypatch) -> None:
    """Anti-vacuity for the test above, and the reason the election is per node: with a per-node output
    filesystem this same non-global-main rank IS its node's writer — gating on global rank 0 would
    silently drop that node's completions."""
    monkeypatch.setattr(cl, "print_prompt_completions_sample", lambda *a, **k: None)
    monkeypatch.setattr(filesystem, "is_global_main_process", lambda: False)
    monkeypatch.setattr(filesystem, "is_local_main_process", lambda: True)
    monkeypatch.setenv("DIST_OUTPUT_SHARED_FILESYSTEM", "0")
    reset_shared_filesystem_consensus()

    trainer = _Trainer(str(tmp_path))
    emit_completion_artifacts(trainer, console=False, save=True)

    assert os.path.exists(_parquet_path(str(tmp_path))), "a per-node output FS lost this node's completions"


def test_a_per_node_writer_writes_its_record_but_leaves_the_console_table_to_rank_zero(tmp_path, monkeypatch) -> None:
    """The writer election is per node on a per-node output filesystem; the console is not: a table
    printed by every node's writer would reach the log once per node."""
    printed = []
    monkeypatch.setattr(cl, "print_prompt_completions_sample", lambda *a, **k: printed.append(a))
    monkeypatch.setattr(cl, "is_rich_available", lambda: True)
    monkeypatch.setattr(ranks, "get_global_rank", lambda: 8)  # the second node's local rank 0
    monkeypatch.setattr(filesystem, "is_local_main_process", lambda: True)
    monkeypatch.setenv("DIST_OUTPUT_SHARED_FILESYSTEM", "0")
    reset_shared_filesystem_consensus()

    emit_completion_artifacts(_Trainer(str(tmp_path)), console=True, save=True)

    assert os.path.exists(_parquet_path(str(tmp_path))), "the node's writer still owes its parquet"
    assert printed == [], "the console table printed on a rank other than global rank 0"


def test_eval_mode_writes_suffixed_parquet(tmp_path, monkeypatch) -> None:
    """model.training=False: the parquet gets the ``_eval`` suffix and never overwrites the train file."""
    monkeypatch.setattr(cl, "print_prompt_completions_sample", lambda *a, **k: None)

    trainer = _Trainer(str(tmp_path))
    emit_completion_artifacts(trainer, console=False, save=True)
    trainer.model.training = False
    trainer._logs = _fresh_logs()
    trainer._logs["completion"] = ["e1", "e2"]
    emit_completion_artifacts(trainer, console=False, save=True)

    train_df = pd.read_parquet(_parquet_path(str(tmp_path)))
    assert list(train_df["completion"]) == ["c1", "c2"], "eval write must not overwrite the train parquet"
    eval_path = os.path.join(str(tmp_path), "completions", "completions_00007_eval.parquet")
    assert os.path.exists(eval_path), "eval-mode write must land in the _eval-suffixed parquet"
    assert list(pd.read_parquet(eval_path)["completion"]) == ["e1", "e2"]


def _buffer_lengths(logs) -> set[int]:
    return {len(v) for slot in logs.values() for v in (slot.values() if isinstance(slot, dict) else (slot,))}


def test_emit_empties_the_buffer_so_the_next_log_writes_only_its_own_rows(tmp_path, monkeypatch) -> None:
    """The async trainer's buffers are unbounded (an eval round is the whole eval set), so a write
    must consume them: rows would otherwise repeat in every later parquet and grow for the run."""
    monkeypatch.setattr(cl, "print_prompt_completions_sample", lambda *a, **k: None)
    trainer = _Trainer(str(tmp_path))
    emit_completion_artifacts(trainer, console=False, save=True)
    assert _buffer_lengths(trainer._logs) == {0}, "every slot, rewards and extras included, is emptied"

    trainer.state.global_step = 8
    emit_completion_artifacts(trainer, console=False, save=True)
    assert not os.path.exists(_parquet_path(str(tmp_path), step=8)), "an emptied buffer writes no second file"


def test_a_non_writer_rank_empties_its_buffer_too(tmp_path, monkeypatch) -> None:
    """Non-writers hold nothing useful after the step; left alone, their unbounded slots would keep
    every row the online trainer's generation path appends on every process."""
    monkeypatch.setattr(cl, "print_prompt_completions_sample", lambda *a, **k: None)
    monkeypatch.setattr(filesystem, "is_global_main_process", lambda: False)
    monkeypatch.setattr(filesystem, "is_local_main_process", lambda: True)
    monkeypatch.delenv("DIST_SHARED_FILESYSTEM", raising=False)
    monkeypatch.delenv("DIST_OUTPUT_SHARED_FILESYSTEM", raising=False)
    reset_shared_filesystem_consensus()

    trainer = _Trainer(str(tmp_path))
    emit_completion_artifacts(trainer, console=False, save=True)
    assert not os.path.exists(_parquet_path(str(tmp_path)))
    assert _buffer_lengths(trainer._logs) == {0}


def test_unbounded_completion_logs_hold_a_round_larger_than_one_generation_batch() -> None:
    """TRL caps each slot at ``generation_batch_size``, which keeps only the tail of an eval round
    larger than one generation batch; the async trainer's buffers take the whole round."""
    logs = cl.unbounded_completion_logs()
    logs["prompt"].extend(f"p{i}" for i in range(432))
    logs["rewards"]["environment_reward"].extend(range(432))
    assert len(logs["prompt"]) == 432 and len(logs["rewards"]["environment_reward"]) == 432
    assert all(slot.maxlen is None for slot in (logs["prompt"], logs["completion"], logs["advantages"]))
    assert set(logs) == {"images", "prompt", "completion", "rewards", "advantages", "extra"}, "TRL's slot layout"


def test_empty_logs_is_noop(tmp_path) -> None:
    """Nothing generated yet (empty buffer): no parquet, no crash."""
    trainer = _Trainer(str(tmp_path))
    trainer._logs["prompt"] = []
    emit_completion_artifacts(trainer, console=True, save=True)
    assert not os.path.exists(_parquet_path(str(tmp_path)))


class _Parallelism:
    is_tp_mode = False
    is_expert_tp_mode = False


class _MetricsHost(rm.RolloutMetricsMixin, TPLeaderBroadcastMixin):
    """Minimal stand-in exposing exactly what ``_populate_completion_logs`` reads."""

    def __init__(self, save=True, console=False):
        self._save_completions = save
        self.log_completions = console
        self.parallelism_config = _Parallelism()
        self._logs = {
            "prompt": [],
            "completion": [],
            "rewards": {"environment_reward": []},
            "advantages": [],
            "extra": defaultdict(list),
        }


def _rollouts(*prompts):
    return [types.SimpleNamespace(prompt=p, trajectory=None) for p in prompts]


def _fake_world(monkeypatch, *, world: int, rank: int, shared_fs: bool) -> None:
    """Simulate rank ``rank`` of a ``world``-rank job whose every gather raises: the record must never gather
    another rank's trajectories. The write's verdict is taken on this rank alone."""

    def refuse(*args, **kwargs):
        raise AssertionError("the completions record gathered across ranks")

    for name in ("gather_object", "all_gather_object", "broadcast_object_list", "all_gather"):
        monkeypatch.setattr(torch.distributed, name, refuse)

    def local_verdict(reason, what, exc_type=RuntimeError):
        if reason:
            raise exc_type(reason)

    monkeypatch.setattr(consensus, "reject_across_ranks", local_verdict)
    monkeypatch.setattr(ranks.dist, "is_available", lambda: True)
    monkeypatch.setattr(ranks.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(ranks.dist, "get_world_size", lambda *a, **k: world)
    monkeypatch.setattr(ranks.dist, "get_rank", lambda *a, **k: rank)
    monkeypatch.setattr(filesystem, "is_local_main_process", lambda: rank % 8 == 0)
    monkeypatch.setenv("DIST_OUTPUT_SHARED_FILESYSTEM", "1" if shared_fs else "0")
    reset_shared_filesystem_consensus()


def _logging_host(tmp_path, step=9) -> _MetricsHost:
    host = _MetricsHost()
    host.args, host.state, host.model = _Args(str(tmp_path)), _State(step=step), _Model(training=True)
    host.num_completions_to_print, host.log_unique_prompts = 2, False
    host._logs = cl.unbounded_completion_logs()
    return host


@pytest.mark.parametrize("shared_fs", [True, False], ids=["shared", "per-node"])
def test_every_rank_keeps_and_writes_only_its_own_rows_with_no_collective(tmp_path, monkeypatch, shared_fs):
    """The rendered trajectories are the heaviest payload the trainer holds: gathered, they stage the world's
    text through a device and, per node, leave every writer holding every rank's record. Each rank, the
    filesystem save ranks or not, keeps its own rows and writes them as its own part."""
    _fake_world(monkeypatch, world=16, rank=9, shared_fs=shared_fs)
    host = _logging_host(tmp_path)

    host._populate_completion_logs(_rollouts("p9"), torch.tensor([1.0]), torch.tensor([0.25]), "train")
    assert list(host._logs["prompt"]) == ["p9"], "a rank holds another rank's rows"
    cl.emit_completion_artifacts(host, console=False, save=True, mode="train")

    assert os.listdir(tmp_path / "completions") == ["completions_00009"], "one directory per step"
    assert os.listdir(tmp_path / "completions" / "completions_00009") == ["rank09.parquet"]
    assert list(pd.read_parquet(_part_path(str(tmp_path), rank=9))["prompt"]) == ["p9"]
    assert _buffer_lengths(host._logs) == {0}


def test_the_ranks_parts_together_are_the_steps_record(tmp_path, monkeypatch):
    """Parts of one step never collide, and the step's directory reads back as every rank's rows."""
    for rank in (0, 3):
        _fake_world(monkeypatch, world=4, rank=rank, shared_fs=True)
        host = _logging_host(tmp_path)
        host._populate_completion_logs(_rollouts(f"p{rank}"), torch.tensor([1.0]), torch.tensor([0.0]), "train")
        cl.emit_completion_artifacts(host, console=False, save=True, mode="train")

    step_dir = tmp_path / "completions" / "completions_00009"
    assert sorted(part.name for part in step_dir.iterdir()) == ["rank00.parquet", "rank03.parquet"]
    assert sorted(pd.read_parquet(step_dir)["prompt"]) == ["p0", "p3"]


def test_a_tp_follower_keeps_no_rows_so_each_episode_is_recorded_once(tmp_path, monkeypatch):
    """Under TP or ETP every rank of a group holds its leader's broadcast rollouts: kept by all, each episode
    would land in the record once per group rank. 4 ranks in TP groups of 2: one part per data-parallel rank."""
    for rank in range(4):
        _fake_world(monkeypatch, world=4, rank=rank, shared_fs=True)
        host = _logging_host(tmp_path)
        host.parallelism_config.is_tp_mode = True
        host._get_tp_rank = lambda rank=rank: rank % 2
        host._get_tp_or_etp_process_group = object
        leader = rank - rank % 2
        host._populate_completion_logs(_rollouts(f"p{leader}"), torch.tensor([1.0]), torch.tensor([0.0]), "train")
        cl.emit_completion_artifacts(host, console=False, save=True, mode="train")

    step_dir = tmp_path / "completions" / "completions_00009"
    assert sorted(part.name for part in step_dir.iterdir()) == ["rank00.parquet", "rank02.parquet"]
    assert sorted(pd.read_parquet(step_dir)["prompt"]) == ["p0", "p2"], "an episode was recorded per TP rank"


_SAMPLE_WORLD = 3
_ROWS_PER_RANK = 5
_LONG = "x" * (cl._TABLE_CELL_CHARS + 100)


class _FakeWandb:
    """What the wandb table path touches: a live run, ``Table`` and ``log``."""

    run = object()

    def __init__(self):
        self.logged = []

    @staticmethod
    def Table(dataframe):
        return dataframe

    def log(self, payload):
        self.logged.append(payload)


def _sample_worker(rank: int, root: str) -> None:
    printed = []
    cl.print_prompt_completions_sample = lambda prompts, completions, *a, **k: printed.append(
        {"prompts": prompts, "completions": completions, "num_samples": a[3]}
    )
    cl.is_rich_available = lambda: True
    fake_wandb = _FakeWandb()
    cl.wandb = fake_wandb
    host = _logging_host(root)
    host.args.report_to = ["wandb"]
    host.num_completions_to_print = 4
    rows = [types.SimpleNamespace(prompt=f"p{rank}-{i}", trajectory=None) for i in range(_ROWS_PER_RANK)]
    host._render_trajectory_for_log = lambda trajectory: _LONG
    host._populate_completion_logs(rows, torch.zeros(_ROWS_PER_RANK), torch.zeros(_ROWS_PER_RANK), "train")
    cl.emit_completion_artifacts(host, console=True, save=True, mode="train")
    tables = [
        {
            "prompts": list(entry["completions"]["prompt"]),
            "cells": sorted({len(c) for c in entry["completions"]["completion"]}),
        }
        for entry in fake_wandb.logged
    ]
    with open(os.path.join(root, f"tables_{rank}.json"), "w") as fh:
        json.dump(
            {
                "printed": [{**p, "completions": sorted({len(c) for c in p["completions"]})} for p in printed],
                "wandb": tables,
            },
            fh,
        )


def test_the_tables_show_a_bounded_sample_of_every_ranks_rows_on_rank_zero(tmp_path):
    """The console and wandb tables cover the world, not rank 0 alone, yet each rank sends only its share of
    ``num_completions_to_print`` (ceil(4 / 3) = 2 of its 5 rows), cut to the table's cell width: the full text
    and every row stay in the rank's own part."""
    run_gloo_ranks(_sample_worker, _SAMPLE_WORLD, str(tmp_path), pg_timeout=datetime.timedelta(seconds=60))
    results = [json.loads((tmp_path / f"tables_{rank}.json").read_text()) for rank in range(_SAMPLE_WORLD)]

    sample = [f"p{rank}-{i}" for rank in range(_SAMPLE_WORLD) for i in (0, 2)]
    cut = len(cl._cell(_LONG))
    assert results[0]["printed"] == [{"prompts": sample, "completions": [cut], "num_samples": 4}], results[0]
    assert results[0]["wandb"] == [{"prompts": sample, "cells": [cut]}], results[0]
    assert all(result == {"printed": [], "wandb": []} for result in results[1:]), "a table left global rank 0"
    for rank in range(_SAMPLE_WORLD):
        part = pd.read_parquet(_part_path(str(tmp_path), rank=rank))
        assert list(part["prompt"]) == [f"p{rank}-{i}" for i in range(_ROWS_PER_RANK)]
        assert set(part["completion"]) == {_LONG}, "the record keeps the full text"


def test_each_row_carries_its_episodes_reward_components_and_scored_term_rationale(tmp_path, monkeypatch) -> None:
    """The parquet row of an episode holds that episode's settled components and every scored term's
    rationale (a veto judge's fired checks with their quotes) as JSON, aligned with its prompt: a vetoed
    solve is told from an honest failure on the record alone. An episode that kept neither (a lost
    rollout, no scored term) records ``{}``."""
    monkeypatch.setattr(cl, "print_prompt_completions_sample", lambda *a, **k: None)
    host = _MetricsHost()
    host.args, host.state, host.model = _Args(str(tmp_path)), _State(step=9), _Model(training=True)
    host.num_completions_to_print, host.log_unique_prompts = 2, False
    host._logs = cl.unbounded_completion_logs()
    vetoed = Trajectory(messages=[Message.user("task"), Message.assistant("print(42)")])
    vetoed.info[REWARD_COMPONENTS_KEY] = {
        "reward/objective": 0.0,
        "reward/submission": 0.0,
        "reward/resubmission": -0.2,
    }
    vetoed.info[REWARD_DETAILS_KEY] = {"audit": "fired hardcoded_output: 'print(42)'"}
    solved = Trajectory(messages=[Message.user("task"), Message.assistant("solve()")])
    solved.info[REWARD_COMPONENTS_KEY] = {"reward/objective": 1.0, "reward/submission": 0.1}
    rollouts = [
        types.SimpleNamespace(prompt="vetoed", trajectory=vetoed),
        types.SimpleNamespace(prompt="solved", trajectory=solved),
        types.SimpleNamespace(prompt="lost", trajectory=None),
    ]

    host._populate_completion_logs(rollouts, torch.tensor([-0.2, 1.1, 0.0]), torch.tensor([-1.0, 1.0, 0.0]), "train")
    emit_completion_artifacts(host, console=False, save=True, mode="train")

    df = pd.read_parquet(_part_path(str(tmp_path)))
    assert list(df["prompt"]) == ["vetoed", "solved", "lost"]
    assert [json.loads(cell) for cell in df[REWARD_COMPONENTS_KEY]] == [
        vetoed.info[REWARD_COMPONENTS_KEY],
        solved.info[REWARD_COMPONENTS_KEY],
        {},
    ]
    assert [json.loads(cell) for cell in df[REWARD_DETAILS_KEY]] == [vetoed.info[REWARD_DETAILS_KEY], {}, {}]


def test_the_record_shows_the_call_a_cut_turn_was_writing_marked_as_never_run() -> None:
    """The completion column renders a turn cut inside its call with the partial call it was writing, tagged
    as never run, after the calls that did; a turn without one renders as before."""
    ran = {"id": "c0", "function": {"name": "run_code", "arguments": '{"code": "print(1)"}'}}
    cut = {"id": "c1", "function": {"name": "run_code", "arguments": '{"code": "# still reasoning about n'}}
    trajectory = Trajectory(
        messages=[
            Message.user("task"),
            Message.assistant("trying", tool_calls=[ran]),
            Message.tool("1", tool_call_id="c0", name="run_code"),
            Message.assistant("again", truncated=True, cut_tool_calls=[cut]),
        ]
    )
    assert rm.RolloutMetricsMixin._render_trajectory_for_log(trajectory) == (
        "[user] task\n"
        '[assistant] trying\n  <tool_call run_code> {"code": "print(1)"}\n'
        "[tool] 1\n"
        f'[assistant] again\n  <tool_call run_code ({CUT_CALL_NOTE})> {{"code": "# still reasoning about n'
    )


def test_a_mode_switch_writes_the_pending_rows_under_their_own_mode(tmp_path, monkeypatch) -> None:
    """With ``logging_steps > 1`` an eval round can begin before the train rows since the last log were
    written; they must reach the train parquet, not the eval one, and never disappear."""
    monkeypatch.setattr(cl, "print_prompt_completions_sample", lambda *a, **k: None)
    host = _MetricsHost()
    host.args, host.state, host.model = _Args(str(tmp_path)), _State(step=9), _Model(training=False)
    host.num_completions_to_print, host.log_unique_prompts = 2, False
    host._logs = cl.unbounded_completion_logs()

    host._populate_completion_logs(_rollouts("train-row"), torch.tensor([1.0]), torch.tensor([0.0]), "train")
    host._populate_completion_logs(_rollouts("eval-row"), torch.tensor([0.0]), torch.tensor([0.0]), "eval")

    train_df = pd.read_parquet(_part_path(str(tmp_path)))
    assert list(train_df["prompt"]) == ["train-row"], "the pending train rows land in the train file"
    assert not os.path.exists(_part_path(str(tmp_path), mode="_eval"))
    assert list(host._logs["prompt"]) == ["eval-row"], "the eval round's own rows wait for the eval log"


def test_completion_logs_skipped_entirely_when_no_artifact_is_wanted(tmp_path) -> None:
    """Neither parquet nor console: the rows are not even rendered."""
    host = _logging_host(tmp_path)
    host._save_completions = host.log_completions = False

    host._populate_completion_logs(_rollouts("p0"), torch.tensor([1.0]), torch.tensor([0.25]), "train")

    assert _buffer_lengths(host._logs) == {0}


class _BaseLog:
    """Stands for the rest of the MRO the mixin delegates to: records TRL's flag as ``log`` saw it."""

    def log(self, logs, start_time=None):
        self.seen_by_base = (dict(logs), self.log_completions)


class _LogHost(cl.DecoupledCompletionsLogMixin, _BaseLog):
    def __init__(self, tmp_path, *, console: bool, save: bool):
        self.log_completions, self._save_completions = console, save
        self.args, self.state, self.model = _Args(str(tmp_path)), _State(step=7), _Model()
        self.num_completions_to_print, self.log_unique_prompts = 2, False
        self._logs = _fresh_logs()


@pytest.mark.parametrize(("console", "save"), [(True, False), (False, True)])
def test_the_log_mixin_keeps_trls_block_off_and_emits_the_decoupled_artifacts(tmp_path, monkeypatch, console, save):
    printed = []
    monkeypatch.setattr(cl, "print_prompt_completions_sample", lambda *a, **k: printed.append(a))
    monkeypatch.setattr(cl, "is_rich_available", lambda: True)
    host = _LogHost(tmp_path, console=console, save=save)

    host.log({"loss": 1.0})

    assert host.seen_by_base == ({"loss": 1.0}, False), "TRL's coupled completions block ran"
    assert host.log_completions is console, "TRL's flag was not restored after the delegated log"
    assert os.path.exists(_parquet_path(str(tmp_path))) is save
    assert len(printed) == int(console)


def test_both_on_policy_trainers_log_through_the_mixin_ahead_of_the_parallelism_logs():
    for cls in (DistributedGRPOTrainer, DistributedAsyncEnvironmentalGRPOTrainer):
        mro = cls.__mro__
        assert mro.index(cl.DecoupledCompletionsLogMixin) < mro.index(DistributedTrainerMixin), cls.__name__


def test_the_init_spine_takes_save_completions_before_trls_ctor_sees_it():
    host = types.SimpleNamespace(_init_distributed_config=lambda kwargs, **_: kwargs)
    kwargs = {"save_completions": False, "args": None}
    _, remaining = OnPolicyGRPOInitMixin._begin_on_policy_init(host, (), kwargs)
    assert host._save_completions is False and "save_completions" not in remaining


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
