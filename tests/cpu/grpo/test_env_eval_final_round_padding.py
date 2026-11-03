#!/usr/bin/env python
"""An eval split's final round is padded to whole batches and the padding is never rolled out or scored. Accelerate's
even-batches shard repeats the split's first rows so all ranks run the same rounds; scored, a repeat counts twice.
At ``num_generations_eval > 1`` accelerate's own ``remainder`` counts the split's rows, not the rows drawn."""

import types
from collections import Counter

import pytest
import torch
from accelerate import PartialState
from accelerate.data_loader import prepare_data_loader
from accelerate.state import GradientState
from torch.utils.data import DataLoader

from halo.environments.episode import RolloutResult
from halo.environments.episode_keys import SOLVE_RATE_KEY
from halo.environments.messages import Message, Trajectory
from halo.trainers.grpo.environmental.step_batch import BatchBuildFence, BatchRows
from halo.trainers.grpo.environmental.trainer import DistributedAsyncEnvironmentalGRPOTrainer
from tests.common.env_grpo_batch import batch_host, row


class _Rollout(str):
    trajectory = None


_Trainer = DistributedAsyncEnvironmentalGRPOTrainer

PartialState()  # accelerate's gather and its loaders read the process state


def _split_rows_host(num_generations_eval: int, num_ranks: int, rank: int):
    host = object.__new__(_Trainer)
    host.accelerator = types.SimpleNamespace(gradient_state=GradientState())
    host.model = types.SimpleNamespace(training=False)
    host.num_generations_eval = num_generations_eval
    host.args = types.SimpleNamespace(seed=42)
    host._dp_metric_gather_scope = (list(range(num_ranks)), num_ranks)
    return host


def _full_pass(num_rows: int, num_generations_eval: int, rows_per_rank: int, num_ranks: int, drop_last=False):
    """Every rank's eval loader as the trainer builds it, iterated in full: per rank, batches drawn and rows kept."""
    split = list(range(num_rows))
    drawn, scored = {}, {}
    for rank in range(num_ranks):
        host = _split_rows_host(num_generations_eval, num_ranks, rank)
        loader = DataLoader(
            split,
            batch_size=rows_per_rank,
            sampler=host._get_eval_sampler(split),
            collate_fn=list,
            drop_last=drop_last,
        )
        drawn[rank], scored[rank] = [], []
        for batch in prepare_data_loader(loader, num_processes=num_ranks, process_index=rank, put_on_device=False):
            drawn[rank].append(batch)
            scored[rank].extend(batch[: host.eval_split_rows(len(batch))])
    return drawn, scored


def _rows(per_rank: dict[int, list]) -> int:
    return sum(len(batch) for batches in per_rank.values() for batch in batches)


@pytest.mark.parametrize(
    ("num_rows", "num_generations_eval", "rows_per_rank", "num_ranks"),
    [(400, 1, 34, 6), (50, 4, 8, 3), (5, 2, 4, 3)],
    ids=["400-rows-on-6-ranks", "groups-of-4", "split-under-one-round"],
)
def test_every_row_the_sampler_draws_is_scored_exactly_once(num_rows, num_generations_eval, rows_per_rank, num_ranks):
    drawn, scored = _full_pass(num_rows, num_generations_eval, rows_per_rank, num_ranks)
    assert len({len(batches) for batches in drawn.values()}) == 1, "every rank must run the same rounds"
    assert _rows(drawn) > num_rows * num_generations_eval, "the final round must have been padded"
    kept = Counter(row for rows in scored.values() for row in rows)
    assert kept == Counter({row: num_generations_eval for row in range(num_rows)})
    assert all(len(rows) % num_generations_eval == 0 for rows in scored.values()), "a group was split"


@pytest.mark.parametrize(
    ("num_rows", "rows_per_rank", "num_ranks", "drop_last"),
    [(48, 8, 6, False), (50, 8, 3, True), (10, 4, 1, False)],
    ids=["whole-rounds", "drop-last", "one-rank-short-tail"],
)
def test_a_pass_without_repeats_keeps_every_row(num_rows, rows_per_rank, num_ranks, drop_last):
    drawn, scored = _full_pass(num_rows, 1, rows_per_rank, num_ranks, drop_last=drop_last)
    assert [row for batch in drawn[0] for row in batch] == scored[0]
    assert sum(len(rows) for rows in scored.values()) == _rows(drawn)


def test_outside_an_eval_loader_every_row_counts():
    assert _split_rows_host(1, 6, 5).eval_split_rows(34) == 34


class _EvalRound:
    """The real round entry; records the rows rolled out and what the batch build and rollout metrics were handed."""

    _collect_and_score_rollouts = _Trainer._collect_and_score_rollouts
    _tokenizer = None  # read only for thinking text, which the stub rollouts carry none of
    _extract_prompts_and_contexts = _Trainer._extract_prompts_and_contexts

    def __init__(self, split_rows: int):
        self.model = types.SimpleNamespace(training=False)
        self.accelerator = types.SimpleNamespace(device=torch.device("cpu"), is_main_process=False)
        # profiling_context reads these on the acquire path; report_to=[] makes it a no-op.
        self.state = types.SimpleNamespace(global_step=0)
        self.args = types.SimpleNamespace(report_to=[])
        self._group_random_effort = False
        self._batch_errors = BatchBuildFence()
        self._prefetch_enabled = False
        self._split_rows = split_rows
        self.rolled_out: list[str] = []
        self._loop = types.SimpleNamespace(run_until_complete=lambda batch: batch)
        self._rollout_manager = types.SimpleNamespace(collect_rollouts=self._collect)

    def eval_split_rows(self, num_rows: int) -> int:
        return self._split_rows

    def _collect(self, prompts, contexts):
        self.rolled_out.extend(prompts)
        return [_Rollout(f"rollout:{p}") for p in prompts]

    def _broadcast_object_from_tp_leader(self, rollout_results):
        return rollout_results

    def _build_training_tensors(self, rollout_results, device, mode, num_padding, reasoning_tokens):
        self.built = (list(rollout_results), mode, num_padding)
        return {}

    def _log_rollout_metrics(self, results, mode, reasoning_tokens):
        self.logged = list(results)


@pytest.mark.parametrize("split_rows", [3, 0], ids=["partial-batch", "all-padding"])
def test_the_eval_round_rolls_out_only_the_split_rows(split_rows):
    host = _EvalRound(split_rows)
    host._collect_and_score_rollouts([{"prompt": f"p{i}"} for i in range(4)])
    episodes = [f"rollout:p{i}" for i in range(split_rows)]
    assert host.rolled_out == [f"p{i}" for i in range(split_rows)]
    assert host.built == (episodes, "eval", 4 - split_rows)
    assert host.logged == episodes


def _episode(prompt: str, reward: float) -> RolloutResult:
    trajectory = Trajectory(messages=[Message.user(prompt), Message.assistant("a")])
    return RolloutResult(prompt=prompt, trajectory=trajectory, total_reward=reward, metrics={SOLVE_RATE_KEY: reward})


def _build_host(num_generations_eval: int):
    """The real batch build in eval mode: one 2-token completion per episode, ``p0``'s an untrainable turn."""
    rows = [row([6, 7], negative_only=True)] + [row([6, 7]) for _ in range(3)]
    return batch_host(
        rows,
        torch.zeros(len(rows), 2),
        training=False,
        num_generations=num_generations_eval,
        scale_rewards="batch",
        std_floor=0.2,
        save_completions=True,
    )


def test_the_batch_keeps_the_padding_inert_and_scores_the_episodes():
    """Read as episodes, the padding's zero rewards would halve the headline reward, shrink the advantage normalizer
    and add a contrast-free group."""
    host = _build_host(num_generations_eval=2)
    episodes = [_episode("p0", 1.0), _episode("p1", 0.0), _episode("p2", 1.0), _episode("p3", 1.0)]
    result = host._build_training_tensors(episodes, torch.device("cpu"), "eval", 2, [[]] * len(episodes))

    assert result["completion_ids"].shape[0] == 6, "the padding keeps the rank's batch its peers' width"
    assert result["completion_mask"][4:].sum() == 0 and result["tool_mask"][4:].sum() == 0
    assert result["num_items_in_batch"].item() == 6, "the eval normalizer counts the episodes' tokens only"
    std = torch.tensor([1.0, 0.0, 1.0, 1.0]).std().item()
    assert result["advantages"].tolist() == pytest.approx([0.5 / std, -0.5 / std, 0, 0, 0, 0], abs=1e-3)

    metrics = host._metrics["eval"]
    assert metrics["reward"] == [pytest.approx(0.75)]
    assert metrics["reward_std"] == [pytest.approx(std)]
    assert metrics["reward/within_group_std"] == [pytest.approx(torch.tensor([1.0, 0.0]).std().item() / 2)]
    assert metrics["sampling/untrainable_rows_frac"] == [pytest.approx(1 / 4)]
    assert metrics["outcome/all_pass_group_frac"] == [pytest.approx(1 / 2)]
    assert list(host._logs["prompt"]) == ["p0", "p1", "p2", "p3"]
    assert list(host._logs["rewards"]["environment_reward"]) == [1.0, 0.0, 1.0, 1.0]


@pytest.mark.parametrize("per_turn", [True, False], ids=["per-turn", "whole-trajectory"])
def test_padding_rows_belong_to_no_rollout(per_turn):
    """A padding row read as a trajectory of its own would sit in every IS mask stage's denominator."""
    turns = [2, 1, 1, 1] if per_turn else [1, 1, 1, 1]
    rows = BatchRows([_episode("p0", 1.0), _episode("p1", 0.0)], turns, 1, per_turn)
    expected = [0, 0, 1, -1, -1, -1] if per_turn else [0, 1, -1, -1, -1]
    assert rows.traj_row_ids(torch.device("cpu")).tolist() == expected


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
