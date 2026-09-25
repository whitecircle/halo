#!/usr/bin/env python
"""The environmental GRPO step diagnostics, each pinned to its value on a hand-computed batch.

They change no numerics, so a wrong one only misleads: a solve-group split that counts invalid
episodes, an effective sample size that skips the masked weights, a covariance whose sign flips, a
truncation alarm that never fires or fires every round. Each is folded across ranks from sums, so the
world value is checked with peers standing in for the other ranks.

    python tests/cpu/grpo/test_env_grpo_step_diagnostics.py
"""

import logging
import math
import types
from collections import defaultdict

import pytest
import torch

import src.trainers.grpo.rollout.rollout_metrics as rm
from src.configs.async_training_config import AsyncTrainingConfig
from src.environments.base import SOLVE_RATE_KEY, Trajectory
from src.environments.episode import RolloutResult
from src.trainers.grpo.environmental import BatchRows, DistributedAsyncEnvironmentalGRPOTrainer
from src.trainers.grpo.rollout.rollout_metrics import WorldMetrics, group_solve_counts
from tests.common.grpo_metrics import attach_world_metrics, flushed_metrics

_Trainer = DistributedAsyncEnvironmentalGRPOTrainer


def _fold(world: WorldMetrics, *peers: WorldMetrics) -> dict[str, list[float]]:
    target = defaultdict(list)
    world.flush(target, gather_fn=lambda own: [*own, *(peer._materialized() for peer in peers)])
    return target


def test_the_effective_sample_frac_folds_the_world_weights():
    # Pooled weights 1, 3, 0 (masked to zero, still a sample), 2; one weight outside the mask is ignored.
    rank0, rank1 = WorldMetrics(), WorldMetrics()
    rank0.effective_sample_frac("ess", torch.tensor([1.0, 3.0, 0.0, 9.0]), torch.tensor([True, True, True, False]))
    rank1.effective_sample_frac("ess", torch.tensor([2.0]), torch.tensor([True]))
    assert _fold(rank0, rank1)["ess"] == [pytest.approx(36 / (4 * 14))]


def test_uniform_weights_are_a_full_sample():
    world = WorldMetrics()
    world.effective_sample_frac("ess", torch.full((5,), 0.7), torch.ones(5, dtype=torch.bool))
    assert _fold(world)["ess"] == [pytest.approx(1.0)]


def test_the_covariance_pools_the_ranks_and_ignores_what_the_mask_drops():
    # Pooled x = 1, 2, 3 and y = 0, 1, 1: E[xy] = 5/3, E[x] E[y] = 2 · 2/3, so 1/3. The masked infinity
    # would turn a masked product into NaN.
    rank0, rank1 = WorldMetrics(), WorldMetrics()
    rank0.covariance(
        "cov", torch.tensor([1.0, 2.0, math.inf]), torch.tensor([0.0, 1.0, 5.0]), torch.tensor([True, True, False])
    )
    rank1.covariance("cov", torch.tensor([3.0]), torch.tensor([1.0]), torch.tensor([True]))
    assert _fold(rank0, rank1)["cov"] == [pytest.approx(1 / 3)]


def test_group_solve_counts_judge_each_group_on_its_verdicts():
    solved = [True, True, True, False, False, None, True, False, True, None, None, None]
    assert group_solve_counts(solved, 3) == (3, 1, 1, 2)


def _episode(solve: float | None, *, error: str | None = None) -> RolloutResult:
    return RolloutResult(
        prompt="p",
        trajectory=Trajectory(done=True),
        error=error,
        metrics={} if solve is None else {SOLVE_RATE_KEY: solve},
    )


def _diagnostics_host():
    return attach_world_metrics(
        types.SimpleNamespace(_metrics={"train": defaultdict(list), "eval": defaultdict(list)})
    )


def _groups_of_two():
    # All solved | none solved (the errored member's forced 1.0 does not vote) | split.
    return [
        _episode(1.0),
        _episode(1.0),
        _episode(0.0),
        _episode(1.0, error="TimeoutError"),
        _episode(1.0),
        _episode(0.0),
    ]


def test_an_eval_round_logs_the_group_split_and_success_at_k():
    host = _diagnostics_host()
    _Trainer._record_step_diagnostics(host, _groups_of_two(), 2, "eval", None, torch.zeros(6), torch.ones(6, 1))
    logged = flushed_metrics(host, "eval")
    assert logged["outcome/all_pass_group_frac"] == [pytest.approx(1 / 3)]
    assert logged["outcome/all_fail_group_frac"] == [pytest.approx(1 / 3)]
    assert logged["outcome/success@2"] == [pytest.approx(2 / 3)]


def test_a_training_round_logs_no_success_at_k_and_an_env_without_a_verdict_logs_no_split():
    host = _diagnostics_host()
    _Trainer._record_step_diagnostics(host, _groups_of_two(), 2, "train", None, torch.zeros(6), torch.ones(6, 1))
    assert not any(key.startswith("outcome/success@") for key in flushed_metrics(host))

    silent = _diagnostics_host()
    episodes = [_episode(None) for _ in range(4)]
    _Trainer._record_step_diagnostics(silent, episodes, 2, "train", None, torch.zeros(4), torch.ones(4, 1))
    assert not flushed_metrics(silent)


def test_the_log_prob_advantage_covariance_reads_the_loss_tokens():
    # Row 0 (advantage +1) is confident, row 1 (advantage -1) is not: the policy is sharpening where it is
    # rewarded, a positive covariance. Loss tokens: -0.1, -0.3 at +1 and -2.0 at -1; the padded tail and
    # the masked tool token are left out. E[xy] = (-0.4 + 2.0) / 3, E[x] = -0.8, E[y] = 1/3.
    logps = torch.tensor([[-0.1, -0.3, -9.0], [-2.0, -5.0, 0.0]])
    loss_mask = torch.tensor([[True, True, False], [True, False, False]])
    host = _diagnostics_host()
    _Trainer._record_step_diagnostics(host, [], 2, "train", logps, torch.tensor([1.0, -1.0]), loss_mask)
    assert flushed_metrics(host)["logps/advantage_cov"] == [pytest.approx((-0.4 + 2.0) / 3 - (-0.8) * (1 / 3))]


def _is_host(clip_max: float):
    return attach_world_metrics(
        types.SimpleNamespace(
            vllm_importance_sampling_clip_max=clip_max,
            _is_mask_config=types.SimpleNamespace(any_mask_active=False),
            _metrics={"train": defaultdict(list)},
        )
    )


def _two_rows():
    # Sampled at log-prob -1 everywhere; the trainer's ratios are 1, 4, 1/4 and 2, 1, 1.
    sampling = [torch.full((3,), -1.0), torch.full((3,), -1.0)]
    recompute = torch.tensor([[0.0, math.log(4), -math.log(4)], [math.log(2), 0.0, 0.0]]) - 1.0
    rows = BatchRows([_episode(None), _episode(None)], [1, 1], 0, True)
    return sampling, recompute, rows


def test_extreme_ratios_are_those_past_the_truncation_point_either_way():
    host = _is_host(clip_max=3.0)
    sampling, recompute, rows = _two_rows()
    ratio, _, corrected, _ = _Trainer._apply_is_correction(
        host, True, recompute, sampling, [True, True], torch.ones(2, 3, dtype=torch.bool), rows, torch.device("cpu")
    )
    _Trainer._score_is_correction(host, ratio, corrected, torch.ones(2, 3, dtype=torch.bool))
    logged = flushed_metrics(host)

    assert logged["sampling/is_ratio_extreme_frac"] == [pytest.approx(2 / 6)], "4 and 1/4 are past 3 and 1/3"
    # The applied weights are the truncated ratios 1, 3, 1/4, 2, 1, 1.
    applied = [1.0, 3.0, 0.25, 2.0, 1.0, 1.0]
    ess = sum(applied) ** 2 / (len(applied) * sum(w * w for w in applied))
    assert logged["sampling/is_ess_frac"] == [pytest.approx(ess, rel=1e-5)]


def test_a_truncation_point_at_one_leaves_no_band_and_logs_no_extreme_fraction():
    host = _is_host(clip_max=1.0)
    sampling, recompute, rows = _two_rows()
    _Trainer._apply_is_correction(
        host, True, recompute, sampling, [True, True], torch.ones(2, 3, dtype=torch.bool), rows, torch.device("cpu")
    )
    assert "sampling/is_ratio_extreme_frac" not in flushed_metrics(host)


class _RoundHost(rm.RolloutMetricsMixin):
    def __init__(self, alarm_rate, *, mask_truncated: bool = False):
        self._metrics = {"train": defaultdict(list), "eval": defaultdict(list)}
        self._truncation_alarm_rate = alarm_rate
        self.args = types.SimpleNamespace(mask_truncated_completions=mask_truncated)


def _round(truncated: int, total: int = 4):
    return [
        types.SimpleNamespace(
            latency=1.0,
            generation_tokens=1,
            requests_expired_in_sync=0,
            episode_length=1,
            success=i >= truncated,
            trajectory=Trajectory(done=True, truncated=i < truncated),
            error=None,
            total_reward=0.0,
            metrics={},
        )
        for i in range(total)
    ]


def test_the_truncation_alarm_fires_on_a_crossing_and_logs_every_round(monkeypatch, caplog):
    monkeypatch.setattr(rm, "gather_object", lambda values: list(values))
    host = _RoundHost(alarm_rate=0.25)
    with caplog.at_level(logging.WARNING, logger=rm.__name__):
        for truncated in (2, 3, 0, 2):
            host._log_rollout_metrics(_round(truncated), "train")

    assert host._metrics["train"]["episode/truncation_alarm"] == [1.0, 1.0, 0.0, 1.0]
    warnings = [r for r in caplog.records if "truncation_alarm_rate" in r.getMessage()]
    assert len(warnings) == 2, "warned on each crossing, not on every round over the line"
    assert "50%" in warnings[0].getMessage()


@pytest.mark.parametrize(
    ("mask_truncated", "said", "unsaid"),
    [
        (False, "priced like a failure", "drops those episodes from the loss"),
        (True, "mask_truncated_completions drops those episodes from the loss", "priced like a failure"),
    ],
    ids=["kept", "masked"],
)
def test_the_truncation_warning_states_what_the_loss_does_with_a_truncated_episode(
    monkeypatch, caplog, mask_truncated, said, unsaid
):
    monkeypatch.setattr(rm, "gather_object", lambda values: list(values))
    host = _RoundHost(alarm_rate=0.25, mask_truncated=mask_truncated)
    with caplog.at_level(logging.WARNING, logger=rm.__name__):
        host._log_rollout_metrics(_round(2), "train")
    (warning,) = [r.getMessage() for r in caplog.records if "truncation_alarm_rate" in r.getMessage()]
    assert said in warning and unsaid not in warning


@pytest.mark.parametrize("rate", [1.0, -0.1])
def test_a_truncation_alarm_rate_that_could_never_or_always_fire_is_refused(rate):
    with pytest.raises(ValueError, match="truncation_alarm_rate"):
        AsyncTrainingConfig(truncation_alarm_rate=rate)


def test_the_truncation_alarm_off_logs_nothing(monkeypatch, caplog):
    monkeypatch.setattr(rm, "gather_object", lambda values: list(values))
    host = _RoundHost(alarm_rate=None)
    with caplog.at_level(logging.WARNING, logger=rm.__name__):
        host._log_rollout_metrics(_round(4), "train")
    assert "episode/truncation_alarm" not in host._metrics["train"]
    assert not caplog.records


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
