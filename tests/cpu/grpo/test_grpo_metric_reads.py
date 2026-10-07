#!/usr/bin/env python
"""GRPO step metrics read the tokens and the ranks they describe.

* ``kl_clamp_frac`` counts the k3 cap over the loss mask the KL term is averaged on (TRL's ``kl``:
  completion mask times tool mask, after the drops), on both on-policy trainers: a cap biting on
  padding or on a tool-output token is no KL the loss trains, so it is in neither side of the ratio.
* ``sampling/forced_close_frac`` is over the tokens that carry sampling log-probs: a fallback row
  without them can never hold a forced close, and its tool tokens only dilute the share.
* ``routing/replay_flip_rate`` and the ``async/prefetch_*`` counters are world totals, a rank with no
  count of its own sending zeros, read on a real 2-rank gloo group.

    python tests/cpu/grpo/test_grpo_metric_reads.py
"""

import datetime
import json
import os
import types
from collections import defaultdict

import pytest
import torch
from accelerate import PartialState

from src.trainers.grpo.objective.logratio import KL_CLAMP_FRAC_KEY, KL_LOGRATIO_CLAMP, kl_clamp_counts
from src.trainers.grpo.online import DistributedGRPOTrainer
from src.trainers.grpo.rollout.async_rollouts import AsyncRolloutMixin
from src.trainers.grpo.rollout.routing_replay import RoutingReplayInjector
from src.trainers.grpo.rollout.trajectory_tokenize import TurnRow
from tests.common.env_grpo_batch import batch_host, build, episode, row
from tests.common.gloo import run_gloo_ranks
from tests.common.routing import BareEPLayer

PartialState()  # the world-metrics flush gathers through accelerate, which reads the process state

# Row 0 is three completion tokens, the middle one tool output; row 1 is one token right-padded to three.
_LOSS_MASK = torch.tensor([[1, 0, 1], [1, 0, 0]])
# The reference sits far above the policy on row 0's first token (trained), its tool token and row 1's
# padding: the cap bites four times, once on a token the KL term averages over.
_OVER_THE_CAP = torch.tensor([[1.0, 1.0, 0.0], [0.0, 1.0, 1.0]]) * (KL_LOGRATIO_CLAMP + 5.0)
_TRAINED_SHARE = 1 / 3


def test_the_clamp_counts_read_the_loss_tokens_alone():
    clamped = _OVER_THE_CAP > KL_LOGRATIO_CLAMP
    assert [int(v) for v in kl_clamp_counts(clamped, _LOSS_MASK)] == [1, 3]


def test_online_kl_clamp_frac_reads_the_post_drop_loss_mask():
    host = types.SimpleNamespace(
        model=types.SimpleNamespace(training=True),
        accelerator=types.SimpleNamespace(gather=lambda x: x),
        _metrics={"train": defaultdict(list)},
    )
    result = {
        "completion_mask": torch.tensor([[1, 1, 1], [1, 0, 0]]),
        "tool_mask": torch.tensor([[1, 0, 1], [1, 1, 1]]),
        "old_per_token_logps": torch.zeros(2, 3),
        "ref_per_token_logps": _OVER_THE_CAP.clone(),
    }
    DistributedGRPOTrainer._clamp_kl_reference(host, result)
    assert host._metrics["train"][KL_CLAMP_FRAC_KEY] == [pytest.approx(_TRAINED_SHARE)]
    assert result["ref_per_token_logps"].max().item() == pytest.approx(KL_LOGRATIO_CLAMP), "the cap still applies"


_FORCED_CLOSE = 77


def _eval_build(rows: list[TurnRow], *, beta: float, is_correction: bool, policy: torch.Tensor):
    """The real eval-mode batch build over ``rows``, the policy's log-probs ``policy`` and a reference over
    the cap on the tokens ``_OVER_THE_CAP`` marks; returns the host whose metrics it logged."""
    host = batch_host(
        rows,
        policy,
        training=False,
        scale_rewards="batch",
        is_correction=is_correction,
        beta=beta,
        reference=policy + _OVER_THE_CAP if beta else None,
        forced_close_ids=(_FORCED_CLOSE,),
    )
    build(host, [episode(1.0), episode(0.0)])
    return host


def test_env_kl_clamp_frac_reads_the_loss_tokens():
    rows = [row([6, 7, 8], loss_mask=[1, 0, 1]), row([6])]
    host = _eval_build(rows, beta=0.04, is_correction=False, policy=torch.full((2, 3), -1.0))
    assert host._metrics["eval"][KL_CLAMP_FRAC_KEY] == [pytest.approx(_TRAINED_SHARE)]


def test_env_forced_close_frac_reads_the_tokens_with_sampling_logprobs():
    """Row 0 is a sampled turn of four tokens whose last is a forced close (sampling log-prob 0); row 1 a
    fallback re-render of four tool and assistant tokens with no sampling log-probs."""
    rows = [
        row([6, 7, 8, _FORCED_CLOSE], sampling=[-1.0, -1.0, -1.0, 0.0]),
        row([9, 10, 11, 12], loss_mask=[1, 0, 0, 1]),
    ]
    host = _eval_build(rows, beta=0.0, is_correction=True, policy=torch.full((2, 4), -1.0))
    assert host._metrics["eval"]["sampling/forced_close_frac"] == [pytest.approx(1 / 4)]


WORLD_SIZE = 2


class _PrefetchHost(AsyncRolloutMixin):
    def __init__(self, hits: int, misses: int, skips: int):
        self._prefetch_hits, self._prefetch_misses, self._prefetch_input_skips = hits, misses, skips


def _counters_worker(rank: int, tmp_dir: str) -> None:
    """Rank 0 prefetched and forced routing; rank 1 missed its every round and forced nothing."""
    host = _PrefetchHost(hits=3, misses=1, skips=0) if rank == 0 else _PrefetchHost(hits=0, misses=4, skips=2)
    layer = BareEPLayer(top_k=2, num_experts=4)
    if rank == 0:
        layer._replay_flip_counts = torch.tensor([3.0, 12.0])
    injector = RoutingReplayInjector([layer], engine_layers=1, layer_indices=[0])
    with open(os.path.join(tmp_dir, f"{rank}.json"), "w") as fh:
        json.dump({"prefetch": host.prefetch_metrics(), "flip": injector.flip_rate()}, fh)


def test_the_counter_metrics_are_world_totals(tmp_path):
    run_gloo_ranks(_counters_worker, WORLD_SIZE, str(tmp_path), pg_timeout=datetime.timedelta(seconds=90))
    prefetch = {
        "async/prefetch_hit_rate": 3 / 8,
        "async/prefetch_hits": 3.0,
        "async/prefetch_misses": 5.0,
        "async/prefetch_input_skips": 2.0,
    }
    for rank in range(WORLD_SIZE):
        read = json.loads((tmp_path / f"{rank}.json").read_text())
        assert read["prefetch"] == pytest.approx(prefetch), f"rank {rank}: {read}"
        assert read["flip"] == pytest.approx(3 / 12), f"rank {rank}: {read}"


def test_a_single_process_reads_its_own_counters():
    assert _PrefetchHost(hits=0, misses=0, skips=0).prefetch_metrics() == {}
    injector = RoutingReplayInjector([BareEPLayer(top_k=2, num_experts=4)], engine_layers=1, layer_indices=[0])
    assert injector.flip_rate() is None, "nothing forced reads as no rate, not 0"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
