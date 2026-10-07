#!/usr/bin/env python
"""Each batch-construction failure is raised at its own phase's fence, ahead of the next phase's work.

Outside routing replay the round entry joins the ``BatchBuildFence`` three times:
* after reading the prompts, for an unusable prompt;
* after the rollouts are acquired and before any synchronous fallback, for a wedged prefetch;
* after tokenization, for a row over the context window.

A wedged rank fenced only later would first pay a whole synchronous round while its peers wait in the
fence. A row failure fenced later would run the recompute forward, whose dispatch is a collective, on a
batch the step then refuses. The round entry and the batch build run for real over a recording fence;
rollout collection, tokenization and the forward are stubs that log their turn.

    python tests/cpu/grpo/test_env_batch_fence_placement.py
"""

from collections import deque
from types import SimpleNamespace

import pytest
import torch
from accelerate import PartialState

from src.trainers.grpo.environmental import BatchBuildFence
from tests.common.env_grpo_batch import batch_host, episode, row

PartialState()  # the world-metrics flush gathers through accelerate, which reads the process state


class _RecordingFence(BatchBuildFence):
    def __init__(self, events: list[str]):
        super().__init__()
        self._events = events

    def reject(self) -> None:
        self._events.append("fence")
        super().reject()


def _round(events: list[str], *, row_error: bool = False, wedged_prefetch: bool = False):
    """A train round of two episodes through the real round entry, logging each phase into ``events``."""
    host = batch_host([row([6]), row([7])], torch.full((2, 1), -1.0), training=True, is_correction=True)
    host._batch_errors = _RecordingFence(events)
    host.accelerator = SimpleNamespace(gather=lambda x: x, device=torch.device("cpu"), is_main_process=False)
    host.args.report_to = []
    host.state = SimpleNamespace(global_step=1)
    host._group_random_effort = False
    host._tokenizer = None  # read only for thinking text, which these episodes carry none of
    host._broadcast_object_from_tp_leader = lambda results: results
    host._log_rollout_metrics = lambda results, mode, reasoning_tokens: None
    host._loop = SimpleNamespace(run_until_complete=lambda results: results)

    def collect(prompts, contexts):
        events.append("rollout")
        return [episode(1.0), episode(0.0)]

    host._rollout_manager = SimpleNamespace(collect_rollouts=collect)
    tokenize, forward = host._tokenize_step_rows, host._get_per_token_logps_and_entropies

    def tokenize_rows(results):
        events.append("tokenize")
        if row_error:
            host._batch_errors.record("Trajectory over the context window")
        return tokenize(results)

    def recompute_forward(*args, **kwargs):
        events.append("recompute forward")
        return forward(*args, **kwargs)

    host._tokenize_step_rows, host._get_per_token_logps_and_entropies = tokenize_rows, recompute_forward
    host._prefetch_enabled = wedged_prefetch
    if wedged_prefetch:
        host._prefetch_pending = deque([(["task", "task"], [None, None])])
        host._try_get_prefetched_results = lambda: None

        def wait_on_a_wedged_worker():
            host._batch_errors.record("the prefetch pipeline is wedged")
            return None

        host._wait_for_inflight_prefetch = wait_on_a_wedged_worker
    return host


_TASKS = [{"prompt": "task"}, {"prompt": "task"}]


def test_a_round_joins_the_fence_at_each_phase_boundary():
    events = []
    _round(events)._generate_and_score_completions_base(_TASKS)
    assert events == ["fence", "fence", "rollout", "tokenize", "fence", "recompute forward"]


def test_a_wedged_prefetch_is_raised_before_the_synchronous_fallback():
    events = []
    with pytest.raises(ValueError, match="wedged"):
        _round(events, wedged_prefetch=True)._generate_and_score_completions_base(_TASKS)
    assert events == ["fence", "fence"], "the wedged rank rolled out a synchronous round first"


def test_a_row_failure_is_raised_before_the_recompute_forward():
    events = []
    with pytest.raises(ValueError, match="context window"):
        _round(events, row_error=True)._generate_and_score_completions_base(_TASKS)
    assert events == ["fence", "fence", "rollout", "tokenize", "fence"]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
