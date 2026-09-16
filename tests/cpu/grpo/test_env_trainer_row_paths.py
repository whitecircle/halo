#!/usr/bin/env python
"""Row construction under ``max_train_row_tokens`` and the empty-capture case, on both tokenize paths.

* The context-window check runs BEFORE the row cap on the per-turn path, as it always did on the
  whole-trajectory path: a row the served model could not have produced is a config error, and a cap
  set below the context must not absorb it as "over cap".
* ``sampling/rows_over_cap_frac`` counts each row once: rows the cap left out over those plus the rows
  that train. An over-cap trajectory comes back as a zero-weight placeholder, which is neither — a
  count that took the placeholder as a built row read eight all-over-cap trajectories as 0.5.
* A zero-token assistant turn (``token_ids == []``) is a capture that succeeded, distinct from a
  missing one (``None``): it yields no row, the rest of the trajectory trains per turn, and the
  re-render fallback (with its server-flag warning) is reserved for a trainable turn with no ids.

    python tests/cpu/grpo/test_env_trainer_row_paths.py
"""

import types
from collections import defaultdict

import pytest
import torch

from src.environments.base import Message, Trajectory
from src.environments.engine_wire import capture_generation_tokens
from src.environments.episode import RolloutResult, TurnGeneration, step_context_from_generation
from src.trainers.grpo.environmental import DistributedAsyncEnvironmentalGRPOTrainer
from tests.common.grpo_metrics import attach_world_metrics, flushed_metrics

_ROLE_TOKENS = {"user": 1001, "assistant": 1002, "tool": 1003, "system": 1004}
_END_TOKEN = 1000


def _flat_render(msgs, add_generation_prompt, _template_kwargs, include_thinking=True):
    """A minimal prefix-monotone template (``<role> <content chars> <end>``) the span locator anchors
    by construction, so the whole-trajectory path runs for real without a tokenizer."""
    ids = []
    for m in msgs:
        ids.append(_ROLE_TOKENS[m.role])
        ids.extend(ord(c) for c in (m.content or ""))
        ids.append(_END_TOKEN)
    if add_generation_prompt:
        ids.append(_ROLE_TOKENS["assistant"])
    return ids


def _trainer(cap: int | None = None, per_turn: bool = True, context_limit: int = 100_000):
    """The trainer's real tokenize methods over a stub state; the fallback re-render is a sentinel."""
    trainer = object.__new__(DistributedAsyncEnvironmentalGRPOTrainer)
    trainer._rollout_routing_replay = False
    trainer._rollout_backend = "vllm"
    trainer._batch_build_error = None
    trainer._warned_capture_missing = False
    trainer._rollout_template_kwargs = {}
    trainer._carry_reasoning = False
    trainer._max_train_row_tokens = cap
    trainer._rows_over_cap = 0
    trainer._train_on_sampled_tokens = per_turn
    trainer._context_limit = lambda: context_limit
    trainer._metrics = {"train": defaultdict(list)}
    attach_world_metrics(trainer)
    trainer._tokenizer = types.SimpleNamespace(name_or_path="stub-model")
    trainer.eos_token_id = 2
    trainer.pad_token_id = 0
    trainer._render_messages_to_ids = _flat_render
    return trainer


def _result(*turns: Message) -> RolloutResult:
    return RolloutResult(prompt="q", trajectory=Trajectory(messages=[Message.user("q"), *turns]))


def _turn(comp: list[int] | None, prompt_len: int = 2, **fields) -> Message:
    return Message.assistant("a", token_ids=comp, prompt_token_ids=list(range(1, prompt_len + 1)), **fields)


# --- the context check precedes the cap on the per-turn path ------------------------------------------


def test_a_per_turn_row_over_the_context_is_recorded_even_when_the_cap_is_lower():
    trainer = _trainer(cap=3, context_limit=4)
    rows = trainer._tokenize_trajectory_turns(_result(_turn([1, 2, 3, 4, 5], prompt_len=3)))  # 8 tokens
    assert trainer._batch_build_error is not None and "context window" in trainer._batch_build_error
    assert trainer._rows_over_cap == 1 and [r.completion_mask.tolist() for r in rows] == [[0]]


def test_a_per_turn_row_under_the_context_but_over_the_cap_is_only_over_cap():
    trainer = _trainer(cap=3, context_limit=100)
    trainer._tokenize_trajectory_turns(_result(_turn([1, 2, 3, 4, 5], prompt_len=3)))
    assert trainer._batch_build_error is None and trainer._rows_over_cap == 1


# --- sampling/rows_over_cap_frac counts each row once --------------------------------------------------


def test_eight_all_over_cap_trajectories_read_one():
    """Whole-trajectory path: every trajectory returns the zero-weight placeholder AND counts as over
    cap; the placeholder must not enter the denominator as a built row."""
    trainer = _trainer(cap=3, per_turn=False)
    results = [_result(Message.assistant("aaaa")) for _ in range(8)]  # every render > 3 tokens
    per_trajectory = trainer._tokenize_step_rows(results)
    assert all(len(rows) == 1 and rows[0].completion_mask.tolist() == [0] for rows in per_trajectory)
    assert flushed_metrics(trainer)["sampling/rows_over_cap_frac"] == [1.0]


def test_a_mixed_step_reads_rows_left_out_over_rows_left_out_plus_rows_that_train():
    """Per-turn path, cap 6: A's two turns are both over (one placeholder), B loses one of two, C's
    single turn fits — 3 rows left out, 2 rows train, so 0.6; a count over built rows read 0.5."""
    trainer = _trainer(cap=6)
    results = [
        _result(_turn([7, 8, 9, 10], prompt_len=4), _turn([7, 8, 9, 10], prompt_len=4)),
        _result(_turn([5, 6]), _turn([7, 8, 9, 10], prompt_len=4)),
        _result(_turn([5])),
    ]
    per_trajectory = trainer._tokenize_step_rows(results)
    assert [len(rows) for rows in per_trajectory] == [1, 1, 1]
    assert per_trajectory[0][0].completion_mask.tolist() == [0]
    assert flushed_metrics(trainer)["sampling/rows_over_cap_frac"] == [pytest.approx(0.6)]


def test_no_cap_logs_no_cap_metric():
    trainer = _trainer(cap=None)
    trainer._tokenize_step_rows([_result(_turn([5, 6]))])
    assert "sampling/rows_over_cap_frac" not in flushed_metrics(trainer)


# --- an empty capture is a capture ----------------------------------------------------------------------


def test_a_zero_token_turn_yields_no_row_and_the_other_turns_still_train_per_turn():
    trainer = _trainer()
    trainer._tokenize_trajectory = lambda result: pytest.fail("a captured trajectory must not re-render")
    rows = trainer._tokenize_trajectory_turns(_result(_turn([5, 6]), _turn([], prompt_len=3), _turn([7])))
    assert [r.completion_ids.tolist() for r in rows] == [[5, 6], [7]]
    assert trainer._warned_capture_missing is False


def test_an_all_empty_trajectory_is_one_masked_row_without_a_re_render():
    trainer = _trainer()
    trainer._tokenize_trajectory = lambda result: pytest.fail("nothing to render: the row is the placeholder")
    rows = trainer._tokenize_trajectory_turns(_result(_turn([]), _turn([], prompt_len=3)))
    assert len(rows) == 1 and rows[0].completion_mask.tolist() == [0]


def test_a_trainable_turn_with_no_capture_still_falls_the_trajectory_back():
    sentinel = (torch.tensor([0]), torch.tensor([99]), torch.tensor([1]))
    trainer = _trainer()
    trainer._tokenize_trajectory = lambda result: sentinel
    rows = trainer._tokenize_trajectory_turns(_result(_turn([5, 6]), _turn(None, prompt_len=3)))
    assert len(rows) == 1 and rows[0].completion_ids.tolist() == [99]
    assert trainer._warned_capture_missing is True


def test_an_untrainable_turn_without_capture_does_not_force_the_fallback():
    """The all-or-nothing check runs over the turns that train: a cut fragment the engine returned no
    ids for is skipped either way, so it must not cost the trajectory its per-turn rows."""
    trainer = _trainer()
    trainer._tokenize_trajectory = lambda result: pytest.fail("the trainable turn is captured; no re-render")
    rows = trainer._tokenize_trajectory_turns(_result(_turn(None, truncated=True), _turn([7, 8])))
    assert [r.completion_ids.tolist() for r in rows] == [[7, 8]]
    assert trainer._warned_capture_missing is False


def test_the_wire_keeps_an_empty_capture_apart_from_a_missing_one():
    vllm_choice = {"logprobs": {"content": []}}
    assert capture_generation_tokens(vllm_choice, {"prompt_token_ids": [1, 2]}, "vllm") == ([], [], [1, 2])
    assert capture_generation_tokens({"logprobs": None}, {}, "vllm")[0] is None
    sglang_choice = {"meta_info": {"output_token_logprobs": []}, "prompt_token_ids": [1, 2]}
    assert capture_generation_tokens(sglang_choice, {}, "sglang") == ([], [], [1, 2])
    assert capture_generation_tokens({"meta_info": {}}, {}, "sglang")[0] is None


def test_the_step_context_forwards_an_empty_capture_and_drops_a_missing_one():
    empty = TurnGeneration(text="", tool_calls=[], reasoning="", tokens=0, finish_reason="stop", token_ids=[])
    ctx = step_context_from_generation(None, empty)
    assert ctx["token_ids"] == []
    missing = TurnGeneration(text="", tool_calls=[], reasoning="", tokens=0, finish_reason="stop", token_ids=None)
    assert "token_ids" not in step_context_from_generation(None, missing)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
