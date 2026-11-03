#!/usr/bin/env python
"""Row construction under ``max_train_row_tokens`` and the empty-capture case, on both tokenize paths. The context
check precedes the cap, which must not absorb a config error; ``[]`` is a capture, ``None`` a missing one."""

import types
from collections import defaultdict

import numpy as np
import pytest
import torch
from trl.trainer.utils import pad

from halo.environments.engine_wire import capture_generation_tokens
from halo.environments.episode import RolloutResult, TurnGeneration, step_context_from_generation
from halo.environments.messages import Message, Trajectory
from halo.trainers.grpo.environmental.routing_replay import RoutingReplayInjector
from halo.trainers.grpo.environmental.step_batch import BatchBuildFence, BatchRows, rollout_valid_mask
from halo.trainers.grpo.environmental.trainer import DistributedAsyncEnvironmentalGRPOTrainer
from halo.trainers.grpo.environmental.trajectory_tokenize import (
    SAMPLED_IDS_MISSING_WARNING,
    UNTRAINABLE_TURNS_ROWLESS_FRAC_KEY,
)
from halo.trainers.grpo.objective.logratio import zero_engine_forced_closes
from tests.common.routing import BareEPLayer, npy_routing_payload
from tests.common.world_metrics import attach_world_metrics, flushed_metrics

_ROLE_TOKENS = {"user": 1001, "assistant": 1002, "tool": 1003, "system": 1004}
_END_TOKEN = 1000
_REASONING_CLOSE = 77


def _flat_render(msgs, add_generation_prompt, _template_kwargs, include_thinking=True):
    """A prefix-monotone template the span locator anchors by construction, so the whole-trajectory path runs."""
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
    trainer._batch_errors = BatchBuildFence()
    trainer._warned_once = set()
    trainer.async_config = types.SimpleNamespace(
        rollout_chat_template_kwargs={},
        max_train_row_tokens=cap,
        train_on_sampled_tokens=per_turn,
        drop_degenerate_groups=False,
    )
    trainer._carry_reasoning = False
    trainer._rows_over_cap = 0
    trainer._context_limit = lambda: context_limit
    trainer._metrics = {"train": defaultdict(list)}
    attach_world_metrics(trainer)
    trainer._tokenizer = types.SimpleNamespace(name_or_path="stub-model")
    trainer.eos_token_id = 2
    trainer.pad_token_id = 0
    trainer._render_messages_to_ids = _flat_render
    trainer.args = types.SimpleNamespace(mask_truncated_completions=False)
    trainer.accelerator = types.SimpleNamespace(gather=lambda x: x)
    return trainer


def _result(*turns: Message) -> RolloutResult:
    return RolloutResult(prompt="q", trajectory=Trajectory(messages=[Message.user("q"), *turns]))


def _turn(comp: list[int] | None, prompt_len: int = 2, **fields) -> Message:
    return Message.assistant("a", token_ids=comp, prompt_token_ids=list(range(1, prompt_len + 1)), **fields)


def test_a_per_turn_row_over_the_context_is_recorded_even_when_the_cap_is_lower():
    trainer = _trainer(cap=3, context_limit=4)
    rows = trainer._tokenize_trajectory_turns(_result(_turn([1, 2, 3, 4, 5], prompt_len=3)))  # 8 tokens
    assert trainer._batch_errors.reason is not None and "context window" in trainer._batch_errors.reason
    assert trainer._rows_over_cap == 1 and [r.completion_mask.tolist() for r in rows] == [[0]]


def test_a_per_turn_row_under_the_context_but_over_the_cap_is_only_over_cap():
    trainer = _trainer(cap=3, context_limit=100)
    trainer._tokenize_trajectory_turns(_result(_turn([1, 2, 3, 4, 5], prompt_len=3)))
    assert trainer._batch_errors.reason is None and trainer._rows_over_cap == 1


def _trajectory_tokens(result: RolloutResult) -> int:
    prompt, completion, _ = _trainer(per_turn=False)._tokenize_trajectory(result)
    return len(prompt) + len(completion)


@pytest.mark.parametrize("per_turn", [False, True], ids=["whole-trajectory", "per-turn"])
def test_a_row_of_exactly_the_context_trains_and_one_token_more_is_recorded(per_turn):
    result = _result(_turn([5, 6, 7], prompt_len=3))
    tokens = 6 if per_turn else _trajectory_tokens(result)
    tokenize = "_tokenize_trajectory_turns" if per_turn else "_tokenize_trajectory"

    at_window = _trainer(per_turn=per_turn, context_limit=tokens)
    getattr(at_window, tokenize)(result)
    assert at_window._batch_errors.reason is None, at_window._batch_errors.reason

    one_over = _trainer(per_turn=per_turn, context_limit=tokens - 1)
    getattr(one_over, tokenize)(result)
    error = one_over._batch_errors.reason
    assert error is not None and f"of {tokens} tokens" in error and f"context window {tokens - 1}" in error, error


def test_a_whole_trajectory_overflow_is_recorded_and_the_steps_first_failure_kept():
    first, second = _result(Message.assistant("aaaa")), _result(Message.assistant("bb"))
    trainer = _trainer(per_turn=False, context_limit=2)
    trainer._tokenize_step_rows([first, second])
    error = trainer._batch_errors.reason
    assert error is not None and error.startswith(f"Trajectory of {_trajectory_tokens(first)} tokens"), error


def test_a_malformed_routed_experts_payload_is_recorded_and_the_turn_trains_unrouted():
    """The engine shipped one layer row per token for a model with two decoder layers."""
    trainer = _trainer()
    trainer._rollout_routing_replay = True
    trainer._routing_injector = RoutingReplayInjector(
        [BareEPLayer(top_k=2, num_experts=8)], engine_layers=2, layer_indices=[1]
    )
    payload = npy_routing_payload(np.zeros((5, 1, 2), dtype=np.int32))
    rows = trainer._tokenize_trajectory_turns(_result(_turn([5, 6, 7], routing_mask=payload, routing_prompt_tokens=2)))
    error = trainer._batch_errors.reason
    assert error is not None and error.startswith("routing_replay='rollout': malformed routed_experts payload:"), error
    assert "decoder layers" in error, error
    assert [row.turn_routing for row in rows] == [None]


def test_eight_all_over_cap_trajectories_read_one():
    """The zero-weight placeholder must not enter the denominator as a built row."""
    trainer = _trainer(cap=3, per_turn=False)
    results = [_result(Message.assistant("aaaa")) for _ in range(8)]  # every render > 3 tokens
    per_trajectory = trainer._tokenize_step_rows(results)
    assert all(len(rows) == 1 and rows[0].completion_mask.tolist() == [0] for rows in per_trajectory)
    assert flushed_metrics(trainer)["sampling/rows_over_cap_frac"] == [1.0]


def test_a_mixed_step_reads_rows_left_out_over_rows_left_out_plus_rows_that_train():
    """Per-turn path, cap 6: 3 rows left out and 2 train, so 0.6; a count over built rows would read 0.5."""
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


def test_a_zero_token_turn_yields_no_row_and_the_other_turns_still_train_per_turn():
    trainer = _trainer()
    trainer._tokenize_trajectory = lambda result: pytest.fail("a captured trajectory must not re-render")
    rows = trainer._tokenize_trajectory_turns(_result(_turn([5, 6]), _turn([], prompt_len=3), _turn([7])))
    assert [r.completion_ids.tolist() for r in rows] == [[5, 6], [7]]
    assert SAMPLED_IDS_MISSING_WARNING not in trainer._warned_once


def _refused_render(*_args):
    raise ValueError("the template rejects this prefix")


# Every way a turn leaves the per-turn batch without dropping its episode.
_EXCLUDED_TURNS = {
    "zero-token": (_turn([]), _turn([], prompt_len=3)),
    "over-cap": (_turn([5, 6, 7, 8], prompt_len=4),),
    "untrainable-render-refused": (_turn([5, 6], prompt_len=0, truncated=True),),
    "mixed": (_turn([]), _turn([5, 6, 7, 8], prompt_len=4), _turn([5, 6], prompt_len=0, truncated=True)),
}


@pytest.mark.parametrize("turns", list(_EXCLUDED_TURNS.values()), ids=list(_EXCLUDED_TURNS))
def test_an_all_excluded_trajectory_is_one_masked_row_without_a_re_render(turns):
    trainer = _trainer(cap=6)
    trainer._render_messages_to_ids = _refused_render
    trainer._tokenize_trajectory = lambda result: pytest.fail("nothing to render: the row is the placeholder")
    result = _result(*turns)
    rows = trainer._tokenize_trajectory_turns(result)
    assert len(rows) == 1 and rows[0].completion_mask.tolist() == [0]
    assert not result.trajectory.episode_invalid, "an excluded turn leaves its episode in the group baseline"
    assert trainer._batch_errors.reason is None


def test_a_trainable_turn_with_no_capture_still_falls_the_trajectory_back():
    sentinel = (torch.tensor([0]), torch.tensor([99]), torch.tensor([1]))
    trainer = _trainer()
    trainer._tokenize_trajectory = lambda result: sentinel
    rows = trainer._tokenize_trajectory_turns(_result(_turn([5, 6]), _turn(None, prompt_len=3)))
    assert len(rows) == 1 and rows[0].completion_ids.tolist() == [99]
    assert SAMPLED_IDS_MISSING_WARNING in trainer._warned_once


def test_an_untrainable_turn_without_capture_does_not_force_the_fallback():
    """The all-or-nothing check runs over the turns that train, so a cut fragment with no ids costs no rows."""
    trainer = _trainer()
    trainer._tokenize_trajectory = lambda result: pytest.fail("the trainable turn is captured; no re-render")
    rows = trainer._tokenize_trajectory_turns(_result(_turn(None, truncated=True), _turn([7, 8])))
    assert [r.completion_ids.tolist() for r in rows] == [[7, 8]]
    assert SAMPLED_IDS_MISSING_WARNING not in trainer._warned_once


def test_untrainable_turns_that_became_no_row_are_counted():
    """Per-turn, the captured cut and empty turns became tagged rows and the uncaptured cut none, so one in three."""
    results = [
        _result(_turn([5, 6], truncated=True), _turn([7])),
        _result(_turn(None, truncated=True), _turn([8], empty=True), _turn([9])),
    ]
    trainer = _trainer()
    per_trajectory = trainer._tokenize_step_rows(results)
    assert [[r.negative_only for r in rows] for rows in per_trajectory] == [[True, False], [True, False]]
    assert flushed_metrics(trainer)[UNTRAINABLE_TURNS_ROWLESS_FRAC_KEY] == [pytest.approx(1 / 3)]

    whole = _trainer(per_turn=False)
    whole._tokenize_step_rows(results)
    assert flushed_metrics(whole)[UNTRAINABLE_TURNS_ROWLESS_FRAC_KEY] == [1.0]


_UNTRAINABLE_FLAGS = [{"truncated": True}, {"empty": True}, {"calls_rejected": True}]
_FLAG_IDS = ["cut", "empty", "invented_calls"]


def _cut_episode_batch(trainer, advantage: float, flag: dict):
    """An untrainable turn whose reasoning the engine closed at its budget, then a finished turn, narrowed the way
    ``_build_training_tensors`` does."""
    cut = _turn([5, _REASONING_CLOSE, 6], token_logprobs=[-0.5, 0.0, -0.3], **flag)
    finished = _turn([7, 8], prompt_len=3, token_logprobs=[-0.2, -0.1])
    result = _result(cut, finished)
    rows = trainer._tokenize_trajectory_turns(result)
    sampling = pad([r.sampling_logps for r in rows], padding_value=0, padding_side="right")
    completion_ids = pad([r.completion_ids for r in rows], padding_value=0, padding_side="right")
    completion_mask = pad([r.completion_mask.bool() for r in rows], padding_value=0, padding_side="right")
    ratio, _forced = zero_engine_forced_closes(
        torch.ones_like(sampling),
        sampling,
        completion_mask,
        torch.ones(len(rows), dtype=torch.bool),
        completion_ids,
        (_REASONING_CLOSE,),
    )
    batch = BatchRows([result], [len(rows)], 0, True)
    step = types.SimpleNamespace(
        completion_mask=completion_mask,
        tool_mask=completion_mask.clone(),
        negative_only=torch.tensor([r.negative_only for r in rows]),
    )
    _comp, tool_mask, _loss, num_items = trainer._narrow_masks_and_normalizer(
        batch,
        step,
        rollout_valid_mask([result], torch.device("cpu")),
        1,
        batch.to_rows(torch.tensor([advantage])),
        "train",
    )
    return rows, ratio, tool_mask, num_items


@pytest.mark.parametrize("flag", _UNTRAINABLE_FLAGS, ids=_FLAG_IDS)
def test_an_untrainable_turn_trains_under_a_negative_advantage_with_its_forced_close_at_ratio_zero(flag):
    rows, ratio, tool_mask, num_items = _cut_episode_batch(_trainer(), -1.0, flag)
    assert [r.negative_only for r in rows] == [True, False]
    assert tool_mask[0].tolist() == [True, True, True], "the cut turn's sampled ids are in the loss"
    assert num_items.item() == 5, "and in the normalizer"
    assert ratio[0].tolist() == [1.0, 0.0, 1.0], "the engine-forced close inside it carries no gradient"


@pytest.mark.parametrize("flag", _UNTRAINABLE_FLAGS, ids=_FLAG_IDS)
@pytest.mark.parametrize("advantage", [1.0, 0.0])
def test_an_untrainable_turn_never_trains_at_a_non_negative_advantage(advantage, flag):
    _rows, _ratio, tool_mask, num_items = _cut_episode_batch(_trainer(), advantage, flag)
    assert not tool_mask[0].any()
    assert tool_mask[1].tolist() == [True, True, False]
    assert num_items.item() == 2


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
