#!/usr/bin/env python
"""Candidate log-probs through the round's row paths. Each candidate slot encodes the row and position it was made for,
so a path that shifts a row or a position (the per-turn rows, the padded stack, the packed slot, TRL's permutation)
reads back the wrong code; padding must stay unused (-1 / -inf)."""

import types
import warnings

import numpy as np
import pytest
import torch

from halo.environments.episodes.candidate_logprobs import NO_CANDIDATE_ID, CandidateLogprobs
from halo.environments.episodes.messages import Message, Trajectory
from halo.environments.rollouts.episode_loop import RolloutResult, TurnGeneration, step_context_from_generation
from halo.trainers.grpo.environmental.packed_rounds import PackedMicrobatchMixin
from halo.trainers.grpo.environmental.step_batch import BatchBuildFence
from halo.trainers.grpo.environmental.trainer import DistributedAsyncEnvironmentalGRPOTrainer as Trainer
from halo.trainers.grpo.environmental.trajectory_tokenize import TurnRow, single_trajectory_row
from halo.trainers.grpo.objective.score_centering import TOPK_IDS_KEY, TOPK_LOGPS_KEY
from halo.trainers.grpo.rounds.packed_rounds import SLOT_ACTIVE_KEY, SLOT_ROWS_KEY
from halo.trainers.grpo.rounds.row_order import permute_rows

K = 3


def _code(row: int, position: int) -> int:
    return 1000 * row + 10 * position


def _candidates(row: int, tokens: int) -> CandidateLogprobs:
    ids = np.array([[_code(row, t) + j for j in range(K)] for t in range(tokens)], dtype=np.int32)
    return CandidateLogprobs(ids, -ids.astype(np.float32) / 1e4)


def _turn_row(row: int, tokens: int, *, candidates: bool = True, trained: bool = True) -> TurnRow:
    return TurnRow(
        prompt_ids=torch.tensor([1, 2]),
        completion_ids=torch.arange(10, 10 + tokens),
        loss_mask=torch.full((tokens,), trained, dtype=torch.bool),
        sampling_logps=None,
        turn_routing=None,
        candidates=_candidates(row, tokens) if candidates else None,
    )


def _host(k: int | None = K):
    host = types.SimpleNamespace(
        pad_token_id=0,
        _is_correction=False,
        async_config=types.SimpleNamespace(score_centering_top_k=k),
        _batch_errors=BatchBuildFence(),
    )
    host._record_rows_without_candidates = types.MethodType(Trainer._record_rows_without_candidates, host)
    return host


def _assert_encodes_its_position(ids: torch.Tensor, logps: torch.Tensor, lengths: list[int], rows: list[int]) -> None:
    for slot, (row, tokens) in enumerate(zip(rows, lengths, strict=True)):
        expected = torch.tensor(_candidates(row, tokens).ids)
        assert torch.equal(ids[slot, :tokens], expected), f"slot {slot} lost row {row}'s positions"
        assert (ids[slot, tokens:] == NO_CANDIDATE_ID).all() and torch.isneginf(logps[slot, tokens:]).all()


def test_stack_rows_pads_each_rows_candidates_to_the_completion_width_in_row_order():
    rows = [_turn_row(0, 4), _turn_row(1, 2), _turn_row(2, 3)]
    step = Trainer._stack_rows(_host(), rows, torch.device("cpu"))
    assert step.topk_ids.shape == (3, 4, K) and step.topk_ids.dtype == torch.int32
    assert step.topk_logps.dtype == torch.float32
    _assert_encodes_its_position(step.topk_ids, step.topk_logps, [4, 2, 3], [0, 1, 2])
    assert Trainer._stack_rows(_host(None), rows, torch.device("cpu")).topk_ids is None


def test_read_only_candidates_from_the_object_store_stack_without_a_warning():
    rows = [_turn_row(0, 4), _turn_row(1, 2)]
    for row in rows:
        row.candidates.ids.setflags(write=False)
        row.candidates.logps.setflags(write=False)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        step = Trainer._stack_rows(_host(), rows, torch.device("cpu"))
    _assert_encodes_its_position(step.topk_ids, step.topk_logps, [4, 2], [0, 1])


def test_a_masked_or_dummy_row_stacks_as_unused_candidates():
    (inert,) = single_trajectory_row((torch.tensor([0]), torch.tensor([1]), torch.tensor([0])))
    step = Trainer._stack_rows(_host(), [_turn_row(0, 2), inert], torch.device("cpu"))
    assert (step.topk_ids[1] == NO_CANDIDATE_ID).all() and torch.isneginf(step.topk_logps[1]).all()


@pytest.mark.parametrize(
    "rows",
    [
        [_turn_row(0, 3, candidates=False)],
        single_trajectory_row((torch.tensor([1, 2]), torch.tensor([3, 4]), torch.tensor([1, 1]))),
        [_turn_row(0, 3)._replace(candidates=_candidates(0, 2))],
    ],
    ids=["turn without candidates", "re-tokenized trajectory row", "candidates shorter than the completion"],
)
def test_a_trained_row_without_aligned_candidates_fails_the_batch_fence(rows):
    host = _host()
    host._record_rows_without_candidates(rows)
    with pytest.raises(ValueError, match="score_centering_top_k=3"):
        host._batch_errors.reject()


def test_rows_that_train_nothing_or_run_without_the_knob_pass_the_fence():
    host = _host()
    host._record_rows_without_candidates([_turn_row(0, 3), _turn_row(1, 2, candidates=False, trained=False)])
    host._batch_errors.reject()
    off = _host(None)
    off._record_rows_without_candidates([_turn_row(0, 3, candidates=False)])
    off._batch_errors.reject()


def test_a_turn_stamps_candidates_only_when_they_cover_every_sampled_id():
    gen = TurnGeneration(
        text="x", tool_calls=[], reasoning="", tokens=3, token_ids=[5, 6, 7], candidates=_candidates(0, 3)
    )
    assert step_context_from_generation(None, gen)["candidates"] is gen.candidates
    short = TurnGeneration(
        text="x", tool_calls=[], reasoning="", tokens=3, token_ids=[5, 6, 7], candidates=_candidates(0, 2)
    )
    assert "candidates" not in step_context_from_generation(None, short)


def test_a_per_turn_row_carries_its_messages_candidates_and_a_missing_capture_is_recorded():
    traj = Trajectory()
    traj.add_message(Message.user("go"))
    traj.add_message(
        Message.assistant("a", token_ids=[7, 8, 9], prompt_token_ids=[1, 2], candidates=_candidates(4, 3))
    )
    traj.add_message(Message.user("more"))
    traj.add_message(Message.assistant("b", token_ids=[5, 6], prompt_token_ids=[1, 2, 3]))
    tokenizer = types.SimpleNamespace(model_max_length=int(1e30))
    stub = types.SimpleNamespace(
        processing_class=tokenizer,
        _tokenizer=tokenizer,
        _tools_schema=None,
        _rollout_routing_replay=False,
        async_config=types.SimpleNamespace(
            rollout_chat_template_kwargs={}, max_train_row_tokens=None, score_centering_top_k=K
        ),
        _carry_reasoning=False,
        _rows_over_cap=0,
        _batch_errors=BatchBuildFence(),
        model=types.SimpleNamespace(config=types.SimpleNamespace(max_position_embeddings=131072)),
    )
    for name in (
        "_context_limit",
        "_tokenize_trajectory_turns",
        "_template_kwargs",
        "_admit_row",
        "_sampled_turn_row",
    ):
        setattr(stub, name, types.MethodType(getattr(Trainer, name), stub))
    stub._record_rows_without_candidates = types.MethodType(Trainer._record_rows_without_candidates, stub)
    rows = stub._tokenize_trajectory_turns(RolloutResult(prompt="go", trajectory=traj))
    assert [row.candidates is not None for row in rows] == [True, False]
    assert torch.equal(torch.tensor(rows[0].candidates.ids), torch.tensor(_candidates(4, 3).ids))
    stub._record_rows_without_candidates(rows)
    with pytest.raises(ValueError, match="1 of 2 training rows"):
        stub._batch_errors.reject()


def test_permuting_rows_moves_each_rows_candidates_with_its_completion():
    rows = [_turn_row(r, n) for r, n in enumerate((4, 2, 3, 1))]
    step = Trainer._stack_rows(_host(), rows, torch.device("cpu"))
    order = torch.tensor([2, 0, 3, 1])
    permuted = permute_rows(
        {TOPK_IDS_KEY: step.topk_ids, TOPK_LOGPS_KEY: step.topk_logps, "completion_ids": step.completion_ids}, order
    )
    _assert_encodes_its_position(permuted[TOPK_IDS_KEY], permuted[TOPK_LOGPS_KEY], [3, 4, 1, 2], order.tolist())
    assert torch.equal(permuted["completion_ids"][0, :3], torch.arange(10, 13))


def test_a_packed_slots_filler_rows_hold_unused_candidates():
    host = object.__new__(PackedMicrobatchMixin)
    host.pad_token_id = 9
    host.args = types.SimpleNamespace(steps_per_generation=2)
    host.loss_type = "grpo"
    step = Trainer._stack_rows(_host(), [_turn_row(0, 2), _turn_row(1, 2)], torch.device("cpu"))
    host._round_store = {
        "advantages": torch.tensor([0.5, -0.5]),
        "prompt_ids": torch.tensor([[1, 2], [3, 4]]),
        "completion_ids": torch.tensor([[5, 6], [6, 7]]),
        "prompt_mask": torch.zeros(2, 2, dtype=torch.bool),
        "importance_sampling_ratio": torch.ones(2, 2),
        TOPK_IDS_KEY: step.topk_ids,
        TOPK_LOGPS_KEY: step.topk_logps,
    }
    slot = {
        SLOT_ROWS_KEY: torch.tensor([1, -1]),
        SLOT_ACTIVE_KEY: torch.tensor([1, 1]),
        "num_items_in_batch": torch.tensor(2),
    }
    view = host._build_slot_view(slot)
    assert torch.equal(view[TOPK_IDS_KEY][0], step.topk_ids[1]), "the real row keeps its own candidates"
    assert (view[TOPK_IDS_KEY][1] == NO_CANDIDATE_ID).all() and torch.isneginf(view[TOPK_LOGPS_KEY][1]).all()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
