#!/usr/bin/env python
"""The rejection-sampling CLI's records: no degenerate preference pairs (with one hypothesis or all-equal scores,
argmax and argmin are the same completion) and one record shape per output format. offline_grpo keeps all-equal
rewards, since the trainer's degenerate-group handling owns that case."""

import json
import types

import numpy as np
import pytest

from halo.entrypoints import rejection_sampling
from halo.entrypoints.rejection_sampling import degenerate_hypotheses_reason


def test_single_hypothesis_is_degenerate_for_every_format():
    for output_format in ("preference", "offline_grpo"):
        reason = degenerate_hypotheses_reason(np.array([0.5]), output_format)
        assert reason is not None and "insufficient" in reason, (output_format, reason)


def test_all_equal_scores_skip_preference_but_not_offline_grpo():
    equal = np.array([0.7, 0.7, 0.7])
    reason = degenerate_hypotheses_reason(equal, "preference")
    assert reason is not None and "equal" in reason
    assert degenerate_hypotheses_reason(equal, "offline_grpo") is None


def test_distinct_scores_pass():
    scores = np.array([0.2, 0.9])
    assert degenerate_hypotheses_reason(scores, "preference") is None
    assert degenerate_hypotheses_reason(scores, "offline_grpo") is None


def test_numpy_and_list_scores_decide_the_same():
    for equal, distinct in ((np.array([0.7, 0.7]), np.array([0.7, 0.8])), ([0.7, 0.7], [0.7, 0.8])):
        assert degenerate_hypotheses_reason(equal, "preference") is not None
        assert degenerate_hypotheses_reason(distinct, "preference") is None


_PROMPT = [{"role": "user", "content": "q"}]
_HYPOTHESES = [{"role": "assistant", "content": "a1"}, {"role": "assistant", "content": "a2"}]
_ARGS = types.SimpleNamespace(id_field="id")


def _rm_record(row, correct_answer=None):
    """The sampler's record: it scores whole conversations and keeps the last turn."""
    conversations = [_PROMPT + [hypothesis] for hypothesis in _HYPOTHESES]
    return rejection_sampling.build_offline_grpo_result(
        row, _PROMPT, conversations, np.array([0.2, 0.9]), _ARGS, correct_answer
    )


@pytest.mark.parametrize(
    ("row", "correct_answer", "expected"),
    [
        ({}, None, {"prompt", "completions", "rewards"}),
        ({"id": "r1"}, None, {"prompt", "completions", "rewards", "id"}),
        ({"id": "r1"}, "42", {"prompt", "completions", "rewards", "id", "target_answer"}),
    ],
    ids=["bare", "with_id", "with_ground_truth"],
)
def test_the_offline_grpo_record_is_exactly_the_trainer_contract(row, correct_answer, expected):
    """Equality, since a column added or dropped here reaches the trainer as a dataset that parses and trains on the
    wrong thing."""
    assert _rm_record(row, correct_answer=correct_answer).keys() == expected


def test_only_a_ground_truth_run_adds_target_answer():
    """``--correct_answer_field`` adds ``target_answer`` and nothing else."""
    extra = _rm_record({"id": "r1"}, correct_answer="42").keys() - _rm_record({"id": "r1"}).keys()
    assert extra == {"target_answer"}


def test_the_sampler_invents_no_id_its_source_row_lacks():
    record = _rm_record({})
    assert "id" not in record, record


def test_the_rewards_survive_json():
    """``json.dumps`` raises on a ``numpy.float64``, so the numpy scores must reach the record as plain floats."""
    record = _rm_record({"id": "r1"})
    assert [type(reward) for reward in record["rewards"]] == [float, float], record["rewards"]
    assert json.loads(json.dumps(record)) == record


_PREF_ARGS = types.SimpleNamespace(
    id_field="id",
    model="gen-model",
    rm_model_id="rm-model",
)
_PREFERENCE_KEYS = {
    "prompt",
    "chosen",
    "chosen_score",
    "rejected",
    "rejected_score",
    "all_generations",
    "all_scores",
    "gen_model",
}


def _rm_pair(row, correct_answer=None):
    conversations = [_PROMPT + [hypothesis] for hypothesis in _HYPOTHESES]
    return rejection_sampling.build_preference_result(
        row, _PROMPT, conversations, np.array([0.2, 0.9]), _PREF_ARGS, correct_answer
    )


def test_the_preference_pair_selects_by_score_and_survives_json():
    """``json.dumps`` raises on a ``numpy.float64``, so the ranked numpy scores must reach the record as plain
    floats."""
    pair = _rm_pair({"id": "r1"})

    assert pair.keys() == _PREFERENCE_KEYS | {"id", "rm_model"}
    assert (pair["chosen_score"], pair["rejected_score"]) == (0.9, 0.2)
    assert json.loads(json.dumps(pair)) == pair


def test_the_preference_pair_invents_no_id_its_source_row_lacks():
    """``--id_field`` need not be a column of local JSONL, so a row without an id must not raise KeyError."""
    record = _rm_pair({})
    assert "id" not in record, record


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
