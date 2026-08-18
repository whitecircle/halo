#!/usr/bin/env python
"""The data contracts of the rejection-sampling CLI: what it reads, and what it emits.

*No degenerate preference pairs.* ``build_preference_result`` keeps argmax/argmin of the scores —
with one hypothesis, or with all-equal scores, that is the SAME completion as both chosen and
rejected, a pair that teaches a preference trainer nothing while inflating the dataset. <2
hypotheses always skip; all-equal scores skip for preference output only (offline_grpo keeps the
full reward vector — the trainer's degenerate-group handling owns that case).

*One record shape per output format.* The sampler writes two trainer contracts — preference and
offline-GRPO — through the shared writers in ``scripts/inference/_common.py``, adding only what its
own scorer knows; the tests below fail the moment the records drift away from those contracts.

Run: pytest tests/cpu/inference/test_rm_rejection_degenerate_pairs.py
"""

import json
import sys
import types

import numpy as np
import pytest

from scripts.inference._common import degenerate_hypotheses_reason
from scripts.inference.reward_model import rm_rejection_sampling as rm_rs


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
    """One rule has to read numpy arrays and plain Python lists the same way, or "all equal" means
    two different things."""
    for equal, distinct in ((np.array([0.7, 0.7]), np.array([0.7, 0.8])), ([0.7, 0.7], [0.7, 0.8])):
        assert degenerate_hypotheses_reason(equal, "preference") is not None
        assert degenerate_hypotheses_reason(distinct, "preference") is None


# --- the offline-GRPO record rm_rejection_sampling writes -----------------------------------------


_PROMPT = [{"role": "user", "content": "q"}]
_HYPOTHESES = [{"role": "assistant", "content": "a1"}, {"role": "assistant", "content": "a2"}]
_ARGS = types.SimpleNamespace(id_field="id")


def _rm_record(row, correct_answer=None):
    """The sampler's record: it scores whole conversations and keeps the last turn."""
    conversations = [_PROMPT + [hypothesis] for hypothesis in _HYPOTHESES]
    return rm_rs.build_offline_grpo_result(row, _PROMPT, conversations, np.array([0.2, 0.9]), _ARGS, correct_answer)


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
    """Equality, not containment: the offline-GRPO trainer reads ``{prompt, completions, rewards}``
    (``agent-docs/data/dataset-formats.md``) plus the two conditional keys, and a column added or dropped
    here reaches the trainer as a dataset that parses and trains on the wrong thing. A subset check
    would pass a writer that silently gained a key."""
    assert _rm_record(row, correct_answer=correct_answer).keys() == expected


def test_only_a_ground_truth_run_adds_target_answer():
    """Only a run with a ground-truth concept (``--correct_answer_field``) may add ``target_answer``
    — and it must add nothing else along with it."""
    extra = _rm_record({"id": "r1"}, correct_answer="42").keys() - _rm_record({"id": "r1"}).keys()
    assert extra == {"target_answer"}


def test_the_sampler_invents_no_id_its_source_row_lacks():
    """A source row without an ``id`` column yields a record without one."""
    record = _rm_record({})
    assert "id" not in record, record


def test_the_rewards_survive_json():
    """The records are written as JSONL. ``json.dumps`` raises on a ``numpy.float64``, so the RM
    sampler's numpy scores have to reach the record as plain floats."""
    record = _rm_record({"id": "r1"})
    assert [type(reward) for reward in record["rewards"]] == [float, float], record["rewards"]
    assert json.loads(json.dumps(record)) == record


# --- the preference record rm_rejection_sampling writes -------------------------------------------


_PREF_ARGS = types.SimpleNamespace(
    id_field="id",
    model_name="gen-model",
    rm_model_path="rm-model",
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
    return rm_rs.build_preference_result(row, _PROMPT, conversations, np.array([0.2, 0.9]), _PREF_ARGS, correct_answer)


def test_the_preference_pair_selects_by_score_and_survives_json():
    """The pair must be argmax/argmin of the scores under the trainers' key spelling, and the scores
    have to reach the record as plain floats — the sampler ranks a numpy vector, and ``json.dumps``
    raises on a ``numpy.float64``."""
    pair = _rm_pair({"id": "r1"})

    assert pair.keys() == _PREFERENCE_KEYS | {"id", "rm_model"}
    assert (pair["chosen_score"], pair["rejected_score"]) == (0.9, 0.2)
    assert json.loads(json.dumps(pair)) == pair


def test_the_preference_pair_invents_no_id_its_source_row_lacks():
    """The CLI reads local JSONL, where ``--id_field`` need not be a column at all: a row without an
    id yields a record without one, not a KeyError."""
    record = _rm_pair({})
    assert "id" not in record, record


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
