#!/usr/bin/env python
"""The reward-model scripts read their JSONL files with every field kept as the value ``json`` parses.

pandas' ``read_json`` rewrites values: by default a string answer ``"579"`` reaches the reward model as
``579.0`` once another row lacks the field, an id ``"007"`` becomes ``7`` and a ``*_at`` column becomes
timestamps; with its inference off an int answer beside a missing one still reads ``5.0`` and ``0.3``
reads ``0.30000000000000004``. The prompts and the resume output go through one reader, so the ids a
re-run skips are the ids the prompts carry.

Run: python tests/cpu/inference/test_rm_jsonl_field_types.py
"""

import json
import types

import pytest

from scripts.inference.reward_model import _common as rm_common

_PROMPT = [{"role": "user", "content": "q"}]
_ROWS = [
    {"id": "007", "prompt": _PROMPT, "correct_answer": "579", "created_at": "2024-01-01"},
    {"id": "008", "prompt": _PROMPT},
]
_ANSWER_ARGS = types.SimpleNamespace(correct_answer_field="correct_answer")


def _args(tmp_path, rows=_ROWS):
    prompts = tmp_path / "prompts.jsonl"
    prompts.write_text("".join(json.dumps(row) + "\n" for row in rows))
    return types.SimpleNamespace(prompts_source=str(prompts), prompt_field="prompt", id_field="id")


def _answer_context(row) -> str | None:
    """The answer line the reward model is shown for ``row``, or None when it is shown none."""
    answer = rm_common.resolve_correct_answer(row, _ANSWER_ARGS)
    context = rm_common.prepend_answer_context(list(_PROMPT), answer)
    return context[0]["content"] if len(context) > len(_PROMPT) else None


def test_string_fields_reach_the_reward_model_unchanged(tmp_path):
    row = rm_common.load_prompts_dataframe(_args(tmp_path)).iloc[0]

    assert row["id"] == "007"
    assert row["created_at"] == "2024-01-01"
    assert row["prompt"] == _PROMPT
    assert _answer_context(row) == "The correct final answer must be: 579"


@pytest.mark.parametrize(
    ("answers", "shown"),
    [
        ([5, "absent"], ["5", None]),
        ([0.3, 2.5], ["0.3", "2.5"]),
        ([7, None], ["7", None]),
    ],
    ids=["int-beside-a-missing-answer", "float", "int-beside-a-null-answer"],
)
def test_numeric_answers_reach_the_reward_model_as_written(answers, shown, tmp_path):
    """A numeric answer column is rendered into the prompt as the JSON wrote it: an int stays an int
    beside a row lacking the field or holding null, and a float keeps its shortest spelling. A null or
    absent answer adds no line."""
    rows = [
        {"id": str(i), "prompt": _PROMPT, **({} if answer == "absent" else {"correct_answer": answer})}
        for i, answer in enumerate(answers)
    ]
    frame = rm_common.load_prompts_dataframe(_args(tmp_path, rows))

    expected = [None if value is None else f"The correct final answer must be: {value}" for value in shown]
    assert [_answer_context(row) for _, row in frame.iterrows()] == expected


def test_a_resumed_id_matches_the_prompt_it_was_written_for(tmp_path):
    prompts = rm_common.load_prompts_dataframe(_args(tmp_path))
    output = tmp_path / "out.jsonl"
    rm_common.append_jsonl_record(output, {"id": "007", "reward": 1.0})

    processed, _existing = rm_common.load_local_jsonl_resume(output, "id")

    assert processed == {"007"}
    assert prompts[~prompts["id"].isin(processed)]["id"].tolist() == ["008"]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
