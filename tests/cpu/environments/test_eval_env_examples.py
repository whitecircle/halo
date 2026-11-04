#!/usr/bin/env python
"""``eval_env.py``'s example builder: every field flag names a column of the split, since a mistyped one skips every
row or drops its value with nothing said. The default answer and id columns may be absent."""

import argparse

import pytest
from datasets import Dataset

from halo.entrypoints import generic_env_eval


class _StopBuild(Exception):
    @classmethod
    def stop(cls):
        raise cls


def _args(dataset: str, **flags) -> argparse.Namespace:
    defaults = {
        "dataset": dataset,
        "config": None,
        "split": "test",
        "prompt_field": "question",
        "answer_field": generic_env_eval.DEFAULT_ANSWER_FIELD,
        "context_fields": [],
        "group_by": None,
        "id_field": generic_env_eval.DEFAULT_ID_FIELD,
        "num_examples": 0,
    }
    return argparse.Namespace(**{**defaults, **flags})


@pytest.fixture
def saved(tmp_path) -> str:
    rows = {"question": ["q1", "q2"], "gold": ["a", "b"], "subject": ["x", "y"], "qid": ["p1", "p2"]}
    Dataset.from_dict(rows).save_to_disk(tmp_path / "ds")
    return str(tmp_path / "ds")


@pytest.mark.parametrize(
    ("flag", "value"),
    [
        ("prompt_field", "prompt"),
        ("answer_field", "answr"),
        ("context_fields", ["choices"]),
        ("group_by", "subjct"),
        ("id_field", "id_"),
    ],
)
def test_a_field_flag_naming_no_column_is_refused_before_any_row(saved, flag, value):
    with pytest.raises(SystemExit, match=f"--{flag} .*: no such column in .*available columns"):
        generic_env_eval.build_column_examples(_args(saved, **{flag: value}))


def test_the_default_answer_and_id_columns_may_be_absent(saved):
    examples = generic_env_eval.build_column_examples(_args(saved, context_fields=["subject"], group_by="subject"))
    assert [example["prompt"] for example in examples] == ["q1", "q2"]
    assert [example["context"] for example in examples] == [{"subject": "x"}, {"subject": "y"}]
    assert [example["id"] for example in examples] == [None, None]
    renamed = generic_env_eval.build_column_examples(_args(saved, answer_field="gold", id_field="qid"))
    assert [(example["context"]["answer"], example["id"]) for example in renamed] == [("a", "p1"), ("b", "p2")]


def test_the_generic_eval_refuses_a_turn_cap_in_env_kwargs(monkeypatch):
    """``--env_kwargs`` is laid over the flags, so it would silently beat ``--max_turns``."""
    built = []
    monkeypatch.setattr(generic_env_eval, "build_environment", lambda *args: built.append(args) or _StopBuild.stop())
    args = argparse.Namespace(training_config=None, env_type="exam_qa", max_turns=None, env_kwargs='{"max_turns": 3}')
    with pytest.raises(SystemExit, match="--max_turns, not --env_kwargs"):
        generic_env_eval.run_generic_env_eval(args)
    assert not built
    args.env_kwargs = '{"open_book": true}'
    with pytest.raises(_StopBuild):
        generic_env_eval.run_generic_env_eval(args)
    assert built == [("exam_qa", {"open_book": True})]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
