#!/usr/bin/env python
"""classification.py must validate its input columns and labels BEFORE the model load.

``tokenize_classification_row`` reads a pre-built ``prompt`` conversation, falling back to
``text_field`` — with neither column present the failure surfaces as a per-row ``KeyError`` deep in
the dataset map, after the full (possibly multi-node) model load. The config-time guard raises
first, naming what is missing.

A single-label ``-1`` "no label" row has no class id: refused in a split the run reads, and the
split holding it left untokenized when the run never reads it (a GLUE-style unlabeled test split
with evaluation off).

Run: pytest tests/cpu/config/test_classification_input_columns.py
"""

import sys
import types
from unittest import mock

import pytest
from accelerate import PartialState
from datasets import Dataset, DatasetDict

from tests.common.utils import load_script_module

_TURNS = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}]


class _ReachedTokenization(Exception):
    """Raised by the stubbed map: the labels were accepted, and it carries the splits handed to it."""


@pytest.fixture(scope="module")
def classification():
    PartialState()  # the script logs through accelerate's logger, which needs the state initialized
    return load_script_module("scripts/training/classification.py", "halo_test_classification_script")


def test_prompt_column_passes(classification):
    classification.require_prompt_or_text_column(["prompt", "label"], text_field=None)


def test_text_field_naming_an_existing_column_passes(classification):
    classification.require_prompt_or_text_column(["text", "label"], text_field="text")


def test_neither_prompt_nor_text_field_raises(classification):
    with pytest.raises(ValueError, match="text_field is not set"):
        classification.require_prompt_or_text_column(["document", "label"], text_field=None)


def test_text_field_naming_a_missing_column_raises_with_the_name(classification):
    with pytest.raises(ValueError, match="text_field='body' names no existing column"):
        classification.require_prompt_or_text_column(["document", "label"], text_field="body")


def test_a_multi_label_sentinel_is_absence_in_every_split(classification):
    labels = {"train": {"-1", "a"}, "test": {"-1", "b"}}
    classification.refuse_unlabeled_rows(labels, ("train", "test"), is_multi_label=True)
    assert classification.build_label_list(labels) == ["a", "b"]


@pytest.mark.parametrize(
    ("split", "remedy"),
    [("train", r"dataset\.filter"), ("test", r"dataset: <hub id>@train\) and cut the eval split")],
)
def test_a_single_label_sentinel_in_a_read_split_is_refused_by_name(classification, split, remedy):
    """A single-label ``-1`` row has no class: passed through it is an out-of-range cross-entropy
    target, and its string spelling a KeyError in the map, both after the model load."""
    labels = {"train": {"a", "b"}, "test": {"a", "b"}}
    labels[split] = labels[split] | {"-1"}
    with pytest.raises(ValueError, match=rf"The '{split}' split .*'no label' sentinel.*{remedy}"):
        classification.refuse_unlabeled_rows(labels, ("train", "test"), is_multi_label=False)


def test_a_single_label_sentinel_in_an_unread_split_is_dropped_from_the_classes(classification):
    labels = {"train": {"a", "b"}, "validation": {"-1", "a"}, "test": {"-1"}}
    classification.refuse_unlabeled_rows(labels, ("train",), is_multi_label=False)
    assert classification.build_label_list(labels) == ["a", "b"]


def test_an_eval_only_label_joins_the_classes(classification):
    assert classification.build_label_list({"train": {"a", "b"}, "test": {"c"}}) == ["a", "b", "c"]


def _run_to_tokenization(classification, tmp_path, eval_strategy: str, test_labels: list) -> list[str]:
    """Run ``main()`` over a single-label dataset to its tokenization map; returns the splits it maps."""
    config = tmp_path / "config.yaml"
    config.write_text(
        f"model_name_or_path: stub/qwen3-4b\ndataset:\n- dummy/dataset\noutput_dir: {tmp_path / 'out'}\n"
        f"bf16: false\nuse_cpu: true\nmax_length: 512\neval_strategy: {eval_strategy}\n"
    )
    train = Dataset.from_dict({"prompt": [_TURNS, _TURNS], "label": [0, 1]})
    test = Dataset.from_dict({"prompt": [_TURNS] * len(test_labels), "label": test_labels})
    runtime = types.SimpleNamespace(parallelism_config=types.SimpleNamespace(), model_source="stub/qwen3-4b")
    model = types.SimpleNamespace(config=types.SimpleNamespace())

    def _map(dataset, *args, **kwargs):
        raise _ReachedTokenization(sorted(dataset))

    patches = [
        mock.patch.object(classification, "run_training", lambda fn: fn),
        mock.patch.object(classification, "init_training_script", return_value=runtime),
        mock.patch.object(
            classification, "load_script_datasets", return_value=(DatasetDict({"train": train, "test": test}), False)
        ),
        mock.patch.object(classification, "require_multimodal_sequence_classification_head"),
        mock.patch.object(classification, "load_script_model", return_value=(model, object())),
        mock.patch.object(classification, "apply_max_length", side_effect=lambda config, args, model, tok: tok),
        mock.patch.object(classification, "setup_peft_model", return_value=None),
        mock.patch.object(classification, "log_model_info"),
        mock.patch.object(classification, "coordinated_map", side_effect=_map),
        mock.patch("src.training.parser.install_log_tee"),
        mock.patch.object(sys, "argv", ["prog", str(config)]),
    ]
    for patch in patches:
        patch.start()
    try:
        classification.main()
    except _ReachedTokenization as reached:
        return reached.args[0]
    finally:
        for patch in reversed(patches):
            patch.stop()
    raise AssertionError("main() returned without reaching the tokenization map")


def test_an_unlabeled_test_split_is_left_untokenized_when_the_run_does_not_evaluate(classification, tmp_path):
    assert _run_to_tokenization(classification, tmp_path, "no", test_labels=[-1, -1]) == ["train"]


def test_an_unlabeled_test_split_is_refused_when_the_run_evaluates(classification, tmp_path):
    with pytest.raises(ValueError, match="The 'test' split"):
        _run_to_tokenization(classification, tmp_path, "steps", test_labels=[-1, -1])


def test_a_labeled_test_split_is_tokenized_when_the_run_evaluates(classification, tmp_path):
    assert _run_to_tokenization(classification, tmp_path, "steps", test_labels=[0, 1]) == ["test", "train"]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
