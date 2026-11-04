#!/usr/bin/env python
"""classification.py validates its input columns and labels before the model load; a missing column would otherwise
surface as a per-row ``KeyError`` deep in the dataset map. A single-label ``-1`` row is refused in a split the
run reads, and that split is left untokenized when the run never reads it."""

import contextlib
import sys
import types
from unittest import mock

import pytest
from datasets import Dataset, DatasetDict

from halo.configs.classification import ClassificationConfig
from tests.common.repo import load_script_module

_CPU_OK = {"output_dir": "/tmp/test_classification_metric_knobs", "bf16": False, "use_cpu": True}
_TURNS = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}]


class _ReachedTokenization(Exception):
    """Raised by the stubbed map: the labels were accepted, and it carries the splits handed to it."""


@pytest.fixture(scope="module")
def classification():
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
    classification.reject_unlabeled_rows(labels, ("train", "test"), is_multi_label=True)
    assert classification.build_label_list(labels) == ["a", "b"]


@pytest.mark.parametrize(
    ("split", "remedy"),
    [("train", r"dataset\.filter"), ("test", r"dataset: <hub id>@train\) and cut the eval split")],
)
def test_a_single_label_sentinel_in_a_read_split_is_refused_by_name(classification, split, remedy):
    """A single-label ``-1`` row passed through is an out-of-range cross-entropy target, and its string spelling a
    KeyError in the map, both after the model load."""
    labels = {"train": {"a", "b"}, "test": {"a", "b"}}
    labels[split] = labels[split] | {"-1"}
    with pytest.raises(ValueError, match=rf"The '{split}' split .*'no label' sentinel.*{remedy}"):
        classification.reject_unlabeled_rows(labels, ("train", "test"), is_multi_label=False)


def test_a_single_label_sentinel_in_an_unread_split_is_dropped_from_the_classes(classification):
    labels = {"train": {"a", "b"}, "validation": {"-1", "a"}, "test": {"-1"}}
    classification.reject_unlabeled_rows(labels, ("train",), is_multi_label=False)
    assert classification.build_label_list(labels) == ["a", "b"]


def test_a_multi_label_dataset_refuses_per_class_metrics_and_takes_a_threshold(classification):
    per_class = ClassificationConfig(compute_per_class_metrics=True, **_CPU_OK)
    with pytest.raises(ValueError, match="multi-label dataset does not support.*compute_per_class_metrics"):
        classification.reject_unread_metric_knobs(per_class, is_multi_label=True)
    classification.reject_unread_metric_knobs(per_class, is_multi_label=False)
    classification.reject_unread_metric_knobs(ClassificationConfig(multi_label_threshold=0.7, **_CPU_OK), True)


def test_a_single_label_dataset_refuses_a_threshold(classification):
    with pytest.raises(ValueError, match="single-label dataset does not support.*multi_label_threshold"):
        classification.reject_unread_metric_knobs(ClassificationConfig(multi_label_threshold=0.7, **_CPU_OK), False)
    classification.reject_unread_metric_knobs(ClassificationConfig(**_CPU_OK), False)


def test_an_eval_only_label_joins_the_classes(classification):
    assert classification.build_label_list({"train": {"a", "b"}, "test": {"c"}}) == ["a", "b", "c"]


def test_an_empty_split_contributes_no_labels(classification):
    """A rank without rows must still reach the label-set gather and not raise before it while its peers wait."""
    empty = Dataset.from_dict({"prompt": [], "label": []})
    ds = DatasetDict({"train": Dataset.from_dict({"prompt": [_TURNS], "label": [0]}), "test": empty})
    assert classification.labels_in_split(ds, "test") == []


def test_an_empty_presharded_train_shard_is_refused_before_its_label_type_is_read(classification):
    """The label type is read from the first train row; a pre-sharded rank without one is refused on every rank, not
    alone ahead of the label-set gather."""
    empty = Dataset.from_dict({"prompt": [], "label": []})
    ds = DatasetDict({"train": empty})
    with pytest.raises(ValueError, match="train shard holds no rows"):
        classification.split_label_sets(ds, dataset_presharded=True)
    assert classification.split_label_sets(ds, dataset_presharded=False) == {"train": set()}


def _labeled_split(labels: list | None, rows: int = 2) -> Dataset:
    """No ``label`` column for ``None``."""
    return Dataset.from_dict(
        {"prompt": [_TURNS] * rows} if labels is None else {"prompt": [_TURNS] * len(labels), "label": labels}
    )


def _run_to_tokenization(
    classification, tmp_path, eval_strategy: str, test_labels: list | None, extra: str = "", train_labels=(0, 1)
) -> list[str]:
    """Run ``main()`` over a single-label dataset to its tokenization map; returns the splits it maps. ``None`` labels
    leave the split without a ``label`` column."""
    config = tmp_path / "config.yaml"
    config.write_text(
        f"model_name_or_path: stub/qwen3-4b\ndataset:\n- dummy/dataset\noutput_dir: {tmp_path / 'out'}\n"
        f"bf16: false\nuse_cpu: true\nmax_length: 512\neval_strategy: {eval_strategy}\n{extra}"
    )
    train = _labeled_split(train_labels and list(train_labels))
    test = _labeled_split(test_labels)
    runtime = types.SimpleNamespace(parallelism_config=types.SimpleNamespace(), model_source="stub/qwen3-4b")
    model = types.SimpleNamespace(config=types.SimpleNamespace())

    def _map(dataset, *args, **kwargs):
        raise _ReachedTokenization(sorted(dataset))

    patches = [
        mock.patch.object(classification, "init_training_script", return_value=runtime),
        mock.patch.object(
            classification, "load_script_datasets", return_value=(DatasetDict({"train": train, "test": test}), False)
        ),
        mock.patch.object(classification, "reject_multimodal_model_without_seq_cls_head"),
        mock.patch.object(classification, "load_model_and_tokenizer", return_value=(model, object())),
        mock.patch.object(classification, "apply_max_length", side_effect=lambda config, args, model, tok: tok),
        mock.patch.object(classification, "configure_adapter_training", return_value=None),
        mock.patch.object(classification, "log_model_info"),
        mock.patch.object(classification, "coordinated_map", side_effect=_map),
        mock.patch("halo.entrypoints.script.install_log_tee"),
        mock.patch.object(sys, "argv", ["prog", str(config)]),
    ]
    with contextlib.ExitStack() as stack:
        for patch in patches:
            stack.enter_context(patch)
        try:
            classification.main()
        except _ReachedTokenization as reached:
            return reached.args[0]
    raise AssertionError("main() returned without reaching the tokenization map")


def test_main_refuses_a_metric_knob_the_dataset_kind_does_not_read(classification, tmp_path):
    with pytest.raises(ValueError, match="single-label dataset does not support.*multi_label_threshold"):
        _run_to_tokenization(classification, tmp_path, "no", [0, 1], extra="multi_label_threshold: 0.7\n")


def test_main_refuses_a_class_weight_count_before_the_load(classification, tmp_path):
    with pytest.raises(ValueError, match="class_weights has 3 entries.*num_labels=2"):
        _run_to_tokenization(classification, tmp_path, "no", [0, 1], extra="class_weights: [1.0, 2.0, 3.0]\n")
    assert _run_to_tokenization(classification, tmp_path, "no", [0, 1], extra="class_weights: [1.0, 2.0]\n")


def test_an_unlabeled_test_split_is_left_untokenized_when_the_run_does_not_evaluate(classification, tmp_path):
    assert _run_to_tokenization(classification, tmp_path, "no", test_labels=[-1, -1]) == ["train"]


def test_an_unlabeled_test_split_is_refused_when_the_run_evaluates(classification, tmp_path):
    with pytest.raises(ValueError, match="The 'test' split"):
        _run_to_tokenization(classification, tmp_path, "steps", test_labels=[-1, -1])


def test_a_labeled_test_split_is_tokenized_when_the_run_evaluates(classification, tmp_path):
    assert _run_to_tokenization(classification, tmp_path, "steps", test_labels=[0, 1]) == ["test", "train"]


def test_a_test_split_with_no_label_column_trains_when_the_run_does_not_evaluate(classification, tmp_path):
    assert _run_to_tokenization(classification, tmp_path, "no", test_labels=None) == ["train"]


def test_a_test_split_with_no_label_column_is_refused_by_name_when_the_run_evaluates(classification, tmp_path):
    with pytest.raises(ValueError, match=r"The 'test' split .*no 'label' column.*eval_strategy: no"):
        _run_to_tokenization(classification, tmp_path, "steps", test_labels=None)


def test_a_train_split_with_no_label_column_is_refused_by_name(classification, tmp_path):
    with pytest.raises(ValueError, match=r"The 'train' split .*no 'label' column.*Add a 'label' column"):
        _run_to_tokenization(classification, tmp_path, "no", test_labels=[0, 1], train_labels=None)


def test_a_rowless_read_split_with_no_label_column_is_not_refused(classification):
    """A pre-sharded rank without rows has no schema to answer for: its peers' verdict decides."""
    ds = DatasetDict({"train": _labeled_split([0, 1]), "test": _labeled_split(None, rows=0)})
    classification.reject_read_splits_without_labels(ds, ("train", "test"))


@pytest.mark.parametrize("dataset_presharded", [False, True], ids=["whole", "presharded"])
def test_an_unread_split_with_no_label_column_adds_no_class(classification, dataset_presharded):
    ds = DatasetDict({"train": _labeled_split([0, 1]), "validation": _labeled_split(None)})
    labels = classification.split_label_sets(ds, dataset_presharded=dataset_presharded)
    assert labels == {"train": {"0", "1"}, "validation": set()}
    assert classification.build_label_list(labels) == ["0", "1"]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
