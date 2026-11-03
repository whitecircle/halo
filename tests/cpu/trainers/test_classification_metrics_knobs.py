#!/usr/bin/env python
"""Classification knobs ``compute_mcc``, ``compute_auc_roc``, ``compute_per_class_metrics`` and ``focal_alpha``.
The confusion matrices and rank orders are small enough to count by hand."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from scipy.special import softmax
from sklearn.metrics import matthews_corrcoef, roc_auc_score
from transformers.trainer_utils import EvalPrediction

from halo.configs.classification import ClassificationConfig
from halo.trainers.reward.classification import ClassificationTrainer, reject_unsupported_classification_args
from halo.trainers.reward.classification_loss import build_classification_loss

# Labels: 4 positives then 6 negatives; argmax gives TP=3 FN=1 FP=2 TN=4, so the two per-class rows differ.
# The logit-0 column varies, so softmax(p1) and sigmoid(logit1) rank the rows differently: 5/6 vs 7/8 AUC.
BINARY_LOGITS = np.array(
    [
        [0.0, 2.0],
        [0.5, 1.5],
        [-1.0, 0.2],
        [1.5, 0.4],
        [0.0, 0.3],
        [-2.0, -1.2],
        [1.0, -1.0],
        [0.5, -0.5],
        [2.0, 0.7],
        [0.0, -0.2],
    ]
)
BINARY_LABELS = np.array([1, 1, 1, 1, 0, 0, 0, 0, 0, 0])
BINARY_MCC = 1.0 / np.sqrt(6.0)  # (3*4 - 2*1) / sqrt(5*4*6*5)
BINARY_AUC_SOFTMAX = 5.0 / 6.0  # 20 of 24 positive/negative pairs correctly ranked
BINARY_AUC_SIGMOID_LOGIT1 = 0.875  # the mis-ranked form the code must not use

# Multiclass fixture: 5/2/1 class support (macro and weighted averages diverge), preds
# [0,0,1,0,2,1,0,2] give per-class f1 [2/3, 1/2, 2/3] -> weighted 0.625 but macro 11/18.
MULTICLASS_LOGITS = np.array(
    [
        [3.0, 0.5, -1.0],
        [2.0, 1.0, 0.0],
        [0.5, 1.5, -0.5],
        [1.0, -1.0, 0.5],
        [0.2, -0.3, 1.0],
        [-1.0, 2.0, 0.3],
        [1.5, 0.7, -0.2],
        [-0.5, 0.1, 2.5],
    ]
)
MULTICLASS_LABELS = np.array([0, 0, 0, 0, 0, 1, 1, 2])

# Multi-label fixture: per-column AUCs 7/9, 6/9 and 9/9 -> macro 22/27; thresholding at 0.5
# (logit >= 0) matches the target row on 5 of 6 rows.
MULTILABEL_LOGITS = np.array(
    [
        [2.0, -1.0, 0.5],
        [1.0, 2.0, -0.5],
        [-1.0, 0.5, 1.5],
        [-2.0, -0.5, -1.5],
        [-1.5, 3.0, -2.0],
        [-0.5, 0.1, 2.0],
    ]
)
MULTILABEL_TARGETS = np.array(
    [
        [1, 0, 1],
        [1, 1, 0],
        [0, 1, 1],
        [0, 0, 0],
        [1, 0, 0],
        [0, 1, 1],
    ]
)
MULTILABEL_AUC_MACRO = 22.0 / 27.0
MULTILABEL_EXACT_MATCH = 5.0 / 6.0


def _config(tmp_path, **overrides) -> ClassificationConfig:
    return ClassificationConfig(output_dir=str(tmp_path), **overrides)


def _metrics(config, logits, labels, *, is_binary=True, is_multi_label=False, label_names_list=None) -> dict:
    trainer = ClassificationTrainer.__new__(ClassificationTrainer)
    trainer.args = config
    trainer.is_binary = is_binary
    trainer.is_multi_label = is_multi_label
    trainer.label_names_list = label_names_list
    return trainer._default_compute_metrics(EvalPrediction(predictions=logits, label_ids=labels))


def _build_loss_fn(config, *, is_multi_label: bool):
    model = SimpleNamespace(config=SimpleNamespace(num_labels=2))
    return build_classification_loss(config, model, None, multi_label=is_multi_label)


def test_shipped_defaults_compute_mcc_and_nothing_else(tmp_path):
    metrics = _metrics(_config(tmp_path), BINARY_LOGITS, BINARY_LABELS)

    assert metrics["mcc"] == pytest.approx(BINARY_MCC)
    assert "auc_roc" not in metrics
    assert [key for key in metrics if "_class_" in key] == []
    # Binary averaging: weighted would give precision 0.72 on this fixture, not 0.6.
    assert metrics["accuracy"] == pytest.approx(0.7)
    assert metrics["precision"] == pytest.approx(0.6)
    assert metrics["recall"] == pytest.approx(0.75)
    assert metrics["f1"] == pytest.approx(2 / 3)


def test_compute_mcc_off_drops_only_the_mcc_key(tmp_path):
    metrics = _metrics(_config(tmp_path, compute_mcc=False), BINARY_LOGITS, BINARY_LABELS)

    assert "mcc" not in metrics
    assert metrics["accuracy"] == pytest.approx(0.7)


def test_compute_mcc_on_multiclass_matches_sklearn(tmp_path):
    metrics = _metrics(_config(tmp_path), MULTICLASS_LOGITS, MULTICLASS_LABELS, is_binary=False)

    preds = MULTICLASS_LOGITS.argmax(axis=-1)
    assert metrics["mcc"] == pytest.approx(matthews_corrcoef(MULTICLASS_LABELS, preds))
    assert metrics["mcc"] != pytest.approx(metrics["accuracy"])


def test_compute_auc_roc_binary_ranks_by_softmax_positive_class(tmp_path):
    metrics = _metrics(_config(tmp_path, compute_auc_roc=True), BINARY_LOGITS, BINARY_LABELS)

    assert metrics["auc_roc"] == pytest.approx(BINARY_AUC_SOFTMAX)
    mis_ranked = roc_auc_score(BINARY_LABELS, 1.0 / (1.0 + np.exp(-BINARY_LOGITS[:, 1])))
    assert mis_ranked == pytest.approx(BINARY_AUC_SIGMOID_LOGIT1)


def test_compute_auc_roc_multiclass_is_one_vs_rest_macro(tmp_path):
    metrics = _metrics(_config(tmp_path, compute_auc_roc=True), MULTICLASS_LOGITS, MULTICLASS_LABELS, is_binary=False)

    probs = softmax(MULTICLASS_LOGITS, axis=-1)
    expected = roc_auc_score(MULTICLASS_LABELS, probs, multi_class="ovr", average="macro")
    assert metrics["auc_roc"] == pytest.approx(expected)
    weighted = roc_auc_score(MULTICLASS_LABELS, probs, multi_class="ovr", average="weighted")
    assert abs(expected - weighted) > 1e-3


def test_compute_auc_roc_multi_label_is_macro_over_columns(tmp_path):
    metrics = _metrics(
        _config(tmp_path, compute_auc_roc=True),
        MULTILABEL_LOGITS,
        MULTILABEL_TARGETS,
        is_binary=False,
        is_multi_label=True,
    )

    assert metrics["auc_roc"] == pytest.approx(MULTILABEL_AUC_MACRO)
    micro = roc_auc_score(MULTILABEL_TARGETS, MULTILABEL_LOGITS, average="micro")
    assert abs(MULTILABEL_AUC_MACRO - micro) > 1e-3


def test_compute_auc_roc_skipped_when_a_class_is_absent_from_the_batch(tmp_path):
    labels = np.array([0, 0, 0, 0, 0, 1, 1, 1])  # class 2 absent, but the head still emits 3 columns
    metrics = _metrics(_config(tmp_path, compute_auc_roc=True), MULTICLASS_LOGITS, labels, is_binary=False)

    assert "auc_roc" not in metrics
    assert metrics["accuracy"] == pytest.approx(0.5)


def test_single_class_binary_batch_omits_auc_roc_instead_of_logging_nan(tmp_path):
    """sklearn 1.9 warns and returns NaN here instead of raising.
    A NaN ``metric_for_best_model`` silently freezes best-model tracking."""
    one_class = np.zeros(len(BINARY_LABELS), dtype=int)
    metrics = _metrics(_config(tmp_path, compute_auc_roc=True), BINARY_LOGITS, one_class)

    assert "auc_roc" not in metrics
    assert not any(isinstance(v, float) and np.isnan(v) for v in metrics.values())
    assert metrics["accuracy"] == pytest.approx((BINARY_LOGITS.argmax(axis=-1) == one_class).mean())


@pytest.mark.parametrize(
    ("labels", "is_binary"),
    [
        (np.zeros(len(BINARY_LABELS), dtype=int), True),  # NaN path (sklearn warns)
        (np.array([0, 0, 0, 0, 0, 1, 1, 1]), False),  # raising path (class 2 absent)
    ],
)
def test_omitted_auc_roc_raises_when_checkpoints_are_ranked_by_it(tmp_path, labels, is_binary):
    """``_determine_best_metric`` would raise a bare KeyError in the save path; fail here with the cause and remedy."""
    config = _config(tmp_path, compute_auc_roc=True, metric_for_best_model="auc_roc")
    logits = BINARY_LOGITS if is_binary else MULTICLASS_LOGITS

    with pytest.raises(ValueError, match="auc_roc is undefined"):
        _metrics(config, logits, labels, is_binary=is_binary)


def test_ranking_on_auc_roc_with_the_knob_off_is_refused_at_construction(tmp_path):
    with pytest.raises(ValueError, match="compute_auc_roc is off"):
        reject_unsupported_classification_args(
            _config(tmp_path, compute_auc_roc=False, metric_for_best_model="auc_roc"), default_metrics=True
        )

    # The eval_-prefixed spelling is caught too, and a metric the trainer always emits stays accepted.
    with pytest.raises(ValueError, match="compute_auc_roc is off"):
        reject_unsupported_classification_args(
            _config(tmp_path, metric_for_best_model="eval_auc_roc"), default_metrics=True
        )
    reject_unsupported_classification_args(_config(tmp_path, metric_for_best_model="f1"), default_metrics=True)
    reject_unsupported_classification_args(
        _config(tmp_path, compute_auc_roc=True, metric_for_best_model="auc_roc"), default_metrics=True
    )
    # A caller's own compute_metrics decides its own keys.
    reject_unsupported_classification_args(_config(tmp_path, metric_for_best_model="auc_roc"), default_metrics=False)


def test_construction_refusals_run_label_smoothing_then_best_metric_then_max_length(tmp_path):
    every_refusal = {"label_smoothing_factor": 0.1, "metric_for_best_model": "auc_roc", "max_length": 0}
    with pytest.raises(ValueError, match="label_smoothing_factor=0.1"):
        reject_unsupported_classification_args(_config(tmp_path, **every_refusal), default_metrics=True)
    every_refusal.pop("label_smoothing_factor")
    with pytest.raises(ValueError, match="compute_auc_roc is off"):
        reject_unsupported_classification_args(_config(tmp_path, **every_refusal), default_metrics=True)
    with pytest.raises(ValueError, match="max_length must be a positive int"):
        reject_unsupported_classification_args(_config(tmp_path, **every_refusal), default_metrics=False)


def test_per_class_metrics_emit_named_keys_with_per_class_values(tmp_path):
    metrics = _metrics(
        _config(tmp_path, compute_per_class_metrics=True),
        BINARY_LOGITS,
        BINARY_LABELS,
        label_names_list=["negative", "positive"],
    )

    assert {key for key in metrics if "_class_" in key} == {
        "f1_class_negative",
        "precision_class_negative",
        "recall_class_negative",
        "f1_class_positive",
        "precision_class_positive",
        "recall_class_positive",
    }
    # Class 0: TP=4 FP=1 FN=2; class 1: TP=3 FP=2 FN=1.
    assert metrics["precision_class_negative"] == pytest.approx(0.8)
    assert metrics["recall_class_negative"] == pytest.approx(2 / 3)
    assert metrics["f1_class_negative"] == pytest.approx(8 / 11)
    assert metrics["precision_class_positive"] == pytest.approx(0.6)
    assert metrics["recall_class_positive"] == pytest.approx(0.75)
    assert metrics["f1_class_positive"] == pytest.approx(2 / 3)


def test_per_class_metrics_fall_back_to_indices_beyond_the_names_list(tmp_path):
    metrics = _metrics(
        _config(tmp_path, compute_per_class_metrics=True),
        MULTICLASS_LOGITS,
        MULTICLASS_LABELS,
        is_binary=False,
        label_names_list=["neg"],
    )

    assert {key for key in metrics if "_class_" in key} == {
        "f1_class_neg",
        "precision_class_neg",
        "recall_class_neg",
        "f1_class_1",
        "precision_class_1",
        "recall_class_1",
        "f1_class_2",
        "precision_class_2",
        "recall_class_2",
    }
    assert metrics["precision_class_neg"] == pytest.approx(0.75)
    assert metrics["recall_class_neg"] == pytest.approx(0.6)
    assert metrics["f1_class_neg"] == pytest.approx(2 / 3)
    assert metrics["f1_class_1"] == pytest.approx(0.5)
    assert metrics["precision_class_2"] == pytest.approx(0.5)
    assert metrics["recall_class_2"] == pytest.approx(1.0)
    # Non-binary aggregates are weighted by support; macro f1 here would be 11/18.
    assert metrics["f1"] == pytest.approx(0.625)
    assert metrics["precision"] == pytest.approx(0.65625)


def test_per_class_metrics_off_emit_no_per_class_keys(tmp_path):
    metrics = _metrics(
        _config(tmp_path, compute_per_class_metrics=False),
        BINARY_LOGITS,
        BINARY_LABELS,
        label_names_list=["negative", "positive"],
    )

    assert [key for key in metrics if "_class_" in key] == []


def test_multi_label_skips_mcc_and_per_class_metrics(tmp_path):
    """``matthews_corrcoef`` and ``average=None`` reject multi-hot targets."""
    metrics = _metrics(
        _config(tmp_path, compute_mcc=True, compute_per_class_metrics=True),
        MULTILABEL_LOGITS,
        MULTILABEL_TARGETS,
        is_binary=False,
        is_multi_label=True,
        label_names_list=["a", "b", "c"],
    )

    assert "mcc" not in metrics
    assert [key for key in metrics if "_class_" in key] == []
    assert metrics["exact_match_accuracy"] == pytest.approx(MULTILABEL_EXACT_MATCH)


def test_focal_alpha_rejected_on_single_label_head(tmp_path):
    """A scalar alpha on a softmax head only rescales the loss uniformly."""
    with pytest.raises(ValueError, match="focal_alpha balances positives against negatives"):
        _build_loss_fn(_config(tmp_path, loss_type="focal", focal_alpha=0.25), is_multi_label=False)

    # Without alpha it builds, so the raise comes from focal_alpha alone.
    assert _build_loss_fn(_config(tmp_path, loss_type="focal"), is_multi_label=False) is not None


def test_focal_alpha_balances_the_multi_label_focal_loss(tmp_path):
    alpha, gamma = 0.25, 2.0
    loss_fn = _build_loss_fn(
        _config(tmp_path, loss_type="focal", focal_alpha=alpha, focal_gamma=gamma), is_multi_label=True
    )

    logits = torch.tensor(MULTILABEL_LOGITS, dtype=torch.float32)
    targets = torch.tensor(MULTILABEL_TARGETS, dtype=torch.float32)
    ce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    focal = ((1 - torch.exp(-ce)) ** gamma) * ce
    expected = ((alpha * targets + (1 - alpha) * (1 - targets)) * focal).mean()

    assert loss_fn.mean(logits, targets).item() == pytest.approx(expected.item(), rel=1e-6)
    assert loss_fn.mean(logits, targets).item() != pytest.approx(focal.mean().item(), rel=1e-3)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
