#!/usr/bin/env python
"""
Tests for ClassificationTrainer derive_class_weights global-count aggregation.

With a pre-sharded dataset every DP rank holds a different shard; class weights must be
derived from the world-summed label counts, not the local shard, or ranks silently train
with divergent loss weighting. These tests simulate two ranks by faking the distributed
primitives inside src.trainers.reward.classification and assert both ranks produce
IDENTICAL weights equal to the global-count-derived ("balanced") weights.

Run: python tests/cpu/trainers/test_classification_class_weights.py
"""

from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import torch
import torch.nn as nn
from sklearn.utils.class_weight import compute_class_weight

from src.trainers.reward.classification import ClassificationTrainer

_MODULE = "src.trainers.reward.classification"


class _FakeDist:
    """Fake of the ``torch.distributed`` surface used by _balanced_class_weights.

    Emulates one rank of a world where ``other_ranks_labels`` are the label shards held
    by the OTHER ranks: MAX all-reduce over the count-vector length, SUM all-reduce over
    the padded counts.
    """

    class ReduceOp:
        MAX = "max"
        SUM = "sum"

    def __init__(self, other_ranks_labels: list[list[int]]):
        self._other_counts = [
            torch.from_numpy(np.bincount(np.asarray(labels, dtype=np.int64))).to(torch.float64)
            for labels in other_ranks_labels
        ]

    @staticmethod
    def is_available() -> bool:
        return True

    @staticmethod
    def is_initialized() -> bool:
        return True

    def all_reduce(self, tensor: torch.Tensor, op) -> None:
        if op == self.ReduceOp.MAX:
            for counts in self._other_counts:
                tensor.copy_(torch.maximum(tensor, torch.tensor(counts.numel(), dtype=tensor.dtype)))
        elif op == self.ReduceOp.SUM:
            for counts in self._other_counts:
                padded = torch.zeros_like(tensor)
                padded[: counts.numel()] = counts.to(tensor.dtype)
                tensor.add_(padded)
        else:
            raise AssertionError(f"unexpected reduce op: {op}")


def _weights_on_rank(rank_labels: list[int], other_ranks_labels: list[list[int]], model=None) -> torch.Tensor:
    """Run _balanced_class_weights as one simulated rank of a fake world."""
    with (
        patch(f"{_MODULE}.dist", _FakeDist(other_ranks_labels)),
        patch(f"{_MODULE}.current_device", return_value=torch.device("cpu")),
    ):
        return ClassificationTrainer._balanced_class_weights({"labels": rank_labels}, model)


def _local_weights(labels: list[int], model=None) -> torch.Tensor:
    """Run _balanced_class_weights with distributed NOT initialized (single-process path)."""
    return ClassificationTrainer._balanced_class_weights({"labels": labels}, model)


# Two DP shards with deliberately different label distributions AND different local max
# labels (rank0 never sees class 2), so per-shard derivation gives divergent weights and
# divergent count-vector lengths.
RANK0_LABELS = [0, 0, 0, 1]
RANK1_LABELS = [1, 1, 2, 2]
# Global: counts [3, 3, 2], n_samples 8, 3 present classes -> balanced 8 / (3 * count).
GLOBAL_WEIGHTS = torch.tensor([8 / 9, 8 / 9, 8 / 6], dtype=torch.float32)


def test_presharded_ranks_produce_identical_global_weights():
    """Both simulated ranks must derive the same weights, equal to the global-count weights."""
    weights_rank0 = _weights_on_rank(RANK0_LABELS, [RANK1_LABELS])
    weights_rank1 = _weights_on_rank(RANK1_LABELS, [RANK0_LABELS])

    assert torch.equal(weights_rank0, weights_rank1), f"rank divergence: {weights_rank0} vs {weights_rank1}"
    assert torch.allclose(weights_rank0, GLOBAL_WEIGHTS, atol=1e-6), weights_rank0

    # Sanity that the assertion above is not vacuous: a per-shard derivation (local labels only)
    # yields DIFFERENT weights, so this test fails if the global gather is dropped.
    assert not torch.allclose(_local_weights(RANK0_LABELS + [2]), GLOBAL_WEIGHTS, atol=1e-6)


def test_build_loss_fn_uses_global_counts():
    """The production path (_build_loss_fn) must weight CrossEntropyLoss by global counts."""
    trainer = ClassificationTrainer.__new__(ClassificationTrainer)
    trainer.is_multi_label = False
    args = SimpleNamespace(loss_type="cross_entropy", class_weights=None, derive_class_weights=True)

    with (
        patch(f"{_MODULE}.dist", _FakeDist([RANK1_LABELS])),
        patch(f"{_MODULE}.current_device", return_value=torch.device("cpu")),
    ):
        loss_fn = trainer._build_loss_fn(args, {"labels": RANK0_LABELS})

    assert isinstance(loss_fn, nn.CrossEntropyLoss)
    assert torch.allclose(loss_fn.weight, GLOBAL_WEIGHTS, atol=1e-6), loss_fn.weight


def test_world_reduce_is_scale_invariant_for_tp_siblings():
    """TP siblings hold the SAME DP shard; the world sum double-counts it uniformly.

    Balanced weights are invariant to a uniform count scale, so the world-group reduce
    must give exactly the single-copy weights.
    """
    labels = [0, 0, 1, 2, 2, 2]
    tp2_weights = _weights_on_rank(labels, [labels])  # sibling contributes identical counts
    assert torch.allclose(tp2_weights, _local_weights(labels), atol=1e-6), tp2_weights


def test_non_distributed_path_matches_sklearn_balanced():
    """Without torch.distributed, weights must match sklearn's 'balanced' derivation."""
    labels = [0, 0, 0, 0, 1, 2, 2]
    expected = compute_class_weight("balanced", classes=np.unique(labels), y=np.asarray(labels))
    weights = _local_weights(labels)
    assert torch.allclose(weights, torch.tensor(expected, dtype=torch.float32), atol=1e-6), weights


def test_absent_class_gets_unit_weight_with_model_num_labels():
    """Classes in [0, num_labels) absent from the global train set keep weight 1.0."""
    model = SimpleNamespace(config=SimpleNamespace(num_labels=4))
    weights = _weights_on_rank(RANK0_LABELS, [RANK1_LABELS], model=model)
    assert weights.numel() == 4
    assert torch.allclose(weights[:3], GLOBAL_WEIGHTS, atol=1e-6), weights
    assert weights[3].item() == 1.0


def test_label_beyond_num_labels_raises():
    """A train label >= model num_labels must fail loud, not scatter out of range."""
    model = SimpleNamespace(config=SimpleNamespace(num_labels=3))
    with pytest.raises(ValueError, match="num_labels"):
        _local_weights([0, 1, 5], model=model)


def test_empty_labels_raise():
    """An empty labels column must fail loud instead of returning an empty weight vector."""
    with pytest.raises(ValueError, match="empty"):
        ClassificationTrainer._balanced_class_weights({"labels": np.array([], dtype=np.int64)}, None)


def test_resolves_singular_label_column_from_script_dataset():
    """The classification script's preprocess_function emits a ``label`` column (singular), while the
    trainer's own _tokenize fallback emits ``labels`` — derive_class_weights must accept either.

    Hardcoding ``train_dataset["labels"]`` raises ``Column 'labels' doesn't exist`` at trainer
    construction on a script-produced Dataset. Uses a real HF Dataset (not the {"labels": ...} dict
    the other tests feed) so the column name is load-bearing.
    """
    from datasets import Dataset

    ds = Dataset.from_dict({"input_ids": [[1], [2], [3], [4]], "label": [0, 0, 0, 1]})
    weights = ClassificationTrainer._balanced_class_weights(ds, None)
    # counts [3, 1], n=4, 2 present -> balanced 4/(2*count) = [2/3, 2].
    assert torch.allclose(weights, torch.tensor([2 / 3, 2.0], dtype=torch.float32), atol=1e-6), weights


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
