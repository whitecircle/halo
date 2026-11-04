#!/usr/bin/env python
"""``balanced_class_weights`` across DP ranks.
With a pre-sharded dataset, local-shard weights would silently diverge across ranks, hence world-summed counts."""

import contextlib
from collections.abc import Iterator
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import torch
from datasets import Dataset
from sklearn.utils.class_weight import compute_class_weight

from halo.trainers.reward.classification_loss import balanced_class_weights, build_classification_loss

_MODULE = "halo.trainers.reward.classification_loss"


class _FakeDist:
    """``balanced_class_weights``'s collectives, as seen by one rank of a world holding ``other_ranks_labels``."""

    class ReduceOp:
        SUM = "sum"

    def __init__(self, other_ranks_labels: list[list[int]]):
        self._other_counts = [
            torch.from_numpy(np.bincount(np.asarray(labels, dtype=np.int64))).to(torch.float64)
            for labels in other_ranks_labels
        ]

    def rank_max(self, values: list[int]) -> list[int]:
        return [max(values[0], *(counts.numel() for counts in self._other_counts))]

    def all_reduce(self, tensor: torch.Tensor, op) -> None:
        if op != self.ReduceOp.SUM:
            raise AssertionError(f"unexpected reduce op: {op}")
        for counts in self._other_counts:
            padded = torch.zeros_like(tensor)
            padded[: counts.numel()] = counts.to(tensor.dtype)
            tensor.add_(padded)


@contextlib.contextmanager
def _fake_world(other_ranks_labels: list[list[int]]) -> Iterator[None]:
    fake = _FakeDist(other_ranks_labels)
    with (
        patch(f"{_MODULE}.dist", fake),
        patch(f"{_MODULE}.rank_max", fake.rank_max),
        patch(f"{_MODULE}.has_process_group", return_value=True),
    ):
        yield


def _weights_on_rank(rank_labels: list[int], other_ranks_labels: list[list[int]], model=None) -> torch.Tensor:
    with _fake_world(other_ranks_labels):
        return balanced_class_weights({"labels": rank_labels}, model)


def _local_weights(labels: list[int], model=None) -> torch.Tensor:
    return balanced_class_weights({"labels": labels}, model)


# Label distributions and local max labels differ (rank0 never sees class 2), so per-shard weights and lengths diverge.
RANK0_LABELS = [0, 0, 0, 1]
RANK1_LABELS = [1, 1, 2, 2]
# Global: counts [3, 3, 2], n_samples 8, 3 present classes -> balanced 8 / (3 * count).
GLOBAL_WEIGHTS = torch.tensor([8 / 9, 8 / 9, 8 / 6], dtype=torch.float32)


def test_presharded_ranks_produce_identical_global_weights():
    weights_rank0 = _weights_on_rank(RANK0_LABELS, [RANK1_LABELS])
    weights_rank1 = _weights_on_rank(RANK1_LABELS, [RANK0_LABELS])

    assert torch.equal(weights_rank0, weights_rank1), f"rank divergence: {weights_rank0} vs {weights_rank1}"
    assert torch.allclose(weights_rank0, GLOBAL_WEIGHTS, atol=1e-6), weights_rank0

    # A per-shard derivation gives different weights, so this fails if the global gather is dropped.
    assert not torch.allclose(_local_weights(RANK0_LABELS + [2]), GLOBAL_WEIGHTS, atol=1e-6)


def test_build_classification_loss_uses_global_counts():
    args = SimpleNamespace(
        loss_type="cross_entropy", class_weights=None, derive_class_weights=True, label_smoothing=0.0
    )

    with _fake_world([RANK1_LABELS]):
        loss = build_classification_loss(
            args, SimpleNamespace(config=SimpleNamespace(num_labels=3)), {"labels": RANK0_LABELS}, multi_label=False
        )

    assert loss.focal_gamma is None and not loss.multi_label
    assert torch.allclose(loss.weight, GLOBAL_WEIGHTS, atol=1e-6), loss.weight


@pytest.mark.parametrize("multi_label", [False, True])
@pytest.mark.parametrize("weights", [[2.0], [1.0, 2.0, 3.0, 4.0]], ids=["short", "long"])
def test_class_weights_must_give_one_weight_per_label(multi_label, weights):
    """A one-entry list would broadcast as ``pos_weight`` over a multi-label head instead of failing the loss."""
    args = SimpleNamespace(
        loss_type="cross_entropy", class_weights=weights, derive_class_weights=False, label_smoothing=0.0
    )
    model = SimpleNamespace(config=SimpleNamespace(num_labels=3))
    with pytest.raises(ValueError, match=f"class_weights has {len(weights)} entries.*num_labels=3"):
        build_classification_loss(args, model, None, multi_label=multi_label)
    args.class_weights = [1.0, 2.0, 3.0]
    assert build_classification_loss(args, model, None, multi_label=multi_label).weight.tolist() == [1.0, 2.0, 3.0]


def test_world_reduce_is_scale_invariant_for_tp_siblings():
    """TP siblings hold the same shard, so the world sum double-counts it uniformly and balanced weights don't move."""
    labels = [0, 0, 1, 2, 2, 2]
    tp2_weights = _weights_on_rank(labels, [labels])
    assert torch.allclose(tp2_weights, _local_weights(labels), atol=1e-6), tp2_weights


def test_non_distributed_path_matches_sklearn_balanced():
    labels = [0, 0, 0, 0, 1, 2, 2]
    expected = compute_class_weight("balanced", classes=np.unique(labels), y=np.asarray(labels))
    weights = _local_weights(labels)
    assert torch.allclose(weights, torch.tensor(expected, dtype=torch.float32), atol=1e-6), weights


def test_absent_class_gets_unit_weight_with_model_num_labels():
    model = SimpleNamespace(config=SimpleNamespace(num_labels=4))
    weights = _weights_on_rank(RANK0_LABELS, [RANK1_LABELS], model=model)
    assert weights.numel() == 4
    assert torch.allclose(weights[:3], GLOBAL_WEIGHTS, atol=1e-6), weights
    assert weights[3].item() == 1.0


def test_label_beyond_num_labels_raises():
    """Such a label would scatter out of range."""
    model = SimpleNamespace(config=SimpleNamespace(num_labels=3))
    with pytest.raises(ValueError, match="num_labels"):
        _local_weights([0, 1, 5], model=model)


def test_empty_labels_raise():
    with pytest.raises(ValueError, match="empty"):
        balanced_class_weights({"labels": np.array([], dtype=np.int64)}, None)


def test_resolves_singular_label_column_from_script_dataset():
    """``tokenize_classification_row`` emits ``label``, the ``_tokenize`` fallback ``labels``; both must work.
    A real HF Dataset makes the column name matter."""
    ds = Dataset.from_dict({"input_ids": [[1], [2], [3], [4]], "label": [0, 0, 0, 1]})
    weights = balanced_class_weights(ds, None)
    # counts [3, 1], n=4, 2 present -> balanced 4/(2*count) = [2/3, 2].
    assert torch.allclose(weights, torch.tensor([2 / 3, 2.0], dtype=torch.float32), atol=1e-6), weights


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
