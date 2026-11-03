"""The classification trainer's objective: one configured loss, in fp32, on pooled logits."""

from dataclasses import dataclass, replace

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from datasets import Dataset

from halo.runtime.collectives import collective_device
from halo.runtime.consensus import rank_max
from halo.runtime.ranks import has_process_group


def loss_inputs_fp32(logits: torch.Tensor, targets: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """A (weighted) cross-entropy is a cancelling difference whose bf16 mantissa resolves ~1e-2 absolute, enough to
    reorder near-tied classes and round the class weights. Float targets follow, since BCE takes its output dtype from
    the target."""
    return logits.float(), targets.float() if targets.is_floating_point() else targets


def focal_loss(logits, labels, gamma=2.0, alpha=None, weight=None, reduction="mean"):
    """Single-label (``[B]`` class ids) and multi-label (``[B, C]`` multi-hot); ``alpha`` is RetinaNet's ``alpha_t``,
    multi-label only. The modulator reads the unweighted cross-entropy: ``exp(-ce)`` on a ``w_y``-weighted one is
    ``p_y ** w_y``, so a class weight would distort the modulator instead of scaling the loss."""
    if labels.dim() == 1:
        ce_loss = F.cross_entropy(logits, labels, reduction="none")
        weighted = ce_loss if weight is None else F.cross_entropy(logits, labels, weight=weight, reduction="none")
        # Every softmax element has exactly one target, so a scalar alpha would be a uniform
        # rescale, not balancing; the builder refuses it there and routes to ``weight``.
        alpha_t = None
    else:
        ce_loss = F.binary_cross_entropy_with_logits(logits, labels, reduction="none")
        weighted = (
            ce_loss
            if weight is None
            else F.binary_cross_entropy_with_logits(logits, labels, pos_weight=weight, reduction="none")
        )
        alpha_t = None if alpha is None else alpha * labels + (1.0 - alpha) * (1.0 - labels)
    pt = torch.exp(-ce_loss)
    focal = ((1 - pt) ** gamma) * weighted
    if alpha_t is not None:
        focal = alpha_t * focal
    if reduction == "none":
        return focal
    if reduction != "mean":
        raise ValueError(f"focal_loss takes reduction 'none' (the pipeline seam) or 'mean', got {reduction!r}")
    return focal.mean()


def balanced_class_weights(train_dataset: Dataset, model: nn.Module | None) -> torch.Tensor:
    """COLLECTIVE on a process group. World label counts: DP shards are disjoint, and TP siblings or replicated
    datasets scale every count alike, to which the balanced weights are invariant."""
    columns = train_dataset.column_names if hasattr(train_dataset, "column_names") else list(train_dataset)
    label_column = next((c for c in ("labels", "label") if c in columns), None)
    if label_column is None:
        raise ValueError(
            f"derive_class_weights: the train dataset has no 'labels'/'label' column (columns: {list(columns)})"
        )
    labels_array = np.asarray(train_dataset[label_column])
    counts = torch.from_numpy(np.bincount(labels_array)).to(torch.float64)

    if has_process_group():
        # Ranks can disagree on the local max label, so pad to a rank-uniform length first.
        (length,) = rank_max([counts.numel()])
        padded = torch.zeros(length, dtype=counts.dtype, device=collective_device())
        padded[: counts.numel()] = counts
        dist.all_reduce(padded, op=dist.ReduceOp.SUM)
        counts = padded.cpu()

    num_labels = getattr(getattr(model, "config", None), "num_labels", None) or counts.numel()
    if counts.numel() > num_labels:
        raise ValueError(
            f"derive_class_weights: train labels reach class {counts.numel() - 1} "
            f"but the model has num_labels={num_labels}"
        )
    present = torch.zeros(num_labels, dtype=torch.bool)
    present[: counts.numel()] = counts > 0
    if not present.any():
        raise ValueError("derive_class_weights: the train dataset 'labels' column is empty")

    full = torch.ones(num_labels, dtype=torch.float32)
    full[present] = (counts.sum() / (present.sum() * counts[counts > 0])).to(torch.float32)
    return full


@dataclass(frozen=True)
class ClassificationLoss:
    """Evaluated in fp32. Never the head's built-in loss, which runs its cross-entropy in bf16 storage, picks single-
    vs multi-label by sniffing the label dtype and scores a single-logit head as regression."""

    multi_label: bool
    # Class weights: CE's ``weight``, or the BCE/focal ``pos_weight`` on a multi-label head.
    weight: torch.Tensor | None = None
    label_smoothing: float = 0.0
    focal_gamma: float | None = None
    focal_alpha: float | None = None

    def to(self, device) -> "ClassificationLoss":
        """Still fp32, since under ``derive_class_weights`` the entries span orders of magnitude."""
        if self.weight is None:
            return self
        weight = self.weight.to(device=device)
        return self if weight is self.weight else replace(self, weight=weight)

    def mean(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self._evaluate(*loss_inputs_fp32(logits, targets), reduction="mean")

    def masked_sum(self, logits: torch.Tensor, targets: torch.Tensor, valid_rows: torch.Tensor) -> torch.Tensor:
        """Summed over ``valid_rows``, additive over pipeline microbatches. Inert rows are zeroed in value space (a
        boolean index syncs the host), logits included, since a saturated logit would make the backward's ``softmax``
        NaN. The extra zeros match summing the kept rows to ~2 fp32 ULP."""
        logits, targets = loss_inputs_fp32(logits, targets)
        row_mask = valid_rows.reshape(-1, *(1,) * (logits.dim() - 1))
        logits = torch.where(row_mask, logits, logits.new_zeros(()))
        per_element = self._evaluate(logits, targets, reduction="none")
        mask = valid_rows.reshape(-1, *(1,) * (per_element.dim() - 1))
        return torch.where(mask, per_element, per_element.new_zeros(())).sum()

    def normalizer(self, targets: torch.Tensor, valid_rows: torch.Tensor) -> torch.Tensor:
        """The denominator of torch's ``reduction="mean"`` over ``valid_rows``: ``B*C`` for BCE and multi-label focal,
        the target classes' weight sum for weighted CE, the row count otherwise (focal means its per-row values even
        when class-weighted)."""
        rows = valid_rows.sum()
        if self.multi_label:
            return (rows * targets.size(-1)).float().clamp(min=1.0)
        if self.focal_gamma is None and self.weight is not None:
            # Inert rows carry LABEL_IGNORE_INDEX; clamp only to keep the gather in range.
            per_row = self.weight[targets.clamp_min(0)]
            weight_sum = torch.where(valid_rows, per_row, per_row.new_zeros(())).sum()
            # A batch of only inert rows would divide zero by zero; a real batch's sum is never replaced.
            return torch.where(weight_sum > 0, weight_sum, weight_sum.new_ones(()))
        return rows.float().clamp(min=1.0)

    def _evaluate(self, logits: torch.Tensor, targets: torch.Tensor, reduction: str) -> torch.Tensor:
        if self.focal_gamma is not None:
            return focal_loss(
                logits,
                targets,
                gamma=self.focal_gamma,
                alpha=self.focal_alpha,
                weight=self.weight,
                reduction=reduction,
            )
        if self.multi_label:
            return F.binary_cross_entropy_with_logits(logits, targets, pos_weight=self.weight, reduction=reduction)
        return F.cross_entropy(
            logits, targets, weight=self.weight, label_smoothing=self.label_smoothing, reduction=reduction
        )


def build_classification_loss(args, model: nn.Module, train_dataset, *, multi_label: bool) -> ClassificationLoss:
    """Refuses the options ``model``'s head cannot honor. COLLECTIVE under ``derive_class_weights``."""
    num_labels = model.config.num_labels
    if not multi_label and num_labels < 2:
        raise ValueError(
            f"Single-label classification needs num_labels >= 2 (got {num_labels}): its objective "
            f"is a softmax cross-entropy over the head's logits. A single-logit head is "
            f"transformers' regression convention (MSE), and regression is not supported; use "
            f"is_multi_label=True for independent sigmoid outputs."
        )
    if multi_label and args.loss_type == "label_smoothing_ce":
        raise ValueError(
            "loss_type='label_smoothing_ce' is single-label only (softmax CE). "
            "Use 'focal' or class_weights (weighted BCE) with multi-label datasets."
        )
    if multi_label and args.derive_class_weights:
        raise ValueError(
            "derive_class_weights is not supported with multi-label datasets — pass class_weights explicitly."
        )

    weight = None
    if args.class_weights is not None:
        weight = torch.tensor(args.class_weights, dtype=torch.float32)
    elif args.derive_class_weights:
        weight = balanced_class_weights(train_dataset, model)

    focal = args.loss_type == "focal"
    if focal and args.focal_alpha is not None and not multi_label:
        raise ValueError(
            "focal_alpha balances positives against negatives (alpha_t = alpha*y + "
            "(1-alpha)*(1-y)), which only exists on a multi-label (sigmoid) head. This is a "
            "single-label softmax head, where every element has exactly one target — a "
            "scalar alpha there is a uniform rescale of the loss (a learning-rate change in "
            "disguise), not balancing. Use class_weights or derive_class_weights for per-class "
            "balancing, and focal_gamma to down-weight easy examples."
        )
    return ClassificationLoss(
        multi_label=multi_label,
        weight=weight,
        label_smoothing=args.label_smoothing if args.loss_type == "label_smoothing_ce" else 0.0,
        focal_gamma=args.focal_gamma if focal else None,
        focal_alpha=args.focal_alpha if focal else None,
    )
