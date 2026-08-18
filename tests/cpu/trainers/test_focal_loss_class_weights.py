#!/usr/bin/env python
"""Focal loss must take its modulating factor from the UNWEIGHTED cross-entropy.

``F.cross_entropy(weight=w, reduction="none")`` returns ``w_y * -log p_y``, so ``exp(-ce)`` on the
weighted value is ``p_y ** w_y``, not ``p_y``. Deriving ``p_t`` from it makes a class weight distort
the ``(1 - p_t) ** gamma`` modulator instead of scaling the loss — the two levers then fight each
other, over-focusing down-weighted classes and under-focusing up-weighted ones.
"""

import pytest
import torch
import torch.nn.functional as F

from src.trainers.reward.classification import ClassificationTrainer

GAMMA = 2.0


def _canonical_focal(logits, labels, weight, gamma=GAMMA):
    """Textbook weighted focal: modulator from the true probability, weight scales the magnitude."""
    log_probs = F.log_softmax(logits.float(), dim=-1)
    p_true = log_probs.gather(-1, labels.unsqueeze(-1)).squeeze(-1).exp()
    per_example = ((1 - p_true) ** gamma) * (-p_true.log())
    if weight is not None:
        per_example = per_example * weight[labels]
    return per_example.mean()


def _logits_and_labels():
    """Six examples spanning confident to badly-wrong, so p_true spreads over ~0.005 to ~0.9.

    The defect's size scales with how far p_true sits from the p_true**w it is confused with, so a
    fixture clustered near one confidence level would not separate the two forms.
    """
    logits = torch.tensor(
        [
            [4.0, 1.0, 0.0],  # confident, correct
            [0.0, 3.0, 0.5],  # confident, correct
            [2.0, 2.5, -3.0],  # badly wrong
            [5.0, 0.0, 0.5],  # very confident, correct
            [1.5, 1.0, 0.5],  # unsure, correct
            [1.0, 0.5, 0.2],  # unsure, wrong
        ]
    )
    return logits, torch.tensor([0, 1, 2, 0, 1, 2])


def test_unweighted_focal_matches_canonical():
    """The unweighted case was already correct and must stay so."""
    logits, labels = _logits_and_labels()
    got = ClassificationTrainer._focal_loss(logits, labels, gamma=GAMMA)
    assert got.item() == pytest.approx(_canonical_focal(logits, labels, None).item(), rel=1e-5)


def test_weighted_focal_matches_canonical():
    """With class weights the loss must equal w_y*(1-p_y)^gamma*(-log p_y), not (1-p^w)^gamma*w*(-log p)."""
    logits, labels = _logits_and_labels()
    weight = torch.tensor([1.0, 5.0, 0.2])

    got = ClassificationTrainer._focal_loss(logits, labels, gamma=GAMMA, weight=weight)
    expected = _canonical_focal(logits, labels, weight)
    assert got.item() == pytest.approx(expected.item(), rel=1e-5)

    # And it must NOT equal the p^w form — otherwise the test would pass on the broken code too.
    weighted_ce = F.cross_entropy(logits, labels, weight=weight, reduction="none")
    broken = (((1 - torch.exp(-weighted_ce)) ** GAMMA) * weighted_ce).mean()
    assert abs(broken.item() - expected.item()) / expected.item() > 0.1, (
        "the broken and correct forms are too close on this fixture to prove anything"
    )


def test_weighted_focal_modulator_is_weight_independent():
    """The per-example modulator must depend only on p_t, so scaling a class weight scales its loss linearly.

    This is the property the defect broke: under ``p_t = exp(-weighted_ce)`` the modulator itself
    moves with the weight, so doubling a class's weight does NOT double its contribution.
    """
    logits, labels = _logits_and_labels()
    single = torch.tensor([1.0, 1.0, 1.0])
    doubled = torch.tensor([1.0, 2.0, 1.0])

    per_example_1 = ClassificationTrainer._focal_loss(
        logits, labels, gamma=GAMMA, weight=single, reduction="none"
    ).sum()
    per_example_2 = ClassificationTrainer._focal_loss(
        logits, labels, gamma=GAMMA, weight=doubled, reduction="none"
    ).sum()

    class_1_mask = labels == 1
    unweighted_terms = _canonical_focal(logits[class_1_mask], labels[class_1_mask], None) * class_1_mask.sum()
    # Doubling class 1's weight must add exactly one more copy of class 1's unweighted focal terms.
    assert (per_example_2 - per_example_1).item() == pytest.approx(unweighted_terms.item(), rel=1e-5)


def test_multilabel_focal_pos_weight_matches_canonical():
    """Multi-label: pos_weight scales the positive term only, and never the modulator."""
    generator = torch.Generator().manual_seed(11)
    logits = torch.randn(4, 3, generator=generator) * 2.0
    labels = torch.tensor([[1.0, 0.0, 1.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    pos_weight = torch.tensor([1.0, 4.0, 0.5])

    raw = F.binary_cross_entropy_with_logits(logits.float(), labels, reduction="none")
    weighted = F.binary_cross_entropy_with_logits(logits.float(), labels, pos_weight=pos_weight, reduction="none")
    expected = (((1 - torch.exp(-raw)) ** GAMMA) * weighted).mean()

    got = ClassificationTrainer._focal_loss(logits, labels, gamma=GAMMA, weight=pos_weight)
    assert got.item() == pytest.approx(expected.item(), rel=1e-5)


def test_per_element_form_recovers_the_mean_for_the_pipeline_normalizer():
    """The PP seam masks and sums the per-element form itself, then divides by the element count —
    that identity must hold when weighted, and it is the only non-mean reduction the loss offers."""
    logits, labels = _logits_and_labels()
    weight = torch.tensor([1.0, 5.0, 0.2])
    summed = ClassificationTrainer._focal_loss(logits, labels, gamma=GAMMA, weight=weight, reduction="none").sum()
    meaned = ClassificationTrainer._focal_loss(logits, labels, gamma=GAMMA, weight=weight, reduction="mean")
    assert (summed / labels.size(0)).item() == pytest.approx(meaned.item(), rel=1e-6)


def test_an_unsupported_reduction_raises_instead_of_silently_meaning():
    """Only the two reductions the trainer asks for exist; anything else was a silent mean."""
    logits, labels = _logits_and_labels()
    with pytest.raises(ValueError, match="reduction"):
        ClassificationTrainer._focal_loss(logits, labels, gamma=GAMMA, reduction="sum")


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
