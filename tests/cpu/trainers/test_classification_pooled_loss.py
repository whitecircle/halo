#!/usr/bin/env python
"""``ClassificationTrainer``'s pooled loss: fp32 evaluation, and a micro-batch path free of host syncs.

Two properties are pinned here, each silent when it breaks:

* **fp32.** The head's logits are the toolkit's bf16 storage dtype, and a class-weighted CE is
  ``w_y * (logit_y - logsumexp(logits))`` — a cancelling difference. Evaluated in bf16, with the
  class weights rounded to bf16 to satisfy torch's dtype rule, the loss lands ~0.5% off its exact
  value while the unweighted path (the model's own head) stays fp32-accurate; the ``pre_fix``
  comparisons below measure that gap, so the assertions cannot pass on a bf16 evaluation.
* **No device→host sync per micro-batch.** The pipeline-parallel token loss runs on every
  micro-batch of every step, so it masks in value space rather than gathering the surviving rows:
  ``bool(valid.any())`` plus two boolean-mask gathers would stall the pipeline three times there.
  The masked form must stay numerically identical to indexing those rows out.

Run: python tests/cpu/trainers/test_classification_pooled_loss.py
"""

import types
from functools import partial

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.data.spans import LABEL_IGNORE_INDEX
from src.trainers.reward.classification import ClassificationTrainer
from src.trainers.reward.pooling import decode_pooling_plane, pooled_outputs

NUM_CLASSES = 6
# Not bf16-representable, and spanning two orders of magnitude — the shape derive_class_weights
# produces on an imbalanced corpus, where rounding the vector is a visible reweighting.
CLASS_WEIGHTS = torch.tensor([0.2571, 17.3, 1.379, 0.6153, 4.77, 0.9091])


def _trainer(loss_fn, *, is_multi_label=False):
    """A trainer stub carrying only what the loss seams read (no model, no accelerator)."""
    trainer = ClassificationTrainer.__new__(ClassificationTrainer)
    trainer._loss_fn = loss_fn
    trainer.is_multi_label = is_multi_label
    return trainer


def _bf16_batch(rows=16, seed=3):
    """(bf16 logits, class targets). The logits are EXACT in bf16, so any difference between the
    paths is the arithmetic's dtype, never the input's."""
    generator = torch.Generator().manual_seed(seed)
    logits = (torch.randn(rows, NUM_CLASSES, generator=generator) * 6.0).bfloat16()
    return logits, torch.randint(0, NUM_CLASSES, (rows,), generator=generator)


def _plane(rows, valid_rows, values, seq=5):
    """A pooling plane: ``LABEL_IGNORE_INDEX`` everywhere except one marker per VALID row."""
    plane = torch.full((rows, seq), LABEL_IGNORE_INDEX, dtype=torch.long)
    for row in range(rows):
        if valid_rows[row]:
            plane[row, row % seq] = int(values[row])
    return plane


# --------------------------------------------------------------------------------------------
# fp32 evaluation
# --------------------------------------------------------------------------------------------


def test_weighted_ce_matches_an_fp64_reference_on_bf16_logits():
    """The weighted path must be as accurate as the unweighted one, not ~0.5% off it.

    Compared against fp64, and against the pre-fix formulation (bare ``F.cross_entropy`` on bf16
    logits with the weights downcast to match), which is what the bound has to separate — a plain
    "close to fp64" tolerance loose enough for bf16 would pass either way.
    """
    logits, targets = _bf16_batch()
    trainer = _trainer(nn.CrossEntropyLoss(weight=CLASS_WEIGHTS.clone()))
    valid = torch.ones(logits.size(0), dtype=torch.bool)

    reference = F.cross_entropy(logits.double(), targets, weight=CLASS_WEIGHTS.double(), reduction="sum")
    got = trainer._pooled_loss(logits, targets, valid)
    pre_fix = F.cross_entropy(logits, targets, weight=CLASS_WEIGHTS.bfloat16(), reduction="sum")

    def relative(value):
        return abs(value.double().item() - reference.item()) / abs(reference.item())

    assert relative(got) < 1e-5, f"weighted CE is {relative(got):.2e} off fp64 — it is not evaluating in fp32"
    assert relative(pre_fix) > 1e-3, (
        f"the pre-fix bf16 path is only {relative(pre_fix):.2e} off fp64, so this batch does not "
        f"separate the two evaluations and the test proves nothing"
    )


def test_weighted_ce_normalizer_matches_an_fp64_reference():
    """The denominator is a sum of the same class weights, so rounding them biases it too — and it
    divides the loss, so the error does not cancel against the numerator's."""
    _, targets = _bf16_batch()
    trainer = _trainer(nn.CrossEntropyLoss(weight=CLASS_WEIGHTS.clone()))
    valid = torch.ones(targets.size(0), dtype=torch.bool)

    reference = CLASS_WEIGHTS.double()[targets].sum()
    got = trainer._pooled_loss_normalizer(targets, valid)
    pre_fix = CLASS_WEIGHTS.bfloat16()[targets].sum()

    assert abs(got.double().item() - reference.item()) / reference.item() < 1e-5
    assert abs(pre_fix.double().item() - reference.item()) / reference.item() > 1e-3


def test_the_weighted_normalizer_never_divides_an_all_inert_batch_by_zero():
    """An all-inert micro-batch sums to zero on both sides, and the runtime's ``sum / normalizer`` is
    then NaN — carried into every stage's reported loss and into the nan/inf filter. The row-count
    branches floor at 1.0; the weight-sum branch owes the same floor, and only at zero, so a real
    batch's denominator (the sum of its rows' class weights) is untouched."""
    trainer = _trainer(nn.CrossEntropyLoss(weight=CLASS_WEIGHTS.clone()))
    targets = torch.full((4,), LABEL_IGNORE_INDEX, dtype=torch.long)

    normalizer = trainer._pooled_loss_normalizer(targets, torch.zeros(4, dtype=torch.bool))

    assert normalizer.item() == 1.0
    assert torch.isfinite(torch.zeros(()) / normalizer).item()


class _StubHead:
    """A sequence-classification model returning fixed bf16 logits — the toolkit's storage dtype."""

    def __init__(self, logits):
        self._logits = logits

    def __call__(self, **_kwargs):
        return types.SimpleNamespace(logits=self._logits)


def _non_pp_loss(trainer, logits, labels):
    """Drive the NON-pipeline seam (``_compute_loss_inner``), where the loss met bf16 logits."""
    inputs = {
        "input_ids": torch.zeros(logits.size(0), 4, dtype=torch.long),
        "attention_mask": torch.ones(logits.size(0), 4, dtype=torch.long),
        "labels": labels,
    }
    return trainer._compute_loss_inner(_StubHead(logits), inputs, return_outputs=False)


def test_non_pp_weighted_loss_evaluates_in_fp32():
    """The seam the bf16 evaluation actually lived on.

    The PP adapter always handed ``_pooled_loss`` fp32 (it floats the pooled logits itself), so only
    this path ever ran a class-weighted CE in bf16 with the weights rounded to match — ~1% off the
    exact value, every step.
    """
    logits, targets = _bf16_batch()
    loss_fn = nn.CrossEntropyLoss(weight=CLASS_WEIGHTS.clone())
    trainer = _trainer(loss_fn)

    got = _non_pp_loss(trainer, logits, targets)
    reference = F.cross_entropy(logits.double(), targets, weight=CLASS_WEIGHTS.double())
    pre_fix = F.cross_entropy(logits, targets, weight=CLASS_WEIGHTS.bfloat16())

    assert got.dtype is torch.float32, f"the non-PP loss reduced in {got.dtype}"
    assert abs(got.double().item() - reference.item()) / reference.item() < 1e-5
    assert abs(pre_fix.double().item() - reference.item()) / reference.item() > 1e-3


def test_non_pp_multi_label_loss_evaluates_in_fp32():
    """Same seam, sigmoid head: BCE takes its output dtype from the target, so a bf16 multi-hot
    label would hold the loss in bf16 even with the logits upcast."""
    generator = torch.Generator().manual_seed(21)
    logits = (torch.randn(8, NUM_CLASSES, generator=generator) * 6.0).bfloat16()
    targets = (torch.rand(8, NUM_CLASSES, generator=generator) > 0.5).bfloat16()
    trainer = _trainer(nn.BCEWithLogitsLoss(pos_weight=CLASS_WEIGHTS.clone()), is_multi_label=True)

    got = _non_pp_loss(trainer, logits, targets)
    reference = F.binary_cross_entropy_with_logits(
        logits.double(), targets.double(), pos_weight=CLASS_WEIGHTS.double()
    )

    assert got.dtype is torch.float32
    assert abs(got.double().item() - reference.item()) / reference.item() < 1e-5


def test_class_weight_vector_is_never_downcast():
    """The weights are moved, never rounded: the loss upcasts the LOGITS to meet them instead.

    Downcasting was permanent — ``_move_loss_weights_to`` writes back onto the loss object, so the
    first bf16 step rounded the vector for the rest of the run.
    """
    logits, targets = _bf16_batch()
    loss_fn = nn.CrossEntropyLoss(weight=CLASS_WEIGHTS.clone())
    trainer = _trainer(loss_fn)

    _non_pp_loss(trainer, logits, targets)

    assert loss_fn.weight.dtype is torch.float32
    assert torch.equal(loss_fn.weight, CLASS_WEIGHTS)


def test_multi_label_bce_stays_fp32_against_a_bf16_target():
    """``binary_cross_entropy_with_logits`` takes its OUTPUT dtype from the TARGET, so upcasting the
    logits alone would leave a bf16 multi-hot target driving the whole loss back into bf16."""
    generator = torch.Generator().manual_seed(11)
    logits = (torch.randn(8, NUM_CLASSES, generator=generator) * 6.0).bfloat16()
    targets = (torch.rand(8, NUM_CLASSES, generator=generator) > 0.5).bfloat16()
    trainer = _trainer(None, is_multi_label=True)

    got = trainer._pooled_loss(logits, targets, torch.ones(8, dtype=torch.bool))
    reference = F.binary_cross_entropy_with_logits(logits.double(), targets.double(), reduction="sum")

    assert got.dtype is torch.float32, f"multi-label BCE reduced in {got.dtype}"
    assert abs(got.double().item() - reference.item()) / reference.item() < 1e-5


@pytest.mark.parametrize(
    "loss_fn, multi_label",
    [
        (None, False),
        (None, True),
        (nn.CrossEntropyLoss(weight=CLASS_WEIGHTS.clone()), False),
        (nn.CrossEntropyLoss(label_smoothing=0.1), False),
        (nn.BCEWithLogitsLoss(pos_weight=CLASS_WEIGHTS.clone()), True),
    ],
)
def test_every_loss_variant_returns_fp32(loss_fn, multi_label):
    """One dtype for every ``_build_loss_fn`` outcome — a variant reducing in bf16 would feed the PP
    runtime a summand it then divides by an fp32 normalizer."""
    generator = torch.Generator().manual_seed(5)
    logits = (torch.randn(8, NUM_CLASSES, generator=generator) * 4.0).bfloat16()
    targets = (
        (torch.rand(8, NUM_CLASSES, generator=generator) > 0.5).bfloat16()
        if multi_label
        else torch.randint(0, NUM_CLASSES, (8,), generator=generator)
    )
    trainer = _trainer(loss_fn, is_multi_label=multi_label)
    assert trainer._pooled_loss(logits, targets, torch.ones(8, dtype=torch.bool)).dtype is torch.float32


def test_focal_loss_variant_returns_fp32():
    """The focal partial reduces itself, so it needs the same upcast at the seam."""
    generator = torch.Generator().manual_seed(6)
    logits = (torch.randn(8, NUM_CLASSES, generator=generator) * 4.0).bfloat16()
    targets = torch.randint(0, NUM_CLASSES, (8,), generator=generator)
    trainer = _trainer(partial(ClassificationTrainer._focal_loss, gamma=2.0, alpha=None, weight=None))
    assert trainer._pooled_loss(logits, targets, torch.ones(8, dtype=torch.bool)).dtype is torch.float32


# --------------------------------------------------------------------------------------------
# The pipeline micro-batch path: mask, never index
# --------------------------------------------------------------------------------------------


class _NoHostSyncTensor(torch.Tensor):
    """A tensor that raises on any op reading a value back to the host.

    Each of these is a stream synchronization on CUDA — ``torch.cuda.set_sync_debug_mode("error")``
    catches them on a GPU, and this is that check's CPU-runnable equivalent. Boolean-mask indexing
    counts: it lowers to ``nonzero``, whose output shape only the device knows.
    """

    _SYNCING = (
        torch.Tensor.item,
        torch.Tensor.tolist,
        torch.Tensor.nonzero,
        torch.Tensor.__bool__,
        torch.Tensor.__int__,
        torch.Tensor.__float__,
        torch.Tensor.__index__,
        torch.masked_select,
    )

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        if func in cls._SYNCING:
            raise AssertionError(f"{getattr(func, '__name__', func)} forces a device→host sync")
        if func is torch.Tensor.__getitem__:
            index = args[1] if isinstance(args[1], tuple) else (args[1],)
            if any(isinstance(part, torch.Tensor) and part.dtype is torch.bool for part in index):
                raise AssertionError("boolean-mask indexing forces a device→host sync")
        return super().__torch_function__(func, types, args, kwargs or {})


def _pp_batch(seed=17, rows=6, seq=5, multi_label=False):
    """(per-token logits, plane, class_targets, valid) with the PP eval path's inert rows present."""
    generator = torch.Generator().manual_seed(seed)
    valid = torch.tensor([True, False, True, True, False, True][:rows])
    values = torch.randint(0, NUM_CLASSES, (rows,), generator=generator)
    plane = _plane(rows, valid, values, seq=seq)
    logits = (torch.randn(rows, seq, NUM_CLASSES, generator=generator) * 5.0).bfloat16()
    class_targets = (torch.rand(rows, NUM_CLASSES, generator=generator) > 0.5).float() if multi_label else None
    return logits, plane, class_targets, valid


# Masking leaves the inert rows in the reduction as exact zeros. They change no individual addition,
# but they lengthen the vector, and torch's pairwise sum blocks on length — so the non-zero terms
# associate differently and the result lands within a couple of fp32 ULP, not bit-exactly. Measured
# worst case over 60 random batches x every loss variant: 2.1e-7.
_REDUCTION_ORDER_TOLERANCE = 1e-6


@pytest.mark.parametrize("multi_label", [False, True])
def test_masked_sum_equals_indexing_the_surviving_rows_out(multi_label):
    """Masking in value space must equal indexing the surviving rows out.

    Bounded against BOTH sides: within reduction-order noise of the indexed answer, and far from the
    unmasked sum — otherwise a tolerance this loose could hide a mask that never applied.
    """
    logits, plane, class_targets, valid = _pp_batch(multi_label=multi_label)
    loss_fn = None if multi_label else nn.CrossEntropyLoss(weight=CLASS_WEIGHTS.clone())
    trainer = _trainer(loss_fn, is_multi_label=multi_label)

    _, positions, plane_values = decode_pooling_plane(plane)
    pooled = pooled_outputs(logits, positions)
    targets = class_targets if multi_label else plane_values.clamp_min(0)
    all_rows = torch.ones(pooled.size(0), dtype=torch.bool)

    masked = trainer._pooled_loss(pooled, targets, valid)
    indexed = trainer._pooled_loss(pooled[valid], targets[valid], all_rows[valid])
    unmasked = trainer._pooled_loss(pooled, targets, all_rows)

    assert masked.item() == pytest.approx(indexed.item(), rel=_REDUCTION_ORDER_TOLERANCE)
    assert abs(unmasked.item() - indexed.item()) / indexed.item() > 0.01, (
        "the inert rows contribute almost nothing to this batch, so the assertion above would pass "
        "with the mask removed"
    )


@pytest.mark.parametrize("multi_label", [False, True])
def test_masked_normalizer_equals_indexing_the_surviving_rows_out(multi_label):
    """Numerator and denominator must agree on which rows exist, so the normalizer gets the same
    treatment — a mismatch would rescale the loss by the inert-row fraction."""
    _, plane, class_targets, valid = _pp_batch(multi_label=multi_label)
    loss_fn = None if multi_label else nn.CrossEntropyLoss(weight=CLASS_WEIGHTS.clone())
    trainer = _trainer(loss_fn, is_multi_label=multi_label)

    _, _, plane_values = decode_pooling_plane(plane)
    targets = class_targets if multi_label else plane_values
    ones = torch.ones(int(valid.sum()), dtype=torch.bool)

    masked = trainer._pooled_loss_normalizer(targets, valid)
    indexed = trainer._pooled_loss_normalizer(targets[valid], ones)
    unmasked = trainer._pooled_loss_normalizer(targets, torch.ones_like(valid))

    assert masked.item() == pytest.approx(indexed.item(), rel=_REDUCTION_ORDER_TOLERANCE)
    assert unmasked.item() != pytest.approx(indexed.item(), rel=_REDUCTION_ORDER_TOLERANCE)


@pytest.mark.parametrize("multi_label", [False, True])
def test_token_loss_runs_without_a_single_host_sync(multi_label):
    """The whole micro-batch loss, under a tensor that raises on any read-back to the host.

    ``bool(valid.any())`` and the two boolean-mask gathers this replaced were three stalls per
    micro-batch — at 512 GPUs, three per micro-batch per rank on the pipeline's critical path.
    """
    logits, plane, class_targets, valid = _pp_batch(multi_label=multi_label)
    trainer = _trainer(
        None if multi_label else nn.CrossEntropyLoss(weight=CLASS_WEIGHTS.clone()), is_multi_label=multi_label
    )

    target = {"labels": plane.as_subclass(_NoHostSyncTensor)}
    if multi_label:
        target["class_targets"] = class_targets.as_subclass(_NoHostSyncTensor)

    loss = trainer._pp_classification_token_loss(logits.as_subclass(_NoHostSyncTensor), target)
    assert torch.isfinite(loss.as_subclass(torch.Tensor)).all()
    _ = valid  # the mask is derived inside the loss; the fixture's copy only documents the batch


def test_normalizer_runs_without_a_host_sync():
    """The step normalizer is off the micro-batch path but shared the same boolean indexing."""
    _, plane, _, _ = _pp_batch()
    trainer = _trainer(nn.CrossEntropyLoss(weight=CLASS_WEIGHTS.clone()))
    normalizer = trainer._pp_classification_normalizer({"labels": plane.as_subclass(_NoHostSyncTensor)})
    assert normalizer.as_subclass(torch.Tensor).item() > 0


def test_all_inert_microbatch_is_zero_and_still_wired_to_the_graph():
    """An all-inert micro-batch must yield a real zero that BACKWARDS.

    The masked sum runs unconditionally — a ``bool(valid.any())`` early-out would host-sync every
    micro-batch — so this is the case that has to survive it: a loss detached from the stage's
    activations leaves the pipeline schedule waiting on a gradient that never arrives.
    """
    rows, seq = 4, 5
    plane = torch.full((rows, seq), LABEL_IGNORE_INDEX, dtype=torch.long)
    logits = torch.randn(rows, seq, NUM_CLASSES, requires_grad=True)
    trainer = _trainer(nn.CrossEntropyLoss(weight=CLASS_WEIGHTS.clone()))

    loss = trainer._pp_classification_token_loss(logits, plane)

    assert loss.item() == 0.0
    assert loss.requires_grad, "an all-inert micro-batch returned a loss detached from the stage output"
    loss.backward()
    assert logits.grad is not None and torch.count_nonzero(logits.grad) == 0


def test_inert_rows_cannot_leak_a_nan_from_a_saturated_logit():
    """An inert row's logits are whatever the padder left, and a saturated one must not poison the
    micro-batch — in the BACKWARD as much as the forward.

    Masking only the loss is not enough: the objective's backward still runs over the row, where
    ``softmax(inf)`` is NaN and the chain rule carries it through the masked zero. The row's logits
    are therefore neutralized before the objective sees them.
    """
    rows, seq = 3, 4
    valid = torch.tensor([True, False, True])
    plane = _plane(rows, valid, torch.tensor([1, 0, 2]), seq=seq)
    logits = torch.randn(rows, seq, NUM_CLASSES)
    logits[1] = float("inf")
    logits.requires_grad_(True)
    trainer = _trainer(nn.CrossEntropyLoss(weight=CLASS_WEIGHTS.clone()))

    loss = trainer._pp_classification_token_loss(logits, plane)
    assert torch.isfinite(loss).all()

    loss.backward()
    assert torch.isfinite(logits.grad).all(), "the saturated inert row leaked a NaN into the gradient"
    assert torch.count_nonzero(logits.grad[1]) == 0


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
