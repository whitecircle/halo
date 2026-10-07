#!/usr/bin/env python
"""``ClassificationTrainer``'s pooled loss: fp32 evaluation, a micro-batch path free of host syncs,
and no token-label check on class ids.

Three properties are pinned here, each silent when it breaks:

* **fp32.** The head's logits are the toolkit's bf16 storage dtype, and a (class-weighted) CE is
  ``w_y * (logit_y - logsumexp(logits))`` — a cancelling difference. Evaluated in bf16 — the class
  weights rounded to bf16 to satisfy torch's dtype rule, or transformers' own head loss, which
  never upcasts — the loss lands ~0.3–1% off its exact value; the ``bf16_path`` / head-loss
  comparisons below measure that gap, so the assertions cannot pass on a bf16 evaluation.
* **No device→host sync per micro-batch.** The pipeline-parallel token loss runs on every
  micro-batch of every step, so it masks in value space rather than gathering the surviving rows:
  ``bool(valid.any())`` plus two boolean-mask gathers would stall the pipeline three times there.
  The masked form must stay numerically identical to indexing those rows out.
* **Class ids are not token ids.** The mixin's empty-label check compares ``labels`` against the
  tokenizer's pad/eos ids, which a class id can equal on every row.

Run: python tests/cpu/trainers/test_classification_pooled_loss.py
"""

import types
from functools import partial

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Qwen3Config, Qwen3ForSequenceClassification

import src.trainers.mixins.base as mixin_base
from src.configs.classification_config import ClassificationConfig
from src.data.spans import LABEL_IGNORE_INDEX
from src.trainers.reward.classification import ClassificationTrainer
from src.trainers.reward.pooling import decode_pooling_plane, pooled_outputs

NUM_CLASSES = 6
# Not bf16-representable, and spanning two orders of magnitude — the shape derive_class_weights
# produces on an imbalanced corpus, where rounding the vector is a visible reweighting.
CLASS_WEIGHTS = torch.tensor([0.2571, 17.3, 1.379, 0.6153, 4.77, 0.9091])
_PAD_ID = 0


def _trainer(loss_fn, *, is_multi_label=False):
    """A trainer stub carrying only what the loss seams read (no model, no accelerator)."""
    trainer = ClassificationTrainer.__new__(ClassificationTrainer)
    trainer._loss_fn = loss_fn
    trainer.is_multi_label = is_multi_label
    trainer.model = types.SimpleNamespace(training=True)  # what eval_split_rows reads: a train step
    return trainer


def _head(dtype, *, num_labels=NUM_CLASSES, seed=0):
    """A tiny randomly-initialised sequence-classification model in ``dtype``.

    The score head is scaled up so the logits span a trained head's range: at the 0.02 init they
    sit near zero, where bf16 and fp32 cross-entropy barely differ.
    """
    torch.manual_seed(seed)
    config = Qwen3Config(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        pad_token_id=_PAD_ID,
        num_labels=num_labels,
    )
    model = Qwen3ForSequenceClassification(config)
    with torch.no_grad():
        model.score.weight.mul_(40.0)
    return model.to(dtype).eval()


def _head_batch(*, num_labels=NUM_CLASSES, multi_label=False, rows=16, seq=12, seed=1):
    """Right-padded rows of random lengths plus their class targets (multi-hot floats when multi-label)."""
    generator = torch.Generator().manual_seed(seed)
    input_ids = torch.randint(1, 64, (rows, seq), generator=generator)
    lengths = torch.randint(3, seq + 1, (rows,), generator=generator)
    attention_mask = (torch.arange(seq)[None] < lengths[:, None]).long()
    if multi_label:
        labels = (torch.rand(rows, num_labels, generator=generator) > 0.5).float()
    else:
        labels = torch.randint(0, num_labels, (rows,), generator=generator)
    return {
        "input_ids": input_ids.masked_fill(attention_mask == 0, _PAD_ID),
        "attention_mask": attention_mask,
        "labels": labels,
    }


def _built_trainer(model, *, is_multi_label=False):
    """A trainer stub whose loss is what ``_build_loss_fn`` returns for the shipped config defaults."""
    trainer = _trainer(None, is_multi_label=is_multi_label)
    trainer._loss_fn = trainer._build_loss_fn(ClassificationConfig(output_dir="unused"), None, model)
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

    Compared against fp64, and against the bf16 formulation (bare ``F.cross_entropy`` on bf16
    logits with the weights downcast to match), which is what the bound has to separate — a plain
    "close to fp64" tolerance loose enough for bf16 would pass either way.
    """
    logits, targets = _bf16_batch()
    trainer = _trainer(nn.CrossEntropyLoss(weight=CLASS_WEIGHTS.clone()))
    valid = torch.ones(logits.size(0), dtype=torch.bool)

    reference = F.cross_entropy(logits.double(), targets, weight=CLASS_WEIGHTS.double(), reduction="sum")
    got = trainer._pooled_loss(logits, targets, valid)
    bf16_path = F.cross_entropy(logits, targets, weight=CLASS_WEIGHTS.bfloat16(), reduction="sum")

    def relative(value):
        return abs(value.double().item() - reference.item()) / abs(reference.item())

    assert relative(got) < 1e-5, f"weighted CE is {relative(got):.2e} off fp64 — it is not evaluating in fp32"
    assert relative(bf16_path) > 1e-3, (
        f"the bf16 path is only {relative(bf16_path):.2e} off fp64, so this batch does not "
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
    bf16_path = CLASS_WEIGHTS.bfloat16()[targets].sum()

    assert abs(got.double().item() - reference.item()) / reference.item() < 1e-5
    assert abs(bf16_path.double().item() - reference.item()) / reference.item() > 1e-3


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
    """The seam a bf16 evaluation would run on.

    The PP adapter hands ``_pooled_loss`` fp32 (it floats the pooled logits itself), so only this
    path could run a class-weighted CE in bf16 with the weights rounded to match — ~1% off the
    exact value, every step.
    """
    logits, targets = _bf16_batch()
    loss_fn = nn.CrossEntropyLoss(weight=CLASS_WEIGHTS.clone())
    trainer = _trainer(loss_fn)

    got = _non_pp_loss(trainer, logits, targets)
    reference = F.cross_entropy(logits.double(), targets, weight=CLASS_WEIGHTS.double())
    bf16_path = F.cross_entropy(logits, targets, weight=CLASS_WEIGHTS.bfloat16())

    assert got.dtype is torch.float32, f"the non-PP loss reduced in {got.dtype}"
    assert abs(got.double().item() - reference.item()) / reference.item() < 1e-5
    assert abs(bf16_path.double().item() - reference.item()) / reference.item() > 1e-3


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

    Downcasting would be permanent — ``_move_loss_weights_to`` writes back onto the loss object, so
    the first bf16 step would round the vector for the rest of the run.
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
    trainer = _trainer(nn.BCEWithLogitsLoss(), is_multi_label=True)

    got = trainer._pooled_loss(logits, targets, torch.ones(8, dtype=torch.bool))
    reference = F.binary_cross_entropy_with_logits(logits.double(), targets.double(), reduction="sum")

    assert got.dtype is torch.float32, f"multi-label BCE reduced in {got.dtype}"
    assert abs(got.double().item() - reference.item()) / reference.item() < 1e-5


@pytest.mark.parametrize(
    "loss_fn, multi_label",
    [
        (nn.CrossEntropyLoss(), False),
        (nn.BCEWithLogitsLoss(), True),
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


@pytest.mark.parametrize("multi_label", [False, True])
def test_the_default_config_builds_plain_ce_or_bce(multi_label):
    """``loss_type: cross_entropy`` with no weights is the trainer's own unweighted objective — the
    shape the seams above are exercised with — never a hand-off to the head's loss."""
    loss_fn = _built_trainer(_head(torch.float32), is_multi_label=multi_label)._loss_fn
    if multi_label:
        assert type(loss_fn) is nn.BCEWithLogitsLoss and loss_fn.pos_weight is None
    else:
        assert type(loss_fn) is nn.CrossEntropyLoss
        assert loss_fn.weight is None and loss_fn.label_smoothing == 0.0


def test_the_default_loss_is_fp32_on_a_bf16_head():
    """The default single-label CE on a real bf16 head: fp32, where the head's own loss is bf16.

    Transformers' ``ForSequenceClassificationLoss`` runs ``cross_entropy`` on the bf16 pooled logits
    and returns bf16, so the default path must never route the labels through the head.
    """
    model = _head(torch.bfloat16)
    inputs = _head_batch()
    trainer = _built_trainer(model)

    got = trainer._compute_loss_inner(model, inputs, return_outputs=False)
    with torch.no_grad():
        head = model(**inputs)
    reference = F.cross_entropy(head.logits.double(), inputs["labels"])

    assert got.dtype is torch.float32, f"the default loss reduced in {got.dtype}"
    assert abs(got.double().item() - reference.item()) / reference.item() < 1e-6
    assert head.loss.dtype is torch.bfloat16
    assert abs(head.loss.double().item() - reference.item()) / reference.item() > 1e-3, (
        "the head's bf16 loss is within 1e-3 of fp64 on this batch, so it does not separate the two "
        "evaluations and the test proves nothing"
    )


@pytest.mark.parametrize("multi_label", [False, True])
def test_the_default_loss_value_is_the_heads_own_on_an_fp32_head(multi_label):
    """Behavior preservation: at fp32 the trainer's default objective IS transformers' head loss — its
    single-label CE on class ids, its BCE on multi-hot targets — bit for bit."""
    model = _head(torch.float32)
    inputs = _head_batch(multi_label=multi_label)
    trainer = _built_trainer(model, is_multi_label=multi_label)

    with torch.no_grad():
        got = trainer._compute_loss_inner(model, inputs, return_outputs=False)
        head_loss = model(**inputs).loss

    assert torch.equal(got, head_loss), f"{got.item()} != head loss {head_loss.item()}"


def test_prediction_step_returns_the_head_logits_and_the_labels():
    """Evaluation hands ``compute_metrics`` the head's pooled logits untouched (storage dtype kept)
    and the batch's labels; only the loss is the trainer's fp32 one."""
    model = _head(torch.bfloat16)
    inputs = _head_batch()
    trainer = _built_trainer(model)
    trainer._pp_runtime = None
    trainer._prepare_inputs = lambda batch: batch
    trainer.accelerator = types.SimpleNamespace(device=torch.device("cpu"))

    loss, logits, labels = trainer.prediction_step(model, inputs, prediction_loss_only=False)
    with torch.no_grad():
        expected_logits = model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"]).logits

    assert loss.dtype is torch.float32 and not loss.requires_grad
    assert torch.equal(logits, expected_logits)
    assert torch.equal(labels, inputs["labels"])


def test_soft_label_targets_are_refused_on_both_paths():
    """A single-label run handed ``[B, C]`` float targets would train soft-label cross-entropy, since
    ``F.cross_entropy`` takes class probabilities there; the PP and non-PP seams refuse it alike."""
    model = _head(torch.float32)
    trainer = _built_trainer(model)
    trainer._pp_pool_pad_id = _PAD_ID
    batch = _head_batch()
    batch["labels"] = F.one_hot(batch["labels"], NUM_CLASSES).float()

    with pytest.raises(ValueError, match="integer class ids"):
        trainer._compute_loss_inner(model, batch, return_outputs=False)
    with pytest.raises(ValueError, match="integer class ids"):
        trainer._pp_classification_batch_transform(batch)


def test_a_single_logit_single_label_head_is_refused():
    """``num_labels == 1`` is transformers' regression convention; softmax CE over one logit is
    identically zero, so a single-label run would train on nothing. Multi-label keeps its one sigmoid."""
    with pytest.raises(ValueError, match="num_labels >= 2"):
        _built_trainer(_head(torch.float32, num_labels=1))
    assert (
        type(_built_trainer(_head(torch.float32, num_labels=1), is_multi_label=True)._loss_fn) is nn.BCEWithLogitsLoss
    )


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
    loss_fn = nn.BCEWithLogitsLoss() if multi_label else nn.CrossEntropyLoss(weight=CLASS_WEIGHTS.clone())
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
    loss_fn = nn.BCEWithLogitsLoss() if multi_label else nn.CrossEntropyLoss(weight=CLASS_WEIGHTS.clone())
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

    ``bool(valid.any())`` and two boolean-mask gathers would be three stalls per
    micro-batch — at 512 GPUs, three per micro-batch per rank on the pipeline's critical path.
    """
    logits, plane, class_targets, valid = _pp_batch(multi_label=multi_label)
    trainer = _trainer(
        nn.BCEWithLogitsLoss() if multi_label else nn.CrossEntropyLoss(weight=CLASS_WEIGHTS.clone()),
        is_multi_label=multi_label,
    )

    target = {"labels": plane.as_subclass(_NoHostSyncTensor)}
    if multi_label:
        target["class_targets"] = class_targets.as_subclass(_NoHostSyncTensor)

    loss = trainer._pp_classification_token_loss(logits.as_subclass(_NoHostSyncTensor), target)
    assert torch.isfinite(loss.as_subclass(torch.Tensor)).all()
    _ = valid  # the mask is derived inside the loss; the fixture's copy only documents the batch


def test_normalizer_runs_without_a_host_sync():
    """The step normalizer is off the micro-batch path but takes the same masked form."""
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


_WEIGHTS = CLASS_WEIGHTS.tolist()


@pytest.mark.parametrize(
    "overrides, multi_label",
    [
        pytest.param({}, False, id="ce"),
        pytest.param({"class_weights": _WEIGHTS}, False, id="weighted-ce"),
        pytest.param(
            {"loss_type": "label_smoothing_ce", "label_smoothing": 0.1, "class_weights": _WEIGHTS},
            False,
            id="weighted-smoothed-ce",
        ),
        pytest.param({"loss_type": "focal", "class_weights": _WEIGHTS}, False, id="weighted-focal"),
        pytest.param({}, True, id="bce"),
        pytest.param({"class_weights": _WEIGHTS}, True, id="pos-weighted-bce"),
        pytest.param({"loss_type": "focal", "focal_alpha": 0.25}, True, id="alpha-focal"),
    ],
)
def test_pp_and_non_pp_score_one_batch_identically(overrides, multi_label):
    """The same batch through both seams: the non-PP head forward + mean, and the PP last stage's
    per-token head outputs, pooled per micro-batch, summed, and divided by the batch normalizer.

    Every ``_build_loss_fn`` variant, on a real bf16 head whose ragged rows make the pooling position
    load-bearing, split into two uneven micro-batches.
    """
    model = _head(torch.bfloat16)
    batch = _head_batch(multi_label=multi_label)
    trainer = _trainer(None, is_multi_label=multi_label)
    trainer._loss_fn = trainer._build_loss_fn(ClassificationConfig(output_dir="unused", **overrides), None, model)
    trainer.model = model
    trainer.eval_split_rows = lambda num_rows: num_rows  # no eval loader is running: every row is real
    trainer._pp_pool_pad_id = _PAD_ID

    with torch.no_grad():
        non_pp = trainer._compute_loss_inner(model, batch, return_outputs=False)
        hidden = model.model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).last_hidden_state
        per_token = model.score(hidden)
    planes = trainer._pp_classification_batch_transform(batch)
    summed = torch.zeros(())
    for rows in (slice(0, 5), slice(5, None)):
        target = {key: planes[key][rows] for key in ("labels", "class_targets") if key in planes}
        summed = summed + trainer._pp_classification_token_loss(per_token[rows], target)
    pp = summed / trainer._pp_classification_normalizer(planes)

    assert pp.item() == pytest.approx(non_pp.item(), rel=_REDUCTION_ORDER_TOLERANCE)


@pytest.mark.parametrize(
    "overrides, multi_label",
    [
        pytest.param({}, False, id="ce"),
        pytest.param({"class_weights": _WEIGHTS}, False, id="weighted-ce"),
        pytest.param({"loss_type": "focal", "class_weights": _WEIGHTS}, False, id="weighted-focal"),
        pytest.param({}, True, id="bce"),
    ],
)
def test_an_eval_split_s_padding_rows_leave_the_loss(overrides, multi_label):
    """The last 6 of 16 rows pad the eval split's final round: the eval loss is the 10 real rows'
    own, as the same rows score alone, and a rank of padding alone scores 0."""
    model = _head(torch.float32)
    batch = _head_batch(multi_label=multi_label)
    trainer = _trainer(None, is_multi_label=multi_label)
    trainer._loss_fn = trainer._build_loss_fn(ClassificationConfig(output_dir="unused", **overrides), None, model)
    alone = {key: value[:10] for key, value in batch.items()}

    with torch.no_grad():
        expected = trainer._compute_loss_inner(model, alone, return_outputs=False)
        trainer.model = types.SimpleNamespace(training=False)
        trainer.eval_split_rows = lambda num_rows: 10
        padded = trainer._compute_loss_inner(model, batch, return_outputs=False)
        trainer.eval_split_rows = lambda num_rows: 0
        empty = trainer._compute_loss_inner(model, batch, return_outputs=False)

    assert padded.item() == pytest.approx(expected.item(), rel=_REDUCTION_ORDER_TOLERANCE)
    assert empty.item() == 0.0


# --------------------------------------------------------------------------------------------
# Class ids are not token ids
# --------------------------------------------------------------------------------------------


def test_compute_loss_never_runs_the_token_label_check(monkeypatch):
    """The mixin's ``_validate_inputs`` reads ``labels`` as token ids and flags a batch whose labels
    are all pad/eos. Class ids are not token ids: under Gemma's pad=0 / eos=1 every binary batch
    "has no training signal", so the check must stay off this trainer's path."""
    warnings = []
    monkeypatch.setattr(mixin_base, "logger", types.SimpleNamespace(warning=warnings.append))
    trainer = _trainer(nn.CrossEntropyLoss())
    trainer.processing_class = types.SimpleNamespace(pad_token_id=0, eos_token_id=1)
    trainer.accelerator = types.SimpleNamespace(device=torch.device("cpu"))
    trainer.state = None
    trainer._warned_empty_labels = False
    logits, _ = _bf16_batch()
    inputs = {
        "input_ids": torch.zeros(logits.size(0), 4, dtype=torch.long),
        "attention_mask": torch.ones(logits.size(0), 4, dtype=torch.long),
        "labels": torch.arange(logits.size(0)) % 2,
    }

    trainer.compute_loss(_StubHead(logits), inputs)
    assert not warnings, f"compute_loss ran the token-label check on class ids: {warnings}"

    trainer._validate_inputs(inputs)
    assert any("no training signal" in message for message in warnings), (
        "the check does not misfire on these class ids, so the assertion above proves nothing"
    )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
