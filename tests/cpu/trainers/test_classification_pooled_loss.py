#!/usr/bin/env python
"""``ClassificationTrainer``'s pooled loss: fp32, no host sync per micro-batch, no token-label check on class ids.
Each breaks silently. CE is a cancelling difference, so in bf16 it lands ~0.3-1% off its exact value."""

import types

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Qwen3Config, Qwen3ForSequenceClassification

import halo.trainers.metrics.token_metrics as token_metrics
from halo.configs.classification import ClassificationConfig
from halo.kernels.logprobs import LABEL_IGNORE_INDEX
from halo.trainers.parallelism.pipeline_loss_adapter import decode_pooling_plane, pooled_logits
from halo.trainers.reward.classification import ClassificationTrainer
from halo.trainers.reward.classification_loss import ClassificationLoss, build_classification_loss, focal_loss

NUM_CLASSES = 6
# Not bf16-representable and spanning two orders of magnitude, as derive_class_weights gives on an imbalanced corpus.
CLASS_WEIGHTS = torch.tensor([0.2571, 17.3, 1.379, 0.6153, 4.77, 0.9091])
_PAD_ID = 0


def _trainer(loss: ClassificationLoss):
    trainer = ClassificationTrainer.__new__(ClassificationTrainer)
    trainer._classification_loss = loss
    trainer.is_multi_label = loss.multi_label
    trainer.model = types.SimpleNamespace(training=True)  # what eval_split_rows reads: a train step
    trainer.accelerator = types.SimpleNamespace(device=torch.device("cpu"))
    trainer._peft_has_been_casted_to_bf16 = False
    return trainer


def _weighted_ce() -> ClassificationLoss:
    return ClassificationLoss(multi_label=False, weight=CLASS_WEIGHTS.clone())


def _head(dtype, *, num_labels=NUM_CLASSES, seed=0):
    """The score head is scaled up: at the 0.02 init the logits sit near zero, where bf16 and fp32 CE barely differ."""
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


def _built_trainer(model, *, is_multi_label=False, **overrides):
    config = ClassificationConfig(output_dir="unused", **overrides)
    return _trainer(build_classification_loss(config, model, None, multi_label=is_multi_label))


def _bf16_batch(rows=16, seed=3):
    """The logits are exact in bf16, so any difference between the paths is the arithmetic's dtype."""
    generator = torch.Generator().manual_seed(seed)
    logits = (torch.randn(rows, NUM_CLASSES, generator=generator) * 6.0).bfloat16()
    return logits, torch.randint(0, NUM_CLASSES, (rows,), generator=generator)


def _plane(rows, valid_rows, values, seq=5):
    plane = torch.full((rows, seq), LABEL_IGNORE_INDEX, dtype=torch.long)
    for row in range(rows):
        if valid_rows[row]:
            plane[row, row % seq] = int(values[row])
    return plane


def test_weighted_ce_matches_an_fp64_reference_on_bf16_logits():
    """Also bounded against the bf16 path: a tolerance around fp64 loose enough for bf16 would pass either way."""
    logits, targets = _bf16_batch()
    valid = torch.ones(logits.size(0), dtype=torch.bool)

    reference = F.cross_entropy(logits.double(), targets, weight=CLASS_WEIGHTS.double(), reduction="sum")
    got = _weighted_ce().masked_sum(logits, targets, valid)
    bf16_path = F.cross_entropy(logits, targets, weight=CLASS_WEIGHTS.bfloat16(), reduction="sum")

    def relative(value):
        return abs(value.double().item() - reference.item()) / abs(reference.item())

    assert relative(got) < 1e-5, f"weighted CE is {relative(got):.2e} off fp64 — it is not evaluating in fp32"
    assert relative(bf16_path) > 1e-3, (
        f"the bf16 path is only {relative(bf16_path):.2e} off fp64, so this batch does not "
        f"separate the two evaluations and the test proves nothing"
    )


def test_weighted_ce_normalizer_matches_an_fp64_reference():
    """Rounding the weights biases the denominator too, and that error does not cancel against the numerator's."""
    _, targets = _bf16_batch()
    valid = torch.ones(targets.size(0), dtype=torch.bool)

    reference = CLASS_WEIGHTS.double()[targets].sum()
    got = _weighted_ce().normalizer(targets, valid)
    bf16_path = CLASS_WEIGHTS.bfloat16()[targets].sum()

    assert abs(got.double().item() - reference.item()) / reference.item() < 1e-5
    assert abs(bf16_path.double().item() - reference.item()) / reference.item() > 1e-3


def test_the_weighted_normalizer_never_divides_an_all_inert_batch_by_zero():
    """``sum / normalizer`` would be NaN on an all-inert micro-batch.
    The floor applies only at zero, so a real batch's denominator is untouched."""
    targets = torch.full((4,), LABEL_IGNORE_INDEX, dtype=torch.long)

    normalizer = _weighted_ce().normalizer(targets, torch.zeros(4, dtype=torch.bool))

    assert normalizer.item() == 1.0
    assert torch.isfinite(torch.zeros(()) / normalizer).item()


class _StubHead:
    def __init__(self, logits):
        self._logits = logits

    def __call__(self, **_kwargs):
        return types.SimpleNamespace(logits=self._logits)


def _non_pp_loss(trainer, logits, labels):
    inputs = {
        "input_ids": torch.zeros(logits.size(0), 4, dtype=torch.long),
        "attention_mask": torch.ones(logits.size(0), 4, dtype=torch.long),
        "labels": labels,
    }
    return trainer.compute_loss(_StubHead(logits), inputs)


def test_non_pp_weighted_loss_evaluates_in_fp32():
    """The PP adapter hands ``masked_sum`` fp32, so only this path could run a weighted CE in bf16, ~1% off."""
    logits, targets = _bf16_batch()
    trainer = _trainer(_weighted_ce())

    got = _non_pp_loss(trainer, logits, targets)
    reference = F.cross_entropy(logits.double(), targets, weight=CLASS_WEIGHTS.double())
    bf16_path = F.cross_entropy(logits, targets, weight=CLASS_WEIGHTS.bfloat16())

    assert got.dtype is torch.float32, f"the non-PP loss reduced in {got.dtype}"
    assert abs(got.double().item() - reference.item()) / reference.item() < 1e-5
    assert abs(bf16_path.double().item() - reference.item()) / reference.item() > 1e-3


def test_non_pp_multi_label_loss_evaluates_in_fp32():
    """BCE takes its output dtype from the target, so a bf16 multi-hot label would hold the loss in bf16."""
    generator = torch.Generator().manual_seed(21)
    logits = (torch.randn(8, NUM_CLASSES, generator=generator) * 6.0).bfloat16()
    targets = (torch.rand(8, NUM_CLASSES, generator=generator) > 0.5).bfloat16()
    trainer = _trainer(ClassificationLoss(multi_label=True, weight=CLASS_WEIGHTS.clone()))

    got = _non_pp_loss(trainer, logits, targets)
    reference = F.binary_cross_entropy_with_logits(
        logits.double(), targets.double(), pos_weight=CLASS_WEIGHTS.double()
    )

    assert got.dtype is torch.float32
    assert abs(got.double().item() - reference.item()) / reference.item() < 1e-5


def test_class_weight_vector_is_never_downcast():
    """The trainer keeps the moved loss, so a downcast on the first bf16 step would last the rest of the run."""
    logits, targets = _bf16_batch()
    trainer = _trainer(_weighted_ce())

    _non_pp_loss(trainer, logits, targets)

    assert trainer._classification_loss.weight.dtype is torch.float32
    assert torch.equal(trainer._classification_loss.weight, CLASS_WEIGHTS)


def test_multi_label_bce_stays_fp32_against_a_bf16_target():
    generator = torch.Generator().manual_seed(11)
    logits = (torch.randn(8, NUM_CLASSES, generator=generator) * 6.0).bfloat16()
    targets = (torch.rand(8, NUM_CLASSES, generator=generator) > 0.5).bfloat16()
    got = ClassificationLoss(multi_label=True).masked_sum(logits, targets, torch.ones(8, dtype=torch.bool))
    reference = F.binary_cross_entropy_with_logits(logits.double(), targets.double(), reduction="sum")

    assert got.dtype is torch.float32, f"multi-label BCE reduced in {got.dtype}"
    assert abs(got.double().item() - reference.item()) / reference.item() < 1e-5


# Every loss the config can build, with the torch module or function that defines its mean.
_LOSS_DEFINITIONS = {
    "ce": (ClassificationLoss(multi_label=False), nn.CrossEntropyLoss()),
    "weighted-ce": (_weighted_ce(), nn.CrossEntropyLoss(weight=CLASS_WEIGHTS.clone())),
    "smoothed-ce": (
        ClassificationLoss(multi_label=False, weight=CLASS_WEIGHTS.clone(), label_smoothing=0.1),
        nn.CrossEntropyLoss(weight=CLASS_WEIGHTS.clone(), label_smoothing=0.1),
    ),
    "bce": (ClassificationLoss(multi_label=True), nn.BCEWithLogitsLoss()),
    "pos-weighted-bce": (
        ClassificationLoss(multi_label=True, weight=CLASS_WEIGHTS.clone()),
        nn.BCEWithLogitsLoss(pos_weight=CLASS_WEIGHTS.clone()),
    ),
    "weighted-focal": (
        ClassificationLoss(multi_label=False, weight=CLASS_WEIGHTS.clone(), focal_gamma=2.0),
        lambda logits, targets: focal_loss(logits, targets, gamma=2.0, weight=CLASS_WEIGHTS),
    ),
    "alpha-focal": (
        ClassificationLoss(multi_label=True, focal_gamma=2.0, focal_alpha=0.25),
        lambda logits, targets: focal_loss(logits, targets, gamma=2.0, alpha=0.25),
    ),
}


def _variant_batch(multi_label: bool, seed: int = 5):
    generator = torch.Generator().manual_seed(seed)
    logits = (torch.randn(8, NUM_CLASSES, generator=generator) * 4.0).bfloat16()
    targets = (
        (torch.rand(8, NUM_CLASSES, generator=generator) > 0.5).bfloat16()
        if multi_label
        else torch.randint(0, NUM_CLASSES, (8,), generator=generator)
    )
    return logits, targets


@pytest.mark.parametrize("variant", sorted(_LOSS_DEFINITIONS))
def test_each_loss_mean_is_its_torch_definition_in_fp32(variant):
    loss, definition = _LOSS_DEFINITIONS[variant]
    logits, targets = _variant_batch(loss.multi_label)
    fp32_targets = targets.float() if targets.is_floating_point() else targets

    assert torch.equal(loss.mean(logits, targets), definition(logits.float(), fp32_targets))


@pytest.mark.parametrize("variant", sorted(_LOSS_DEFINITIONS))
def test_every_loss_variant_returns_fp32(variant):
    """A variant reducing in bf16 would feed the PP runner a summand it then divides by an fp32 normalizer."""
    loss, _ = _LOSS_DEFINITIONS[variant]
    logits, targets = _variant_batch(loss.multi_label)
    assert loss.masked_sum(logits, targets, torch.ones(8, dtype=torch.bool)).dtype is torch.float32


@pytest.mark.parametrize("multi_label", [False, True])
def test_the_default_config_builds_plain_ce_or_bce(multi_label):
    loss = _built_trainer(_head(torch.float32), is_multi_label=multi_label)._classification_loss
    assert loss == ClassificationLoss(multi_label=multi_label)


def test_the_default_loss_is_fp32_on_a_bf16_head():
    """``ForSequenceClassificationLoss`` returns bf16 on bf16 pooled logits, so labels must not go through the head."""
    model = _head(torch.bfloat16)
    inputs = _head_batch()
    trainer = _built_trainer(model)

    got = trainer.compute_loss(model, inputs)
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
    model = _head(torch.float32)
    inputs = _head_batch(multi_label=multi_label)
    trainer = _built_trainer(model, is_multi_label=multi_label)

    with torch.no_grad():
        got = trainer.compute_loss(model, inputs)
        head_loss = model(**inputs).loss

    assert torch.equal(got, head_loss), f"{got.item()} != head loss {head_loss.item()}"


def test_prediction_step_returns_the_head_logits_and_the_labels():
    model = _head(torch.bfloat16)
    inputs = _head_batch()
    trainer = _built_trainer(model)
    trainer._pp_runner = None
    trainer._prepare_inputs = lambda batch: batch

    loss, logits, labels = trainer.prediction_step(model, inputs, prediction_loss_only=False)
    with torch.no_grad():
        expected_logits = model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"]).logits

    assert loss.dtype is torch.float32 and not loss.requires_grad
    assert torch.equal(logits, expected_logits)
    assert torch.equal(labels, inputs["labels"])


def test_soft_label_targets_are_refused_on_both_paths():
    """``F.cross_entropy`` reads ``[B, C]`` float targets as class probabilities, which trains soft-label CE."""
    model = _head(torch.float32)
    trainer = _built_trainer(model)
    trainer._pp_pool_pad_id = _PAD_ID
    batch = _head_batch()
    batch["labels"] = F.one_hot(batch["labels"], NUM_CLASSES).float()

    with pytest.raises(ValueError, match="integer class ids"):
        trainer.compute_loss(model, batch)
    with pytest.raises(ValueError, match="integer class ids"):
        trainer._pp_classification_batch_transform(batch)


def test_a_single_logit_single_label_head_is_refused():
    """``num_labels == 1`` is transformers' regression convention; softmax CE over one logit is identically zero."""
    with pytest.raises(ValueError, match="num_labels >= 2"):
        _built_trainer(_head(torch.float32, num_labels=1))
    assert _built_trainer(_head(torch.float32, num_labels=1), is_multi_label=True)._classification_loss.multi_label


class _NoHostSyncTensor(torch.Tensor):
    """Raises on any host read-back: a CPU stand-in for ``torch.cuda.set_sync_debug_mode("error")``.
    Boolean-mask indexing counts, since it lowers to ``nonzero``, whose output shape only the device knows."""

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
    generator = torch.Generator().manual_seed(seed)
    valid = torch.tensor([True, False, True, True, False, True][:rows])
    values = torch.randint(0, NUM_CLASSES, (rows,), generator=generator)
    plane = _plane(rows, valid, values, seq=seq)
    logits = (torch.randn(rows, seq, NUM_CLASSES, generator=generator) * 5.0).bfloat16()
    class_targets = (torch.rand(rows, NUM_CLASSES, generator=generator) > 0.5).float() if multi_label else None
    return logits, plane, class_targets, valid


# Inert rows stay in the reduction as exact zeros. They lengthen the vector, and torch's pairwise sum blocks on length,
# so the result lands within a couple of fp32 ULP. Measured worst case over 60 random batches x each loss: 2.1e-7.
_REDUCTION_ORDER_TOLERANCE = 1e-6


@pytest.mark.parametrize("multi_label", [False, True])
def test_masked_sum_equals_indexing_the_surviving_rows_out(multi_label):
    """Bounded on both sides: a tolerance this loose could otherwise hide a mask that never applied."""
    logits, plane, class_targets, valid = _pp_batch(multi_label=multi_label)
    loss = ClassificationLoss(multi_label=True) if multi_label else _weighted_ce()

    decoded = decode_pooling_plane(plane)
    pooled = pooled_logits(logits, decoded)
    targets = class_targets if multi_label else decoded.values.clamp_min(0)
    all_rows = torch.ones(pooled.size(0), dtype=torch.bool)

    masked = loss.masked_sum(pooled, targets, valid)
    indexed = loss.masked_sum(pooled[valid], targets[valid], all_rows[valid])
    unmasked = loss.masked_sum(pooled, targets, all_rows)

    assert masked.item() == pytest.approx(indexed.item(), rel=_REDUCTION_ORDER_TOLERANCE)
    assert abs(unmasked.item() - indexed.item()) / indexed.item() > 0.01, (
        "the inert rows contribute almost nothing to this batch, so the assertion above would pass "
        "with the mask removed"
    )


@pytest.mark.parametrize("multi_label", [False, True])
def test_masked_normalizer_equals_indexing_the_surviving_rows_out(multi_label):
    """Numerator and denominator must agree on which rows exist, or the loss rescales by the inert-row fraction."""
    _, plane, class_targets, valid = _pp_batch(multi_label=multi_label)
    loss = ClassificationLoss(multi_label=True) if multi_label else _weighted_ce()

    targets = class_targets if multi_label else decode_pooling_plane(plane).values
    ones = torch.ones(int(valid.sum()), dtype=torch.bool)

    masked = loss.normalizer(targets, valid)
    indexed = loss.normalizer(targets[valid], ones)
    unmasked = loss.normalizer(targets, torch.ones_like(valid))

    assert masked.item() == pytest.approx(indexed.item(), rel=_REDUCTION_ORDER_TOLERANCE)
    assert unmasked.item() != pytest.approx(indexed.item(), rel=_REDUCTION_ORDER_TOLERANCE)


@pytest.mark.parametrize("multi_label", [False, True])
def test_token_loss_runs_without_a_single_host_sync(multi_label):
    """``bool(valid.any())`` and two boolean-mask gathers would be three stalls per micro-batch."""
    logits, plane, class_targets, valid = _pp_batch(multi_label=multi_label)
    trainer = _trainer(ClassificationLoss(multi_label=True) if multi_label else _weighted_ce())

    target = {"labels": plane.as_subclass(_NoHostSyncTensor)}
    if multi_label:
        target["class_targets"] = class_targets.as_subclass(_NoHostSyncTensor)

    loss = trainer._pp_classification_token_loss(logits.as_subclass(_NoHostSyncTensor), target)
    assert torch.isfinite(loss.as_subclass(torch.Tensor)).all()
    _ = valid  # the loss derives the mask itself


def test_normalizer_runs_without_a_host_sync():
    _, plane, _, _ = _pp_batch()
    trainer = _trainer(_weighted_ce())
    normalizer = trainer._pp_classification_normalizer({"labels": plane.as_subclass(_NoHostSyncTensor)})
    assert normalizer.as_subclass(torch.Tensor).item() > 0


def test_all_inert_microbatch_is_zero_and_still_wired_to_the_graph():
    """A loss detached from the stage's activations leaves the pipeline waiting on a gradient that never arrives."""
    rows, seq = 4, 5
    plane = torch.full((rows, seq), LABEL_IGNORE_INDEX, dtype=torch.long)
    logits = torch.randn(rows, seq, NUM_CLASSES, requires_grad=True)
    trainer = _trainer(_weighted_ce())

    loss = trainer._pp_classification_token_loss(logits, plane)

    assert loss.item() == 0.0
    assert loss.requires_grad, "an all-inert micro-batch returned a loss detached from the stage output"
    loss.backward()
    assert logits.grad is not None and torch.count_nonzero(logits.grad) == 0


def test_inert_rows_cannot_leak_a_nan_from_a_saturated_logit():
    """An inert row's logits are whatever the padder left, so they are neutralized before the objective.
    Masking only the loss would still leave ``softmax(inf)`` NaN in the backward, carried through the masked zero."""
    rows, seq = 3, 4
    valid = torch.tensor([True, False, True])
    plane = _plane(rows, valid, torch.tensor([1, 0, 2]), seq=seq)
    logits = torch.randn(rows, seq, NUM_CLASSES)
    logits[1] = float("inf")
    logits.requires_grad_(True)
    trainer = _trainer(_weighted_ce())

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
    """Every built loss, on a bf16 head whose ragged rows make pooling position matter, in two uneven micro-batches."""
    model = _head(torch.bfloat16)
    batch = _head_batch(multi_label=multi_label)
    trainer = _built_trainer(model, is_multi_label=multi_label, **overrides)
    trainer.model = model
    trainer.eval_split_rows = lambda num_rows: num_rows  # no eval loader is running: every row is real
    trainer._pp_pool_pad_id = _PAD_ID

    with torch.no_grad():
        non_pp = trainer.compute_loss(model, batch)
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
    """The last 6 of 16 rows pad the final round: the loss is the 10 real rows', and a padding-only rank scores 0."""
    model = _head(torch.float32)
    batch = _head_batch(multi_label=multi_label)
    trainer = _built_trainer(model, is_multi_label=multi_label, **overrides)
    alone = {key: value[:10] for key, value in batch.items()}

    with torch.no_grad():
        expected = trainer.compute_loss(model, alone)
        trainer.model = types.SimpleNamespace(training=False)
        trainer.eval_split_rows = lambda num_rows: 10
        padded = trainer.compute_loss(model, batch)
        trainer.eval_split_rows = lambda num_rows: 0
        empty = trainer.compute_loss(model, batch)

    assert padded.item() == pytest.approx(expected.item(), rel=_REDUCTION_ORDER_TOLERANCE)
    assert empty.item() == 0.0


def test_compute_loss_never_runs_the_token_label_check(monkeypatch):
    """Under Gemma's pad=0 / eos=1 every binary batch would read as all pad/eos and warn of no training signal."""
    warnings = []
    monkeypatch.setattr(token_metrics, "logger", types.SimpleNamespace(warning=warnings.append))
    trainer = _trainer(ClassificationLoss(multi_label=False))
    trainer.processing_class = types.SimpleNamespace(pad_token_id=0, eos_token_id=1)
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

    trainer._warn_once_on_empty_labels(inputs)
    assert any("no training signal" in message for message in warnings), (
        "the check does not misfire on these class ids, so the assertion above proves nothing"
    )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
