#!/usr/bin/env python
"""The SDPG self-distillation trainer's objective (arXiv:2606.04036) on stub models, and its vision-feature reuse gate.
``L = L_sft + beta(k) * L_OPD + alpha * L_ref``, each term a per-sample token mean.
The stub scores a token by its position and sequence length, so a misaligned branch pairing lands on another number."""

import types
from unittest import mock

import pytest
import torch
from torch import nn
from torch.nn.functional import cross_entropy

import halo.kernels.logprobs as logprob_kernels
from halo.configs.distillation import SDPGArguments
from halo.kernels.logprobs import LABEL_IGNORE_INDEX
from halo.trainers.distillation.divergences import reverse_kl_loss, unnormalized_kl_loss
from halo.trainers.distillation.self_distillation import DistributedSelfDistillationTrainer
from halo.trainers.metrics import world_metrics
from halo.trainers.metrics.world_metrics import WorldMetrics
from halo.trainers.sft import DistributedSFTTrainer

VOCAB = 13
EOS = 2
TEMPERATURE = 1.5
BETA = 0.75
ALPHA = 0.4
HINT = [11, 12, 11]
# Per row: (prompt, response); responses end in EOS, one carries a mid-response EOS too.
ROWS = [([5, 6, 7], [8, 9, EOS]), ([4, 3], [10, EOS, 6, 7, EOS]), ([9, 9, 9, 9], [3, EOS])]
WEIGHTS = torch.tensor([0.5, 2.0, 1.25])


def _branch(rows, hinted):
    """Right-padded ``(input_ids, attention_mask, labels)``, labels on the response tokens only."""
    sequences = [prompt + (HINT if hinted else []) + response for prompt, response in rows]
    width = max(map(len, sequences))
    ids = torch.zeros(len(rows), width, dtype=torch.long)
    mask = torch.zeros_like(ids)
    labels = torch.full_like(ids, LABEL_IGNORE_INDEX)
    for row, (sequence, (_, response)) in enumerate(zip(sequences, rows, strict=True)):
        ids[row, : len(sequence)] = torch.tensor(sequence)
        mask[row, : len(sequence)] = 1
        labels[row, len(sequence) - len(response) : len(sequence)] = torch.tensor(response, dtype=torch.long)
    return ids, mask, labels


class _PositionTable(nn.Module):
    """Keyed by token, position and sequence length, so the two branches score a shared response token differently."""

    def __init__(self, seed: int):
        super().__init__()
        self.table = nn.Parameter(torch.randn(VOCAB * 32, VOCAB, generator=torch.Generator().manual_seed(seed)))

    def forward(self, input_ids, attention_mask, use_cache=None):
        positions = torch.arange(input_ids.size(1)).expand_as(input_ids)
        key = input_ids + VOCAB * ((positions + 2 * attention_mask.sum(-1, keepdim=True)) % 32)
        return types.SimpleNamespace(logits=self.table[key])


def _trainer(*, confidence_weight_opd=True, opd_exclude_eos=True, reference_kl_coef=ALPHA):
    trainer = object.__new__(DistributedSelfDistillationTrainer)
    trainer.model = _PositionTable(seed=0)
    trainer._reference_model = _PositionTable(seed=1)
    trainer.sdpg = SDPGArguments(sdpg_beta_base=BETA, sdpg_temperature=TEMPERATURE)
    trainer.sdpg_loss_fn = reverse_kl_loss
    trainer.reference_kl_coef = reference_kl_coef
    trainer.reference_kl_loss_fn = unnormalized_kl_loss
    trainer._reference_vocab_width = None
    trainer.confidence_weight_opd = confidence_weight_opd
    trainer.opd_exclude_eos = opd_exclude_eos
    trainer._stop_token_ids, trainer._stop_ids_tensor = {EOS}, None
    trainer.state = types.SimpleNamespace(global_step=4, max_steps=10)
    trainer._warned_empty_labels = True
    trainer._vision_reuse_setup, trainer._vision_reuse_active = True, False
    trainer.stored = []
    mixin_store = trainer.store_metrics
    trainer.store_metrics = lambda metrics, train_eval="train", rows=1: (
        trainer.stored.append((metrics, train_eval, rows)),
        mixin_store(metrics, train_eval, rows),
    )
    return trainer


def _stored(trainer, train_eval: str = "train") -> dict:
    """The latest value of each metric stored. A trainer that stops storing through the mixin fails here."""
    return {key: value for metrics, mode, _ in trainer.stored if mode == train_eval for key, value in metrics.items()}


def _inputs():
    ids, mask, labels = _branch(ROWS, hinted=False)
    teacher_ids, teacher_mask, teacher_labels = _branch(ROWS, hinted=True)
    return {
        "input_ids": ids,
        "attention_mask": mask,
        "labels": labels,
        "teacher_input_ids": teacher_ids,
        "teacher_attention_mask": teacher_mask,
        "teacher_labels": teacher_labels,
        "confidence_weights": WEIGHTS,
    }


def _response_rows(logits, labels, exclude_eos):
    """Each row's next-token logits at its supervised positions, spelled per row: ``[(n_b, V)]``."""
    rows = []
    for row in range(labels.size(0)):
        positions = [t for t in range(labels.size(1) - 1) if labels[row, t + 1] != LABEL_IGNORE_INDEX]
        if exclude_eos:
            positions = [t for t in positions if labels[row, t + 1] != EOS]
        rows.append(logits[row, positions])
    return rows


def _oracle(*, confidence_weight_opd=True, opd_exclude_eos=True, reference_kl_coef=ALPHA, real_rows=None):
    """``(loss, sft, opd, ref)`` from per-row loops over the hand-paired response positions of ``real_rows`` rows."""
    model, reference = _PositionTable(seed=0), _PositionTable(seed=1)
    inputs = _inputs()
    with torch.no_grad():
        student = model(inputs["input_ids"], inputs["attention_mask"]).logits
        teacher = model(inputs["teacher_input_ids"], inputs["teacher_attention_mask"]).logits
        ref = reference(inputs["input_ids"], inputs["attention_mask"]).logits
    labels = inputs["labels"]
    sft_rows, opd_rows, ref_rows = [], [], []
    for row, (s, t, r) in enumerate(
        zip(
            _response_rows(student, labels, False),
            _response_rows(teacher, inputs["teacher_labels"], opd_exclude_eos),
            _response_rows(ref, labels, False),
            strict=True,
        )
    ):
        targets = labels[row][labels[row] != LABEL_IGNORE_INDEX]
        sft_rows.append(cross_entropy(s, targets))
        keep = targets != EOS if opd_exclude_eos else torch.ones_like(targets, dtype=torch.bool)
        opd_rows.append(reverse_kl_loss(s[keep], t, TEMPERATURE).sum(-1).mean())
        ref_rows.append(unnormalized_kl_loss(s, r, TEMPERATURE).sum(-1).mean())
    weights = WEIGHTS[:real_rows]  # None slices the whole batch
    sft = (torch.stack(sft_rows[:real_rows]) * weights).mean()
    opd = (torch.stack(opd_rows[:real_rows]) * (weights if confidence_weight_opd else 1.0)).mean()
    ref = torch.stack(ref_rows[:real_rows]).mean()
    return sft + BETA * opd + reference_kl_coef * ref, sft, opd, ref


@pytest.mark.parametrize("confidence_weight_opd", [True, False])
@pytest.mark.parametrize("opd_exclude_eos", [True, False])
def test_the_loss_is_sft_plus_scheduled_opd_plus_the_reference_anchor(confidence_weight_opd, opd_exclude_eos):
    trainer = _trainer(confidence_weight_opd=confidence_weight_opd, opd_exclude_eos=opd_exclude_eos)
    loss = trainer.compute_loss(trainer.model, _inputs())
    stored = _stored(trainer)
    expected, sft, opd, ref = _oracle(confidence_weight_opd=confidence_weight_opd, opd_exclude_eos=opd_exclude_eos)
    torch.testing.assert_close(loss.detach(), expected)
    torch.testing.assert_close(stored["sft_loss"], sft)
    torch.testing.assert_close(stored["opd_loss"], opd)
    torch.testing.assert_close(stored["reference_kl"], ref)
    assert stored["opd_beta"] == BETA
    assert all(not value.requires_grad for value in stored.values() if torch.is_tensor(value)), (
        "metrics must be detached"
    )
    train_rows = {rows for _, mode, rows in trainer.stored if mode == "train"}
    assert train_rows == {1}, "a train micro-batch weighs 1, whatever its row count"


def test_an_eval_split_s_padding_rows_leave_every_term():
    """Row 2 pads the eval split's final round."""
    trainer = _trainer()
    trainer.model.eval()
    trainer.eval_split_rows = lambda num_rows: 2
    loss = trainer.compute_loss(trainer.model, _inputs())
    expected, sft, opd, ref = _oracle(real_rows=2)

    torch.testing.assert_close(loss.detach(), expected)
    stored = _stored(trainer, "eval")
    torch.testing.assert_close(stored["sft_loss"], sft)
    torch.testing.assert_close(stored["opd_loss"], opd)
    torch.testing.assert_close(stored["reference_kl"], ref)
    assert {rows for _, mode, rows in trainer.stored if mode == "eval"} == {2}


def test_a_rank_of_eval_padding_alone_scores_zero():
    """Each term means over no sample, which is 0, not NaN."""
    trainer = _trainer()
    trainer.model.eval()
    trainer.eval_split_rows = lambda num_rows: 0

    assert trainer.compute_loss(trainer.model, _inputs()).item() == 0.0


def test_backward_holds_no_fp32_vocab_plane(monkeypatch):
    """Each term scores token chunks under a checkpoint, so bf16 logits keep no fp32 ``[tokens, V]`` autograd plane."""
    monkeypatch.setattr(logprob_kernels, "LOGIT_CHUNK_ELEMENTS", 2 * VOCAB)
    trainer = _trainer()
    trainer.model.to(torch.bfloat16)
    trainer._reference_model.to(torch.bfloat16)
    saved = []

    def record(tensor):
        saved.append(tensor)
        return tensor

    with torch.autograd.graph.saved_tensors_hooks(record, lambda tensor: tensor):
        trainer.compute_loss(trainer.model, _inputs())
    planes = [tuple(t.shape) for t in saved if t.dtype == torch.float32 and t.dim() and t.size(-1) == VOCAB]
    assert not planes, f"fp32 vocab-wide tensors held for backward: {planes}"


def test_the_anchor_is_off_at_a_zero_coefficient():
    trainer = _trainer(reference_kl_coef=0.0)
    expected, _, _, _ = _oracle(reference_kl_coef=0.0)
    torch.testing.assert_close(trainer.compute_loss(trainer.model, _inputs()).detach(), expected)
    assert "reference_kl" not in _stored(trainer)


def _opd(trainer, student_logits, student_labels, teacher_logits, teacher_labels, weights):
    return trainer._response_divergence(
        trainer.sdpg_loss_fn,
        student_logits,
        trainer._opd_mask(student_labels),
        teacher_logits,
        trainer._opd_mask(teacher_labels),
        weights,
    )


def _loop_opd(student_logits, student_labels, teacher_logits, teacher_labels, weights, stop_ids):
    """A per-row reference loop, on batches whose two branches carry equal per-row counts."""
    student_shift, teacher_shift = student_logits[:, :-1], teacher_logits[:, :-1]
    student_ids = student_labels[:, 1:]
    student_mask = student_ids != LABEL_IGNORE_INDEX
    teacher_mask = teacher_labels[:, 1:] != LABEL_IGNORE_INDEX
    per_sample = []
    for b in range(student_shift.size(0)):
        student_rows, teacher_rows = student_shift[b][student_mask[b]], teacher_shift[b][teacher_mask[b]]
        keep = ~torch.isin(student_ids[b][student_mask[b]], stop_ids)
        if not keep.any():
            per_sample.append(student_logits.new_zeros(()))
            continue
        per_sample.append(reverse_kl_loss(student_rows[keep], teacher_rows[keep], TEMPERATURE).sum(-1).mean())
    return (torch.stack(per_sample) * weights).mean()


@pytest.mark.parametrize("seed", range(4))
def test_the_masked_gather_matches_a_per_row_reference_loop(seed):
    """Random logits and offsets, an EOS-only row (no OPD tokens) and an empty one: value and gradient."""
    generator = torch.Generator().manual_seed(seed)
    rows = [([1, 2], [3, 4, EOS]), ([5], [EOS]), ([6, 7, 8], [9, 3, 3, 4, EOS]), ([5, 5], [])]
    _, _, labels = _branch(rows, hinted=False)
    _, _, teacher_labels = _branch(rows, hinted=True)
    student = torch.randn(len(rows), labels.size(1), VOCAB, generator=generator).requires_grad_(True)
    teacher = torch.randn(len(rows), teacher_labels.size(1), VOCAB, generator=generator)
    weights = torch.rand(len(rows), generator=generator)

    trainer = _trainer()
    got = _opd(trainer, student, labels, teacher, teacher_labels, weights)
    (got_grad,) = torch.autograd.grad(got, student)
    expected = _loop_opd(student, labels, teacher, teacher_labels, weights, torch.tensor([EOS]))
    (expected_grad,) = torch.autograd.grad(expected, student)
    torch.testing.assert_close(got, expected)
    torch.testing.assert_close(got_grad, expected_grad)


@pytest.mark.parametrize("totals_match", [False, True], ids=["one-row-longer", "rows-swap-a-token"])
def test_misaligned_branches_raise_instead_of_pairing_different_tokens(totals_match):
    """Per row, not in total: counts that differ with equal totals would pair rows across samples without any raise."""
    trainer = _trainer(opd_exclude_eos=False)
    _, _, labels = _branch(ROWS, hinted=False)
    _, _, teacher_labels = _branch(ROWS, hinted=True)
    first_supervised = int((teacher_labels[0] != LABEL_IGNORE_INDEX).nonzero()[0])
    teacher_labels[0, first_supervised - 1] = 5
    if totals_match:
        last_supervised = int((teacher_labels[1] != LABEL_IGNORE_INDEX).nonzero()[-1])
        teacher_labels[1, last_supervised] = LABEL_IGNORE_INDEX
    logits = torch.zeros(len(ROWS), teacher_labels.size(1), VOCAB)
    with pytest.raises(RuntimeError, match="response alignment contract"):
        _opd(trainer, logits[:, : labels.size(1)], labels, logits, teacher_labels, None)


def test_one_mask_on_both_sides_skips_the_count_check(monkeypatch):
    """The reference term uses one mask for both sides, so comparing it with itself is a wasted host sync."""
    trainer = _trainer()
    _, _, labels = _branch(ROWS, hinted=False)
    logits = torch.zeros(len(ROWS), labels.size(1), VOCAB)
    compared = []
    real_equal = torch.equal
    monkeypatch.setattr(torch, "equal", lambda a, b: compared.append(True) or real_equal(a, b))
    trainer._reference_loss(logits, logits, labels)
    assert compared == []
    _opd(trainer, logits, labels, logits, labels.clone(), None)
    assert compared == [True]


@pytest.mark.parametrize(
    "trainer_cls,flushes", [(DistributedSFTTrainer, 1), (DistributedSelfDistillationTrainer, 0)], ids=["sft", "self"]
)
def test_only_a_loss_recording_the_sft_metrics_flushes_them_per_log(trainer_cls, flushes):
    """That loss records nothing for the per-log flush, which would add a world collective carrying nothing."""
    trainer = object.__new__(trainer_cls)
    trainer.model = types.SimpleNamespace(training=True)
    trainer._loss_metrics = {"train": WorldMetrics(), "eval": WorldMetrics()}
    with mock.patch.object(world_metrics, "_world_states", return_value=({}, ())) as folded:
        trainer._drain_loss_metrics()
    assert folded.call_count == flushes


def _vision_reuse_probe(model_type):
    """Returns ``(activated, wrapper_installed)`` for a fake VLM of ``model_type``."""

    def original_get_image_features(*args, **kwargs):
        return "features"

    inner = nn.Module()
    inner.config = types.SimpleNamespace(model_type=model_type)
    inner.model = types.SimpleNamespace(get_image_features=original_get_image_features)
    me = types.SimpleNamespace(
        _vision_reuse_setup=False,
        _vision_reuse_active=False,
        _vision_record=False,
        _vision_replay=False,
        _vision_cache=None,
    )
    activated = DistributedSelfDistillationTrainer._install_vision_reuse(me, inner)
    return activated, inner.model.get_image_features is not original_get_image_features


def test_vision_reuse_only_for_validated_families():
    """Replaying the student's features on the teacher pass is sound only where both passes call ``get_image_features``
    once with identical inputs (LFM2-VL)."""
    assert _vision_reuse_probe("lfm2_vl") == (True, True)
    for model_type in ("qwen3_vl", "gemma4", "mistral3", None):
        assert _vision_reuse_probe(model_type) == (False, False), model_type


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
