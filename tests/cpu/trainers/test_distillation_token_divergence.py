#!/usr/bin/env python
"""The chunked per-token reduction every distillation trainer scores its divergences through.

``token_divergence`` runs a divergence one token chunk at a time under a non-reentrant checkpoint, so
backward keeps the two logits planes and no fp32 ``[tokens, V]`` plane of the divergence; the gold-token
gate and the hard-label cross-entropy chunk the same way. Chunking is only legitimate if the numbers do
not move, so every divergence is held to the whole-plane form (values and student gradients, across
several chunks, on the shifted views the trainers pass), and the saved-for-backward set is checked for
fp32 vocab-wide tensors: the whole-plane form keeps several, the chunked one none.

Run: pytest tests/cpu/trainers/test_distillation_token_divergence.py
"""

import pytest
import torch

import halo.kernels.logprobs as logprob_kernels
from halo.kernels.logprobs import LABEL_IGNORE_INDEX
from halo.trainers.distillation.divergences import (
    DIVERGENCES,
    call_divergence,
    gold_token_probs,
    hard_labels_coefficient,
    shifted_token_cross_entropy,
    token_divergence,
)

BATCH, SEQ, VOCAB = 2, 17, 37
TEMPERATURE = 1.7
# Token rows per chunk: SEQ - 1 = 16 shifted rows split into 4 chunks per sequence.
CHUNK_ROWS = 4


@pytest.fixture(autouse=True)
def _small_chunks(monkeypatch):
    monkeypatch.setattr(logprob_kernels, "LOGIT_CHUNK_ELEMENTS", CHUNK_ROWS * VOCAB)


def _inputs(dtype=torch.float32, seed=0):
    generator = torch.Generator().manual_seed(seed)
    student = (torch.randn(BATCH, SEQ, VOCAB, generator=generator) * 3).to(dtype)
    teacher = (student.float() + torch.randn(BATCH, SEQ, VOCAB, generator=generator)).to(dtype)
    labels = torch.randint(0, VOCAB, (BATCH, SEQ - 1), generator=generator)
    labels[0, :3] = LABEL_IGNORE_INDEX
    return student, teacher, labels


def _whole_plane(loss_fn, student, teacher, labels):
    loss = call_divergence(loss_fn, student, teacher, TEMPERATURE, labels)
    return loss.sum(-1) if loss.dim() == student.dim() else loss


def _saved_fp32_planes(fn, *inputs):
    """Run ``fn`` recording what autograd saves; return it and the shapes of the fp32 vocab-wide saved tensors
    that are not views of ``inputs`` (the checkpoint keeps its chunk inputs, which allocate nothing)."""
    saved = []

    def record(tensor):
        saved.append(tensor)
        return tensor

    with torch.autograd.graph.saved_tensors_hooks(record, lambda tensor: tensor):
        out = fn()
    owned = {tensor.untyped_storage().data_ptr() for tensor in inputs}
    planes = [
        tuple(t.shape)
        for t in saved
        if t.dtype == torch.float32 and t.dim() and t.size(-1) == VOCAB and t.untyped_storage().data_ptr() not in owned
    ]
    return out, planes


@pytest.mark.parametrize("name", sorted(DIVERGENCES))
def test_every_divergence_matches_the_whole_plane_on_the_shifted_view(name):
    """Values and the student gradient equal the unchunked form within fp32 rounding, on the
    ``logits[:, :-1]`` view the teacher and SDPG trainers pass."""
    loss_fn = DIVERGENCES[name]
    base, teacher, labels = _inputs()
    chunked_in = base.clone().requires_grad_(True)
    whole_in = base.clone().requires_grad_(True)

    got = token_divergence(loss_fn, chunked_in[:, :-1], teacher[:, :-1], TEMPERATURE, labels)
    want = _whole_plane(loss_fn, whole_in[:, :-1], teacher[:, :-1], labels)

    assert got.shape == (BATCH, SEQ - 1) and got.dtype == torch.float32
    torch.testing.assert_close(got, want)
    weights = torch.rand(BATCH, SEQ - 1, generator=torch.Generator().manual_seed(3))
    (got * weights).sum().backward()
    (want * weights).sum().backward()
    torch.testing.assert_close(chunked_in.grad, whole_in.grad)


@pytest.mark.parametrize("name", sorted(DIVERGENCES))
def test_backward_holds_no_fp32_plane_of_any_divergence(name):
    """On bf16 logits, as the trainers score them, the chunked form saves no fp32 vocab-wide tensor for
    backward, where the whole-plane form saves its upcasts and log-softmax outputs."""
    loss_fn = DIVERGENCES[name]
    base, teacher, labels = _inputs(torch.bfloat16)
    student = base.clone().requires_grad_(True)

    _, planes = _saved_fp32_planes(
        lambda: token_divergence(loss_fn, student[:, :-1], teacher[:, :-1], TEMPERATURE, labels), student, teacher
    )
    _, whole_planes = _saved_fp32_planes(
        lambda: _whole_plane(loss_fn, student[:, :-1], teacher[:, :-1], labels), student, teacher
    )
    assert not planes, f"{name}: fp32 vocab-wide tensors held for backward: {planes}"
    assert whole_planes, "the oracle must hold the planes the chunking removes, or the check above proves nothing"


@pytest.mark.parametrize("name", ["reverse_kl", "forward_kl", "unnormalized_kl"])
def test_gathered_response_rows_match_the_whole_plane(name):
    """Self-distillation scores ``[N, V]`` rows gathered by the response masks."""
    loss_fn = DIVERGENCES[name]
    student, teacher, _ = _inputs(seed=1)
    rows = student.flatten(0, 1).clone().requires_grad_(True)
    reference = rows.detach().clone().requires_grad_(True)
    got = token_divergence(loss_fn, rows, teacher.flatten(0, 1), TEMPERATURE)
    want = _whole_plane(loss_fn, reference, teacher.flatten(0, 1), None)
    assert got.shape == (BATCH * SEQ,)
    torch.testing.assert_close(got, want)
    got.sum().backward()
    want.sum().backward()
    torch.testing.assert_close(rows.grad, reference.grad)


@pytest.mark.parametrize("shape", [(0, SEQ, VOCAB), (0, VOCAB)], ids=["no-sequence", "no-row"])
def test_an_empty_batch_scores_an_empty_result(shape):
    """An eval rank holding only a split's final-round padding scores no row, without raising."""
    empty = torch.zeros(shape)
    got = token_divergence(DIVERGENCES["reverse_kl"], empty, empty, TEMPERATURE)
    assert got.shape == shape[:-1] and got.dtype == torch.float32


def test_a_gold_token_divergence_without_hard_labels_raises():
    student, teacher, _ = _inputs()
    with pytest.raises(ValueError, match="needs hard_labels"):
        token_divergence(DIVERGENCES["slim"], student, teacher, TEMPERATURE)


def test_the_gold_token_gate_matches_the_whole_plane_across_chunks():
    student, teacher, labels = _inputs(torch.bfloat16, seed=2)
    got = hard_labels_coefficient(student[:, :-1], teacher[:, :-1], labels)
    want = (1 - gold_token_probs(student[:, :-1], labels)) * gold_token_probs(teacher[:, :-1], labels)
    assert not got.requires_grad
    torch.testing.assert_close(got, want)


def test_the_hard_label_cross_entropy_holds_no_fp32_plane():
    base, _, labels = _inputs(torch.bfloat16, seed=4)
    logits = base.clone().requires_grad_(True)
    token_ce, planes = _saved_fp32_planes(lambda: shifted_token_cross_entropy(logits[:, :-1], labels), logits)
    assert token_ce.dtype == torch.float32
    assert not planes, f"fp32 vocab-wide tensors held for backward: {planes}"
    assert (token_ce[labels == LABEL_IGNORE_INDEX] == 0).all()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
