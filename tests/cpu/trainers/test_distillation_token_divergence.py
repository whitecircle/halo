#!/usr/bin/env python
"""The chunked per-token reduction every distillation trainer scores its divergences through.
Chunks run under a non-reentrant checkpoint, so backward keeps no fp32 ``[tokens, V]`` plane; values must not move."""

import multiprocessing

import pytest
import torch

import halo.kernels.logprobs as logprob_kernels
from halo.kernels.logprobs import LABEL_IGNORE_INDEX, selective_logprobs
from halo.trainers.distillation.divergences import (
    DIVERGENCES,
    call_divergence,
    global_token_mean,
    gold_token_probs,
    hard_labels_coefficient,
    shifted_token_cross_entropy,
    token_divergence,
    token_divergence_and_cross_entropy,
)
from tests.common.memory import host_peak_rss_gib, reset_host_peak_rss

BATCH, SEQ, VOCAB = 2, 17, 37
TEMPERATURE = 1.7
# Token rows per chunk: SEQ - 1 = 16 shifted rows split into 4 chunks per sequence.
CHUNK_ROWS = 4
# One bf16 [B, S, V] student plane is 128 MiB, so a whole gradient plane stands far above the allocator's noise.
PEAK_SHAPE = (2, 1024, 32768)
DISTILL_ALPHA = 0.5


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
    """Excludes views of ``inputs``: the checkpoint keeps its chunk inputs, which allocate nothing."""
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
    """On the ``logits[:, :-1]`` view the teacher and SDPG trainers pass."""
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


def test_the_cross_entropy_scores_and_trains_what_the_log_prob_form_does():
    """The selective log-prob form is the oracle, bit for bit."""
    base, _, labels = _inputs(torch.bfloat16, seed=5)
    logits, oracle_in = base.clone().requires_grad_(True), base.clone().requires_grad_(True)
    supervised = labels != LABEL_IGNORE_INDEX
    got = shifted_token_cross_entropy(logits[:, :-1], labels)
    want = -selective_logprobs(oracle_in[:, :-1], labels.masked_fill(~supervised, 0), supervised)
    assert torch.equal(got, want)
    weights = torch.rand(BATCH, SEQ - 1, generator=torch.Generator().manual_seed(6))
    (got * weights).sum().backward()
    (want * weights).sum().backward()
    assert torch.equal(logits.grad, oracle_in.grad)


@pytest.mark.parametrize("name", sorted(DIVERGENCES))
def test_one_pass_scores_and_trains_what_two_passes_do(name):
    loss_fn = DIVERGENCES[name]
    base, teacher, labels = _inputs(torch.bfloat16, seed=7)
    joint_in, split_in = base.clone().requires_grad_(True), base.clone().requires_grad_(True)
    divergence, cross_entropy = token_divergence_and_cross_entropy(
        loss_fn, joint_in[:, :-1], teacher[:, :-1], TEMPERATURE, labels
    )
    want_divergence = token_divergence(loss_fn, split_in[:, :-1], teacher[:, :-1], TEMPERATURE, labels)
    want_cross_entropy = shifted_token_cross_entropy(split_in[:, :-1], labels)
    assert torch.equal(divergence, want_divergence)
    assert torch.equal(cross_entropy, want_cross_entropy)
    weights = torch.rand(2, BATCH, SEQ - 1, generator=torch.Generator().manual_seed(8))
    ((divergence * weights[0]).sum() + (cross_entropy * weights[1]).sum()).backward()
    ((want_divergence * weights[0]).sum() + (want_cross_entropy * weights[1]).sum()).backward()
    assert torch.equal(joint_in.grad, split_in.grad)


def _peak_gradient_planes(one_pass: bool) -> float:
    """In student planes, above the logits it reads. Runs in a fresh process so no earlier allocation is reused."""
    batch, seq, vocab = PEAK_SHAPE
    generator = torch.Generator().manual_seed(0)
    student = torch.randn(batch, seq + 1, vocab, generator=generator).to(torch.bfloat16).requires_grad_(True)
    teacher = torch.randn(batch, seq + 1, vocab, generator=generator).to(torch.bfloat16)
    labels = torch.randint(0, vocab, (batch, seq), generator=generator)
    supervised = labels != LABEL_IGNORE_INDEX
    loss_fn = DIVERGENCES["forward_kl"]
    reset_host_peak_rss()
    baseline = host_peak_rss_gib()
    if one_pass:
        divergence, cross_entropy = token_divergence_and_cross_entropy(
            loss_fn, student[:, :-1], teacher[:, :-1], 1.0, labels
        )
    else:
        divergence = token_divergence(loss_fn, student[:, :-1], teacher[:, :-1], 1.0, labels)
        cross_entropy = shifted_token_cross_entropy(student[:, :-1], labels)
    loss = DISTILL_ALPHA * global_token_mean(divergence, supervised)
    loss = loss + (1 - DISTILL_ALPHA) * global_token_mean(cross_entropy, supervised)
    loss.backward()
    plane_gib = batch * seq * vocab * student.element_size() / 1024**3
    return (host_peak_rss_gib() - baseline) / plane_gib


def test_one_pass_builds_one_student_gradient_plane_fewer():
    """Two passes hold the divergence's student gradient while building the cross-entropy's, about one plane more."""
    with multiprocessing.get_context("spawn").Pool(1, maxtasksperchild=1) as pool:
        one_pass = pool.apply(_peak_gradient_planes, (True,))
    with multiprocessing.get_context("spawn").Pool(1, maxtasksperchild=1) as pool:
        two_passes = pool.apply(_peak_gradient_planes, (False,))
    assert two_passes - one_pass > 0.5, f"peak above the logits: one pass {one_pass:.2f}, two {two_passes:.2f} planes"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
