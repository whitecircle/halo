#!/usr/bin/env python
"""The chunked head scoring ``m`` targets per row (a sampled token and its candidates) under one ``log Z``, against a
dense float64 softmax over the full vocabulary."""

import pytest
import torch

import halo.kernels.head_logprobs as head_logprobs
from halo.kernels.head_logprobs import chunked_head_logprobs

ROWS, HIDDEN, VOCAB, M = 11, 12, 70, 5
SEQ_TILE, VOCAB_TILE = 4, 16
HEADS = {
    "plain": {"temperature": 1.0, "logit_scale": None, "softcap": None},
    "temperature_softcap": {"temperature": 0.7, "logit_scale": None, "softcap": 2.0},
    "every_term": {"temperature": 1.3, "logit_scale": 2.5, "softcap": 3.0},
}


@pytest.fixture(autouse=True)
def small_tiles(monkeypatch):
    monkeypatch.setattr(head_logprobs, "SEQ_TILE", SEQ_TILE)
    monkeypatch.setattr(head_logprobs, "VOCAB_TILE", VOCAB_TILE)


def _inputs(seed=0):
    generator = torch.Generator().manual_seed(seed)
    hidden = torch.randn(ROWS, HIDDEN, generator=generator, dtype=torch.float64)
    weight = torch.randn(VOCAB, HIDDEN, generator=generator, dtype=torch.float64) * 0.4
    bias = torch.randn(VOCAB, generator=generator, dtype=torch.float64) * 0.2
    # Candidates spread over every vocab tile (70 ids across tiles of 16), masked slots, and a candidate that repeats
    # the sampled token.
    targets = torch.randint(0, VOCAB, (ROWS, M), generator=generator)
    targets[:, 0] = torch.arange(ROWS) * 6 % VOCAB
    targets[3, 2] = targets[3, 0]
    targets[::4, 3] = -1
    targets[5, 1:] = -1
    upstream = torch.randn(ROWS, M, generator=generator, dtype=torch.float64)
    return hidden, weight, bias, targets, upstream


def _reference(hidden, weight, bias, targets, upstream, temperature, logit_scale, softcap):
    leaves = [t.clone().requires_grad_(True) for t in (hidden, weight, bias)]
    logits = leaves[0] @ leaves[1].T + leaves[2]
    if logit_scale is not None:
        logits = logits * logit_scale
    if softcap is not None:
        logits = torch.tanh(logits / softcap) * softcap
    logps = torch.log_softmax(logits / temperature, dim=-1).gather(1, targets.clamp(min=0))
    valid = targets >= 0
    (logps * torch.where(valid, upstream, 0.0)).sum().backward()
    return logps.detach().masked_fill(~valid, float("-inf")), *(leaf.grad for leaf in leaves)


def _kernel(hidden, weight, bias, targets, upstream, temperature, logit_scale, softcap):
    leaves = [t.clone().requires_grad_(True) for t in (hidden, weight, bias)]
    logps, _ = chunked_head_logprobs(
        leaves[0], leaves[1], targets, leaves[2], temperature, logit_scale=logit_scale, softcap=softcap
    )
    # An upstream gradient that is NaN on masked slots must not leak: the objective's masked terms can be.
    (torch.where(targets >= 0, logps, 0.0) * upstream).sum().backward()
    return logps.detach(), *(leaf.grad for leaf in leaves)


@pytest.mark.parametrize("head", sorted(HEADS))
def test_candidate_logprobs_and_gradients_match_a_dense_float64_softmax(head):
    inputs = _inputs()
    got, ref = _kernel(*inputs, **HEADS[head]), _reference(*inputs, **HEADS[head])
    assert torch.equal(got[0] == float("-inf"), inputs[3] < 0)
    for g, r in zip(got, ref, strict=True):
        torch.testing.assert_close(g.double(), r, atol=2e-5, rtol=2e-5)


def test_masked_candidates_contribute_no_gradient():
    hidden, weight, bias, targets, upstream = _inputs()
    full = _kernel(hidden, weight, bias, targets, upstream, **HEADS["plain"])
    upstream = upstream.clone()
    upstream[targets < 0] = 1e6
    changed = _kernel(hidden, weight, bias, targets, upstream, **HEADS["plain"])
    for a, b in zip(full[1:], changed[1:], strict=True):
        assert torch.equal(a, b)


@pytest.mark.parametrize("head", sorted(HEADS))
def test_one_column_equals_the_flat_path_bitwise(head):
    hidden, weight, bias, targets, upstream = _inputs()
    flat_targets, flat_upstream = targets[:, 0].contiguous(), upstream[:, 0].contiguous()
    flat = _kernel_flat(hidden, weight, bias, flat_targets, flat_upstream, **HEADS[head])
    column = _kernel(hidden, weight, bias, flat_targets[:, None], flat_upstream[:, None], **HEADS[head])
    assert torch.equal(column[0][:, 0], flat[0])
    for a, b in zip(column[1:], flat[1:], strict=True):
        assert torch.equal(a, b)


def _kernel_flat(hidden, weight, bias, targets, upstream, temperature, logit_scale, softcap):
    leaves = [t.clone().requires_grad_(True) for t in (hidden, weight, bias)]
    logps, _ = chunked_head_logprobs(
        leaves[0], leaves[1], targets, leaves[2], temperature, logit_scale=logit_scale, softcap=softcap
    )
    (logps * upstream).sum().backward()
    return logps.detach(), *(leaf.grad for leaf in leaves)


def test_leading_dimensions_and_entropy_follow_the_targets():
    hidden, weight, bias, targets, _ = _inputs()
    rows = 8
    batched_hidden, batched_targets = hidden[:rows].reshape(2, 4, HIDDEN), targets[:rows].reshape(2, 4, M)
    logps, entropy = chunked_head_logprobs(batched_hidden, weight, batched_targets, bias, 0.9, compute_entropy=True)
    flat_logps, flat_entropy = chunked_head_logprobs(
        hidden[:rows], weight, targets[:rows], bias, 0.9, compute_entropy=True
    )
    assert logps.shape == (2, 4, M) and entropy.shape == (2, 4)
    assert torch.equal(logps.reshape(rows, M), flat_logps)
    assert torch.equal(entropy.reshape(rows), flat_entropy)


def test_entropy_is_per_row_and_independent_of_the_targets():
    hidden, weight, bias, targets, _ = _inputs()
    head = HEADS["temperature_softcap"]
    _, entropy = chunked_head_logprobs(hidden, weight, targets, bias, compute_entropy=True, **head)
    _, flat_entropy = chunked_head_logprobs(hidden, weight, targets[:, 0], bias, compute_entropy=True, **head)
    logits = torch.tanh((hidden @ weight.T + bias) / head["softcap"]) * head["softcap"]
    logp = torch.log_softmax(logits / head["temperature"], dim=-1)
    reference = -(logp.exp() * logp).sum(-1)
    assert entropy.shape == (ROWS,)
    assert torch.equal(entropy, flat_entropy)
    torch.testing.assert_close(entropy.double(), reference, atol=2e-5, rtol=2e-5)
    assert chunked_head_logprobs(hidden, weight, targets, bias, **head)[1] is None


def test_a_wrong_backward_fails_the_gradient_check(monkeypatch):
    # Negative control: dropping the softmax term's weight (sum of the unmasked upstreams) must be caught.
    inputs = _inputs()
    original = head_logprobs._backward_tile

    def one_weight_softmax(raw, bias, targets, vocab_start, log_z, grad, *rest):
        if targets.dim() == 2:
            grad = grad.clone()
            grad[:, 1:] = 0.0
        return original(raw, bias, targets, vocab_start, log_z, grad, *rest)

    monkeypatch.setattr(head_logprobs, "_backward_tile", one_weight_softmax)
    monkeypatch.setattr(head_logprobs, "_COMPILED_TILES", (head_logprobs._forward_tile, one_weight_softmax))
    got = _kernel(*inputs, **HEADS["plain"])
    ref = _reference(*inputs, **HEADS["plain"])
    assert not torch.allclose(got[1].double(), ref[1], atol=2e-5, rtol=2e-5)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
