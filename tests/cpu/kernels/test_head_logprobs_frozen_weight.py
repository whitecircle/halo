#!/usr/bin/env python
"""The chunked head backward computes only the gradients autograd asks for, so a frozen ``lm_head`` builds no fp32
``[V, H]`` buffer."""

import pytest
import torch
from torch.utils._python_dispatch import TorchDispatchMode

import halo.kernels.head_logprobs as head_logprobs
from halo.kernels.head_logprobs import chunked_head_logprobs

ROWS, HIDDEN, VOCAB = 13, 24, 40
SEQ_TILE, VOCAB_TILE = 5, 16
TILES = -(-ROWS // SEQ_TILE) * -(-VOCAB // VOCAB_TILE)
GEMMS = (torch.ops.aten.mm.default, torch.ops.aten.addmm.default)


class _BackwardOps(TorchDispatchMode):
    """The GEMM count and the shape and dtype of every tensor an op returns."""

    def __init__(self):
        super().__init__()
        self.gemms = 0
        self.outputs: list[tuple[tuple[int, ...], torch.dtype]] = []

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        out = func(*args, **(kwargs or {}))
        if func in GEMMS:
            self.gemms += 1
        for t in out if isinstance(out, (tuple, list)) else (out,):
            if isinstance(t, torch.Tensor):
                self.outputs.append((tuple(t.shape), t.dtype))
        return out


def _inputs(with_bias):
    torch.manual_seed(0)
    hidden = torch.randn(ROWS, HIDDEN)
    weight = torch.randn(VOCAB, HIDDEN) * 0.2
    bias = torch.randn(VOCAB) * 0.1 if with_bias else None
    targets = torch.randint(0, VOCAB, (ROWS,))
    upstream = torch.randn(ROWS)
    return hidden, weight, bias, targets, upstream


def _backward(hidden, weight, bias, targets, upstream, *, train_head):
    """``(grads, ops)``."""
    hidden = hidden.clone().requires_grad_(True)
    weight = weight.clone().requires_grad_(train_head)
    bias = None if bias is None else bias.clone().requires_grad_(train_head)
    logps = chunked_head_logprobs(hidden, weight, targets, bias, 0.7, logit_scale=1.3, softcap=4.0)[0]
    with _BackwardOps() as ops:
        (logps * upstream).sum().backward()
    return (hidden.grad, weight.grad, None if bias is None else bias.grad), ops


@pytest.fixture(autouse=True)
def _small_tiles(monkeypatch):
    monkeypatch.setattr(head_logprobs, "SEQ_TILE", SEQ_TILE)
    monkeypatch.setattr(head_logprobs, "VOCAB_TILE", VOCAB_TILE)


@pytest.mark.parametrize("with_bias", [False, True])
def test_frozen_head_builds_no_weight_gradient(with_bias):
    (grad_hidden, grad_weight, grad_bias), ops = _backward(*_inputs(with_bias), train_head=False)
    assert grad_weight is None and grad_bias is None
    assert (VOCAB, HIDDEN) not in [shape for shape, _ in ops.outputs], "a frozen head built a [V, H] buffer"
    assert ops.gemms == 2 * TILES, "per tile: the logits recompute and the hidden-gradient GEMM, nothing else"
    assert grad_hidden is not None and torch.isfinite(grad_hidden).all()


@pytest.mark.parametrize("with_bias", [False, True])
def test_frozen_head_hidden_gradient_is_bitwise_the_trainable_one(with_bias):
    inputs = _inputs(with_bias)
    frozen, _ = _backward(*inputs, train_head=False)
    trainable, ops = _backward(*inputs, train_head=True)
    assert torch.equal(frozen[0], trainable[0])
    assert ops.gemms == 3 * TILES, "a trainable head runs the weight-gradient GEMM on every tile"


@pytest.mark.parametrize("with_bias", [False, True])
def test_trainable_head_gradients_match_autograd(with_bias):
    hidden, weight, bias, targets, upstream = _inputs(with_bias)
    (grad_hidden, grad_weight, grad_bias), _ = _backward(hidden, weight, bias, targets, upstream, train_head=True)

    ref_leaves = [t.clone().requires_grad_(True) for t in (hidden, weight)]
    ref_bias = None if bias is None else bias.clone().requires_grad_(True)
    logits = ref_leaves[0] @ ref_leaves[1].t()
    if ref_bias is not None:
        logits = logits + ref_bias
    logits = (torch.tanh(logits * 1.3 / 4.0) * 4.0) / 0.7
    ref = torch.log_softmax(logits, dim=-1).gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    (ref * upstream).sum().backward()

    torch.testing.assert_close(grad_hidden, ref_leaves[0].grad, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(grad_weight, ref_leaves[1].grad, rtol=1e-5, atol=1e-6)
    if bias is not None:
        torch.testing.assert_close(grad_bias, ref_bias.grad, rtol=1e-5, atol=1e-6)


def test_frozen_hidden_builds_no_hidden_gradient():
    hidden, weight, _, targets, upstream = _inputs(False)
    weight = weight.clone().requires_grad_(True)
    logps = chunked_head_logprobs(hidden, weight, targets, None, 1.0)[0]
    with _BackwardOps() as ops:
        (logps * upstream).sum().backward()
    assert weight.grad is not None
    assert ops.gemms == 2 * TILES
    assert (ROWS, HIDDEN) not in [shape for shape, _ in ops.outputs]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
