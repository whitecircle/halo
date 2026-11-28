#!/usr/bin/env python
"""The chunked head sweep with LoRA deltas on the head: each adapter's ``scaling · x @ A.T @ B.T`` joins every tile in
the forward and the recompute backward, as its low-rank pair, never as a ``[V, H]`` sum."""

import pytest
import torch
from torch.utils._python_dispatch import TorchDispatchMode

import halo.kernels.head_logprobs as head_logprobs
from halo.kernels.head_logprobs import LowRankDelta, chunked_head_logprobs

ROWS, HIDDEN, VOCAB = 13, 6, 23
RANKS = (3, 4)
SCALINGS = (0.7, 2.0)
TEMPERATURE = 0.9
LOGIT_SCALE, SOFTCAP = 1.3, 4.0


@pytest.fixture(autouse=True)
def _small_tiles(monkeypatch):
    """Several sequence and vocab tiles, the last of each ragged, so a factor or gradient kept across tiles shows."""
    monkeypatch.setattr(head_logprobs, "SEQ_TILE", 5)
    monkeypatch.setattr(head_logprobs, "VOCAB_TILE", 7)


def _leaf(*shape, trains: bool = True) -> torch.Tensor:
    return torch.randn(*shape, dtype=torch.float64).requires_grad_(trains)


def _operands(adapters_train: bool = True):
    torch.manual_seed(0)
    hidden, weight, bias = _leaf(ROWS, HIDDEN), _leaf(VOCAB, HIDDEN), _leaf(VOCAB)
    deltas = [
        LowRankDelta(
            _leaf(ROWS, HIDDEN), _leaf(r, HIDDEN, trains=adapters_train), _leaf(VOCAB, r, trains=adapters_train), s
        )
        for r, s in zip(RANKS, SCALINGS, strict=True)
    ]
    return hidden, weight, bias, deltas


def _targets(candidates: bool) -> torch.Tensor:
    generator = torch.Generator().manual_seed(1)
    if not candidates:
        return torch.randint(0, VOCAB, (ROWS,), generator=generator)
    targets = torch.randint(0, VOCAB, (ROWS, 3), generator=generator)
    targets[0, 2] = -1
    return targets


def _reference(hidden, weight, bias, deltas, targets):
    """``(logps, entropy)`` from the materialized logits, autograd through every operand."""
    logits = hidden @ weight.T + bias + sum(d.scaling * (d.inputs @ d.down.T @ d.up.T) for d in deltas)
    logits = torch.tanh(logits * LOGIT_SCALE / SOFTCAP) * SOFTCAP / TEMPERATURE
    log_probs = logits.log_softmax(-1)
    entropy = -(log_probs.exp() * log_probs).sum(-1)
    if targets.dim() == 1:
        return log_probs.gather(-1, targets[:, None])[:, 0], entropy
    picked = log_probs.gather(-1, targets.clamp(min=0))
    return picked.masked_fill(targets < 0, float("-inf")), entropy


def _swept(hidden, weight, bias, deltas, targets):
    return chunked_head_logprobs(
        hidden,
        weight,
        targets,
        bias,
        TEMPERATURE,
        logit_scale=LOGIT_SCALE,
        softcap=SOFTCAP,
        compute_entropy=True,
        low_rank=deltas,
    )


def _weighted(logps: torch.Tensor) -> torch.Tensor:
    weights = torch.randn(logps.shape, dtype=torch.float64, generator=torch.Generator().manual_seed(2))
    return (torch.where(logps.isfinite(), logps, 0.0) * weights).sum()


@pytest.mark.parametrize("candidates", [False, True], ids=["sampled", "candidates"])
def test_the_sweep_matches_autograd_through_two_adapters(candidates):
    """Log-probs, entropy and the gradient of every operand, each adapter's input, ``A`` and ``B`` included."""
    hidden, weight, bias, deltas = _operands()
    targets = _targets(candidates)
    operands = [hidden, weight, bias, *(t for d in deltas for t in (d.inputs, d.down, d.up))]
    logps, entropy = _swept(hidden, weight, bias, deltas, targets)
    got = torch.autograd.grad(_weighted(logps), operands)
    want_logps, want_entropy = _reference(hidden, weight, bias, deltas, targets)
    want = torch.autograd.grad(_weighted(want_logps), operands)
    without_adapters, _ = _swept(hidden, weight, bias, [], targets)
    assert (without_adapters - want_logps).abs().nan_to_num().max() > 1e-2, "the deltas must move the log-probs"
    torch.testing.assert_close(logps, want_logps.float(), rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(entropy, want_entropy.float(), rtol=1e-5, atol=1e-5)
    names = ["hidden", "weight", "bias", *(f"{n}{i}" for i in range(len(deltas)) for n in ("inputs", "down", "up"))]
    for name, grad, expected in zip(names, got, want, strict=True):
        torch.testing.assert_close(grad, expected, rtol=1e-5, atol=1e-5, msg=name)


GEMMS = (torch.ops.aten.mm.default, torch.ops.aten.addmm.default)


class _OpOutputs(TorchDispatchMode):
    """The shape of every tensor an op returns, and of every GEMM's output apart."""

    def __init__(self):
        super().__init__()
        self.shapes: set[tuple[int, ...]] = set()
        self.gemm_shapes: set[tuple[int, ...]] = set()

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        out = func(*args, **(kwargs or {}))
        for t in out if isinstance(out, (tuple, list)) else (out,):
            if isinstance(t, torch.Tensor):
                self.shapes.add(tuple(t.shape))
                if func in GEMMS:
                    self.gemm_shapes.add(tuple(t.shape))
        return out


def test_adapters_train_without_a_vocab_by_hidden_tensor():
    """A frozen base weight and trained adapters: neither direction builds ``W + B·A``, whole or per vocab tile, nor
    a ``[V, H]`` gradient; ``B`` meets ``A`` only through the ``[rows, r]`` factor."""
    hidden, weight, bias, deltas = _operands()
    weight.requires_grad_(False)
    with _OpOutputs() as ops:
        logps, _ = _swept(hidden, weight, bias, deltas, _targets(False))
        logps.sum().backward()
    vocab_tiles = {VOCAB, head_logprobs.VOCAB_TILE, VOCAB % head_logprobs.VOCAB_TILE}
    assert (VOCAB, HIDDEN) not in ops.shapes, "a [V, H] tensor was built"
    merged = {shape for shape in ops.gemm_shapes if shape in {(width, HIDDEN) for width in vocab_tiles}}
    assert not merged, f"a GEMM built a vocab tile of W + B·A: {merged}"
    assert all(d.down.grad is not None and d.up.grad is not None for d in deltas)


def test_frozen_adapters_take_no_gradient():
    """Frozen ``A`` and ``B`` build no gradient buffer; the hidden state still gets the delta's share of its own."""
    hidden, weight, bias, deltas = _operands(adapters_train=False)
    logps, _ = _swept(hidden, weight, bias, deltas, _targets(False))
    logps.sum().backward()
    assert all(d.down.grad is None and d.up.grad is None for d in deltas)
    want_logps, _ = _reference(hidden, weight, bias, deltas, _targets(False))
    (want,) = torch.autograd.grad(want_logps.sum(), deltas[0].inputs)
    torch.testing.assert_close(deltas[0].inputs.grad, want, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize(
    ("field", "shape"),
    [
        ("inputs", (ROWS + 1, HIDDEN)),
        ("down", (RANKS[0], HIDDEN + 1)),
        ("up", (VOCAB + 1, RANKS[0])),
        ("up", (VOCAB, 2)),
    ],
    ids=["extra-input-rows", "down-wider-than-hidden", "extra-vocab-rows", "rank-mismatch"],
)
def test_a_delta_that_does_not_match_the_head_raises(field, shape):
    """The sweep slices ``inputs`` by row and ``up`` by vocab tile, so a larger tensor would lose its extra rows."""
    hidden, weight, bias, deltas = _operands()
    deltas[0] = LowRankDelta(**{**vars(deltas[0]), field: _leaf(*shape)})
    with pytest.raises(ValueError, match=f"low_rank\\[0\\].*{field}"):
        _swept(hidden, weight, bias, deltas, _targets(False))


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
