#!/usr/bin/env python
"""The chunked head's compiled tiles (``halo/kernels/head_logprobs.py``) against the same functions run eagerly, on
inductor's CPU backend. Nothing a run meets mid-way may add a graph."""

import pytest
import torch
import torch._dynamo
from torch._dynamo.utils import counters

from halo.kernels import head_logprobs

HIDDEN = 16
# No tile is one row or one column wide, a size dynamo specializes.
SEQ_TILE, VOCAB_TILE = 16, 64
ROWS, VOCAB = 40, 3 * VOCAB_TILE + 35
DEGENERATE_SHAPES = ((2 * SEQ_TILE + 1, VOCAB), (1, VOCAB), (ROWS, 3 * VOCAB_TILE + 1))

# Gemma 2/3's head is a softcap without bias or logit scale; "every_term" moves the log-probs by whole nats.
HEAD_PATHS = {
    "plain": {"bias": False, "temperature": 1.0, "logit_scale": None, "softcap": None, "entropy": False},
    "gemma_softcap": {"bias": False, "temperature": 0.8, "logit_scale": None, "softcap": 1.5, "entropy": True},
    "every_term": {"bias": True, "temperature": 0.7, "logit_scale": 3.0, "softcap": 2.0, "entropy": True},
}


@pytest.fixture
def small_tiles(monkeypatch):
    monkeypatch.setattr(head_logprobs, "SEQ_TILE", SEQ_TILE)
    monkeypatch.setattr(head_logprobs, "VOCAB_TILE", VOCAB_TILE)
    torch._dynamo.reset()
    counters.clear()
    yield
    torch._dynamo.reset()


def _inputs(rows, vocab, seed):
    generator = torch.Generator().manual_seed(seed)
    hidden = torch.randn(rows, HIDDEN, generator=generator)
    weight = torch.randn(vocab, HIDDEN, generator=generator) * 0.3
    bias = torch.randn(vocab, generator=generator) * 0.2
    targets = torch.randint(0, vocab, (rows,), generator=generator)
    targets[::7] = -100  # A target in no tile scores against log Z alone.
    upstream = torch.randn(rows, generator=generator)
    return hidden, weight, bias, targets, upstream


def _sweep(
    monkeypatch,
    compiled,
    inputs,
    bias,
    temperature,
    logit_scale,
    softcap,
    entropy,
    expand_upstream=False,
    frozen_bias=False,
):
    """``expand_upstream``'s gradient reaches the kernel expanded (stride 0); ``frozen_bias`` takes the bias bf16
    and untrained."""
    monkeypatch.setattr(head_logprobs, "_COMPILED_DEVICE_TYPES", frozenset({"cpu"}) if compiled else frozenset())
    hidden, weight, bias_value, targets, upstream = inputs
    leaves = [hidden.clone().requires_grad_(True), weight.clone().requires_grad_(True)]
    head_bias = None
    if bias and frozen_bias:
        head_bias = bias_value.bfloat16()
    elif bias:
        head_bias = bias_value.clone().requires_grad_(True)
        leaves.append(head_bias)
    logps, ent = head_logprobs.chunked_head_logprobs(
        leaves[0],
        leaves[1],
        targets,
        head_bias,
        temperature,
        logit_scale=logit_scale,
        softcap=softcap,
        compute_entropy=entropy,
    )
    (logps.sum() if expand_upstream else (logps * upstream).sum()).backward()
    return logps.detach(), ent, *(leaf.grad for leaf in leaves)


def _assert_same(got, ref):
    for g, r in zip(got, ref, strict=True):
        if r is None:
            assert g is None
            continue
        torch.testing.assert_close(g, r, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("path", sorted(HEAD_PATHS))
def test_compiled_tiles_match_the_eager_tiles_on_every_head_path(monkeypatch, small_tiles, path):
    inputs = _inputs(ROWS, VOCAB, seed=0)
    compiled = _sweep(monkeypatch, True, inputs, **HEAD_PATHS[path])
    assert counters["stats"]["unique_graphs"] == 2, "one forward and one backward graph serve the whole sweep"
    _assert_same(compiled, _sweep(monkeypatch, False, inputs, **HEAD_PATHS[path]))


def test_one_graph_serves_new_rows_vocabularies_and_scalar_values(monkeypatch, small_tiles):
    # A new value must reach the kernel, not recompile it, and not be served from the first value's graph.
    path = HEAD_PATHS["every_term"]
    _sweep(monkeypatch, True, _inputs(ROWS, VOCAB, seed=1), **path)
    assert counters["stats"]["unique_graphs"] == 2
    with torch._dynamo.config.patch(error_on_recompile=True):
        for seed, (rows, vocab, temperature, logit_scale, softcap) in enumerate(
            ((23, 2 * VOCAB_TILE, 1.3, 2.5, 1.0), (57, 4 * VOCAB_TILE + 9, 0.6, 4.0, 3.0), (9, 50, 1.0, 1.0, 5.0)),
            start=2,
        ):
            inputs = _inputs(rows, vocab, seed)
            head = {**path, "temperature": temperature, "logit_scale": logit_scale, "softcap": softcap}
            _assert_same(_sweep(monkeypatch, True, inputs, **head), _sweep(monkeypatch, False, inputs, **head))
    assert counters["stats"]["unique_graphs"] == 2


def test_nothing_a_run_meets_mid_way_compiles_a_graph(monkeypatch, small_tiles):
    # Each can arrive at a data-dependent step on one rank, where a compile stalls that rank alone.
    path = HEAD_PATHS["every_term"]
    _sweep(monkeypatch, True, _inputs(SEQ_TILE - 6, VOCAB, seed=1), **path)
    assert counters["stats"]["unique_graphs"] == 2
    with torch._dynamo.config.patch(error_on_recompile=True):
        for seed, (rows, vocab) in enumerate(((ROWS, VOCAB), *DEGENERATE_SHAPES), start=2):
            inputs = _inputs(rows, vocab, seed)
            _assert_same(_sweep(monkeypatch, True, inputs, **path), _sweep(monkeypatch, False, inputs, **path))

        inputs = _inputs(ROWS, VOCAB, seed=8)
        # A view of another tensor, as one row's completion ids are.
        batch = torch.zeros(3, ROWS + 9, dtype=torch.long)
        batch[1, -ROWS:] = inputs[3]
        row_view = (*inputs[:3], batch[1:2, -ROWS:], inputs[4])
        _assert_same(_sweep(monkeypatch, True, row_view, **path), _sweep(monkeypatch, False, row_view, **path))

        with torch.autocast("cpu", dtype=torch.bfloat16):
            under_autocast = _sweep(monkeypatch, True, inputs, **path)
        _assert_same(under_autocast, _sweep(monkeypatch, False, inputs, **path))

        ones_upstream = (*inputs[:4], torch.ones(ROWS))
        _assert_same(
            _sweep(monkeypatch, True, inputs, **path, expand_upstream=True),
            _sweep(monkeypatch, False, ones_upstream, **path),
        )

        _assert_same(
            _sweep(monkeypatch, True, inputs, **path, frozen_bias=True),
            _sweep(monkeypatch, False, inputs, **path, frozen_bias=True),
        )
    assert counters["stats"]["unique_graphs"] == 2


def test_candidate_targets_compile_one_graph_pair_of_their_own(monkeypatch, small_tiles):
    # The (rows, m) form is a run's second scoring form: its pair compiles at its first call, and a new candidate
    # count, row count or vocabulary then reuses it.
    path = HEAD_PATHS["every_term"]

    def candidates(rows, vocab, seed, m):
        hidden, weight, bias, _, _ = _inputs(rows, vocab, seed)
        generator = torch.Generator().manual_seed(seed)
        targets = torch.randint(-1, vocab, (rows, m), generator=generator)
        return hidden, weight, bias, targets, torch.randn(rows, m, generator=generator)

    def run(compiled, inputs):
        monkeypatch.setattr(head_logprobs, "_COMPILED_DEVICE_TYPES", frozenset({"cpu"}) if compiled else frozenset())
        hidden, weight, bias, targets, upstream = inputs
        leaves = [t.clone().requires_grad_(True) for t in (hidden, weight, bias)]
        logps, _ = head_logprobs.chunked_head_logprobs(
            *leaves[:2],
            targets,
            leaves[2],
            path["temperature"],
            logit_scale=path["logit_scale"],
            softcap=path["softcap"],
        )
        (torch.where(targets >= 0, logps, 0.0) * upstream).sum().backward()
        return logps.detach(), *(leaf.grad for leaf in leaves)

    _sweep(monkeypatch, True, _inputs(ROWS, VOCAB, seed=1), **path)
    assert counters["stats"]["unique_graphs"] == 2
    first = candidates(ROWS, VOCAB, 2, 4)
    _assert_same(run(True, first), run(False, first))
    assert counters["stats"]["unique_graphs"] == 4
    with torch._dynamo.config.patch(error_on_recompile=True):
        for seed, (rows, vocab, m) in enumerate(((23, 2 * VOCAB_TILE, 9), (SEQ_TILE - 3, VOCAB, 2)), start=3):
            inputs = candidates(rows, vocab, seed, m)
            _assert_same(run(True, inputs), run(False, inputs))
    assert counters["stats"]["unique_graphs"] == 4


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
