#!/usr/bin/env python
"""CPU test: the pipeline CE — label-roll shift and chunked-checkpoint equivalence.

``causal_lm_token_loss`` rolls labels instead of slicing logits and chunks the fp32 CE under a
non-reentrant checkpoint. Both must be exact: values AND gradients must match the naive
sliced/monolithic formulation, including across the chunk boundary.

    python tests/cpu/parallelism/test_pp_losses.py
"""

import collections

import pytest
import torch
import torch.nn.functional as F

import src.kernels.logprobs as logprob_kernels
from src.distributed.pipeline_parallel.stage_losses import causal_lm_token_loss, fused_causal_lm_token_loss
from src.kernels.logprobs import token_logprobs
from src.models.head_transform import IDENTITY_HEAD_TRANSFORM
from tests.common.utils import autograd_nodes


def _naive_reference(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    return F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)).float(),
        shift_labels.view(-1),
        ignore_index=-100,
        reduction="sum",
    )


def _make(batch: int, seq: int, vocab: int = 37, seed: int = 3):
    g = torch.Generator().manual_seed(seed)
    logits = torch.randn(batch, seq, vocab, generator=g, requires_grad=True)
    labels = torch.randint(0, vocab, (batch, seq), generator=g)
    labels[:, :3] = -100  # ignored prefix
    return logits, labels


@pytest.mark.parametrize("batch,seq", [(2, 16), (3, 33)])
def test_single_chunk_matches_naive(batch, seq):
    logits, labels = _make(batch, seq)
    loss = causal_lm_token_loss(logits, labels)
    ref = _naive_reference(logits.detach().clone().requires_grad_(True), labels)
    assert torch.allclose(loss, ref, atol=1e-5)


def test_multi_chunk_values_and_grads_match(monkeypatch):
    # Several chunks with an uneven tail: the path a wrong chunk boundary breaks.
    monkeypatch.setattr(logprob_kernels, "LOGIT_CHUNK_ELEMENTS", 50 * 37)  # 50 rows at vocab 37
    logits, labels = _make(4, 40)  # 160 tokens -> 4 chunks (50/50/50/10)
    loss = causal_lm_token_loss(logits, labels)
    loss.backward()

    ref_logits = logits.detach().clone().requires_grad_(True)
    ref = _naive_reference(ref_logits, labels)
    ref.backward()

    assert torch.allclose(loss, ref, atol=1e-5)
    assert torch.allclose(logits.grad, ref_logits.grad, atol=1e-6), (
        f"max grad diff {(logits.grad - ref_logits.grad).abs().max()}"
    )


def test_all_ignored_chunk_is_inert(monkeypatch):
    monkeypatch.setattr(logprob_kernels, "LOGIT_CHUNK_ELEMENTS", 8 * 37)  # 8 rows at vocab 37
    logits, labels = _make(1, 32)
    labels[:] = -100
    loss = causal_lm_token_loss(logits, labels)
    assert float(loss) == 0.0
    loss.backward()
    assert torch.count_nonzero(logits.grad) == 0


_VOCAB, _HIDDEN = 37, 11


def _fused_loss(hidden, labels):
    head = torch.nn.Linear(_HIDDEN, _VOCAB, bias=False, dtype=torch.bfloat16).requires_grad_(False)
    return fused_causal_lm_token_loss(head, IDENTITY_HEAD_TRANSFORM, hidden, labels)


# Every loss that chunks through ``checkpointed_chunks``, with its input's feature width and the inputs
# it splits: the CE sums split the flattened plane once, the per-token log-probs each row. Each chunk
# budget is counted in vocab columns.
_ROWS = 3
_CHUNKED_LOSSES = {
    "causal_lm_token_loss": (causal_lm_token_loss, _VOCAB, 1),
    "fused_causal_lm_token_loss": (_fused_loss, _HIDDEN, 1),
    "token_logprobs": (lambda rows, labels: torch.mul(*token_logprobs(rows, labels)).sum(), _VOCAB, _ROWS),
}


def _chunked_run(name: str, chunk_rows: int, monkeypatch):
    """Forward + backward of one chunked loss at ``chunk_rows``; returns the value, the input
    gradient and the autograd graph's node-type counts."""
    loss_fn, features, _ = _CHUNKED_LOSSES[name]
    monkeypatch.setattr(logprob_kernels, "LOGIT_CHUNK_ELEMENTS", chunk_rows * _VOCAB)
    torch.manual_seed(7)
    rows = torch.randn(_ROWS, 20, features, dtype=torch.bfloat16).requires_grad_(True)
    labels = torch.randint(0, _VOCAB, (_ROWS, 20))
    labels[:, :4] = -100
    value = loss_fn(rows, labels)
    nodes = collections.Counter(type(node).__name__ for node in autograd_nodes(value))
    value.backward()
    return value.detach(), rows.grad, nodes


@pytest.mark.parametrize("name", list(_CHUNKED_LOSSES))
def test_chunked_losses_do_not_depend_on_the_chunk_count(name, monkeypatch):
    """60 tokens in one chunk vs in seven uneven ones (9 rows each, a 6-row tail) for the CE sums, and
    each row's positions in one chunk vs three for the per-token log-probs: every per-token quantity
    is computed per row, so the input gradient is bitwise identical, and only the scalar reductions
    over chunks reassociate (fp32 summation order)."""
    whole_value, whole_grad, _ = _chunked_run(name, 64, monkeypatch)
    chunked_value, chunked_grad, _ = _chunked_run(name, 9, monkeypatch)

    torch.testing.assert_close(chunked_value, whole_value, rtol=1e-6, atol=0)
    if name == "fused_causal_lm_token_loss":
        # The hidden-state gradient is a per-chunk GEMM, whose blocking may follow the chunk's rows.
        torch.testing.assert_close(chunked_grad, whole_grad, rtol=torch.finfo(torch.bfloat16).eps, atol=0)
    else:
        assert torch.equal(chunked_grad, whole_grad), f"max grad diff {(chunked_grad - whole_grad).abs().max()}"


@pytest.mark.parametrize("name", list(_CHUNKED_LOSSES))
def test_the_chunks_share_one_split_backward(name, monkeypatch):
    """Each chunk is a view from ONE ``split`` of its input (the flattened plane, or one row of it),
    whose backward concatenates the chunk gradients once. A per-chunk slice instead runs one
    SliceBackward per chunk, each allocating a zero-filled plane the size of the whole input — the
    backward grows with the chunk count in time and in memory."""
    _, _, whole_nodes = _chunked_run(name, 64, monkeypatch)
    _, _, chunked_nodes = _chunked_run(name, 9, monkeypatch)

    split_nodes = sum(count for node, count in chunked_nodes.items() if node.startswith("Split"))
    splits = _CHUNKED_LOSSES[name][2]
    assert split_nodes == splits, f"expected {splits} split(s) over the chunked input, got {dict(chunked_nodes)}"
    assert chunked_nodes["SliceBackward0"] == whole_nodes["SliceBackward0"], (
        f"chunking added {chunked_nodes['SliceBackward0'] - whole_nodes['SliceBackward0']} slice backwards"
    )


def test_fp64_logits_keep_fp64_log_probs_and_cross_entropy():
    """The upcast is to fp32 at least: an fp64 oracle (the CPU parity tests) must not be rounded to
    fp32 inside the loss it checks, and a bf16 production plane pays exactly the fp32 it always did."""
    torch.manual_seed(5)
    logits = torch.randn(2, 9, _VOCAB, dtype=torch.float64)
    labels = torch.randint(0, _VOCAB, (2, 9))
    logps, _ = token_logprobs(logits, labels)
    reference = torch.log_softmax(logits[:, :-1], dim=-1).gather(-1, labels[:, 1:, None]).squeeze(-1)

    assert logps.dtype is torch.float64
    torch.testing.assert_close(logps, reference, rtol=1e-12, atol=0)
    assert token_logprobs(logits.bfloat16(), labels)[0].dtype is torch.float32

    ce = causal_lm_token_loss(logits, labels)
    assert ce.dtype is torch.float64
    torch.testing.assert_close(ce, _naive_reference(logits, labels).to(torch.float64), rtol=1e-6, atol=0)
    torch.testing.assert_close(ce, -reference.sum(), rtol=1e-12, atol=0)
    assert causal_lm_token_loss(logits.bfloat16(), labels).dtype is torch.float32


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
