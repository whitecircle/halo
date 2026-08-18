"""CPU tests for the chunked GRPO log-prob / entropy helpers (src/trainers/grpo/chunked_logprobs.py).

The chunked path must compute the SAME per-token log-probs and entropy as the full-logits path TRL
uses (``selective_log_softmax`` / ``entropy_from_logits`` over ``hidden @ lm_head.weight.T / temp``),
without materializing the ``[B, T, vocab]`` logits. These tests assert that equivalence to tight
tolerance and FAIL if the chunked math drifts from the reference.

    python tests/cpu/grpo/test_chunked_grpo_logprobs.py
"""

import sys

import pytest
import torch
from trl.trainer.utils import entropy_from_logits, selective_log_softmax

from src.distributed.runtime import materialize_dtensor
from src.trainers.grpo.mixins.chunked_logprobs import (
    _VOCAB_CHUNK,
    ChunkedGRPOLogprobsMixin,
    chunked_selective_log_softmax,
    chunked_selective_log_softmax_with_entropy,
)

# Vocab spans multiple chunk tiles so the online-softmax accumulation across chunks is exercised.
VOCAB = _VOCAB_CHUNK * 2 + 137
HIDDEN = 32
B, T = 2, 16


def _ref_logits(hidden, weight, bias, temperature):
    logits = torch.matmul(hidden, weight.t())
    if bias is not None:
        logits = logits + bias
    return logits / temperature


def test_chunked_logps_match_full_path():
    torch.manual_seed(0)
    hidden = torch.randn(B, T, HIDDEN, dtype=torch.float32)
    weight = torch.randn(VOCAB, HIDDEN, dtype=torch.float32) * 0.05
    ids = torch.randint(0, VOCAB, (B, T))
    for temperature in (1.0, 0.7):
        ref = selective_log_softmax(_ref_logits(hidden, weight, None, temperature), ids)
        got = chunked_selective_log_softmax(hidden, weight, ids, None, temperature)
        assert got.shape == (B, T)
        torch.testing.assert_close(got, ref, atol=1e-4, rtol=1e-4)


def test_chunked_logps_match_with_bias():
    torch.manual_seed(1)
    hidden = torch.randn(B, T, HIDDEN, dtype=torch.float32)
    weight = torch.randn(VOCAB, HIDDEN, dtype=torch.float32) * 0.05
    bias = torch.randn(VOCAB, dtype=torch.float32) * 0.1
    ids = torch.randint(0, VOCAB, (B, T))
    ref = selective_log_softmax(_ref_logits(hidden, weight, bias, 1.0), ids)
    got = chunked_selective_log_softmax(hidden, weight, ids, bias, 1.0)
    torch.testing.assert_close(got, ref, atol=1e-4, rtol=1e-4)


def test_chunked_logps_backward_matches():
    torch.manual_seed(2)
    weight = torch.randn(VOCAB, HIDDEN, dtype=torch.float32) * 0.05
    ids = torch.randint(0, VOCAB, (B, T))

    h_ref = torch.randn(B, T, HIDDEN, dtype=torch.float32, requires_grad=True)
    selective_log_softmax(_ref_logits(h_ref, weight, None, 1.0), ids).sum().backward()

    h_chunk = h_ref.detach().clone().requires_grad_(True)
    chunked_selective_log_softmax(h_chunk, weight, ids, None, 1.0).sum().backward()

    torch.testing.assert_close(h_chunk.grad, h_ref.grad, atol=1e-4, rtol=1e-4)


def test_fused_entropy_matches_full_path():
    # Both outputs are checked: a fused sweep can corrupt the diagnostic or the loss numerator alone.
    torch.manual_seed(3)
    hidden = torch.randn(B, T, HIDDEN, dtype=torch.float32)
    weight = torch.randn(VOCAB, HIDDEN, dtype=torch.float32) * 0.05
    bias = torch.randn(VOCAB, dtype=torch.float32) * 0.1
    ids = torch.randint(0, VOCAB, (B, T))
    for temperature in (1.0, 0.7):
        for b in (None, bias):
            ref_entropy = entropy_from_logits(_ref_logits(hidden, weight, b, temperature))
            ref_logps = selective_log_softmax(_ref_logits(hidden, weight, b, temperature), ids)
            logps, entropy = chunked_selective_log_softmax_with_entropy(hidden, weight, ids, b, temperature)
            assert entropy.shape == (B, T)
            assert not entropy.requires_grad  # detached diagnostic
            torch.testing.assert_close(entropy, ref_entropy, atol=1e-4, rtol=1e-4)
            torch.testing.assert_close(logps, ref_logps, atol=1e-4, rtol=1e-4)


def test_fused_backward_matches_plain_chunked():
    # Both delegate backward to Liger's recompute path over the same saved tensors, so atol/rtol are 0.
    torch.manual_seed(4)
    weight = torch.randn(VOCAB, HIDDEN, dtype=torch.float32) * 0.05
    ids = torch.randint(0, VOCAB, (B, T))

    h_plain = torch.randn(B, T, HIDDEN, dtype=torch.float32, requires_grad=True)
    chunked_selective_log_softmax(h_plain, weight, ids, None, 1.0).sum().backward()

    h_fused = h_plain.detach().clone().requires_grad_(True)
    logps, _entropy = chunked_selective_log_softmax_with_entropy(h_fused, weight, ids, None, 1.0)
    logps.sum().backward()

    torch.testing.assert_close(h_fused.grad, h_plain.grad, atol=0.0, rtol=0.0)


def test_materialize_passthrough_for_plain_tensors():
    t = torch.randn(4, 4)
    assert materialize_dtensor(t) is t
    assert materialize_dtensor(None) is None


class _DenseHarness(ChunkedGRPOLogprobsMixin):
    """Minimal harness exposing ``_dense_last_hidden_state`` with a backbone whose hidden state is a
    differentiable function of a single leaf param — so we can assert each row's backward reaches it.

    ``_get_last_hidden_state`` faithfully emulates TRL's: forward → drop the final position (the causal
    next-token shift) → keep the last ``logits_to_keep``. A single-token span therefore yields a
    0-length hidden, exactly the shape a naive empty-row connection breaks on."""

    def __init__(self, hidden_size: int):
        self.param = torch.zeros(hidden_size, requires_grad=True)

    def _get_last_hidden_state(self, model, input_ids, attention_mask, logits_to_keep):
        h = self.param.numel()
        seq_len = input_ids.size(1)
        full = (self.param + 1.0).reshape(1, 1, h).expand(1, seq_len, h)  # (1, L, H), connected to param
        full = full[:, :-1, :]  # drop final position (causal shift) — 1-token span → 0-length
        return full[:, -logits_to_keep:, :]


def _dense_batch():
    # Row 1 is a dummy padding row: n_comp == 0, so its 1-token span collapses to a 0-length hidden.
    input_ids = torch.tensor([[5, 6, 7], [5, 0, 0]])
    attention_mask = torch.tensor([[1, 1, 1], [1, 0, 0]])
    return input_ids, attention_mask, 2  # logits_to_keep


def test_dense_empty_row_stays_connected_to_graph():
    # A disconnected zeros row gets its backward pruned, so the FSDP/EP grad collectives vanish only on
    # the rank that padded more dummy rows — a NCCL deadlock. The shape assert covers the widened span.
    harness = _DenseHarness(hidden_size=8)
    input_ids, attention_mask, ltk = _dense_batch()

    out = harness._dense_last_hidden_state(None, input_ids, attention_mask, ltk)
    assert out.shape == (2, ltk, 8)

    harness.param.grad = None
    out[1].sum().backward()  # the dummy row alone must still reach the param
    assert harness.param.grad is not None and harness.param.grad.abs().sum() > 0, (
        "empty (n_comp==0) row is disconnected from the model graph — its backward would be pruned, "
        "desyncing FSDP/EP collectives across ranks"
    )


def test_dense_real_row_values_unchanged_by_fix():
    # Keeping the empty row connected must not perturb a real row's values.
    harness = _DenseHarness(hidden_size=8)
    input_ids, attention_mask, ltk = _dense_batch()
    out = harness._dense_last_hidden_state(None, input_ids, attention_mask, ltk)

    expected_real = harness.param.detach() + 1.0
    torch.testing.assert_close(out[0, 0], expected_real)
    torch.testing.assert_close(out[0, 1], expected_real)
    # Row 1 connects position 0 only; its value is irrelevant (masked downstream), position 1 stays zero.
    assert torch.count_nonzero(out[1, 1]) == 0


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
