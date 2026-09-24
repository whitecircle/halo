#!/usr/bin/env python
"""The chunked GRPO log-prob sweep keeps fp32 logits and fp32-accumulated gradients on bf16 operands.

Each logits tile is a tensor-core matmul with an fp32 result, so logits of a peaked distribution are not
rounded to bf16 (at |logit| ≈ 20 a bf16 step is 0.125 nats), and the backward's two GEMMs accumulate in
fp32. Checked against an fp64 reference computed from the same bf16 values, at a vocabulary wider than
one tile and more rows than one sequence tile.
"""

import pytest
import torch

from src.trainers.grpo.mixins.chunked_logprobs import chunked_selective_log_softmax

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

ROWS, HIDDEN, VOCAB = 5000, 1024, 40_000
# Logit spread of a trained model's head: most mass on a few tokens, |logit| in the tens.
LOGIT_STD = 6.0


def _inputs():
    generator = torch.Generator(device="cuda").manual_seed(0)
    hidden = torch.randn(ROWS, HIDDEN, generator=generator, device="cuda")
    weight = torch.randn(VOCAB, HIDDEN, generator=generator, device="cuda") * (LOGIT_STD / HIDDEN**0.5)
    targets = (hidden @ weight.t()).argmax(dim=-1)
    grad = torch.randn(ROWS, generator=generator, device="cuda")
    return hidden.bfloat16(), weight.bfloat16(), targets, grad


def _reference(hidden, weight, targets, grad):
    h = hidden.double().requires_grad_(True)
    w = weight.double().requires_grad_(True)
    logps = torch.log_softmax(h @ w.t(), dim=-1).gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    (logps * grad.double()).sum().backward()
    return logps.detach(), h.grad, w.grad


def _rel_fro(a, b):
    return ((a.double() - b).norm() / b.norm()).item()


def test_chunked_logprobs_and_grads_track_fp64():
    hidden, weight, targets, grad = _inputs()
    ref_logps, ref_dh, ref_dw = _reference(hidden, weight, targets, grad)

    h = hidden.clone().requires_grad_(True)
    w = weight.clone().requires_grad_(True)
    logps = chunked_selective_log_softmax(h.unsqueeze(0), w, targets.unsqueeze(0), None, 1.0).squeeze(0)
    (logps * grad).sum().backward()

    logp_err = (logps.double() - ref_logps).abs().max().item()
    dh_err, dw_err = _rel_fro(h.grad, ref_dh), _rel_fro(w.grad, ref_dw)
    # bf16 storage of the gradients alone costs ~1.7e-3 relative (the floor); a bf16-rounded logits tile
    # costs several times that and ~1e-1 nats of log-prob at this spread.
    assert logp_err < 5e-3, f"log-prob max |Δ| vs fp64 {logp_err:.3e}"
    assert dh_err < 4e-3 and dw_err < 4e-3, f"grad rel-fro vs fp64: dh {dh_err:.3e} dw {dw_err:.3e}"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
