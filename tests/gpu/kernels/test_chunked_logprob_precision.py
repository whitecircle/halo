#!/usr/bin/env python
"""The chunked GRPO log-prob sweep keeps fp32 logits and fp32-accumulated gradients on bf16 operands.

Each logits tile is a tensor-core matmul with an fp32 result, so logits of a peaked distribution are not
rounded to bf16 (at |logit| ≈ 20 a bf16 step is 0.125 nats), and the backward's two GEMMs accumulate in
fp32. Checked against an fp64 reference computed from the same bf16 values, over nine vocabulary tiles
(the last one ragged) and two sequence tiles: end to end, and on the backward's fp32 gradients before
their cast to the parameters' bf16, where rounding a tile's product or the running sum to bf16 shows.

Run with 1 GPU:
    torchrun --nproc_per_node=1 tests/gpu/kernels/test_chunked_logprob_precision.py
"""

import torch

from src.trainers.grpo.mixins.chunked_logprobs import (
    _VOCAB_CHUNK,
    _selective_logprob_backward,
    _selective_logprob_entropy_forward,
    chunked_selective_log_softmax,
)
from tests.common.harness import gpu_test_main
from tests.common.utils import fro_rel_err

ROWS, HIDDEN, VOCAB = 5000, 1024, 8 * _VOCAB_CHUNK + 1000
# Logit spread of a trained model's head: most mass on a few tokens, |logit| in the tens.
LOGIT_STD = 6.0
# A bf16-rounded logits tile costs ~1e-1 nats of log-prob at this spread.
LOGP_MAX_ERR = 5e-3
# The bf16 gradient tile each backward GEMM takes costs ~1.6e-3 relative on the fp32 gradients, and a
# tile's product or the running sum rounded to bf16 costs as much again (>= 2.3e-3). The bf16 cast of
# the returned gradients adds ~1.7e-3 in quadrature (~2.3e-3 end to end), where a bf16 running sum over
# the nine vocab tiles reaches ~4e-3.
GRAD_FP32_REL_FRO = 2e-3
GRAD_REL_FRO = 3e-3


def _inputs(device):
    generator = torch.Generator(device=device).manual_seed(0)
    hidden = torch.randn(ROWS, HIDDEN, generator=generator, device=device)
    weight = torch.randn(VOCAB, HIDDEN, generator=generator, device=device) * (LOGIT_STD / HIDDEN**0.5)
    targets = (hidden @ weight.t()).argmax(dim=-1)
    grad = torch.randn(ROWS, generator=generator, device=device)
    return hidden.bfloat16(), weight.bfloat16(), targets, grad


def _reference(hidden, weight, targets, grad):
    h = hidden.double().requires_grad_(True)
    w = weight.double().requires_grad_(True)
    logps = torch.log_softmax(h @ w.t(), dim=-1).gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    (logps * grad.double()).sum().backward()
    return logps.detach(), h.grad, w.grad


def run(ctx) -> dict:
    hidden, weight, targets, grad = _inputs(ctx.device)
    ref_logps, ref_dh, ref_dw = _reference(hidden, weight, targets, grad)

    h = hidden.clone().requires_grad_(True)
    w = weight.clone().requires_grad_(True)
    logps = chunked_selective_log_softmax(h.unsqueeze(0), w, targets.unsqueeze(0), None, 1.0).squeeze(0)
    (logps * grad).sum().backward()

    _, log_z, _ = _selective_logprob_entropy_forward(hidden, weight, targets, None, 1.0, None, None, False)
    dh_fp32, dw_fp32, _ = _selective_logprob_backward(hidden, weight, targets, None, log_z, grad, 1.0, None, None)

    logp_err = (logps.double() - ref_logps).abs().max().item()
    dh_err, dw_err = fro_rel_err(h.grad, ref_dh), fro_rel_err(w.grad, ref_dw)
    dh_fp32_err, dw_fp32_err = fro_rel_err(dh_fp32, ref_dh), fro_rel_err(dw_fp32, ref_dw)
    return {
        "checks": {
            "logprob_max_error_vs_fp64": logp_err < LOGP_MAX_ERR,
            "hidden_grad_rel_fro_vs_fp64": dh_err < GRAD_REL_FRO,
            "weight_grad_rel_fro_vs_fp64": dw_err < GRAD_REL_FRO,
            "fp32_hidden_grad_rel_fro_vs_fp64": dh_fp32_err < GRAD_FP32_REL_FRO,
            "fp32_weight_grad_rel_fro_vs_fp64": dw_fp32_err < GRAD_FP32_REL_FRO,
        },
        "metrics": {
            "logprob_max_error": logp_err,
            "hidden_grad_rel_fro": dh_err,
            "weight_grad_rel_fro": dw_err,
            "fp32_hidden_grad_rel_fro": dh_fp32_err,
            "fp32_weight_grad_rel_fro": dw_fp32_err,
        },
    }


main = gpu_test_main(exact_world_size=1, prefix="chunked_logprob_precision", partial_state=False)(run)

if __name__ == "__main__":
    main()
