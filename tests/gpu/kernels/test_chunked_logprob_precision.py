#!/usr/bin/env python
"""The head log-prob kernel keeps fp32 logits and fp32-accumulated gradients on bf16 operands, against an fp64
reference.

At |logit| ≈ 20 a bf16 step is 0.125 nats. Compiled tiles are held to eager ones on every head term, and sweeps at new
sizes and scalar values must reuse the compiled graphs, so no scalar is frozen in.
"""

import torch
import torch._dynamo
from torch._dynamo.exc import RecompileError
from torch._dynamo.utils import counters

from halo.kernels import head_logprobs
from halo.kernels.head_logprobs import (
    SEQ_TILE,
    VOCAB_TILE,
    _logprobs_backward,
    _logprobs_forward,
    chunked_head_logprobs,
)
from tests.common.harness import gpu_test_main
from tests.common.numerics import fro_rel_err

ROWS, HIDDEN, VOCAB = 5000, 1024, 8 * VOCAB_TILE + 1000
# A trained head's spread: most mass on a few tokens, |logit| in the tens.
LOGIT_STD = 6.0
# A bf16-rounded logits tile costs ~1e-1 nats of log-prob at this spread.
LOGP_MAX_ERR = 5e-3
# The bf16 gradient tile each backward GEMM takes costs ~1.6e-3 relative, and a tile's product or running sum rounded
# to bf16 costs as much again (>= 2.3e-3). The bf16 cast of the returned gradients adds ~1.7e-3 in quadrature, where a
# bf16 running sum over nine tiles reaches ~4e-3.
GRAD_FP32_REL_FRO = 2e-3
GRAD_REL_FRO = 3e-3
# fp32 reassociation moves log Z (up to ~40, ulp 3.8e-6) by a few ulps, and a gradient element on a bf16 boundary flips
# by one bf16 ulp; two orders below the fp64 bounds, while a frozen scalar moves log-probs by nats. The entropy, log Z
# less a mean logit of about its size, carries that error into a value of order 1.
LOGP_COMPILED_VS_EAGER = 3e-5
ENTROPY_COMPILED_VS_EAGER = 1e-4
GRAD_COMPILED_VS_EAGER_REL_FRO = 2e-4
# Within ~1e-6 of fp64 at this spread; after the tile's bf16 cast off by ~2.3e-3 (measured on a CPU emulation).
BIAS_GRAD_FP32_REL_FRO = 1e-4
BIAS_STD = 1.0
# Gemma 2/3's head cap, a temperature and the fused entropy; every term at once; then other values, which must reach
# the compiled graph rather than recompile it or be frozen out of it.
SOFTCAP_HEAD = {"temperature": 0.8, "softcap": 30.0, "compute_entropy": True}
EVERY_TERM_SCALARS = {"temperature": 0.8, "logit_scale": 0.5, "softcap": 30.0, "compute_entropy": True}
NEW_SCALARS = {"temperature": 1.1, "logit_scale": 0.7, "softcap": 20.0, "compute_entropy": True}
# Far below a sequence tile, as a short batch's sweep is.
SMALL_ROWS = (2, 37, 300)


def _inputs(device):
    generator = torch.Generator(device=device).manual_seed(0)
    hidden = torch.randn(ROWS, HIDDEN, generator=generator, device=device)
    weight = torch.randn(VOCAB, HIDDEN, generator=generator, device=device) * (LOGIT_STD / HIDDEN**0.5)
    targets = (hidden @ weight.t()).argmax(dim=-1)
    grad = torch.randn(ROWS, generator=generator, device=device)
    bias = torch.randn(VOCAB, generator=generator, device=device) * BIAS_STD
    return hidden.bfloat16(), weight.bfloat16(), targets, grad, bias.bfloat16()


def _reference(hidden, weight, targets, grad, bias=None):
    leaves = [hidden.double().requires_grad_(True), weight.double().requires_grad_(True)]
    logits = leaves[0] @ leaves[1].t()
    if bias is not None:
        leaves.append(bias.double().requires_grad_(True))
        logits = logits + leaves[2]
    logps = torch.log_softmax(logits, dim=-1).gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    (logps * grad.double()).sum().backward()
    return logps.detach(), leaves[0].grad, leaves[1].grad, leaves[2].grad if bias is not None else None


def _fp32_sweep(
    hidden, weight, targets, grad, bias=None, temperature=1.0, logit_scale=None, softcap=None, compute_entropy=False
):
    logps, log_z, entropy = _logprobs_forward(
        hidden, weight, targets, bias, temperature, logit_scale, softcap, compute_entropy
    )
    needs_grad = (True, True, bias is not None)
    dh, dw, db = _logprobs_backward(
        hidden, weight, targets, bias, log_z, grad, temperature, logit_scale, softcap, needs_grad
    )
    return logps, entropy, dh, dw, db


def _eager_fp32_sweep(*args, **head):
    compiled_on = head_logprobs._COMPILED_DEVICE_TYPES
    head_logprobs._COMPILED_DEVICE_TYPES = frozenset()
    try:
        return _fp32_sweep(*args, **head)
    finally:
        head_logprobs._COMPILED_DEVICE_TYPES = compiled_on


def _hold_to_eager(name, compiled, eager, checks, metrics) -> None:
    logp_diff, entropy_diff = ((compiled[i] - eager[i]).abs().max().item() for i in (0, 1))
    grad_diff = max(fro_rel_err(compiled[i], eager[i]) for i in (2, 3, 4) if eager[i] is not None)
    metrics[f"{name}_compiled_vs_eager_logprob_max"] = logp_diff
    metrics[f"{name}_compiled_vs_eager_entropy_max"] = entropy_diff
    metrics[f"{name}_compiled_vs_eager_grad_rel_fro"] = grad_diff
    checks[f"{name}_compiled_logprobs_match_eager"] = logp_diff < LOGP_COMPILED_VS_EAGER
    checks[f"{name}_compiled_entropy_matches_eager"] = entropy_diff < ENTROPY_COMPILED_VS_EAGER
    checks[f"{name}_compiled_grads_match_eager"] = grad_diff < GRAD_COMPILED_VS_EAGER_REL_FRO


def _reuse_and_hold(name, inputs, head, checks, metrics) -> None:
    """A recompile fails the case; then hold it to eager."""
    try:
        with torch._dynamo.config.patch(error_on_recompile=True):
            compiled = _fp32_sweep(*inputs, **head)
    except RecompileError:
        checks[f"{name}_reuse_the_graphs"] = False
        return
    checks[f"{name}_reuse_the_graphs"] = True
    _hold_to_eager(name, compiled, _eager_fp32_sweep(*inputs, **head), checks, metrics)


def run(ctx) -> dict:
    hidden, weight, targets, grad, bias = _inputs(ctx.device)
    ref_logps, ref_dh, ref_dw, _ = _reference(hidden, weight, targets, grad)

    h = hidden.clone().requires_grad_(True)
    w = weight.clone().requires_grad_(True)
    logps, _ = chunked_head_logprobs(h, w, targets, None, 1.0)
    (logps * grad).sum().backward()
    _, _, dh_fp32, dw_fp32, _ = _fp32_sweep(hidden, weight, targets, grad)

    logp_err = (logps.double() - ref_logps).abs().max().item()
    dh_err, dw_err = fro_rel_err(h.grad, ref_dh), fro_rel_err(w.grad, ref_dw)
    dh_fp32_err, dw_fp32_err = fro_rel_err(dh_fp32, ref_dh), fro_rel_err(dw_fp32, ref_dw)
    metrics = {
        "logprob_max_error": logp_err,
        "hidden_grad_rel_fro": dh_err,
        "weight_grad_rel_fro": dw_err,
        "fp32_hidden_grad_rel_fro": dh_fp32_err,
        "fp32_weight_grad_rel_fro": dw_fp32_err,
    }
    checks = {
        "logprob_max_error_vs_fp64": logp_err < LOGP_MAX_ERR,
        "hidden_grad_rel_fro_vs_fp64": dh_err < GRAD_REL_FRO,
        "weight_grad_rel_fro_vs_fp64": dw_err < GRAD_REL_FRO,
        "fp32_hidden_grad_rel_fro_vs_fp64": dh_fp32_err < GRAD_FP32_REL_FRO,
        "fp32_weight_grad_rel_fro_vs_fp64": dw_fp32_err < GRAD_FP32_REL_FRO,
        "tiles_ran_compiled": counters["stats"]["unique_graphs"] >= 2,
    }

    for name, head in (
        ("plain", {}),
        ("softcap", SOFTCAP_HEAD),
        ("every_term", {"bias": bias, **EVERY_TERM_SCALARS}),
    ):
        compiled = _fp32_sweep(hidden, weight, targets, grad, **head)
        _hold_to_eager(name, compiled, _eager_fp32_sweep(hidden, weight, targets, grad, **head), checks, metrics)

    # The eager tiles share the bias gradient's column-sum function, so only fp64 can see the sum after the bf16 cast.
    biased = _fp32_sweep(hidden, weight, targets, grad, bias=bias)
    ref_biased_logps, _, _, ref_db = _reference(hidden, weight, targets, grad, bias)
    metrics["bias_logprob_max_error"] = (biased[0].double() - ref_biased_logps).abs().max().item()
    metrics["fp32_bias_grad_rel_fro"] = fro_rel_err(biased[4], ref_db)
    checks["bias_logprob_max_error_vs_fp64"] = metrics["bias_logprob_max_error"] < LOGP_MAX_ERR
    checks["fp32_bias_grad_rel_fro_vs_fp64"] = metrics["fp32_bias_grad_rel_fro"] < BIAS_GRAD_FP32_REL_FRO

    # New sizes, small row counts, tiles one row or column wide (which run eagerly), and new scalars on the every-term
    # path.
    reuse_cases = {
        "new_rows_and_vocab": (3000, 5 * VOCAB_TILE + 77, {}),
        **{f"{rows}_rows": (rows, VOCAB, {}) for rows in SMALL_ROWS},
        "one_row_and_one_column_tiles": (SEQ_TILE + 1, 2 * VOCAB_TILE + 1, {}),
        "new_scalars": (ROWS, VOCAB, {"bias": bias, **NEW_SCALARS}),
    }
    for name, (rows, vocab, head) in reuse_cases.items():
        inputs = (hidden[:rows], weight[:vocab], targets[:rows] % vocab, grad[:rows])
        _reuse_and_hold(name, inputs, head, checks, metrics)

    # A run's first sweep can be a short batch; a kernel specialized to the first sizes inductor saw would fail full
    # tiles.
    torch._dynamo.reset()
    short = SMALL_ROWS[1]
    _fp32_sweep(hidden[:short], weight, targets[:short], grad[:short])
    _reuse_and_hold("full_tiles_after_a_short_first_sweep", (hidden, weight, targets, grad), {}, checks, metrics)
    return {"checks": checks, "metrics": metrics}


main = gpu_test_main(exact_world_size=1, prefix="chunked_logprob_precision", partial_state=False)(run)

if __name__ == "__main__":
    main()
