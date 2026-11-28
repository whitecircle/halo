#!/usr/bin/env python
"""pytorch/pytorch#196678 on the GPU at hand. After a model load's guard, a right-padded causal SDPA call whose every
logsumexp sits below -ln(FLT_MAX) gives finite gradients within bf16 rounding of an fp64 reference, at key lengths 64
mod 128 and beside them. Forced cuDNN on an affected Hopper still returns NaN in dQ at those lengths, so the check reaches the
bug, and Blackwell keeps cuDNN SDPA on."""

import contextlib
import math

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
from transformers import LlamaConfig

from halo.model_loading.model_preparation import apply_family_patches
from halo.models.attention.backend_setup import CUDNN_SDPA_HOPPER_FIXED_VERSION
from tests.common.harness import gpu_test_main, record_check
from tests.common.rank_log import log

# 704 and 1472 are 64 mod 128, the bug's lengths; 640 and 768 are not.
_LENGTHS = (640, 704, 768, 1472)
_BUGGY_LENGTHS = (704, 1472)
_BATCH, _HEADS, _HEAD_DIM = 3, 4, 128
# A bf16 kernel against the fp64 math reference, as a fraction of the reference's largest entry.
_RELATIVE_TOLERANCE = 2e-2


def _affected() -> bool:
    return (
        tuple(torch.cuda.get_device_capability()) == (9, 0)
        and torch.backends.cudnn.version() < CUDNN_SDPA_HOPPER_FIXED_VERSION
    )


def _inputs(length: int):
    """The issue's construction under a right-padded causal mask of three rows: every logit about -120, so every query
    row's logsumexp is below -ln(FLT_MAX), and each padding query still reaches its row's real keys."""
    generator = torch.Generator(device="cuda").manual_seed(length)
    shape = (_BATCH, _HEADS, length, _HEAD_DIM)

    def draw(scale: float) -> torch.Tensor:
        return torch.randn(shape, generator=generator, device="cuda", dtype=torch.bfloat16) * scale

    query, key, value, grad = draw(0.1), draw(0.1), draw(1.0), draw(1.0)
    query[..., 0], key[..., 0] = 1.0, -120 * math.sqrt(_HEAD_DIM)
    lengths = torch.tensor([length, length - 100, length // 2], device="cuda")
    positions = torch.arange(length, device="cuda")
    causal = positions[:, None] >= positions[None, :]
    real_keys = positions[None, :] < lengths[:, None]
    mask = (causal[None] & real_keys[:, None, :])[:, None]
    return query, key, value, mask, grad


def _grads(query, key, value, mask, grad, backend: SDPBackend | None = None) -> tuple[torch.Tensor, ...]:
    leaves = [tensor.detach().clone().requires_grad_() for tensor in (query, key, value)]
    scope = sdpa_kernel([backend]) if backend is not None else contextlib.nullcontext()
    with scope:
        F.scaled_dot_product_attention(*leaves, attn_mask=mask).backward(grad)
    return tuple(leaf.grad for leaf in leaves)


def _load_guard() -> None:
    config = LlamaConfig(
        vocab_size=64, hidden_size=_HEADS * _HEAD_DIM, intermediate_size=64, num_hidden_layers=1, num_attention_heads=4
    )
    apply_family_patches(config, "sdpa")


def check_the_load_turns_cudnn_sdpa_off_on_an_affected_hopper_alone() -> None:
    _load_guard()
    cudnn_on = torch.backends.cuda.cudnn_sdp_enabled()
    log(
        f"  capability {torch.cuda.get_device_capability()}, cuDNN {torch.backends.cudnn.version()}: cuDNN SDPA "
        f"{'on' if cudnn_on else 'off'}"
    )
    assert cudnn_on is not _affected()
    assert torch.backends.cuda.flash_sdp_enabled() and torch.backends.cuda.mem_efficient_sdp_enabled()


def check_default_dispatch_is_finite_and_matches_an_fp64_reference() -> None:
    """dQ is compared outside dimension 0. Every key holds the same value there, so its true dQ is zero (each softmax
    row's gradient sums to zero), and any bf16 kernel returns rounding noise in it larger than every real entry."""
    _load_guard()
    for length in _LENGTHS:
        inputs = _inputs(length)
        query, key, value, mask, grad = inputs
        default = _grads(*inputs)
        reference = _grads(query.double(), key.double(), value.double(), mask, grad.double(), SDPBackend.MATH)
        for name, got, want in zip(("dQ", "dK", "dV"), default, reference, strict=True):
            assert torch.isfinite(got).all(), f"{name} non-finite at length {length}"
            if name == "dQ":
                got, want = got[..., 1:], want[..., 1:]
            gap = ((got.double() - want).abs().max() / want.abs().max()).item()
            log(f"  length {length}, {name}: {gap:.3g} of the fp64 reference's largest entry")
            assert gap <= _RELATIVE_TOLERANCE, f"{name} at length {length} is {gap:.3g} off the fp64 reference"


def check_forced_cudnn_on_an_affected_hopper_still_hits_the_bug() -> None:
    """The control: without it, finite gradients above could come from inputs that never reach the bug."""
    for length in _BUGGY_LENGTHS:
        dq, _, _ = _grads(*_inputs(length), backend=SDPBackend.CUDNN_ATTENTION)
        bad = (~torch.isfinite(dq)).sum().item()
        log(f"  forced cuDNN, length {length}: {bad} non-finite dQ values")
        assert bad > 0, f"forced cuDNN gave a finite dQ at length {length}, so the construction misses the bug"


@gpu_test_main(exact_world_size=1, prefix="cudnn_sdpa_hopper_guard", partial_state=False)
def run(ctx) -> dict:
    checks: dict[str, bool] = {}
    record_check(
        checks, "guard_follows_the_architecture", check_the_load_turns_cudnn_sdpa_off_on_an_affected_hopper_alone
    )
    record_check(checks, "default_dispatch_finite", check_default_dispatch_is_finite_and_matches_an_fp64_reference)
    if _affected():
        record_check(
            checks, "forced_cudnn_reaches_the_bug", check_forced_cudnn_on_an_affected_hopper_still_hits_the_bug
        )
    return {"checks": checks}


if __name__ == "__main__":
    run()
