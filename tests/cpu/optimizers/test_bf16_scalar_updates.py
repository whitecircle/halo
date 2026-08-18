"""bf16 weight writes in the scalar/1-D parameter paths.

Two properties that fail silently — no error, no NaN, just weights that stop moving or replicas that
stop agreeing:

* An AdamW step is ~``lr`` in magnitude. bf16's ULP at value 1.0 is 3.9e-3, so round-to-nearest
  truncates any step below ~2e-3 to zero forever — norm weights (value ~1) freeze at every usual LR.
* Bias correction reads ``state["step"]``. If that counter only advances when a grad is present, two
  replicas that disagree about grad presence for one step disagree about every later step size.
"""

import pytest
import torch
from torch import nn

from src.optimizers.adamw_bf16 import AdamWBF16
from src.optimizers.muon import create_muon_optimizer

LR = 3e-4  # the documented Muon default, and a typical fine-tune LR
STEPS = 200


def _frozen_fraction(optimizer_factory, value: float = 1.0, numel: int = 256) -> float:
    """Fraction of elements whose bf16 weight is byte-for-byte unchanged after STEPS updates."""
    torch.manual_seed(0)
    param = nn.Parameter(torch.full((numel,), value, dtype=torch.bfloat16))
    before = param.detach().clone()
    optimizer = optimizer_factory([param])

    generator = torch.Generator().manual_seed(1)
    for _ in range(STEPS):
        param.grad = torch.randn(numel, generator=generator, dtype=torch.bfloat16) * 0.01
        optimizer.step()

    return (param.detach() == before).float().mean().item()


def test_stochastic_rounding_keeps_bf16_norm_weights_moving():
    """AdamWBF16 must move a bf16 param sitting at 1.0; plain AdamW provably cannot at this LR.

    This is the mechanism behind routing Muon's scalar branch through AdamWBF16: torch.optim.AdamW
    writes bf16 with round-to-nearest, so every norm weight in the model is frozen for the whole run.
    """
    nearest_frozen = _frozen_fraction(lambda params: torch.optim.AdamW(params, lr=LR, weight_decay=0.0, foreach=False))
    sr_frozen = _frozen_fraction(lambda params: AdamWBF16(params, lr=LR, weight_decay=0.0, use_triton=False))

    assert nearest_frozen == 1.0, (
        f"expected round-to-nearest AdamW to freeze every element at lr={LR} (that is the premise of "
        f"this test); got {nearest_frozen:.3f} frozen"
    )
    assert sr_frozen < 0.9, (
        f"AdamWBF16 left {sr_frozen:.1%} of a bf16 param at value 1.0 completely unmoved after "
        f"{STEPS} steps at lr={LR}. Stochastic rounding is not reaching the weight write, so these "
        "parameters are frozen for the entire run."
    )


def test_muon_routes_scalar_params_through_the_stochastic_rounding_optimizer():
    """Muon's 1-D / embedding / head branch must not be a nearest-rounding AdamW.

    Muon requires bf16 weights, so a plain torch.optim.AdamW on that branch hits the freeze above.
    """
    model = nn.Sequential(nn.Linear(32, 32, bias=True), nn.LayerNorm(32)).to(torch.bfloat16)
    optimizer = create_muon_optimizer(model, lr=LR, ns_use_kernels=False)

    scalar_optimizer = optimizer.scalar_optimizer
    assert scalar_optimizer is not None, "expected 1-D params to be routed to a scalar optimizer"
    assert isinstance(scalar_optimizer, AdamWBF16), (
        f"Muon's scalar optimizer is {type(scalar_optimizer).__name__}; it must be AdamWBF16 so the "
        "bf16 writes for norms/biases/embeddings use stochastic rounding rather than round-to-nearest."
    )


def _step_with_optional_grad(optimizer, param, grad):
    param.grad = grad
    optimizer.step()


def test_step_counter_advances_even_when_a_replica_has_no_grad():
    """Replicas must not diverge because one of them saw grad=None for a step.

    ``state["step"]`` drives bias correction, so if it only advances on grad presence, a replica that
    skipped one step permanently applies a different step size to an identically-replicated weight —
    with no error raised anywhere. Sparse-routed experts, an idle VLM tower and router tie-breaks all
    produce rank-nonuniform grad presence.
    """
    torch.manual_seed(0)
    start = torch.full((64,), 0.5, dtype=torch.bfloat16)
    replica_a = nn.Parameter(start.clone())
    replica_b = nn.Parameter(start.clone())

    opt_a = AdamWBF16([replica_a], lr=1e-2, weight_decay=0.0, use_triton=False)
    opt_b = AdamWBF16([replica_b], lr=1e-2, weight_decay=0.0, use_triton=False)

    generator = torch.Generator().manual_seed(7)
    grads = [torch.randn(64, generator=generator, dtype=torch.bfloat16) * 0.1 for _ in range(4)]

    # Step 0: replica B has no gradient for this parameter; both must still advance their counters.
    _step_with_optional_grad(opt_a, replica_a, grads[0].clone())
    replica_b.grad = None
    opt_b.step()

    for grad in grads[1:]:
        _step_with_optional_grad(opt_a, replica_a, grad.clone())
        _step_with_optional_grad(opt_b, replica_b, grad.clone())

    step_a = opt_a.state[replica_a]["step"]
    step_b = opt_b.state[replica_b]["step"]
    assert step_a == step_b, (
        f"replicas disagree on the Adam step counter ({step_a} vs {step_b}) because one of them saw "
        "grad=None once. Bias correction is derived from this counter, so every subsequent update "
        "differs between replicas of the same weight."
    )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
