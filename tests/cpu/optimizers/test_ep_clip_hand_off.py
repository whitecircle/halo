"""The EP clip applies its coefficient exactly once: an AdamWBF16 that steps the clipped parameters takes
it as a pending scale and applies it inside its step, leaving the gradients unscaled; any other
optimizer gets the gradients scaled in place. Either way the step equals clip-then-step.

A clip that computes the coefficient and never hands it over trains unclipped with no other symptom, so
the step itself is compared against a reference that clips first.
"""

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from src.optimizers.adamw_bf16 import AdamWBF16
from src.trainers.mixins.grad_clip import clip_coefficient
from src.trainers.mixins.grad_sync import GradientSyncMixin

MAX_NORM = 0.5


class _EPTrainer(GradientSyncMixin):
    """The attributes ``ep_clip_grad_norm_`` reads, on one rank with EP layers and no TP."""

    def __init__(self, optimizer, params):
        self.optimizer = optimizer
        self.accelerator = SimpleNamespace(clip_grad_norm_=None)
        self.parallelism_config = SimpleNamespace(is_tp_mode=False)
        self._has_ep_layers = True
        self._params = params
        self._patch_gradient_clipping_for_ep()

    def _sync_deferred_expert_grads(self) -> None:
        pass

    def _compute_global_grad_norm(self) -> torch.Tensor:
        return torch.linalg.vector_norm(
            torch.stack([torch.linalg.vector_norm(p.grad, dtype=torch.float32) for p in self._params])
        )


def _params(dtype: torch.dtype = torch.float32, seed: int = 0) -> list[nn.Parameter]:
    generator = torch.Generator().manual_seed(seed)
    params = [nn.Parameter(torch.randn(8, 4, generator=generator).to(dtype)) for _ in range(3)]
    for p in params:
        p.grad = (torch.randn(p.shape, generator=generator) * 3).to(dtype)  # well past MAX_NORM
    return params


def _clip_then_step(optimizer_cls, dtype: torch.dtype = torch.float32) -> torch.optim.Optimizer:
    """The reference: clip the gradients in place, then step; returns the stepped optimizer."""
    params = _params(dtype)
    norm = torch.linalg.vector_norm(
        torch.stack([torch.linalg.vector_norm(p.grad, dtype=torch.float32) for p in params])
    )
    for p in params:
        p.grad.mul_(clip_coefficient(MAX_NORM, norm))
    optimizer = optimizer_cls(params, lr=1e-2)
    optimizer.step()
    return optimizer


def _assert_steps_match(optimizer: torch.optim.Optimizer, reference: torch.optim.Optimizer) -> None:
    """Parameters and first moments equal the reference bit for bit. A first Adam step is nearly
    scale-invariant (``m / sqrt(v)``), so the moment is what shows the scale the step applied."""
    got, want = optimizer.param_groups[0]["params"], reference.param_groups[0]["params"]
    for p, ref in zip(got, want, strict=True):
        assert torch.equal(p.detach(), ref.detach()), "the step did not apply the clip coefficient as the clip would"
        if "exp_avg" in reference.state[ref]:
            assert torch.equal(optimizer.state[p]["exp_avg"], reference.state[ref]["exp_avg"])


@pytest.mark.parametrize("dtype", (torch.float32, torch.bfloat16), ids=("fp32", "bf16"))
def test_adamw_bf16_takes_the_scale_and_steps_as_if_clipped_first(dtype):
    """Bit for bit, in both dtypes: for bf16 the clip's in-place multiply rounds the coefficient and the
    product to bf16, and the deferred scale must round the same way (the coefficient here is not exact in
    bf16)."""
    params = _params(dtype)
    grads_before = [p.grad.clone() for p in params]
    optimizer = AdamWBF16(params, lr=1e-2)
    trainer = _EPTrainer(optimizer, params)

    norm = trainer.accelerator.clip_grad_norm_(params, MAX_NORM)

    assert torch.equal(optimizer._grad_scale, clip_coefficient(MAX_NORM, norm).reshape(()))
    for p, before in zip(params, grads_before, strict=True):
        assert torch.equal(p.grad, before), "the deferred path must leave the gradients unscaled"
    assert optimizer._grad_scale.bfloat16().float() != optimizer._grad_scale  # premise: rounding matters
    optimizer.step()
    _assert_steps_match(optimizer, _clip_then_step(AdamWBF16, dtype))


@pytest.mark.parametrize("optimizer_cls", (torch.optim.AdamW, torch.optim.SGD))
def test_another_optimizer_gets_the_gradients_scaled_in_place(optimizer_cls):
    params = _params()
    grads_before = [p.grad.clone() for p in params]
    optimizer = optimizer_cls(params, lr=1e-2)
    trainer = _EPTrainer(optimizer, params)

    norm = trainer.accelerator.clip_grad_norm_(params, MAX_NORM)

    coefficient = clip_coefficient(MAX_NORM, norm)
    for p, before in zip(params, grads_before, strict=True):
        torch.testing.assert_close(p.grad, before * coefficient, rtol=0, atol=0)
    optimizer.step()
    _assert_steps_match(optimizer, _clip_then_step(optimizer_cls))


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
