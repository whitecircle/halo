#!/usr/bin/env python
"""Custom optimizers must stay LR-scheduler-compatible (CPU).

torch's ``LRScheduler.__init__`` wraps ``optimizer.step`` via ``patch_track_step_called``, which
reads ``step.__func__`` — so ``step`` must be a genuine bound method when the scheduler is built.
An instance-level plain-function monkeypatch of ``step`` (how a step hook could naively be
attached) crashes every scheduler at construction. Pinned for AdamWBF16, Muon and FlashAdamW:

- a fresh optimizer's ``step`` is a bound method exposing ``__func__``;
- ``LambdaLR`` construction succeeds, and ``opt.step(); sched.step()`` drives EVERY param group's
  lr along the lambda (Muon's shadowing ``param_groups`` property must serve the live dicts the
  scheduler writes into, or the new lr lands in a dead attribute);
- premise control: an instance-level plain-function ``step`` really does crash ``LambdaLR`` — the
  failure mode the bound-method contract exists to prevent.

Muon's fused matrix step is Triton (CUDA-only); on CPU the grads that would reach it are left
``None`` so the step skips those params while the scheduler seam — construction, step tracking,
lr propagation — is exercised in full. FlashAdamW refuses CPU params outright at construction, so
its leg runs on CUDA when present and skips cleanly otherwise.

Run: pytest tests/cpu/optimizers/test_scheduler_compat.py  (or python <file>)
"""

import inspect

import pytest
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR

from src.optimizers.adamw_bf16 import AdamWBF16
from src.optimizers.muon import create_muon_optimizer

BASE_LR = 1e-2
DECAY = 0.5
SCHED_STEPS = 3


def _tiny_model() -> nn.Module:
    torch.manual_seed(0)
    # 2D Linear weight (Muon path) + 1D bias/LayerNorm params (scalar path); bf16 for AdamWBF16/Muon.
    return nn.Sequential(nn.Linear(8, 8, dtype=torch.bfloat16), nn.LayerNorm(8, dtype=torch.bfloat16))


def _set_grads(model: nn.Module, max_ndim: int | None = None) -> None:
    """Synthetic grads; ``max_ndim=1`` leaves 2D+ grads ``None`` (skips Triton-only paths on CPU)."""
    torch.manual_seed(1)
    for p in model.parameters():
        if max_ndim is not None and p.ndim > max_ndim:
            continue
        p.grad = torch.randn_like(p) * 1e-3


def _assert_scheduler_drives_lr(opt, model: nn.Module, *, grad_max_ndim: int | None = None) -> None:
    assert inspect.ismethod(opt.step), "optimizer.step must be a bound method"
    assert hasattr(opt.step, "__func__"), "LRScheduler's patch_track_step_called reads step.__func__"
    sched = LambdaLR(opt, lr_lambda=lambda epoch: DECAY**epoch)
    for k in range(1, SCHED_STEPS + 1):
        _set_grads(model, max_ndim=grad_max_ndim)
        opt.step()
        opt.zero_grad()
        sched.step()
        expected = BASE_LR * DECAY**k
        for i, group in enumerate(opt.param_groups):
            assert group["lr"] == pytest.approx(expected), (
                f"group {i}: lr {group['lr']} != {expected} after {k} scheduler steps"
            )


def test_adamw_bf16_scheduler_compat():
    model = _tiny_model()
    opt = AdamWBF16(model.parameters(), lr=BASE_LR)
    _assert_scheduler_drives_lr(opt, model)


def test_muon_scheduler_compat():
    model = _tiny_model()
    # No CUDA -> the kernels probe fails -> pure-torch Newton-Schulz fallback; 2D grads stay None
    # on CPU (fused momentum kernel is Triton), the scalar AdamW leg steps for real.
    opt = create_muon_optimizer(model, lr=BASE_LR)
    _assert_scheduler_drives_lr(opt, model, grad_max_ndim=1)


def test_flash_adamw_scheduler_compat():
    pytest.importorskip("flashoptim", reason="flashoptim not installed")
    if not torch.cuda.is_available():
        pytest.skip("flashoptim refuses CPU params at construction (quantized states need CUDA)")
    from src.optimizers.flash_adamw import create_flash_adamw_optimizer

    model = _tiny_model().cuda()
    opt = create_flash_adamw_optimizer(model, lr=BASE_LR)
    _assert_scheduler_drives_lr(opt, model)


def test_plain_function_step_breaks_scheduler_premise():
    """The failure the bound-method contract prevents: an instance-attribute function has no
    ``__func__``, so ``LambdaLR`` (``patch_track_step_called``) crashes at construction."""
    model = _tiny_model()
    opt = AdamWBF16(model.parameters(), lr=BASE_LR)
    opt.step = lambda closure=None: None  # instance-level plain function (the broken patch style)
    assert not inspect.ismethod(opt.step)
    with pytest.raises(AttributeError):
        LambdaLR(opt, lr_lambda=lambda epoch: 1.0)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
