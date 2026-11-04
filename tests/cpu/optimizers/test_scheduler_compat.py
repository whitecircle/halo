#!/usr/bin/env python
"""Custom optimizers stay LR-scheduler-compatible: ``patch_track_step_called`` reads ``step.__func__``, so ``step``
must be a bound method. FlashAdamW refuses CPU params, so it runs on CUDA when present."""

import inspect

import pytest
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR

from halo.optimizers.adamw_bf16 import AdamWBF16
from halo.optimizers.flash_adamw import build_flash_adamw_optimizer
from halo.optimizers.muon import build_muon_optimizer

BASE_LR = 1e-2
DECAY = 0.5
SCHED_STEPS = 3


def _tiny_model() -> nn.Module:
    torch.manual_seed(0)
    # 2D Linear weight (Muon path) + 1D bias/LayerNorm params (scalar path).
    return nn.Sequential(nn.Linear(8, 8, dtype=torch.bfloat16), nn.LayerNorm(8, dtype=torch.bfloat16))


def _set_grads(model: nn.Module, max_ndim: int | None = None) -> None:
    """``max_ndim=1`` leaves 2D+ grads ``None``, skipping Triton-only paths on CPU."""
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
    opt = AdamWBF16(model.named_parameters(), lr=BASE_LR)
    _assert_scheduler_drives_lr(opt, model)


def test_muon_scheduler_compat():
    model = _tiny_model()
    # No CUDA makes the kernels probe fall back to pure-torch Newton-Schulz; 2D grads stay None since the fused
    # momentum kernel is Triton.
    opt = build_muon_optimizer(model, lr=BASE_LR)
    _assert_scheduler_drives_lr(opt, model, grad_max_ndim=1)


def test_flash_adamw_scheduler_compat():
    pytest.importorskip("flashoptim", reason="flashoptim not installed")
    if not torch.cuda.is_available():
        pytest.skip("flashoptim refuses CPU params at construction (quantized states need CUDA)")

    model = _tiny_model().cuda()
    opt = build_flash_adamw_optimizer(model, lr=BASE_LR)
    _assert_scheduler_drives_lr(opt, model)


def test_plain_function_step_breaks_scheduler_premise():
    """An instance-attribute function has no ``__func__``, so ``LambdaLR`` crashes at construction."""
    model = _tiny_model()
    opt = AdamWBF16(model.named_parameters(), lr=BASE_LR)
    opt.step = lambda closure=None: None
    assert not inspect.ismethod(opt.step)
    with pytest.raises(AttributeError):
        LambdaLR(opt, lr_lambda=lambda epoch: 1.0)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
