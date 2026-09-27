"""The EP clip hands its coefficient to the optimizer only when the optimizer steps exactly the clipped
parameters that hold gradients; anything else keeps the in-place gradient scaling."""

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from src.optimizers.adamw_bf16 import AdamWBF16
from src.trainers.mixins.grad_sync import GradientSyncMixin


def _params(n: int) -> list[nn.Parameter]:
    params = [nn.Parameter(torch.zeros(4, dtype=torch.bfloat16)) for _ in range(n)]
    for p in params:
        p.grad = torch.ones_like(p)
    return params


def _target(trainer, params):
    return GradientSyncMixin._grad_scale_deferring_optimizer(trainer, params)


def test_adamw_bf16_owning_every_param_takes_the_scale():
    params = _params(3)
    optimizer = AdamWBF16(params)
    assert _target(SimpleNamespace(optimizer=optimizer), params) is optimizer
    # accelerate wraps the optimizer; the inner one is the target
    assert _target(SimpleNamespace(optimizer=SimpleNamespace(optimizer=optimizer)), params) is optimizer


def test_a_clipped_param_outside_the_optimizer_keeps_in_place_scaling():
    params = _params(3)
    assert _target(SimpleNamespace(optimizer=AdamWBF16(params[:2])), params) is None


def test_an_optimizer_stepping_unclipped_grads_keeps_in_place_scaling():
    # The step would scale params[2] although the clip did not select it.
    params = _params(3)
    assert _target(SimpleNamespace(optimizer=AdamWBF16(params)), params[:2]) is None


def test_unclipped_optimizer_params_without_grads_do_not_block_deferral():
    params = _params(3)
    params[2].grad = None
    optimizer = AdamWBF16(params)
    assert _target(SimpleNamespace(optimizer=optimizer), params[:2]) is optimizer


def test_an_optimizer_without_deferral_keeps_in_place_scaling():
    params = _params(2)
    assert _target(SimpleNamespace(optimizer=torch.optim.AdamW(params)), params) is None
    assert _target(SimpleNamespace(optimizer=None), params) is None


def test_params_without_grads_do_not_block_deferral():
    params = _params(3)
    params[2].grad = None
    assert _target(SimpleNamespace(optimizer=AdamWBF16(params[:2])), params) is not None


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
