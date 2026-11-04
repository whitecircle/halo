"""The non-finite step guard composed as every distributed trainer composes it, on a toy model, for one-step-at-a-time
tests."""

from types import SimpleNamespace

import torch
from transformers import Trainer

from halo.trainers.parallelism.grad_clipping import GradientClippingMixin
from halo.trainers.step_guard import NonFiniteStepGuardMixin


class HFBase:
    """The transformers methods the guard extends: the norm, and a log/save/evaluate pass recording the optimizer
    state's count."""

    _get_grad_norm = Trainer._get_grad_norm

    def _maybe_log_save_evaluate(self, tr_loss, *args, **kwargs):
        self.counted_at_save = [int(state["step"]) for state in self.optimizer.state.values()]


class Guarded(GradientClippingMixin, NonFiniteStepGuardMixin, HFBase):
    """The guard behind the clipping mixin and ahead of the Trainer base, as in every distributed trainer."""


def guarded(
    limit: int = 3,
    optimizer_cls=torch.optim.AdamW,
    scaler=None,
    install: bool = True,
    device: str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> Guarded:
    """The guard over a ``Linear(4, 2)``; ``install`` routes the optimizer's step through it, as ``create_optimizer``
    does."""
    torch.manual_seed(0)
    me = object.__new__(Guarded)
    me.model = torch.nn.Linear(4, 2).to(device=device, dtype=dtype)
    me._max_consecutive_nonfinite_steps = limit
    me.state = SimpleNamespace(global_step=0)
    me.accelerator = SimpleNamespace(
        distributed_type="NO", scaler=scaler, clip_grad_norm_=torch.nn.utils.clip_grad_norm_
    )
    me.optimizer = optimizer_cls(me.model.named_parameters(), lr=0.1)
    # transformers' loss accumulator, and its value before the step that a skip restores.
    me._tr_loss, me._tr_loss_before_step = torch.zeros((), device=device), torch.zeros((), device=device)
    if install:
        me._install_nonfinite_step_skip()
    return me


def guarded_step(me: Guarded, grad: float) -> None:
    """One optimizer step on gradients of ``grad``, in the trainer's order: norm, then step."""
    for param in me.model.parameters():
        param.grad = torch.full_like(param, grad)
    me._get_grad_norm(me.model, grad_norm=None)
    me.optimizer.step()
    me.state.global_step += 1
